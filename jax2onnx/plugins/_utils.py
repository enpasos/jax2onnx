# jax2onnx/plugins/_utils.py

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Final, Sequence, cast
import numpy as np
import onnx_ir as ir

if TYPE_CHECKING:
    from jax2onnx.converter.conversion_api import _IRBuildContext as IRBuildContext  # type: ignore


_DTYPE_PAIRS: Final[Sequence[tuple[ir.DataType, np.dtype[Any]]]] = (
    (ir.DataType.DOUBLE, np.dtype(np.float64)),
    (ir.DataType.FLOAT, np.dtype(np.float32)),
    (ir.DataType.FLOAT16, np.dtype(np.float16)),
    (ir.DataType.INT64, np.dtype(np.int64)),
    (ir.DataType.INT32, np.dtype(np.int32)),
    (ir.DataType.INT16, np.dtype(np.int16)),
    (ir.DataType.INT8, np.dtype(np.int8)),
    (ir.DataType.UINT64, np.dtype(np.uint64)),
    (ir.DataType.UINT32, np.dtype(np.uint32)),
    (ir.DataType.UINT16, np.dtype(np.uint16)),
    (ir.DataType.UINT8, np.dtype(np.uint8)),
    (ir.DataType.BOOL, np.dtype(np.bool_)),
)


_IR_TO_NP_DTYPE: dict[ir.DataType, np.dtype[Any]] = dict(_DTYPE_PAIRS)


def _const_tensor_numpy(value: ir.Value) -> np.ndarray[Any, np.dtype[Any]] | None:
    tensor = ir.convenience.get_const_tensor(value)
    if tensor is None:
        return None
    try:
        return cast(np.ndarray[Any, np.dtype[Any]], tensor.numpy())
    except AttributeError:
        try:
            return cast(np.ndarray[Any, np.dtype[Any]], np.asarray(tensor))
        except Exception:
            return None
    except Exception:
        return None


def cast_param_like(
    ctx: "IRBuildContext",
    param: ir.Value,
    like: ir.Value,
    name_hint: str = "param_cast",
) -> ir.Value:
    """
    If `param`'s dtype differs from `like`'s dtype, insert a CastLike node to cast
    `param` to `like`'s dtype. Never casts `like`. Returns the value to use downstream.

    Safe to call when dtypes are unknown (no-op). Preserves shape.
    """
    p_dt = param.dtype
    l_dt = like.dtype
    if p_dt is None or l_dt is None or p_dt == l_dt:
        return param

    if param.const_value is not None:
        np_arr = _const_tensor_numpy(param)
        target_np = _IR_TO_NP_DTYPE.get(l_dt)
        if np_arr is not None and target_np is not None:
            if np_arr.dtype != target_np:
                np_arr = np_arr.astype(target_np, copy=False)
                param.const_value = ir.tensor(np_arr)
            param.type = ir.TensorType(l_dt)
            if param.shape is None and hasattr(np_arr, "shape"):
                param.shape = ir.Shape(tuple(int(d) for d in np_arr.shape))
            return param

    builder = getattr(ctx, "builder", None)
    if builder is None:
        return ctx.cast_like(param, exemplar=like, name_hint=name_hint)

    cast_name = ctx.fresh_name(name_hint)
    out = builder.CastLike(
        param,
        like,
        _outputs=[cast_name],
    )
    out.type = ir.TensorType(l_dt)
    out.shape = param.shape
    return out


def inline_reshape_initializer(
    ctx, val: ir.Value, new_shape: tuple[int, ...], name_hint: str
) -> ir.Value:
    """
    If `val` is a constant initializer, create a new initializer with the data
    reshaped to `new_shape` and return it. Otherwise, return `val` unchanged.
    """
    if val.const_value is None:
        return val  # not a constant → caller must insert a Reshape node

    arr = _const_tensor_numpy(val)
    if arr is None:
        return val

    np_arr = np.asarray(arr).reshape(new_shape)
    # Preserve dtype from the original value when available
    target_dtype = val.dtype
    if target_dtype is not None and target_dtype in _IR_TO_NP_DTYPE:
        np_arr = np_arr.astype(_IR_TO_NP_DTYPE[target_dtype], copy=False)

    return ctx.builder.add_initializer_from_array(
        name=ctx.fresh_name(name_hint), array=np_arr
    )


def normalize_builder_outputs(
    outputs: ir.Value | Sequence[ir.Value],
) -> tuple[ir.Value, ...]:
    """
    Normalize ONNX IR builder outputs to a tuple of values.

    The underlying builder collapses op results to a bare ``ir.Value`` whenever
    the requested output arity is 1, even if the caller passed a one-element
    ``_outputs`` sequence. Multi-result lowerings that iterate over the builder
    result should use this helper so the single-output edge case behaves like
    the general multi-output path.
    """

    if isinstance(outputs, ir.Value):
        return (outputs,)
    return tuple(outputs)
