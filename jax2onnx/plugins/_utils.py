# jax2onnx/plugins/_utils.py

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import onnx_ir as ir

if TYPE_CHECKING:
    from jax2onnx.converter.conversion_api import _IRBuildContext as IRBuildContext  # type: ignore


_IR_TO_NP_DTYPE: dict[ir.DataType, np.dtype] = {}
_dtype_pairs = [
    ("DOUBLE", np.float64),
    ("FLOAT", np.float32),
    ("FLOAT16", np.float16),
    ("INT64", np.int64),
    ("INT32", np.int32),
    ("INT16", np.int16),
    ("INT8", np.int8),
    ("UINT64", np.uint64),
    ("UINT32", np.uint32),
    ("UINT16", np.uint16),
    ("UINT8", np.uint8),
    ("BOOL", np.bool_),
]
for name, np_dt in _dtype_pairs:
    enum = getattr(ir.DataType, name, None)
    if enum is None or np_dt is None:
        continue
    _IR_TO_NP_DTYPE[enum] = np.dtype(np_dt)
del name, np_dt, enum


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
    p_ty = getattr(param, "type", None)
    l_ty = getattr(like, "type", None)
    p_dt = getattr(p_ty, "dtype", None)
    l_dt = getattr(l_ty, "dtype", None)
    if p_dt is None or l_dt is None or p_dt == l_dt:
        return param

    const_tensor = getattr(param, "const_value", None)
    if const_tensor is not None:
        try:
            np_arr = const_tensor.numpy()
        except Exception:
            np_arr = None
        target_np = _IR_TO_NP_DTYPE.get(l_dt)
        if np_arr is not None and target_np is not None:
            if np_arr.dtype != target_np:
                np_arr = np_arr.astype(target_np, copy=False)
                param.const_value = ir.tensor(np_arr)
            param.type = ir.TensorType(l_dt)
            if getattr(param, "shape", None) is None and hasattr(np_arr, "shape"):
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


# --- NEW: inline reshape for constant parameters (no runtime node) ---
def inline_reshape_initializer(
    ctx, val: ir.Value, new_shape: tuple[int, ...], name_hint: str
) -> ir.Value:
    """
    If `val` is a constant initializer, create a new initializer with the data
    reshaped to `new_shape` and return it. Otherwise, return `val` unchanged.
    """
    arr = getattr(val, "const_value", None)
    if arr is None:
        return val  # not a constant → caller must insert a Reshape node

    np_arr = np.asarray(arr).reshape(new_shape)
    reshaped = ir.Value(
        name=ctx.fresh_name(name_hint),
        type=val.type,  # keep dtype
        shape=ir.Shape(tuple(int(s) for s in new_shape)),
        const_value=ir.tensor(np_arr),
    )
    ctx._initializers.append(reshaped)
    return reshaped
