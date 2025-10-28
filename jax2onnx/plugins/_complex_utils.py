# jax2onnx/plugins/_complex_utils.py

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Dict, Final

import numpy as np
import onnx_ir as ir
from onnx_ir import Attr, AttributeType

from jax2onnx.plugins._ir_shapes import (
    _ensure_value_metadata,
    _stamp_type_and_shape,
)

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from jax2onnx.converter.ir_context import IRContext


_COMPLEX_PAIR: Final[Dict[ir.DataType, ir.DataType]] = {
    ir.DataType.COMPLEX64: ir.DataType.FLOAT,
    ir.DataType.COMPLEX128: ir.DataType.DOUBLE,
}

_BASE_TO_COMPLEX: Final[Dict[ir.DataType, ir.DataType]] = {
    ir.DataType.FLOAT: ir.DataType.COMPLEX64,
    ir.DataType.DOUBLE: ir.DataType.COMPLEX128,
}


def _shape_tuple(val: ir.Value) -> tuple[object, ...]:
    """Return a tuple view of `val`'s shape metadata."""
    shape_obj = getattr(val, "shape", None)
    if isinstance(shape_obj, ir.Shape):
        return tuple(shape_obj.dims)
    if isinstance(shape_obj, Sequence) and not isinstance(shape_obj, (str, bytes)):
        return tuple(shape_obj)
    return ()


def _base_dtype_for_complex(dtype: ir.DataType) -> ir.DataType:
    if dtype not in _COMPLEX_PAIR:
        raise ValueError(f"Expected complex dtype, received: {dtype}")
    return _COMPLEX_PAIR[dtype]


def _complex_dtype_for_base(dtype: ir.DataType) -> ir.DataType:
    if dtype not in _BASE_TO_COMPLEX:
        raise ValueError(f"Unsupported real base dtype for complex packing: {dtype}")
    return _BASE_TO_COMPLEX[dtype]


def ensure_complex_dtype(
    ctx: "IRContext",
    value: ir.Value,
    target_dtype: ir.DataType,
    *,
    name_hint: str = "complex_cast",
) -> ir.Value:
    """
    Ensure `value` carries `target_dtype`, inserting a Cast node when needed.
    """
    if target_dtype not in _COMPLEX_PAIR:
        raise ValueError(
            f"Target dtype must be complex64/complex128, got {target_dtype}"
        )

    current_dtype = value.dtype
    if current_dtype == target_dtype:
        return value
    if current_dtype is None:
        value.type = ir.TensorType(target_dtype)
        return value

    cast_out = ir.Value(
        name=ctx.fresh_name(name_hint),
        type=ir.TensorType(target_dtype),
        shape=value.shape,
    )
    ctx.builder.add_node(
        "Cast",
        inputs=[value],
        outputs=[cast_out],
        attributes=[Attr("to", AttributeType.INT, int(target_dtype.value))],
        name=ctx.fresh_name("Cast"),
    )
    _ensure_value_metadata(ctx, cast_out)
    return cast_out


def pack_native_complex(
    ctx: "IRContext", tensor: ir.Value, *, name_hint: str = "complex_pack"
) -> ir.Value:
    """
    Convert a native complex tensor to a `[... , 2]` float view suitable for ONNX DFT/STFT.
    """
    dtype = tensor.dtype
    if dtype not in _COMPLEX_PAIR:
        raise ValueError(f"pack_native_complex expects complex dtype, received {dtype}")

    base_dtype = _base_dtype_for_complex(dtype)
    dims = _shape_tuple(tensor)
    axis_index = len(dims)

    real_part = ctx.builder.Real(tensor, _outputs=[ctx.fresh_name(f"{name_hint}_real")])
    imag_part = ctx.builder.Imag(tensor, _outputs=[ctx.fresh_name(f"{name_hint}_imag")])
    for part in (real_part, imag_part):
        part.type = ir.TensorType(base_dtype)
        _stamp_type_and_shape(part, dims)
        _ensure_value_metadata(ctx, part)

    unsqueeze_axes = [axis_index]
    real_unsq = ctx.builder.Unsqueeze(
        real_part,
        axes=unsqueeze_axes,
        _outputs=[ctx.fresh_name(f"{name_hint}_real_unsq")],
    )
    imag_unsq = ctx.builder.Unsqueeze(
        imag_part,
        axes=unsqueeze_axes,
        _outputs=[ctx.fresh_name(f"{name_hint}_imag_unsq")],
    )
    for part in (real_unsq, imag_unsq):
        part.type = ir.TensorType(base_dtype)
        _stamp_type_and_shape(part, dims + (1,))
        _ensure_value_metadata(ctx, part)

    packed = ctx.builder.Concat(
        real_unsq,
        imag_unsq,
        axis=axis_index,
        _outputs=[ctx.fresh_name(f"{name_hint}_ri")],
    )
    packed.type = ir.TensorType(base_dtype)
    _stamp_type_and_shape(packed, dims + (2,))
    _ensure_value_metadata(ctx, packed)
    return packed


def unpack_to_native_complex(
    ctx: "IRContext",
    tensor: ir.Value,
    *,
    name_hint: str = "complex_unpack",
    target_dtype: ir.DataType | None = None,
) -> ir.Value:
    """
    Convert a `[... , 2]` float tensor (real/imag channels) back to a native complex tensor.
    """
    base_dtype = tensor.dtype
    if base_dtype not in _BASE_TO_COMPLEX:
        raise ValueError(
            f"unpack_to_native_complex expects float pair tensor, received dtype {base_dtype}"
        )

    dims = _shape_tuple(tensor)
    if not dims:
        raise ValueError("Cannot unpack scalar tensor without real/imag axis")
    axis_index = len(dims) - 1
    trailing = dims[-1]
    if isinstance(trailing, (int, np.integer)) and int(trailing) != 2:
        raise ValueError(
            f"Expected trailing dimension of size 2 for real/imag channels, got {trailing}"
        )

    idx_vals: list[ir.Value] = []
    for idx in (0, 1):
        init = ctx.builder.add_initializer_from_scalar(
            name=ctx.fresh_name(f"{name_hint}_idx{idx}"),
            value=np.asarray(idx, dtype=np.int64),
        )
        _stamp_type_and_shape(init, ())
        idx_vals.append(init)

    gather_real = ctx.builder.Gather(
        tensor,
        idx_vals[0],
        axis=axis_index,
        _outputs=[ctx.fresh_name(f"{name_hint}_real")],
    )
    gather_imag = ctx.builder.Gather(
        tensor,
        idx_vals[1],
        axis=axis_index,
        _outputs=[ctx.fresh_name(f"{name_hint}_imag")],
    )
    base_dims = dims[:-1]
    for part in (gather_real, gather_imag):
        part.type = ir.TensorType(base_dtype)
        _stamp_type_and_shape(part, base_dims)
        _ensure_value_metadata(ctx, part)

    complex_dtype = target_dtype or _complex_dtype_for_base(base_dtype)
    complex_val = ctx.builder.Complex(
        gather_real,
        gather_imag,
        _outputs=[ctx.fresh_name(f"{name_hint}_complex")],
    )
    complex_val.type = ir.TensorType(complex_dtype)
    _stamp_type_and_shape(complex_val, base_dims)
    _ensure_value_metadata(ctx, complex_val)
    return complex_val
