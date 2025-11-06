# jax2onnx/plugins/_complex_utils.py

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Dict, Final, Tuple, cast

import numpy as np
import onnx_ir as ir
from onnx_ir import Attr, AttributeType

from jax2onnx.plugins._ir_shapes import (
    DimValue,
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

_COMPLEX_TO_NP_DTYPE: Final[Dict[ir.DataType, np.dtype]] = {
    ir.DataType.COMPLEX64: np.dtype(np.complex64),
    ir.DataType.COMPLEX128: np.dtype(np.complex128),
}

COMPLEX_DTYPES: Final[frozenset[ir.DataType]] = frozenset(_COMPLEX_PAIR.keys())
REAL_PAIR_DTYPES: Final[frozenset[ir.DataType]] = frozenset(
    {ir.DataType.FLOAT, ir.DataType.DOUBLE}
)


def coerce_dim_values(dims: Tuple[object, ...]) -> Tuple[DimValue, ...]:
    out: list[DimValue] = []
    for dim in dims:
        if isinstance(dim, (int, np.integer)):
            out.append(int(dim))
        elif isinstance(dim, ir.SymbolicDim):
            out.append(dim)
        elif dim is None:
            out.append(None)
        else:
            text = str(dim)
            out.append(ir.SymbolicDim(text) if text else None)
    return tuple(out)


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
    View a native complex tensor as a `[... , 2]` float pair without relying on complex ops.

    The caller is responsible for ensuring the actual runtime inputs provide the packed
    representation (real/imag stacked in the trailing dimension).
    """
    dtype = tensor.dtype
    if dtype not in _COMPLEX_PAIR:
        raise ValueError(f"pack_native_complex expects complex dtype, received {dtype}")

    base_dtype = _base_dtype_for_complex(dtype)
    enable_double = getattr(
        getattr(ctx, "builder", None), "enable_double_precision", False
    )
    if enable_double and base_dtype == ir.DataType.FLOAT:
        base_dtype = ir.DataType.DOUBLE
    dims = _shape_tuple(tensor)
    packed_shape = dims + (2,)

    tensor.dtype = base_dtype
    tensor.type = ir.TensorType(base_dtype)
    coerced_shape = coerce_dim_values(packed_shape)
    tensor.shape = ir.Shape(coerced_shape)
    _stamp_type_and_shape(tensor, coerced_shape)
    _ensure_value_metadata(ctx, tensor)
    return tensor


def is_packed_complex_tensor(value: ir.Value) -> bool:
    """
    True when `value` already stores real/imag channels in its trailing dimension.
    """
    dtype = value.dtype
    if dtype is None:
        type_obj = getattr(value, "type", None)
        if isinstance(type_obj, ir.TensorType):
            dtype = type_obj.dtype
    if dtype not in REAL_PAIR_DTYPES:
        return False
    dims = _shape_tuple(value)
    if not dims:
        return False
    last = dims[-1]
    if isinstance(last, ir.SymbolicDim):
        # Try to interpret symbolic dims like '2'
        try:
            return int(str(last)) == 2
        except Exception:
            return False
    if isinstance(last, (int, np.integer)):
        return int(last) == 2
    return False


def cast_real_tensor(
    ctx: "IRContext",
    value: ir.Value,
    target_dtype: ir.DataType,
    *,
    name_hint: str,
) -> ir.Value:
    """
    Ensure a real-valued tensor carries `target_dtype`.
    """
    if value.dtype == target_dtype:
        return value
    cast_value = cast(
        ir.Value,
        ctx.builder.Cast(
            value,
            to=int(target_dtype.value),
            _outputs=[ctx.fresh_name(name_hint)],
        ),
    )
    cast_value.type = ir.TensorType(target_dtype)
    cast_value.dtype = target_dtype
    dims = coerce_dim_values(_shape_tuple(value))
    _stamp_type_and_shape(cast_value, dims)
    _ensure_value_metadata(ctx, cast_value)
    return cast_value


def ensure_packed_real_pair(
    ctx: "IRContext", value: ir.Value, *, name_hint: str
) -> tuple[ir.Value, ir.DataType]:
    """
    Return `(packed_tensor, base_dtype)` for complex or packed-complex values.
    """
    dtype = value.dtype
    if dtype is None:
        type_obj = getattr(value, "type", None)
        if isinstance(type_obj, ir.TensorType):
            dtype = type_obj.dtype
    if dtype is None:
        raise ValueError("Value is missing dtype metadata for complex handling")
    if dtype in _COMPLEX_PAIR:
        packed = pack_native_complex(ctx, value, name_hint=name_hint)
        base_dtype = packed.dtype
        if base_dtype is None:
            base_dtype = _base_dtype_for_complex(dtype)
            packed.dtype = base_dtype
            packed.type = ir.TensorType(base_dtype)
        return packed, base_dtype
    if is_packed_complex_tensor(value):
        base_dtype = value.dtype
        if base_dtype is None:
            type_obj = getattr(value, "type", None)
            if isinstance(type_obj, ir.TensorType):
                base_dtype = type_obj.dtype
        if base_dtype is None:
            raise ValueError("Packed complex tensor missing dtype metadata")
        return value, base_dtype
    raise ValueError(
        f"Expected complex dtype or packed tensor with trailing channel, got {dtype}"
    )


def resolve_common_real_dtype(lhs: ir.DataType, rhs: ir.DataType) -> ir.DataType:
    """
    Pick a shared real dtype for packed complex operations.
    """
    if lhs == rhs:
        return lhs
    if ir.DataType.DOUBLE in (lhs, rhs):
        return ir.DataType.DOUBLE
    return ir.DataType.FLOAT


def split_packed_real_imag(
    ctx: "IRContext",
    value: ir.Value,
    base_dtype: ir.DataType,
    *,
    prefix: str,
) -> tuple[ir.Value, ir.Value]:
    """
    Extract real and imaginary channels from a packed tensor `[... , 2]`.
    """
    dims = _shape_tuple(value)
    if not dims:
        raise ValueError("Packed complex tensor must have rank >= 1")
    trailing = dims[-1]
    if isinstance(trailing, (int, np.integer)) and int(trailing) != 2:
        raise ValueError(
            f"Expected trailing dimension of size 2 for real/imag channels, got {trailing}"
        )
    axis = len(dims) - 1
    base_dims = dims[:-1]
    dim_values = coerce_dim_values(base_dims)

    idx_vals: list[ir.Value] = []
    for idx in (0, 1):
        init = ctx.builder.add_initializer_from_scalar(
            name=ctx.fresh_name(f"{prefix}_idx{idx}"),
            value=np.asarray(idx, dtype=np.int64),
        )
        init.type = ir.TensorType(ir.DataType.INT64)
        _stamp_type_and_shape(init, ())
        _ensure_value_metadata(ctx, init)
        idx_vals.append(init)

    parts: list[ir.Value] = []
    for idx_val, channel in zip(idx_vals, ("real", "imag"), strict=True):
        gathered = cast(
            ir.Value,
            ctx.builder.Gather(
                value,
                idx_val,
                axis=axis,
                _outputs=[ctx.fresh_name(f"{prefix}_{channel}")],
            ),
        )
        gathered.type = ir.TensorType(base_dtype)
        gathered.dtype = base_dtype
        _stamp_type_and_shape(gathered, dim_values)
        _ensure_value_metadata(ctx, gathered)
        parts.append(gathered)
    real_part, imag_part = parts
    return real_part, imag_part


def pack_real_imag_pair(
    ctx: "IRContext",
    real: ir.Value,
    imag: ir.Value,
    base_dtype: ir.DataType,
    *,
    name_hint: str,
    output_name: str | None = None,
) -> ir.Value:
    """
    Pack real/imag tensors with identical shapes into a `[... , 2]` tensor.
    """
    real_dims = _shape_tuple(real)
    imag_dims = _shape_tuple(imag)
    if real_dims != imag_dims:
        raise ValueError(
            "Real and imaginary tensors must share the same shape for packing"
        )
    axis = len(real_dims)
    coerce_dim_values(real_dims)
    axes_arr = np.asarray([axis], dtype=np.int64)
    axes_init = ctx.builder.add_initializer_from_array(
        name=ctx.fresh_name(f"{name_hint}_axes"),
        array=axes_arr,
    )
    axes_init.type = ir.TensorType(ir.DataType.INT64)
    _stamp_type_and_shape(axes_init, (1,))
    _ensure_value_metadata(ctx, axes_init)

    def _unsqueeze(value: ir.Value, channel: str) -> ir.Value:
        unsqueezed = cast(
            ir.Value,
            ctx.builder.Unsqueeze(
                value,
                axes_init,
                _outputs=[ctx.fresh_name(f"{name_hint}_{channel}_unsq")],
            ),
        )
        unsqueezed.type = ir.TensorType(base_dtype)
        unsqueezed.dtype = base_dtype
        unsqueezed_dims = coerce_dim_values(real_dims + (1,))
        _stamp_type_and_shape(unsqueezed, unsqueezed_dims)
        _ensure_value_metadata(ctx, unsqueezed)
        return unsqueezed

    real_unsq = _unsqueeze(real, "real")
    imag_unsq = _unsqueeze(imag, "imag")

    concat_name = output_name or ctx.fresh_name(name_hint)
    packed = cast(
        ir.Value,
        ctx.builder.Concat(
            real_unsq,
            imag_unsq,
            axis=axis,
            _outputs=[concat_name],
        ),
    )
    packed.type = ir.TensorType(base_dtype)
    packed.dtype = base_dtype
    packed_dims = coerce_dim_values(real_dims + (2,))
    _stamp_type_and_shape(packed, packed_dims)
    _ensure_value_metadata(ctx, packed)
    return packed


def conjugate_packed_tensor(
    ctx: "IRContext",
    value: ir.Value,
    base_dtype: ir.DataType,
    *,
    prefix: str,
    output_name: str | None = None,
) -> ir.Value:
    """
    Return the conjugate of a `[... , 2]` packed tensor.
    """
    real_part, imag_part = split_packed_real_imag(
        ctx, value, base_dtype, prefix=f"{prefix}_split"
    )
    neg_imag = cast(
        ir.Value,
        ctx.builder.Neg(
            imag_part,
            _outputs=[ctx.fresh_name(f"{prefix}_neg_imag")],
        ),
    )
    neg_imag.type = ir.TensorType(base_dtype)
    dims = coerce_dim_values(_shape_tuple(imag_part))
    _stamp_type_and_shape(neg_imag, dims)
    _ensure_value_metadata(ctx, neg_imag)

    return pack_real_imag_pair(
        ctx,
        real_part,
        neg_imag,
        base_dtype,
        name_hint=f"{prefix}_conj",
        output_name=output_name,
    )


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
    dim_values = coerce_dim_values(base_dims)
    for part in (gather_real, gather_imag):
        part.type = ir.TensorType(base_dtype)
        _stamp_type_and_shape(part, dim_values)
        _ensure_value_metadata(ctx, part)

    complex_dtype = target_dtype or _complex_dtype_for_base(base_dtype)
    complex_val = cast(
        ir.Value,
        ctx.builder.Complex(
            gather_real,
            gather_imag,
            _outputs=[ctx.fresh_name(f"{name_hint}_complex")],
        ),
    )
    complex_val.type = ir.TensorType(complex_dtype)
    _stamp_type_and_shape(complex_val, dim_values)
    _ensure_value_metadata(ctx, complex_val)
    return complex_val
