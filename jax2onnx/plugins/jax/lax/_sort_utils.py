# jax2onnx/plugins/jax/lax/_sort_utils.py

from __future__ import annotations

from typing import cast

import numpy as np
import onnx_ir as ir

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.ir_utils import ir_dtype_to_numpy
from jax2onnx.plugins._ir_shapes import (
    DimInput,
    _ensure_value_metadata,
    _stamp_type_and_shape,
)


def _value_dtype(value: ir.Value) -> ir.DataType | None:
    dtype = getattr(getattr(value, "type", None), "dtype", None)
    if isinstance(dtype, ir.DataType):
        return dtype
    dtype = getattr(value, "dtype", None)
    return dtype if isinstance(dtype, ir.DataType) else None


def _is_floating_dtype(dtype: ir.DataType | None) -> bool:
    np_dtype = ir_dtype_to_numpy(dtype, default=None)
    return np_dtype is not None and np.issubdtype(np_dtype, np.floating)


def _as_dim_tuple(shape: tuple[object, ...]) -> tuple[DimInput, ...]:
    return tuple(cast(DimInput, dim) for dim in shape)


def make_nan_last_sort_key(
    ctx: LoweringContextProtocol,
    value: ir.Value,
    shape: tuple[object, ...],
    *,
    name_hint: str,
) -> tuple[ir.Value, bool]:
    dtype = _value_dtype(value)
    if dtype is None or not _is_floating_dtype(dtype):
        return value, False

    np_dtype = ir_dtype_to_numpy(dtype, default=None)
    if np_dtype is None:
        return value, False

    is_nan = cast(
        ir.Value,
        ctx.builder.IsNaN(value, _outputs=[ctx.fresh_name(f"{name_hint}_is_nan")]),
    )
    is_nan.type = ir.TensorType(ir.DataType.BOOL)
    is_nan.dtype = ir.DataType.BOOL
    _stamp_type_and_shape(is_nan, _as_dim_tuple(shape))
    _ensure_value_metadata(ctx, is_nan)

    pos_inf = ctx.builder.add_initializer_from_array(
        name=ctx.fresh_name(f"{name_hint}_pos_inf"),
        array=np.asarray(np.inf, dtype=np_dtype),
    )
    pos_inf.type = ir.TensorType(dtype)
    pos_inf.dtype = dtype
    _stamp_type_and_shape(pos_inf, ())
    _ensure_value_metadata(ctx, pos_inf)

    key = cast(
        ir.Value,
        ctx.builder.Where(
            is_nan,
            pos_inf,
            value,
            _outputs=[ctx.fresh_name(f"{name_hint}_key")],
        ),
    )
    key.type = ir.TensorType(dtype)
    key.dtype = dtype
    _stamp_type_and_shape(key, _as_dim_tuple(shape))
    _ensure_value_metadata(ctx, key)
    return key, True


def gather_original_by_sort_indices(
    ctx: LoweringContextProtocol,
    value: ir.Value,
    indices: ir.Value,
    *,
    axis: int,
    shape: tuple[object, ...],
    name_hint: str,
) -> ir.Value:
    gathered = cast(
        ir.Value,
        ctx.builder.GatherElements(
            value,
            indices,
            axis=int(axis),
            _outputs=[ctx.fresh_name(name_hint)],
        ),
    )
    value_dtype = _value_dtype(value)
    if value_dtype is not None:
        gathered.type = ir.TensorType(value_dtype)
        gathered.dtype = value_dtype
    _stamp_type_and_shape(gathered, _as_dim_tuple(shape))
    _ensure_value_metadata(ctx, gathered)
    return gathered
