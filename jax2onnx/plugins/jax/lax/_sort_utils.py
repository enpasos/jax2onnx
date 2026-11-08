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
) -> tuple[ir.Value, ir.Value | None]:
    """Return a numeric key and the original NaN classification, if needed."""
    dtype = _value_dtype(value)
    if dtype is None or not _is_floating_dtype(dtype):
        return value, None

    np_dtype = ir_dtype_to_numpy(dtype, default=None)
    if np_dtype is None:
        return value, None

    is_nan = ctx.builder.op(
        "IsNaN",
        [value],
        output=ir.Value(name=ctx.fresh_name(f"{name_hint}_is_nan")),
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

    key = ctx.builder.op(
        "Where",
        [is_nan, pos_inf, value],
        output=ir.Value(name=ctx.fresh_name(f"{name_hint}_key")),
    )
    key.type = ir.TensorType(dtype)
    key.dtype = dtype
    _stamp_type_and_shape(key, _as_dim_tuple(shape))
    _ensure_value_metadata(ctx, key)
    return key, is_nan


def gather_original_by_sort_indices(
    ctx: LoweringContextProtocol,
    value: ir.Value,
    indices: ir.Value,
    *,
    axis: int,
    shape: tuple[object, ...],
    name_hint: str,
) -> ir.Value:
    gathered = ctx.builder.op(
        "GatherElements",
        [value, indices],
        {"axis": int(axis)},
        output=ir.Value(name=ctx.fresh_name(name_hint)),
    )
    value_dtype = _value_dtype(value)
    if value_dtype is not None:
        gathered.type = ir.TensorType(value_dtype)
        gathered.dtype = value_dtype
    _stamp_type_and_shape(gathered, _as_dim_tuple(shape))
    _ensure_value_metadata(ctx, gathered)
    return gathered


def stabilize_nan_last_sort_indices(
    ctx: LoweringContextProtocol,
    indices: ir.Value,
    is_nan: ir.Value,
    k: ir.Value,
    *,
    axis: int,
    shape: tuple[object, ...],
    name_hint: str,
) -> ir.Value:
    """Stably partition numeric-sort indices into non-NaNs followed by NaNs.

    The numeric key maps NaNs to +Inf, so those two groups can tie. Sorted TopK
    breaks ties by lower input index; a second TopK on the original NaN mask
    therefore separates the groups without changing either group's order.
    Compose the permutations, leaving callers to gather untouched source data.
    """
    sorted_is_nan = gather_original_by_sort_indices(
        ctx,
        is_nan,
        indices,
        axis=axis,
        shape=shape,
        name_hint=f"{name_hint}_sorted_is_nan",
    )
    partition_key = ctx.builder.op(
        "Cast",
        [sorted_is_nan],
        {"to": int(ir.DataType.INT32)},
        output=ir.Value(
            name=ctx.fresh_name(f"{name_hint}_partition_key"),
            type=ir.TensorType(ir.DataType.INT32),
        ),
    )
    _stamp_type_and_shape(partition_key, _as_dim_tuple(shape))
    _ensure_value_metadata(ctx, partition_key)

    partition_values = ir.Value(
        name=ctx.fresh_name(f"{name_hint}_partition_values"),
        type=ir.TensorType(ir.DataType.INT32),
    )
    partition_indices = ir.Value(
        name=ctx.fresh_name(f"{name_hint}_partition_indices"),
        type=ir.TensorType(ir.DataType.INT64),
    )
    ctx.builder.op_multi_out(
        "TopK",
        [partition_key, k],
        {"axis": int(axis), "largest": 0, "sorted": 1},
        outputs=[partition_values, partition_indices],
    )
    for output in (partition_values, partition_indices):
        _stamp_type_and_shape(output, _as_dim_tuple(shape))
        _ensure_value_metadata(ctx, output)

    return gather_original_by_sort_indices(
        ctx,
        indices,
        partition_indices,
        axis=axis,
        shape=shape,
        name_hint=f"{name_hint}_indices",
    )
