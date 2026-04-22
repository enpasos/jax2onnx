# jax2onnx/plugins/_attention_utils.py

from __future__ import annotations

from typing import Any, Union

import numpy as np
import onnx_ir as ir

from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins.jax.lax._index_utils import (
    _const_i64,
    _gather_int_scalar,
    _shape_of,
    _unsqueeze_scalar,
)


DimLike = Union[int, str]


def _dtype_enum_from_value(val: ir.Value) -> ir.DataType:
    dtype = getattr(getattr(val, "type", None), "dtype", None)
    if dtype is None:
        raise TypeError("Missing dtype on value; ensure inputs are typed.")
    return dtype


def expand_grouped_kv_heads(
    ctx: Any,
    source_val: ir.Value,
    *,
    q_num_heads: int,
    kv_num_heads: int,
    batch_dim: DimLike,
    seq_dim: DimLike,
    head_dim: DimLike,
    prefix: str,
    op_name: str,
) -> ir.Value:
    if q_num_heads == kv_num_heads:
        return source_val
    if q_num_heads % kv_num_heads != 0:
        raise ValueError(
            f"{op_name} requires q_num_heads to be divisible by kv_num_heads"
        )

    builder = getattr(ctx, "builder", None)
    if builder is None:
        raise AttributeError("IR build context missing builder for GQA expansion")

    group_size = q_num_heads // kv_num_heads
    source_dtype = _dtype_enum_from_value(source_val)

    source_shape = _shape_of(ctx, source_val, f"{prefix}_shape")
    batch_scalar = _gather_int_scalar(ctx, source_shape, 0, f"{prefix}_batch")
    seq_scalar = _gather_int_scalar(ctx, source_shape, 1, f"{prefix}_seq")
    head_scalar = _gather_int_scalar(ctx, source_shape, 3, f"{prefix}_head")

    unsqueeze_axes = _const_i64(
        ctx,
        np.asarray([3], dtype=np.int64),
        f"{prefix}_unsqueeze_axes",
    )
    unsqueezed = builder.Unsqueeze(
        source_val,
        unsqueeze_axes,
        _outputs=[ctx.fresh_name(f"{prefix}_unsqueezed")],
    )
    unsqueezed.type = ir.TensorType(source_dtype)
    _stamp_type_and_shape(unsqueezed, (batch_dim, seq_dim, kv_num_heads, 1, head_dim))
    _ensure_value_metadata(ctx, unsqueezed)

    tile_repeats = _const_i64(
        ctx,
        np.asarray([1, 1, 1, group_size, 1], dtype=np.int64),
        f"{prefix}_tile_repeats",
    )
    tiled = builder.Tile(
        unsqueezed,
        tile_repeats,
        _outputs=[ctx.fresh_name(f"{prefix}_tiled")],
    )
    tiled.type = ir.TensorType(source_dtype)
    _stamp_type_and_shape(
        tiled, (batch_dim, seq_dim, kv_num_heads, group_size, head_dim)
    )
    _ensure_value_metadata(ctx, tiled)

    batch_vec = _unsqueeze_scalar(ctx, batch_scalar, 0, f"{prefix}_batch_vec")
    seq_vec = _unsqueeze_scalar(ctx, seq_scalar, 0, f"{prefix}_seq_vec")
    q_heads_vec = _const_i64(
        ctx,
        np.asarray([q_num_heads], dtype=np.int64),
        f"{prefix}_q_heads_vec",
    )
    head_vec = _unsqueeze_scalar(ctx, head_scalar, 0, f"{prefix}_head_vec")
    reshape_shape = builder.Concat(
        batch_vec,
        seq_vec,
        q_heads_vec,
        head_vec,
        axis=0,
        _outputs=[ctx.fresh_name(f"{prefix}_reshape_shape")],
    )
    reshape_shape.type = ir.TensorType(ir.DataType.INT64)
    _stamp_type_and_shape(reshape_shape, (4,))
    _ensure_value_metadata(ctx, reshape_shape)

    expanded = builder.Reshape(
        tiled,
        reshape_shape,
        _outputs=[ctx.fresh_name(f"{prefix}_expanded")],
    )
    expanded.type = ir.TensorType(source_dtype)
    _stamp_type_and_shape(expanded, (batch_dim, seq_dim, q_num_heads, head_dim))
    _ensure_value_metadata(ctx, expanded)
    return expanded


__all__ = ["DimLike", "expand_grouped_kv_heads"]
