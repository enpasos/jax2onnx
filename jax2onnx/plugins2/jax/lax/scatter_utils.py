"""IR helpers for lax scatter family in plugins2.

This module currently implements a narrow path that covers element-wise
scatter updates where each index resolves a concrete position across every
operand axis (i.e. the scatter depth equals the operand rank and no window
dimensions are present).  That is enough for the first batch of tests we
enable for the plugins2 port; support for windowed scatter updates will be
added incrementally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple

import numpy as np
import onnx_ir as ir
from onnx_ir import Attr as IRAttr, AttributeType as IRAttrType

from jax2onnx.plugins2._ir_shapes import _ensure_value_info, _stamp_type_and_shape
from jax2onnx.plugins2.jax.lax._index_utils import (
    _cast_to_i64,
    _const_i64,
    _scalar_i64,
)


@dataclass(frozen=True)
class ScatterSpec:
    """Minimal shape metadata extracted from ``ScatterDimensionNumbers``."""

    update_window_dims: Tuple[int, ...]
    inserted_window_dims: Tuple[int, ...]
    scatter_dims_to_operand_dims: Tuple[int, ...]


def _normalize_dimension_numbers(dnums_like: Any) -> ScatterSpec:
    """Convert a lax ``ScatterDimensionNumbers`` (or dict) into ``ScatterSpec``."""

    if dnums_like is None:
        raise ValueError("scatter lowering requires dimension_numbers")

    def _get(name: str) -> Tuple[int, ...]:
        if hasattr(dnums_like, name):
            value = getattr(dnums_like, name)
        elif isinstance(dnums_like, dict):
            value = dnums_like.get(name, ())
        else:
            raise ValueError(f"scatter lowering missing field '{name}'")
        return tuple(int(v) for v in value)

    return ScatterSpec(
        update_window_dims=_get("update_window_dims"),
        inserted_window_dims=_get("inserted_window_dims"),
        scatter_dims_to_operand_dims=_get("scatter_dims_to_operand_dims"),
    )


def _require_simple_elementwise(spec: ScatterSpec, operand_rank: int) -> None:
    """Ensure the current lowering only sees element-wise scatter patterns."""

    if spec.update_window_dims:
        raise NotImplementedError(
            "scatter window dimensions are not supported in plugins2 yet"
        )
    scatter_axes = spec.scatter_dims_to_operand_dims
    if len(scatter_axes) != operand_rank:
        raise NotImplementedError(
            "scatter lowering currently supports only full-depth indices"
        )
    if tuple(sorted(scatter_axes)) != tuple(range(operand_rank)):
        raise NotImplementedError(
            "scatter axes must cover each operand axis exactly once"
        )


def _reshape_indices_to_2d(
    ctx: Any,
    indices_val: ir.Value,
    batch_rank: int,
    operand_rank: int,
) -> Tuple[ir.Value, ir.Value]:
    """Return ``(indices_2d, num_updates_scalar)``.

    ``indices_2d`` is shaped ``(N, operand_rank)`` with scatter components
    ordered to match operand axis order.  ``num_updates_scalar`` is an INT64
    scalar ``N`` that can be re-used when reshaping updates.
    """

    indices_shape = ir.Value(
        name=ctx.fresh_name("scatter_idx_shape"),
        type=ir.TensorType(ir.DataType.INT64),
        shape=ir.Shape((None,)),
    )
    ctx.add_node(
        ir.Node(
            op_type="Shape",
            domain="",
            inputs=[indices_val],
            outputs=[indices_shape],
            name=ctx.fresh_name("Shape"),
        )
    )
    _ensure_value_info(ctx, indices_shape)

    axes0 = _const_i64(ctx, np.asarray([0], dtype=np.int64), "scatter_axes0")

    if batch_rank > 0:
        batch_starts = _const_i64(ctx, np.asarray([0], dtype=np.int64), "scatter_bs")
        batch_ends = _const_i64(
            ctx, np.asarray([batch_rank], dtype=np.int64), "scatter_be"
        )
        batch_axes = axes0
        batch_steps = _const_i64(ctx, np.asarray([1], dtype=np.int64), "scatter_bt")
        batch_shape = ir.Value(
            name=ctx.fresh_name("scatter_batch_shape"),
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape((batch_rank,)),
        )
        ctx.add_node(
            ir.Node(
                op_type="Slice",
                domain="",
                inputs=[
                    indices_shape,
                    batch_starts,
                    batch_ends,
                    batch_axes,
                    batch_steps,
                ],
                outputs=[batch_shape],
                name=ctx.fresh_name("Slice"),
            )
        )
        _ensure_value_info(ctx, batch_shape)

        num_updates = ir.Value(
            name=ctx.fresh_name("scatter_num_updates"),
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape(()),
        )
        ctx.add_node(
            ir.Node(
                op_type="ReduceProd",
                domain="",
                inputs=[batch_shape],
                outputs=[num_updates],
                name=ctx.fresh_name("ReduceProd"),
                attributes=[IRAttr("keepdims", IRAttrType.INT, 0)],
            )
        )
        _ensure_value_info(ctx, num_updates)
    else:
        num_updates = _scalar_i64(ctx, 1, "scatter_num_updates")

    last_start = _const_i64(ctx, np.asarray([batch_rank], dtype=np.int64), "scatter_ls")
    last_end = _const_i64(
        ctx, np.asarray([batch_rank + 1], dtype=np.int64), "scatter_le"
    )
    last_axes = axes0
    last_steps = _const_i64(ctx, np.asarray([1], dtype=np.int64), "scatter_lt")
    depth_vec = ir.Value(
        name=ctx.fresh_name("scatter_depth_vec"),
        type=ir.TensorType(ir.DataType.INT64),
        shape=ir.Shape((1,)),
    )
    ctx.add_node(
        ir.Node(
            op_type="Slice",
            domain="",
            inputs=[indices_shape, last_start, last_end, last_axes, last_steps],
            outputs=[depth_vec],
            name=ctx.fresh_name("Slice"),
        )
    )
    _ensure_value_info(ctx, depth_vec)

    num_updates_vec = ir.Value(
        name=ctx.fresh_name("scatter_num_updates_vec"),
        type=ir.TensorType(ir.DataType.INT64),
        shape=ir.Shape((1,)),
    )
    ctx.add_node(
        ir.Node(
            op_type="Unsqueeze",
            domain="",
            inputs=[num_updates, axes0],
            outputs=[num_updates_vec],
            name=ctx.fresh_name("Unsqueeze"),
        )
    )
    _ensure_value_info(ctx, num_updates_vec)

    shape_2d = ir.Value(
        name=ctx.fresh_name("scatter_indices_shape2d"),
        type=ir.TensorType(ir.DataType.INT64),
        shape=ir.Shape((2,)),
    )
    ctx.add_node(
        ir.Node(
            op_type="Concat",
            domain="",
            inputs=[num_updates_vec, depth_vec],
            outputs=[shape_2d],
            name=ctx.fresh_name("Concat"),
            attributes=[IRAttr("axis", IRAttrType.INT, 0)],
        )
    )
    _ensure_value_info(ctx, shape_2d)

    indices_2d = ir.Value(
        name=ctx.fresh_name("scatter_indices_2d"),
        type=ir.TensorType(ir.DataType.INT64),
        shape=ir.Shape((None, operand_rank)),
    )
    ctx.add_node(
        ir.Node(
            op_type="Reshape",
            domain="",
            inputs=[indices_val, shape_2d],
            outputs=[indices_2d],
            name=ctx.fresh_name("Reshape"),
        )
    )
    _ensure_value_info(ctx, indices_2d)

    return indices_2d, num_updates


def _reorder_indices_columns(
    ctx: Any,
    indices_2d: ir.Value,
    scatter_axes: Sequence[int],
) -> ir.Value:
    """Ensure the final column order matches ``range(operand_rank)``."""

    operand_rank = len(scatter_axes)
    order = np.argsort(np.asarray(scatter_axes, dtype=np.int64))
    if np.array_equal(order, np.arange(operand_rank, dtype=np.int64)):
        return indices_2d

    order_const = _const_i64(ctx, order, "scatter_order")
    reordered = ir.Value(
        name=ctx.fresh_name("scatter_indices_reordered"),
        type=ir.TensorType(ir.DataType.INT64),
        shape=ir.Shape((None, operand_rank)),
    )
    ctx.add_node(
        ir.Node(
            op_type="Gather",
            domain="",
            inputs=[indices_2d, order_const],
            outputs=[reordered],
            name=ctx.fresh_name("Gather"),
            attributes=[IRAttr("axis", IRAttrType.INT, 1)],
        )
    )
    _ensure_value_info(ctx, reordered)
    return reordered


def _reshape_updates_flat(
    ctx: Any,
    updates_val: ir.Value,
    num_updates: ir.Value,
) -> ir.Value:
    """Flatten updates to shape ``(N,)`` using ``num_updates`` as dynamic dim."""

    axes0 = _const_i64(ctx, np.asarray([0], dtype=np.int64), "scatter_axes0")
    num_updates_vec = ir.Value(
        name=ctx.fresh_name("scatter_updates_shape"),
        type=ir.TensorType(ir.DataType.INT64),
        shape=ir.Shape((1,)),
    )
    ctx.add_node(
        ir.Node(
            op_type="Unsqueeze",
            domain="",
            inputs=[num_updates, axes0],
            outputs=[num_updates_vec],
            name=ctx.fresh_name("Unsqueeze"),
        )
    )
    _ensure_value_info(ctx, num_updates_vec)

    updates_flat = ir.Value(
        name=ctx.fresh_name("scatter_updates_flat"),
        type=ir.TensorType(updates_val.type.dtype),
        shape=ir.Shape((None,)),
    )
    ctx.add_node(
        ir.Node(
            op_type="Reshape",
            domain="",
            inputs=[updates_val, num_updates_vec],
            outputs=[updates_flat],
            name=ctx.fresh_name("Reshape"),
        )
    )
    _ensure_value_info(ctx, updates_flat)
    return updates_flat


def lower_scatter_elementwise(
    ctx: Any,
    *,
    operand_val: ir.Value,
    indices_val: ir.Value,
    updates_val: ir.Value,
    operand_shape: Sequence[Any],
    indices_shape: Sequence[Any],
    updates_shape: Sequence[Any],
    spec: ScatterSpec,
    reduction: str,
    out_val: ir.Value,
) -> None:
    """Lower the supported element-wise scatter variant to ``ScatterND``."""

    operand_rank = len(operand_shape)
    _require_simple_elementwise(spec, operand_rank)

    scatter_axes = spec.scatter_dims_to_operand_dims

    if indices_shape:
        index_depth = indices_shape[-1]
        if (
            isinstance(index_depth, (int, np.integer))
            and int(index_depth) != operand_rank
        ):
            raise NotImplementedError(
                "scatter lowering expects index depth equal to operand rank"
            )
    batch_rank = max(len(indices_shape) - 1, 0)

    indices_i64 = _cast_to_i64(ctx, indices_val, "scatter_indices_i64")
    indices_2d, num_updates = _reshape_indices_to_2d(
        ctx, indices_i64, batch_rank, operand_rank
    )
    indices_ordered = _reorder_indices_columns(ctx, indices_2d, scatter_axes)

    updates_flat = _reshape_updates_flat(ctx, updates_val, num_updates)

    reduction_norm = (reduction or "none").lower()
    if reduction_norm not in {"none", "add", "max", "min", "mul"}:
        raise ValueError(f"unsupported scatter reduction '{reduction}'")

    attributes = []
    if reduction_norm != "none":
        attributes.append(IRAttr("reduction", IRAttrType.STRING, reduction_norm))

    scatter_node = ir.Node(
        op_type="ScatterND",
        domain="",
        inputs=[operand_val, indices_ordered, updates_flat],
        outputs=[out_val],
        name=ctx.fresh_name("ScatterND"),
        attributes=attributes,
    )

    ctx.add_node(scatter_node)

    _stamp_type_and_shape(out_val, tuple(operand_shape))
    out_val.type = ir.TensorType(operand_val.type.dtype)
    out_val.dtype = operand_val.type.dtype
    _ensure_value_info(ctx, out_val)


def ensure_supported_mode(mode: Any) -> None:
    """Reject unsupported scatter modes early."""

    if mode is None:
        return
    mode_name = getattr(mode, "name", None)
    if mode_name is not None and mode_name.upper() in {
        "FILL_OR_DROP",
        "PROMISE_IN_BOUNDS",
    }:
        return
    as_str = str(mode).upper()
    if any(token in as_str for token in ("FILL_OR_DROP", "PROMISE_IN_BOUNDS")):
        return
    raise NotImplementedError(f"scatter mode '{mode}' not supported in plugins2 yet")


def lower_scatter_common(
    ctx: Any,
    eqn,
    *,
    reduction: str,
) -> None:
    """Shared lowering for scatter, scatter_add, scatter_min/max/mul."""

    operand_var, indices_var, updates_var = eqn.invars
    out_var = eqn.outvars[0]

    params = getattr(eqn, "params", {})
    spec = _normalize_dimension_numbers(params.get("dimension_numbers"))
    ensure_supported_mode(params.get("mode"))

    operand_val = ctx.get_value_for_var(
        operand_var, name_hint=ctx.fresh_name("scatter_operand")
    )
    indices_val = ctx.get_value_for_var(
        indices_var, name_hint=ctx.fresh_name("scatter_indices")
    )
    updates_val = ctx.get_value_for_var(
        updates_var, name_hint=ctx.fresh_name("scatter_updates")
    )
    out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("scatter_out"))

    operand_shape = tuple(getattr(operand_var.aval, "shape", ()))
    indices_shape = tuple(getattr(indices_var.aval, "shape", ()))
    updates_shape = tuple(getattr(updates_var.aval, "shape", ()))

    lower_scatter_elementwise(
        ctx,
        operand_val=operand_val,
        indices_val=indices_val,
        updates_val=updates_val,
        operand_shape=operand_shape,
        indices_shape=indices_shape,
        updates_shape=updates_shape,
        spec=spec,
        reduction=reduction,
        out_val=out_val,
    )
