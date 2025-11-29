# jax2onnx/plugins/jax/lax/scatter_utils.py

"""IR helpers for the lax scatter family in plugins.

The lowering keeps a few key invariants in sync with the ONNX backend:

* guard `Where`/`If` emission so all inputs share an explicit broadcast shape;
* support element-wise scatter (index rank == operand rank) and a prefix-slice
  variant where the remaining axes form a contiguous window;
* harmonize float dtypes (updates follow operand) to avoid ORT type drift.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import onnx_ir as ir
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._loop_extent_meta import set_axis0_override
from jax2onnx.plugins.jax.lax._index_utils import (
    _builder_op,
    _cast_to_i64,
    _const_i64,
    _gather_int_scalar,
    _scalar_i64,
    _shape_of,
    _unsqueeze_scalar,
)


def _maybe_static_extent(value: Any) -> int | None:
    if isinstance(value, (int, np.integer)):
        return int(value)
    return None


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


def _classify_scatter_pattern(spec: ScatterSpec, operand_rank: int) -> str:
    """Return the supported scatter kind: ``"elementwise"`` or ``"slice"``."""

    scatter_axes = tuple(int(a) for a in spec.scatter_dims_to_operand_dims)
    if any(ax < 0 or ax >= operand_rank for ax in scatter_axes):
        raise NotImplementedError("scatter axes out of operand rank range")

    if len(scatter_axes) == operand_rank:
        if spec.update_window_dims:
            raise NotImplementedError(
                "window dims not supported for fully elementwise scatter"
            )
        if tuple(sorted(scatter_axes)) != tuple(range(operand_rank)):
            raise NotImplementedError(
                "scatter axes must cover each operand axis exactly once"
            )
        return "elementwise"

    # For now support slices when scatter axes form a leading prefix.
    expected_prefix = tuple(range(len(scatter_axes)))
    if tuple(scatter_axes) != expected_prefix:
        raise NotImplementedError(
            "scatter lowering currently supports prefix scatter axes only"
        )

    # Basic sanity on metadata for slice updates to avoid overly general cases.
    if spec.inserted_window_dims and tuple(spec.inserted_window_dims) != tuple(
        range(len(spec.inserted_window_dims))
    ):
        raise NotImplementedError("unsupported inserted_window_dims pattern")

    return "slice"


def _compute_window_operand_dims(
    spec: ScatterSpec, operand_rank: int
) -> Tuple[int, ...]:
    """Return operand axes that participate in the window portion of updates."""

    inserted = set(spec.inserted_window_dims)
    scatter_axes = tuple(int(a) for a in spec.scatter_dims_to_operand_dims)

    all_window = [axis for axis in range(operand_rank) if axis not in inserted]
    excl_scatter_window = [axis for axis in all_window if axis not in scatter_axes]

    update_len = len(spec.update_window_dims)
    if update_len == len(all_window):
        return tuple(all_window)
    if update_len == len(excl_scatter_window):
        return tuple(excl_scatter_window)
    raise NotImplementedError(
        "scatter lowering: unsupported update_window_dims configuration"
    )


def _mul_scalars(ctx: Any, lhs: ir.Value, rhs: ir.Value, name_hint: str) -> ir.Value:
    dtype = getattr(lhs.type, "dtype", ir.DataType.INT64)
    return _builder_op(
        ctx,
        "Mul",
        [lhs, rhs],
        name_hint=name_hint,
        dtype=dtype,
        shape=(),
    )


def _make_constant_of_shape(
    ctx: Any,
    shape_tensor: ir.Value,
    value: np.ndarray,
    name_hint: str,
) -> ir.Value:
    return _builder_op(
        ctx,
        "ConstantOfShape",
        [shape_tensor],
        name_hint=name_hint,
        dtype=ir.DataType.INT64,
        shape=(None,),
        attributes={"value": ir.tensor(value)},
    )


def _reshape_indices_to_2d(
    ctx: Any,
    indices_val: ir.Value,
    batch_rank: int,
    index_depth: int,
) -> Tuple[ir.Value, ir.Value]:
    """Return ``(indices_2d, num_updates_scalar)``.

    ``indices_2d`` is shaped ``(N, operand_rank)`` with scatter components
    ordered to match operand axis order.  ``num_updates_scalar`` is an INT64
    scalar ``N`` that can be re-used when reshaping updates.
    """

    indices_shape = _builder_op(
        ctx,
        "Shape",
        [indices_val],
        name_hint="scatter_idx_shape",
        dtype=ir.DataType.INT64,
        shape=(None,),
    )

    axes0 = _const_i64(ctx, np.asarray([0], dtype=np.int64), "scatter_axes0")

    if batch_rank > 0:
        batch_starts = _const_i64(ctx, np.asarray([0], dtype=np.int64), "scatter_bs")
        batch_ends = _const_i64(
            ctx, np.asarray([batch_rank], dtype=np.int64), "scatter_be"
        )
        batch_axes = axes0
        batch_steps = _const_i64(ctx, np.asarray([1], dtype=np.int64), "scatter_bt")
        batch_shape = _builder_op(
            ctx,
            "Slice",
            [indices_shape, batch_starts, batch_ends, batch_axes, batch_steps],
            name_hint="scatter_batch_shape",
            dtype=ir.DataType.INT64,
            shape=(batch_rank,),
        )

        num_updates = _builder_op(
            ctx,
            "ReduceProd",
            [batch_shape],
            name_hint="scatter_num_updates",
            dtype=ir.DataType.INT64,
            shape=(),
            attributes={"keepdims": 0},
        )
    else:
        num_updates = _scalar_i64(ctx, 1, "scatter_num_updates")

    last_start = _const_i64(ctx, np.asarray([batch_rank], dtype=np.int64), "scatter_ls")
    last_end = _const_i64(
        ctx, np.asarray([batch_rank + 1], dtype=np.int64), "scatter_le"
    )
    last_axes = axes0
    last_steps = _const_i64(ctx, np.asarray([1], dtype=np.int64), "scatter_lt")
    depth_vec = _builder_op(
        ctx,
        "Slice",
        [indices_shape, last_start, last_end, last_axes, last_steps],
        name_hint="scatter_depth_vec",
        dtype=ir.DataType.INT64,
        shape=(1,),
    )

    num_updates_vec = _builder_op(
        ctx,
        "Unsqueeze",
        [num_updates, axes0],
        name_hint="scatter_num_updates_vec",
        dtype=ir.DataType.INT64,
        shape=(1,),
    )

    shape_2d = _builder_op(
        ctx,
        "Concat",
        [num_updates_vec, depth_vec],
        name_hint="scatter_indices_shape2d",
        dtype=ir.DataType.INT64,
        shape=(2,),
        attributes={"axis": 0},
    )

    indices_2d = _builder_op(
        ctx,
        "Reshape",
        [indices_val, shape_2d],
        name_hint="scatter_indices_2d",
        dtype=ir.DataType.INT64,
        shape=(None, index_depth),
    )

    return indices_2d, num_updates


def _reorder_indices_columns(
    ctx: Any,
    indices_2d: ir.Value,
    scatter_axes: Sequence[int],
) -> ir.Value:
    """Ensure the final column order matches ``range(operand_rank)``."""

    index_depth = len(scatter_axes)
    order = np.argsort(np.asarray(scatter_axes, dtype=np.int64))
    if np.array_equal(order, np.arange(index_depth, dtype=np.int64)):
        return indices_2d

    order_const = _const_i64(ctx, order, "scatter_order")
    return _builder_op(
        ctx,
        "Gather",
        [indices_2d, order_const],
        name_hint="scatter_indices_reordered",
        dtype=ir.DataType.INT64,
        shape=(None, index_depth),
        attributes={"axis": 1},
    )


def _reshape_updates_flat(
    ctx: Any,
    updates_val: ir.Value,
    num_updates: ir.Value,
) -> ir.Value:
    """Flatten updates to shape ``(N,)`` using ``num_updates`` as dynamic dim."""

    axes0 = _const_i64(ctx, np.asarray([0], dtype=np.int64), "scatter_axes0")
    num_updates_vec = _builder_op(
        ctx,
        "Unsqueeze",
        [num_updates, axes0],
        name_hint="scatter_updates_shape",
        dtype=ir.DataType.INT64,
        shape=(1,),
    )

    updates_flat = _builder_op(
        ctx,
        "Reshape",
        [updates_val, num_updates_vec],
        name_hint="scatter_updates_flat",
        dtype=getattr(updates_val.type, "dtype", None),
        shape=(None,),
    )
    return updates_flat


def _prepare_updates_for_scatternd(
    ctx: Any,
    updates_val: ir.Value,
    num_updates: ir.Value,
    slice_shape: Sequence[Any],
    *,
    operand_val: ir.Value,
    operand_shape: Sequence[Any],
    index_depth: int,
) -> ir.Value:
    """Return updates shaped as expected by ``ScatterND`` for the pattern."""

    if slice_shape:
        axes0 = _const_i64(ctx, np.asarray([0], dtype=np.int64), "scatter_axes0")

        num_updates_vec = _builder_op(
            ctx,
            "Unsqueeze",
            [num_updates, axes0],
            name_hint="scatter_updates_num",
            dtype=ir.DataType.INT64,
            shape=(1,),
        )

        operand_shape_val = _builder_op(
            ctx,
            "Shape",
            [operand_val],
            name_hint="scatter_operand_shape",
            dtype=ir.DataType.INT64,
            shape=(len(operand_shape),),
        )

        slice_start = _const_i64(
            ctx,
            np.asarray([index_depth], dtype=np.int64),
            "scatter_slice_start",
        )
        slice_end = _const_i64(
            ctx,
            np.asarray([len(operand_shape)], dtype=np.int64),
            "scatter_slice_end",
        )
        slice_steps = _const_i64(
            ctx, np.asarray([1], dtype=np.int64), "scatter_slice_step"
        )

        slice_dims = _builder_op(
            ctx,
            "Slice",
            [operand_shape_val, slice_start, slice_end, axes0, slice_steps],
            name_hint="scatter_slice_dims",
            dtype=ir.DataType.INT64,
            shape=(len(slice_shape),),
        )

        target_shape = _builder_op(
            ctx,
            "Concat",
            [num_updates_vec, slice_dims],
            name_hint="scatter_updates_shape",
            dtype=ir.DataType.INT64,
            shape=(1 + len(slice_shape),),
            attributes={"axis": 0},
        )

        updates_shaped = _builder_op(
            ctx,
            "Reshape",
            [updates_val, target_shape],
            name_hint="scatter_updates_shaped",
            dtype=getattr(updates_val.type, "dtype", None),
            shape=(None,) + tuple(slice_shape),
        )
        return updates_shaped

    return _reshape_updates_flat(ctx, updates_val, num_updates)


def _flatten_updates_after_permute(
    ctx: Any,
    updates_perm_val: ir.Value,
    num_updates: ir.Value,
    num_updates_vec: ir.Value,
    window_total: ir.Value,
    window_total_vec: ir.Value,
    *,
    gather_idx: ir.Value | None = None,
) -> tuple[ir.Value, ir.Value, ir.Value, ir.Value]:
    """Reshape permuted updates tensor to ``(N, window_total)`` and flatten."""

    reshape_updates_shape = _builder_op(
        ctx,
        "Concat",
        [num_updates_vec, window_total_vec],
        name_hint="scatter_updates_reshape_shape",
        dtype=ir.DataType.INT64,
        shape=(2,),
        attributes={"axis": 0},
    )

    upd_dtype = getattr(getattr(updates_perm_val, "type", None), "dtype", None)
    updates_2d = _builder_op(
        ctx,
        "Reshape",
        [updates_perm_val, reshape_updates_shape],
        name_hint="scatter_updates_2d",
        dtype=upd_dtype,
        shape=(None, None),
    )

    if gather_idx is not None:
        updates_2d = _builder_op(
            ctx,
            "Gather",
            [updates_2d, gather_idx],
            name_hint="scatter_updates_valid",
            dtype=upd_dtype,
            shape=(None, None),
            attributes={"axis": 0},
        )

    total_updates = _mul_scalars(
        ctx, num_updates, window_total, "scatter_total_updates"
    )
    total_updates_vec = _unsqueeze_scalar(
        ctx, total_updates, 0, "scatter_total_updates_vec"
    )

    upd_dtype = getattr(getattr(updates_perm_val, "type", None), "dtype", None)
    updates_flat = _builder_op(
        ctx,
        "Reshape",
        [updates_2d, total_updates_vec],
        name_hint="scatter_updates_flat",
        dtype=upd_dtype,
        shape=(None,),
    )

    return updates_flat, updates_2d, total_updates, total_updates_vec


def _resolve_operand_to_update_map(
    spec: ScatterSpec, operand_rank: int
) -> tuple[Dict[int, int], Dict[int, int]]:
    """Return mappings from operand axes to update axes.

    ``full_map`` contains every operand axis participating in the window
    portion (including scatter axes). ``window_map`` filters out scatter axes so
    callers can treat them as batch axes when permuting update tensors.
    """

    window_operand_dims = _compute_window_operand_dims(spec, operand_rank)
    update_window_dims = tuple(int(dim) for dim in spec.update_window_dims)

    if len(window_operand_dims) != len(update_window_dims):
        raise NotImplementedError(
            "scatter lowering: mismatched update_window_dims metadata"
        )

    full_map: Dict[int, int] = {
        operand_axis: update_window_dims[i]
        for i, operand_axis in enumerate(window_operand_dims)
    }

    scatter_axes = {int(ax) for ax in spec.scatter_dims_to_operand_dims}
    window_map = {
        axis: update_axis
        for axis, update_axis in full_map.items()
        if axis not in scatter_axes
    }

    return full_map, window_map


def _compute_window_sizes(
    ctx: Any,
    updates_val: ir.Value,
    operand_rank: int,
    operand_to_update: Dict[int, int],
) -> tuple[
    list[ir.Value],
    ir.Value,
    ir.Value,
    ir.Value,
    ir.Value,
    list[ir.Value],
]:
    updates_shape_val = _shape_of(ctx, updates_val, "scatter_updates_shape")

    size_scalars: list[ir.Value] = []
    size_unsqueezed: list[ir.Value] = []
    window_total: ir.Value | None = None

    for axis in range(operand_rank):
        upd_axis = operand_to_update.get(axis)
        if upd_axis is not None:
            size_scalar = None
            upd_shape_dims = getattr(getattr(updates_val, "shape", None), "dims", None)
            if (
                upd_shape_dims is not None
                and len(upd_shape_dims) > upd_axis
                and upd_shape_dims[upd_axis] is not None
            ):
                try:
                    upd_dim_int = int(upd_shape_dims[upd_axis])
                except Exception:
                    upd_dim_int = None
                if upd_dim_int is not None:
                    size_scalar = _scalar_i64(
                        ctx, upd_dim_int, f"scatter_window_size_static_{axis}"
                    )
            if size_scalar is None:
                size_scalar = _gather_int_scalar(
                    ctx, updates_shape_val, upd_axis, f"scatter_window_size_{axis}"
                )
            if os.environ.get("J2O_DEBUG_SCATTER_SIZES") == "1":
                print(
                    "[scatter_window_size]",
                    axis,
                    upd_axis,
                    getattr(getattr(updates_val, "shape", None), "dims", None),
                    flush=True,
                )
        else:
            size_scalar = _scalar_i64(ctx, 1, f"scatter_window_size_const_{axis}")
        size_scalars.append(size_scalar)

        size_unsq = _unsqueeze_scalar(
            ctx, size_scalar, 0, f"scatter_window_size_vec_{axis}"
        )
        size_unsqueezed.append(size_unsq)

        window_total = (
            size_scalar
            if window_total is None
            else _mul_scalars(
                ctx, window_total, size_scalar, f"scatter_window_total_{axis}"
            )
        )

    if window_total is None:
        window_total = _scalar_i64(ctx, 1, "scatter_window_total_one")

    window_shape_val = _builder_op(
        ctx,
        "Concat",
        size_unsqueezed,
        name_hint="scatter_window_shape",
        dtype=ir.DataType.INT64,
        shape=(operand_rank,),
        attributes={"axis": 0},
    )

    window_total_vec = _unsqueeze_scalar(
        ctx, window_total, 0, "scatter_window_total_vec"
    )

    return (
        size_scalars,
        window_total,
        window_shape_val,
        window_total_vec,
        updates_shape_val,
        size_unsqueezed,
    )


def _build_window_offsets_matrix(
    ctx: Any,
    operand_rank: int,
    size_scalars: Sequence[ir.Value],
    window_shape_val: ir.Value,
    window_total_vec: ir.Value,
    zero_scalar: ir.Value,
    one_scalar: ir.Value,
) -> ir.Value:
    axes_range_cols: list[ir.Value] = []
    for axis in range(operand_rank):
        size_scalar = size_scalars[axis]
        range_out = _builder_op(
            ctx,
            "Range",
            [zero_scalar, size_scalar, one_scalar],
            name_hint=f"scatter_range_axis{axis}",
            dtype=ir.DataType.INT64,
            shape=(None,),
        )
        range_out = _builder_op(
            ctx,
            "Identity",
            [range_out],
            name_hint=f"scatter_range_identity_{axis}",
            dtype=ir.DataType.INT64,
            shape=(None,),
        )

        axes_unsq = [i for i in range(operand_rank) if i != axis]
        if axes_unsq:
            axes_tensor = _const_i64(
                ctx, np.asarray(axes_unsq, dtype=np.int64), f"scatter_unsq_axes_{axis}"
            )
            range_unsq = _builder_op(
                ctx,
                "Unsqueeze",
                [range_out, axes_tensor],
                name_hint=f"scatter_range_unsq_{axis}",
                dtype=ir.DataType.INT64,
                shape=tuple([1] * axis + [None] + [1] * (operand_rank - axis - 1)),
            )
        else:
            range_unsq = range_out

        range_b = _builder_op(
            ctx,
            "Expand",
            [range_unsq, window_shape_val],
            name_hint=f"scatter_range_b_{axis}",
            dtype=ir.DataType.INT64,
            shape=tuple([None] * operand_rank),
        )

        range_flat = _builder_op(
            ctx,
            "Reshape",
            [range_b, window_total_vec],
            name_hint=f"scatter_range_flat_{axis}",
            dtype=ir.DataType.INT64,
            shape=(None,),
        )

        axes_last = _const_i64(
            ctx, np.asarray([1], dtype=np.int64), f"scatter_range_unsq_last_{axis}"
        )
        range_col = _builder_op(
            ctx,
            "Unsqueeze",
            [range_flat, axes_last],
            name_hint=f"scatter_range_col_{axis}",
            dtype=ir.DataType.INT64,
            shape=(None, 1),
        )
        axes_range_cols.append(range_col)

    return _builder_op(
        ctx,
        "Concat",
        axes_range_cols,
        name_hint="scatter_window_offsets",
        dtype=ir.DataType.INT64,
        shape=(None, operand_rank),
        attributes={"axis": 1},
    )


def _filter_fill_or_drop_updates(
    ctx: Any,
    indices_2d: ir.Value,
    updates_perm_val: ir.Value,
    *,
    scatter_axes: Sequence[int],
    size_scalars: Sequence[ir.Value],
    operand_shape_val: ir.Value,
    num_updates: ir.Value,
    mode: Any,
) -> tuple[ir.Value, ir.Value, ir.Value, ir.Value | None]:
    if not scatter_axes:
        return indices_2d, updates_perm_val, num_updates, None

    mode_name = getattr(mode, "name", str(mode)).upper() if mode is not None else ""
    if "FILL_OR_DROP" not in mode_name:
        return indices_2d, updates_perm_val, num_updates, None

    zero_scalar = _scalar_i64(ctx, 0, "scatter_zero_ref")
    axes1 = _const_i64(ctx, np.asarray([1], dtype=np.int64), "scatter_axes1")
    mask_val: ir.Value | None = None

    for axis in scatter_axes:
        col_index = scatter_axes.index(axis)
        gather_idx = _const_i64(
            ctx, np.asarray([col_index], dtype=np.int64), f"scatter_fill_idx_{axis}"
        )
        col_val = _builder_op(
            ctx,
            "Gather",
            [indices_2d, gather_idx],
            name_hint=f"scatter_base_col_{axis}",
            dtype=ir.DataType.INT64,
            shape=(None, 1),
            attributes={"axis": 1},
        )

        ge_zero = _builder_op(
            ctx,
            "GreaterOrEqual",
            [col_val, zero_scalar],
            name_hint=f"scatter_ge0_{axis}",
            dtype=ir.DataType.BOOL,
            shape=(None, 1),
        )

        dim_scalar = _gather_int_scalar(
            ctx, operand_shape_val, axis, f"scatter_dim_{axis}"
        )
        limit_scalar = _builder_op(
            ctx,
            "Sub",
            [dim_scalar, size_scalars[axis]],
            name_hint=f"scatter_limit_{axis}",
            dtype=ir.DataType.INT64,
            shape=(),
        )

        le_limit = _builder_op(
            ctx,
            "LessOrEqual",
            [col_val, limit_scalar],
            name_hint=f"scatter_le_{axis}",
            dtype=ir.DataType.BOOL,
            shape=(None, 1),
        )

        axis_valid = _builder_op(
            ctx,
            "And",
            [ge_zero, le_limit],
            name_hint=f"scatter_valid_axis_{axis}",
            dtype=ir.DataType.BOOL,
            shape=(None, 1),
        )

        if mask_val is None:
            mask_val = axis_valid
        else:
            mask_val = _builder_op(
                ctx,
                "And",
                [mask_val, axis_valid],
                name_hint="scatter_valid_mask",
                dtype=ir.DataType.BOOL,
                shape=(None, 1),
            )

    if mask_val is None:
        return indices_2d, updates_perm_val, num_updates, None

    mask_flat = _builder_op(
        ctx,
        "Squeeze",
        [mask_val, axes1],
        name_hint="scatter_valid_flat",
        dtype=ir.DataType.BOOL,
        shape=(None,),
    )

    valid_idx = _builder_op(
        ctx,
        "NonZero",
        [mask_flat],
        name_hint="scatter_valid_idx",
        dtype=ir.DataType.INT64,
        shape=(1, None),
    )

    valid_idx_t = _builder_op(
        ctx,
        "Transpose",
        [valid_idx],
        name_hint="scatter_valid_idx_t",
        dtype=ir.DataType.INT64,
        shape=(None, 1),
        attributes={"perm": (1, 0)},
    )

    valid_idx_flat = _builder_op(
        ctx,
        "Squeeze",
        [valid_idx_t, axes1],
        name_hint="scatter_valid_idx_flat",
        dtype=ir.DataType.INT64,
        shape=(None,),
    )

    valid_shape = _shape_of(ctx, valid_idx_flat, "scatter_valid_shape")
    num_valid = _gather_int_scalar(ctx, valid_shape, 0, "scatter_num_valid")

    depth = getattr(getattr(indices_2d.type, "shape", None), "dims", None)
    depth_dim = int(depth[1]) if depth and depth[1] is not None else None
    indices_filtered = _builder_op(
        ctx,
        "Gather",
        [indices_2d, valid_idx_flat],
        name_hint="scatter_indices_valid",
        dtype=ir.DataType.INT64,
        shape=(None, depth_dim),
        attributes={"axis": 0},
    )

    return indices_filtered, updates_perm_val, num_valid, valid_idx_flat


def _create_zero_column(
    ctx: Any, num_updates_vec: ir.Value, name_hint: str
) -> tuple[ir.Value, ir.Value]:
    one_const = _const_i64(ctx, np.asarray([1], dtype=np.int64), f"{name_hint}_one")
    column_shape = _builder_op(
        ctx,
        "Concat",
        [num_updates_vec, one_const],
        name_hint=f"{name_hint}_shape",
        dtype=ir.DataType.INT64,
        shape=(2,),
        attributes={"axis": 0},
    )

    zero_column = _make_constant_of_shape(
        ctx, column_shape, np.asarray([0], dtype=np.int64), name_hint
    )
    _stamp_type_and_shape(zero_column, (None, 1))
    return zero_column, column_shape


def _build_base_matrix(
    ctx: Any,
    indices_2d: ir.Value,
    scatter_axes: Sequence[int],
    num_updates_vec: ir.Value,
    zero_column: ir.Value,
    column_shape: ir.Value,
    size_scalars: Sequence[ir.Value],
    operand_shape_val: ir.Value,
    mode: Any,
) -> ir.Value:
    scatter_axes = tuple(int(a) for a in scatter_axes)
    operand_rank = len(size_scalars)
    base_cols: list[ir.Value] = []
    mode_name = getattr(mode, "name", str(mode)).upper() if mode is not None else ""
    clip_mode = "CLIP" in mode_name

    zero_scalar = _scalar_i64(ctx, 0, "scatter_zero_scalar")

    for axis in range(operand_rank):
        if axis in scatter_axes:
            col_pos = scatter_axes.index(axis)
            gather_idx = _const_i64(
                ctx, np.asarray([col_pos], dtype=np.int64), f"scatter_base_idx_{axis}"
            )
            col_val = _builder_op(
                ctx,
                "Gather",
                [indices_2d, gather_idx],
                name_hint=f"scatter_base_col_{axis}",
                dtype=ir.DataType.INT64,
                shape=(None, 1),
                attributes={"axis": 1},
            )

            if clip_mode:
                operand_dim = _gather_int_scalar(
                    ctx, operand_shape_val, axis, f"scatter_operand_dim_{axis}"
                )
                window_extent = size_scalars[axis]
                max_start = _builder_op(
                    ctx,
                    "Sub",
                    [operand_dim, window_extent],
                    name_hint=f"scatter_max_start_{axis}",
                    dtype=ir.DataType.INT64,
                    shape=(),
                )

                max_start_nneg = _builder_op(
                    ctx,
                    "Max",
                    [max_start, zero_scalar],
                    name_hint=f"scatter_max_start_nneg_{axis}",
                    dtype=ir.DataType.INT64,
                    shape=(),
                )

                max_start_vec = _unsqueeze_scalar(
                    ctx, max_start_nneg, 0, f"scatter_max_start_vec_{axis}"
                )

                max_broadcast = _builder_op(
                    ctx,
                    "Expand",
                    [max_start_vec, column_shape],
                    name_hint=f"scatter_max_broadcast_{axis}",
                    dtype=ir.DataType.INT64,
                    shape=(None, 1),
                )

                col_nonneg = _builder_op(
                    ctx,
                    "Max",
                    [col_val, zero_column],
                    name_hint=f"scatter_base_ge0_{axis}",
                    dtype=ir.DataType.INT64,
                    shape=(None, 1),
                )

                col_val = _builder_op(
                    ctx,
                    "Min",
                    [col_nonneg, max_broadcast],
                    name_hint=f"scatter_base_clamped_{axis}",
                    dtype=ir.DataType.INT64,
                    shape=(None, 1),
                )

            base_cols.append(col_val)
        else:
            base_cols.append(zero_column)

    base_matrix = _builder_op(
        ctx,
        "Concat",
        base_cols,
        name_hint="scatter_base_matrix",
        dtype=ir.DataType.INT64,
        shape=(None, operand_rank),
        attributes={"axis": 1},
    )
    return base_matrix


def _expand_indices_with_offsets(
    ctx: Any,
    base_matrix: ir.Value,
    window_offsets: ir.Value,
    num_updates_vec: ir.Value,
    window_total_vec: ir.Value,
    operand_rank: int,
    total_updates_vec: ir.Value,
) -> ir.Value:
    one_vec = _const_i64(ctx, np.asarray([1], dtype=np.int64), "scatter_one_vec")
    rank_vec = _const_i64(
        ctx, np.asarray([operand_rank], dtype=np.int64), "scatter_rank_vec"
    )

    reshape_base_shape = _builder_op(
        ctx,
        "Concat",
        [num_updates_vec, one_vec, rank_vec],
        name_hint="scatter_base_reshape_shape",
        dtype=ir.DataType.INT64,
        shape=(3,),
        attributes={"axis": 0},
    )

    base_reshaped = _builder_op(
        ctx,
        "Reshape",
        [base_matrix, reshape_base_shape],
        name_hint="scatter_base_reshaped",
        dtype=ir.DataType.INT64,
        shape=(None, None, operand_rank),
    )

    rank_vec2 = _const_i64(
        ctx, np.asarray([operand_rank], dtype=np.int64), "scatter_rank_vec2"
    )
    expand_shape = _builder_op(
        ctx,
        "Concat",
        [num_updates_vec, window_total_vec, rank_vec2],
        name_hint="scatter_expand_shape",
        dtype=ir.DataType.INT64,
        shape=(3,),
        attributes={"axis": 0},
    )

    base_expanded = _builder_op(
        ctx,
        "Expand",
        [base_reshaped, expand_shape],
        name_hint="scatter_base_expanded",
        dtype=ir.DataType.INT64,
        shape=(None, None, operand_rank),
    )

    one_vec2 = _const_i64(ctx, np.asarray([1], dtype=np.int64), "scatter_one_vec2")
    rank_vec3 = _const_i64(
        ctx, np.asarray([operand_rank], dtype=np.int64), "scatter_rank_vec3"
    )
    reshape_offsets_shape = _builder_op(
        ctx,
        "Concat",
        [one_vec2, window_total_vec, rank_vec3],
        name_hint="scatter_offsets_shape",
        dtype=ir.DataType.INT64,
        shape=(3,),
        attributes={"axis": 0},
    )

    offsets_reshaped = _builder_op(
        ctx,
        "Reshape",
        [window_offsets, reshape_offsets_shape],
        name_hint="scatter_offsets_reshaped",
        dtype=ir.DataType.INT64,
        shape=(1, None, operand_rank),
    )

    indices_expanded = _builder_op(
        ctx,
        "Add",
        [base_expanded, offsets_reshaped],
        name_hint="scatter_indices_expanded",
        dtype=ir.DataType.INT64,
        shape=(None, None, operand_rank),
    )

    rank_vec4 = _const_i64(
        ctx, np.asarray([operand_rank], dtype=np.int64), "scatter_rank_vec4"
    )
    final_shape = _builder_op(
        ctx,
        "Concat",
        [total_updates_vec, rank_vec4],
        name_hint="scatter_indices_final_shape",
        dtype=ir.DataType.INT64,
        shape=(2,),
        attributes={"axis": 0},
    )

    indices_flat = _builder_op(
        ctx,
        "Reshape",
        [indices_expanded, final_shape],
        name_hint="scatter_indices_flat",
        dtype=ir.DataType.INT64,
        shape=(None, operand_rank),
    )
    return indices_flat


def _lower_scatter_window_full(
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
    mode: Any,
) -> bool:
    """Lower general window scatter by expanding to element-wise ScatterND."""

    operand_rank = len(operand_shape)
    scatter_axes = tuple(int(a) for a in spec.scatter_dims_to_operand_dims)
    if operand_rank == 0 or not scatter_axes:
        return False

    try:
        operand_to_update_full, operand_to_update_window = (
            _resolve_operand_to_update_map(spec, operand_rank)
        )
    except NotImplementedError:
        return False

    batch_rank = max(len(indices_shape) - 1, 0)
    indices_i64 = _cast_to_i64(ctx, indices_val, "scatter_indices_i64")
    indices_2d, num_updates = _reshape_indices_to_2d(
        ctx, indices_i64, batch_rank, len(scatter_axes)
    )
    (
        size_scalars,
        window_total,
        window_shape_val,
        window_total_vec,
        _,
        size_unsqueezed,
    ) = _compute_window_sizes(ctx, updates_val, operand_rank, operand_to_update_full)

    hints = getattr(ctx, "_scatter_window_hints", None)
    if hints is None or not isinstance(hints, dict):
        hints = {}
        setattr(ctx, "_scatter_window_hints", hints)
    for axis in range(operand_rank):
        if axis in scatter_axes:
            continue
        hints.setdefault(axis, []).append(size_unsqueezed[axis])
        if os.environ.get("J2O_DEBUG_SCATTER_HINTS") == "1":
            try:
                val = size_unsqueezed[axis]
                dims = getattr(getattr(val, "shape", None), "dims", None)
                print(
                    "[scatter_hint]",
                    axis,
                    getattr(val, "name", None),
                    dims,
                    flush=True,
                )
            except Exception:
                pass

    updates_rank = len(updates_shape)
    window_axes_update = {
        update_axis for update_axis in operand_to_update_window.values()
    }
    batch_axes = [ax for ax in range(updates_rank) if ax not in window_axes_update]

    # Preserve the relative order of scatter-related update axes to keep the
    # perm stable, but place them ahead of the window axes.
    scatter_update_axes = [
        operand_to_update_full[axis]
        for axis in scatter_axes
        if axis in operand_to_update_full
    ]
    for axis in scatter_update_axes:
        if axis not in batch_axes:
            batch_axes.append(axis)
    batch_axes.sort()

    window_axes_ordered = [
        operand_to_update_window[axis]
        for axis in range(operand_rank)
        if axis in operand_to_update_window
    ]
    perm = batch_axes + window_axes_ordered
    updates_perm_val = updates_val
    if perm != list(range(updates_rank)):
        perm_shape_dims: list[Any] = []
        for idx in perm:
            if idx < len(updates_shape):
                perm_shape_dims.append(updates_shape[idx])
            else:
                perm_shape_dims.append(None)
        dtype = getattr(getattr(updates_val, "type", None), "dtype", None)
        updates_perm_val = _builder_op(
            ctx,
            "Transpose",
            [updates_val],
            name_hint="scatter_updates_perm",
            dtype=dtype,
            shape=tuple(perm_shape_dims),
            attributes={"perm": tuple(perm)},
        )

    operand_shape_val = _shape_of(ctx, operand_val, "scatter_operand_shape")
    (
        indices_2d,
        updates_perm_val,
        num_updates,
        gather_idx,
    ) = _filter_fill_or_drop_updates(
        ctx,
        indices_2d,
        updates_perm_val,
        scatter_axes=scatter_axes,
        size_scalars=size_scalars,
        operand_shape_val=operand_shape_val,
        num_updates=num_updates,
        mode=mode,
    )

    num_updates_vec = _unsqueeze_scalar(ctx, num_updates, 0, "scatter_num_updates_vec")

    updates_flat, _, total_updates, total_updates_vec = _flatten_updates_after_permute(
        ctx,
        updates_perm_val,
        num_updates,
        num_updates_vec,
        window_total,
        window_total_vec,
        gather_idx=gather_idx,
    )

    zero_scalar = _scalar_i64(ctx, 0, "scatter_zero")
    one_scalar = _scalar_i64(ctx, 1, "scatter_one")
    window_offsets = _build_window_offsets_matrix(
        ctx,
        operand_rank,
        size_scalars,
        window_shape_val,
        window_total_vec,
        zero_scalar,
        one_scalar,
    )

    zero_column, column_shape = _create_zero_column(
        ctx, num_updates_vec, "scatter_zero_column"
    )
    base_matrix = _build_base_matrix(
        ctx,
        indices_2d,
        scatter_axes,
        num_updates_vec,
        zero_column,
        column_shape,
        size_scalars,
        operand_shape_val,
        mode,
    )

    indices_flat = _expand_indices_with_offsets(
        ctx,
        base_matrix,
        window_offsets,
        num_updates_vec,
        window_total_vec,
        operand_rank,
        total_updates_vec,
    )

    reduction_norm = (reduction or "none").lower()
    if reduction_norm not in {"none", "add", "max", "min", "mul"}:
        raise ValueError(f"unsupported scatter reduction '{reduction}'")

    attr_map = {"reduction": reduction_norm} if reduction_norm != "none" else None
    _builder_op(
        ctx,
        "ScatterND",
        [operand_val, indices_flat, updates_flat],
        name_hint=out_val.name or ctx.fresh_name("ScatterND"),
        dtype=getattr(getattr(operand_val, "type", None), "dtype", None),
        shape=tuple(operand_shape),
        attributes=attr_map,
        output=out_val,
    )

    _stamp_type_and_shape(out_val, tuple(operand_shape))
    out_val.type = ir.TensorType(operand_val.type.dtype)
    out_val.dtype = operand_val.type.dtype
    _ensure_value_metadata(ctx, out_val)
    axis0_extent = _maybe_static_extent(operand_shape[0] if operand_shape else None)
    if axis0_extent and axis0_extent > 1:
        set_axis0_override(out_val, axis0_extent)
    return True


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
    """Lower supported scatter variants to ``ScatterND``."""

    operand_rank = len(operand_shape)
    pattern = _classify_scatter_pattern(spec, operand_rank)

    scatter_axes = spec.scatter_dims_to_operand_dims
    index_depth = len(scatter_axes)

    if indices_shape:
        shape_depth = indices_shape[-1]
        if isinstance(shape_depth, (int, np.integer)):
            if int(shape_depth) != index_depth:
                raise NotImplementedError(
                    "scatter index depth does not match scatter axes"
                )
    batch_rank = max(len(indices_shape) - 1, 0)

    indices_i64 = _cast_to_i64(ctx, indices_val, "scatter_indices_i64")
    indices_2d, num_updates = _reshape_indices_to_2d(
        ctx, indices_i64, batch_rank, index_depth
    )
    indices_ordered = _reorder_indices_columns(ctx, indices_2d, scatter_axes)

    slice_shape = () if pattern == "elementwise" else operand_shape[index_depth:]
    updates_prepared = _prepare_updates_for_scatternd(
        ctx,
        updates_val,
        num_updates,
        slice_shape,
        operand_val=operand_val,
        operand_shape=operand_shape,
        index_depth=index_depth,
    )

    reduction_norm = (reduction or "none").lower()
    if reduction_norm not in {"none", "add", "max", "min", "mul"}:
        raise ValueError(f"unsupported scatter reduction '{reduction}'")

    attr_map = {"reduction": reduction_norm} if reduction_norm != "none" else None
    produced = _builder_op(
        ctx,
        "ScatterND",
        [operand_val, indices_ordered, updates_prepared],
        name_hint=out_val.name or ctx.fresh_name("ScatterND"),
        dtype=getattr(getattr(operand_val, "type", None), "dtype", None),
        shape=tuple(operand_shape),
        attributes=attr_map,
        output=out_val,
    )

    _stamp_type_and_shape(out_val, tuple(operand_shape))
    out_val.type = ir.TensorType(operand_val.type.dtype)
    out_val.dtype = operand_val.type.dtype
    _ensure_value_metadata(ctx, out_val)
    axis0_extent = _maybe_static_extent(operand_shape[0] if operand_shape else None)
    if axis0_extent and axis0_extent > 1:
        set_axis0_override(out_val, axis0_extent)
        set_axis0_override(produced, axis0_extent)


def ensure_supported_mode(mode: Any) -> None:
    """Reject unsupported scatter modes early."""

    if mode is None:
        return
    mode_name = getattr(mode, "name", None)
    if mode_name is not None and mode_name.upper() in {
        "FILL_OR_DROP",
        "PROMISE_IN_BOUNDS",
        "CLIP",
    }:
        return
    as_str = str(mode).upper()
    if any(token in as_str for token in ("FILL_OR_DROP", "PROMISE_IN_BOUNDS", "CLIP")):
        return
    raise NotImplementedError(f"scatter mode '{mode}' not supported in plugins yet")


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
    mode = params.get("mode")
    ensure_supported_mode(mode)

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

    try:
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
        return
    except NotImplementedError:
        pass

    if _lower_scatter_window_full(
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
        mode=mode,
    ):
        return

    raise NotImplementedError("scatter pattern not supported in plugins")
