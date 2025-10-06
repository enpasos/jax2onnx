# jax2onnx/plugins/jax/lax/scatter_utils.py

"""IR helpers for the lax scatter family in plugins.

The lowering keeps a few key invariants in sync with the ONNX backend:

* guard `Where`/`If` emission so all inputs share an explicit broadcast shape;
* support element-wise scatter (index rank == operand rank) and a prefix-slice
  variant where the remaining axes form a contiguous window;
* harmonize float dtypes (updates follow operand) to avoid ORT type drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import onnx_ir as ir
from jax2onnx.plugins._ir_shapes import _ensure_value_info, _stamp_type_and_shape
from jax2onnx.plugins.jax.lax._index_utils import (
    _cast_to_i64,
    _const_i64,
    _scalar_i64,
)


def _builder_op(
    ctx: Any,
    op_type: str,
    inputs: Sequence[ir.Value | None],
    *,
    name_hint: str,
    dtype: ir.DataType | None = None,
    shape: Sequence[int | None] | None = None,
    attributes: Dict[str, Any] | None = None,
    output: ir.Value | None = None,
) -> ir.Value:
    if output is None:
        output = ir.Value(
            name=ctx.fresh_name(name_hint),
            type=ir.TensorType(dtype) if dtype is not None else None,
            shape=ir.Shape(tuple(shape)) if shape is not None else None,
        )
    else:
        if dtype is not None:
            output.type = ir.TensorType(dtype)
        if shape is not None:
            output.shape = ir.Shape(tuple(shape))
    result = ctx.builder.op(
        op_type,
        list(inputs),
        attributes or {},
        output=output,
        name=output.name,
    )
    if dtype is not None:
        result.type = ir.TensorType(dtype)
    if shape is not None:
        _stamp_type_and_shape(result, tuple(shape))
    _ensure_value_info(ctx, result)
    return result


def _emit_into(
    ctx: Any,
    op_type: str,
    inputs: Sequence[ir.Value | None],
    *,
    output: ir.Value,
    attributes: Dict[str, Any] | None = None,
) -> ir.Value:
    dtype = getattr(getattr(output, "type", None), "dtype", None)
    shape_obj = getattr(output, "shape", None)
    shape = None
    if shape_obj is not None and hasattr(shape_obj, "dims"):
        dims = tuple(shape_obj.dims)
        shape = dims
    return _builder_op(
        ctx,
        op_type,
        inputs,
        name_hint=output.name,
        dtype=dtype,
        shape=shape,
        attributes=attributes,
        output=output,
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


def _shape_of(ctx: Any, value: ir.Value, name_hint: str) -> ir.Value:
    return _builder_op(
        ctx,
        "Shape",
        [value],
        name_hint=name_hint,
        dtype=ir.DataType.INT64,
        shape=(None,),
    )


def _gather_int_scalar(
    ctx: Any, shape_val: ir.Value, axis: int, name_hint: str
) -> ir.Value:
    indices = _const_i64(ctx, np.asarray([axis], dtype=np.int64), f"{name_hint}_idx")
    gathered = _builder_op(
        ctx,
        "Gather",
        [shape_val, indices],
        name_hint=name_hint,
        dtype=ir.DataType.INT64,
        shape=(1,),
        attributes={"axis": 0},
    )

    axes = _const_i64(ctx, np.asarray([0], dtype=np.int64), f"{name_hint}_sq")
    scalar = _builder_op(
        ctx,
        "Squeeze",
        [gathered, axes],
        name_hint=f"{name_hint}_scalar",
        dtype=ir.DataType.INT64,
        shape=(),
    )
    return scalar


def _unsqueeze_scalar(
    ctx: Any, scalar: ir.Value, axis: int, name_hint: str
) -> ir.Value:
    axes = _const_i64(ctx, np.asarray([axis], dtype=np.int64), f"{name_hint}_axes")
    return _builder_op(
        ctx,
        "Unsqueeze",
        [scalar, axes],
        name_hint=name_hint,
        dtype=ir.DataType.INT64,
        shape=(1,),
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

    reshape_updates_shape = ir.Value(
        name=ctx.fresh_name("scatter_updates_reshape_shape"),
        type=ir.TensorType(ir.DataType.INT64),
        shape=ir.Shape((2,)),
    )
    _builder_op(
        ctx,
        "Concat",
        [num_updates_vec, window_total_vec],
        name_hint="scatter_updates_reshape_shape",
        dtype=ir.DataType.INT64,
        shape=(2,),
        attributes={"axis": 0},
        output=reshape_updates_shape,
    )
    _ensure_value_info(ctx, reshape_updates_shape)

    updates_2d = ir.Value(
        name=ctx.fresh_name("scatter_updates_2d"),
        type=ir.TensorType(updates_perm_val.type.dtype),
        shape=ir.Shape((None, None)),
    )
    upd_dtype = getattr(getattr(updates_perm_val, "type", None), "dtype", None)
    _builder_op(
        ctx,
        "Reshape",
        [updates_perm_val, reshape_updates_shape],
        name_hint="scatter_updates_2d",
        dtype=upd_dtype,
        shape=(None, None),
        output=updates_2d,
    )
    _ensure_value_info(ctx, updates_2d)

    if gather_idx is not None:
        updates_2d_filtered = ir.Value(
            name=ctx.fresh_name("scatter_updates_valid"),
            type=ir.TensorType(updates_perm_val.type.dtype),
            shape=ir.Shape((None, None)),
        )
        _builder_op(
            ctx,
            "Gather",
            [updates_2d, gather_idx],
            name_hint="scatter_updates_valid",
            dtype=upd_dtype,
            shape=(None, None),
            attributes={"axis": 0},
            output=updates_2d_filtered,
        )
        _ensure_value_info(ctx, updates_2d_filtered)
        updates_2d = updates_2d_filtered

    total_updates = _mul_scalars(
        ctx, num_updates, window_total, "scatter_total_updates"
    )
    total_updates_vec = _unsqueeze_scalar(
        ctx, total_updates, 0, "scatter_total_updates_vec"
    )

    updates_flat = ir.Value(
        name=ctx.fresh_name("scatter_updates_flat"),
        type=ir.TensorType(updates_perm_val.type.dtype),
        shape=ir.Shape((None,)),
    )
    upd_dtype = getattr(getattr(updates_perm_val, "type", None), "dtype", None)
    _builder_op(
        ctx,
        "Reshape",
        [updates_2d, total_updates_vec],
        name_hint="scatter_updates_flat",
        dtype=upd_dtype,
        shape=(None,),
        output=updates_flat,
    )
    _ensure_value_info(ctx, updates_flat)

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
            size_scalar = _gather_int_scalar(
                ctx, updates_shape_val, upd_axis, f"scatter_window_size_{axis}"
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
        col_val = ir.Value(
            name=ctx.fresh_name(f"scatter_base_col_{axis}"),
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape((None, 1)),
        )
        _emit_into(
            ctx,
            "Gather",
            [indices_2d, gather_idx],
            output=col_val,
            attributes={"axis": 1},
        )
        _ensure_value_info(ctx, col_val)

        ge_zero = ir.Value(
            name=ctx.fresh_name(f"scatter_ge0_{axis}"),
            type=ir.TensorType(ir.DataType.BOOL),
            shape=ir.Shape((None, 1)),
        )
        _emit_into(
            ctx,
            "GreaterOrEqual",
            [col_val, zero_scalar],
            output=ge_zero,
        )
        _ensure_value_info(ctx, ge_zero)

        dim_scalar = _gather_int_scalar(
            ctx, operand_shape_val, axis, f"scatter_dim_{axis}"
        )
        limit_scalar = ir.Value(
            name=ctx.fresh_name(f"scatter_limit_{axis}"),
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape(()),
        )
        _emit_into(
            ctx,
            "Sub",
            [dim_scalar, size_scalars[axis]],
            output=limit_scalar,
        )
        _ensure_value_info(ctx, limit_scalar)

        le_limit = ir.Value(
            name=ctx.fresh_name(f"scatter_le_{axis}"),
            type=ir.TensorType(ir.DataType.BOOL),
            shape=ir.Shape((None, 1)),
        )
        _emit_into(
            ctx,
            "LessOrEqual",
            [col_val, limit_scalar],
            output=le_limit,
        )
        _ensure_value_info(ctx, le_limit)

        axis_valid = ir.Value(
            name=ctx.fresh_name(f"scatter_valid_axis_{axis}"),
            type=ir.TensorType(ir.DataType.BOOL),
            shape=ir.Shape((None, 1)),
        )
        _emit_into(
            ctx,
            "And",
            [ge_zero, le_limit],
            output=axis_valid,
        )
        _ensure_value_info(ctx, axis_valid)

        if mask_val is None:
            mask_val = axis_valid
        else:
            combined = ir.Value(
                name=ctx.fresh_name("scatter_valid_mask"),
                type=ir.TensorType(ir.DataType.BOOL),
                shape=ir.Shape((None, 1)),
            )
            _emit_into(
                ctx,
                "And",
                [mask_val, axis_valid],
                output=combined,
            )
            _ensure_value_info(ctx, combined)
            mask_val = combined

    if mask_val is None:
        return indices_2d, updates_perm_val, num_updates, None

    mask_flat = ir.Value(
        name=ctx.fresh_name("scatter_valid_flat"),
        type=ir.TensorType(ir.DataType.BOOL),
        shape=ir.Shape((None,)),
    )
    _emit_into(
        ctx,
        "Squeeze",
        [mask_val, axes1],
        output=mask_flat,
    )
    _ensure_value_info(ctx, mask_flat)

    valid_idx = ir.Value(
        name=ctx.fresh_name("scatter_valid_idx"),
        type=ir.TensorType(ir.DataType.INT64),
        shape=ir.Shape((1, None)),
    )
    _emit_into(
        ctx,
        "NonZero",
        [mask_flat],
        output=valid_idx,
    )
    _ensure_value_info(ctx, valid_idx)

    valid_idx_t = ir.Value(
        name=ctx.fresh_name("scatter_valid_idx_t"),
        type=ir.TensorType(ir.DataType.INT64),
        shape=ir.Shape((None, 1)),
    )
    _emit_into(
        ctx,
        "Transpose",
        [valid_idx],
        output=valid_idx_t,
        attributes={"perm": (1, 0)},
    )
    _ensure_value_info(ctx, valid_idx_t)

    valid_idx_flat = ir.Value(
        name=ctx.fresh_name("scatter_valid_idx_flat"),
        type=ir.TensorType(ir.DataType.INT64),
        shape=ir.Shape((None,)),
    )
    _emit_into(
        ctx,
        "Squeeze",
        [valid_idx_t, axes1],
        output=valid_idx_flat,
    )
    _ensure_value_info(ctx, valid_idx_flat)

    valid_shape = _shape_of(ctx, valid_idx_flat, "scatter_valid_shape")
    num_valid = _gather_int_scalar(ctx, valid_shape, 0, "scatter_num_valid")

    depth = getattr(getattr(indices_2d.type, "shape", None), "dims", None)
    depth_dim = int(depth[1]) if depth and depth[1] is not None else None
    indices_filtered = ir.Value(
        name=ctx.fresh_name("scatter_indices_valid"),
        type=ir.TensorType(ir.DataType.INT64),
        shape=ir.Shape((None, depth_dim)),
    )
    _emit_into(
        ctx,
        "Gather",
        [indices_2d, valid_idx_flat],
        output=indices_filtered,
        attributes={"axis": 0},
    )
    _ensure_value_info(ctx, indices_filtered)

    return indices_filtered, updates_perm_val, num_valid, valid_idx_flat


def _create_zero_column(
    ctx: Any, num_updates_vec: ir.Value, name_hint: str
) -> tuple[ir.Value, ir.Value]:
    column_shape = ir.Value(
        name=ctx.fresh_name(f"{name_hint}_shape"),
        type=ir.TensorType(ir.DataType.INT64),
        shape=ir.Shape((2,)),
    )
    one_const = _const_i64(ctx, np.asarray([1], dtype=np.int64), f"{name_hint}_one")
    _emit_into(
        ctx,
        "Concat",
        [num_updates_vec, one_const],
        output=column_shape,
        attributes={"axis": 0},
    )
    _ensure_value_info(ctx, column_shape)

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
            col_val = ir.Value(
                name=ctx.fresh_name(f"scatter_base_col_{axis}"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape((None, 1)),
            )
            _emit_into(
                ctx,
                "Gather",
                [indices_2d, gather_idx],
                output=col_val,
                attributes={"axis": 1},
            )
            _ensure_value_info(ctx, col_val)

            if clip_mode:
                operand_dim = _gather_int_scalar(
                    ctx, operand_shape_val, axis, f"scatter_operand_dim_{axis}"
                )
                window_extent = size_scalars[axis]
                max_start = ir.Value(
                    name=ctx.fresh_name(f"scatter_max_start_{axis}"),
                    type=ir.TensorType(ir.DataType.INT64),
                    shape=ir.Shape(()),
                )
                _emit_into(
                    ctx,
                    "Sub",
                    [operand_dim, window_extent],
                    output=max_start,
                )
                _ensure_value_info(ctx, max_start)

                max_start_nneg = ir.Value(
                    name=ctx.fresh_name(f"scatter_max_start_nneg_{axis}"),
                    type=ir.TensorType(ir.DataType.INT64),
                    shape=ir.Shape(()),
                )
                _emit_into(
                    ctx,
                    "Max",
                    [max_start, zero_scalar],
                    output=max_start_nneg,
                )
                _ensure_value_info(ctx, max_start_nneg)

                max_start_vec = _unsqueeze_scalar(
                    ctx, max_start_nneg, 0, f"scatter_max_start_vec_{axis}"
                )

                max_broadcast = ir.Value(
                    name=ctx.fresh_name(f"scatter_max_broadcast_{axis}"),
                    type=ir.TensorType(ir.DataType.INT64),
                    shape=ir.Shape((None, 1)),
                )
                _emit_into(
                    ctx,
                    "Expand",
                    [max_start_vec, column_shape],
                    output=max_broadcast,
                )
                _ensure_value_info(ctx, max_broadcast)

                col_nonneg = ir.Value(
                    name=ctx.fresh_name(f"scatter_base_ge0_{axis}"),
                    type=ir.TensorType(ir.DataType.INT64),
                    shape=ir.Shape((None, 1)),
                )
                _emit_into(
                    ctx,
                    "Max",
                    [col_val, zero_column],
                    output=col_nonneg,
                )
                _ensure_value_info(ctx, col_nonneg)

                col_clamped = ir.Value(
                    name=ctx.fresh_name(f"scatter_base_clamped_{axis}"),
                    type=ir.TensorType(ir.DataType.INT64),
                    shape=ir.Shape((None, 1)),
                )
                _emit_into(
                    ctx,
                    "Min",
                    [col_nonneg, max_broadcast],
                    output=col_clamped,
                )
                _ensure_value_info(ctx, col_clamped)
                col_val = col_clamped

            base_cols.append(col_val)
        else:
            base_cols.append(zero_column)

    base_matrix = ir.Value(
        name=ctx.fresh_name("scatter_base_matrix"),
        type=ir.TensorType(ir.DataType.INT64),
        shape=ir.Shape((None, operand_rank)),
    )
    _emit_into(
        ctx,
        "Concat",
        base_cols,
        output=base_matrix,
        attributes={"axis": 1},
    )
    _ensure_value_info(ctx, base_matrix)
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
    reshape_base_shape = ir.Value(
        name=ctx.fresh_name("scatter_base_reshape_shape"),
        type=ir.TensorType(ir.DataType.INT64),
        shape=ir.Shape((3,)),
    )
    one_vec = _const_i64(ctx, np.asarray([1], dtype=np.int64), "scatter_one_vec")
    rank_vec = _const_i64(
        ctx, np.asarray([operand_rank], dtype=np.int64), "scatter_rank_vec"
    )
    _emit_into(
        ctx,
        "Concat",
        [num_updates_vec, one_vec, rank_vec],
        output=reshape_base_shape,
        attributes={"axis": 0},
    )
    _ensure_value_info(ctx, reshape_base_shape)

    base_reshaped = ir.Value(
        name=ctx.fresh_name("scatter_base_reshaped"),
        type=ir.TensorType(ir.DataType.INT64),
        shape=ir.Shape((None, None, operand_rank)),
    )
    _emit_into(
        ctx,
        "Reshape",
        [base_matrix, reshape_base_shape],
        output=base_reshaped,
    )
    _ensure_value_info(ctx, base_reshaped)

    expand_shape = ir.Value(
        name=ctx.fresh_name("scatter_expand_shape"),
        type=ir.TensorType(ir.DataType.INT64),
        shape=ir.Shape((3,)),
    )
    rank_vec2 = _const_i64(
        ctx, np.asarray([operand_rank], dtype=np.int64), "scatter_rank_vec2"
    )
    _emit_into(
        ctx,
        "Concat",
        [num_updates_vec, window_total_vec, rank_vec2],
        output=expand_shape,
        attributes={"axis": 0},
    )
    _ensure_value_info(ctx, expand_shape)

    base_expanded = ir.Value(
        name=ctx.fresh_name("scatter_base_expanded"),
        type=ir.TensorType(ir.DataType.INT64),
        shape=ir.Shape((None, None, operand_rank)),
    )
    _emit_into(
        ctx,
        "Expand",
        [base_reshaped, expand_shape],
        output=base_expanded,
    )
    _ensure_value_info(ctx, base_expanded)

    reshape_offsets_shape = ir.Value(
        name=ctx.fresh_name("scatter_offsets_shape"),
        type=ir.TensorType(ir.DataType.INT64),
        shape=ir.Shape((3,)),
    )
    one_vec2 = _const_i64(ctx, np.asarray([1], dtype=np.int64), "scatter_one_vec2")
    rank_vec3 = _const_i64(
        ctx, np.asarray([operand_rank], dtype=np.int64), "scatter_rank_vec3"
    )
    _emit_into(
        ctx,
        "Concat",
        [one_vec2, window_total_vec, rank_vec3],
        output=reshape_offsets_shape,
        attributes={"axis": 0},
    )
    _ensure_value_info(ctx, reshape_offsets_shape)

    offsets_reshaped = ir.Value(
        name=ctx.fresh_name("scatter_offsets_reshaped"),
        type=ir.TensorType(ir.DataType.INT64),
        shape=ir.Shape((1, None, operand_rank)),
    )
    _emit_into(
        ctx,
        "Reshape",
        [window_offsets, reshape_offsets_shape],
        output=offsets_reshaped,
    )
    _ensure_value_info(ctx, offsets_reshaped)

    indices_expanded = ir.Value(
        name=ctx.fresh_name("scatter_indices_expanded"),
        type=ir.TensorType(ir.DataType.INT64),
        shape=ir.Shape((None, None, operand_rank)),
    )
    _emit_into(
        ctx,
        "Add",
        [base_expanded, offsets_reshaped],
        output=indices_expanded,
    )
    _ensure_value_info(ctx, indices_expanded)

    final_shape = ir.Value(
        name=ctx.fresh_name("scatter_indices_final_shape"),
        type=ir.TensorType(ir.DataType.INT64),
        shape=ir.Shape((2,)),
    )
    rank_vec4 = _const_i64(
        ctx, np.asarray([operand_rank], dtype=np.int64), "scatter_rank_vec4"
    )
    _emit_into(
        ctx,
        "Concat",
        [total_updates_vec, rank_vec4],
        output=final_shape,
        attributes={"axis": 0},
    )
    _ensure_value_info(ctx, final_shape)

    indices_flat = ir.Value(
        name=ctx.fresh_name("scatter_indices_flat"),
        type=ir.TensorType(ir.DataType.INT64),
        shape=ir.Shape((None, operand_rank)),
    )
    _emit_into(
        ctx,
        "Reshape",
        [indices_expanded, final_shape],
        output=indices_flat,
    )
    _ensure_value_info(ctx, indices_flat)
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
        updates_perm_val = ir.Value(
            name=ctx.fresh_name("scatter_updates_perm"),
            type=updates_val.type,
            shape=ir.Shape(tuple(perm_shape_dims)),
        )
        _emit_into(
            ctx,
            "Transpose",
            [updates_val],
            output=updates_perm_val,
            attributes={"perm": tuple(perm)},
        )
        _ensure_value_info(ctx, updates_perm_val)

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
    _ensure_value_info(ctx, out_val)
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
    _builder_op(
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
    _ensure_value_info(ctx, out_val)


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
