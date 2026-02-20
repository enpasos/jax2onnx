# jax2onnx/plugins/jax/lax/dynamic_update_slice.py

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins.jax.lax._index_utils import _const_i64, _scalar_i64, _cast_to_i64
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover - typing only
    from jax2onnx.converter.ir_context import IRContext


def _binary_scalar(
    ctx: "IRContext", op: str, lhs: ir.Value, rhs: ir.Value, name_hint: str
) -> ir.Value:
    result = getattr(ctx.builder, op)(
        lhs,
        rhs,
        _outputs=[ctx.fresh_name(name_hint)],
    )
    result.type = ir.TensorType(getattr(lhs.type, "dtype", ir.DataType.INT64))
    _stamp_type_and_shape(result, ())
    _ensure_value_metadata(ctx, result)
    return result


def _const_scalar_i64(value: ir.Value) -> int | None:
    const_val = getattr(value, "const_value", None)
    if const_val is None:
        return None
    try:
        arr = np.asarray(const_val.numpy())
    except Exception:
        try:
            arr = np.asarray(const_val)
        except Exception:
            return None
    if arr.size != 1:
        return None
    try:
        return int(arr.reshape(()))
    except (TypeError, ValueError):
        return None


@register_primitive(
    jaxpr_primitive=jax.lax.dynamic_update_slice_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.dynamic_update_slice.html",
    onnx=[
        {
            "component": "TensorScatter",
            "doc": "https://onnx.ai/onnx/operators/onnx__TensorScatter.html",
        },
        {
            "component": "ScatterND",
            "doc": "https://onnx.ai/onnx/operators/onnx__ScatterND.html",
        },
    ],
    since="0.8.1",
    context="primitives.lax",
    component="dynamic_update_slice",
    testcases=[
        {
            "testcase": "dus_1d_scalar_update",
            "callable": lambda ref, val, idx: jax.lax.dynamic_update_slice(
                ref, val, (idx,)
            ),
            "input_shapes": [(10,), (1,), ()],
            "input_dtypes": [np.float32, np.float32, np.int32],
            "expected_output_shapes": [(10,)],
            "post_check_onnx_graph": EG(
                ["ScatterND:10"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dus_1d_block_update",
            "callable": lambda ref, upd, idx: jax.lax.dynamic_update_slice(
                ref, upd, (idx,)
            ),
            "input_shapes": [(10,), (3,), ()],
            "input_dtypes": [np.float32, np.float32, np.int32],
            "expected_output_shapes": [(10,)],
            "post_check_onnx_graph": EG(
                ["ScatterND:10"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dus_2d_block_update",
            "callable": lambda ref, upd, i, j: jax.lax.dynamic_update_slice(
                ref, upd, (i, j)
            ),
            "input_shapes": [(4, 4), (2, 2), (), ()],
            "input_dtypes": [np.float32, np.float32, np.int32, np.int32],
            "expected_output_shapes": [(4, 4)],
            "post_check_onnx_graph": EG(
                ["ScatterND:4x4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dus_3d_block_update",
            "callable": lambda ref, upd, i, j, k: jax.lax.dynamic_update_slice(
                ref, upd, (i, j, k)
            ),
            "input_shapes": [(3, 4, 4), (1, 2, 2), (), (), ()],
            "input_dtypes": [np.float32, np.float32, np.int32, np.int32, np.int32],
            "expected_output_shapes": [(3, 4, 4)],
            "post_check_onnx_graph": EG(
                ["ScatterND:3x4x4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dus_4d_block_update",
            "callable": lambda ref, upd, a, b, c, d: jax.lax.dynamic_update_slice(
                ref, upd, (a, b, c, d)
            ),
            "input_shapes": [(5, 10, 10, 1), (1, 5, 5, 1), (), (), (), ()],
            "input_dtypes": [
                np.float32,
                np.float32,
                np.int32,
                np.int32,
                np.int32,
                np.int32,
            ],
            "expected_output_shapes": [(5, 10, 10, 1)],
            "post_check_onnx_graph": EG(
                ["ScatterND:5x10x10x1"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dus_tensorscatter_axis1_opset24",
            "callable": lambda ref, upd, idx: jax.lax.dynamic_update_slice(
                ref, upd, (0, idx, 0)
            ),
            "input_shapes": [(2, 5, 3), (2, 2, 3), ()],
            "input_dtypes": [np.float32, np.float32, np.int32],
            "expected_output_shapes": [(2, 5, 3)],
            "opset_version": 24,
            "skip_numeric_validation": True,
            "post_check_onnx_graph": EG(
                ["TensorScatter:2x5x3"],
                no_unused_inputs=True,
            ),
            "run_only_f32_variant": True,
        },
    ],
)
class DynamicUpdateSlicePlugin(PrimitiveLeafPlugin):
    """IR-only lowering of ``lax.dynamic_update_slice`` via ScatterND."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        ref_var = eqn.invars[0]
        upd_var = eqn.invars[1]
        start_vars = list(eqn.invars[2:])
        out_var = eqn.outvars[0]

        ref_val = ctx.get_value_for_var(ref_var, name_hint=ctx.fresh_name("dus_ref"))
        upd_val = ctx.get_value_for_var(upd_var, name_hint=ctx.fresh_name("dus_update"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("dus_out"))

        rank = len(getattr(ref_var.aval, "shape", ()))
        if rank != len(start_vars):
            raise ValueError(
                f"dynamic_update_slice expects {rank} start indices but got {len(start_vars)}"
            )
        opset = int(getattr(ctx.builder, "opset", 21))

        ref_shape_static = tuple(getattr(getattr(ref_var, "aval", None), "shape", ()))
        upd_shape_static = tuple(getattr(getattr(upd_var, "aval", None), "shape", ()))

        # Fast path: map a strict cache-like case to TensorScatter (opset >= 24).
        # Conditions:
        # - rank >= 2 with dim 0 treated as batch axis
        # - exactly one non-batch axis has a different update extent
        # - all non-axis start indices are compile-time zero
        # - non-axis dimensions match between ref and update
        if (
            opset >= 24
            and rank >= 2
            and len(ref_shape_static) == rank
            and len(upd_shape_static) == rank
            and all(isinstance(d, (int, np.integer)) for d in ref_shape_static)
            and all(isinstance(d, (int, np.integer)) for d in upd_shape_static)
            and isinstance(ref_shape_static[0], (int, np.integer))
            and isinstance(upd_shape_static[0], (int, np.integer))
            and int(ref_shape_static[0]) == int(upd_shape_static[0])
        ):
            candidate_axes = [
                ax
                for ax in range(1, rank)
                if int(ref_shape_static[ax]) != int(upd_shape_static[ax])
            ]
            if len(candidate_axes) == 1:
                seq_axis = int(candidate_axes[0])
                starts_i64: list[ir.Value] = []
                non_axis_starts_are_zero = True
                for axis, start_var in enumerate(start_vars):
                    start_val = ctx.get_value_for_var(
                        start_var, name_hint=ctx.fresh_name("dus_start")
                    )
                    start_const = _const_scalar_i64(start_val)
                    start_i64 = _cast_to_i64(ctx, start_val, f"dus_start_{axis}_i64")
                    starts_i64.append(start_i64)
                    if axis != seq_axis and start_const != 0:
                        non_axis_starts_are_zero = False
                        break
                if non_axis_starts_are_zero and all(
                    int(ref_shape_static[ax]) == int(upd_shape_static[ax])
                    for ax in range(rank)
                    if ax != seq_axis
                ):
                    seq_start = starts_i64[seq_axis]
                    zero_i64 = _scalar_i64(ctx, 0, "dus_tensorscatter_zero")
                    one_i64 = _scalar_i64(ctx, 1, "dus_tensorscatter_one")

                    ref_shape = ctx.builder.Shape(
                        ref_val, _outputs=[ctx.fresh_name("dus_ref_shape")]
                    )
                    ref_shape.type = ir.TensorType(ir.DataType.INT64)
                    _stamp_type_and_shape(ref_shape, (rank,))
                    _ensure_value_metadata(ctx, ref_shape)

                    upd_shape = ctx.builder.Shape(
                        upd_val, _outputs=[ctx.fresh_name("dus_upd_shape")]
                    )
                    upd_shape.type = ir.TensorType(ir.DataType.INT64)
                    _stamp_type_and_shape(upd_shape, (rank,))
                    _ensure_value_metadata(ctx, upd_shape)

                    axis_idx = _const_i64(
                        ctx,
                        np.asarray(seq_axis, dtype=np.int64),
                        f"dus_tensorscatter_axis_{seq_axis}",
                    )
                    ref_dim = ctx.builder.Gather(
                        ref_shape,
                        axis_idx,
                        axis=0,
                        _outputs=[ctx.fresh_name("dus_tensorscatter_ref_dim")],
                    )
                    ref_dim.type = ir.TensorType(ir.DataType.INT64)
                    _stamp_type_and_shape(ref_dim, ())
                    _ensure_value_metadata(ctx, ref_dim)

                    upd_dim = ctx.builder.Gather(
                        upd_shape,
                        axis_idx,
                        axis=0,
                        _outputs=[ctx.fresh_name("dus_tensorscatter_upd_dim")],
                    )
                    upd_dim.type = ir.TensorType(ir.DataType.INT64)
                    _stamp_type_and_shape(upd_dim, ())
                    _ensure_value_metadata(ctx, upd_dim)

                    cond_neg = ctx.builder.Less(
                        seq_start,
                        zero_i64,
                        _outputs=[ctx.fresh_name("dus_tensorscatter_start_neg")],
                    )
                    cond_neg.type = ir.TensorType(ir.DataType.BOOL)
                    _stamp_type_and_shape(cond_neg, ())
                    _ensure_value_metadata(ctx, cond_neg)

                    start_plus_ref = _binary_scalar(
                        ctx,
                        "Add",
                        seq_start,
                        ref_dim,
                        "dus_tensorscatter_start_plus_ref",
                    )
                    start_norm = ctx.builder.Where(
                        cond_neg,
                        start_plus_ref,
                        seq_start,
                        _outputs=[ctx.fresh_name("dus_tensorscatter_start_norm")],
                    )
                    start_norm.type = ir.TensorType(ir.DataType.INT64)
                    _stamp_type_and_shape(start_norm, ())
                    _ensure_value_metadata(ctx, start_norm)

                    max_start = _binary_scalar(
                        ctx, "Sub", ref_dim, upd_dim, "dus_tensorscatter_max_start"
                    )
                    start_ge0 = _binary_scalar(
                        ctx, "Max", start_norm, zero_i64, "dus_tensorscatter_start_ge0"
                    )
                    start_clamped = _binary_scalar(
                        ctx,
                        "Min",
                        start_ge0,
                        max_start,
                        "dus_tensorscatter_start_clamped",
                    )

                    batch_axis_idx = _const_i64(
                        ctx,
                        np.asarray(0, dtype=np.int64),
                        "dus_tensorscatter_batch_axis",
                    )
                    batch_dim = ctx.builder.Gather(
                        ref_shape,
                        batch_axis_idx,
                        axis=0,
                        _outputs=[ctx.fresh_name("dus_tensorscatter_batch_dim")],
                    )
                    batch_dim.type = ir.TensorType(ir.DataType.INT64)
                    _stamp_type_and_shape(batch_dim, ())
                    _ensure_value_metadata(ctx, batch_dim)

                    unsq_axes = _const_i64(
                        ctx,
                        np.asarray([0], dtype=np.int64),
                        "dus_tensorscatter_unsq_axes",
                    )
                    start_vec = ctx.builder.Unsqueeze(
                        start_clamped,
                        unsq_axes,
                        _outputs=[ctx.fresh_name("dus_tensorscatter_start_vec")],
                    )
                    start_vec.type = ir.TensorType(ir.DataType.INT64)
                    _stamp_type_and_shape(start_vec, (1,))
                    _ensure_value_metadata(ctx, start_vec)

                    batch_vec = ctx.builder.Unsqueeze(
                        batch_dim,
                        unsq_axes,
                        _outputs=[ctx.fresh_name("dus_tensorscatter_batch_vec")],
                    )
                    batch_vec.type = ir.TensorType(ir.DataType.INT64)
                    _stamp_type_and_shape(batch_vec, (1,))
                    _ensure_value_metadata(ctx, batch_vec)

                    write_indices = ctx.builder.Expand(
                        start_vec,
                        batch_vec,
                        _outputs=[ctx.fresh_name("dus_tensorscatter_write_idx")],
                    )
                    write_indices.type = ir.TensorType(ir.DataType.INT64)
                    _stamp_type_and_shape(write_indices, (int(ref_shape_static[0]),))
                    _ensure_value_metadata(ctx, write_indices)

                    out_name = getattr(out_spec, "name", None) or ctx.fresh_name(
                        "TensorScatter"
                    )
                    result = ctx.builder.TensorScatter(
                        ref_val,
                        upd_val,
                        write_indices,
                        axis=seq_axis,
                        mode="none",
                        _outputs=[out_name],
                    )
                    if getattr(out_spec, "type", None) is not None:
                        result.type = out_spec.type
                    else:
                        result.type = ref_val.type
                    if getattr(out_spec, "shape", None) is not None:
                        result.shape = out_spec.shape
                    else:
                        _stamp_type_and_shape(result, ref_shape_static)
                    _ensure_value_metadata(ctx, result)
                    ctx.bind_value_for_var(out_var, result)
                    return

        zero_i64 = _scalar_i64(ctx, 0, "dus_zero")
        one_i64 = _scalar_i64(ctx, 1, "dus_one")

        ref_shape = ctx.builder.Shape(
            ref_val, _outputs=[ctx.fresh_name("dus_ref_shape")]
        )
        ref_shape.type = ir.TensorType(ir.DataType.INT64)
        _stamp_type_and_shape(ref_shape, (rank,))
        _ensure_value_metadata(ctx, ref_shape)

        upd_shape = ctx.builder.Shape(
            upd_val, _outputs=[ctx.fresh_name("dus_upd_shape")]
        )
        upd_shape.type = ir.TensorType(ir.DataType.INT64)
        _stamp_type_and_shape(upd_shape, (rank,))
        _ensure_value_metadata(ctx, upd_shape)

        # Prepare per-axis metadata
        upd_dims: list[ir.Value] = []
        start_clamped: list[ir.Value] = []

        for axis, start_var in enumerate(start_vars):
            start_val = ctx.get_value_for_var(
                start_var, name_hint=ctx.fresh_name("dus_start")
            )
            start_i64 = _cast_to_i64(ctx, start_val, f"dus_start_{axis}_i64")

            axis_idx = _const_i64(
                ctx, np.asarray(axis, dtype=np.int64), f"dus_axis_{axis}"
            )

            upd_dim = ctx.builder.Gather(
                upd_shape,
                axis_idx,
                axis=0,
                _outputs=[ctx.fresh_name("dus_upd_dim")],
            )
            upd_dim.type = ir.TensorType(ir.DataType.INT64)
            _stamp_type_and_shape(upd_dim, ())
            _ensure_value_metadata(ctx, upd_dim)
            upd_dims.append(upd_dim)

            ref_dim = ctx.builder.Gather(
                ref_shape,
                axis_idx,
                axis=0,
                _outputs=[ctx.fresh_name("dus_ref_dim")],
            )
            ref_dim.type = ir.TensorType(ir.DataType.INT64)
            _stamp_type_and_shape(ref_dim, ())
            _ensure_value_metadata(ctx, ref_dim)

            cond_neg = ctx.builder.Less(
                start_i64,
                zero_i64,
                _outputs=[ctx.fresh_name("dus_start_neg")],
            )
            cond_neg.type = ir.TensorType(ir.DataType.BOOL)
            _stamp_type_and_shape(cond_neg, ())
            _ensure_value_metadata(ctx, cond_neg)

            start_plus_ref = _binary_scalar(
                ctx, "Add", start_i64, ref_dim, "dus_start_plus_ref"
            )
            start_norm = ctx.builder.Where(
                cond_neg,
                start_plus_ref,
                start_i64,
                _outputs=[ctx.fresh_name("dus_start_norm")],
            )
            start_norm.type = ir.TensorType(ir.DataType.INT64)
            _stamp_type_and_shape(start_norm, ())
            _ensure_value_metadata(ctx, start_norm)

            max_start = _binary_scalar(ctx, "Sub", ref_dim, upd_dim, "dus_max_start")
            start_ge0 = _binary_scalar(
                ctx, "Max", start_norm, zero_i64, "dus_start_ge0"
            )
            clamped = _binary_scalar(
                ctx, "Min", start_ge0, max_start, "dus_start_clamped"
            )
            start_clamped.append(clamped)

        # Total number of update elements: ReduceProd(upd_shape)
        numel_upd = ctx.builder.ReduceProd(
            upd_shape,
            keepdims=0,
            _outputs=[ctx.fresh_name("dus_numel")],
        )
        numel_upd.type = ir.TensorType(ir.DataType.INT64)
        _stamp_type_and_shape(numel_upd, ())
        _ensure_value_metadata(ctx, numel_upd)

        # Build expanded coordinate grids and stack as indices
        idx_components: list[ir.Value] = []
        rank_shape = tuple([None] * rank)
        for axis in range(rank):
            dim = upd_dims[axis]

            range_out = ctx.builder.Range(
                zero_i64,
                dim,
                one_i64,
                _outputs=[ctx.fresh_name("dus_range")],
            )
            range_out.type = ir.TensorType(ir.DataType.INT64)
            _stamp_type_and_shape(range_out, (None,))
            _ensure_value_metadata(ctx, range_out)

            axes_unsq = [ax for ax in range(rank) if ax != axis]
            if axes_unsq:
                axes_tensor = _const_i64(
                    ctx, np.asarray(axes_unsq, dtype=np.int64), f"dus_unsq_axes_{axis}"
                )
                range_unsq = ctx.builder.Unsqueeze(
                    range_out,
                    axes_tensor,
                    _outputs=[ctx.fresh_name("dus_range_unsq")],
                )
                range_unsq.type = ir.TensorType(ir.DataType.INT64)
                unsq_shape = tuple([1] * axis + [None] + [1] * (rank - axis - 1))
                _stamp_type_and_shape(range_unsq, unsq_shape)
                _ensure_value_metadata(ctx, range_unsq)
            else:
                range_unsq = range_out

            range_exp = ctx.builder.Expand(
                range_unsq,
                upd_shape,
                _outputs=[ctx.fresh_name("dus_range_exp")],
            )
            range_exp.type = ir.TensorType(ir.DataType.INT64)
            _stamp_type_and_shape(range_exp, rank_shape)
            _ensure_value_metadata(ctx, range_exp)

            start_b = ctx.builder.Expand(
                start_clamped[axis],
                upd_shape,
                _outputs=[ctx.fresh_name("dus_start_b")],
            )
            start_b.type = ir.TensorType(ir.DataType.INT64)
            _stamp_type_and_shape(start_b, rank_shape)
            _ensure_value_metadata(ctx, start_b)

            idx_axis = ctx.builder.Add(
                start_b,
                range_exp,
                _outputs=[ctx.fresh_name("dus_idx_axis")],
            )
            idx_axis.type = ir.TensorType(ir.DataType.INT64)
            _stamp_type_and_shape(idx_axis, rank_shape)
            _ensure_value_metadata(ctx, idx_axis)

            axes_last = _const_i64(
                ctx, np.asarray(rank, dtype=np.int64), f"dus_axes_last_{axis}"
            )
            idx_unsq = ctx.builder.Unsqueeze(
                idx_axis,
                axes_last,
                _outputs=[ctx.fresh_name("dus_idx_unsq")],
            )
            idx_unsq.type = ir.TensorType(ir.DataType.INT64)
            _stamp_type_and_shape(idx_unsq, tuple([None] * rank + [1]))
            _ensure_value_metadata(ctx, idx_unsq)
            idx_components.append(idx_unsq)

        indices_nd = ctx.builder.Concat(
            *idx_components,
            axis=rank,
            _outputs=[ctx.fresh_name("dus_indices_nd")],
        )
        indices_nd.type = ir.TensorType(ir.DataType.INT64)
        _stamp_type_and_shape(indices_nd, tuple([None] * rank + [rank]))
        _ensure_value_metadata(ctx, indices_nd)

        axes0 = _const_i64(ctx, np.asarray(0, dtype=np.int64), "dus_axes0")
        numel_vec = ctx.builder.Unsqueeze(
            numel_upd,
            axes0,
            _outputs=[ctx.fresh_name("dus_numel_vec")],
        )
        numel_vec.type = ir.TensorType(ir.DataType.INT64)
        _stamp_type_and_shape(numel_vec, (1,))
        _ensure_value_metadata(ctx, numel_vec)

        rank_const = _scalar_i64(ctx, rank, "dus_rank_scalar")
        rank_vec = ctx.builder.Unsqueeze(
            rank_const,
            axes0,
            _outputs=[ctx.fresh_name("dus_rank_vec")],
        )
        rank_vec.type = ir.TensorType(ir.DataType.INT64)
        _stamp_type_and_shape(rank_vec, (1,))
        _ensure_value_metadata(ctx, rank_vec)

        shape_2d = ctx.builder.Concat(
            numel_vec,
            rank_vec,
            axis=0,
            _outputs=[ctx.fresh_name("dus_shape2d")],
        )
        shape_2d.type = ir.TensorType(ir.DataType.INT64)
        _stamp_type_and_shape(shape_2d, (2,))
        _ensure_value_metadata(ctx, shape_2d)

        indices_2d = ctx.builder.Reshape(
            indices_nd,
            shape_2d,
            _outputs=[ctx.fresh_name("dus_indices_2d")],
        )
        indices_2d.type = ir.TensorType(ir.DataType.INT64)
        _stamp_type_and_shape(indices_2d, (None, rank))
        _ensure_value_metadata(ctx, indices_2d)

        updates_shape_1d = ctx.builder.Unsqueeze(
            numel_upd,
            axes0,
            _outputs=[ctx.fresh_name("dus_updates_shape1d")],
        )
        updates_shape_1d.type = ir.TensorType(ir.DataType.INT64)
        _stamp_type_and_shape(updates_shape_1d, (1,))
        _ensure_value_metadata(ctx, updates_shape_1d)

        upd_flat = ctx.builder.Reshape(
            upd_val,
            updates_shape_1d,
            _outputs=[ctx.fresh_name("dus_upd_flat")],
        )
        upd_flat.type = ir.TensorType(getattr(ref_val.type, "dtype", ir.DataType.FLOAT))
        _stamp_type_and_shape(upd_flat, (None,))
        _ensure_value_metadata(ctx, upd_flat)

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("ScatterND")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("ScatterND")

        result = ctx.builder.ScatterND(
            ref_val,
            indices_2d,
            upd_flat,
            _outputs=[desired_name],
        )

        target_shape = tuple(getattr(ref_var.aval, "shape", ()))
        _stamp_type_and_shape(result, target_shape)
        result_dtype = getattr(getattr(ref_val, "type", None), "dtype", None)
        if result_dtype is not None:
            result.type = ir.TensorType(result_dtype)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)
