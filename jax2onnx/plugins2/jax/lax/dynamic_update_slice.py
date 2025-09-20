from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np
import onnx_ir as ir
from onnx_ir import Attr as IRAttr, AttributeType as IRAttrType

from jax2onnx.plugins2._ir_shapes import _ensure_value_info, _stamp_type_and_shape
from jax2onnx.plugins2.jax.lax._index_utils import _const_i64, _scalar_i64, _cast_to_i64
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover - typing only
    from jax2onnx.converter2.ir_context import IRContext


def _add_binary_scalar(
    ctx: "IRContext", op_type: str, lhs: ir.Value, rhs: ir.Value, name_hint: str
) -> ir.Value:
    out = ir.Value(
        name=ctx.fresh_name(name_hint),
        type=ir.TensorType(lhs.type.dtype),
        shape=ir.Shape(()),
    )
    ctx.add_node(
        ir.Node(
            op_type=op_type,
            domain="",
            inputs=[lhs, rhs],
            outputs=[out],
            name=ctx.fresh_name(op_type),
        )
    )
    _stamp_type_and_shape(out, ())
    _ensure_value_info(ctx, out)
    return out


def _add_elementwise(
    ctx: "IRContext",
    op_type: str,
    inputs: list[ir.Value],
    shape_dims: tuple[int | None, ...],
    dtype: ir.DataType,
    name_hint: str,
) -> ir.Value:
    out = ir.Value(
        name=ctx.fresh_name(name_hint),
        type=ir.TensorType(dtype),
        shape=ir.Shape(shape_dims),
    )
    ctx.add_node(
        ir.Node(
            op_type=op_type,
            domain="",
            inputs=inputs,
            outputs=[out],
            name=ctx.fresh_name(op_type),
        )
    )
    _stamp_type_and_shape(out, shape_dims)
    _ensure_value_info(ctx, out)
    return out


@register_primitive(
    jaxpr_primitive=jax.lax.dynamic_update_slice_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.dynamic_update_slice.html",
    onnx=[
        {
            "component": "ScatterND",
            "doc": "https://onnx.ai/onnx/operators/onnx__ScatterND.html",
        }
    ],
    since="v0.1.0",
    context="primitives2.lax",
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
            "use_onnx_ir": True,
        },
        {
            "testcase": "dus_1d_block_update",
            "callable": lambda ref, upd, idx: jax.lax.dynamic_update_slice(
                ref, upd, (idx,)
            ),
            "input_shapes": [(10,), (3,), ()],
            "input_dtypes": [np.float32, np.float32, np.int32],
            "expected_output_shapes": [(10,)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "dus_2d_block_update",
            "callable": lambda ref, upd, i, j: jax.lax.dynamic_update_slice(
                ref, upd, (i, j)
            ),
            "input_shapes": [(4, 4), (2, 2), (), ()],
            "input_dtypes": [np.float32, np.float32, np.int32, np.int32],
            "expected_output_shapes": [(4, 4)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "dus_3d_block_update",
            "callable": lambda ref, upd, i, j, k: jax.lax.dynamic_update_slice(
                ref, upd, (i, j, k)
            ),
            "input_shapes": [(3, 4, 4), (1, 2, 2), (), (), ()],
            "input_dtypes": [np.float32, np.float32, np.int32, np.int32, np.int32],
            "expected_output_shapes": [(3, 4, 4)],
            "use_onnx_ir": True,
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
            "use_onnx_ir": True,
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
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("dus_out"))

        rank = len(getattr(ref_var.aval, "shape", ()))
        if rank != len(start_vars):
            raise ValueError(
                f"dynamic_update_slice expects {rank} start indices but got {len(start_vars)}"
            )

        zero_i64 = _scalar_i64(ctx, 0, "dus_zero")
        one_i64 = _scalar_i64(ctx, 1, "dus_one")

        ref_shape = ir.Value(
            name=ctx.fresh_name("dus_ref_shape"),
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape((rank,)),
        )
        ctx.add_node(
            ir.Node(
                op_type="Shape",
                domain="",
                inputs=[ref_val],
                outputs=[ref_shape],
                name=ctx.fresh_name("Shape"),
            )
        )
        _stamp_type_and_shape(ref_shape, tuple([rank]))
        _ensure_value_info(ctx, ref_shape)

        upd_shape = ir.Value(
            name=ctx.fresh_name("dus_upd_shape"),
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape((rank,)),
        )
        ctx.add_node(
            ir.Node(
                op_type="Shape",
                domain="",
                inputs=[upd_val],
                outputs=[upd_shape],
                name=ctx.fresh_name("Shape"),
            )
        )
        _stamp_type_and_shape(upd_shape, tuple([rank]))
        _ensure_value_info(ctx, upd_shape)

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

            upd_dim = ir.Value(
                name=ctx.fresh_name("dus_upd_dim"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape(()),
            )
            ctx.add_node(
                ir.Node(
                    op_type="Gather",
                    domain="",
                    inputs=[upd_shape, axis_idx],
                    outputs=[upd_dim],
                    name=ctx.fresh_name("Gather"),
                    attributes=[IRAttr("axis", IRAttrType.INT, 0)],
                )
            )
            _stamp_type_and_shape(upd_dim, ())
            _ensure_value_info(ctx, upd_dim)
            upd_dims.append(upd_dim)

            ref_dim = ir.Value(
                name=ctx.fresh_name("dus_ref_dim"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape(()),
            )
            ctx.add_node(
                ir.Node(
                    op_type="Gather",
                    domain="",
                    inputs=[ref_shape, axis_idx],
                    outputs=[ref_dim],
                    name=ctx.fresh_name("Gather"),
                    attributes=[IRAttr("axis", IRAttrType.INT, 0)],
                )
            )
            _stamp_type_and_shape(ref_dim, ())
            _ensure_value_info(ctx, ref_dim)

            cond_neg = ir.Value(
                name=ctx.fresh_name("dus_start_neg"),
                type=ir.TensorType(ir.DataType.BOOL),
                shape=ir.Shape(()),
            )
            ctx.add_node(
                ir.Node(
                    op_type="Less",
                    domain="",
                    inputs=[start_i64, zero_i64],
                    outputs=[cond_neg],
                    name=ctx.fresh_name("Less"),
                )
            )
            _stamp_type_and_shape(cond_neg, ())
            _ensure_value_info(ctx, cond_neg)

            start_plus_ref = _add_binary_scalar(
                ctx, "Add", start_i64, ref_dim, "dus_start_plus_ref"
            )
            start_norm = ir.Value(
                name=ctx.fresh_name("dus_start_norm"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape(()),
            )
            ctx.add_node(
                ir.Node(
                    op_type="Where",
                    domain="",
                    inputs=[cond_neg, start_plus_ref, start_i64],
                    outputs=[start_norm],
                    name=ctx.fresh_name("Where"),
                )
            )
            _stamp_type_and_shape(start_norm, ())
            _ensure_value_info(ctx, start_norm)

            max_start = _add_binary_scalar(
                ctx, "Sub", ref_dim, upd_dim, "dus_max_start"
            )
            start_ge0 = _add_binary_scalar(
                ctx, "Max", start_norm, zero_i64, "dus_start_ge0"
            )
            clamped = _add_binary_scalar(
                ctx, "Min", start_ge0, max_start, "dus_start_clamped"
            )
            start_clamped.append(clamped)

        # Total number of update elements: ReduceProd(upd_shape)
        numel_upd = ir.Value(
            name=ctx.fresh_name("dus_numel"),
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape(()),
        )
        ctx.add_node(
            ir.Node(
                op_type="ReduceProd",
                domain="",
                inputs=[upd_shape],
                outputs=[numel_upd],
                name=ctx.fresh_name("ReduceProd"),
                attributes=[IRAttr("keepdims", IRAttrType.INT, 0)],
            )
        )
        _stamp_type_and_shape(numel_upd, ())
        _ensure_value_info(ctx, numel_upd)

        # Build expanded coordinate grids and stack as indices
        idx_components: list[ir.Value] = []
        rank_shape = tuple([None] * rank)
        for axis in range(rank):
            dim = upd_dims[axis]

            range_out = ir.Value(
                name=ctx.fresh_name("dus_range"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape((None,)),
            )
            ctx.add_node(
                ir.Node(
                    op_type="Range",
                    domain="",
                    inputs=[zero_i64, dim, one_i64],
                    outputs=[range_out],
                    name=ctx.fresh_name("Range"),
                )
            )
            _ensure_value_info(ctx, range_out)

            axes_unsq = [ax for ax in range(rank) if ax != axis]
            if axes_unsq:
                axes_tensor = _const_i64(
                    ctx, np.asarray(axes_unsq, dtype=np.int64), f"dus_unsq_axes_{axis}"
                )
                range_unsq = ir.Value(
                    name=ctx.fresh_name("dus_range_unsq"),
                    type=ir.TensorType(ir.DataType.INT64),
                    shape=ir.Shape(
                        tuple([1] * axis + [None] + [1] * (rank - axis - 1))
                    ),
                )
                ctx.add_node(
                    ir.Node(
                        op_type="Unsqueeze",
                        domain="",
                        inputs=[range_out, axes_tensor],
                        outputs=[range_unsq],
                        name=ctx.fresh_name("Unsqueeze"),
                    )
                )
                _ensure_value_info(ctx, range_unsq)
            else:
                range_unsq = range_out

            range_exp = ir.Value(
                name=ctx.fresh_name("dus_range_exp"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape(rank_shape),
            )
            ctx.add_node(
                ir.Node(
                    op_type="Expand",
                    domain="",
                    inputs=[range_unsq, upd_shape],
                    outputs=[range_exp],
                    name=ctx.fresh_name("Expand"),
                )
            )
            _ensure_value_info(ctx, range_exp)

            start_b = ir.Value(
                name=ctx.fresh_name("dus_start_b"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape(rank_shape),
            )
            ctx.add_node(
                ir.Node(
                    op_type="Expand",
                    domain="",
                    inputs=[start_clamped[axis], upd_shape],
                    outputs=[start_b],
                    name=ctx.fresh_name("Expand"),
                )
            )
            _ensure_value_info(ctx, start_b)

            idx_axis = ir.Value(
                name=ctx.fresh_name("dus_idx_axis"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape(rank_shape),
            )
            ctx.add_node(
                ir.Node(
                    op_type="Add",
                    domain="",
                    inputs=[start_b, range_exp],
                    outputs=[idx_axis],
                    name=ctx.fresh_name("Add"),
                )
            )
            _ensure_value_info(ctx, idx_axis)

            axes_last = _const_i64(
                ctx, np.asarray(rank, dtype=np.int64), f"dus_axes_last_{axis}"
            )
            idx_unsq = ir.Value(
                name=ctx.fresh_name("dus_idx_unsq"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape(tuple([None] * rank + [1])),
            )
            ctx.add_node(
                ir.Node(
                    op_type="Unsqueeze",
                    domain="",
                    inputs=[idx_axis, axes_last],
                    outputs=[idx_unsq],
                    name=ctx.fresh_name("Unsqueeze"),
                )
            )
            _ensure_value_info(ctx, idx_unsq)
            idx_components.append(idx_unsq)

        indices_nd = ir.Value(
            name=ctx.fresh_name("dus_indices_nd"),
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape(tuple([None] * rank + [rank])),
        )
        ctx.add_node(
            ir.Node(
                op_type="Concat",
                domain="",
                inputs=idx_components,
                outputs=[indices_nd],
                name=ctx.fresh_name("Concat"),
                attributes=[IRAttr("axis", IRAttrType.INT, rank)],
            )
        )
        _ensure_value_info(ctx, indices_nd)

        axes0 = _const_i64(ctx, np.asarray(0, dtype=np.int64), "dus_axes0")
        numel_vec = ir.Value(
            name=ctx.fresh_name("dus_numel_vec"),
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape((1,)),
        )
        ctx.add_node(
            ir.Node(
                op_type="Unsqueeze",
                domain="",
                inputs=[numel_upd, axes0],
                outputs=[numel_vec],
                name=ctx.fresh_name("Unsqueeze"),
            )
        )
        _ensure_value_info(ctx, numel_vec)

        rank_const = _scalar_i64(ctx, rank, "dus_rank_scalar")
        rank_vec = ir.Value(
            name=ctx.fresh_name("dus_rank_vec"),
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape((1,)),
        )
        ctx.add_node(
            ir.Node(
                op_type="Unsqueeze",
                domain="",
                inputs=[rank_const, axes0],
                outputs=[rank_vec],
                name=ctx.fresh_name("Unsqueeze"),
            )
        )
        _ensure_value_info(ctx, rank_vec)

        shape_2d = ir.Value(
            name=ctx.fresh_name("dus_shape2d"),
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape((2,)),
        )
        ctx.add_node(
            ir.Node(
                op_type="Concat",
                domain="",
                inputs=[numel_vec, rank_vec],
                outputs=[shape_2d],
                name=ctx.fresh_name("Concat"),
                attributes=[IRAttr("axis", IRAttrType.INT, 0)],
            )
        )
        _ensure_value_info(ctx, shape_2d)

        indices_2d = ir.Value(
            name=ctx.fresh_name("dus_indices_2d"),
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape((None, rank)),
        )
        ctx.add_node(
            ir.Node(
                op_type="Reshape",
                domain="",
                inputs=[indices_nd, shape_2d],
                outputs=[indices_2d],
                name=ctx.fresh_name("Reshape"),
            )
        )
        _ensure_value_info(ctx, indices_2d)

        updates_shape_1d = ir.Value(
            name=ctx.fresh_name("dus_updates_shape1d"),
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape((1,)),
        )
        ctx.add_node(
            ir.Node(
                op_type="Unsqueeze",
                domain="",
                inputs=[numel_upd, axes0],
                outputs=[updates_shape_1d],
                name=ctx.fresh_name("Unsqueeze"),
            )
        )
        _ensure_value_info(ctx, updates_shape_1d)

        upd_flat = ir.Value(
            name=ctx.fresh_name("dus_upd_flat"),
            type=ir.TensorType(ref_val.type.dtype),
            shape=ir.Shape((None,)),
        )
        ctx.add_node(
            ir.Node(
                op_type="Reshape",
                domain="",
                inputs=[upd_val, updates_shape_1d],
                outputs=[upd_flat],
                name=ctx.fresh_name("Reshape"),
            )
        )
        _ensure_value_info(ctx, upd_flat)

        ctx.add_node(
            ir.Node(
                op_type="ScatterND",
                domain="",
                inputs=[ref_val, indices_2d, upd_flat],
                outputs=[out_val],
                name=ctx.fresh_name("ScatterND"),
            )
        )

        target_shape = tuple(getattr(ref_var.aval, "shape", ()))
        _stamp_type_and_shape(out_val, target_shape)
        out_val.type = ir.TensorType(ref_val.type.dtype)
        out_val.dtype = ref_val.type.dtype
        _ensure_value_info(ctx, out_val)
