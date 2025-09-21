from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.plugins2._ir_shapes import _ensure_value_info, _stamp_type_and_shape
from jax2onnx.plugins2.jax.lax._index_utils import _const_i64, _scalar_i64
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover - typing only
    from jax2onnx.converter2.ir_context import IRContext


@register_primitive(
    jaxpr_primitive=jax.lax.rev_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.rev.html",
    onnx=[
        {"component": "Flip", "doc": "https://onnx.ai/onnx/operators/onnx__Flip.html"}
    ],
    since="v0.7.5",
    context="primitives2.lax",
    component="rev",
    testcases=[
        {
            "testcase": "rev_vector",
            "callable": lambda x: jax.lax.rev(x, (0,)),
            "input_shapes": [(5,)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "rev_matrix_axes01",
            "callable": lambda x: jax.lax.rev(x, (0, 1)),
            "input_shapes": [(3, 4)],
            "use_onnx_ir": True,
        },
    ],
)
class RevPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.rev`` to a sequence of Gather ops (no Flip dependency)."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        axes_param = tuple(int(a) for a in eqn.params.get("dimensions", ()))
        input_shape = tuple(getattr(x_var.aval, "shape", ()))
        rank = len(input_shape)

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("rev_in"))
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("rev_out"))

        if rank == 0 or not axes_param:
            ctx.add_node(
                ir.Node(
                    op_type="Identity",
                    domain="",
                    inputs=[x_val],
                    outputs=[out_val],
                    name=ctx.fresh_name("Identity"),
                )
            )
            _stamp_type_and_shape(out_val, input_shape)
            _ensure_value_info(ctx, out_val)
            return

        canonical_axes: list[int] = []
        for axis in axes_param:
            canonical = axis % rank if axis < 0 else axis
            if canonical < 0 or canonical >= rank:
                raise ValueError(
                    f"Axis {axis} out of range for lax.rev with rank {rank}"
                )
            if canonical not in canonical_axes:
                canonical_axes.append(canonical)

        one = _scalar_i64(ctx, 1, "rev_one")
        neg_one = _scalar_i64(ctx, -1, "rev_neg_one")

        current_val = x_val
        for idx, axis in enumerate(canonical_axes):
            shape_val = ir.Value(
                name=ctx.fresh_name("rev_shape"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape((rank,)),
            )
            ctx.add_node(
                ir.Node(
                    op_type="Shape",
                    domain="",
                    inputs=[current_val],
                    outputs=[shape_val],
                    name=ctx.fresh_name("Shape"),
                )
            )
            _stamp_type_and_shape(shape_val, (rank,))
            _ensure_value_info(ctx, shape_val)

            axis_const = _const_i64(ctx, np.asarray(axis, dtype=np.int64), "rev_axis")

            dim_len = ir.Value(
                name=ctx.fresh_name("rev_dim_len"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape(()),
            )
            ctx.add_node(
                ir.Node(
                    op_type="Gather",
                    domain="",
                    inputs=[shape_val, axis_const],
                    outputs=[dim_len],
                    name=ctx.fresh_name("Gather"),
                    attributes=[ir.Attr("axis", ir.AttributeType.INT, 0)],
                )
            )
            _stamp_type_and_shape(dim_len, ())
            _ensure_value_info(ctx, dim_len)

            start_val = ir.Value(
                name=ctx.fresh_name("rev_start"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape(()),
            )
            ctx.add_node(
                ir.Node(
                    op_type="Sub",
                    domain="",
                    inputs=[dim_len, one],
                    outputs=[start_val],
                    name=ctx.fresh_name("Sub"),
                )
            )
            _stamp_type_and_shape(start_val, ())
            _ensure_value_info(ctx, start_val)

            range_val = ir.Value(
                name=ctx.fresh_name("rev_range"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape((None,)),
            )
            ctx.add_node(
                ir.Node(
                    op_type="Range",
                    domain="",
                    inputs=[start_val, neg_one, neg_one],
                    outputs=[range_val],
                    name=ctx.fresh_name("Range"),
                )
            )
            _stamp_type_and_shape(range_val, (None,))
            _ensure_value_info(ctx, range_val)

            target_val = (
                out_val
                if idx == len(canonical_axes) - 1
                else ir.Value(
                    name=ctx.fresh_name("rev_out"),
                    type=getattr(current_val, "type", None),
                    shape=current_val.shape,
                )
            )

            ctx.add_node(
                ir.Node(
                    op_type="Gather",
                    domain="",
                    inputs=[current_val, range_val],
                    outputs=[target_val],
                    name=ctx.fresh_name("Gather"),
                    attributes=[ir.Attr("axis", ir.AttributeType.INT, axis)],
                )
            )
            _stamp_type_and_shape(target_val, input_shape)
            _ensure_value_info(ctx, target_val)

            current_val = target_val
