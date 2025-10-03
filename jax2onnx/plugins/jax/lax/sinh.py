from typing import TYPE_CHECKING

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    pass


def _like(ctx, exemplar: ir.Value, name_hint: str) -> ir.Value:
    return ir.Value(
        name=ctx.fresh_name(name_hint), type=exemplar.type, shape=exemplar.shape
    )


@register_primitive(
    jaxpr_primitive=jax.lax.sinh_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.sinh.html",
    onnx=[
        {
            "component": "Sinh",
            "doc": "https://onnx.ai/onnx/operators/onnx__Sinh.html",
        }
    ],
    since="v0.4.4",
    context="primitives.lax",
    component="sinh",
    testcases=[
        {
            "testcase": "sinh",
            "callable": lambda x: jax.lax.sinh(x),
            "input_shapes": [(3,)],
        }
    ],
)
class SinhPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx, eqn):
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("sinh_in"))
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("sinh_out"))

        x_dtype = np.dtype(getattr(x_var.aval, "dtype", np.float32))
        if x_dtype == np.float32:
            node = ir.Node(
                op_type="Sinh",
                domain="",
                inputs=[x_val],
                outputs=[out_val],
                name=ctx.fresh_name("Sinh"),
            )
            ctx.add_node(node)
            return

        exp_x = _like(ctx, x_val, "sinh_exp")
        ctx.add_node(
            ir.Node(
                op_type="Exp",
                domain="",
                inputs=[x_val],
                outputs=[exp_x],
                name=ctx.fresh_name("Exp"),
            )
        )

        neg_x = _like(ctx, x_val, "sinh_neg")
        ctx.add_node(
            ir.Node(
                op_type="Neg",
                domain="",
                inputs=[x_val],
                outputs=[neg_x],
                name=ctx.fresh_name("Neg"),
            )
        )

        exp_neg_x = _like(ctx, x_val, "sinh_exp_neg")
        ctx.add_node(
            ir.Node(
                op_type="Exp",
                domain="",
                inputs=[neg_x],
                outputs=[exp_neg_x],
                name=ctx.fresh_name("Exp"),
            )
        )

        diff_val = _like(ctx, x_val, "sinh_diff")
        ctx.add_node(
            ir.Node(
                op_type="Sub",
                domain="",
                inputs=[exp_x, exp_neg_x],
                outputs=[diff_val],
                name=ctx.fresh_name("Sub"),
            )
        )

        half = ctx.bind_const_for_var(object(), np.asarray(0.5, dtype=x_dtype))
        ctx.add_node(
            ir.Node(
                op_type="Mul",
                domain="",
                inputs=[diff_val, half],
                outputs=[out_val],
                name=ctx.fresh_name("Mul"),
            )
        )
