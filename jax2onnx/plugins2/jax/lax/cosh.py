from typing import TYPE_CHECKING

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    pass


def _like(ctx, exemplar: ir.Value, name_hint: str) -> ir.Value:
    return ir.Value(
        name=ctx.fresh_name(name_hint), type=exemplar.type, shape=exemplar.shape
    )


@register_primitive(
    jaxpr_primitive=jax.lax.cosh_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.cosh.html",
    onnx=[
        {
            "component": "Cosh",
            "doc": "https://onnx.ai/onnx/operators/onnx__Cosh.html",
        }
    ],
    since="v0.4.4",
    context="primitives2.lax",
    component="cosh",
    testcases=[
        {
            "testcase": "cosh",
            "callable": lambda x: jax.lax.cosh(x),
            "input_shapes": [(3,)],
        }
    ],
)
class CoshPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx, eqn):
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("cosh_in"))
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("cosh_out"))

        x_dtype = np.dtype(getattr(x_var.aval, "dtype", np.float32))
        if x_dtype == np.float32:
            node = ir.Node(
                op_type="Cosh",
                domain="",
                inputs=[x_val],
                outputs=[out_val],
                name=ctx.fresh_name("Cosh"),
            )
            ctx.add_node(node)
            return

        exp_x = _like(ctx, x_val, "cosh_exp")
        ctx.add_node(
            ir.Node(
                op_type="Exp",
                domain="",
                inputs=[x_val],
                outputs=[exp_x],
                name=ctx.fresh_name("Exp"),
            )
        )

        neg_x = _like(ctx, x_val, "cosh_neg")
        ctx.add_node(
            ir.Node(
                op_type="Neg",
                domain="",
                inputs=[x_val],
                outputs=[neg_x],
                name=ctx.fresh_name("Neg"),
            )
        )

        exp_neg_x = _like(ctx, x_val, "cosh_exp_neg")
        ctx.add_node(
            ir.Node(
                op_type="Exp",
                domain="",
                inputs=[neg_x],
                outputs=[exp_neg_x],
                name=ctx.fresh_name("Exp"),
            )
        )

        sum_val = _like(ctx, x_val, "cosh_sum")
        ctx.add_node(
            ir.Node(
                op_type="Add",
                domain="",
                inputs=[exp_x, exp_neg_x],
                outputs=[sum_val],
                name=ctx.fresh_name("Add"),
            )
        )

        half = ctx.bind_const_for_var(object(), np.asarray(0.5, dtype=x_dtype))
        ctx.add_node(
            ir.Node(
                op_type="Mul",
                domain="",
                inputs=[sum_val, half],
                outputs=[out_val],
                name=ctx.fresh_name("Mul"),
            )
        )
