from typing import TYPE_CHECKING

import jax
import onnx_ir as ir

from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    pass


@register_primitive(
    jaxpr_primitive=jax.lax.exp_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.exp.html",
    onnx=[
        {
            "component": "Exp",
            "doc": "https://onnx.ai/onnx/operators/onnx__Exp.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="exp",
    testcases=[
        {
            "testcase": "exp",
            "callable": lambda x: jax.lax.exp(x),
            "input_shapes": [(3,)],
        }
    ],
)
class ExpPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx, eqn):
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("exp_in"))
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("exp_out"))

        node = ir.Node(
            op_type="Exp",
            domain="",
            inputs=[x_val],
            outputs=[out_val],
            name=ctx.fresh_name("Exp"),
        )
        ctx.add_node(node)
