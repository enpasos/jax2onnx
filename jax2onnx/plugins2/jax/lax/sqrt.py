from typing import TYPE_CHECKING

import jax
import onnx_ir as ir

from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    pass


@register_primitive(
    jaxpr_primitive=jax.lax.sqrt_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.sqrt.html",
    onnx=[
        {
            "component": "Sqrt",
            "doc": "https://onnx.ai/onnx/operators/onnx__Sqrt.html",
        }
    ],
    since="v0.2.0",
    context="primitives2.lax",
    component="sqrt",
    testcases=[
        {
            "testcase": "sqrt",
            "callable": lambda x: jax.lax.sqrt(x),
            "input_shapes": [(3,)],
        }
    ],
)
class SqrtPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx, eqn):
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("sqrt_in"))
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("sqrt_out"))

        node = ir.Node(
            op_type="Sqrt",
            domain="",
            inputs=[x_val],
            outputs=[out_val],
            name=ctx.fresh_name("Sqrt"),
        )
        ctx.add_node(node)
