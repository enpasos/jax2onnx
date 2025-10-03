from typing import TYPE_CHECKING

import jax
import onnx_ir as ir

from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    pass


@register_primitive(
    jaxpr_primitive=jax.lax.sign_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.sign.html",
    onnx=[
        {
            "component": "Sign",
            "doc": "https://onnx.ai/onnx/operators/onnx__Sign.html",
        }
    ],
    since="v0.5.0",
    context="primitives.lax",
    component="sign",
    testcases=[
        {
            "testcase": "sign",
            "callable": lambda x: jax.lax.sign(x),
            "input_shapes": [(3,)],
        },
    ],
)
class SignPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx, eqn):
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("sign_in"))
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("sign_out"))

        node = ir.Node(
            op_type="Sign",
            domain="",
            inputs=[x_val],
            outputs=[out_val],
            name=ctx.fresh_name("Sign"),
        )
        ctx.add_node(node)
