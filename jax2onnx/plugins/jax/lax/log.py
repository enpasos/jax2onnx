from typing import TYPE_CHECKING

import jax
import onnx_ir as ir

from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    pass


@register_primitive(
    jaxpr_primitive=jax.lax.log_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.log.html",
    onnx=[
        {
            "component": "Log",
            "doc": "https://onnx.ai/onnx/operators/onnx__Log.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="log",
    testcases=[
        {
            "testcase": "log",
            "callable": lambda x: jax.lax.log(x),
            "input_shapes": [(3,)],
        }
    ],
)
class LogPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx, eqn):
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("log_in"))
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("log_out"))

        node = ir.Node(
            op_type="Log",
            domain="",
            inputs=[x_val],
            outputs=[out_val],
            name=ctx.fresh_name("Log"),
        )
        ctx.add_node(node)
