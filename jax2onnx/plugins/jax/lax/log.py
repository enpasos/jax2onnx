# jax2onnx/plugins/jax/lax/log.py

from typing import Any

from jax import core
import jax

from jax2onnx.converter.typing_support import LoweringContextProtocol

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

JaxprEqn = getattr(core, "JaxprEqn", Any)


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
            "post_check_onnx_graph": EG(
                ["Log:3"],
                no_unused_inputs=True,
            ),
        }
    ],
)
class LogPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("log_in"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("log_out"))

        result = ctx.builder.Log(x_val, _outputs=[out_spec.name])
        result.type = out_spec.type
        result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)
