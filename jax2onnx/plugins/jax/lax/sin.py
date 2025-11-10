# jax2onnx/plugins/jax/lax/sin.py

from typing import Any

from jax import core
import jax

from jax2onnx.converter.typing_support import LoweringContextProtocol

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

JaxprEqn = getattr(core, "JaxprEqn", Any)


@register_primitive(
    jaxpr_primitive=jax.lax.sin_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.sin.html",
    onnx=[
        {
            "component": "Sin",
            "doc": "https://onnx.ai/onnx/operators/onnx__Sin.html",
        }
    ],
    since="v0.4.4",
    context="primitives.lax",
    component="sin",
    testcases=[
        {
            "testcase": "sin",
            "callable": lambda x: jax.lax.sin(x),
            "input_shapes": [(3,)],
            "post_check_onnx_graph": EG(
                ["Sin:3"],
                no_unused_inputs=True,
            ),
        }
    ],
)
class SinPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("sin_in"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("sin_out"))

        result = ctx.builder.Sin(x_val, _outputs=[out_spec.name])
        result.type = out_spec.type
        result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)
