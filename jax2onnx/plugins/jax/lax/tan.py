# jax2onnx/plugins/jax/lax/tan.py

from typing import Any

from jax import core
import jax

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

JaxprEqn = getattr(core, "JaxprEqn", Any)


@register_primitive(
    jaxpr_primitive=jax.lax.tan_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.tan.html",
    onnx=[
        {
            "component": "Tan",
            "doc": "https://onnx.ai/onnx/operators/onnx__Tan.html",
        }
    ],
    since="0.12.1",
    context="primitives.lax",
    component="tan",
    testcases=[
        {
            "testcase": "tan",
            "callable": lambda x: jax.lax.tan(x),
            "input_shapes": [(3,)],
            "disable_float64_test": True,
            "post_check_onnx_graph": EG(
                ["Tan:3"],
                no_unused_inputs=True,
            ),
        }
    ],
)
class TanPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("tan_in"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("tan_out"))

        # Avoid reusing an output name that may already have a producer.
        output_name = out_spec.name
        producer_attr = getattr(out_spec, "producer", None)
        if callable(producer_attr) and producer_attr() is not None:
            output_name = ctx.fresh_name("tan_out")

        result = ctx.builder.Tan(x_val, _outputs=[output_name])
        result.type = out_spec.type
        result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)
