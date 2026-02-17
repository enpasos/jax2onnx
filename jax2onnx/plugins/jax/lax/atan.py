# jax2onnx/plugins/jax/lax/atan.py

from typing import Any

from jax import core
import jax

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

JaxprEqn = getattr(core, "JaxprEqn", Any)


@register_primitive(
    jaxpr_primitive=jax.lax.atan_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.atan.html",
    onnx=[
        {
            "component": "Atan",
            "doc": "https://onnx.ai/onnx/operators/onnx__Atan.html",
        }
    ],
    since="0.12.1",
    context="primitives.lax",
    component="atan",
    testcases=[
        {
            "testcase": "atan",
            "callable": lambda x: jax.lax.atan(x),
            "input_shapes": [(3,)],
            "disable_float64_test": True,
            "post_check_onnx_graph": EG(
                ["Atan:3"],
                no_unused_inputs=True,
            ),
        }
    ],
)
class AtanPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("atan_in"))
        desired_name = ctx.fresh_name("atan_out")
        out_spec = ctx.get_value_for_var(out_var, name_hint=desired_name)

        output_name = desired_name
        producer_attr = getattr(out_spec, "producer", None)
        if callable(producer_attr):
            producer = producer_attr()
        else:
            producer = producer_attr
        if producer is not None:
            output_name = ctx.fresh_name("atan_out")

        result = ctx.builder.Atan(x_val, _outputs=[output_name])
        result.type = out_spec.type
        result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)
