# jax2onnx/plugins/jax/lax/abs.py

from typing import Any

from jax import core
import jax

from jax2onnx.converter.typing_support import LoweringContextProtocol

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

JaxprEqn = getattr(core, "JaxprEqn", Any)


@register_primitive(
    jaxpr_primitive=jax.lax.abs_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.abs.html",
    onnx=[
        {
            "component": "Abs",
            "doc": "https://onnx.ai/onnx/operators/onnx__Abs.html",
        }
    ],
    since="v0.5.0",
    context="primitives.lax",
    component="abs",
    testcases=[
        {
            "testcase": "abs",
            "callable": lambda x: jax.lax.abs(x),
            "input_shapes": [(3,)],
            "post_check_onnx_graph": EG(
                ["Abs:3"],
                no_unused_inputs=True,
            ),
        }
    ],
)
class AbsPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("abs_in"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("abs_out"))

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("abs_out")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("abs_out")

        result = ctx.builder.Abs(x_val, _outputs=[desired_name])
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)
