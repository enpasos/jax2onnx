# jax2onnx/plugins/jax/lax/erf.py

from typing import Any

from jax import core
import jax

from jax2onnx.converter.typing_support import LoweringContextProtocol

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

JaxprEqn = getattr(core, "JaxprEqn", Any)


@register_primitive(
    jaxpr_primitive=jax.lax.erf_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.erf.html",
    onnx=[
        {
            "component": "Erf",
            "doc": "https://onnx.ai/onnx/operators/onnx__Erf.html",
        }
    ],
    since="v0.4.4",
    context="primitives.lax",
    component="erf",
    testcases=[
        {
            "testcase": "erf",
            "callable": lambda x: jax.lax.erf(x),
            "input_shapes": [(3,)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Erf:3"],
                no_unused_inputs=True,
            ),
        }
    ],
)
class ErfPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("erf_in"))
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("erf_out"))

        result = ctx.builder.Erf(x_val, _outputs=[out_val.name])
        result.type = out_val.type
        result.shape = out_val.shape
        ctx.bind_value_for_var(out_var, result)
