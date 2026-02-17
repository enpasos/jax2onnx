# jax2onnx/plugins/jax/lax/asinh.py

from typing import Any

from jax import core
import jax

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

JaxprEqn = getattr(core, "JaxprEqn", Any)


@register_primitive(
    jaxpr_primitive=jax.lax.asinh_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.asinh.html",
    onnx=[
        {
            "component": "Asinh",
            "doc": "https://onnx.ai/onnx/operators/onnx__Asinh.html",
        }
    ],
    since="0.12.1",
    context="primitives.lax",
    component="asinh",
    testcases=[
        {
            "testcase": "asinh",
            "callable": lambda x: jax.lax.asinh(x),
            "input_shapes": [(3,)],
            "disable_float64_test": True,
            "post_check_onnx_graph": EG(
                ["Asinh:3"],
                no_unused_inputs=True,
            ),
        }
    ],
)
class AsinhPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("asinh_in"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("asinh_out"))

        desired_name = out_spec.name
        if out_spec.producer() is not None:
            desired_name = ctx.fresh_name(out_spec.name)

        result = ctx.builder.Asinh(x_val, _outputs=[desired_name])
        result.type = out_spec.type
        result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)
