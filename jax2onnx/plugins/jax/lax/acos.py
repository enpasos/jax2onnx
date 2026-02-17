# jax2onnx/plugins/jax/lax/acos.py

from typing import Any

from jax import core
import jax
import numpy as np

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

JaxprEqn = getattr(core, "JaxprEqn", Any)


@register_primitive(
    jaxpr_primitive=jax.lax.acos_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.acos.html",
    onnx=[
        {
            "component": "Acos",
            "doc": "https://onnx.ai/onnx/operators/onnx__Acos.html",
        }
    ],
    since="0.12.1",
    context="primitives.lax",
    component="acos",
    testcases=[
        {
            "testcase": "acos",
            "callable": lambda x: jax.lax.acos(x),
            "input_values": [np.array([-1.0, -0.5, 0.5], dtype=np.float32)],
            "disable_float64_test": True,
            "post_check_onnx_graph": EG(
                ["Acos:3"],
                no_unused_inputs=True,
            ),
        }
    ],
)
class AcosPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("acos_in"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("acos_out"))

        desired_name = out_spec.name
        if out_spec.producer() is not None:
            desired_name = ctx.fresh_name("acos_out")

        result = ctx.builder.Acos(x_val, _outputs=[desired_name])
        result.type = out_spec.type
        result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)
