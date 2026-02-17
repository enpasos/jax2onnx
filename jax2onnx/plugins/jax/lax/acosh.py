# jax2onnx/plugins/jax/lax/acosh.py

from typing import Any

from jax import core
import jax
import numpy as np

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

JaxprEqn = getattr(core, "JaxprEqn", Any)


@register_primitive(
    jaxpr_primitive=jax.lax.acosh_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.acosh.html",
    onnx=[
        {
            "component": "Acosh",
            "doc": "https://onnx.ai/onnx/operators/onnx__Acosh.html",
        }
    ],
    since="0.12.1",
    context="primitives.lax",
    component="acosh",
    testcases=[
        {
            "testcase": "acosh",
            "callable": lambda x: jax.lax.acosh(x),
            "input_values": [np.array([1.0, 1.5, 3.0], dtype=np.float32)],
            "disable_float64_test": True,
            "post_check_onnx_graph": EG(
                ["Acosh:3"],
                no_unused_inputs=True,
            ),
        }
    ],
)
class AcoshPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("acosh_in"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("acosh_out"))

        result = ctx.builder.Acosh(x_val, _outputs=[out_spec.name])
        result.type = out_spec.type
        result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)
