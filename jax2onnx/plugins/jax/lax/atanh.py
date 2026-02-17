# jax2onnx/plugins/jax/lax/atanh.py

from typing import Any

from jax import core
import jax
import numpy as np

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

JaxprEqn = getattr(core, "JaxprEqn", Any)


@register_primitive(
    jaxpr_primitive=jax.lax.atanh_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.atanh.html",
    onnx=[
        {
            "component": "Atanh",
            "doc": "https://onnx.ai/onnx/operators/onnx__Atanh.html",
        }
    ],
    since="0.12.1",
    context="primitives.lax",
    component="atanh",
    testcases=[
        {
            "testcase": "atanh",
            "callable": lambda x: jax.lax.atanh(x),
            "input_values": [np.array([-0.9, 0.0, 0.9], dtype=np.float32)],
            "disable_float64_test": True,
            "post_check_onnx_graph": EG(
                ["Atanh:3"],
                no_unused_inputs=True,
            ),
        }
    ],
)
class AtanhPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("atanh_in"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("atanh_out"))

        desired_name = out_spec.name
        if hasattr(out_spec, "producer") and out_spec.producer() is not None:
            desired_name = ctx.fresh_name(desired_name)

        result = ctx.builder.Atanh(x_val, _outputs=[desired_name])
        result.type = out_spec.type
        result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)
