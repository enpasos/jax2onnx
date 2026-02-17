# jax2onnx/plugins/jax/lax/asin.py

from typing import Any

from jax import core
import jax
import numpy as np

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

JaxprEqn = getattr(core, "JaxprEqn", Any)


@register_primitive(
    jaxpr_primitive=jax.lax.asin_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.asin.html",
    onnx=[
        {
            "component": "Asin",
            "doc": "https://onnx.ai/onnx/operators/onnx__Asin.html",
        }
    ],
    since="0.12.1",
    context="primitives.lax",
    component="asin",
    testcases=[
        {
            "testcase": "asin",
            "callable": lambda x: jax.lax.asin(x),
            "input_values": [np.array([-1.0, -0.5, 0.5], dtype=np.float32)],
            "disable_float64_test": True,
            "post_check_onnx_graph": EG(
                ["Asin:3"],
                no_unused_inputs=True,
            ),
        }
    ],
)
class AsinPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("asin_in"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("asin_out"))

        # Compute a safe output name: reuse out_spec.name if it has no producer,
        # otherwise allocate a fresh, unique name.
        desired_name = out_spec.name
        if hasattr(out_spec, "producer"):
            producer_fn = out_spec.producer
            if callable(producer_fn) and producer_fn() is not None:
                desired_name = ctx.fresh_name(desired_name)

        result = ctx.builder.Asin(x_val, _outputs=[desired_name])
        result.type = out_spec.type
        result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)
