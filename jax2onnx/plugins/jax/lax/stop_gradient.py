from typing import TYPE_CHECKING

import jax
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.stop_gradient_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.stop_gradient.html",
    onnx=[
        {
            "component": "Identity",
            "doc": "https://onnx.ai/onnx/operators/onnx__Identity.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="stop_gradient",
    testcases=[
        {
            "testcase": "stop_gradient",
            "callable": lambda x: jax.lax.stop_gradient(x),
            "input_shapes": [(3,)],
        }
    ],
)
class StopGradientPlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.stop_gradient to ONNX Identity."""

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX stop_gradient primitive as Identity."""
        input_names = [s.get_name(inp) for inp in node_inputs]
        output_name = s.get_var_name(node_outputs[0])
        node = helper.make_node(
            "Identity",
            inputs=input_names,
            outputs=[output_name],
            name=s.get_unique_name("stop_gradient"),
        )
        s.add_node(node)
