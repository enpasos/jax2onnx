from typing import TYPE_CHECKING

import jax
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.div_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.div.html",
    onnx=[
        {
            "component": "Div",
            "doc": "https://onnx.ai/onnx/operators/onnx__Div.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="div",
    testcases=[
        {
            "testcase": "div",
            "callable": lambda x1, x2: x1 / x2,
            "input_shapes": [(3,), (3,)],
        }
    ],
)
class DivPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting jax.lax.div to ONNX.
    """

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX div primitive."""
        input_names = [s.get_name(inp) for inp in node_inputs]
        output_name = s.get_name(node_outputs[0])  # Use s.get_name
        node = helper.make_node(
            "Div",
            inputs=input_names,
            outputs=[output_name],
            name=s.get_unique_name("div"),
        )
        s.add_node(node)
