from typing import TYPE_CHECKING

import jax
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.sin_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.sin.html",
    onnx=[
        {
            "component": "Sin",
            "doc": "https://onnx.ai/onnx/operators/onnx__Sin.html",
        }
    ],
    since="v0.4.4",
    context="primitives.lax",
    component="sin",
    testcases=[
        {
            "testcase": "sin",
            "callable": lambda x: jax.lax.sin(x),
            "input_shapes": [(3,)],
        }
    ],
)
class SinPlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.sin to ONNX Sin."""

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX sin primitive."""
        input_name = s.get_name(node_inputs[0])
        output_name = s.get_var_name(node_outputs[0])
        node = helper.make_node(
            "Sin",
            inputs=[input_name],
            outputs=[output_name],
            name=s.get_unique_name("sin"),
        )
        s.add_node(node)
