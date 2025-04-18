"""
Plugin for handling the JAX convert_element_type primitive.

This plugin converts JAX's convert_element_type primitive to ONNX Cast operation.
"""

from typing import TYPE_CHECKING

import jax
import numpy as np
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.convert_element_type_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.convert_element_type.html",
    onnx=[
        {
            "component": "Cast",
            "doc": "https://onnx.ai/onnx/operators/onnx__Cast.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="convert_element_type",
    testcases=[
        {
            "testcase": "convert_element_type",
            "callable": lambda x: x.astype(np.int16),
            "input_shapes": [(3,)],
        }
    ],
)
class ConvertElementTypePlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting jax.lax.convert_element_type to ONNX.
    """

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX convert_element_type primitive."""
        input_names = [s.get_name(inp) for inp in node_inputs]
        output_name = s.get_name(node_outputs[0])  # Use s.get_name
        new_dtype = s.builder._numpy_dtype_to_onnx(params["new_dtype"])
        node = helper.make_node(
            "Cast",
            inputs=input_names,
            outputs=[output_name],
            name=s.get_unique_name("convert_element_type"),
            to=new_dtype,
        )
        s.add_node(node)
