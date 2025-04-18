from typing import TYPE_CHECKING

import jax
import numpy as np
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.square_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.square.html",
    onnx=[
        {
            "component": "Mul",
            "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="square",
    testcases=[
        {
            "testcase": "square",
            "callable": lambda x: jax.lax.square(x),
            "input_shapes": [(3,)],
        }
    ],
)
class SquarePlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.square to ONNX Mul."""

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX square primitive."""
        input_name = s.get_name(node_inputs[0])
        output_name = s.get_var_name(node_outputs[0])
        power_value = np.array(2, dtype=np.int32)
        power_name = s.get_constant_name(power_value)
        node = helper.make_node(
            "Pow",
            inputs=[input_name, power_name],
            outputs=[output_name],
            name=s.get_unique_name("square"),
        )
        s.add_node(node)
