from typing import TYPE_CHECKING

import jax
import numpy as np
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.integer_pow_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.integer_pow.html",
    onnx=[
        {
            "component": "Pow",
            "doc": "https://onnx.ai/onnx/operators/onnx__Pow.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="integer_pow",
    testcases=[
        {
            "testcase": "integer_pow",
            "callable": lambda x: jax.lax.integer_pow(x, 2),
            "input_shapes": [(3,)],
        }
    ],
)
class IntegerPowPlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.integer_pow to ONNX Pow."""

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX integer pow primitive."""
        input_name = s.get_name(node_inputs[0])
        output_name = s.get_var_name(node_outputs[0])
        exponent = params.get("y", 2)  # Default exponent is 2 if not provided

        # Promote exponent to the same dtype as the base *before* Pow.
        aval = node_inputs[0].aval
        promoted = np.array(exponent, dtype=aval.dtype)
        y_name = s.get_constant_name(promoted)

        node = helper.make_node(
            "Pow",
            inputs=[input_name, y_name],
            outputs=[output_name],
            name=s.get_unique_name("integer_pow"),
        )
        s.add_node(node)
