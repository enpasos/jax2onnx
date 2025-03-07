import jax
import numpy as np
from typing import TYPE_CHECKING, List, Dict, Any
from onnx import helper

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def get_primitive():
    return jax.lax.integer_pow_p


def get_handler(s: "Jaxpr2OnnxConverter"):
    def _handle_integer_pow(node_inputs, node_outputs, params):
        """Handle JAX integer pow primitive."""
        input_name = s.get_name(node_inputs[0])
        output_name = s.get_var_name(node_outputs[0])
        exponent = params.get("y", 2)  # Default exponent is 2 if not provided
        power_value = np.array(exponent, dtype=np.int32)
        power_name = s.get_constant_name(power_value)
        node = helper.make_node(
            "Pow",
            inputs=[input_name, power_name],
            outputs=[output_name],
            name=s.get_unique_name("integer_pow"),
        )
        s.add_node(node)

    return _handle_integer_pow


def get_metadata() -> List[Dict[str, Any]]:
    """
    Return metadata describing the plugin.

    This could include documentation links, test cases, version information, etc.
    For now, we return an empty list.
    """
    return []
