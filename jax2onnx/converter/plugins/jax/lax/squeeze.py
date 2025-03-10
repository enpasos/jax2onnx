import jax
import numpy as np
from typing import TYPE_CHECKING
from onnx import helper

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def get_primitive():
    return jax.lax.squeeze_p


def get_handler(s: "Jaxpr2OnnxConverter"):
    def _handle_squeeze(node_inputs, node_outputs, params):
        """Handle JAX squeeze primitive."""
        input_name = s.get_name(node_inputs[0])
        output_name = s.get_var_name(node_outputs[0])
        dims = params["dimensions"]

        # Use the new add_initializer method
        axes_name = s.get_unique_name("squeeze_axes")
        s.add_initializer(name=axes_name, vals=dims)

        node = helper.make_node(
            "Squeeze",
            inputs=[input_name, axes_name],
            outputs=[output_name],
            name=s.get_unique_name("squeeze"),
        )
        s.add_node(node)

    return _handle_squeeze


def get_metadata() -> dict:
    """
    Return metadata describing the plugin.

    This could include documentation links, test cases, version information, etc.
    For now, we return an empty list.
    """
    return {}
