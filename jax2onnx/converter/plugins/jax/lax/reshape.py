import jax
import numpy as np
from typing import TYPE_CHECKING
from onnx import helper

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def get_primitive():
    return jax.lax.reshape_p


def get_handler(s: "Jaxpr2OnnxConverter"):
    def _handle_reshape(node_inputs, node_outputs, params):
        """Handle JAX reshape primitive."""
        input_name = s.get_name(node_inputs[0])
        output_name = s.get_var_name(node_outputs[0])
        new_shape = params["new_sizes"]
        input_shape = node_inputs[0].aval.shape
        # Detect if reshape is redundant for bias broadcasting:
        if len(new_shape) == 2 and new_shape[0] == 1 and input_shape == (new_shape[1],):
            s.var_to_name[node_outputs[0]] = input_name
            return

        # Use the new add_initializer method
        shape_name = s.get_unique_name("reshape_shape")
        s.add_initializer(name=shape_name, vals=new_shape)

        node = helper.make_node(
            "Reshape",
            inputs=[input_name, shape_name],
            outputs=[output_name],
            name=s.get_unique_name("reshape"),
        )
        s.add_node(node)

    return _handle_reshape


def get_metadata() -> dict:
    """
    Return metadata describing the plugin.

    This could include documentation links, test cases, version information, etc.
    For now, we return an empty list.
    """
    return {}
