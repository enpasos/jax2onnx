import jax
from typing import TYPE_CHECKING, List, Dict, Any
from onnx import helper

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def get_primitive():
    return jax.lax.select_n_p


def get_handler(s: "Jaxpr2OnnxConverter"):
    def _handle_select_n(node_inputs, node_outputs, params):
        """Handle JAX select_n primitive."""
        condition_name = s.get_name(node_inputs[0])
        false_name = s.get_name(node_inputs[1])
        true_name = s.get_name(node_inputs[2])
        output_name = s.get_var_name(node_outputs[0])
        node = helper.make_node(
            "Where",
            inputs=[condition_name, true_name, false_name],
            outputs=[output_name],
            name=s.get_unique_name("where"),
        )
        s.add_node(node)

    return _handle_select_n


def get_metadata() -> dict:
    """
    Return metadata describing the plugin.

    This could include documentation links, test cases, version information, etc.
    For now, we return an empty list.
    """
    return {}
