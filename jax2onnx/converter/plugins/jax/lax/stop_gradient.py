import jax
from typing import TYPE_CHECKING
from onnx import helper

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def get_primitive():
    return jax.lax.stop_gradient_p


def get_handler(s: "Jaxpr2OnnxConverter"):
    def _handle_stop_gradient(node_inputs, node_outputs, params):
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

    return _handle_stop_gradient


def get_metadata() -> dict:
    """
    Return metadata describing the plugin.

    This could include documentation links, test cases, version information, etc.
    For now, we return an empty list.
    """
    return {}
