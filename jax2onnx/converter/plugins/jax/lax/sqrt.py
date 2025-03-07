import jax
from typing import TYPE_CHECKING, List, Dict, Any
from onnx import helper

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def get_primitive():
    return jax.lax.sqrt_p


def get_handler(s: "Jaxpr2OnnxConverter"):
    def _handle_sqrt(node_inputs, node_outputs, params):
        """Handle JAX sqrt primitive."""
        input_name = s.get_name(node_inputs[0])
        output_name = s.get_var_name(node_outputs[0])
        node = helper.make_node(
            "Sqrt",
            inputs=[input_name],
            outputs=[output_name],
            name=s.get_unique_name("sqrt"),
        )
        s.add_node(node)

    return _handle_sqrt


def get_metadata() -> List[Dict[str, Any]]:
    """
    Return metadata describing the plugin.

    This could include documentation links, test cases, version information, etc.
    For now, we return an empty list.
    """
    return []
