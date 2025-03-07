import jax
from typing import TYPE_CHECKING, List, Dict, Any
from onnx import helper

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def get_primitive():
    return jax.lax.exp_p


def get_handler(s: "Jaxpr2OnnxConverter"):
    def _handle_exp(node_inputs, node_outputs, params):
        """Handle JAX exp primitive."""
        input_name = s.get_name(node_inputs[0])
        output_name = s.get_var_name(node_outputs[0])
        node = helper.make_node(
            "Exp",
            inputs=[input_name],
            outputs=[output_name],
            name=s.get_unique_name("exp"),
        )
        s.add_node(node)

    return _handle_exp


def get_metadata() -> List[Dict[str, Any]]:
    """
    Return metadata describing the plugin.

    This could include documentation links, test cases, version information, etc.
    For now, we return an empty list.
    """
    return []
