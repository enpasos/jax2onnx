import jax
from typing import TYPE_CHECKING, List, Dict, Any
from onnx import helper

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def get_primitive():
    return jax.lax.eq_p


def get_handler(s: "Jaxpr2OnnxConverter"):
    def _handle_eq(node_inputs, node_outputs, params):
        """Handle JAX eq primitive."""
        input_names = [s.get_name(inp) for inp in node_inputs]
        output_name = s.get_var_name(node_outputs[0])
        node = helper.make_node(
            "Equal",
            inputs=input_names,
            outputs=[output_name],
            name=s.get_unique_name("eq"),
        )
        s.add_node(node)

    return _handle_eq


def get_metadata() -> List[Dict[str, Any]]:
    """
    Return metadata describing the plugin.

    This could include documentation links, test cases, version information, etc.
    For now, we return an empty list.
    """
    return []
