import jax
from typing import TYPE_CHECKING
from onnx import helper

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def get_primitive():
    return jax.lax.and_p


def get_handler(s: "Jaxpr2OnnxConverter"):
    def _handle_and(node_inputs, node_outputs, params):
        """Handle JAX and primitive."""
        input_names = [s.get_name(inp) for inp in node_inputs]
        output_name = s.get_var_name(node_outputs[0])
        node = helper.make_node(
            "And",
            inputs=input_names,
            outputs=[output_name],
            name=s.get_unique_name("and"),
        )
        s.add_node(node)

    return _handle_and
