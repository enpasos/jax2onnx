import jax
from typing import TYPE_CHECKING
from onnx import helper

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def get_primitive():
    return jax.lax.max_p


def get_handler(s: "Jaxpr2OnnxConverter"):
    def _handle_max(node_inputs, node_outputs, params):
        """Handle JAX max primitive."""
        input_names = [s.get_name(inp) for inp in node_inputs]
        output_name = s.get_var_name(node_outputs[0])
        node = helper.make_node(
            "Max",
            inputs=input_names,
            outputs=[output_name],
            name=s.get_unique_name("max"),
        )
        s.add_node(node)

    return _handle_max
