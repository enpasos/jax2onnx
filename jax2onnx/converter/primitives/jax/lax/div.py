import jax
from typing import TYPE_CHECKING
from onnx import helper

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def get_primitive():
    return jax.lax.div_p


def get_handler(s: "Jaxpr2OnnxConverter"):
    def _handle_div(node_inputs, node_outputs, params):
        """Handle JAX div primitive."""
        input_names = [s.get_name(inp) for inp in node_inputs]
        output_name = s.get_var_name(node_outputs[0])
        node = helper.make_node(
            "Div",
            inputs=input_names,
            outputs=[output_name],
            name=s.get_unique_name("div"),
        )
        s.add_node(node)

    return _handle_div
