import jax
from typing import TYPE_CHECKING
from onnx import helper

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def get_primitive():
    return jax.lax.ne_p


def get_handler(s: "Jaxpr2OnnxConverter"):
    def _handle_ne(node_inputs, node_outputs, params):
        """Handle JAX ne primitive."""
        input_names = [s.get_name(inp) for inp in node_inputs]
        eq_output = s.get_unique_name("equal_output")
        output_name = s.get_var_name(node_outputs[0])
        node_1 = helper.make_node(
            "Equal",
            inputs=input_names,
            outputs=[eq_output],
            name=s.get_unique_name("ne_eq"),
        )
        s.add_node(node_1)
        node_2 = helper.make_node(
            "Not",
            inputs=[eq_output],
            outputs=[output_name],
            name=s.get_unique_name("ne_not"),
        )
        s.add_node(node_2)

    return _handle_ne
