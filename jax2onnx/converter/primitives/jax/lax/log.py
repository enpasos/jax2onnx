import jax
from typing import TYPE_CHECKING
from onnx import helper

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def get_primitive():
    return jax.lax.log_p


def get_handler(s: "Jaxpr2OnnxConverter"):
    def _handle_log(node_inputs, node_outputs, params):
        """Handle JAX log primitive."""
        input_name = s.get_name(node_inputs[0])
        output_name = s.get_var_name(node_outputs[0])
        node = helper.make_node(
            "Log",
            inputs=[input_name],
            outputs=[output_name],
            name=s.get_unique_name("log"),
        )
        s.add_node(node)

    return _handle_log
