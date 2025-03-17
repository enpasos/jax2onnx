import jax
from typing import TYPE_CHECKING
from onnx import helper

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def get_primitive():
    return jax.lax.tanh_p


def get_handler(s: "Jaxpr2OnnxConverter"):
    def _handle_tanh(node_inputs, node_outputs, params):
        """Handle JAX tanh primitive."""
        input_name = s.get_name(node_inputs[0])
        output_name = s.get_var_name(node_outputs[0])
        node = helper.make_node(
            "Tanh",
            inputs=[input_name],
            outputs=[output_name],
            name=s.get_unique_name("tanh"),
        )
        s.add_node(node)

    return _handle_tanh


def get_metadata() -> dict:
    """Return metadata describing this plugin and its test cases."""
    return {
        "jaxpr_primitive": "tanh",
        "jax_doc": "https://docs.jax.dev/en/latest/_autosummary/jax.lax.tanh.html",
        "onnx": [
            {
                "component": "Tanh",
                "doc": "https://onnx.ai/onnx/operators/onnx__Tanh.html",
            }
        ],
        "since": "v0.2.0",
        "context": "plugins.lax",
        "testcases": [
            {
                "testcase": "tanh",
                "callable": lambda x: jax.lax.tanh(x),
                "input_shapes": [(3,)],
            }
        ],
    }
