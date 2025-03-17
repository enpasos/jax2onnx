import jax
from typing import TYPE_CHECKING
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


def get_metadata() -> dict:
    """Return metadata describing this plugin and its test cases."""
    return {
        "jaxpr_primitive": "sqrt",
        "jax_doc": "https://docs.jax.dev/en/latest/_autosummary/jax.lax.sqrt.html",
        "onnx": [
            {
                "component": "Sqrt",
                "doc": "https://onnx.ai/onnx/operators/onnx__Sqrt.html",
            }
        ],
        "since": "v0.2.0",
        "context": "plugins.lax",
        "testcases": [
            {
                "testcase": "sqrt",
                "callable": lambda x: jax.lax.sqrt(x),
                "input_shapes": [(3,)],
            }
        ],
    }
