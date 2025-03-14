import jax
from typing import TYPE_CHECKING
from onnx import helper

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def get_primitive():
    return jax.lax.transpose_p


def get_handler(s: "Jaxpr2OnnxConverter"):
    def _handle_transpose(node_inputs, node_outputs, params):
        """Handle JAX transpose primitive."""
        input_name = s.get_name(node_inputs[0])
        output_name = s.get_var_name(node_outputs[0])
        permutation = params["permutation"]
        node = helper.make_node(
            "Transpose",
            inputs=[input_name],
            outputs=[output_name],
            name=s.get_unique_name("transpose"),
            perm=permutation,
        )
        s.add_node(node)

    return _handle_transpose


def get_metadata() -> dict:
    """Return metadata describing this plugin and its test cases."""
    return {
        "jaxpr_primitive": "transpose",
        "jax_doc": "https://docs.jax.dev/en/latest/_autosummary/jax.lax.transpose.html",
        "onnx": [
            {
                "component": "Transpose",
                "doc": "https://onnx.ai/onnx/operators/onnx__Transpose.html",
            }
        ],
        "since": "v0.2.0",
        "context": "plugins.lax",
        "testcases": [
            {
                "testcase": "transpose_basic",
                "callable": lambda x: jax.lax.transpose(x, (1, 0)),
                "input_shapes": [(2, 3)],
            }
        ],
    }
