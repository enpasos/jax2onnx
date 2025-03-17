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


def get_metadata() -> dict:
    """Return metadata describing this plugin and its test cases."""
    return {
        "jaxpr_primitive": "max",
        "jax_doc": "https://docs.jax.dev/en/latest/_autosummary/jax.lax.max.html",
        "onnx": [
            {
                "component": "Max",
                "doc": "https://onnx.ai/onnx/operators/onnx__Max.html",
            }
        ],
        "since": "v0.2.0",
        "context": "plugins.lax",
        "testcases": [
            {
                "testcase": "max",
                "callable": lambda x1, x2: jax.lax.max(x1, x2),
                "input_shapes": [(3,), (3,)],
            }
        ],
    }
