import jax
from typing import TYPE_CHECKING
from onnx import helper

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def get_primitive():
    return jax.lax.min_p


def get_handler(s: "Jaxpr2OnnxConverter"):
    def _handle_min(node_inputs, node_outputs, params):
        """Handle JAX min primitive."""
        input_names = [s.get_name(inp) for inp in node_inputs]
        output_name = s.get_var_name(node_outputs[0])
        node = helper.make_node(
            "Min",
            inputs=input_names,
            outputs=[output_name],
            name=s.get_unique_name("min"),
        )
        s.add_node(node)

    return _handle_min


def get_metadata() -> dict:
    """Return metadata describing this plugin and its test cases."""
    return {
        "jaxpr_primitive": "min",
        "jax_doc": "https://docs.jax.dev/en/latest/_autosummary/jax.lax.min.html",
        "onnx": [
            {
                "component": "Min",
                "doc": "https://onnx.ai/onnx/operators/onnx__Min.html",
            }
        ],
        "since": "v0.1.0",
        "context": "plugins.lax",
        "testcases": [
            {
                "testcase": "min_test1",
                "callable": lambda x1, x2: jax.lax.min(x1, x2),
                "input_shapes": [(3,), (3,)],
            }
        ],
    }
