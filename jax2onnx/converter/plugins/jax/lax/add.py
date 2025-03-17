import jax
from typing import TYPE_CHECKING
from onnx import helper

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def get_primitive():
    return jax.lax.add_p


def get_handler(s: "Jaxpr2OnnxConverter"):
    def _handle_add(node_inputs, node_outputs, params):
        """Handle JAX add primitive."""
        input_names = [s.get_name(inp) for inp in node_inputs]
        output_name = s.get_var_name(node_outputs[0])
        node = helper.make_node(
            "Add",
            inputs=input_names,
            outputs=[output_name],
            name=s.get_unique_name("add"),
        )
        s.add_node(node)

    return _handle_add


def get_metadata() -> dict:
    """Return metadata describing this plugin and its test cases."""
    return {
        "jaxpr_primitive": "add",
        "jax_doc": "https://docs.jax.dev/en/latest/_autosummary/jax.lax.add.html",
        "onnx": [
            {
                "component": "Add",
                "doc": "https://onnx.ai/onnx/operators/onnx__Add.html",
            }
        ],
        "since": "v0.2.0",
        "context": "plugins.lax",
        "testcases": [
            {
                "testcase": "add",
                "callable": lambda x1, x2: x1 + x2,
                "input_shapes": [(3,), (3,)],
            }
        ],
    }
