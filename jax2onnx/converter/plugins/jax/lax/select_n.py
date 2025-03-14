import jax
from typing import TYPE_CHECKING, List, Dict, Any
from onnx import helper

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def get_primitive():
    return jax.lax.select_n_p


def get_handler(s: "Jaxpr2OnnxConverter"):
    def _handle_select_n(node_inputs, node_outputs, params):
        """Handle JAX select_n primitive."""
        condition_name = s.get_name(node_inputs[0])
        false_name = s.get_name(node_inputs[1])
        true_name = s.get_name(node_inputs[2])
        output_name = s.get_var_name(node_outputs[0])
        node = helper.make_node(
            "Where",
            inputs=[condition_name, true_name, false_name],
            outputs=[output_name],
            name=s.get_unique_name("where"),
        )
        s.add_node(node)

    return _handle_select_n


def get_metadata() -> dict:
    """Return metadata describing this plugin and its test cases."""
    return {
        "jaxpr_primitive": "select_n",
        "jax_doc": "https://docs.jax.dev/en/latest/_autosummary/jax.lax.select_n.html",
        "onnx": [
            {
                "component": "Where",
                "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
            }
        ],
        "since": "v0.2.0",
        "context": "plugins.lax",
        "testcases": [
            {
                "testcase": "select_n",
                "callable": lambda pred, x, y: jax.lax.select(pred, x, y),
                "input_shapes": [(3,), (3,), (3,)],
            }
        ],
    }
