import jax
import numpy as np
from typing import TYPE_CHECKING
from onnx import helper

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def get_primitive():
    return jax.lax.square_p


def get_handler(s: "Jaxpr2OnnxConverter"):
    def _handle_square(node_inputs, node_outputs, params):
        """Handle JAX square primitive."""
        input_name = s.get_name(node_inputs[0])
        output_name = s.get_var_name(node_outputs[0])
        power_value = np.array(2, dtype=np.int32)
        power_name = s.get_constant_name(power_value)
        node = helper.make_node(
            "Pow",
            inputs=[input_name, power_name],
            outputs=[output_name],
            name=s.get_unique_name("square"),
        )
        s.add_node(node)

    return _handle_square


def get_metadata() -> dict:
    """Return metadata describing this plugin and its test cases."""
    return {
        "jaxpr_primitive": "square",
        "jax_doc": "https://docs.jax.dev/en/latest/_autosummary/jax.lax.square.html",
        "onnx": [
            {
                "component": "Mul",
                "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html",
            }
        ],
        "since": "v0.2.0",
        "context": "plugins.lax",
        "testcases": [
            {
                "testcase": "square",
                "callable": lambda x: jax.lax.square(x),
                "input_shapes": [(3,)],
            }
        ],
    }
