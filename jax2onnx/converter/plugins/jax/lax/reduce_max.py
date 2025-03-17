import numpy as np
import jax
from typing import TYPE_CHECKING
from onnx import helper

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def get_primitive():
    return jax.lax.reduce_max_p


def get_handler(s: "Jaxpr2OnnxConverter"):
    def _handle_reduce_max(node_inputs, node_outputs, params):
        """Handle JAX reduce_max primitive."""
        input_name = s.get_name(node_inputs[0])
        output_name = s.get_var_name(node_outputs[0])
        axes = params["axes"]
        axes_name = s.get_constant_name(np.array(axes, dtype=np.int64))
        node = helper.make_node(
            "ReduceMax",
            inputs=[input_name, axes_name],
            outputs=[output_name],
            name=s.get_unique_name("reduce_max"),
            keepdims=0 if not params.get("keepdims", False) else 1,
        )
        s.add_node(node)

    return _handle_reduce_max


def get_metadata() -> dict:
    """Return metadata describing this plugin and its test cases."""
    return {
        "jaxpr_primitive": "reduce_max",
        "jax_doc": "https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce_max.html",
        "onnx": [
            {
                "component": "ReduceMax",
                "doc": "https://onnx.ai/onnx/operators/onnx__ReduceMax.html",
            }
        ],
        "since": "v0.2.0",
        "context": "plugins.lax",
        "testcases": [
            {
                "testcase": "reduce_max",
                "callable": lambda x: jax.lax.reduce_max(x, axes=(0,)),
                "input_shapes": [(3, 3)],
            }
        ],
    }
