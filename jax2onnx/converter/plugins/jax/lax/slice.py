import jax
import numpy as np
from typing import TYPE_CHECKING
from onnx import helper

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def get_primitive():
    return jax.lax.slice_p


def get_handler(s: "Jaxpr2OnnxConverter"):
    def _handle_slice(node_inputs, node_outputs, params):
        """Handle JAX slice primitive."""
        input_name = s.get_name(node_inputs[0])
        output_name = s.get_var_name(node_outputs[0])
        start_indices = params["start_indices"]
        limit_indices = params["limit_indices"]
        axes = list(range(len(start_indices)))
        starts_name = s.get_constant_name(np.array(start_indices, dtype=np.int64))
        ends_name = s.get_constant_name(np.array(limit_indices, dtype=np.int64))
        axes_name = s.get_constant_name(np.array(axes, dtype=np.int64))
        inputs_list = [input_name, starts_name, ends_name, axes_name]
        if "strides" in params and params["strides"]:
            strides = params["strides"]
            steps_name = s.get_constant_name(np.array(strides, dtype=np.int64))
            inputs_list.append(steps_name)
        node = helper.make_node(
            "Slice",
            inputs=inputs_list,
            outputs=[output_name],
            name=s.get_unique_name("slice"),
        )
        s.add_node(node)

    return _handle_slice


def get_metadata() -> dict:
    """Return metadata describing this plugin and its test cases."""
    return {
        "jaxpr_primitive": "slice",
        "jax_doc": "https://docs.jax.dev/en/latest/_autosummary/jax.lax.slice.html",
        "onnx": [
            {
                "component": "Slice",
                "doc": "https://onnx.ai/onnx/operators/onnx__Slice.html",
            }
        ],
        "since": "v0.1.0",
        "context": "plugins.lax",
        "testcases": [
            {
                "testcase": "slice_test1",
                "callable": lambda x: x[1:3],
                "input_shapes": [(5,)],
            }
        ],
    }
