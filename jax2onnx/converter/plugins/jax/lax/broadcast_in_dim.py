import jax
import numpy as np
from typing import TYPE_CHECKING
from onnx import helper

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def get_primitive():
    return jax.lax.broadcast_in_dim_p


def get_handler(s: "Jaxpr2OnnxConverter"):
    def _handle_broadcast_in_dim(node_inputs, node_outputs, params):
        """Handle JAX broadcast_in_dim primitive."""
        input_name = s.get_name(node_inputs[0])
        output_name = s.get_var_name(node_outputs[0])
        broadcast_dimensions = params["broadcast_dimensions"]
        shape = params["shape"]
        shape_name = s.get_constant_name(np.array(shape, dtype=np.int64))
        # First, reshape input to add singleton dimensions where necessary.
        reshape_output = s.get_unique_name("reshape_output")
        reshape_shape = []
        idx = 0
        for i in range(len(shape)):
            if i in broadcast_dimensions:
                if idx < len(node_inputs[0].aval.shape):
                    reshape_shape.append(node_inputs[0].aval.shape[idx])
                else:
                    reshape_shape.append(1)
                idx += 1
            else:
                reshape_shape.append(1)
        reshape_shape_name = s.get_constant_name(
            np.array(reshape_shape, dtype=np.int64)
        )
        node_reshape = helper.make_node(
            "Reshape",
            inputs=[input_name, reshape_shape_name],
            outputs=[reshape_output],
            name=s.get_unique_name("reshape_for_broadcast"),
        )
        s.add_node(node_reshape)
        # Then, expand to the target shape.
        node_expand = helper.make_node(
            "Expand",
            inputs=[reshape_output, shape_name],
            outputs=[output_name],
            name=s.get_unique_name("expand"),
        )
        s.add_node(node_expand)

    return _handle_broadcast_in_dim


def get_metadata() -> dict:
    """Return metadata describing this plugin and its test cases."""
    return {
        "jaxpr_primitive": "broadcast_in_dim",
        "jax_doc": "https://docs.jax.dev/en/latest/_autosummary/jax.lax.broadcast_in_dim.html",
        "onnx": [
            {
                "component": "Expand",
                "doc": "https://onnx.ai/onnx/operators/onnx__Expand.html",
            }
        ],
        "since": "v0.2.0",
        "context": "plugins.lax",
        "testcases": [
            {
                "testcase": "broadcast_in_dim",
                "callable": lambda x: jax.lax.broadcast_in_dim(
                    x, (3,), broadcast_dimensions=(0,)
                ),
                "input_shapes": [(3,)],
            }
        ],
    }
