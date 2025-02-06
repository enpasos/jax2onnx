# file: jax2onnx/plugins/squeeze.py

# JAX API reference: https://docs.jax.dev/en/latest/_autosummary/jax.numpy.squeeze.html#jax.numpy.squeeze
# ONNX Operator: https://onnx.ai/onnx/operators/onnx__Squeeze.html

import jax.numpy as jnp
import onnx
import onnx.helper as oh
import numpy as np
from jax2onnx.to_onnx import Z

from functools import partial

def build_squeeze_onnx_node(z, parameters=None):
    """
    Converts JAX numpy.squeeze operation to ONNX Squeeze operation.

    Args:
        z (Z): A container with input shapes, names, and the ONNX graph.
        parameters (dict, optional): Dictionary containing 'axes' specifying dimensions to remove.

    Returns:
        Z: Updated instance with new shapes and names.
    """
    if parameters is None or "axes" not in parameters:
        raise ValueError("Squeeze operation requires 'axes' parameter.")

    onnx_graph = z.onnx_graph
    input_name = z.names[0]
    input_shape = z.shapes[0]

    axes = parameters["axes"]
    node_name = f"node{onnx_graph.next_id()}"
    output_name = f"{node_name}_output"

    # Add Squeeze node to the ONNX graph
    onnx_graph.add_node(
        oh.make_node(
            "Squeeze",
            inputs=[input_name, f"{node_name}_axes"],
            outputs=[output_name],
            name=node_name,
        )
    )

    # Add initializer for squeeze axes
    onnx_graph.add_initializer(
        oh.make_tensor(
            f"{node_name}_axes",
            onnx.TensorProto.INT64,
            [len(axes)],
            np.array(axes, dtype=np.int64),
        )
    )

    # Compute new shape after squeeze
    output_shape = [dim for i, dim in enumerate(input_shape) if i not in axes]
    onnx_graph.add_local_outputs([output_shape], [output_name])

    # Corrected jax_function
    jax_function = partial(jnp.squeeze, axis=tuple(axes))

    return Z([output_shape], [output_name], onnx_graph, jax_function=jax_function)

# Attach ONNX conversion method to JAX squeeze function
jnp.squeeze.to_onnx = build_squeeze_onnx_node


def get_test_params():
    """
    Defines test parameters for verifying the ONNX conversion of the Squeeze operation.

    Returns:
        list: A list of test cases with expected squeeze parameters.
    """
    return [
        {
            "model_name": "squeeze_single_dim",
            "input_shapes": [(1, 49, 10)],  # Single batch dimension
            "to_onnx": jnp.squeeze.to_onnx,
            "export": {"axes": [0]},  # Removing the batch dimension
        },
        {
            "model_name": "squeeze_multiple_dims",
            "input_shapes": [(1, 49, 1, 10)],  # Multiple singleton dimensions
            "to_onnx": jnp.squeeze.to_onnx,
            "export": {"axes": [0, 2]},  # Removing batch and last singleton dimension
        },
        {
            "model_name": "squeeze_vit_output",
            "input_shapes": [(1, 1, 10)],  # Common ViT output shape
            "to_onnx": jnp.squeeze.to_onnx,
            "export": {"axes": [1]},  # Removing the second singleton dimension
        },
        {
            "model_name": "squeeze_no_change",
            "input_shapes": [(3, 49, 10)],  # No singleton dimensions
            "to_onnx": jnp.squeeze.to_onnx,
            "export": {"axes": []},  # No dimensions should be removed
        }
    ]
