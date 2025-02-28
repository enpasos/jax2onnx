# file: jax2onnx/plugins/squeeze.py

# JAX API: https://docs.jax.dev/en/latest/_autosummary/jax.numpy.squeeze.html#jax.numpy.squeeze
# ONNX Operator: https://onnx.ai/onnx/operators/onnx__Squeeze.html

from functools import partial

import jax.numpy as jnp
import numpy as np
import onnx
import onnx.helper as oh

from jax2onnx.convert import Z


def build_squeeze_onnx_node(z, **params):
    """
    Converts JAX numpy.squeeze operation to ONNX Squeeze operation.

    Args:
        z (Z): A container with input shapes, names, and the ONNX graph.
        **params: Dictionary containing 'axes' specifying dimensions to remove.

    Returns:
        Z: Updated instance with new shapes and names.
    """
    if "axes" not in params:
        raise ValueError("Squeeze operation requires 'axes' parameter.")

    onnx_graph = z.onnx_graph
    input_name = z.names[0]
    input_shape = z.shapes[0]

    axes = params["axes"]
    valid_axes = []
    for axis in axes:
        if axis == 0 and onnx_graph.dynamic_batch_dim:
            continue  # Skip dynamic batch dimension
        if isinstance(input_shape[axis], int) and input_shape[axis] == 1:
            valid_axes.append(axis)

    if not valid_axes:
        # If no valid axes, return the input as is (identity operation)
        output_name = input_name
        output_shape = input_shape
    else:
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
                [len(valid_axes)],
                np.array(valid_axes, dtype=np.int64),
            )
        )

        # Compute new shape after squeeze
        output_shape = [dim for i, dim in enumerate(input_shape) if i not in valid_axes]
        if onnx_graph.dynamic_batch_dim:
            output_shape[0] = -1  # Handle dynamic batch dimension

    onnx_graph.add_local_outputs([output_shape], [output_name])

    # Corrected jax_function
    jax_function = (
        partial(jnp.squeeze, axis=tuple(valid_axes)) if valid_axes else lambda x: x
    )

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
            "jax_component": "jax.numpy.squeeze",
            "jax_doc": "https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.squeeze.html",
            "onnx": [
                {
                    "component": "Squeeze",
                    "doc": "https://onnx.ai/onnx/operators/onnx__Squeeze.html",
                },
            ],
            "since": "v0.1.0",
            "testcases": [
                {
                    "testcase": "squeeze_single_dim",
                    "input_shapes": [(1, 49, 10)],  # Single batch dimension
                    "component": jnp.squeeze,
                    "params": {"axes": [0]},  # Removing the batch dimension
                },
                {
                    "testcase": "squeeze_multiple_dims",
                    "input_shapes": [(1, 49, 1, 10)],  # Multiple singleton dimensions
                    "component": jnp.squeeze,
                    "params": {
                        "axes": [0, 2]
                    },  # Removing batch and last singleton dimension
                },
                {
                    "testcase": "squeeze_vit_output",
                    "input_shapes": [(1, 1, 10)],  # Common ViT output shape
                    "component": jnp.squeeze,
                    "params": {"axes": [1]},  # Removing the second singleton dimension
                },
                # Add a test case for dynamic batch dimensions
                {
                    "testcase": "squeeze_dynamic_batch",
                    "input_shapes": [
                        ["B", 1, 10]
                    ],  # Dynamic batch with singleton dimension
                    "component": jnp.squeeze,
                    "params": {"axes": [1]},  # Removing only the singleton dimension
                },
            ],
        }
    ]
