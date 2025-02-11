# file: jax2onnx/plugins/transpose.py

# JAX API reference: https://docs.jax.dev/en/latest/_autosummary/jax.numpy.transpose.html#jax.numpy.transpose
# ONNX Operator: https://onnx.ai/onnx/operators/onnx__Transpose.html

import jax.numpy as jnp
import onnx.helper as oh

from jax2onnx.to_onnx import Z


def build_transpose_onnx_node(z, **params):
    """
    Converts JAX numpy.transpose operation to ONNX Transpose operation.

    Args:
        z (Z): A container with input shapes, names, and the ONNX graph.
        **params: Dictionary containing 'axes' to specify the permutation.

    Returns:
        Z: Updated instance with new shapes and names.
    """
    if "axes" not in params:
        raise ValueError("Transpose operation requires 'axes' parameter.")

    onnx_graph = z.onnx_graph
    input_name = z.names[0]
    input_shape = z.shapes[0]
    axes = params["axes"]
    output_shape = [input_shape[i] for i in axes]

    node_name = f"node{onnx_graph.next_id()}"
    output_name = f"{node_name}_output"

    # Add Transpose node to the ONNX graph
    onnx_graph.add_node(
        oh.make_node(
            "Transpose",
            inputs=[input_name],
            outputs=[output_name],
            perm=axes,
            name=node_name,
        )
    )

    # Update output shapes
    onnx_graph.add_local_outputs([output_shape], [output_name])

    def jax_function(x):
        return jnp.transpose(x, axes=tuple(axes))

    return Z([output_shape], [output_name], onnx_graph, jax_function=jax_function)


# Attach ONNX conversion method to JAX transpose function
jnp.transpose.to_onnx = build_transpose_onnx_node


def get_test_params():
    return [
        {
            "jax_component": "jax.numpy.transpose",
            "jax_doc": "https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.transpose.html",
            "onnx": [
                {
                    "component": "Transpose",
                    "doc": "https://onnx.ai/onnx/operators/onnx__Transpose.html",
                },
            ],
            "since": "v0.1.0",
            "testcases": [
                {
                    "testcase": "transpose_basic",
                    "input_shapes": [(2, 3)],
                    "component": jnp.transpose,
                    "params": {"axes": [1, 0]},  # Change "perm" to "axes"
                },
                {
                    "testcase": "transpose_reverse",
                    "input_shapes": [(2, 3, 4)],
                    "component": jnp.transpose,
                    "params": {"axes": [2, 1, 0]},  # Change "perm" to "axes"
                },
                {
                    "testcase": "transpose_4d",
                    "input_shapes": [(1, 2, 3, 4)],
                    "component": jnp.transpose,
                    "params": {"axes": [0, 2, 3, 1]},
                },
                {
                    "testcase": "transpose_square_matrix",
                    "input_shapes": [(5, 5)],
                    "component": jnp.transpose,
                    "params": {"axes": [1, 0]},
                },
                {
                    "testcase": "transpose_high_dim",
                    "input_shapes": [(2, 3, 4, 5, 6)],
                    "component": jnp.transpose,
                    "params": {"axes": [4, 3, 2, 1, 0]},
                },
            ],
        }
    ]
