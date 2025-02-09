# file: jax2onnx/plugins/pool.py

import flax.nnx as nnx
import jax.numpy as jnp
import onnx.helper as oh
from functools import partial

from jax2onnx.transpose_utils import jax_shape_to_onnx_shape, onnx_shape_to_jax_shape


def to_onnx(pool_type, jax_function, z, **params):
    """
    Constructs an ONNX node for pooling operations (AveragePool, MaxPool).

    Args:
        pool_type (str): The type of ONNX pooling operation ('AveragePool' or 'MaxPool').
        jax_function: The JAX function used for pooling.
        z (Z): A container with input shapes, names, and the ONNX graph.
        **params: Additional parameters containing 'window_shape', 'strides', and 'padding'.

    Returns:
        Z: Updated instance with new shapes and names.
    """

    input_shapes = z.shapes
    input_names = z.names
    onnx_graph = z.onnx_graph
    window_shape = params.get("window_shape", (2, 2))
    strides = params.get("strides", (2, 2))
    padding = params.get("padding", "VALID")

    # Convert padding string to ONNX-compatible pads
    if isinstance(padding, str):
        if padding == "VALID":
            pads = [0] * len(window_shape) * 2
        elif padding == "SAME":
            pads = [k // 2 for k in window_shape] * 2
        else:
            raise ValueError(f"Unsupported padding type: {padding}")
    else:
        pads = padding

    node_name = f"node{onnx_graph.next_id()}"

    onnx_output_names = [f"{node_name}_output"]

    onnx_graph.add_node(
        oh.make_node(
            pool_type,
            inputs=input_names,
            outputs=onnx_output_names,
            name=node_name,
            kernel_shape=window_shape,
            strides=strides,
            pads=pads,
        )
    )

    input_shape = input_shapes[0]

    # Transform shape to JAX format (B, H, W, C)
    input_shape_jax = onnx_shape_to_jax_shape(input_shape)
    jax_example_input = jnp.zeros(input_shape_jax)

    jax_function = partial(
        jax_function, window_shape=window_shape, strides=strides, padding=padding
    )

    jax_example_output = jax_function(jax_example_input)
    jax_example_output_shape = jax_example_output.shape
    output_shapes = [jax_shape_to_onnx_shape(jax_example_output_shape)]

    onnx_graph.add_local_outputs(output_shapes, onnx_output_names)

    z.jax_function = jax_function
    z.shapes = output_shapes
    z.names = onnx_output_names
    return z


# Assign ONNX node builders to Flax pooling functions
nnx.avg_pool.to_onnx = lambda z, **params: to_onnx(
    "AveragePool", nnx.avg_pool, z, **params
)
nnx.max_pool.to_onnx = lambda z, **params: to_onnx("MaxPool", nnx.max_pool, z, **params)
# nnx.min_pool.to_onnx = lambda z, **params: to_onnx('MinPool', nnx.min_pool, z, **params)


def get_test_params():
    """
    Returns test parameters for verifying the ONNX conversion of pooling operations.

    Returns:
        list: A list of dictionaries, each defining a test case.
    """
    return [
        {
            "testcase": "avg_pool",
            "input_shapes": [(1, 32, 32, 3)],  # JAX shape: (B, H, W, C)
            "to_onnx": nnx.avg_pool.to_onnx,
            "params": {
                "window_shape": (2, 2),
                "strides": (2, 2),
                "padding": "VALID",
                "pre_transpose": [
                    (0, 3, 1, 2)
                ],  # Convert JAX (B, H, W, C) to ONNX (B, C, H, W)
                "post_transpose": [
                    (0, 2, 3, 1)
                ],  # Convert ONNX output back to JAX format
            },
        },
        {
            "testcase": "max_pool",
            "input_shapes": [(1, 32, 32, 3)],  # JAX shape: (B, H, W, C)
            "to_onnx": nnx.max_pool.to_onnx,
            "params": {
                "window_shape": (2, 2),
                "strides": (2, 2),
                "padding": "VALID",
                "pre_transpose": [
                    (0, 3, 1, 2)
                ],  # Convert JAX (B, H, W, C) to ONNX (B, C, H, W)
                "post_transpose": [
                    (0, 2, 3, 1)
                ],  # Convert ONNX output back to JAX format
            },
        },
        # {
        #     "testcase": "min_pool",
        #     "input_shapes": [(1, 32, 32, 3)],  # JAX shape: (B, H, W, C)
        #     "to_onnx": nnx.min_pool.to_onnx,
        #     "params": {
        #         "window_shape": (2, 2), "strides": (2, 2), "padding": "VALID",
        #         "pre_transpose": [(0, 3, 1, 2)],  # Convert JAX (B, H, W, C) to ONNX (B, C, H, W)
        #         "post_transpose": [(0, 2, 3, 1)],  # Convert ONNX output back to JAX format
        #     }
        # },
    ]
