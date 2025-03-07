# file: jax2onnx/plugins/pool.py

from functools import partial

import flax.nnx as nnx
import jax.numpy as jnp
import onnx.helper as oh

from obsolete.convert import Z
from obsolete.transpose_utils import jax_shape_to_onnx_shape, onnx_shape_to_jax_shape


def to_onnx(pool_type, jax_function, z: Z, **params) -> Z:
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

    # Convert padding string to ONNX-compatible format
    if isinstance(padding, str):
        if padding == "VALID":
            pads = [0] * len(window_shape) * 2
        elif padding == "SAME":
            input_height, input_width = input_shapes[0][2], input_shapes[0][3]

            pad_h = max(
                (input_height - 1) * strides[0] + window_shape[0] - input_height, 0
            )
            pad_w = max(
                (input_width - 1) * strides[1] + window_shape[1] - input_width, 0
            )

            # Ensure pads are **smaller than the kernel size**
            pad_h = min(pad_h, window_shape[0] - 1)
            pad_w = min(pad_w, window_shape[1] - 1)

            pads = [pad_h // 2, pad_w // 2, pad_h - pad_h // 2, pad_w - pad_w // 2]
        else:
            raise ValueError(f"Unsupported padding type: {padding}")
    else:
        pads = padding  # Assume it's a valid tuple

    node_name = f"node{onnx_graph.next_id()}"
    onnx_output_names = [f"{node_name}_output"]

    # Add the ONNX Pooling node
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

    # Compute expected JAX output shape
    input_shape = input_shapes[0]
    input_shape_jax = list(onnx_shape_to_jax_shape(input_shape))

    # Remember the original batch dimension value
    original_batch_dim = input_shape_jax[0]

    # Use a concrete batch dimension of 1 for dummy test input
    input_shape_jax[0] = 1

    jax_example_input = jnp.zeros(input_shape_jax)

    jax_function_partial = partial(
        jax_function, window_shape=window_shape, strides=strides, padding=padding
    )

    jax_example_output = jax_function_partial(jax_example_input)
    jax_example_output_shape = list(jax_example_output.shape)

    # Restore the original batch dimension value
    input_shape_jax[0] = original_batch_dim
    jax_example_output_shape[0] = original_batch_dim

    output_shapes = [jax_shape_to_onnx_shape(tuple(jax_example_output_shape))]

    # Store expected output in ONNX graph
    onnx_graph.add_local_outputs(output_shapes, onnx_output_names)

    # ✅ Store `jax_function` with parameters so testing works correctly
    z.jax_function = jax_function_partial
    z.shapes = output_shapes
    z.names = onnx_output_names
    return z


# Assign ONNX node builders to Flax pooling functions
nnx.avg_pool.to_onnx = lambda z, **params: to_onnx(
    "AveragePool", nnx.avg_pool, z, **params
)
nnx.max_pool.to_onnx = lambda z, **params: to_onnx("MaxPool", nnx.max_pool, z, **params)


def get_test_params():
    """
    Returns test parameters for verifying the ONNX conversion of pooling operations.

    Returns:
        list: A list of dictionaries, each defining a test case.
    """
    return [
        {
            "jax_component": "flax.nnx.avg_pool",
            "jax_doc": "https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.avg_pool",
            "onnx": [
                {
                    "component": "AveragePool",
                    "doc": "https://onnx.ai/onnx/operators/onnx__AveragePool.html",
                }
            ],
            "since": "v0.1.0",
            "testcases": [
                {
                    "testcase": "avg_pool",
                    "input_shapes": [(1, 32, 32, 3)],  # JAX shape: (B, H, W, C)
                    "component": nnx.avg_pool,
                    "params": {
                        "window_shape": (2, 2),
                        "strides": (2, 2),
                        "padding": "SAME",
                        "pre_transpose": [
                            (0, 3, 1, 2)
                        ],  # Convert JAX (B, H, W, C) to ONNX (B, C, H, W)
                        "post_transpose": [
                            (0, 2, 3, 1)
                        ],  # Convert ONNX output back to JAX format
                    },
                }
            ],
        },
        {
            "jax_component": "flax.nnx.max_pool",
            "jax_doc": "https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.max_pool",
            "onnx": [
                {
                    "component": "MaxPool",
                    "doc": "https://onnx.ai/onnx/operators/onnx__MaxPool.html",
                }
            ],
            "since": "v0.1.0",
            "testcases": [
                {
                    "testcase": "max_pool",
                    "input_shapes": [(1, 32, 32, 3)],  # JAX shape: (B, H, W, C)
                    "component": nnx.max_pool,
                    "params": {
                        "window_shape": (2, 2),
                        "strides": (2, 2),
                        "padding": "SAME",
                        "pre_transpose": [
                            (0, 3, 1, 2)
                        ],  # Convert JAX (B, H, W, C) to ONNX (B, C, H, W)
                        "post_transpose": [
                            (0, 2, 3, 1)
                        ],  # Convert ONNX output back to JAX format
                    },
                },
            ],
        },
    ]
