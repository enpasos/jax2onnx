# file: jax2onnx/plugins/pool.py

import flax.nnx as nnx
import jax.numpy as jnp
import onnx.helper as oh

from transpose_utils import jax_shape_to_onnx_shape, onnx_shape_to_jax_shape


def build_pool_onnx_node(function, pool_type, input_shapes, input_names, onnx_graph, parameters):
    """
    Constructs an ONNX node for pooling operations (AveragePool, MaxPool).

    Args:
        pool_type (str): The type of ONNX pooling operation ('AveragePool' or 'MaxPool').
        input_shapes (list of tuples): Input tensor shapes.
        input_names (list of str): Names of input tensors.
        onnx_graph: The ONNX graph object where the node will be added.
        parameters (dict): Dictionary containing kernel shape, strides, and padding.

    Returns:
        tuple:
            - output_shapes (list of tuples): Shape of the output tensor.
            - onnx_output_names (list of str): Names of the generated ONNX output tensors.
    """


    window_shape = parameters.get("window_shape", (2, 2))
    strides = parameters.get("strides", (2, 2))
    padding = parameters.get("padding", "VALID")

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

    node_name = f"node{onnx_graph.counter_plusplus()}"


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

    # assume that the input shape is (B, C, H, W) in the ONNX format

    # transform shape to JAX format (B, H, W, C)
    input_shape_jax = onnx_shape_to_jax_shape(input_shape)
    # construct jax_example_input
    jax_example_input = jnp.zeros(input_shape_jax)

    if pool_type == 'AveragePool':
        jax_example_output = nnx.avg_pool(jax_example_input, window_shape=window_shape, strides=strides, padding=padding)
    else:
        jax_example_output =  nnx.max_pool(jax_example_input, window_shape=window_shape, strides=strides, padding=padding)

    jax_example_output_shape = jax_example_output.shape
    output_shapes = [jax_shape_to_onnx_shape(jax_example_output_shape)]


    onnx_graph.add_local_outputs(output_shapes, onnx_output_names)

    return output_shapes, onnx_output_names


# Assign ONNX node builders to Flax pooling functions
nnx.avg_pool.build_onnx = lambda function, *args: build_pool_onnx_node(function, 'AveragePool', *args)
nnx.max_pool.build_onnx = lambda function, *args: build_pool_onnx_node(function, 'MaxPool', *args)


def get_test_params():
    """
    Returns test parameters for verifying the ONNX conversion of pooling operations.

    Returns:
        list: A list of dictionaries, each defining a test case.
    """
    return [
        {
            "model_name": "avg_pool",
            "model": lambda: lambda x: nnx.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID'),
            "input_shapes": [(1, 32, 32, 3)],  # JAX shape: (B, H, W, C)
            "build_onnx": nnx.avg_pool.build_onnx,
            "export": {
                "window_shape": (2, 2), "strides": (2, 2), "padding": "VALID",
                "pre_transpose": [(0, 3, 1, 2)],  # Convert JAX (B, H, W, C) to ONNX (B, C, H, W)
                "post_transpose": [(0, 2, 3, 1)],  # Convert ONNX output back to JAX format
            }
        },
        {
            "model_name": "max_pool",
            "model": lambda: lambda x: nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID'),
            "input_shapes": [(1, 32, 32, 3)],  # JAX shape: (B, H, W, C)
            "build_onnx": nnx.max_pool.build_onnx,
            "export": {
                "window_shape": (2, 2), "strides": (2, 2), "padding": "VALID",
                "pre_transpose": [(0, 3, 1, 2)],  # Convert JAX (B, H, W, C) to ONNX (B, C, H, W)
                "post_transpose": [(0, 2, 3, 1)],  # Convert ONNX output back to JAX format
            }
        },
    ]
