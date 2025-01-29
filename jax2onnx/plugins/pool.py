# file: jax2onnx/plugins/pool.py
import onnx.helper as oh
import jax
import jax.numpy as jnp
import flax
import onnx


# Average Pooling
def build_avg_pool_onnx_node(jax_inputs, input_names, onnx_graph, parameters):
    kernel_shape = parameters.get("kernel_shape", (2, 2))
    strides = parameters.get("strides", (2, 2))
    padding = parameters.get("padding", "VALID")

    # Convert padding string to ONNX-compatible pads
    if isinstance(padding, str):
        if padding == "VALID":
            pads = [0] * len(kernel_shape) * 2
        elif padding == "SAME":
            pads = [k // 2 for k in kernel_shape] * 2
        else:
            raise ValueError(f"Unsupported padding type: {padding}")
    else:
        pads = padding

    # Perform average pooling in JAX
    jax_outputs = [flax.nnx.avg_pool(jax_inputs[0], window_shape=kernel_shape, strides=strides, padding=padding)]

    node_name = f"node{onnx_graph.get_counter()}"
    output_names = [f"{node_name}_output"]
    onnx_graph.increment_counter()

    # Add the AveragePool node with the necessary attributes
    onnx_graph.add_node(
        oh.make_node(
            "AveragePool",
            inputs=input_names,
            outputs=output_names,
            name=node_name,
            kernel_shape=kernel_shape,
            strides=strides,
            pads=pads,
        )
    )

    onnx_graph.add_local_outputs(jax_outputs, output_names)
    return jax_outputs, output_names

# Assign ONNX node builder to avg_pool function
flax.nnx.avg_pool.build_onnx_node = build_avg_pool_onnx_node


# Max Pooling
def build_max_pool_onnx_node(jax_inputs, input_names, onnx_graph, parameters):
    kernel_shape = parameters.get("kernel_shape", (2, 2))
    strides = parameters.get("strides", (2, 2))
    padding = parameters.get("padding", "VALID")

    # Convert padding string to ONNX-compatible pads
    if isinstance(padding, str):
        if padding == "VALID":
            pads = [0] * len(kernel_shape) * 2
        elif padding == "SAME":
            pads = [k // 2 for k in kernel_shape] * 2
        else:
            raise ValueError(f"Unsupported padding type: {padding}")
    else:
        pads = padding

    # Perform max pooling in JAX
    jax_outputs = [flax.nnx.max_pool(jax_inputs[0], window_shape=kernel_shape, strides=strides, padding=padding)]

    node_name = f"node{onnx_graph.get_counter()}"
    output_names = [f"{node_name}_output"]
    onnx_graph.increment_counter()

    # Add the MaxPool node with the necessary attributes
    onnx_graph.add_node(
        oh.make_node(
            "MaxPool",
            inputs=input_names,
            outputs=output_names,
            name=node_name,
            kernel_shape=kernel_shape,
            strides=strides,
            pads=pads,
        )
    )

    onnx_graph.add_local_outputs(jax_outputs, output_names)
    return jax_outputs, output_names

# Assign ONNX node builder to max_pool function
flax.nnx.max_pool.build_onnx_node = build_max_pool_onnx_node


# Example test parameters
def get_test_params():
    return [

        {
            "model_name": "avg_pool",
            "model": lambda: lambda x: flax.nnx.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID'),
            "input_shapes": [(1, 32, 32, 3)],
            "build_onnx_node": flax.nnx.avg_pool.build_onnx_node,
            "parameters": {"kernel_shape": (2, 2), "strides": (2, 2), "padding": "VALID"},
        },

        {
            "model_name": "max_pool",
            "model": lambda: lambda x: flax.nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID'),
            "input_shapes": [(1, 32, 32, 3)],
            "build_onnx_node": flax.nnx.max_pool.build_onnx_node,
            "parameters": {"kernel_shape": (2, 2), "strides": (2, 2), "padding": "VALID"},
        },
    ]
