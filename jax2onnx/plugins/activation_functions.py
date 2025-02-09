# file: jax2onnx/plugins/activation_functions.py

import jax
import onnx.helper as oh
from jax2onnx.to_onnx import Z

from functools import partial


def build_generic_onnx_node(op_type, jax_function, z, parameters=None):
    """
    Creates an ONNX node for standard activation functions (e.g., ReLU, Sigmoid).

    Args:
        op_type (str): The corresponding ONNX operation type (e.g., 'Relu', 'Sigmoid').
        z (Z): A container with input shapes, names, and the ONNX graph.
        parameters (dict, optional): Additional parameters for ONNX, if applicable.

    Returns:
        Z: Updated instance with new shapes and names.
    """

    onnx_graph = z.onnx_graph
    input_shape = z.shapes[0]
    input_name = z.names[0]
    z.jax_function = jax_function

    node_name = f"node{onnx_graph.next_id()}"
    output_names = [f"{node_name}_output"]

    # Add the activation function node to the ONNX graph
    onnx_graph.add_node(
        oh.make_node(
            op_type,
            inputs=[input_name],
            outputs=output_names,
            name=node_name,
        )
    )

    # Activation functions do not change the tensor shape
    output_shapes = [input_shape]
    onnx_graph.add_local_outputs(output_shapes, output_names)

    z.shapes = output_shapes
    z.names = output_names
    return z


# Attach ONNX conversion methods to JAX activation functions
jax.nn.relu.to_onnx = lambda *args: build_generic_onnx_node("Relu", jax.nn.relu, *args)
jax.nn.sigmoid.to_onnx = lambda *args: build_generic_onnx_node(
    "Sigmoid", jax.nn.sigmoid, *args
)
jax.nn.tanh.to_onnx = lambda *args: build_generic_onnx_node("Tanh", jax.nn.tanh, *args)
jax.nn.softmax.to_onnx = lambda *args: build_generic_onnx_node(
    "Softmax", jax.nn.softmax, *args
)
jax.nn.elu.to_onnx = lambda *args: build_generic_onnx_node("Elu", jax.nn.elu, *args)
jax.nn.softplus.to_onnx = lambda *args: build_generic_onnx_node(
    "Softplus", jax.nn.softplus, *args
)


def build_log_softmax_onnx_node(z, parameters=None):
    """
    Creates an ONNX node for LogSoftmax with configurable axis.

    Args:
        z (Z): A container with input shapes, names, and the ONNX graph.
        parameters (dict, optional): Dictionary containing 'axis' (default: -1).

    Returns:
        Z: Updated instance with new shapes and names.
    """
    onnx_graph = z.onnx_graph
    input_shape = z.shapes[0]
    input_name = z.names[0]

    node_name = f"node{onnx_graph.next_id()}"
    axis = parameters.get("axis", -1) if parameters else -1
    output_names = [f"{node_name}_output"]

    # Add LogSoftmax node to the ONNX graph
    onnx_graph.add_node(
        oh.make_node(
            "LogSoftmax",
            inputs=[input_name],
            outputs=output_names,
            name=node_name,
            axis=axis,
        )
    )

    output_shapes = [input_shape]
    onnx_graph.add_local_outputs(output_shapes, output_names)

    return Z(output_shapes, output_names, onnx_graph, jax_function=jax.nn.log_softmax)


jax.nn.log_softmax.to_onnx = build_log_softmax_onnx_node


def build_leaky_relu_onnx_node(z, parameters=None):
    """
    Creates an ONNX node for LeakyReLU with configurable alpha.

    Args:
        z (Z): A container with input shapes, names, and the ONNX graph.
        parameters (dict, optional): Dictionary containing 'alpha' (default: 0.01).

    Returns:
        Z: Updated instance with new shapes and names.
    """
    onnx_graph = z.onnx_graph
    input_shape = z.shapes[0]
    input_name = z.names[0]

    node_name = f"node{onnx_graph.next_id()}"
    alpha = parameters.get("alpha", 0.01) if parameters else 0.01
    output_names = [f"{node_name}_output"]

    # Add LeakyReLU node to the ONNX graph
    onnx_graph.add_node(
        oh.make_node(
            "LeakyRelu",
            inputs=[input_name],
            outputs=output_names,
            name=node_name,
            alpha=alpha,
        )
    )

    output_shapes = [input_shape]
    onnx_graph.add_local_outputs(output_shapes, output_names)

    return Z(output_shapes, output_names, onnx_graph, jax_function=jax.nn.leaky_relu)


jax.nn.leaky_relu.to_onnx = build_leaky_relu_onnx_node


def build_gelu_onnx_node(z, parameters=None):
    """
    Creates an ONNX node for GELU activation function.

    Args:
        z (Z): A container with input shapes, names, and the ONNX graph.
        parameters (dict, optional): Dictionary containing ONNX-specific parameters (not used).

    Returns:
        Z: Updated instance with new shapes and names.
    """
    onnx_graph = z.onnx_graph
    input_shape = z.shapes[0]
    input_name = z.names[0]

    node_name = f"node{onnx_graph.next_id()}"
    output_names = [f"{node_name}_output"]

    # Add GELU node to the ONNX graph
    onnx_graph.add_node(
        oh.make_node(
            "Gelu",
            inputs=[input_name],
            outputs=output_names,
            name=node_name,
        )
    )

    output_shapes = [input_shape]
    onnx_graph.add_local_outputs(output_shapes, output_names)
    return Z(
        output_shapes,
        output_names,
        onnx_graph,
        jax_function=partial(jax.nn.gelu, approximate=False),
    )


jax.nn.gelu.to_onnx = build_gelu_onnx_node


def get_test_params():
    """
    Defines test parameters for verifying the ONNX conversion of activation functions.

    Returns:
        list: A list of dictionaries, each representing a test case.
    """
    return [
        {
            "testcase": "relu",
            "input_shapes": [(1, 10)],
            "to_onnx": jax.nn.relu.to_onnx,
        },
        {
            "testcase": "sigmoid",
            "input_shapes": [(1, 10)],
            "to_onnx": jax.nn.sigmoid.to_onnx,
        },
        {
            "testcase": "tanh",
            "input_shapes": [(1, 10)],
            "to_onnx": jax.nn.tanh.to_onnx,
        },
        {
            "testcase": "softmax",
            "input_shapes": [(1, 10)],
            "to_onnx": jax.nn.softmax.to_onnx,
        },
        {
            "testcase": "log_softmax",
            "input_shapes": [(1, 10)],
            "to_onnx": jax.nn.log_softmax.to_onnx,
            "parameters": {"axis": -1},
        },
        {
            "testcase": "leaky_relu",
            "input_shapes": [(1, 10)],
            "to_onnx": jax.nn.leaky_relu.to_onnx,
            "parameters": {"alpha": 0.02},
        },
        {
            "testcase": "elu",
            "input_shapes": [(1, 10)],
            "to_onnx": jax.nn.elu.to_onnx,
        },
        {
            "testcase": "softplus",
            "input_shapes": [(1, 10)],
            "to_onnx": jax.nn.softplus.to_onnx,
        },
        {
            "testcase": "gelu",
            "input_shapes": [(1, 10)],
            "to_onnx": jax.nn.gelu.to_onnx,
        },
        {
            "testcase": "gelu2",
            "input_shapes": [(1, 512)],
            "to_onnx": jax.nn.gelu.to_onnx,
        },
    ]
