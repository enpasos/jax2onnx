# file: jax2onnx/plugins/activation_functions.py

import onnx.helper as oh
import jax
import onnx
from transpose_utils import jax_shape_to_onnx_shape


def build_generic_onnx_node(op_type, input_shapes, input_names, onnx_graph, parameters=None):
    """
    Create a generic ONNX node for activation functions.

    Args:
        op_type (str): The type of ONNX operation (e.g., 'Relu', 'Sigmoid').
        input_shapes (list of tuples): Input tensor shapes.
        input_names (list of str): Corresponding input names in ONNX format.
        onnx_graph: The ONNX graph object where the node will be added.
        parameters (optional): Additional parameters for the ONNX node (if any).

    Returns:
        tuple:
            - output_shapes (list of tuples): Shape of the output tensor.
            - onnx_output_names (list of str): Names of the generated ONNX output tensors.
    """
    node_name = f"node{onnx_graph.counter_plusplus()}"


    # Define ONNX output names
    onnx_output_names = [f'{node_name}_output']

    # Create ONNX node
    onnx_graph.add_node(
        oh.make_node(
            op_type,
            inputs=input_names,
            outputs=onnx_output_names,
            name=node_name,
        )
    )

    # Activation functions do not change the shape
    output_shapes = input_shapes

    # Register outputs in ONNX graph
    onnx_graph.add_local_outputs(output_shapes, onnx_output_names)

    return output_shapes, onnx_output_names


# Attach ONNX conversion methods to JAX activation functions
jax.nn.relu.build_onnx_node = lambda *args: build_generic_onnx_node('Relu', *args)
jax.nn.sigmoid.build_onnx_node = lambda *args: build_generic_onnx_node('Sigmoid', *args)
jax.nn.tanh.build_onnx_node = lambda *args: build_generic_onnx_node('Tanh', *args)
jax.nn.softmax.build_onnx_node = lambda *args: build_generic_onnx_node('Softmax', *args)
jax.nn.elu.build_onnx_node = lambda *args: build_generic_onnx_node('Elu', *args)
jax.nn.softplus.build_onnx_node = lambda *args: build_generic_onnx_node('Softplus', *args)


# LogSoftmax (requires axis parameter)
def build_log_softmax_onnx_node(input_shapes, input_names, onnx_graph, parameters=None):
    """
    Create an ONNX node for LogSoftmax with axis handling.

    Args:
        input_shapes (list of tuples): Input tensor shapes.
        input_names (list of str): Corresponding input names in ONNX format.
        onnx_graph: The ONNX graph object where the node will be added.
        parameters (optional): Dictionary containing axis information.

    Returns:
        tuple:
            - output_shapes (list of tuples): Shape of the output tensor.
            - onnx_output_names (list of str): Names of the generated ONNX output tensors.
    """
    node_name = f"node{onnx_graph.counter_plusplus()}"


    axis = parameters[0].get('axis', -1) if parameters else -1

    onnx_output_names = [f'{node_name}_output']

    onnx_graph.add_node(
        oh.make_node(
            'LogSoftmax',
            inputs=input_names,
            outputs=onnx_output_names,
            name=node_name,
            axis=axis,
        )
    )

    output_shapes = input_shapes
    onnx_graph.add_local_outputs(output_shapes, onnx_output_names)

    return output_shapes, onnx_output_names


jax.nn.log_softmax.build_onnx_node = build_log_softmax_onnx_node


# LeakyReLU (requires alpha parameter)
def build_leaky_relu_onnx_node(input_shapes, input_names, onnx_graph, parameters=None):
    """
    Create an ONNX node for LeakyReLU with alpha handling.

    Args:
        input_shapes (list of tuples): Input tensor shapes.
        input_names (list of str): Corresponding input names in ONNX format.
        onnx_graph: The ONNX graph object where the node will be added.
        parameters (optional): Dictionary containing alpha value.

    Returns:
        tuple:
            - output_shapes (list of tuples): Shape of the output tensor.
            - onnx_output_names (list of str): Names of the generated ONNX output tensors.
    """
    node_name = f"node{onnx_graph.counter_plusplus()}"


    alpha = parameters[0].get('alpha', 0.01) if parameters else 0.01

    onnx_output_names = [f'{node_name}_output']

    onnx_graph.add_node(
        oh.make_node(
            'LeakyRelu',
            inputs=input_names,
            outputs=onnx_output_names,
            name=node_name,
            alpha=alpha,
        )
    )

    output_shapes = input_shapes
    onnx_graph.add_local_outputs(output_shapes, onnx_output_names)

    return output_shapes, onnx_output_names


jax.nn.leaky_relu.build_onnx_node = build_leaky_relu_onnx_node


# Define test parameters for activation functions
def get_test_params():
    """
    Returns test parameters for verifying the ONNX conversion of activation functions.

    Returns:
        list: A list of dictionaries, each defining a test case.
    """
    return [
        {
            "model_name": "relu",
            "model": lambda: lambda x: jax.nn.relu(x),
            "input_shapes": [(1, 10)],
            "build_onnx_node": jax.nn.relu.build_onnx_node,
        },
        {
            "model_name": "sigmoid",
            "model": lambda: lambda x: jax.nn.sigmoid(x),
            "input_shapes": [(1, 10)],
            "build_onnx_node": jax.nn.sigmoid.build_onnx_node,
        },
        {
            "model_name": "tanh",
            "model": lambda: lambda x: jax.nn.tanh(x),
            "input_shapes": [(1, 10)],
            "build_onnx_node": jax.nn.tanh.build_onnx_node,
        },
        {
            "model_name": "softmax",
            "model": lambda: lambda x: jax.nn.softmax(x),
            "input_shapes": [(1, 10)],
            "build_onnx_node": jax.nn.softmax.build_onnx_node,
        },
        {
            "model_name": "log_softmax",
            "model": lambda: lambda x: jax.nn.log_softmax(x),
            "input_shapes": [(1, 10)],
            "build_onnx_node": jax.nn.log_softmax.build_onnx_node,
            "parameters": [{"axis": -1}],
        },
        {
            "model_name": "leaky_relu",
            "model": lambda: lambda x: jax.nn.leaky_relu(x),
            "input_shapes": [(1, 10)],
            "build_onnx_node": jax.nn.leaky_relu.build_onnx_node,
            "parameters": [{"alpha": 0.02}],
        },
        {
            "model_name": "elu",
            "model": lambda: lambda x: jax.nn.elu(x),
            "input_shapes": [(1, 10)],
            "build_onnx_node": jax.nn.elu.build_onnx_node,
        },
        {
            "model_name": "softplus",
            "model": lambda: lambda x: jax.nn.softplus(x),
            "input_shapes": [(1, 10)],
            "build_onnx_node": jax.nn.softplus.build_onnx_node,
        },
    ]
