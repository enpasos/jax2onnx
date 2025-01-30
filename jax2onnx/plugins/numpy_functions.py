# file: jax2onnx/plugins/numpy_functions.py

import onnx.helper as oh
import jax.numpy as jnp


def build_add_onnx_node(input_shapes, input_names, onnx_graph, parameters=None):
    """
    Constructs an ONNX node for element-wise addition.

    Args:
        input_shapes (list of tuples): List of input tensor shapes.
        input_names (list of str): Names of input tensors.
        onnx_graph: The ONNX graph object where the node will be added.
        parameters (optional): Additional parameters, currently unused.

    Returns:
        tuple:
            - output_shapes (list of tuples): Shape of the output tensor.
            - onnx_output_names (list of str): Names of the generated ONNX output tensors.
    """
    node_name = f"node{onnx_graph.counter_plusplus()}"


    # Element-wise addition does not change shape, assuming broadcasting is valid
    output_shapes = [input_shapes[0]]

    onnx_output_names = [f'{node_name}_output']

    onnx_graph.add_node(
        oh.make_node(
            'Add',
            inputs=input_names,
            outputs=onnx_output_names,
            name=node_name,
        )
    )

    onnx_graph.add_local_outputs(output_shapes, onnx_output_names)
    return output_shapes, onnx_output_names


# Assign ONNX node builder to jax.numpy.add
jnp.add.build_onnx_node = build_add_onnx_node


def build_concat_onnx_node(input_shapes, input_names, onnx_graph, parameters):
    """
    Constructs an ONNX node for concatenation along a specified axis.

    Args:
        input_shapes (list of tuples): List of input tensor shapes.
        input_names (list of str): Names of input tensors.
        onnx_graph: The ONNX graph object where the node will be added.
        parameters (dict): Dictionary containing 'axis' information.

    Returns:
        tuple:
            - output_shapes (list of tuples): Shape of the output tensor.
            - onnx_output_names (list of str): Names of the generated ONNX output tensors.
    """
    if not isinstance(parameters, list) or not isinstance(parameters[0], dict):
        raise TypeError("Expected parameters to be a list with a dictionary containing 'axis'.")

    axis = parameters[0].get("axis", 0)  # Default axis is 0

    node_name = f"node{onnx_graph.counter_plusplus()}"


    # Compute the output shape by summing the sizes along the concatenation axis
    output_shape = list(input_shapes[0])
    output_shape[axis] = sum(shape[axis] for shape in input_shapes)

    output_shapes = [tuple(output_shape)]
    onnx_output_names = [f"{node_name}_output"]

    onnx_graph.add_node(
        oh.make_node(
            "Concat",
            inputs=input_names,
            outputs=onnx_output_names,
            name=node_name,
            axis=axis,
        )
    )

    onnx_graph.add_local_outputs(output_shapes, onnx_output_names)
    return output_shapes, onnx_output_names


# Assign ONNX node builder to jax.numpy.concatenate
jnp.concatenate.build_onnx_node = build_concat_onnx_node


def get_test_params():
    """
    Returns test parameters for verifying the ONNX conversion of numpy functions.

    Returns:
        list: A list of dictionaries, each defining a test case.
    """
    return [
        {
            "model_name": "add",
            "model": lambda: lambda x, y: jnp.add(x, y),
            "input_shapes": [(1, 10), (1, 10)],  # Two input shapes for element-wise addition
            "build_onnx_node": jnp.add.build_onnx_node,
        },
        {
            "model_name": "concat",
            "model": lambda: lambda x, y: jnp.concatenate([x, y], axis=1),
            "input_shapes": [(1, 10), (1, 10)],  # Compatible shapes for axis=1
            "build_onnx_node": jnp.concatenate.build_onnx_node,
            "export": [{"axis": 1}],  # Correct axis for concatenation
        },
    ]
