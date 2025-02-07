# file: jax2onnx/plugins/numpy_functions.py

import jax.numpy as jnp
import onnx.helper as oh


def build_add_onnx_node( z, parameters=None):
    """
    Constructs an ONNX node for element-wise addition.

    Args:
        z (Z): A container with input shapes, names, and the ONNX graph.
        parameters (dict, optional): Additional parameters (unused for Add).

    Returns:
        Z: Updated instance with new shapes and names.
    """
    onnx_graph = z.onnx_graph
    input_shapes = z.shapes
    input_names = z.names

    node_name = f"node{onnx_graph.next_id()}"
    output_shapes = [input_shapes[0]]  # Element-wise addition does not change shape
    output_names = [f"{node_name}_output"]

    # Add ONNX Add node
    onnx_graph.add_node(
        oh.make_node(
            "Add",
            inputs=input_names,
            outputs=output_names,
            name=node_name,
        )
    )

    onnx_graph.add_local_outputs(output_shapes, output_names)

    z.shapes = output_shapes
    z.names = output_names
    z.jax_function =  jnp.add
    return z


# Assign ONNX node builder to jax.numpy.add
jnp.add.to_onnx =  build_add_onnx_node


def build_concat_onnx_node( z, parameters):
    """
    Constructs an ONNX node for concatenation along a specified axis.

    Args:
        jax_function: The JAX function (for reference and testing, unused in ONNX).
        z (Z): A container with input shapes, names, and the ONNX graph.
        parameters (dict): Dictionary containing 'axis' information.

    Returns:
        Z: Updated instance with new shapes and names.
    """
    if not isinstance(parameters, dict) or "axis" not in parameters:
        raise TypeError("Expected parameters to be a dictionary containing 'axis'.")

    axis = parameters["axis"]  # Extract the axis parameter


    onnx_graph = z.onnx_graph
    input_shapes = z.shapes
    input_names = z.names

    node_name = f"node{onnx_graph.next_id()}"

    # Compute the output shape by summing the sizes along the concatenation axis
    output_shape = list(input_shapes[0])
    output_shape[axis] = sum(shape[axis] for shape in input_shapes)

    output_shapes = [tuple(output_shape)]
    output_names = [f"{node_name}_output"]

    # Add ONNX Concat node
    onnx_graph.add_node(
        oh.make_node(
            "Concat",
            inputs=input_names,
            outputs=output_names,
            name=node_name,
            axis=axis,
        )
    )

    onnx_graph.add_local_outputs(output_shapes, output_names)

    z.shapes = output_shapes
    z.names = output_names
    z.jax_function =   lambda *args: jnp.concatenate(args, axis=axis)  # Define the JAX function
    return z


# Assign ONNX node builder to jax.numpy.concatenate
jnp.concatenate.to_onnx = build_concat_onnx_node


def get_test_params():
    """
    Returns test parameters for verifying the ONNX conversion of numpy functions.

    Returns:
        list: A list of dictionaries, each defining a test case.
    """
    return [
        {
            "model_name": "add",
            "input_shapes": [(1, 10), (1, 10)],  # Two input shapes for element-wise addition
            "to_onnx": jnp.add.to_onnx,
        },
        {
            "model_name": "concat",
            "input_shapes": [(1, 10), (1, 10)],  # Compatible shapes for axis=1
            "to_onnx": jnp.concatenate.to_onnx,
            "export": {"axis": 1},  # Correct axis for concatenation
        },
    ]
