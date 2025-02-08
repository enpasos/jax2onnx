# file: jax2onnx/plugins/reshape.py

import jax.numpy as jnp
import onnx
import onnx.helper as oh
from jax2onnx.to_onnx import pre_transpose, post_transpose


def to_onnx_reshape(z, parameters):
    """
    Converts `jax.numpy.reshape` into an ONNX `Reshape` node.

    Args:
        z (Z): A container with input shapes, names, and the ONNX graph.
        parameters (dict): Dictionary containing the target shape.

    Returns:
        Z: Updated instance with new shapes and names.
    """

    if isinstance(parameters, list):
        if not parameters or not isinstance(parameters[0], dict):
            raise ValueError(
                "Parameters for reshape must be a dictionary containing 'shape'."
            )
        params = parameters[0]  # Extract dictionary from list
    elif isinstance(parameters, dict):
        params = parameters  # Use as-is
    else:
        raise ValueError(
            "Parameters for reshape must be a dictionary or a list containing a dictionary with 'shape'."
        )

    if "shape" not in params:
        raise ValueError("Parameters for reshape must include 'shape'.")

    new_shape = tuple(params["shape"])

    onnx_graph = z.onnx_graph

    # Apply pre-transpose if necessary
    z = pre_transpose(z, parameters)

    node_name = f"node{onnx_graph.next_id()}"
    input_name = z.names[0]
    output_names = [f"{node_name}_output"]

    # Create a shape tensor
    shape_tensor_name = f"{node_name}_shape"
    onnx_graph.add_initializer(
        oh.make_tensor(
            name=shape_tensor_name,
            data_type=onnx.TensorProto.INT64,
            dims=[len(new_shape)],
            vals=list(new_shape),
        )
    )

    # Add Reshape node
    onnx_graph.add_node(
        oh.make_node(
            "Reshape",
            inputs=[input_name, shape_tensor_name],
            outputs=output_names,
            name=node_name,
        )
    )

    # Compute output shapes
    output_shapes = [new_shape]

    # Register final output in ONNX graph
    onnx_graph.add_local_outputs(output_shapes, output_names)

    # Apply post-transpose if necessary
    z.shapes = output_shapes
    z.names = output_names
    z = post_transpose(z, parameters)

    z.jax_function = lambda x: jnp.reshape(x, new_shape)
    return z


# Assign the reshape node builder
jnp.reshape.to_onnx = to_onnx_reshape


def get_test_params():
    """
    Returns test parameters for verifying the ONNX conversion of reshape.

    Returns:
        list: A list of dictionaries, each defining a test case.
    """
    return [
        {
            "model_name": "reshape",
            # "model": lambda: lambda x: jnp.reshape(x, shape=(10, 3)),
            "input_shapes": [(30,)],
            "to_onnx": jnp.reshape.to_onnx,
            "export": {"shape": (10, 3)},
        },
        {
            "model_name": "reshape2",
            # "model": lambda: lambda x: jnp.reshape(x, (3, 3136)),
            "input_shapes": [(3, 7, 7, 64)],
            "to_onnx": jnp.reshape.to_onnx,
            "export": {
                "shape": (3, 3136),
            },
        },
    ]
