# file: jax2onnx/plugins/reshape.py
import jax
import jax.numpy as jnp
import onnx
import onnx.helper as oh

from jax2onnx.to_onnx import pre_transpose, post_transpose


def build_reshape_onnx_node(function, input_shapes, input_names, onnx_graph, parameters):
    """
    Constructs an ONNX node for a reshape operation.

    Args:
        input_shapes (list of tuples): Input tensor shapes.
        input_names (list of str): Names of input tensors.
        onnx_graph: The ONNX graph object where the node will be added.
        parameters (dict or list of dict): Dictionary containing the target shape.

    Returns:
        tuple:
            - output_shapes (list of tuples): Shape of the output tensor.
            - output_names (list of str): Names of the generated ONNX output tensors.
    """
    # Ensure parameters is always a dictionary
    if isinstance(parameters, list):
        if not parameters or not isinstance(parameters[0], dict):
            raise ValueError("Parameters for reshape must be a dictionary containing 'shape'.")
        params = parameters[0]  # Extract dictionary from list
    elif isinstance(parameters, dict):
        params = parameters  # Use as-is
    else:
        raise ValueError("Parameters for reshape must be a dictionary or a list containing a dictionary with 'shape'.")

    if "shape" not in params:
        raise ValueError("Parameters for reshape must include 'shape'.")

    new_shape = tuple(params["shape"])

    # Apply pre-transpose if necessary

    input_shapes, transposed_input_names = pre_transpose(input_shapes, input_names,  onnx_graph, parameters)
    input_name = transposed_input_names[0]

    node_name = f"node{onnx_graph.counter_plusplus()}"

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
            inputs=[input_name, shape_tensor_name],  # Ensure input_name is a string
            outputs=output_names,
            name=node_name,
        )
    )

    # Compute output shapes
    output_shapes = [new_shape]

    # Register final output in ONNX graph
    onnx_graph.add_local_outputs(output_shapes, output_names)

    # Apply post-transpose if necessary
    final_output_shapes, final_output_names = post_transpose(output_shapes, output_names, onnx_graph, parameters)

    return final_output_shapes, final_output_names


# Assign the reshape node builder
jax.numpy.reshape.to_onnx = build_reshape_onnx_node


# Example test parameters
def get_test_params():
    return [
        {
            "model_name": "reshape",
            "model": lambda: lambda x: jnp.reshape(x, shape=(10, 3)),
            "input_shapes": [(30,)],
            "to_onnx": jnp.reshape.to_onnx,
            "export": {
                "shape": (10, 3)
            },  # Simple reshape with no transpositions
        },

        {
            "model_name": "reshape2",
            "model": lambda: lambda x: jnp.reshape(x, (3, 3136)),
            "input_shapes": [(3, 7, 7, 64)],
            "to_onnx": jnp.reshape.to_onnx,
            "export": {
                "shape": (3, 3136),
            },
        },
    ]
