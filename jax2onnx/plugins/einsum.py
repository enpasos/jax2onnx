# file: jax2onnx/plugins/einsum.py

import jax.numpy as jnp
import onnx.helper as oh


def to_onnx_einsum(z, **params):
    """
    Converts `jax.numpy.einsum` into an ONNX Einsum node.

    Args:
        z (Z): A container with input shapes, names, and the ONNX graph.
        **params: Dictionary containing 'equation' information.

    Returns:
        Z: Updated instance with new shapes and names.
    """

    if "equation" not in params:
        raise ValueError("Einsum requires an 'equation' parameter.")

    equation = params["equation"]

    onnx_graph = z.onnx_graph
    input_shapes = z.shapes
    input_names = z.names

    # Generate a unique node name
    node_name = f"node{onnx_graph.next_id()}"

    # Compute output shape using JAX einsum for shape inference
    jnp_inputs = [jnp.zeros(shape) for shape in input_shapes]
    jax_output = jnp.einsum(equation, *jnp_inputs)
    output_shape = jax_output.shape
    output_names = [f"{node_name}_output"]

    # Add the Einsum node to the ONNX graph
    onnx_graph.add_node(
        oh.make_node(
            "Einsum",
            inputs=input_names,
            outputs=output_names,
            name=node_name,
            equation=equation,
        )
    )

    onnx_graph.add_local_outputs([output_shape], output_names)

    # Update and return Z
    z.shapes = [output_shape]
    z.names = output_names
    z.jax_function = lambda a, b: jnp.einsum(equation, a, b)
    return z


# Register the ONNX node builder for einsum
jnp.einsum.to_onnx = to_onnx_einsum


def get_test_params():
    """
    Returns test parameters for verifying the ONNX conversion of einsum.

    Returns:
        list: A list of dictionaries, each defining a test case.
    """

    equation = "BNHE,BMHE->BNHM"
    return [
        {
            "model_name": "einsum",
            "input_shapes": [(1, 64, 8, 32), (1, 128, 8, 32)],
            "to_onnx": jnp.einsum.to_onnx,
            "params": {"equation": equation},
        },
    ]
