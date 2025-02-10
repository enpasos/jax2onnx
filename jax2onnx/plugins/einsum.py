# file: jax2onnx/plugins/einsum.py

import jax.numpy as jnp
import onnx.helper as oh

from jax2onnx.to_onnx import Z
from jax2onnx.typing_helpers import Supports2Onnx


def build_einsum_onnx_node(jax_function: Supports2Onnx, z: Z, **params) -> Z:
    """
    Converts `jax.numpy.einsum` into an ONNX Einsum node.

    Args:
        jax_function: The JAX einsum function.
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
    z.jax_function = lambda *args: jnp.einsum(equation, *args)
    return z


# Register the ONNX node builder for einsum
jnp.einsum.to_onnx = lambda *args, **kwargs: build_einsum_onnx_node(
    jnp.einsum, *args, **kwargs
)


def get_test_params():
    """
    Returns test parameters for verifying the ONNX conversion of einsum.

    Returns:
        list: A list of dictionaries, each defining a test case.
    """

    return [
        {
            "testcase": "einsum_attention",
            "input_shapes": [(1, 64, 8, 32), (1, 128, 8, 32)],
            "component": jnp.einsum,
            "params": {"equation": "BNHE,BMHE->BNHM"},
        },
        {
            "testcase": "einsum_matmul",
            "input_shapes": [(32, 64), (64, 128)],
            "component": jnp.einsum,
            "params": {"equation": "ij,jk->ik"},
        },
        {
            "testcase": "einsum_batch_matmul",
            "input_shapes": [(10, 32, 64), (10, 64, 128)],
            "component": jnp.einsum,
            "params": {"equation": "bij,bjk->bik"},
        },
    ]
