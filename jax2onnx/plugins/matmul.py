# file: jax2onnx/plugins/matmul.py

import jax.numpy as jnp
import onnx.helper as oh


def to_onnx_matmul(z, parameters=None):
    """
    Constructs an ONNX node for jax.numpy.matmul, ensuring proper handling of batch dimensions
    and transpositions to match ONNX's MatMul behavior.

    Args:
        z (Z): A container with input shapes, names, and the ONNX graph.
        parameters (dict, optional): Additional parameters (currently unused).

    Returns:
        Z: Updated instance with new shapes and names.
    """
    if parameters is None:
        parameters = {}

    if len(z.shapes) != 2:
        raise ValueError("MatMul requires exactly two inputs.")

    input_shapes = z.shapes
    input_names = z.names
    onnx_graph = z.onnx_graph

    input_shape_A = list(map(int, input_shapes[0]))
    input_shape_B = list(map(int, input_shapes[1]))
    input_name_A = input_names[0]
    input_name_B = input_names[1]

    node_name = f"node{onnx_graph.next_id()}"

    # Compute output shape
    batch_dims = input_shape_A[:-2]  # Assume broadcasting rules for batch dims
    M, K_A = input_shape_A[-2], input_shape_A[-1]
    K_B, N = input_shape_B[-2], input_shape_B[-1]

    if K_A != K_B:
        raise ValueError(
            f"Incompatible MatMul shapes: {input_shape_A} x {input_shape_B}"
        )

    output_shape = batch_dims + [M, N]

    # Create MatMul node
    matmul_out_name = f"{node_name}_matmul"
    onnx_graph.add_node(
        oh.make_node(
            "MatMul",
            inputs=[input_name_A, input_name_B],
            outputs=[matmul_out_name],
            name=f"{node_name}_matmul",
        )
    )

    onnx_graph.add_local_outputs([output_shape], [matmul_out_name])

    # Update and return Z
    z.shapes = [output_shape]
    z.names = [matmul_out_name]
    z.jax_function = jnp.matmul
    return z


# Attach the ONNX conversion function to `jax.numpy.matmul`
jnp.matmul.to_onnx = to_onnx_matmul


def get_test_params():
    """
    Returns test parameters for verifying the ONNX conversion of MatMul.

    Returns:
        list: A list of dictionaries, each defining a test case.
    """
    return [
        {
            "testcase": "matmul_2d",
            "input_shapes": [(3, 4), (4, 3)],
            "to_onnx": jnp.matmul.to_onnx,
        },
        {
            "testcase": "matmul_3d",
            "input_shapes": [(2, 3, 4), (2, 4, 3)],
            "to_onnx": jnp.matmul.to_onnx,
        },
        {
            "testcase": "matmul_4d",
            "input_shapes": [(1, 2, 3, 4), (1, 2, 4, 3)],
            "to_onnx": jnp.matmul.to_onnx,
        },
    ]
