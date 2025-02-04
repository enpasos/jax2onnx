# file: jax2onnx/plugins/matmul.py

import jax.numpy as jnp
import onnx.helper as oh


def build_onnx_matmul(function, input_shapes, input_names, onnx_graph, parameters=None):
    """
    Constructs an ONNX node for `jax.numpy.matmul`, ensuring proper handling of batch dimensions
    and transpositions to match ONNX's `MatMul` behavior.
    """
    if parameters is None:
        parameters = {}

    if len(input_shapes) != 2:
        raise ValueError("MatMul requires exactly two inputs.")

    input_shape_A = [int(dim) for dim in input_shapes[0]]
    input_shape_B = [int(dim) for dim in input_shapes[1]]
    input_name_A = input_names[0]
    input_name_B = input_names[1]

    node_name = f"node{onnx_graph.counter_plusplus()}"

    # Compute output shape
    batch_dims = input_shape_A[:-2]  # Assume broadcasting rules for batch dims
    M, K_A = input_shape_A[-2], input_shape_A[-1]
    K_B, N = input_shape_B[-2], input_shape_B[-1]

    if K_A != K_B:
        raise ValueError(f"Incompatible MatMul shapes: {input_shape_A} x {input_shape_B}")

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

    print(f"DEBUG: MatMul Input A Shape: {input_shape_A}")
    print(f"DEBUG: MatMul Input B Shape: {input_shape_B}")
    print(f"DEBUG: MatMul Output Shape: {output_shape}")

    return [output_shape], [matmul_out_name]


jnp.matmul.build_onnx = build_onnx_matmul


def get_test_params():
    """
    Returns test parameters for verifying the ONNX conversion of MatMul.

    Returns:
        list: A list of dictionaries, each defining a test case.
    """
    return [
        {
            "model_name": "matmul_2d",
            "model": lambda: lambda a, b: jnp.matmul(a, b),
            "input_shapes": [(3, 4), (4, 3)],
            "build_onnx": jnp.matmul.build_onnx,
            "export": {},
        },
        {
            "model_name": "matmul_3d",
            "model": lambda: lambda a, b: jnp.matmul(a, b),
            "input_shapes": [(2, 3, 4), (2, 4, 3)],
            "build_onnx": jnp.matmul.build_onnx,
            "export": {},
        },
        {
            "model_name": "matmul_4d",
            "model": lambda: lambda a, b: jnp.matmul(a, b),
            "input_shapes": [(1, 2, 3, 4), (1, 2, 4, 3)],
            "build_onnx": jnp.matmul.build_onnx,
            "export": {},
        },
    ]
