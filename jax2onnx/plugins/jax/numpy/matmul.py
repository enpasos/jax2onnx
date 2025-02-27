# file: jax2onnx/plugins/matmul.py

import jax.numpy as jnp
import onnx.helper as oh

from jax2onnx.convert import Z
from jax2onnx.typing_helpers import Supports2Onnx


def build_matmul_onnx_node(jax_function: Supports2Onnx, z: Z, **params) -> Z:
    """
    Constructs an ONNX node for `jax.numpy.matmul`, ensuring proper handling of batch dimensions
    and transpositions to match ONNX's MatMul behavior.

    Args:
        jax_function: The JAX function (here, `jax.numpy.matmul`).
        z (Z): A container with input shapes, names, and the ONNX graph.
        **params: Additional parameters (currently unused).

    Returns:
        Z: Updated instance with new shapes and names.
    """
    if len(z.shapes) != 2:
        raise ValueError("MatMul requires exactly two inputs.")

    input_shapes = z.shapes
    input_names = z.names
    onnx_graph = z.onnx_graph

    input_shape_A = input_shapes[0]
    input_shape_B = input_shapes[1]
    input_name_A = input_names[0]
    input_name_B = input_names[1]

    node_name = f"node{onnx_graph.next_id()}"

    # raise exception if input shapes are not compatible
    if len(input_shape_A) < 1 or len(input_shape_B) < 1:
        raise ValueError(
            f"MatMul requires at least one dimension for each input: {input_shape_A} x {input_shape_B}"
        )
    # raise exception if input shapes are not equal:

    # Handle dynamic batch dimensions
    if isinstance(input_shape_A[0], str):
        batch_dim = input_shape_A[0]
        input_shape_A = list(input_shape_A[1:])
    else:
        batch_dim = None
        input_shape_A = list(map(int, input_shape_A))  # Convert to Python int list

    if len(input_shape_B) > 2 and isinstance(input_shape_B[0], str):
        batch_dim = input_shape_B[0]
        input_shape_B = list(input_shape_B[1:])
    elif isinstance(input_shape_B[1], str):
        batch_dim = input_shape_B[1]
        input_shape_B = [input_shape_B[0]]
    else:
        batch_dim = None
        input_shape_B = list(map(int, input_shape_B))  # Convert to Python int list

    # Compute output shape
    if len(input_shape_A) == 1 and len(input_shape_B) == 1:
        # Both inputs are 1D
        if input_shape_A[0] != input_shape_B[0]:
            raise ValueError(
                f"Incompatible MatMul shapes: {input_shape_A} x {input_shape_B}"
            )
        output_shape = []
    elif len(input_shape_A) == 2 and len(input_shape_B) == 2:
        # Both inputs are 2D
        if batch_dim:
            M, K_A = input_shape_A
            K_B, N = input_shape_B
            if K_A != K_B:
                raise ValueError(
                    f"Incompatible MatMul shapes: {input_shape_A} x {input_shape_B}"
                )
            output_shape = list((batch_dim,) + (M, N))
        else:
            M, K_A = input_shape_A
            K_B, N = input_shape_B
            if K_A != K_B:
                raise ValueError(
                    f"Incompatible MatMul shapes: {input_shape_A} x {input_shape_B}"
                )
            output_shape = [M, N]
    elif len(input_shape_A) == 2 and len(input_shape_B) == 1:
        # First input is 2D, second input is 1D
        M, K_A = input_shape_A
        K_B = input_shape_B[0]
        if K_A != K_B:
            raise ValueError(
                f"Incompatible MatMul shapes: {input_shape_A} x {input_shape_B}"
            )
        output_shape = [M]
    elif len(input_shape_A) == 1 and len(input_shape_B) == 2:
        # First input is 1D, second input is 2D
        K_A = input_shape_A[0]
        K_B, N = input_shape_B
        if K_A != K_B:
            raise ValueError(
                f"Incompatible MatMul shapes: {input_shape_A} x {input_shape_B}"
            )
        output_shape = [N]
    else:
        # Handle higher dimensions
        # batch_dims = tuple(
        #     input_shape_A[:-2]
        # )

        # Assume broadcasting rules for batch dims
        M, K_A = input_shape_A[-2], input_shape_A[-1]
        K_B, N = input_shape_B[-2], input_shape_B[-1]
        if K_A != K_B:
            raise ValueError(
                f"Incompatible MatMul shapes: {input_shape_A} x {input_shape_B}"
            )
        output_shape = list((batch_dim,) + (M, N))

    if batch_dim:
        output_shape = [batch_dim] + output_shape

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
    z.jax_function = jax_function
    return z


# ✅ Attach `to_onnx` method to `jax.numpy.matmul`
jnp.matmul.to_onnx = lambda *args, **kwargs: build_matmul_onnx_node(
    jnp.matmul, *args, **kwargs
)


def get_test_params():
    """
    Returns test parameters for verifying the ONNX conversion of MatMul.
    """
    return [
        {
            "jax_component": "jax.numpy.matmul",
            "jax_doc": "https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.matmul.html",
            "onnx": [
                {
                    "component": "MatMul",
                    "doc": "https://onnx.ai/onnx/operators/onnx__MatMul.html",
                },
            ],
            "since": "v0.1.0",
            "testcases": [
                {
                    "testcase": "matmul_2d",
                    "input_shapes": [(3, 4), (4, 3)],
                    "batch_input_shapes": [("B", 4), (4, "B")],
                    "component": jnp.matmul,
                },
                {
                    "testcase": "matmul_3d",
                    "input_shapes": [(2, 3, 4), (2, 4, 3)],
                    "batch_input_shapes": [("B", 3, 4), ("B", 4, 3)],
                    "component": jnp.matmul,
                },
                {
                    "testcase": "matmul_4d",
                    "input_shapes": [(1, 2, 3, 4), (1, 2, 4, 3)],
                    "batch_input_shapes": [("B", 2, 3, 4), ("B", 2, 4, 3)],
                    "component": jnp.matmul,
                },
            ],
        }
    ]
