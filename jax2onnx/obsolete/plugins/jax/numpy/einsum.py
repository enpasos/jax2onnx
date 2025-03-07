# file: jax2onnx/plugins/einsum.py

import jax.numpy as jnp
import onnx.helper as oh

from obsolete.convert import Z
from obsolete.typing_helpers import Supports2Onnx


def build_einsum_onnx_node(jax_function: Supports2Onnx, z: Z, **params) -> Z:
    """Convert `jax.numpy.einsum` into an ONNX Einsum node."""
    if "equation" not in params:
        raise ValueError("Einsum requires an 'equation' parameter.")

    equation = params["equation"]
    onnx_graph = z.onnx_graph
    input_shapes = z.shapes
    input_names = z.names

    # Generate a unique node name
    node_name = f"node{onnx_graph.next_id()}"

    # For shape inference, use the original equation
    infer_equation = equation

    # Compute output shape using JAX einsum for shape inference
    jnp_inputs = []
    for shape in input_shapes:
        shape_list = list(shape)
        # Replace any string dimension with 1 for shape inference
        for i in range(len(shape_list)):
            if not isinstance(shape_list[i], int) or shape_list[i] < 0:
                shape_list[i] = 1
        jnp_input = jnp.zeros(shape_list)
        jnp_inputs.append(jnp_input)

    jax_output = jnp.einsum(infer_equation, *jnp_inputs)
    output_shape = list(jax_output.shape)

    # Restore the original batch dimension value
    if onnx_graph.dynamic_batch_dim:
        output_shape[0] = -1  # Handle dynamic batch dimension
    else:
        output_shape[0] = input_shapes[0][0]

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


def get_test_params() -> list:
    """Return test parameters for verifying the ONNX conversion of einsum."""
    return [
        {
            "jax_component": "jax.numpy.einsum",
            "jax_doc": "https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.einsum.html",
            "onnx": [
                {
                    "component": "Einsum",
                    "doc": "https://onnx.ai/onnx/operators/onnx__Einsum.html",
                },
            ],
            "since": "v0.1.0",
            "testcases": [
                {
                    "testcase": "einsum_attention",
                    "input_shapes": [(1, 64, 8, 32), (1, 128, 8, 32)],
                    "component": jnp.einsum,
                    "params": {"equation": "BNHE,BMHE->BNHM"},
                    "generate_derived_batch_dim_testcases": False,
                },
                {
                    "testcase": "einsum_matmul_static",  # Renamed to distinguish from dynamic version
                    "input_shapes": [(32, 64), (64, 128)],
                    "component": jnp.einsum,
                    "params": {"equation": "ij,jk->ik"},
                    "generate_derived_batch_dim_testcases": False,
                },
                {
                    "testcase": "einsum_matmul",  # This is the one that gets tested with dynamic batch
                    "input_shapes": [
                        (32, 64),
                        (64, 32),
                    ],  # Compatible dimensions for matrix multiplication
                    "component": jnp.einsum,
                    "params": {
                        "equation": "ij,jk->ik"
                    },  # Standard matrix multiplication,
                    "generate_derived_batch_dim_testcases": False,
                },
                {
                    "testcase": "einsum_batch_matmul",
                    "input_shapes": [(10, 32, 64), (10, 64, 128)],
                    "component": jnp.einsum,
                    "params": {"equation": "bij,bjk->bik"},
                    "generate_derived_batch_dim_testcases": False,
                },
                # Update dynamic batch matmul test case to use compatible shapes and equation
                {
                    "testcase": "einsum_dynamic_batch_matmul",
                    "input_shapes": [
                        ["B", 64],
                        [64, 32],
                    ],  # Compatible dimensions for matrix multiplication
                    "component": jnp.einsum,
                    "params": {
                        "equation": "bi,ij->bj"
                    },  # Correct equation for these shapes,
                    "generate_derived_batch_dim_testcases": False,
                },
                # Add another dynamic batch test case
                {
                    "testcase": "einsum_dynamic_batch_matmul_batched",
                    "input_shapes": [["B", 32, 64], ["B", 64, 32]],
                    "component": jnp.einsum,
                    "params": {"equation": "bij,bjk->bik"},
                    "generate_derived_batch_dim_testcases": False,
                },
            ],
        }
    ]
