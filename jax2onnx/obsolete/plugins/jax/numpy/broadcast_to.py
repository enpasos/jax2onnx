# file: jax2onnx/plugins/jax/numpy/broadcast_to.py

import jax.numpy as jnp
import onnx
import onnx.helper as oh

from obsolete.convert import Z


def build_broadcast_onnx_node(z, **params):
    """
    Converts JAX lax.broadcast operation to ONNX Expand operation.

    Args:
        z (Z): A container with input shapes, names, and the ONNX graph.
        **params: Dictionary containing 'sizes' specifying the broadcast sizes along each axis.

    Returns:
        Z: Updated instance with new shapes and names.
    """
    if "sizes" not in params:
        raise ValueError("Broadcast operation requires 'sizes' parameter.")

    onnx_graph = z.onnx_graph
    input_names = z.names
    input_shapes = z.shapes

    sizes = params["sizes"]
    node_name = f"node{onnx_graph.next_id()}"
    output_name = f"{node_name}_output"

    # Check if we have multiple inputs and need to handle dynamic batch dimension
    has_batch_dim = any(dim == "B" for dim in sizes)
    has_second_input = len(input_shapes) > 1

    if has_batch_dim and has_second_input:
        # Extract the batch dimension from the shape of the second input at runtime
        shape_node_name = f"{node_name}_shape"
        batch_dim_node_name = f"{node_name}_batch_dim"
        sizes_node_name = f"{node_name}_dynamic_sizes"

        # Get shape of the second input
        onnx_graph.add_node(
            oh.make_node(
                "Shape",
                inputs=[input_names[1]],
                outputs=[shape_node_name],
                name=shape_node_name,
            )
        )

        # Get the batch dimension (first dimension)
        onnx_graph.add_node(
            oh.make_node(
                "Gather",
                inputs=[shape_node_name, f"{batch_dim_node_name}_index"],
                outputs=[batch_dim_node_name],
                name=batch_dim_node_name,
            )
        )

        # Add index initializer for Gather
        onnx_graph.add_initializer(
            oh.make_tensor(
                f"{batch_dim_node_name}_index",
                onnx.TensorProto.INT64,
                [1],
                [0],  # Get the first dimension
            )
        )

        # Create static part of sizes
        onnx_graph.add_initializer(
            oh.make_tensor(
                f"{node_name}_static_sizes",
                onnx.TensorProto.INT64,
                [len(sizes)],
                [0 if dim == "B" else dim for dim in sizes],
            )
        )

        # Create a mask for where to insert batch dimension
        onnx_graph.add_initializer(
            oh.make_tensor(
                f"{node_name}_mask",
                onnx.TensorProto.BOOL,
                [len(sizes)],
                [dim == "B" for dim in sizes],
            )
        )

        # Create a Tensor with batch dim repeated len(sizes) times
        onnx_graph.add_node(
            oh.make_node(
                "Expand",
                inputs=[batch_dim_node_name, f"{node_name}_dim_shape"],
                outputs=[f"{batch_dim_node_name}_expanded"],
                name=f"{batch_dim_node_name}_expand",
            )
        )

        # Add shape for the expansion
        onnx_graph.add_initializer(
            oh.make_tensor(
                f"{node_name}_dim_shape",
                onnx.TensorProto.INT64,
                [1],
                [len(sizes)],
            )
        )

        # Use Where to merge static sizes and dynamic batch dim
        onnx_graph.add_node(
            oh.make_node(
                "Where",
                inputs=[
                    f"{node_name}_mask",
                    f"{batch_dim_node_name}_expanded",
                    f"{node_name}_static_sizes",
                ],
                outputs=[sizes_node_name],
                name=f"{node_name}_where",
            )
        )

        # Add Expand node with dynamic sizes
        onnx_graph.add_node(
            oh.make_node(
                "Expand",
                inputs=[input_names[0], sizes_node_name],
                outputs=[output_name],
                name=node_name,
            )
        )
    else:
        # Handle static case
        # Add Expand node to the ONNX graph
        onnx_graph.add_node(
            oh.make_node(
                "Expand",
                inputs=[input_names[0], f"{node_name}_sizes"],
                outputs=[output_name],
                name=node_name,
            )
        )

        # Add initializer for sizes
        onnx_graph.add_initializer(
            oh.make_tensor(
                f"{node_name}_sizes",
                onnx.TensorProto.INT64,
                [len(sizes)],
                jnp.array([0 if dim == "B" else dim for dim in sizes], dtype=jnp.int64),
            )
        )

    # Return with appropriate JAX function for testing
    return Z(
        shapes=[tuple(sizes)],
        names=[output_name],
        onnx_graph=onnx_graph,
        jax_function=lambda x, y=None: jnp.broadcast_to(
            x,
            (
                [y.shape[0] if dim == "B" else dim for dim in sizes]
                if y is not None and len(sizes) > 0 and "B" in sizes
                else sizes
            ),
        ),
    )


# Attach ONNX conversion method to JAX lax.broadcast function
jnp.broadcast_to.to_onnx = build_broadcast_onnx_node


def get_test_params() -> list:
    """
    Defines test parameters for verifying the ONNX conversion of the Broadcast operation.

    Returns:
        list: A list of test cases with expected broadcast parameters.
    """
    return [
        {
            "jax_component": "jax.lax.broadcast",
            "jax_doc": "https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.broadcast.html",
            "onnx": [
                {
                    "component": "Expand",
                    "doc": "https://onnx.ai/onnx/operators/onnx__Expand.html",
                },
            ],
            "since": "v0.1.0",
            "testcases": [
                {
                    "testcase": "broadcast_a",
                    "input_shapes": [(1, 1, 5)],
                    "component": jnp.broadcast_to,
                    "params": {"sizes": [3, 1, 5]},
                    "generate_derived_batch_dim_testcases": False,
                },
                {
                    "testcase": "broadcast_b",
                    "input_shapes": [(1, 1, 5), ("B", 24, 24, 7)],
                    "component": jnp.broadcast_to,
                    "params": {"sizes": ["B", 1, 5]},
                    "dynamic_batch_dim": True,
                    "generate_derived_batch_dim_testcases": False,
                },
            ],
        }
    ]
