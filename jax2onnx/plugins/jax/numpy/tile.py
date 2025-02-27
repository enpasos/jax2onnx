# file: jax2onnx/plugins/tile.py

# JAX API: https://docs.jax.dev/en/latest/_autosummary/jax.numpy.tile.html#jax.numpy.tile
# ONNX Operator: https://onnx.ai/onnx/operators/onnx__Tile.html


import jax.numpy as jnp
import numpy as np
import onnx
import onnx.helper as oh

from jax2onnx.convert import Z


def build_tile_onnx_node(z, **params):
    """
    Converts JAX numpy.tile operation to ONNX Tile operation.

    Args:
        z (Z): A container with input shapes, names, and the ONNX graph.
        **params: Dictionary containing 'repeats' specifying repetitions along each axis.

    Returns:
        Z: Updated instance with new shapes and names.
    """
    if "repeats" not in params:
        raise ValueError("Tile operation requires 'repeats' parameter.")

    onnx_graph = z.onnx_graph
    input_name = z.names[0]

    repeats = params["repeats"]
    node_name = f"node{onnx_graph.next_id()}"
    output_name = f"{node_name}_output"

    # Add Tile node to the ONNX graph
    onnx_graph.add_node(
        oh.make_node(
            "Tile",
            inputs=[input_name, f"{node_name}_repeats"],
            outputs=[output_name],
            name=node_name,
        )
    )

    # Add initializer for repeats
    onnx_graph.add_initializer(
        oh.make_tensor(
            f"{node_name}_repeats",
            onnx.TensorProto.INT64,
            [len(repeats)],
            np.array(repeats, dtype=np.int64),
        )
    )

    return Z(
        shapes=[tuple(repeats)],
        names=[output_name],
        onnx_graph=onnx_graph,
        jax_function=lambda x: jnp.tile(x, repeats),
    )


# Attach ONNX conversion method to JAX tile function
jnp.tile.to_onnx = build_tile_onnx_node


def get_test_params():
    """
    Defines test parameters for verifying the ONNX conversion of the Tile operation.

    Returns:
        list: A list of test cases with expected tile parameters.
    """
    return [
        {
            "jax_component": "jax.numpy.tile",
            "jax_doc": "https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.tile.html",
            "onnx": [
                {
                    "component": "Tile",
                    "doc": "https://onnx.ai/onnx/operators/onnx__Tile.html",
                },
            ],
            "since": "v0.1.0",
            "testcases": [
                {
                    "testcase": "tile_a",
                    "input_shapes": [(2, 3)],
                    "component": jnp.tile,
                    "params": {"repeats": [1, 2]},
                },
                {
                    "testcase": "tile_b",
                    "input_shapes": [(1, 5, 5)],
                    "component": jnp.tile,
                    "params": {"repeats": [1, 2, 1]},
                },
                {
                    "testcase": "tile_c",
                    "input_shapes": [(3, 3)],
                    "component": jnp.tile,
                    "params": {"repeats": [1, 4]},
                },
            ],
        }
    ]
