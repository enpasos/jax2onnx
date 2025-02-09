# file: jax2onnx/plugins/tile.py

# JAX API: https://docs.jax.dev/en/latest/_autosummary/jax.numpy.tile.html#jax.numpy.tile
# ONNX Operator: https://onnx.ai/onnx/operators/onnx__Tile.html

import jax.numpy as jnp
import onnx
import onnx.helper as oh
import numpy as np
from jax2onnx.to_onnx import Z
from functools import partial


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
    input_shape = z.shapes[0]

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

    # Compute new shape after tiling
    output_shape = [dim * repeats[i] for i, dim in enumerate(input_shape)]
    onnx_graph.add_local_outputs([output_shape], [output_name])

    # Corrected jax_function
    jax_function = partial(jnp.tile, reps=tuple(repeats))

    return Z([output_shape], [output_name], onnx_graph, jax_function=jax_function)


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
            "testcase": "tile_2x",
            "input_shapes": [(2, 3)],
            "to_onnx": jnp.tile.to_onnx,
            "params": {"repeats": [2, 2]},
        },
        {
            "testcase": "tile_1d",
            "input_shapes": [(4,)],
            "to_onnx": jnp.tile.to_onnx,
            "params": {"repeats": [3]},
        },
        {
            "testcase": "tile_batch_dim",
            "input_shapes": [(1, 5, 5)],
            "to_onnx": jnp.tile.to_onnx,
            "params": {"repeats": [2, 1, 1]},
        },
        {
            "testcase": "tile_large",
            "input_shapes": [(3, 3)],
            "to_onnx": jnp.tile.to_onnx,
            "params": {"repeats": [4, 4]},
        },
    ]
