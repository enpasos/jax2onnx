"""
Plugin for handling the JAX random_split primitive.

This plugin converts JAX's random_split primitive to ONNX operations.
"""

from typing import TYPE_CHECKING

import jax
import numpy as np

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax._src.prng.random_split_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/jax.random.html#jax.random.split",
    onnx=[
        {
            "component": "Reshape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
        },
        {
            "component": "Tile",
            "doc": "https://onnx.ai/onnx/operators/onnx__Tile.html",
        },
    ],
    since="v0.4.0",
    context="primitives.jax.prng",
    component="random_split",
    testcases=[],
)
class RandomSplitPlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.random.split to ONNX Reshape and Tile operations."""

    def to_onnx(
        self, converter: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params
    ):
        """
        Convert jax.random.split to ONNX operations.

        This implementation uses Reshape and Tile operations to simulate JAX's random key splitting.

        Arguments:
            converter: The Jaxpr2OnnxConverter instance
            node_inputs: Input variables to the primitive
            node_outputs: Output variables from the primitive
            params: Parameters for the primitive
        """
        input_name = converter.get_name(node_inputs[0])
        output_name = converter.get_name(node_outputs[0])
        intermediate = converter.get_unique_name("random_split:x")

        # Create shape constants for reshape and tile
        reshape_name = converter.get_constant_name(np.array([1, 2], dtype=np.int64))
        repeat_name = converter.get_constant_name(
            np.array([params["shape"][0], 1], dtype=np.int64)
        )

        # Create reshape node
        node1 = converter.builder.create_node(
            "Reshape",
            [input_name, reshape_name],
            [intermediate],
            name=converter.get_unique_name("random_split:reshape"),
        )

        # Create tile node
        node2 = converter.builder.create_node(
            "Tile",
            [intermediate, repeat_name],
            [output_name],
            name=converter.get_unique_name("random_split:tile"),
        )

        # Add nodes to the graph
        converter.add_node(node1)
        converter.add_node(node2)
