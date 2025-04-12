"""
Plugin for handling the JAX random_seed primitive.

This plugin converts JAX's random_seed primitive to ONNX operations.
"""

from typing import TYPE_CHECKING

import jax

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax._src.prng.random_seed_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/jax.random.html#jax.random.seed",
    onnx=[
        {
            "component": "Identity",
            "doc": "https://onnx.ai/onnx/operators/onnx__Identity.html",
        }
    ],
    since="v0.4.0",
    context="primitives.jax.prng",
    component="random_seed",
    testcases=[],  # Add test cases if available
)
class RandomSeedPlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.random.seed to ONNX Identity."""

    def to_onnx(
        self, converter: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params
    ):
        """
        Convert jax.random.seed to ONNX operations.

        Arguments:
            converter: The Jaxpr2OnnxConverter instance
            node_inputs: Input variables to the primitive
            node_outputs: Output variables from the primitive
            params: Parameters for the primitive
        """
        input_names = [converter.get_name(inp) for inp in node_inputs]
        output_names = [converter.get_name(out) for out in node_outputs]
        if not output_names:
            return

        node = converter.builder.create_node(
            "Identity",
            input_names,
            output_names,
            name=converter.get_unique_name(
                f"identity_{jax._src.prng.random_seed_p.name}"
            ),
        )
        converter.add_node(node)
