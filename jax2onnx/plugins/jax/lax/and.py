from typing import TYPE_CHECKING

import jax
from onnx import helper
import jax.numpy as jnp

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.and_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.html#jax.lax.bitwise_and",
    onnx=[
        {
            "component": "And",
            "doc": "https://onnx.ai/onnx/operators/onnx__And.html",
        },
        {
            "component": "BitwiseAnd",
            "doc": "https://onnx.ai/onnx/operators/onnx__BitwiseAnd.html",
        },
    ],
    since="v0.6.5",
    context="primitives.lax",
    component="and",
    testcases=[
        {
            "testcase": "and_bool",
            "callable": lambda x, y: jax.lax.bitwise_and(x, y),
            "input_values": [
                jnp.array([True, True, False, False]),
                jnp.array([True, False, True, False]),
            ],
        },
        {
            "testcase": "and_int",
            "callable": lambda x, y: jax.lax.bitwise_and(x, y),
            "input_values": [
                jnp.array([1, 2, 3], dtype=jnp.int32),
                jnp.array([3, 1, 2], dtype=jnp.int32),
            ],
        },
    ],
)
class AndPlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.and to ONNX And or BitwiseAnd."""

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX and primitive."""
        input_name1 = s.get_name(node_inputs[0])
        input_name2 = s.get_name(node_inputs[1])
        output_name = s.get_var_name(node_outputs[0])

        input_dtype = node_inputs[0].aval.dtype
        is_bool = jnp.issubdtype(input_dtype, jnp.bool_)

        op_type = "And" if is_bool else "BitwiseAnd"

        node = helper.make_node(
            op_type,
            inputs=[input_name1, input_name2],
            outputs=[output_name],
            name=s.get_unique_name("and"),
        )
        s.add_node(node)
