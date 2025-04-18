from typing import TYPE_CHECKING

import jax
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.neg_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.neg.html",
    onnx=[
        {
            "component": "Neg",
            "doc": "https://onnx.ai/onnx/operators/onnx__Neg.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="neg",
    testcases=[
        {
            "testcase": "neg",
            "callable": lambda x: jax.lax.neg(x),
            "input_shapes": [(3,)],
        }
    ],
)
class NegPlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.neg to ONNX Neg."""

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX neg primitive."""
        input_names = [s.get_name(inp) for inp in node_inputs]
        output_name = s.get_var_name(node_outputs[0])
        node = helper.make_node(
            "Neg",
            inputs=input_names,
            outputs=[output_name],
            name=s.get_unique_name("neg"),
        )
        s.add_node(node)
