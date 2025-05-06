from typing import TYPE_CHECKING

import jax
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.abs_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.abs.html",
    onnx=[
        {
            "component": "Abs",
            "doc": "https://onnx.ai/onnx/operators/onnx__Abs.html",
        }
    ],
    since="v0.5.0",
    context="primitives.lax",
    component="abs",
    testcases=[
        {
            "testcase": "abs",
            "callable": lambda x: jax.lax.abs(x),
            "input_shapes": [(3,)],
        },
    ],
)
class AbsPlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.abs to ONNX Abs."""

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX abs primitive."""
        input_name = s.get_name(node_inputs[0])
        output_name = s.get_var_name(node_outputs[0])
        node = helper.make_node(
            "Abs",
            inputs=[input_name],
            outputs=[output_name],
            name=s.get_unique_name("abs"),
        )
        s.add_node(node)
