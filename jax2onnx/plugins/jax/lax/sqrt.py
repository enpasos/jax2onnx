from typing import TYPE_CHECKING

import jax
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.sqrt_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.sqrt.html",
    onnx=[
        {
            "component": "Sqrt",
            "doc": "https://onnx.ai/onnx/operators/onnx__Sqrt.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="sqrt",
    testcases=[
        {
            "testcase": "sqrt",
            "callable": lambda x: jax.lax.sqrt(x),
            "input_shapes": [(3,)],
        }
    ],
)
class SqrtPlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.sqrt to ONNX Sqrt."""

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX sqrt primitive."""
        input_name = s.get_name(node_inputs[0])
        output_name = s.get_var_name(node_outputs[0])
        node = helper.make_node(
            "Sqrt",
            inputs=[input_name],
            outputs=[output_name],
            name=s.get_unique_name("sqrt"),
        )
        s.add_node(node)
