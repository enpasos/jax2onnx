from typing import TYPE_CHECKING

import jax
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.cosh_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.cosh.html",
    onnx=[
        {
            "component": "Cosh",
            "doc": "https://onnx.ai/onnx/operators/onnx__Cosh.html",
        }
    ],
    since="v0.4.4",
    context="primitives.lax",
    component="cosh",
    testcases=[
        {
            "testcase": "cosh",
            "callable": lambda x: jax.lax.cosh(x),
            "input_shapes": [(3,)],
        }
    ],
)
class CoshPlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.cosh to ONNX Cosh."""

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX cosh primitive."""
        input_name = s.get_name(node_inputs[0])
        output_name = s.get_var_name(node_outputs[0])
        node = helper.make_node(
            "Cosh",
            inputs=[input_name],
            outputs=[output_name],
            name=s.get_unique_name("cosh"),
        )
        s.add_node(node)
