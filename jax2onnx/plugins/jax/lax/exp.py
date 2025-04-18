from typing import TYPE_CHECKING

import jax
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.exp_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.exp.html",
    onnx=[
        {
            "component": "Exp",
            "doc": "https://onnx.ai/onnx/operators/onnx__Exp.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="exp",
    testcases=[
        {
            "testcase": "exp",
            "callable": lambda x: jax.lax.exp(x),
            "input_shapes": [(3,)],
        }
    ],
)
class ExpPlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.exp to ONNX Exp."""

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX exp primitive."""
        input_name = s.get_name(node_inputs[0])
        output_name = s.get_var_name(node_outputs[0])
        node = helper.make_node(
            "Exp",
            inputs=[input_name],
            outputs=[output_name],
            name=s.get_unique_name("exp"),
        )
        s.add_node(node)
