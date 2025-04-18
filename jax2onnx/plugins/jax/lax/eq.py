from typing import TYPE_CHECKING

import jax
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.eq_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.eq.html",
    onnx=[
        {
            "component": "Equal",
            "doc": "https://onnx.ai/onnx/operators/onnx__Equal.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="eq",
    testcases=[
        {
            "testcase": "eq",
            "callable": lambda x1, x2: x1 == x2,
            "input_shapes": [(3,), (3,)],
        }
    ],
)
class EqPlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.eq to ONNX Equal."""

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX eq primitive."""
        input_names = [s.get_name(inp) for inp in node_inputs]
        output_name = s.get_var_name(node_outputs[0])
        node = helper.make_node(
            "Equal",
            inputs=input_names,
            outputs=[output_name],
            name=s.get_unique_name("eq"),
        )
        s.add_node(node)
