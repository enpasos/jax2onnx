from typing import TYPE_CHECKING

import jax
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.ge_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.ge.html",
    onnx=[
        {
            "component": "GreaterOrEqual",
            "doc": "https://onnx.ai/onnx/operators/onnx__GreaterOrEqual.html",
        }
    ],
    since="v0.7.4",
    context="primitives.lax",
    component="greater_equal",
    testcases=[
        {
            "testcase": "greater_equal",
            "callable": lambda x, y: x >= y,
            "input_shapes": [(3,), (3,)],
        },
    ],
)
class GreaterEqualPlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.ge to ONNX GreaterOrEqual."""

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX greater_equal primitive."""
        input_names = [s.get_name(inp) for inp in node_inputs]
        output_name = s.get_var_name(node_outputs[0])
        node = helper.make_node(
            "GreaterOrEqual",
            inputs=input_names,
            outputs=[output_name],
            name=s.get_unique_name("greater_equal"),
        )
        s.add_node(node)
