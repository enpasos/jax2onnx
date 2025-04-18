from typing import TYPE_CHECKING

import jax
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.max_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.max.html",
    onnx=[
        {
            "component": "Max",
            "doc": "https://onnx.ai/onnx/operators/onnx__Max.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="max",
    testcases=[
        {
            "testcase": "max",
            "callable": lambda x1, x2: jax.lax.max(x1, x2),
            "input_shapes": [(3,), (3,)],
        }
    ],
)
class MaxPlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.max to ONNX Max."""

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX max primitive."""
        input_names = [s.get_name(inp) for inp in node_inputs]
        output_name = s.get_var_name(node_outputs[0])
        node = helper.make_node(
            "Max",
            inputs=input_names,
            outputs=[output_name],
            name=s.get_unique_name("max"),
        )
        s.add_node(node)
