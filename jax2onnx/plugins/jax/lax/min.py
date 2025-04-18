from typing import TYPE_CHECKING

import jax
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.min_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.min.html",
    onnx=[
        {
            "component": "Min",
            "doc": "https://onnx.ai/onnx/operators/onnx__Min.html",
        }
    ],
    since="v0.1.0",
    context="primitives.lax",
    component="min",
    testcases=[
        {
            "testcase": "min_test1",
            "callable": lambda x1, x2: jax.lax.min(x1, x2),
            "input_shapes": [(3,), (3,)],
        }
    ],
)
class MinPlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.min to ONNX Min."""

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX min primitive."""
        input_names = [s.get_name(inp) for inp in node_inputs]
        output_name = s.get_var_name(node_outputs[0])
        node = helper.make_node(
            "Min",
            inputs=input_names,
            outputs=[output_name],
            name=s.get_unique_name("min"),
        )
        s.add_node(node)
