from typing import TYPE_CHECKING

import jax
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.add_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.add.html",
    onnx=[
        {
            "component": "Add",
            "doc": "https://onnx.ai/onnx/operators/onnx__Add.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="add",
    testcases=[
        {
            "testcase": "add",
            "callable": lambda x1, x2: x1 + x2,
            "input_shapes": [(3,), (3,)],
        }
    ],
)
class AddPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting jax.lax.add to ONNX.
    """

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX add primitive."""
        input_names = [s.get_name(inp) for inp in node_inputs]
        output_name = s.get_name(
            node_outputs[0]
        )  # Corrected: get_name, not get_var_name
        node = helper.make_node(
            "Add",
            inputs=input_names,
            outputs=[output_name],
            name=s.get_unique_name("add"),
        )
        s.add_node(node)
