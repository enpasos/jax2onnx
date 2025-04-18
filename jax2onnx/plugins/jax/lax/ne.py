from typing import TYPE_CHECKING

import jax
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.ne_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.ne.html",
    onnx=[
        {
            "component": "Equal",
            "doc": "https://onnx.ai/onnx/operators/onnx__Equal.html",
        },
        {
            "component": "Not",
            "doc": "https://onnx.ai/onnx/operators/onnx__Not.html",
        },
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="ne",
    testcases=[
        {
            "testcase": "ne",
            "callable": lambda x1, x2: x1 != x2,
            "input_shapes": [(3,), (3,)],
        }
    ],
)
class NePlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.ne to ONNX Equal and Not."""

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX ne primitive."""
        input_names = [s.get_name(inp) for inp in node_inputs]
        eq_output = s.get_unique_name("equal_output")
        output_name = s.get_var_name(node_outputs[0])

        # Add value info for eq_output
        s.add_shape_info(eq_output, shape=node_inputs[0].aval.shape, dtype="bool")

        node_1 = helper.make_node(
            "Equal",
            inputs=input_names,
            outputs=[eq_output],
            name=s.get_unique_name("ne_eq"),
        )
        s.add_node(node_1)

        node_2 = helper.make_node(
            "Not",
            inputs=[eq_output],
            outputs=[output_name],
            name=s.get_unique_name("ne_not"),
        )
        s.add_node(node_2)
