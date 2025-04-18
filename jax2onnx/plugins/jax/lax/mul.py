from typing import TYPE_CHECKING

import jax
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.mul_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.mul.html",
    onnx=[
        {
            "component": "Mul",
            "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html",
        }
    ],
    since="v0.1.0",
    context="primitives.lax",
    component="mul",
    testcases=[
        {
            "testcase": "mul_test1",
            "callable": lambda x1, x2: x1 * x2,
            "input_shapes": [(3,), (3,)],
        },
        {
            "testcase": "mul_test2",
            "callable": lambda x1, x2: x1 * x2,
            "input_shapes": [(2, 2), (2, 2)],
        },
    ],
)
class MulPlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.mul to ONNX Mul."""

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX mul primitive."""
        input_names = [s.get_name(inp) for inp in node_inputs]
        output_name = s.get_var_name(node_outputs[0])
        node = helper.make_node(
            "Mul",
            inputs=input_names,
            outputs=[output_name],
            name=s.get_unique_name("mul"),
        )
        s.add_node(node)
