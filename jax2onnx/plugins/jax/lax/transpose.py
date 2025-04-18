from typing import TYPE_CHECKING

import jax
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.transpose_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.transpose.html",
    onnx=[
        {
            "component": "Transpose",
            "doc": "https://onnx.ai/onnx/operators/onnx__Transpose.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="transpose",
    testcases=[
        {
            "testcase": "transpose_basic",
            "callable": lambda x: jax.lax.transpose(x, (1, 0)),
            "input_shapes": [(2, 3)],
        }
    ],
)
class TransposePlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.transpose to ONNX Transpose."""

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX transpose primitive."""
        input_name = s.get_name(node_inputs[0])
        output_name = s.get_var_name(node_outputs[0])
        permutation = params["permutation"]
        node = helper.make_node(
            "Transpose",
            inputs=[input_name],
            outputs=[output_name],
            name=s.get_unique_name("transpose"),
            perm=permutation,
        )
        s.add_node(node)
