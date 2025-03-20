import jax
from typing import TYPE_CHECKING
from onnx import helper
from jax2onnx.plugin_system import register_primitive, PrimitivePlugin

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.squeeze_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.squeeze.html",
    onnx=[
        {
            "component": "Squeeze",
            "doc": "https://onnx.ai/onnx/operators/onnx__Squeeze.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    testcases=[
        {
            "testcase": "squeeze",
            "callable": lambda x: jax.lax.squeeze(x, (0,)),
            "input_shapes": [(1, 3)],
        }
    ],
)
class SqueezePlugin(PrimitivePlugin):
    """Plugin for converting jax.lax.squeeze to ONNX Squeeze."""

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX squeeze primitive."""
        input_name = s.get_name(node_inputs[0])
        output_name = s.get_var_name(node_outputs[0])
        dims = params["dimensions"]

        axes_name = s.get_unique_name("squeeze_axes")
        s.add_initializer(name=axes_name, vals=dims)

        node = helper.make_node(
            "Squeeze",
            inputs=[input_name, axes_name],
            outputs=[output_name],
            name=s.get_unique_name("squeeze"),
        )
        s.add_node(node)
