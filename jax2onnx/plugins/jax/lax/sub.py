import jax
from typing import TYPE_CHECKING
from onnx import helper
from jax2onnx.plugin_system import register_plugin, PrimitivePlugin

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


@register_plugin(
    jaxpr_primitive=jax.lax.sub_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.sub.html",
    onnx=[
        {
            "component": "Sub",
            "doc": "https://onnx.ai/onnx/operators/onnx__Sub.html",
        }
    ],
    since="v0.1.0",
    context="plugins.lax",
    testcases=[
        {
            "testcase": "sub_test1",
            "callable": lambda x1, x2: x1 - x2,
            "input_shapes": [(3,), (3,)],
        },
        {
            "testcase": "sub_test2",
            "callable": lambda x1, x2: jax.lax.sub(x1, x2),
            "input_shapes": [(2, 2), (2, 2)],
        },
    ],
)
class SubPlugin(PrimitivePlugin):
    """Plugin for converting jax.lax.sub to ONNX Sub."""

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX sub primitive."""
        input_names = [s.get_name(inp) for inp in node_inputs]
        output_name = s.get_var_name(node_outputs[0])
        node = helper.make_node(
            "Sub",
            inputs=input_names,
            outputs=[output_name],
            name=s.get_unique_name("sub"),
        )
        s.add_node(node)
