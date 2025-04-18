from typing import TYPE_CHECKING

import jax
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.select_n_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.select_n.html",
    onnx=[
        {
            "component": "Where",
            "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="select_n",
    testcases=[
        # {
        #     "testcase": "select_n",
        #     "callable": lambda pred, x, y: jax.lax.select_n(pred, x, y),
        #     "input_shapes": [(3,), (3,), (3,)],
        #     "test_data": [(jnp.array([True, False, True]), jnp.array([1, 2, 3]), jnp.array([4, 5, 6]))]
        # },
        # {
        #     "testcase": "select_n_float",
        #     "callable": lambda pred, x, y: jax.lax.select_n(pred, x, y),
        #     "input_shapes": [(2, 2), (2, 2), (2, 2)],
        #     "test_data": [(jnp.array([[True, False], [False, True]]), jnp.array([[1.0, 2.0], [3.0, 4.0]]), jnp.array([[5.0, 6.0], [7.0, 8.0]]))]
        # },
        # {
        #     "testcase": "select_n_int",
        #     "callable": lambda pred, x, y: jax.lax.select_n(pred, x, y),
        #     "input_shapes": [(1, 4), (1, 4), (1, 4)],
        #     "test_data": [(jnp.array([[True, True, False, False]]), jnp.array([[1, 2, 3, 4]]), jnp.array([[5, 6, 7, 8]]))]
        # }
    ],
)
class SelectNPlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.select_n to ONNX Where."""

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX select_n primitive by mapping it to ONNX's Where operator."""
        condition_name = s.get_name(node_inputs[0])
        false_name = s.get_name(node_inputs[1])  # JAX 'x' maps to ONNX 'Y'
        true_name = s.get_name(node_inputs[2])  # JAX 'y' maps to ONNX 'X'
        output_name = s.get_var_name(node_outputs[0])

        node = helper.make_node(
            "Where",
            inputs=[condition_name, true_name, false_name],  # ONNX: condition, X, Y
            outputs=[output_name],
            name=s.get_unique_name("where"),
        )
        s.add_node(node)
