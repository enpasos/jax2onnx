import jax
from typing import TYPE_CHECKING
from onnx import helper

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def get_primitive():
    return jax.lax.squeeze_p


def get_handler(s: "Jaxpr2OnnxConverter"):
    def _handle_squeeze(node_inputs, node_outputs, params):
        """Handle JAX squeeze primitive."""
        input_name = s.get_name(node_inputs[0])
        output_name = s.get_var_name(node_outputs[0])
        dims = params["dimensions"]

        # Use the new add_initializer method
        axes_name = s.get_unique_name("squeeze_axes")
        s.add_initializer(name=axes_name, vals=dims)

        node = helper.make_node(
            "Squeeze",
            inputs=[input_name, axes_name],
            outputs=[output_name],
            name=s.get_unique_name("squeeze"),
        )
        s.add_node(node)

    return _handle_squeeze


def get_metadata() -> dict:
    """Return metadata describing this plugin and its test cases."""
    return {
        "jaxpr_primitive": "squeeze",
        "jax_doc": "https://docs.jax.dev/en/latest/_autosummary/jax.lax.squeeze.html",
        "onnx": [
            {
                "component": "Squeeze",
                "doc": "https://onnx.ai/onnx/operators/onnx__Squeeze.html",
            }
        ],
        "since": "v0.2.0",
        "context": "plugins.lax",
        "testcases": [
            {
                "testcase": "squeeze",
                "callable": lambda x: jax.lax.squeeze(x, (0,)),
                "input_shapes": [(1, 3)],
            }
        ],
    }
