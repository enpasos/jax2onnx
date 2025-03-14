import jax
from typing import TYPE_CHECKING
from onnx import helper

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def get_primitive():
    return jax.lax.concatenate_p


def get_handler(s: "Jaxpr2OnnxConverter"):
    def _handle_concatenate(node_inputs, node_outputs, params):
        """Handle JAX concatenate primitive."""
        input_names = [s.get_name(inp) for inp in node_inputs]
        output_name = s.get_var_name(node_outputs[0])
        dimension = params["dimension"]
        node = helper.make_node(
            "Concat",
            inputs=input_names,
            outputs=[output_name],
            name=s.get_unique_name("concat"),
            axis=dimension,
        )
        s.add_node(node)

    return _handle_concatenate


def get_metadata() -> dict:
    """Return metadata describing this plugin and its test cases."""
    return {
        "jaxpr_primitive": "concatenate",
        "jax_doc": "https://docs.jax.dev/en/latest/_autosummary/jax.lax.concatenate.html",
        "onnx": [
            {
                "component": "Concat",
                "doc": "https://onnx.ai/onnx/operators/onnx__Concat.html",
            }
        ],
        "since": "v0.2.0",
        "context": "plugins.lax",
        "testcases": [
            {
                "testcase": "concatenate",
                "callable": lambda *args: jax.lax.concatenate(args, dimension=0),
                "input_shapes": [(3,), (3,)],
            }
        ],
    }
