import jax
from typing import TYPE_CHECKING, List, Dict, Any
from onnx import helper

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def get_primitive():
    return jax.lax.gather_p


def get_handler(s: "Jaxpr2OnnxConverter"):
    def _handle_gather(node_inputs, node_outputs, params):
        """Handle JAX gather primitive."""
        input_names = [s.get_name(inp) for inp in node_inputs]
        output_name = s.get_var_name(node_outputs[0])
        node = helper.make_node(
            "GatherElements",
            inputs=input_names,
            outputs=[output_name],
            name=s.get_unique_name("gather"),
        )
        s.add_node(node)

    return _handle_gather


def get_metadata() -> dict:
    """Return metadata describing this plugin and its test cases."""
    return {
        "jaxpr_primitive": "gather",
        "jax_doc": "https://docs.jax.dev/en/latest/_autosummary/jax.lax.gather.html",
        "onnx": [
            {
                "component": "Gather",
                "doc": "https://onnx.ai/onnx/operators/onnx__Gather.html",
            }
        ],
        "since": "v0.2.0",
        "context": "plugins.lax",
        "testcases": [
            {
                "testcase": "gather",
                "callable": lambda x, indices: jax.lax.gather(
                    x, indices, dimension_numbers=(((0,), (0,)), ())
                ),
                "input_shapes": [(3, 3), (2,)],
            }
        ],
    }
