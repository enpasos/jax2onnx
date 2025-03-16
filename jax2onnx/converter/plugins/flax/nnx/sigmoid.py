# file: jax2onnx/converter/primitives/flax/nnx/linear_general.py

from jax import core
from jax.extend.core import Primitive
from flax import nnx
from onnx import helper
import contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


nnx.sigmoid_p = Primitive("nnx.sigmoid")


def get_primitive():
    return nnx.sigmoid_p


def _get_monkey_patch():
    def sigmoid(x):
        def sigmoid_abstract_eval(x):
            return core.ShapedArray(x.shape, x.dtype)

        nnx.sigmoid_p.multiple_results = False
        nnx.sigmoid_p.def_abstract_eval(sigmoid_abstract_eval)
        return nnx.sigmoid_p.bind(x)

    return sigmoid


@contextlib.contextmanager
def temporary_patch():
    # Save the original function
    original_fn = nnx.sigmoid
    # Patch the function by replacing it in the module namespace.
    nnx.sigmoid = _get_monkey_patch()
    try:
        yield
    finally:
        # Restore the original function
        nnx.sigmoid = original_fn


def get_handler(s: "Jaxpr2OnnxConverter"):
    def handle_sigmoid(node_inputs, node_outputs, params):
        input_var = node_inputs[0]
        output_var = node_outputs[0]

        input_name = s.get_name(input_var)
        output_name = s.get_name(output_var)

        sigmoid_node = helper.make_node(
            "Sigmoid",
            inputs=[input_name],
            outputs=[output_name],
            name=s.get_unique_name("sigmoid"),
        )
        s.add_node(sigmoid_node)

    return handle_sigmoid


def get_metadata() -> dict:
    """Return metadata describing this plugin and its test cases."""
    return {
        "jaxpr_primitive": "nnx.sigmoid",
        "jax_doc": "https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/activations.html#flax.nnx.sigmoid",
        "onnx": [
            {
                "component": "Sigmoid",
                "doc": "https://onnx.ai/onnx/operators/onnx__Sigmoid.html",
            }
        ],
        "since": "v0.1.0",
        "context": "plugins.nnx",
        "testcases": [
            {
                "testcase": "sigmoid",
                "callable": lambda x: nnx.sigmoid(x),
                "input_shapes": [(3,)],
            }
        ],
    }
