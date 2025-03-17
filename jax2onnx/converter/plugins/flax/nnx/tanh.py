# file: jax2onnx/converter/primitives/flax/nnx/linear_general.py

from jax import core
from jax.extend.core import Primitive
from flax import nnx
from onnx import helper
import contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


nnx.tanh_p = Primitive("nnx.tanh")


def get_primitive():
    return nnx.tanh_p


def _get_monkey_patch():
    def tanh(x):
        def tanh_abstract_eval(x):
            return core.ShapedArray(x.shape, x.dtype)

        nnx.tanh_p.multiple_results = False
        nnx.tanh_p.def_abstract_eval(tanh_abstract_eval)
        return nnx.tanh_p.bind(x)

    return tanh


@contextlib.contextmanager
def temporary_patch():
    # Save the original function
    original_fn = nnx.tanh
    # Patch the function by replacing it in the module namespace.
    nnx.tanh = _get_monkey_patch()
    try:
        yield
    finally:
        # Restore the original function
        nnx.tanh = original_fn


def get_handler(s: "Jaxpr2OnnxConverter"):
    def handle_tanh(node_inputs, node_outputs, params):
        input_var = node_inputs[0]
        output_var = node_outputs[0]

        input_name = s.get_name(input_var)
        output_name = s.get_name(output_var)

        tanh_node = helper.make_node(
            "Tanh",
            inputs=[input_name],
            outputs=[output_name],
            name=s.get_unique_name("tanh"),
        )
        s.add_node(tanh_node)

    return handle_tanh


def get_metadata() -> dict:
    """Return metadata describing this plugin and its test cases."""
    return {
        "jaxpr_primitive": "nnx.tanh",
        "jax_doc": "https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/activations.html#flax.nnx.tanh",
        "onnx": [
            {
                "component": "Tanh",
                "doc": "https://onnx.ai/onnx/operators/onnx__Tanh.html",
            }
        ],
        "since": "v0.1.0",
        "context": "plugins.nnx",
        "testcases": [
            {
                "testcase": "tanh",
                "callable": lambda x: nnx.tanh(x),
                "input_shapes": [(3,)],
            }
        ],
    }
