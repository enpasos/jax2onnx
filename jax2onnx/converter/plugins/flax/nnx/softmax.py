from jax import core
from jax.extend.core import Primitive
from flax import nnx
from onnx import helper
import contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

nnx.softmax_p = Primitive("nnx.softmax")


def get_primitive():
    return nnx.softmax_p


def _get_monkey_patch():
    def softmax(x, axis=-1):
        def softmax_abstract_eval(x, axis=-1):
            # The output shape is the same as the input.
            return core.ShapedArray(x.shape, x.dtype)

        nnx.softmax_p.multiple_results = False
        nnx.softmax_p.def_abstract_eval(softmax_abstract_eval)
        return nnx.softmax_p.bind(x, axis=axis)

    return softmax


@contextlib.contextmanager
def temporary_patch():
    # Save the original function
    original_fn = nnx.softmax
    # Patch the function by replacing it in the module namespace.
    nnx.softmax = _get_monkey_patch()
    try:
        yield
    finally:
        # Restore the original function
        nnx.softmax = original_fn


def get_handler(s: "Jaxpr2OnnxConverter"):
    def handle_softmax(node_inputs, node_outputs, params):
        input_var = node_inputs[0]
        output_var = node_outputs[0]

        input_name = s.get_name(input_var)
        output_name = s.get_name(output_var)

        # Retrieve the axis parameter (defaulting to -1 if not provided)
        axis = params.get("axis", -1)

        softmax_node = helper.make_node(
            "Softmax",
            inputs=[input_name],
            outputs=[output_name],
            name=s.get_unique_name("softmax"),
            axis=axis,
        )
        s.add_node(softmax_node)

    return handle_softmax


def get_metadata() -> dict:
    """Return metadata describing this plugin and its test cases."""
    return {
        "jaxpr_primitive": "nnx.softmax",
        "jax_doc": "https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.softmax.html",
        "onnx": [
            {
                "component": "Softmax",
                "doc": "https://onnx.ai/onnx/operators/onnx__Softmax.html",
            }
        ],
        "since": "v0.1.0",
        "context": "plugins.nnx",
        "testcases": [
            {
                "testcase": "softmax",
                "callable": lambda x: nnx.softmax(x),
                "input_shapes": [(3,)],
            }
        ],
    }
