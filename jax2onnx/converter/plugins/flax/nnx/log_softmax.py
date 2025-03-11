from jax import core
from jax.extend.core import Primitive
from flax import nnx
from onnx import helper
import contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

nnx.log_softmax_p = Primitive("nnx.log_softmax")


def get_primitive():
    return nnx.log_softmax_p


def _get_monkey_patch():
    def log_softmax(x, axis=-1):
        def log_softmax_abstract_eval(x, axis=-1):
            # Output shape and type remain the same as input.
            return core.ShapedArray(x.shape, x.dtype)

        nnx.log_softmax_p.multiple_results = False
        nnx.log_softmax_p.def_abstract_eval(log_softmax_abstract_eval)
        return nnx.log_softmax_p.bind(x, axis=axis)

    return log_softmax


@contextlib.contextmanager
def temporary_patch():
    # Save the original function.
    original_fn = nnx.log_softmax
    # Replace the function in the module namespace with our patched version.
    nnx.log_softmax = _get_monkey_patch()
    try:
        yield
    finally:
        # Restore the original function.
        nnx.log_softmax = original_fn


def get_handler(s: "Jaxpr2OnnxConverter"):
    def handle_log_softmax(node_inputs, node_outputs, params):
        input_var = node_inputs[0]
        output_var = node_outputs[0]

        input_name = s.get_name(input_var)
        output_name = s.get_name(output_var)

        # Retrieve the axis parameter (defaulting to -1 if not provided)
        axis = params.get("axis", -1)

        log_softmax_node = helper.make_node(
            "LogSoftmax",
            inputs=[input_name],
            outputs=[output_name],
            name=s.get_unique_name("log_softmax"),
            axis=axis,
        )
        s.add_node(log_softmax_node)

    return handle_log_softmax


def get_metadata() -> dict:
    """Return metadata describing this plugin and its test cases."""
    return {
        "jaxpr_primitive": "nnx.log_softmax",
        "jax_doc": "https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.log_softmax.html",
        "onnx": [
            {
                "component": "LogSoftmax",
                "doc": "https://onnx.ai/onnx/operators/onnx__LogSoftmax.html",
            }
        ],
        "since": "v0.1.0",
        "context": "plugins.nnx",
        "testcases": [
            {
                "testcase": "log_softmax",
                "callable": lambda x: nnx.log_softmax(x),
                "input_shapes": [(3,)],
            }
        ],
    }
