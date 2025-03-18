from jax import core
from jax.extend.core import Primitive
from flax import nnx
from onnx import helper
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

# Define the LogSoftmax primitive
nnx.log_softmax_p = Primitive("nnx.log_softmax")


def get_primitive():
    """Returns the nnx.log_softmax primitive."""
    return nnx.log_softmax_p


def log_softmax_abstract_eval(x, axis=-1):
    """Abstract evaluation function for LogSoftmax."""
    return core.ShapedArray(x.shape, x.dtype)


# Register abstract evaluation function
nnx.log_softmax_p.def_abstract_eval(log_softmax_abstract_eval)


def log_softmax(x, axis=-1):
    """Defines the primitive binding for LogSoftmax."""
    return nnx.log_softmax_p.bind(x, axis=axis)


def patch_info():
    """Provides patching information for LogSoftmax."""
    return {
        "patch_targets": [nnx],
        "patch_function": lambda _: log_softmax,
        "target_attribute": "log_softmax",
    }


def _get_monkey_patch():
    """Returns a patched version of LogSoftmax's call method."""

    def patched_log_softmax_call(self, x):
        return log_softmax(x, axis=self.axis if hasattr(self, "axis") else -1)

    return patched_log_softmax_call


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
