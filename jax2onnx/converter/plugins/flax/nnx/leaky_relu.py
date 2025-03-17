from jax import core
from jax.extend.core import Primitive
from flax import nnx
from onnx import helper
import contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

nnx.leaky_relu_p = Primitive("nnx.leaky_relu")


def get_primitive():
    return nnx.leaky_relu_p


def _get_monkey_patch():
    def leaky_relu(x, negative_slope=0.01):
        def leaky_relu_abstract_eval(x, negative_slope=0.01):
            # The output shape and dtype remain the same as the input.
            return core.ShapedArray(x.shape, x.dtype)

        nnx.leaky_relu_p.multiple_results = False
        nnx.leaky_relu_p.def_abstract_eval(leaky_relu_abstract_eval)
        return nnx.leaky_relu_p.bind(x, negative_slope=negative_slope)

    return leaky_relu


@contextlib.contextmanager
def temporary_patch():
    # Save the original function.
    original_fn = nnx.leaky_relu
    # Patch the function by replacing it in the module namespace.
    nnx.leaky_relu = _get_monkey_patch()
    try:
        yield
    finally:
        # Restore the original function.
        nnx.leaky_relu = original_fn


def get_handler(s: "Jaxpr2OnnxConverter"):
    def handle_leaky_relu(node_inputs, node_outputs, params):
        input_var = node_inputs[0]
        output_var = node_outputs[0]

        input_name = s.get_name(input_var)
        output_name = s.get_name(output_var)

        # Retrieve the negative_slope parameter (defaulting to 0.01 if not provided)
        negative_slope = params.get("negative_slope", 0.01)

        leaky_relu_node = helper.make_node(
            "LeakyRelu",
            inputs=[input_name],
            outputs=[output_name],
            name=s.get_unique_name("leaky_relu"),
            alpha=negative_slope,
        )
        s.add_node(leaky_relu_node)

    return handle_leaky_relu


def get_metadata() -> dict:
    """Return metadata describing this plugin and its test cases."""
    return {
        "jaxpr_primitive": "nnx.leaky_relu",
        "jax_doc": "https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.leaky_relu.html",
        "onnx": [
            {
                "component": "LeakyRelu",
                "doc": "https://onnx.ai/onnx/operators/onnx__LeakyRelu.html",
            }
        ],
        "since": "v0.1.0",
        "context": "plugins.nnx",
        "testcases": [
            {
                "testcase": "leaky_relu",
                "callable": lambda x: nnx.leaky_relu(x),
                "input_shapes": [(3,)],
            }
        ],
    }
