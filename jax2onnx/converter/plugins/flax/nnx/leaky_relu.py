import contextlib
from typing import TYPE_CHECKING

from flax import nnx
from jax import core
from jax.extend.core import Primitive
from onnx import helper

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

# Define the LeakyReLU primitive
nnx.leaky_relu_p = Primitive("nnx.leaky_relu")
nnx.leaky_relu_p.multiple_results = False  # ✅ Set at initialization


def get_primitive():
    """Returns the nnx.leaky_relu primitive."""
    return nnx.leaky_relu_p


def leaky_relu_abstract_eval(x, negative_slope=0.01):
    """Abstract evaluation function for LeakyReLU."""
    return core.ShapedArray(x.shape, x.dtype)


# Register abstract evaluation function
nnx.leaky_relu_p.def_abstract_eval(leaky_relu_abstract_eval)


def leaky_relu(x, negative_slope=0.01):
    """Defines the primitive binding for LeakyReLU."""
    return nnx.leaky_relu_p.bind(x, negative_slope=negative_slope)


def patch_info():
    """Provides patching information for LeakyReLU."""
    return {
        "patch_targets": [nnx],
        "patch_function": lambda _: leaky_relu,
        "target_attribute": "leaky_relu",
    }


def get_handler(s: "Jaxpr2OnnxConverter"):
    """Handles conversion of LeakyReLU to ONNX format."""

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
    """Returns metadata describing this plugin and its test cases."""
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
