from jax import core
from jax.extend.core import Primitive
from flax import nnx
from onnx import helper
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

# Define the Sigmoid primitive
nnx.sigmoid_p = Primitive("nnx.sigmoid")


def get_primitive():
    """Returns the nnx.sigmoid primitive."""
    return nnx.sigmoid_p


def sigmoid_abstract_eval(x):
    """Abstract evaluation function for Sigmoid."""
    return core.ShapedArray(x.shape, x.dtype)


# Register abstract evaluation function
nnx.sigmoid_p.def_abstract_eval(sigmoid_abstract_eval)


def sigmoid(x):
    """Defines the primitive binding for Sigmoid."""
    return nnx.sigmoid_p.bind(x)


def patch_info():
    """Provides patching information for Sigmoid."""
    return {
        "patch_targets": [nnx],
        "patch_function": lambda _: sigmoid,
        "target_attribute": "sigmoid",
    }


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
