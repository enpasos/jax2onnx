from jax import core
from jax.extend.core import Primitive
from flax import nnx
from onnx import helper
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

# Define the ReLU primitive
nnx.relu_p = Primitive("nnx.relu")


def get_primitive():
    """Returns the nnx.relu primitive."""
    return nnx.relu_p


def relu_abstract_eval(x):
    """Abstract evaluation function for ReLU."""
    return core.ShapedArray(x.shape, x.dtype)


# Register abstract evaluation function
nnx.relu_p.def_abstract_eval(relu_abstract_eval)


def relu(x):
    """Defines the primitive binding for ReLU."""
    return nnx.relu_p.bind(x)


def patch_info():
    """Provides patching information for ReLU."""
    return {
        "patch_targets": [nnx],
        "patch_function": lambda _: relu,
        "target_attribute": "relu",
    }


def get_handler(s: "Jaxpr2OnnxConverter"):
    def handle_relu(node_inputs, node_outputs, params):
        input_var = node_inputs[0]
        output_var = node_outputs[0]

        input_name = s.get_name(input_var)
        output_name = s.get_name(output_var)

        relu_node = helper.make_node(
            "Relu",
            inputs=[input_name],
            outputs=[output_name],
            name=s.get_unique_name("relu"),
        )
        s.add_node(relu_node)

    return handle_relu


def get_metadata() -> dict:
    """Return metadata describing this plugin and its test cases."""
    return {
        "jaxpr_primitive": "nnx.relu",
        "jax_doc": "https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.relu.html",
        "onnx": [
            {
                "component": "Relu",
                "doc": "https://onnx.ai/onnx/operators/onnx__Relu.html",
            }
        ],
        "since": "v0.1.0",
        "context": "plugins.nnx",
        "testcases": [
            {
                "testcase": "relu",
                "callable": lambda x: nnx.relu(x),
                "input_shapes": [(3,)],
            },
            {
                "testcase": "relu_2",
                "callable": lambda x: nnx.relu(x),
                "input_shapes": [(2, 28, 28, 32)],
            },
        ],
    }
