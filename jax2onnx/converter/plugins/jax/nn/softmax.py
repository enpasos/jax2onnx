from jax import core
from jax.extend.core import Primitive
import jax.nn as nn
from onnx import helper
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

# Define a new primitive for softmax
nn.softmax_p = Primitive("nn.softmax")


def get_primitive():
    """Returns the nn.softmax primitive."""
    return nn.softmax_p


def softmax_abstract_eval(x, axis=-1):
    """Computes the output shape for nn.softmax."""
    return core.ShapedArray(x.shape, x.dtype)


# Register abstract evaluation function
nn.softmax_p.def_abstract_eval(softmax_abstract_eval)


def softmax(x, axis=-1):
    """Defines the primitive binding for Softmax."""
    return nn.softmax_p.bind(x, axis=axis)


def patch_info():
    """Provides patching information for Softmax."""
    return {
        "patch_targets": [nn],
        "patch_function": lambda _: softmax,
        "target_attribute": "softmax",
    }


def get_handler(s: "Jaxpr2OnnxConverter"):
    def handle_softmax(node_inputs, node_outputs, params):
        """Handles ONNX conversion for nn.softmax."""
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
    """Returns metadata describing this plugin and its test cases."""
    return {
        "jaxpr_primitive": "nn.softmax",
        "jax_doc": "https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.softmax.html",
        "onnx": [
            {
                "component": "Softmax",
                "doc": "https://onnx.ai/onnx/operators/onnx__Softmax.html",
            }
        ],
        "since": "v0.1.0",
        "context": "plugins.nn",
        "testcases": [
            {
                "testcase": "softmax",
                "callable": lambda x: nn.softmax(x),
                "input_shapes": [(3,)],
            },
            {
                "testcase": "softmax_2d",
                "callable": lambda x: nn.softmax(x, axis=1),
                "input_shapes": [(4, 5)],
            },
            {
                "testcase": "softmax_3d",
                "callable": lambda x: nn.softmax(x, axis=2),
                "input_shapes": [(2, 3, 4)],
            },
        ],
    }
