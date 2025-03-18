from jax import core
from jax.extend.core import Primitive
from flax import nnx
from onnx import helper
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

# Define the Softplus primitive
nnx.softplus_p = Primitive("nnx.softplus")


def get_primitive():
    """Returns the nnx.softplus primitive."""
    return nnx.softplus_p


def softplus_abstract_eval(x):
    """Abstract evaluation function for Softplus."""
    return core.ShapedArray(x.shape, x.dtype)


# Register abstract evaluation function
nnx.softplus_p.def_abstract_eval(softplus_abstract_eval)


def softplus(x):
    """Defines the primitive binding for Softplus."""
    return nnx.softplus_p.bind(x)


def patch_info():
    """Provides patching information for Softplus."""
    return {
        "patch_targets": [nnx],
        "patch_function": lambda _: softplus,
        "target_attribute": "softplus",
    }


def get_handler(s: "Jaxpr2OnnxConverter"):
    def handle_softplus(node_inputs, node_outputs, params):
        input_var = node_inputs[0]
        output_var = node_outputs[0]

        input_name = s.get_name(input_var)
        output_name = s.get_name(output_var)

        softplus_node = helper.make_node(
            "Softplus",
            inputs=[input_name],
            outputs=[output_name],
            name=s.get_unique_name("softplus"),
        )
        s.add_node(softplus_node)

    return handle_softplus


def get_metadata() -> dict:
    """Return metadata describing this plugin and its test cases."""
    return {
        "jaxpr_primitive": "nnx.softplus",
        "jax_doc": "https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.softplus.html",
        "onnx": [
            {
                "component": "Softplus",
                "doc": "https://onnx.ai/onnx/operators/onnx__Softplus.html",
            }
        ],
        "since": "v0.1.0",
        "context": "plugins.nnx",
        "testcases": [
            {
                "testcase": "softplus",
                "callable": lambda x: nnx.softplus(x),
                "input_shapes": [(3,)],
            }
        ],
    }
