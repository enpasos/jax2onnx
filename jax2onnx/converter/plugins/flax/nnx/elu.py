from jax import core
from jax.extend.core import Primitive
from flax import nnx
from onnx import helper
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

# Define the ELU primitive
nnx.elu_p = Primitive("nnx.elu")
nnx.elu_p.multiple_results = False  # ✅ Set at initialization


def get_primitive():
    """Returns the nnx.elu primitive."""
    return nnx.elu_p


def elu_abstract_eval(x, alpha=1.0):
    """Abstract evaluation function for ELU."""
    return core.ShapedArray(x.shape, x.dtype)


# Register abstract evaluation function
nnx.elu_p.def_abstract_eval(elu_abstract_eval)


def elu(x, alpha=1.0):
    """Defines the primitive binding for ELU."""
    return nnx.elu_p.bind(x, alpha=alpha)


def patch_info():
    """Provides patching information for ELU."""
    return {
        "patch_targets": [nnx],
        "patch_function": lambda _: elu,
        "target_attribute": "elu",
    }


def get_handler(s: "Jaxpr2OnnxConverter"):
    """Handles conversion of ELU to ONNX format."""

    def handle_elu(node_inputs, node_outputs, params):
        input_var = node_inputs[0]
        output_var = node_outputs[0]

        input_name = s.get_name(input_var)
        output_name = s.get_name(output_var)

        # Retrieve the alpha parameter (defaulting to 1.0 if not provided)
        alpha = params.get("alpha", 1.0)

        elu_node = helper.make_node(
            "Elu",
            inputs=[input_name],
            outputs=[output_name],
            name=s.get_unique_name("elu"),
            alpha=alpha,
        )
        s.add_node(elu_node)

    return handle_elu


def get_metadata() -> dict:
    """Returns metadata describing this plugin and its test cases."""
    return {
        "jaxpr_primitive": "nnx.elu",
        "jax_doc": "https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.elu.html",
        "onnx": [
            {
                "component": "Elu",
                "doc": "https://onnx.ai/onnx/operators/onnx__Elu.html",
            }
        ],
        "since": "v0.1.0",
        "context": "plugins.nnx",
        "testcases": [
            {
                "testcase": "elu",
                "callable": lambda x: nnx.elu(x),
                "input_shapes": [(3,)],
            }
        ],
    }
