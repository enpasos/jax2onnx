from jax import core
from jax.extend.core import Primitive
from flax import nnx
from onnx import helper
import contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

nnx_elu_p = Primitive("nnx.elu")


def get_primitive():
    return nnx_elu_p


def _get_monkey_patch():
    def elu(x, alpha=1.0):
        def elu_abstract_eval(x, alpha=1.0):
            # The output shape and type remain the same as the input.
            return core.ShapedArray(x.shape, x.dtype)

        nnx_elu_p.multiple_results = False
        nnx_elu_p.def_abstract_eval(elu_abstract_eval)
        return nnx_elu_p.bind(x, alpha=alpha)

    return elu


@contextlib.contextmanager
def temporary_patch():
    # Save the original function
    original_fn = nnx.elu
    # Patch the function by replacing it in the module namespace.
    nnx.elu = _get_monkey_patch()
    try:
        yield
    finally:
        # Restore the original function
        nnx.elu = original_fn


def get_handler(s: "Jaxpr2OnnxConverter"):
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
    """Return metadata describing this plugin and its test cases."""
    return {
        "jaxpr_primitive": "nnx.elu",
        # "jaxpr_primitive": "nnx.elu",
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
