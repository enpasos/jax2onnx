from jax import core
from jax.extend.core import Primitive
from flax import nnx
from onnx import helper
import contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

nnx.softplus_p = Primitive("nnx.softplus")


def get_primitive():
    return nnx.softplus_p


def _get_monkey_patch():
    def softplus(x):
        def softplus_abstract_eval(x):
            # The output shape and dtype are the same as the input.
            return core.ShapedArray(x.shape, x.dtype)

        nnx.softplus_p.multiple_results = False
        nnx.softplus_p.def_abstract_eval(softplus_abstract_eval)
        return nnx.softplus_p.bind(x)

    return softplus


@contextlib.contextmanager
def temporary_patch():
    # Save the original function.
    original_fn = nnx.softplus
    # Replace the function in the module namespace with our patched version.
    nnx.softplus = _get_monkey_patch()
    try:
        yield
    finally:
        # Restore the original function.
        nnx.softplus = original_fn


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
