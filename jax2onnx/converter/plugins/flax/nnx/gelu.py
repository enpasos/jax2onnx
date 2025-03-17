from jax import core
from jax.extend.core import Primitive
from flax import nnx
from onnx import helper
import contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

nnx.gelu_p = Primitive("nnx.gelu")


def get_primitive():
    return nnx.gelu_p


def _get_monkey_patch():
    def gelu(x, approximate=True):
        def gelu_abstract_eval(x, approximate=True):
            # The output shape and dtype remain the same as the input.
            return core.ShapedArray(x.shape, x.dtype)

        nnx.gelu_p.multiple_results = False
        nnx.gelu_p.def_abstract_eval(gelu_abstract_eval)
        return nnx.gelu_p.bind(x, approximate=approximate)

    return gelu


@contextlib.contextmanager
def temporary_patch():
    # Save the original function.
    original_fn = nnx.gelu
    # Patch the function by replacing it in the module namespace.
    nnx.gelu = _get_monkey_patch()
    try:
        yield
    finally:
        # Restore the original function.
        nnx.gelu = original_fn


def get_handler(s: "Jaxpr2OnnxConverter"):
    def handle_gelu(node_inputs, node_outputs, params):
        input_var = node_inputs[0]
        output_var = node_outputs[0]

        input_name = s.get_name(input_var)
        output_name = s.get_name(output_var)

        # Retrieve the approximate parameter (defaulting to True if not provided)
        approximate = params.get("approximate", True)
        # ONNX Gelu operator expects an 'approximation' attribute:
        # "tanh" for approximate GELU, "none" for the exact version.
        approximation = "tanh" if approximate else "none"

        gelu_node = helper.make_node(
            "Gelu",
            inputs=[input_name],
            outputs=[output_name],
            name=s.get_unique_name("gelu"),
            approximate=approximation,
        )
        s.add_node(gelu_node)

    return handle_gelu


def get_metadata() -> dict:
    """Return metadata describing this plugin and its test cases."""
    return {
        "jaxpr_primitive": "nnx.gelu",
        "jax_doc": "https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.gelu.html",
        "onnx": [
            {
                "component": "Gelu",
                "doc": "https://onnx.ai/onnx/operators/onnx__Gelu.html",
            }
        ],
        "since": "v0.1.0",
        "context": "plugins.nnx",
        "testcases": [
            {
                "testcase": "gelu",
                "callable": lambda x: nnx.gelu(x, approximate=False),
                "input_shapes": [(1,)],
            },
            {
                "testcase": "gelu_1",
                "callable": lambda x: nnx.gelu(x, approximate=False),
                "input_shapes": [(1, 10)],
            },
            {
                "testcase": "gelu_2",
                "callable": lambda x: nnx.gelu(x, approximate=True),
                "input_shapes": [(1,)],
            },
            {
                "testcase": "gelu_3",
                "callable": lambda x: nnx.gelu(x, approximate=True),
                "input_shapes": [(1, 10)],
            },
        ],
    }
