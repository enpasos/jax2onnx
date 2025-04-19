from typing import TYPE_CHECKING

from flax import nnx
from jax import core
from jax.extend.core import Primitive
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Define the Sigmoid primitive
nnx.sigmoid_p = Primitive("nnx.sigmoid")
nnx.sigmoid_p.multiple_results = False  # Correct initialization


@register_primitive(
    jaxpr_primitive=nnx.sigmoid_p.name,
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/activations.html#flax.nnx.sigmoid",
    onnx=[
        {
            "component": "Sigmoid",
            "doc": "https://onnx.ai/onnx/operators/onnx__Sigmoid.html",
        }
    ],
    since="v0.1.0",
    context="primitives.nnx",
    component="sigmoid",
    testcases=[
        {
            "testcase": "sigmoid",
            "callable": lambda x: nnx.sigmoid(x),
            "input_shapes": [(3,)],
        }
    ],
)
class SigmoidPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting flax.nnx.sigmoid to ONNX.
    """

    @staticmethod
    def abstract_eval(x):
        """Abstract evaluation function for Sigmoid."""
        return core.ShapedArray(x.shape, x.dtype)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handles conversion of Sigmoid to ONNX format."""
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

    @staticmethod
    def get_monkey_patch():
        """Provides patching information for Sigmoid."""

        def patched_sigmoid(x):
            return nnx.sigmoid_p.bind(x)

        return patched_sigmoid

    @staticmethod
    def patch_info():
        """Provides patching information for Sigmoid."""
        return {
            "patch_targets": [nnx],
            "patch_function": lambda _: SigmoidPlugin.get_monkey_patch(),
            "target_attribute": "sigmoid",
        }


# Register abstract evaluation function
nnx.sigmoid_p.def_abstract_eval(SigmoidPlugin.abstract_eval)
