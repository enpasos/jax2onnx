from typing import TYPE_CHECKING

from flax import nnx
from jax.extend.core import Primitive
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Define the Softmax primitive
nnx.softmax_p = Primitive("nnx.softmax")
nnx.softmax_p.multiple_results = False  # Correct initialization


@register_primitive(
    jaxpr_primitive=nnx.softmax_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.softmax.html",
    onnx=[
        {
            "component": "Softmax",
            "doc": "https://onnx.ai/onnx/operators/onnx__Softmax.html",
        }
    ],
    since="v0.1.0",
    context="primitives.nnx",
    component="softmax",
    testcases=[
        {
            "testcase": "softmax",
            "callable": lambda x: nnx.softmax(x),
            "input_shapes": [("B", 2)],
        }
    ],
)
class SoftmaxPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting flax.nnx.softmax to ONNX.
    """

    @staticmethod
    def abstract_eval(x, axis=-1):
        """Abstract evaluation function for Softmax."""
        # Use update instead of creating a new ShapedArray to avoid issues with unhashable tracers
        return x.update(shape=x.shape, dtype=x.dtype, weak_type=False)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handles conversion of Softmax to ONNX format."""
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

    @staticmethod
    def get_monkey_patch():
        """Provides patching information for Softmax."""

        def patched_softmax(x, axis=-1):
            return nnx.softmax_p.bind(x, axis=axis)

        return patched_softmax

    @staticmethod
    def patch_info():
        """Provides patching information for Softmax."""
        return {
            "patch_targets": [nnx],
            "patch_function": lambda _: SoftmaxPlugin.get_monkey_patch(),
            "target_attribute": "softmax",
        }


# Register abstract evaluation function
nnx.softmax_p.def_abstract_eval(SoftmaxPlugin.abstract_eval)
