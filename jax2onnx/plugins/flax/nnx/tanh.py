from typing import TYPE_CHECKING

from flax import nnx
from jax import core
from jax.extend.core import Primitive
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Define the Tanh primitive
nnx.tanh_p = Primitive("nnx.tanh")
nnx.tanh_p.multiple_results = False  # Correct initialization


@register_primitive(
    jaxpr_primitive=nnx.tanh_p.name,
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/activations.html#flax.nnx.tanh",
    onnx=[
        {
            "component": "Tanh",
            "doc": "https://onnx.ai/onnx/operators/onnx__Tanh.html",
        }
    ],
    since="v0.1.0",
    context="primitives.nnx",
    component="tanh",
    testcases=[
        {
            "testcase": "tanh",
            "callable": lambda x: nnx.tanh(x),
            "input_shapes": [(3,)],
        }
    ],
)
class TanhPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting flax.nnx.tanh to ONNX.
    """

    @staticmethod
    def abstract_eval(x):
        """Abstract evaluation function for Tanh."""
        return core.ShapedArray(x.shape, x.dtype)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handles conversion of Tanh to ONNX format."""
        input_var = node_inputs[0]
        output_var = node_outputs[0]

        input_name = s.get_name(input_var)
        output_name = s.get_name(output_var)

        tanh_node = helper.make_node(
            "Tanh",
            inputs=[input_name],
            outputs=[output_name],
            name=s.get_unique_name("tanh"),
        )
        s.add_node(tanh_node)

    @staticmethod
    def get_monkey_patch():
        """Provides patching information for Tanh."""

        def patched_tanh(x):
            return nnx.tanh_p.bind(x)

        return patched_tanh

    @staticmethod
    def patch_info():
        """Provides patching information for Tanh."""
        return {
            "patch_targets": [nnx],
            "patch_function": lambda _: TanhPlugin.get_monkey_patch(),
            "target_attribute": "tanh",
        }


# Register abstract evaluation function
nnx.tanh_p.def_abstract_eval(TanhPlugin.abstract_eval)
