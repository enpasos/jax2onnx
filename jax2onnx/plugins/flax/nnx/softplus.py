from typing import TYPE_CHECKING

from flax import nnx
from jax import core
from jax.extend.core import Primitive
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Define the Softplus primitive
nnx.softplus_p = Primitive("nnx.softplus")
nnx.softplus_p.multiple_results = False  # Correct initialization


@register_primitive(
    jaxpr_primitive=nnx.softplus_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.softplus.html",
    onnx=[
        {
            "component": "Softplus",
            "doc": "https://onnx.ai/onnx/operators/onnx__Softplus.html",
        }
    ],
    since="v0.1.0",
    context="primitives.nnx",
    component="softplus",
    testcases=[
        {
            "testcase": "softplus",
            "callable": lambda x: nnx.softplus(x),
            "input_shapes": [(3,)],
            "run_only_f32_variant": True,
        }
    ],
)
class SoftplusPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting flax.nnx.softplus to ONNX.
    """

    @staticmethod
    def abstract_eval(x):
        """Abstract evaluation function for Softplus."""
        return core.ShapedArray(x.shape, x.dtype)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handles conversion of Softplus to ONNX format."""
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

    @staticmethod
    def get_monkey_patch():
        """Provides patching information for Softplus."""

        def patched_softplus(x):
            return nnx.softplus_p.bind(x)

        return patched_softplus

    @staticmethod
    def patch_info():
        """Provides patching information for Softplus."""
        return {
            "patch_targets": [nnx],
            "patch_function": lambda _: SoftplusPlugin.get_monkey_patch(),
            "target_attribute": "softplus",
        }


# Register abstract evaluation function
nnx.softplus_p.def_abstract_eval(SoftplusPlugin.abstract_eval)
