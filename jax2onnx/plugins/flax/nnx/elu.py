from typing import TYPE_CHECKING

from flax import nnx
from jax import core
from jax.extend.core import Primitive
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Define the ELU primitive
nnx.elu_p = Primitive("nnx.elu")
nnx.elu_p.multiple_results = False  # Correctly set at initialization


@register_primitive(
    jaxpr_primitive=nnx.elu_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.elu.html",
    onnx=[
        {
            "component": "Elu",
            "doc": "https://onnx.ai/onnx/operators/onnx__Elu.html",
        }
    ],
    since="v0.1.0",
    context="primitives.nnx",
    component="elu",
    testcases=[
        {
            "testcase": "elu",
            "callable": lambda x: nnx.elu(x),
            "input_shapes": [(3,)],
            "run_only_f32_variant": True,
        }
    ],
)
class EluPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting flax.nnx.elu to ONNX.
    """

    @staticmethod
    def abstract_eval(x, alpha=1.0):
        """Abstract evaluation function for ELU."""
        return core.ShapedArray(x.shape, x.dtype)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handles conversion of ELU to ONNX format."""
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

    @staticmethod
    def get_monkey_patch():
        """Provides patching information for ELU."""

        def patched_elu(x, alpha=1.0):
            return nnx.elu_p.bind(x, alpha=alpha)

        return patched_elu

    @staticmethod
    def patch_info():
        """Provides patching information for ELU."""
        return {
            "patch_targets": [nnx],
            "patch_function": lambda _: EluPlugin.get_monkey_patch(),
            "target_attribute": "elu",
        }


# Register abstract evaluation function
nnx.elu_p.def_abstract_eval(EluPlugin.abstract_eval)
