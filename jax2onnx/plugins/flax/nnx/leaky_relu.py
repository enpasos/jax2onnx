from typing import TYPE_CHECKING

from flax import nnx
from jax import core
from jax.extend.core import Primitive
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Define the LeakyReLU primitive
nnx.leaky_relu_p = Primitive("nnx.leaky_relu")
nnx.leaky_relu_p.multiple_results = False  # Correctly set at initialization


@register_primitive(
    jaxpr_primitive=nnx.leaky_relu_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.leaky_relu.html",
    onnx=[
        {
            "component": "LeakyRelu",
            "doc": "https://onnx.ai/onnx/operators/onnx__LeakyRelu.html",
        }
    ],
    since="v0.1.0",
    context="primitives.nnx",
    component="leaky_relu",
    testcases=[
        {
            "testcase": "leaky_relu",
            "callable": lambda x: nnx.leaky_relu(x),
            "input_shapes": [(3,)],
            "run_only_f32_variant": True,
        }
    ],
)
class LeakyReluPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting flax.nnx.leaky_relu to ONNX.
    """

    @staticmethod
    def abstract_eval(x, negative_slope=0.01):
        """Abstract evaluation function for LeakyReLU."""
        return core.ShapedArray(x.shape, x.dtype)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handles conversion of LeakyReLU to ONNX format."""
        input_var = node_inputs[0]
        output_var = node_outputs[0]

        input_name = s.get_name(input_var)
        output_name = s.get_name(output_var)

        # Retrieve the negative_slope parameter (defaulting to 0.01 if not provided)
        negative_slope = params.get("negative_slope", 0.01)

        leaky_relu_node = helper.make_node(
            "LeakyRelu",
            inputs=[input_name],
            outputs=[output_name],
            name=s.get_unique_name("leaky_relu"),
            alpha=negative_slope,
        )
        s.add_node(leaky_relu_node)

    @staticmethod
    def get_monkey_patch():
        """Provides patching information for LeakyReLU."""

        def patched_leaky_relu(x, negative_slope=0.01):
            return nnx.leaky_relu_p.bind(x, negative_slope=negative_slope)

        return patched_leaky_relu

    @staticmethod
    def patch_info():
        """Provides patching information for LeakyReLU."""
        return {
            "patch_targets": [nnx],
            "patch_function": lambda _: LeakyReluPlugin.get_monkey_patch(),
            "target_attribute": "leaky_relu",
        }


# Register abstract evaluation function
nnx.leaky_relu_p.def_abstract_eval(LeakyReluPlugin.abstract_eval)
