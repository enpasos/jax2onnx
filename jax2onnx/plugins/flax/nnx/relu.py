from typing import TYPE_CHECKING

from flax import nnx
from jax import core
from jax.extend.core import Primitive
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Define the ReLU primitive
nnx.relu_p = Primitive("nnx.relu")
nnx.relu_p.multiple_results = False  # Correct initialization


@register_primitive(
    jaxpr_primitive=nnx.relu_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.relu.html",
    onnx=[
        {
            "component": "Relu",
            "doc": "https://onnx.ai/onnx/operators/onnx__Relu.html",
        }
    ],
    since="v0.1.0",
    context="primitives.nnx",
    component="relu",
    testcases=[
        {
            "testcase": "relu_1d",
            "callable": lambda x: nnx.relu(x),
            "input_shapes": [(3,)],
        },
        {
            "testcase": "relu_4d",
            "callable": lambda x: nnx.relu(x),
            "input_shapes": [("B", 28, 28, 32)],
        },
    ],
)
class ReluPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting flax.nnx.relu to ONNX.
    """

    @staticmethod
    def abstract_eval(x):
        """Abstract evaluation function for ReLU."""
        return core.ShapedArray(x.shape, x.dtype)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handles conversion of ReLU to ONNX format."""
        input_var = node_inputs[0]
        output_var = node_outputs[0]

        input_name = s.get_name(input_var)
        output_name = s.get_name(output_var)

        relu_node = helper.make_node(
            "Relu",
            inputs=[input_name],
            outputs=[output_name],
            name=s.get_unique_name("relu"),
        )
        s.add_node(relu_node)

    @staticmethod
    def get_monkey_patch():
        """Provides patching information for ReLU."""

        def patched_relu(x):
            return nnx.relu_p.bind(x)

        return patched_relu

    @staticmethod
    def patch_info():
        """Provides patching information for ReLU."""
        return {
            "patch_targets": [nnx],
            "patch_function": lambda _: ReluPlugin.get_monkey_patch(),
            "target_attribute": "relu",
        }


# Register abstract evaluation function
nnx.relu_p.def_abstract_eval(ReluPlugin.abstract_eval)
