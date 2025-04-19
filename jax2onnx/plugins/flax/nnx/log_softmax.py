from typing import TYPE_CHECKING

from flax import nnx
from jax import core
from jax.extend.core import Primitive
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Define the LogSoftmax primitive
nnx.log_softmax_p = Primitive("nnx.log_softmax")
nnx.log_softmax_p.multiple_results = False  # Correct initialization


@register_primitive(
    jaxpr_primitive=nnx.log_softmax_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.log_softmax.html",
    onnx=[
        {
            "component": "LogSoftmax",
            "doc": "https://onnx.ai/onnx/operators/onnx__LogSoftmax.html",
        }
    ],
    since="v0.1.0",
    context="primitives.nnx",
    component="log_softmax",
    testcases=[
        {
            "testcase": "log_softmax",
            "callable": lambda x: nnx.log_softmax(x),
            "input_shapes": [(3,)],
        }
    ],
)
class LogSoftmaxPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting flax.nnx.log_softmax to ONNX.
    """

    @staticmethod
    def abstract_eval(x, axis=-1):
        """Abstract evaluation function for LogSoftmax."""
        return core.ShapedArray(x.shape, x.dtype)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handles conversion of LogSoftmax to ONNX format."""
        input_var = node_inputs[0]
        output_var = node_outputs[0]

        input_name = s.get_name(input_var)
        output_name = s.get_name(output_var)

        # Retrieve the axis parameter (defaulting to -1 if not provided)
        axis = params.get("axis", -1)

        log_softmax_node = helper.make_node(
            "LogSoftmax",
            inputs=[input_name],
            outputs=[output_name],
            name=s.get_unique_name("log_softmax"),
            axis=axis,
        )
        s.add_node(log_softmax_node)

    @staticmethod
    def get_monkey_patch():
        """Provides patching information for LogSoftmax."""

        def patched_log_softmax(x, axis=-1):
            return nnx.log_softmax_p.bind(x, axis=axis)

        return patched_log_softmax

    @staticmethod
    def patch_info():
        """Provides patching information for LogSoftmax."""
        return {
            "patch_targets": [nnx],
            "patch_function": lambda _: LogSoftmaxPlugin.get_monkey_patch(),
            "target_attribute": "log_softmax",
        }


# Register abstract evaluation function
nnx.log_softmax_p.def_abstract_eval(LogSoftmaxPlugin.abstract_eval)
