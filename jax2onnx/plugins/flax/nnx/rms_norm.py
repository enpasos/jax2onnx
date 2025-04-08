"""
RMS Norm Plugin for JAX to ONNX conversion.

This plugin enables conversion of flax.nnx.RMSNorm layers to ONNX format.
It transforms JAX’s rms_norm operations into an ONNX RMSNormalization operator
with necessary Transpose operations for NHWC to NCHW conversion.
"""

from typing import TYPE_CHECKING
from flax import nnx
from jax import core
from onnx import helper
from jax.extend.core import Primitive
from jax2onnx.plugin_system import register_primitive, PrimitivePlugin
import numpy as np

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

# Define a new primitive for RMS norm.
nnx.rms_norm_p = Primitive("nnx.rms_norm")
nnx.rms_norm_p.multiple_results = False  # Set at initialization


@register_primitive(
    jaxpr_primitive=nnx.rms_norm_p.name,
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.RMSNorm",
    onnx=[
        {
            "component": "RMSNormalization",
            "doc": "https://example.com/onnx_RMSNormalization_doc",  # Replace with an appropriate doc link if available.
        },
    ],
    since="v0.3.0",
    context="primitives.nnx",
    testcases=[
        {
            "testcase": "rms_norm",
            "callable": nnx.RMSNorm(6, rngs=nnx.Rngs(0)),
            "input_shapes": [(11, 2, 2, 6)],
        },
        {
            "testcase": "rms_norm_2",
            "callable": nnx.RMSNorm(num_features=20, rngs=nnx.Rngs(0)),
            "input_shapes": [(2, 20)],
        },
    ],
)
class RMSNormPlugin(PrimitivePlugin):
    """
    Plugin for converting flax.nnx.RMSNorm to ONNX.

    Converts an RMSNorm operation into an RMSNormalization operator
    with necessary Transpose operations for NHWC to NCHW conversion.
    """

    @staticmethod
    def abstract_eval(x, scale, *args, **kwargs):
        """Abstract evaluation function for rms_norm."""
        return core.ShapedArray(x.shape, x.dtype)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handles conversion of rms_norm to ONNX format."""
        input_name = s.get_name(node_inputs[0])
        scale_name = s.get_name(node_inputs[1])
        final_output_name = s.get_name(node_outputs[0])
        epsilon = params.get("epsilon", 1e-5)

        node_inputs[0].aval.shape  # e.g. (11, 2, 2, 64) or (2, 20)

        # Calculate mean of squares
        mean_square_name = s.get_unique_name("mean_square")
        mean_square_node = helper.make_node(
            "ReduceMean",
            inputs=[input_name],
            outputs=[mean_square_name],
            name=s.get_unique_name("reduce_mean_square"),
            keepdims=1,  # Change to 1
        )
        s.add_node(mean_square_node)

        # Subtract mean square from input
        sub_name = s.get_unique_name("sub")
        sub_node = helper.make_node(
            "Sub",
            inputs=[input_name, mean_square_name],
            outputs=[sub_name],
            name=s.get_unique_name("sub_mean_square"),
        )
        s.add_node(sub_node)

        # Square the result
        square_name = s.get_unique_name("square")
        square_node = helper.make_node(
            "Pow",
            inputs=[sub_name, s.get_constant_name(np.array(2.0, dtype=np.float32))],
            outputs=[square_name],
            name=s.get_unique_name("square"),
        )
        s.add_node(square_node)

        # Calculate mean of squares
        mean_square_2_name = s.get_unique_name("mean_square_2")
        mean_square_2_node = helper.make_node(
            "ReduceMean",
            inputs=[square_name],
            outputs=[mean_square_2_name],
            name=s.get_unique_name("reduce_mean_square_2"),
            keepdims=1,  # Change to 1
        )
        s.add_node(mean_square_2_node)

        # Add epsilon
        add_epsilon_name = s.get_unique_name("add_epsilon")
        add_epsilon_node = helper.make_node(
            "Add",
            inputs=[
                mean_square_2_name,
                s.get_constant_name(np.array(epsilon, dtype=np.float32)),
            ],
            outputs=[add_epsilon_name],
            name=s.get_unique_name("add_epsilon"),
        )
        s.add_node(add_epsilon_node)

        # Calculate sqrt
        sqrt_name = s.get_unique_name("sqrt")
        sqrt_node = helper.make_node(
            "Sqrt",
            inputs=[add_epsilon_name],
            outputs=[sqrt_name],
            name=s.get_unique_name("sqrt"),
        )
        s.add_node(sqrt_node)

        # Divide input by sqrt
        div_name = s.get_unique_name("div")
        div_node = helper.make_node(
            "Div",
            inputs=[input_name, sqrt_name],
            outputs=[div_name],
            name=s.get_unique_name("div"),
        )
        s.add_node(div_node)

        # Multiply by scale
        s.get_unique_name("mul")
        mul_node = helper.make_node(
            "Mul",
            inputs=[div_name, scale_name],
            outputs=[final_output_name],
            name=s.get_unique_name("mul"),
        )
        s.add_node(mul_node)

    @staticmethod
    def _rms_norm(x, scale, epsilon):
        nnx.rms_norm_p.multiple_results = False
        return nnx.rms_norm_p.bind(x, scale, epsilon=epsilon)

    @staticmethod
    def rms_norm(x, scale, epsilon):
        """Binding function for rms_norm."""
        return RMSNormPlugin._rms_norm(x, scale, epsilon)

    @staticmethod
    def get_monkey_patch():
        """Returns a patched version of RMSNorm.__call__."""

        def patched_rms_norm_call(self, x):
            return RMSNormPlugin._rms_norm(
                x,
                self.scale.value,
                epsilon=self.epsilon,
            )

        return patched_rms_norm_call

    @staticmethod
    def patch_info():
        """Provides patching information."""
        return {
            "patch_targets": [nnx.RMSNorm],
            "patch_function": lambda _: RMSNormPlugin.get_monkey_patch(),
            "target_attribute": "__call__",
        }


# Register abstract evaluation function.
nnx.rms_norm_p.def_abstract_eval(RMSNormPlugin.abstract_eval)
