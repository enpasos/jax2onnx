from typing import TYPE_CHECKING

import jax
import jax.nn as nn
from jax import core
from jax.extend.core import Primitive
from jax.interpreters import batching
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Define a new primitive for softmax
nn.softmax_p = Primitive("nn.softmax")
nn.softmax_p.multiple_results = False  # Correct initialization


@register_primitive(
    jaxpr_primitive=nn.softmax_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.softmax.html",
    onnx=[
        {
            "component": "Softmax",
            "doc": "https://onnx.ai/onnx/operators/onnx__Softmax.html",
        }
    ],
    since="v0.1.0",
    context="primitives.nn",
    component="softmax",
    testcases=[
        {
            "testcase": "softmax",
            "callable": lambda x: nn.softmax(x),
            "input_shapes": [(3,)],
        },
        {
            "testcase": "softmax_2d",
            "callable": lambda x: nn.softmax(x, axis=1),
            "input_shapes": [(4, 5)],
        },
        {
            "testcase": "softmax_3d",
            "callable": lambda x: nn.softmax(x, axis=2),
            "input_shapes": [(2, 3, 4)],
        },
    ],
)
class SoftmaxPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting jax.nn.softmax to ONNX.
    """

    @staticmethod
    def abstract_eval(x, axis=-1):
        """Computes the output shape for nn.softmax."""
        return core.ShapedArray(x.shape, x.dtype)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handles ONNX conversion for nn.softmax."""
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
    def _softmax(x, axis=-1):
        """Defines the primitive binding for Softmax."""
        return nn.softmax_p.bind(x, axis=axis)

    @staticmethod
    def get_monkey_patch():
        """Provides patching information for Softmax."""

        def patched_softmax(x, axis=-1):
            return SoftmaxPlugin._softmax(x, axis)

        return patched_softmax

    @staticmethod
    def patch_info():
        """Provides patching information for Softmax."""
        return {
            "patch_targets": [nn],
            "patch_function": lambda _: SoftmaxPlugin.get_monkey_patch(),
            "target_attribute": "softmax",
        }


# Register abstract evaluation function
nn.softmax_p.def_abstract_eval(SoftmaxPlugin.abstract_eval)


# Define the batching rule for nn.softmax
def softmax_batching_rule(batched_args, batch_dims, axis=-1):
    """Batching rule for jax.nn.softmax."""
    (x,) = batched_args
    (bdim,) = batch_dims

    # If axis is negative, make it positive
    if axis < 0:
        axis = x.ndim + axis

    # If the batched dimension is at or before the axis, we need to increment the axis
    axis + (1 if bdim <= axis else 0)

    # Move the batch dimension to the front
    if bdim != 0:
        x = batching.moveaxis(x, bdim, 0)

    x_max_reduced = jax.lax.reduce_max(x, axes=(axis,))  # Use axis instead of new_axis
    x_max = jax.lax.expand_dims(
        x_max_reduced, dimensions=(axis,)
    )  # Use axis instead of new_axis
    shifted = x - x_max
    exp_shifted = jax.lax.exp(shifted)
    sum_exp_reduced = jax.lax.reduce_sum(
        exp_shifted, axes=(axis,)
    )  # Reduce without keepdims
    sum_exp = jax.lax.expand_dims(
        sum_exp_reduced, dimensions=(axis,)
    )  # Add the dimension back
    result = exp_shifted / sum_exp

    # Return result with batch dimension at position 0
    return result, 0


# Register the batching rule
batching.primitive_batchers[nn.softmax_p] = softmax_batching_rule
