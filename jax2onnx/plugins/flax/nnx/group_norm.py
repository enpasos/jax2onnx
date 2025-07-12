"""
Group Norm Plugin for JAX to ONNX conversion.

This plugin enables conversion of flax.nnx.GroupNorm layers to ONNX format.
It transforms JAX’s group_norm operations into an ONNX GroupNormalization operator
with necessary Transpose operations for NHWC to NCHW conversion.
"""

from typing import TYPE_CHECKING

import jax.numpy as jnp
from flax import nnx
from jax import core
from jax.extend.core import Primitive
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Define a new primitive for group norm.
nnx.group_norm_p = Primitive("nnx.group_norm")
nnx.group_norm_p.multiple_results = False  # Set at initialization


@register_primitive(
    jaxpr_primitive=nnx.group_norm_p.name,
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.GroupNorm",
    onnx=[
        {
            "component": "GroupNormalization",
            "doc": "https://onnx.ai/onnx/operators/onnx__GroupNormalization.html",
        },
    ],
    since="v0.3.0",
    context="primitives.nnx",
    component="group_norm",
    testcases=[
        {
            "testcase": "group_norm",
            "callable": nnx.GroupNorm(num_features=64, rngs=nnx.Rngs(0)),
            "input_shapes": [(11, 2, 2, 64)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "group_norm_no_bias_no_scale",
            "callable": nnx.GroupNorm(
                32, num_groups=8, use_bias=False, use_scale=False, rngs=nnx.Rngs(0)
            ),
            "input_shapes": [("B", 16, 16, 32)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "group_norm_bias_no_scale",
            "callable": nnx.GroupNorm(
                32, num_groups=8, use_bias=True, use_scale=False, rngs=nnx.Rngs(0)
            ),
            "input_shapes": [("B", 16, 16, 32)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "group_norm_no_bias_scale",
            "callable": nnx.GroupNorm(
                32, num_groups=8, use_bias=False, use_scale=True, rngs=nnx.Rngs(0)
            ),
            "input_shapes": [("B", 16, 16, 32)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "group_norm_bias_scale",
            "callable": nnx.GroupNorm(
                32, num_groups=8, use_bias=True, use_scale=True, rngs=nnx.Rngs(0)
            ),
            "input_shapes": [("B", 16, 16, 32)],
            "run_only_f32_variant": True,
        },
    ],
)
class GroupNormPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting flax.nnx.GroupNorm to ONNX.

    Converts a GroupNorm operation into a GroupNormalization operator
    with necessary Transpose operations for NHWC to NCHW conversion.
    """

    @staticmethod
    def abstract_eval(x, scale, bias, *args, **kwargs):
        """Abstract evaluation function for group_norm."""
        return core.ShapedArray(x.shape, x.dtype)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handles conversion of group_norm to ONNX format."""
        input_var, scale_var, bias_var = node_inputs
        input_name = s.get_name(input_var)
        scale_name = s.get_name(scale_var)
        bias_name = s.get_name(bias_var)

        final_output_name = s.get_name(node_outputs[0])
        epsilon = params.get("epsilon", 1e-5)
        num_groups = params.get("num_groups")
        jax_shape = input_var.aval.shape

        if len(jax_shape) == 4:
            pre_transpose_name = s.get_unique_name("gn_pre_transpose")
            pre_transpose_node = helper.make_node(
                "Transpose",
                inputs=[input_name],
                outputs=[pre_transpose_name],
                name=s.get_unique_name("gn_transpose_pre"),
                perm=[0, 3, 1, 2],  # NHWC -> NCHW
            )
            s.add_node(pre_transpose_node)
            pre_transposed_shape = (
                jax_shape[0],
                jax_shape[3],
                jax_shape[1],
                jax_shape[2],
            )
            s.add_shape_info(pre_transpose_name, pre_transposed_shape)

            gn_output_name = s.get_unique_name("gn_output")
            group_norm_node = helper.make_node(
                "GroupNormalization",
                inputs=[pre_transpose_name, scale_name, bias_name],
                outputs=[gn_output_name],
                name=s.get_unique_name("group_norm"),
                epsilon=epsilon,
                num_groups=num_groups,
            )
            s.add_node(group_norm_node)
            s.add_shape_info(gn_output_name, pre_transposed_shape)

            post_transpose_node = helper.make_node(
                "Transpose",
                inputs=[gn_output_name],
                outputs=[final_output_name],
                name=s.get_unique_name("gn_transpose_post"),
                perm=[0, 2, 3, 1],  # NCHW -> NHWC
            )
            s.add_node(post_transpose_node)
        else:
            group_norm_node = helper.make_node(
                "GroupNormalization",
                inputs=[input_name, scale_name, bias_name],
                outputs=[final_output_name],
                name=s.get_unique_name("group_norm"),
                epsilon=epsilon,
                num_groups=num_groups,
            )
            s.add_node(group_norm_node)

    @staticmethod
    def _group_norm(x, scale, bias, epsilon, num_groups):
        nnx.group_norm_p.multiple_results = False
        return nnx.group_norm_p.bind(
            x, scale, bias, epsilon=epsilon, num_groups=num_groups
        )

    @staticmethod
    def group_norm(x, scale, bias, epsilon, num_groups):
        """Binding function for group_norm."""
        return GroupNormPlugin._group_norm(x, scale, bias, epsilon, num_groups)

    @staticmethod
    def get_monkey_patch():
        """Returns a patched version of GroupNorm.__call__."""

        def patched_group_norm_call(self, x):
            num_features = x.shape[-1]
            param_dtype = self.param_dtype if self.param_dtype is not None else x.dtype

            if (
                self.use_scale
                and self.scale is not None
                and self.scale.value.shape[-1] == num_features
            ):
                # learned γ matches the current feature count – use it
                scale_value = self.scale.value
            else:
                # shape‐mismatch → fall back to "identity" γ = 1
                scale_value = jnp.ones((num_features,), dtype=param_dtype)

            if (
                self.use_bias
                and self.bias is not None
                and self.bias.value.shape[-1] == num_features
            ):
                beta_value = self.bias.value
            else:
                # shape‐mismatch → β = 0
                beta_value = jnp.zeros((num_features,), dtype=param_dtype)

            return GroupNormPlugin._group_norm(
                x,
                scale_value,
                beta_value,
                epsilon=self.epsilon,
                num_groups=self.num_groups,
            )

        return patched_group_norm_call

    @staticmethod
    def patch_info():
        """Provides patching information."""
        return {
            "patch_targets": [nnx.GroupNorm],
            "patch_function": lambda _: GroupNormPlugin.get_monkey_patch(),
            "target_attribute": "__call__",
        }


# Register abstract evaluation function.
nnx.group_norm_p.def_abstract_eval(GroupNormPlugin.abstract_eval)
