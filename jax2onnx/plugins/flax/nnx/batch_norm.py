"""
Batch Norm Plugin for JAX to ONNX conversion.

This plugin enables conversion of flax.nnx.BatchNorm layers to ONNX format.
It transforms JAX’s batch_norm operations into an ONNX BatchNormalization operator
with necessary Transpose operations for NHWC to NCHW conversion.

The conversion process involves:
  1. Handling input shape and transpositions.
  2. Providing an abstract evaluation for JAX's tracing system.
  3. Converting the operation to ONNX using BatchNormalization and Transpose nodes.
  4. Monkey-patching BatchNorm.__call__ to redirect calls to our primitive.
"""

import contextlib
from typing import TYPE_CHECKING
from flax import nnx
from jax import core
from onnx import helper
from jax.extend.core import Primitive
from jax2onnx.plugin_system import register_primitive, PrimitivePlugin

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

# Define a new primitive for batch norm.
nnx.batch_norm_p = Primitive("nnx.batch_norm")
nnx.batch_norm_p.multiple_results = False  # Set once at initialization


@register_primitive(
    jaxpr_primitive=nnx.batch_norm_p.name,
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.BatchNorm",
    onnx=[
        {
            "component": "BatchNormalization",
            "doc": "https://onnx.ai/onnx/operators/onnx__BatchNormalization.html",
        },
    ],
    since="v0.1.0",
    context="primitives.nnx",
    testcases=[
        {
            "testcase": "batch_norm",
            "callable": nnx.BatchNorm(
                num_features=64, epsilon=1e-5, momentum=0.9, rngs=nnx.Rngs(0)
            ),
            "input_shapes": [(11, 2, 2, 64)],
        },
        {
            "testcase": "batch_norm_2",
            "callable": nnx.BatchNorm(num_features=20, rngs=nnx.Rngs(0)),
            "input_shapes": [(2, 20)],
        },
    ],
)
class BatchNormPlugin(PrimitivePlugin):
    """
    Plugin for converting flax.nnx.BatchNorm to ONNX.

    Converts a BatchNorm operation into a BatchNormalization operator
    with necessary Transpose operations for NHWC to NCHW conversion.
    """

    @staticmethod
    def abstract_eval(x, scale, bias, mean, var, *args, **kwargs):
        """Abstract evaluation function for batch_norm."""
        return core.ShapedArray(x.shape, x.dtype)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handles conversion of batch_norm to ONNX format."""
        input_name = s.get_name(node_inputs[0])
        scale_name = s.get_name(node_inputs[1])
        bias_name = s.get_name(node_inputs[2])
        mean_name = s.get_name(node_inputs[3])
        variance_name = s.get_name(node_inputs[4])
        final_output_name = s.get_name(node_outputs[0])
        epsilon = params.get("epsilon")

        jax_shape = node_inputs[0].aval.shape  # e.g. (11, 2, 2, 64) or (2,20)

        if len(jax_shape) == 4:
            pre_transpose_name = s.get_unique_name("bn_pre_transpose")
            pre_transpose_node = helper.make_node(
                "Transpose",
                inputs=[input_name],
                outputs=[pre_transpose_name],
                name=s.get_unique_name("bn_transpose_pre"),
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

            bn_output_name = s.get_unique_name("bn_output")
            batch_norm_node = helper.make_node(
                "BatchNormalization",
                inputs=[
                    pre_transpose_name,
                    scale_name,
                    bias_name,
                    mean_name,
                    variance_name,
                ],
                outputs=[bn_output_name],
                name=s.get_unique_name("batch_norm"),
                epsilon=epsilon,
            )
            s.add_node(batch_norm_node)
            s.add_shape_info(bn_output_name, pre_transposed_shape)

            post_transpose_node = helper.make_node(
                "Transpose",
                inputs=[bn_output_name],
                outputs=[final_output_name],
                name=s.get_unique_name("bn_transpose_post"),
                perm=[0, 2, 3, 1],  # NCHW -> NHWC
            )
            s.add_node(post_transpose_node)

        else:
            batch_norm_node = helper.make_node(
                "BatchNormalization",
                inputs=[input_name, scale_name, bias_name, mean_name, variance_name],
                outputs=[final_output_name],
                name=s.get_unique_name("batch_norm"),
                epsilon=epsilon,
            )
            s.add_node(batch_norm_node)

    @staticmethod
    def _batch_norm(x, scale, bias, mean, var, epsilon, use_running_average, momentum):
        nnx.batch_norm_p.multiple_results = False
        return nnx.batch_norm_p.bind(
            x,
            scale,
            bias,
            mean,
            var,
            epsilon=epsilon,
            use_running_average=use_running_average,
            momentum=momentum,
        )

    @staticmethod
    def batch_norm(x, scale, bias, mean, var, epsilon, use_running_average, momentum):
        """Binding function for batch_norm."""
        return BatchNormPlugin._batch_norm(
            x, scale, bias, mean, var, epsilon, use_running_average, momentum
        )

    @staticmethod
    def get_monkey_patch():
        """Returns a patched version of BatchNorm.__call__."""

        def patched_batch_norm_call(self, x):
            return BatchNormPlugin._batch_norm(
                x,
                self.scale.value,
                self.bias.value,
                self.mean.value,
                self.var.value,
                epsilon=self.epsilon,
                use_running_average=self.use_running_average,
                momentum=self.momentum,
            )

        return patched_batch_norm_call

    @staticmethod
    def patch_info():
        """Provides patching information."""
        return {
            "patch_targets": [nnx.BatchNorm],
            "patch_function": lambda _: BatchNormPlugin.get_monkey_patch(),
            "target_attribute": "__call__",
        }


# Register abstract evaluation function.
nnx.batch_norm_p.def_abstract_eval(BatchNormPlugin.abstract_eval)
