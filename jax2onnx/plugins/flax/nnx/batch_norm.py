"""
Batch Norm Plugin for JAX to ONNX conversion.

This plugin enables conversion of flax.nnx.BatchNorm layers to ONNX format.
It transforms JAX’s batch_norm operations into an ONNX BatchNormalization operator.
If a BatchNorm layer is provided in training mode (`use_running_average=False`),
it will be automatically converted to inference mode with a warning.

The conversion process involves:
  1. Defining a JAX primitive for BatchNorm's inference behavior.
  2. Providing an abstract evaluation for JAX's tracing system.
  3. Converting the operation to an ONNX BatchNormalization node.
  4. Monkey-patching BatchNorm.__call__ to redirect calls to our primitive,
     ensuring inference parameters (running mean/var) and default scale/bias
     are used.
"""

from typing import TYPE_CHECKING
import logging

import jax.numpy as jnp
from flax import nnx
from jax import core
from jax.extend.core import Primitive
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Define the BatchNorm primitive
nnx.batch_norm_p = Primitive("nnx.batch_norm")
nnx.batch_norm_p.multiple_results = False


@register_primitive(
    jaxpr_primitive=nnx.batch_norm_p.name,
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.BatchNorm",
    onnx=[
        {
            "component": "BatchNormalization",
            "doc": "https://onnx.ai/onnx/operators/onnx__BatchNormalization.html",
        }
    ],
    since="v0.1.0",
    context="primitives.nnx",
    component="batch_norm",
    testcases=[
        {
            "testcase": "batch_norm_no_bias_no_scale",
            "callable": nnx.BatchNorm(
                num_features=8,
                use_running_average=True,
                use_bias=False,
                use_scale=False,
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [("B", 8)],
        },
        {
            "testcase": "batch_norm_bias_no_scale",
            "callable": nnx.BatchNorm(
                num_features=8,
                use_running_average=True,
                use_bias=True,
                use_scale=False,
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [("B", 8)],
        },
        {
            "testcase": "batch_norm_no_bias_scale",
            "callable": nnx.BatchNorm(
                num_features=8,
                use_running_average=True,
                use_bias=False,
                use_scale=True,
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [("B", 8)],
        },
        {
            "testcase": "batch_norm_bias_scale",
            "callable": nnx.BatchNorm(
                num_features=8,
                use_running_average=True,
                use_bias=True,
                use_scale=True,
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [("B", 8)],
        },
        {
            "testcase": "batch_norm_4d",
            "callable": nnx.BatchNorm(
                num_features=3, use_running_average=True, rngs=nnx.Rngs(0)
            ),
            "input_shapes": [("B", 4, 4, 3)],
        },
        {
            "testcase": "batch_norm_4d_no_bias_no_scale",
            "callable": nnx.BatchNorm(
                num_features=3,
                use_running_average=True,
                use_bias=False,
                use_scale=False,
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [("B", 4, 4, 3)],
        },
        {
            "testcase": "batch_norm_training_mode_fallback",
            "callable": nnx.BatchNorm(
                num_features=8,
                use_running_average=False,
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [("B", 8)],
        },
    ],
)
class BatchNormPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting flax.nnx.BatchNorm to ONNX.
    """

    @staticmethod
    def abstract_eval(x, scale, bias, mean, var, **kwargs):
        """Abstract evaluation function for BatchNorm."""
        return core.ShapedArray(x.shape, x.dtype)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handles conversion of BatchNorm to ONNX format."""
        x_var, scale_var, bias_var, mean_var, var_var = node_inputs

        input_name = s.get_name(x_var)
        scale_name = s.get_name(scale_var)
        bias_name = s.get_name(bias_var)
        mean_name = s.get_name(mean_var)
        variance_name = s.get_name(var_var)
        output_name = s.get_name(node_outputs[0])

        epsilon = params.get("epsilon", 1e-5)
        momentum = params.get("momentum", 0.9)

        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=[input_name, scale_name, bias_name, mean_name, variance_name],
            outputs=[output_name],
            name=s.get_unique_name("batch_norm"),
            epsilon=epsilon,
            momentum=momentum,
        )
        s.add_node(bn_node)
        s.add_shape_info(output_name, x_var.aval.shape, x_var.aval.dtype)

    @staticmethod
    def _batch_norm(x, scale, bias, mean, var, epsilon, momentum):
        """Defines the primitive binding for BatchNorm."""
        return nnx.batch_norm_p.bind(
            x, scale, bias, mean, var, epsilon=epsilon, momentum=momentum
        )

    @staticmethod
    def get_monkey_patch():
        """Returns a patched version of BatchNorm's call method."""

        def patched_batch_norm_call(self, x, use_running_average=None, *, mask=None):
            if not self.use_running_average:
                logging.warning(
                    "BatchNorm is being converted with use_running_average=False. "
                    "The ONNX model will be created in inference mode."
                )

            param_dtype = self.param_dtype if self.param_dtype is not None else x.dtype

            if self.use_scale:
                scale_value = self.scale.value
            else:
                scale_value = jnp.ones((self.num_features,), dtype=param_dtype)

            if self.use_bias:
                bias_value = self.bias.value
            else:
                bias_value = jnp.zeros((self.num_features,), dtype=param_dtype)

            return BatchNormPlugin._batch_norm(
                x,
                scale_value,
                bias_value,
                self.mean.value,
                self.var.value,
                epsilon=self.epsilon,
                momentum=self.momentum,
            )

        return patched_batch_norm_call

    @staticmethod
    def patch_info():
        """Provides patching information for BatchNorm."""
        return {
            "patch_targets": [nnx.BatchNorm],
            "patch_function": lambda _: BatchNormPlugin.get_monkey_patch(),
            "target_attribute": "__call__",
        }


# Register abstract evaluation function
nnx.batch_norm_p.def_abstract_eval(BatchNormPlugin.abstract_eval)
