"""
Batch Norm Plugin for JAX to ONNX conversion.

This plugin enables conversion of flax.nnx.BatchNorm layers to ONNX format.
It transforms JAX’s batch_norm operations into an ONNX BatchNormalization operator.
If a BatchNorm layer is provided in training mode (use_running_average=False),
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


# ---------------------------------------------------------------------
# Python implementation **mirrors the layout trick** used in the ONNX
# graph: for rank > 2 we convert NHWC → NCHW, apply the formula, then
# convert back.  For 2-D tensors (N,C) no transpose is needed.
def _batch_norm_impl(x, scale, bias, mean, var, *, epsilon, momentum):
    del momentum  # inference-only

    rank = x.ndim

    if rank > 2:
        # Move channel (last axis) to position 1:  NHWC → NCHW
        x_nchw = jnp.moveaxis(x, -1, 1)

        # broadcast params over N, spatial dims
        param_shape = (1, -1) + (1,) * (rank - 2)  # (1,C,1,1,…)
        scale_ = jnp.reshape(scale, param_shape).astype(x.dtype, copy=False)
        bias_ = jnp.reshape(bias, param_shape).astype(x.dtype, copy=False)
        mean_ = jnp.reshape(mean, param_shape).astype(x.dtype, copy=False)
        var_ = jnp.reshape(var, param_shape).astype(x.dtype, copy=False)

        y = (x_nchw - mean_) * scale_ / jnp.sqrt(var_ + epsilon) + bias_

        # back to NHWC
        return jnp.moveaxis(y, 1, -1)

    # rank == 1 or 2  →  channel is already axis −1 == 1
    param_shape = (1,) * (rank - 1) + (-1,)  # (1,C) or (1,C)
    scale = jnp.reshape(scale, param_shape).astype(x.dtype, copy=False)
    bias = jnp.reshape(bias, param_shape).astype(x.dtype, copy=False)
    mean = jnp.reshape(mean, param_shape).astype(x.dtype, copy=False)
    var = jnp.reshape(var, param_shape).astype(x.dtype, copy=False)
    return (x - mean) * scale / jnp.sqrt(var + epsilon) + bias


# Register that implementation on the primitive
nnx.batch_norm_p.def_impl(_batch_norm_impl)
# ---------------------------------------------------------------------


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
            "run_only_f32_variant": True,
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
            "run_only_f32_variant": True,
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
            "run_only_f32_variant": True,
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
            "run_only_f32_variant": True,
        },
        # {
        #     "testcase": "batch_norm_4d",
        #     "callable": nnx.BatchNorm(
        #         num_features=3, use_running_average=True, rngs=nnx.Rngs(0)
        #     ),
        #     "input_shapes": [("B", 4, 4, 3)],
        #     "run_only_f32_variant": True,
        # },
        # {
        #     "testcase": "batch_norm_4d_no_bias_no_scale",
        #     "callable": nnx.BatchNorm(
        #         num_features=3,
        #         use_running_average=True,
        #         use_bias=False,
        #         use_scale=False,
        #         rngs=nnx.Rngs(0),
        #     ),
        #     "input_shapes": [("B", 4, 4, 3)],
        #     "run_only_f32_variant": True,
        # },
        # {
        #     "testcase": "batch_norm_training_mode_fallback",
        #     "callable": nnx.BatchNorm(
        #         num_features=8,
        #         use_running_average=False,
        #         rngs=nnx.Rngs(0),
        #     ),
        #     "input_shapes": [("B", 8)],
        #     "run_only_f32_variant": True,
        # },
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
        x_var, scale_var, bias_var, mean_var, var_var = node_inputs
        (out_var,) = node_outputs

        in_name = s.get_name(x_var)
        scale_name = s.get_name(scale_var)
        bias_name = s.get_name(bias_var)
        mean_name = s.get_name(mean_var)
        var_name = s.get_name(var_var)
        out_name = s.get_name(out_var)

        eps = params.get("epsilon", 1e-5)
        momentum = params.get("momentum", 0.9)

        shape = x_var.aval.shape
        rank = len(shape)

        # --- 1️⃣  NHWC → NCHW if needed ---------------------------------
        if rank > 2:
            perm = [0, rank - 1] + list(range(1, rank - 1))
            pre_trans = s.get_unique_name("bn_pre_transpose")
            s.add_node(
                helper.make_node(
                    "Transpose",
                    inputs=[in_name],
                    outputs=[pre_trans],
                    perm=perm,
                )
            )
            bn_in_name = pre_trans
            # Intermediate output in NCHW
            bn_out = s.get_unique_name("bn_nchw_out")
        else:
            bn_in_name = in_name
            bn_out = out_name

        # BatchNormalization itself
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=[bn_in_name, scale_name, bias_name, mean_name, var_name],
            outputs=[bn_out],
            name=s.get_unique_name("batch_norm"),
            epsilon=eps,
            momentum=momentum,
        )
        s.add_node(bn_node)

        # --- 2️⃣  NCHW → NHWC (restore original layout) ------------------
        if rank > 2:
            inv_perm = [0] + list(range(2, rank)) + [1]
            s.add_node(
                helper.make_node(
                    "Transpose",
                    inputs=[bn_out],
                    outputs=[out_name],
                    perm=inv_perm,
                )
            )

        # Tell the converter about the output tensor
        s.add_shape_info(out_name, x_var.aval.shape, x_var.aval.dtype)

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
            # ↪ patch every future BatchNorm subclass
            "patch_targets": [nnx.BatchNorm],
            # plugin runner will call this with the original attribute
            "patch_function": lambda _: BatchNormPlugin.get_monkey_patch(),
            "target_attribute": "__call__",
        }


# Register abstract evaluation function
nnx.batch_norm_p.def_abstract_eval(BatchNormPlugin.abstract_eval)
