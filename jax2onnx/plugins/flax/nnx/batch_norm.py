# file: jax2onnx/plugins/flax/nnx/batch_norm.py

"""
Batch Norm Plugin for JAX to ONNX conversion.

This plugin enables conversion of flax.nnx.BatchNorm layers (in inference mode)
to ONNX format. It transforms JAX’s batch_norm operations into an ONNX
BatchNormalization operator with necessary Transpose operations for
NHWC/NLC to NCHW/NCL conversion and adds required shape information.

The conversion process involves:
  1. Defining a JAX primitive for BatchNorm's inference behavior.
  2. Providing an abstract evaluation for JAX's tracing system.
  3. Handling input shape, transpositions (2D, 3D, 4D), and adding ONNX shape info.
  4. Converting the operation to ONNX using BatchNormalization and Transpose nodes.
  5. Monkey-patching BatchNorm.__call__ to redirect calls to our primitive,
     ensuring inference parameters (running mean/var) and default scale/bias
     are used.
"""

from typing import TYPE_CHECKING

import numpy as np
from flax import nnx
from jax import core
from jax.extend.core import Primitive
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Define the BatchNorm primitive
nnx.batch_norm_p = Primitive("nnx.batch_norm")
nnx.batch_norm_p.multiple_results = False  # Correctly set at initialization


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
            "testcase": "batch_norm_simple",
            "callable": nnx.BatchNorm(
                num_features=1, use_running_average=True, epsilon=1e-5, rngs=nnx.Rngs(0)
            ),
            "input_shapes": [(2, 1)],
        },
        {
            "testcase": "batch_norm_2d",
            "callable": nnx.BatchNorm(
                num_features=8, use_running_average=True, epsilon=1e-5, rngs=nnx.Rngs(0)
            ),
            "input_shapes": [(16, 8)],
        },
        {
            "testcase": "batch_norm_2d_use_bias_false",
            "callable": nnx.BatchNorm(
                num_features=8,
                use_running_average=True,
                use_bias=False,
                epsilon=1e-5,
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [(16, 8)],
        },
        {
            "testcase": "batch_norm_2d_use_scale_false",
            "callable": nnx.BatchNorm(
                num_features=8,
                use_running_average=True,
                use_scale=False,
                epsilon=1e-5,
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [(16, 8)],
        },
        {
            "testcase": "batch_norm_4d",
            "callable": nnx.BatchNorm(
                num_features=3, use_running_average=True, epsilon=1e-5, rngs=nnx.Rngs(0)
            ),
            "input_shapes": [(2, 4, 4, 3)],
        },
        {
            "testcase": "batch_norm_4d_use_bias_false",
            "callable": nnx.BatchNorm(
                num_features=3,
                use_running_average=True,
                use_bias=False,
                epsilon=1e-5,
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [(2, 4, 4, 3)],
        },
        {
            "testcase": "batch_norm_4d_use_scale_false",
            "callable": nnx.BatchNorm(
                num_features=3,
                use_running_average=True,
                use_scale=False,
                epsilon=1e-5,
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [(2, 4, 4, 3)],
        },
        {
            "testcase": "batch_norm_minimal",
            "callable": nnx.BatchNorm(
                num_features=1, use_running_average=True, epsilon=1e-5, rngs=nnx.Rngs(0)
            ),
            "input_shapes": [(1, 1)],
            "input_values": np.array([[0.0]], dtype=np.float32),
        },
    ],
)
class BatchNormPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting flax.nnx.BatchNorm to ONNX.
    """

    @staticmethod
    def abstract_eval(x, **kwargs):
        """Abstract evaluation function for BatchNorm."""
        return core.ShapedArray(x.shape, x.dtype)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handles conversion of BatchNorm to ONNX format."""
        input_name = s.get_name(node_inputs[0])
        output_name = s.get_name(node_outputs[0])

        params.get("use_running_average", False)
        params.get("axis", -1)
        params.get("axis_index_groups", None)
        params.get("axis_name", None)
        use_bias = params.get("use_bias", True)
        params.get("use_fast_variance", False)
        use_scale = params.get("use_scale", True)
        dtype = getattr(node_inputs[0].aval, "dtype", np.float32)
        params.get("num_features", None)
        bias = params.get("bias", None)
        scale = params.get("scale", None)
        var = params.get("var", None)
        mean = params.get("mean", None)
        epsilon = params.get("epsilon")
        momentum = params.get("momentum", 0.9)
        params.get("use_running_average", False)

        # --- handle NHWC/NCHW transpose and param shapes ---
        input_shape = node_inputs[0].aval.shape
        input_rank = len(input_shape)
        channel_axis = input_rank - 1  # NHWC/NLC: last axis
        C = input_shape[channel_axis]

        # For NHWC->NCHW and NLC->NCL, channel order is preserved, so no reordering is needed.
        # Just ensure parameters are 1D (C,) and ONNX initializers.
        def ensure_onnx_param(param, name, default_value):
            arr = (
                np.array(param, dtype=dtype)
                if param is not None
                else np.full((C,), default_value, dtype=dtype)
            )
            arr = arr.reshape((C,))
            if arr.shape != (C,):
                raise ValueError(
                    f"BatchNorm param '{name}' must be shape ({C},), got {arr.shape}"
                )
            return s.builder.get_constant_name(arr)

        scale_name = (
            ensure_onnx_param(scale, "scale", 1.0)
            if use_scale
            else s.builder.get_constant_name(np.ones((C,), dtype=dtype))
        )
        bias_name = (
            ensure_onnx_param(bias, "bias", 0.0)
            if use_bias
            else s.builder.get_constant_name(np.zeros((C,), dtype=dtype))
        )
        mean_name = ensure_onnx_param(mean, "mean", 0.0)
        variance_name = ensure_onnx_param(var, "var", 1.0)

        # Insert transpose if needed
        pre_perm = post_perm = None
        if input_rank == 4:
            pre_perm = [0, 3, 1, 2]
            post_perm = [0, 2, 3, 1]
        elif input_rank == 3:
            pre_perm = [0, 2, 1]
            post_perm = [0, 2, 1]

        bn_input = input_name
        if pre_perm:
            transposed_input = s.get_unique_name("bn_pre_transpose")
            pre_transpose_node = helper.make_node(
                "Transpose",
                inputs=[input_name],
                outputs=[transposed_input],
                name=s.get_unique_name("bn_transpose_pre"),
                perm=pre_perm,
            )
            s.add_node(pre_transpose_node)
            transposed_shape = tuple(input_shape[i] for i in pre_perm)
            s.add_shape_info(transposed_input, transposed_shape, dtype)
            bn_input = transposed_input

        bn_output = output_name
        if post_perm:
            bn_output = s.get_unique_name("bn_output")

        inputs = [bn_input, scale_name, bias_name, mean_name, variance_name]

        # Set ONNX momentum attribute to match JAX/Flax value
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=inputs,
            outputs=[bn_output],
            name=s.get_unique_name("batch_norm"),
            epsilon=epsilon,
            momentum=momentum,  # Explicitly set momentum to match params
        )
        s.add_node(bn_node)
        if post_perm:
            s.add_shape_info(bn_output, transposed_shape, dtype)
        if post_perm:
            post_transpose_node = helper.make_node(
                "Transpose",
                inputs=[bn_output],
                outputs=[output_name],
                name=s.get_unique_name("bn_transpose_post"),
                perm=post_perm,
            )
            s.add_node(post_transpose_node)
            s.add_shape_info(output_name, input_shape, dtype)

    @staticmethod
    def _batch_norm(
        x,
        use_running_average,
        axis,
        axis_index_groups,
        axis_name,
        bias,
        dtype,
        epsilon,
        mean,
        momentum,
        num_features,
        scale,
        use_bias,
        use_fast_variance,
        use_scale,
        var,
    ):
        """Defines the primitive binding for BatchNorm."""
        return nnx.batch_norm_p.bind(
            x,
            use_running_average=use_running_average,
            axis=axis,
            axis_index_groups=axis_index_groups,
            axis_name=axis_name,
            bias=bias,
            dtype=dtype,
            epsilon=epsilon,
            mean=mean,
            momentum=momentum,
            num_features=num_features,
            scale=scale,
            use_bias=use_bias,
            use_fast_variance=use_fast_variance,
            use_scale=use_scale,
            var=var,
        )

    @staticmethod
    def get_monkey_patch():
        """Returns a patched version of BatchNorm's call method."""

        def patched_batch_norm_call(self, x, use_running_average=None, *, mask=None):
            return BatchNormPlugin._batch_norm(
                x,
                self.use_running_average,
                self.axis,
                self.axis_index_groups,
                self.axis_name,
                self.bias.value if getattr(self, "bias", None) is not None else None,
                self.dtype,
                self.epsilon,
                self.mean.value,
                self.momentum,
                self.num_features,
                self.scale.value if getattr(self, "scale", None) is not None else None,
                self.use_bias,
                self.use_fast_variance,
                self.use_scale,
                self.var.value,
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
