from collections.abc import Sequence
from typing import TYPE_CHECKING

from flax import nnx
from jax import core
from jax.extend.core import Primitive
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Define the MaxPool primitive
nnx.max_pool_p = Primitive("nnx.max_pool")
nnx.max_pool_p.multiple_results = False  # Correct initialization


@register_primitive(
    jaxpr_primitive=nnx.max_pool_p.name,
    jax_doc="https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.max_pool",
    onnx=[
        {
            "component": "MaxPool",
            "doc": "https://onnx.ai/onnx/operators/onnx__MaxPool.html",
        },
        {
            "component": "Transpose",
            "doc": "https://onnx.ai/onnx/operators/onnx__Transpose.html",
        },
    ],
    since="v0.1.0",
    context="primitives.nnx",
    component="max_pool",
    testcases=[
        {
            "testcase": "max_pool",
            "callable": lambda x: nnx.max_pool(
                x, window_shape=(2, 2), strides=(2, 2), padding="VALID"
            ),
            "input_shapes": [(1, 32, 32, 3)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "max_pool_same_padding",
            "callable": lambda x: nnx.max_pool(
                x, window_shape=(2, 2), strides=(2, 2), padding="SAME"
            ),
            "input_shapes": [(1, 32, 32, 3)],
            "run_only_f32_variant": True,
        },
    ],
)
class MaxPoolPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting flax.nnx.max_pool to ONNX.
    """

    @staticmethod
    def _compute_max_pool_output_shape(
        x_shape: tuple[int, ...],
        window_shape: Sequence[int],
        strides: Sequence[int],
        padding: str,
        input_format: str = "NHWC",
    ) -> tuple[int, ...]:
        """Compute output shape for MaxPool operation."""
        if input_format == "NHWC":
            spatial_dims = x_shape[1:-1]  # Extract H, W from NHWC
            batch_dim = x_shape[0]
            channel_dim = x_shape[-1]
        elif input_format == "NCHW":
            spatial_dims = x_shape[2:]  # Extract H, W from NCHW
            batch_dim = x_shape[0]
            channel_dim = x_shape[1]
        else:
            raise ValueError("Invalid input_format. Must be 'NHWC' or 'NCHW'.")

        out_dims = []
        for dim, w, s in zip(spatial_dims, window_shape, strides, strict=False):
            if padding.upper() == "VALID":
                out_dim = (dim - w) // s + 1
            elif padding.upper() == "SAME":
                out_dim = -(-dim // s)  # Equivalent to ceil(dim / s)
            else:
                raise ValueError("Unsupported padding: " + padding)
            out_dims.append(out_dim)

        return (
            (batch_dim, *out_dims, channel_dim)
            if input_format == "NHWC"
            else (batch_dim, channel_dim, *out_dims)
        )

    @staticmethod
    def abstract_eval(x, window_shape, strides, padding):
        """Abstract evaluation function for MaxPool."""
        out_shape = MaxPoolPlugin._compute_max_pool_output_shape(
            x.shape, window_shape, strides, padding, input_format="NHWC"
        )
        return core.ShapedArray(out_shape, x.dtype)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handles conversion of MaxPool to ONNX format."""
        input_var = node_inputs[0]
        input_name = s.get_name(input_var)
        final_output_name = s.get_name(node_outputs[0])

        window_shape = params.get("window_shape")
        strides = params.get("strides")
        padding = params.get("padding")

        jax_input_shape = input_var.aval.shape

        # === Pre-Transpose: NHWC -> NCHW ===
        pre_transpose_name = s.get_unique_name("pre_transpose")
        pre_transpose_node = helper.make_node(
            "Transpose",
            inputs=[input_name],
            outputs=[pre_transpose_name],
            name=s.get_unique_name("transpose_pre"),
            perm=[0, 3, 1, 2],  # NHWC -> NCHW
        )
        s.add_node(pre_transpose_node)
        pre_transposed_shape = (
            jax_input_shape[0],
            jax_input_shape[3],
            jax_input_shape[1],
            jax_input_shape[2],
        )
        s.add_shape_info(pre_transpose_name, pre_transposed_shape)

        # === MaxPool Node in ONNX (operates in NCHW) ===
        pool_out_name = s.get_unique_name("max_pool_output")

        if padding.upper() == "SAME":
            pads = []
            for i in range(len(window_shape)):
                in_dim = pre_transposed_shape[2 + i]  # NCHW
                out_dim = -(-in_dim // strides[i])  # ceil(in_dim / strides[i])
                total_pad = max(
                    0, (out_dim - 1) * strides[i] + window_shape[i] - in_dim
                )
                pad_before = total_pad // 2
                pad_after = total_pad - pad_before
                pads.extend([pad_before, pad_after])
        else:
            pads = [0] * (2 * len(window_shape))  # [0, 0, 0, 0] for 2D

        max_pool_node = helper.make_node(
            "MaxPool",
            inputs=[pre_transpose_name],
            outputs=[pool_out_name],
            name=s.get_unique_name("max_pool"),
            kernel_shape=window_shape,
            strides=strides,
            pads=pads,
        )
        s.add_node(max_pool_node)

        maxpool_output_shape_nchw = MaxPoolPlugin._compute_max_pool_output_shape(
            pre_transposed_shape, window_shape, strides, padding, input_format="NCHW"
        )
        s.add_shape_info(pool_out_name, maxpool_output_shape_nchw)

        # === Post-Transpose: NCHW -> NHWC ===
        post_transpose_node = helper.make_node(
            "Transpose",
            inputs=[pool_out_name],
            outputs=[final_output_name],
            name=s.get_unique_name("transpose_post"),
            perm=[0, 2, 3, 1],  # NCHW -> NHWC
        )
        s.add_node(post_transpose_node)

        final_output_shape = MaxPoolPlugin._compute_max_pool_output_shape(
            jax_input_shape, window_shape, strides, padding, input_format="NHWC"
        )
        s.add_shape_info(final_output_name, final_output_shape)  # Correctly track shape

    @staticmethod
    def _max_pool(x, window_shape, strides, padding):
        """Defines the primitive binding for MaxPool."""
        return nnx.max_pool_p.bind(
            x, window_shape=window_shape, strides=strides, padding=padding
        )

    @staticmethod
    def get_monkey_patch():
        """Provides patching information for MaxPool."""

        def patched_max_pool(x, window_shape, strides, padding):
            return MaxPoolPlugin._max_pool(x, window_shape, strides, padding)

        return patched_max_pool

    @staticmethod
    def patch_info():
        """Provides patching information for MaxPool."""
        return {
            "patch_targets": [nnx],
            "patch_function": lambda _: MaxPoolPlugin.get_monkey_patch(),
            "target_attribute": "max_pool",
        }


# Register abstract evaluation function
nnx.max_pool_p.def_abstract_eval(MaxPoolPlugin.abstract_eval)
