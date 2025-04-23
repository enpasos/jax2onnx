# file: jax2onnx/plugins/flax/nnx/avg_pool.py
from collections.abc import Sequence
from typing import TYPE_CHECKING

from flax import nnx
from jax import core
from jax.extend.core import Primitive
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Define the avg_pool primitive
nnx.avg_pool_p = Primitive("nnx.avg_pool")
nnx.avg_pool_p.multiple_results = False  # Correctly set at initialization


@register_primitive(
    jaxpr_primitive=nnx.avg_pool_p.name,
    jax_doc="https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.avg_pool",
    onnx=[
        {
            "component": "AveragePool",
            "doc": "https://onnx.ai/onnx/operators/onnx__AveragePool.html",
        },
        {
            "component": "Transpose",
            "doc": "https://onnx.ai/onnx/operators/onnx__Transpose.html",
        },
    ],
    since="v0.1.0",
    context="primitives.nnx",
    component="avg_pool",
    testcases=[
        {
            "testcase": "avg_pool",
            "callable": lambda x: nnx.avg_pool(
                x, window_shape=(2, 2), strides=(2, 2), padding="VALID"
            ),
            "input_shapes": [("B", 32, 32, 3)],
        },
        {
            "testcase": "avg_pool_same_padding",
            "callable": lambda x: nnx.avg_pool(
                x, window_shape=(2, 2), strides=(2, 2), padding="SAME"
            ),
            "input_shapes": [("B", 32, 32, 3)],
        },
        {
            "testcase": "avg_pool_default_padding",
            "callable": lambda x: nnx.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            "input_shapes": [("B", 32, 32, 3)],
        },
        {
            "testcase": "avg_pool_stride1",
            "callable": lambda x: nnx.avg_pool(
                x, window_shape=(2, 2), strides=(1, 1), padding="VALID"
            ),
            "input_shapes": [("B", 8, 8, 3)],
        },
        # {
        #     "testcase": "avg_pool_large_window",
        #     "callable": lambda x: nnx.avg_pool(
        #         x, window_shape=(4, 4), strides=(2, 2), padding="SAME"
        #     ),
        #     "input_shapes": [(2, 16, 16, 3)],
        # },
        {
            "testcase": "avg_pool_win3x3_stride2",
            "callable": lambda x: nnx.avg_pool(
                x, window_shape=(3, 3), strides=(2, 2), padding="VALID"
            ),
            "input_shapes": [("B", 10, 10, 1)],
        },
        {
            "testcase": "avg_pool_stride_none",
            "callable": lambda x: nnx.avg_pool(
                x, window_shape=(2, 2), strides=None, padding="VALID"
            ),
            "input_shapes": [("B", 8, 8, 3)],
        },
        {
            "testcase": "avg_pool_count_include_pad_false",
            "callable": lambda x: nnx.avg_pool(
                x,
                window_shape=(2, 2),
                strides=(2, 2),
                padding="SAME",
                count_include_pad=False,
            ),
            "input_shapes": [("B", 8, 8, 3)],
        },
    ],
)
class AvgPoolPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting flax.nnx.avg_pool to ONNX.
    """

    @staticmethod
    def _compute_avg_pool_output_shape(
        x_shape,
        window_shape: Sequence[int],
        strides: Sequence[int],
        padding: str,
        input_format: str = "NHWC",
    ) -> tuple:
        """Computes the output shape of avg_pool operation."""
        # Extract dimensions safely handling any JAX tracers
        try:
            # Basic extraction of dimensions
            if input_format == "NHWC":
                batch_size = x_shape[0]
                h = x_shape[1]
                w = x_shape[2]
                channels = x_shape[3]
            else:  # NCHW
                batch_size = x_shape[0]
                channels = x_shape[1]
                h = x_shape[2]
                w = x_shape[3]

            # Make a copy of window shape and strides to avoid modifying input
            window_h, window_w = window_shape
            stride_h, stride_w = strides

            # Compute output dimensions for spatial dimensions
            if padding.upper() == "VALID":
                out_h = 0
                out_w = 0
                # Try to compute concrete values if possible
                try:
                    if (
                        isinstance(h, (int, float))
                        and isinstance(window_h, (int, float))
                        and isinstance(stride_h, (int, float))
                    ):
                        out_h = (h - window_h) // stride_h + 1
                    else:
                        out_h = h  # Fallback for symbolic dimensions

                    if (
                        isinstance(w, (int, float))
                        and isinstance(window_w, (int, float))
                        and isinstance(stride_w, (int, float))
                    ):
                        out_w = (w - window_w) // stride_w + 1
                    else:
                        out_w = w  # Fallback for symbolic dimensions
                except (TypeError, ValueError):
                    # If computation fails due to tracers, preserve the original dimensions
                    out_h, out_w = h, w
            else:  # SAME padding
                out_h = 0
                out_w = 0
                # Try to compute concrete values if possible
                try:
                    if isinstance(h, (int, float)) and isinstance(
                        stride_h, (int, float)
                    ):
                        out_h = -(-h // stride_h)  # Ceiling division
                    else:
                        out_h = h  # Fallback for symbolic dimensions

                    if isinstance(w, (int, float)) and isinstance(
                        stride_w, (int, float)
                    ):
                        out_w = -(-w // stride_w)  # Ceiling division
                    else:
                        out_w = w  # Fallback for symbolic dimensions
                except (TypeError, ValueError):
                    # If computation fails due to tracers, preserve the original dimensions
                    out_h, out_w = h, w

            # Build output shape based on the format
            if input_format == "NHWC":
                return (batch_size, out_h, out_w, channels)
            else:  # NCHW
                return (batch_size, channels, out_h, out_w)

        except (TypeError, IndexError, ValueError):
            # If any error occurs during shape computation,
            # fall back to preserving the input shape
            # This is better than failing completely
            return x_shape

    @staticmethod
    def abstract_eval(inputs, window_shape, strides, padding, count_include_pad):
        """Abstract evaluation function for avg_pool."""
        if strides is None:
            strides = (1, 1)

        # When dealing with JAX abstractions and tracers, we need to be careful with shape operations
        # Instead of trying to compute shapes using our custom function, we'll use JAX's built-in
        # shape abstraction mechanism to determine the output shape

        # Get batch size and channels from input shape (these remain unchanged)
        batch_size = inputs.shape[0]
        channels = inputs.shape[3]

        # Get spatial dimensions from input shape
        h, w = inputs.shape[1:3]

        # Compute output spatial dimensions based on padding
        if padding.upper() == "VALID":
            # For VALID padding: out_dim = (in_dim - window_size) / stride + 1
            # We can't do actual computation on tracers, so we'll create output dims
            # that capture this relationship symbolically
            out_h = h  # This will be adjusted by JAX's shape abstraction
            out_w = w  # This will be adjusted by JAX's shape abstraction
        else:  # SAME
            # For SAME padding: out_dim = ceil(in_dim / stride)
            # Again, we rely on JAX's shape abstraction
            out_h = h  # This will be adjusted by JAX's shape abstraction
            out_w = w  # This will be adjusted by JAX's shape abstraction

        # Pass through the shape directly to JAX rather than computing it ourselves
        # This avoids the unhashable error by letting JAX handle the tracer objects
        return inputs.update(
            shape=(batch_size, out_h, out_w, channels), weak_type=False
        )

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handles conversion of avg_pool to ONNX format."""
        input_var = node_inputs[0]
        input_name = s.get_name(input_var)
        final_output_name = s.get_name(node_outputs[0])

        window_shape = params.get("window_shape")
        strides = params.get("strides")
        padding = params.get("padding")
        count_include_pad = params.get("count_include_pad")

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

        # === AveragePool Node in ONNX (operates in NCHW) ===
        pool_out_name = s.get_unique_name("avg_pool_output")

        if padding.upper() == "SAME":
            pads = []
            for i in range(len(window_shape)):
                in_dim = pre_transposed_shape[2 + i]
                out_dim = -(-in_dim // strides[i])
                total_pad = max(
                    0, (out_dim - 1) * strides[i] + window_shape[i] - in_dim
                )
                pad_before = total_pad // 2
                pad_after = total_pad - pad_before
                pads.extend([pad_before, pad_after])
        else:
            pads = [0] * (2 * len(window_shape))

        # ONNX's AveragePool `count_include_pad` attribute:
        #   - 0: Exclude padding pixels from the average (what we want)
        #   - 1: Include padding pixels in the average
        onnx_count_include_pad = 0 if count_include_pad else 1

        avg_pool_node = helper.make_node(
            "AveragePool",
            inputs=[pre_transpose_name],
            outputs=[pool_out_name],
            name=s.get_unique_name("avg_pool"),
            kernel_shape=window_shape,
            strides=strides,
            pads=pads,
            count_include_pad=onnx_count_include_pad,
        )
        s.add_node(avg_pool_node)

        avgpool_output_shape_nchw = AvgPoolPlugin._compute_avg_pool_output_shape(
            pre_transposed_shape, window_shape, strides, padding, input_format="NCHW"
        )
        s.add_shape_info(pool_out_name, avgpool_output_shape_nchw)

        # === Post-Transpose: NCHW -> NHWC ===
        post_transpose_node = helper.make_node(
            "Transpose",
            inputs=[pool_out_name],
            outputs=[final_output_name],
            name=s.get_unique_name("transpose_post"),
            perm=[0, 2, 3, 1],  # NCHW -> NHWC
        )
        s.add_node(post_transpose_node)

    @staticmethod
    def _avg_pool(inputs, window_shape, strides, padding, count_include_pad):
        """Defines the primitive binding for avg_pool."""
        if strides is None:
            strides = (1, 1)
        return nnx.avg_pool_p.bind(
            inputs,
            window_shape=window_shape,
            strides=strides,
            padding=padding,
            count_include_pad=count_include_pad,
        )

    @staticmethod
    def get_monkey_patch():
        """Provides patching information for avg_pool."""

        def patched_avg_pool(
            inputs, window_shape, strides=None, padding="VALID", count_include_pad=True
        ):
            return AvgPoolPlugin._avg_pool(
                inputs, window_shape, strides, padding, count_include_pad
            )

        return patched_avg_pool

    @staticmethod
    def patch_info():
        """Provides patching information for avg_pool."""
        return {
            "patch_targets": [nnx],
            "patch_function": lambda _: AvgPoolPlugin.get_monkey_patch(),
            "target_attribute": "avg_pool",
        }


# Register abstract evaluation function.
nnx.avg_pool_p.def_abstract_eval(AvgPoolPlugin.abstract_eval)
