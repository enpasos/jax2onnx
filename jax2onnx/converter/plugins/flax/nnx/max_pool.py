# file: jax2onnx/converter/plugins/flax/nnx/max_pool.py

from jax import core
from jax.extend.core import Primitive
from flax import nnx
from onnx import helper
import contextlib
from typing import TYPE_CHECKING, Tuple, Sequence

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

nnx.max_pool_p = Primitive("nnx.max_pool")


def get_primitive():
    return nnx.max_pool_p


def _compute_max_pool_output_shape(
    x_shape: Tuple[int, ...],
    window_shape: Sequence[int],
    strides: Sequence[int],
    padding: str,
    input_format: str = "NHWC",  # Add input format parameter
) -> Tuple[int, ...]:
    # Compute the output shape for the spatial dimensions.
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
    for dim, w, s in zip(spatial_dims, window_shape, strides):
        if padding.upper() == "VALID":
            out_dim = (dim - w) // s + 1
        elif padding.upper() == "SAME":
            out_dim = -(-dim // s)  # Equivalent to ceil(dim / s)
        else:
            raise ValueError("Unsupported padding: " + padding)
        out_dims.append(out_dim)

    if input_format == "NHWC":
        return (batch_dim,) + tuple(out_dims) + (channel_dim,)
    else:  # input_format == "NCHW":
        return (batch_dim, channel_dim) + tuple(out_dims)


def _get_monkey_patch():
    def max_pool(x, window_shape, strides, padding):
        def max_pool_abstract_eval(x, window_shape, strides, padding):
            out_shape = _compute_max_pool_output_shape(
                x.shape,
                window_shape,
                strides,
                padding,
                input_format="NHWC",  # Input is NHWC
            )
            return core.ShapedArray(out_shape, x.dtype)

        nnx.max_pool_p.multiple_results = False
        nnx.max_pool_p.def_abstract_eval(max_pool_abstract_eval)
        return nnx.max_pool_p.bind(
            x, window_shape=window_shape, strides=strides, padding=padding
        )

    return max_pool


@contextlib.contextmanager
def temporary_patch():
    original_fn = nnx.max_pool
    nnx.max_pool = _get_monkey_patch()
    try:
        yield
    finally:
        nnx.max_pool = original_fn


def get_handler(s: "Jaxpr2OnnxConverter"):
    def handle_max_pool(node_inputs, node_outputs, params):
        # Expect node_inputs: [input]
        input_var = node_inputs[0]
        input_name = s.get_name(input_var)
        final_output_name = s.get_name(node_outputs[0])

        window_shape = params.get("window_shape")  # e.g. (2, 2)
        strides = params.get("strides")  # e.g. (2, 2)
        padding = params.get("padding")  # e.g. "VALID"

        # The JAX input is in NHWC, e.g. (1, 32, 32, 3)
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
        # Compute and record the pre-transposed shape: (B, C, H, W)
        pre_transposed_shape = (
            jax_input_shape[0],
            jax_input_shape[3],
            jax_input_shape[1],
            jax_input_shape[2],
        )
        s.add_shape_info(pre_transpose_name, pre_transposed_shape)

        # === MaxPool Node in ONNX (operates in NCHW) ===
        pool_out_name = s.get_unique_name("max_pool_output")

        # Handle padding.  ONNX MaxPool 'pads' attribute takes [x1_begin, x2_begin, ..., x1_end, x2_end,...]
        if padding.upper() == "SAME":
            # Calculate pads for SAME padding.  This simulates the JAX behavior.
            pads = []
            for i in range(len(window_shape)):
                in_dim = pre_transposed_shape[2 + i]  # NCHW
                out_dim = -(-in_dim // strides[i])  #  ceil(in_dim / strides[i])
                total_pad = max(
                    0, (out_dim - 1) * strides[i] + window_shape[i] - in_dim
                )
                pad_before = total_pad // 2
                pad_after = total_pad - pad_before
                pads.extend([pad_before, pad_after])
        else:  # padding == "VALID" or anything else (already checked in _compute_max_pool_output_shape)
            pads = [0] * (2 * len(window_shape))  # [0, 0, 0, 0] for 2D

        max_pool_node = helper.make_node(
            "MaxPool",
            inputs=[pre_transpose_name],
            outputs=[pool_out_name],
            name=s.get_unique_name("max_pool"),
            kernel_shape=window_shape,
            strides=strides,
            pads=pads,  # Explicitly set pads
        )
        s.add_node(max_pool_node)

        # Compute the ONNX output shape in NCHW.
        maxpool_output_shape_nchw = _compute_max_pool_output_shape(
            pre_transposed_shape,
            window_shape,
            strides,
            padding,
            input_format="NCHW",  # ONNX is NCHW
        )
        s.add_shape_info(pool_out_name, maxpool_output_shape_nchw)

        # === Post-Transpose: NCHW -> NHWC ===
        s.get_unique_name("post_transpose")
        post_transpose_node = helper.make_node(
            "Transpose",
            inputs=[pool_out_name],
            outputs=[final_output_name],
            name=s.get_unique_name("transpose_post"),
            perm=[0, 2, 3, 1],  # NCHW -> NHWC
        )
        s.add_node(post_transpose_node)

        # Final output shape (NHWC) - Use the common shape calculation.
        final_output_shape = _compute_max_pool_output_shape(
            jax_input_shape,
            window_shape,
            strides,
            padding,
            input_format="NHWC",  # Output is NHWC
        )
        s.add_shape_info(final_output_name, final_output_shape)

    return handle_max_pool


def get_metadata() -> dict:
    return {
        "jaxpr_primitive": "nnx.max_pool",
        "jax_doc": "https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.max_pool",
        "onnx": [
            {
                "component": "MaxPool",
                "doc": "https://onnx.ai/onnx/operators/onnx__MaxPool.html",
            },
            {
                "component": "Transpose",
                "doc": "https://onnx.ai/onnx/operators/onnx__Transpose.html",
            },
        ],
        "since": "v0.1.0",
        "context": "plugins.nnx",
        "testcases": [
            {
                "testcase": "max_pool",
                "callable": lambda x: nnx.max_pool(
                    x, window_shape=(2, 2), strides=(2, 2), padding="VALID"
                ),
                "input_shapes": [(1, 32, 32, 3)],
            },
            {
                "testcase": "max_pool_same_padding",
                "callable": lambda x: nnx.max_pool(
                    x, window_shape=(2, 2), strides=(2, 2), padding="SAME"
                ),
                "input_shapes": [(1, 32, 32, 3)],
            },
        ],
    }
