# file: jax2onnx/converter/plugins/flax/nnx/avg_pool.py

from jax import core
from jax.extend.core import Primitive
from flax import nnx
from onnx import helper
import contextlib
from typing import TYPE_CHECKING, Tuple, Sequence

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

# Create a new primitive for avg_pool.
nnx.avg_pool_p = Primitive("nnx.avg_pool")


def get_primitive():
    return nnx.avg_pool_p


def _compute_avg_pool_output_shape(
    x_shape: Tuple[int, ...],
    window_shape: Sequence[int],
    strides: Sequence[int],
    padding: str,
    input_format: str = "NHWC",
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
    else:  # input_format == "NCHW"
        return (batch_dim, channel_dim) + tuple(out_dims)


def _get_monkey_patch():
    def avg_pool(
        inputs, window_shape, strides=None, padding="VALID", count_include_pad=True
    ):
        def avg_pool_abstract_eval(
            inputs, window_shape, strides, padding, count_include_pad
        ):
            out_shape = _compute_avg_pool_output_shape(
                inputs.shape, window_shape, strides, padding, input_format="NHWC"
            )
            return core.ShapedArray(out_shape, inputs.dtype)

        nnx.avg_pool_p.multiple_results = False
        nnx.avg_pool_p.def_abstract_eval(avg_pool_abstract_eval)
        if strides is None:
            strides = (1, 1)
        # Pass the static parameters as keyword arguments.
        return nnx.avg_pool_p.bind(
            inputs,
            window_shape=window_shape,
            strides=strides,
            padding=padding,
            count_include_pad=count_include_pad,
        )

    # Since nnx.avg_pool is a plain function, our patched version does not include a 'self' parameter.
    def patched_avg_pool_call(
        inputs, window_shape, strides=None, padding="VALID", count_include_pad=True
    ):
        if strides is None:
            strides = (1, 1)
        return avg_pool(inputs, window_shape, strides, padding, count_include_pad)

    return patched_avg_pool_call


@contextlib.contextmanager
def temporary_patch():
    original_fn = nnx.avg_pool
    nnx.avg_pool = _get_monkey_patch()
    try:
        yield
    finally:
        nnx.avg_pool = original_fn


def get_handler(s: "Jaxpr2OnnxConverter"):
    def handle_avg_pool(node_inputs, node_outputs, params):
        # Expect node_inputs: [input]
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

        avg_pool_node = helper.make_node(
            "AveragePool",  # Use AveragePool instead of MaxPool
            inputs=[pre_transpose_name],
            outputs=[pool_out_name],
            name=s.get_unique_name("avg_pool"),
            kernel_shape=window_shape,
            strides=strides,
            pads=pads,
            count_include_pad=0,  # Do not include padding in the averaging calculation
        )
        s.add_node(avg_pool_node)

        avgpool_output_shape_nchw = _compute_avg_pool_output_shape(
            pre_transposed_shape, window_shape, strides, padding, input_format="NCHW"
        )
        s.add_shape_info(pool_out_name, avgpool_output_shape_nchw)

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

        _compute_avg_pool_output_shape(
            jax_input_shape, window_shape, strides, padding, input_format="NHWC"
        )
        # s.add_shape_info(final_output_name, final_output_shape)

    return handle_avg_pool


def get_metadata() -> dict:
    return {
        "jaxpr_primitive": "avg_pool",
        "jax_doc": "https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.avg_pool",
        "onnx": [
            {
                "component": "AveragePool",
                "doc": "https://onnx.ai/onnx/operators/onnx__AveragePool.html",
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
                "testcase": "avg_pool",
                "callable": lambda x: nnx.avg_pool(
                    x, window_shape=(2, 2), strides=(2, 2), padding="VALID"
                ),
                "input_shapes": [(1, 32, 32, 3)],
            },
            {
                "testcase": "avg_pool_same_padding",
                "callable": lambda x: nnx.avg_pool(
                    x, window_shape=(2, 2), strides=(2, 2), padding="SAME"
                ),
                "input_shapes": [(1, 32, 32, 3)],
            },
            {
                "testcase": "avg_pool_default_padding",
                "callable": lambda x: nnx.avg_pool(
                    x, window_shape=(2, 2), strides=(2, 2)
                ),
                "input_shapes": [(1, 32, 32, 3)],
            },
        ],
    }
