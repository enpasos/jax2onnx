import numpy as np
from jax import core, numpy as jnp
from jax.core import Primitive
from flax import nnx
from onnx import helper
import contextlib
from typing import TYPE_CHECKING, Tuple, Sequence

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

max_pool_p = Primitive("max_pool")


def get_primitive():
    return max_pool_p


def _compute_max_pool_output_shape(
    x_shape: Tuple[int, ...],
    window_shape: Sequence[int],
    strides: Sequence[int],
    padding: str,
) -> Tuple[int, ...]:
    # Compute the output shape for the spatial dimensions.
    spatial_dims = x_shape[-len(window_shape) :]
    out_dims = []
    for dim, w, s in zip(spatial_dims, window_shape, strides):
        if padding.upper() == "VALID":
            out_dim = (dim - w) // s + 1
        elif padding.upper() == "SAME":
            out_dim = -(-dim // s)
        else:
            raise ValueError("Unsupported padding: " + padding)
        out_dims.append(out_dim)
    return x_shape[: -len(window_shape)] + tuple(out_dims)


def _get_monkey_patch():
    def max_pool(x, window_shape, strides, padding):
        def max_pool_abstract_eval(x, window_shape, strides, padding):
            out_shape = _compute_max_pool_output_shape(
                x.shape, window_shape, strides, padding
            )
            return core.ShapedArray(out_shape, x.dtype)

        max_pool_p.multiple_results = False
        max_pool_p.def_abstract_eval(max_pool_abstract_eval)
        return max_pool_p.bind(
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
        # Expect node_inputs: input.
        input_var = node_inputs[0]
        # Get original name.
        input_name = s.get_name(input_var)
        pool_out_name = s.get_unique_name("max_pool_output")
        final_output_name = s.get_name(node_outputs[0])

        window_shape = params.get("window_shape")
        strides = params.get("strides")
        padding = params.get("padding")

        # === Pre-Transpose: NHWC -> NCHW ===
        pre_transpose_name = s.get_unique_name("pre_transpose")
        pre_transpose_node = helper.make_node(
            "Transpose",
            inputs=[input_name],
            outputs=[pre_transpose_name],
            name=s.get_unique_name("transpose_pre"),
            perm=[0, 3, 1, 2],
        )
        s.add_node(pre_transpose_node)

        # Create the MaxPool node.
        # For ONNX, pooling parameters apply to the H and W dimensions.
        pads = []
        if padding.upper() == "SAME":
            input_shape = input_var.aval.shape  # (B, H, W, C)
            for i in range(2):
                in_dim = input_shape[i + 1]
                filt_dim = window_shape[i]
                stride = strides[i]
                out_dim = -(-in_dim // stride)
                pad_total = max((out_dim - 1) * stride + filt_dim - in_dim, 0)
                pad_begin = pad_total // 2
                pad_end = pad_total - pad_begin
                pads.extend([pad_begin, pad_end])
        else:
            pads = [0, 0, 0, 0]
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

        # === Post-Transpose: NCHW -> NHWC ===
        post_transpose_node = helper.make_node(
            "Transpose",
            inputs=[pool_out_name],
            outputs=[final_output_name],
            name=s.get_unique_name("transpose_post"),
            perm=[0, 2, 3, 1],
        )
        s.add_node(post_transpose_node)

    return handle_max_pool


def get_metadata() -> dict:
    return {
        "jaxpr_primitive": "max_pool",
        "jax_doc": "https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.max_pool.html",
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
        "testcases": [
            {
                "testcase": "max_pool",
                "callable": lambda x: nnx.max_pool(
                    x, window_shape=(2, 2), strides=(2, 2), padding="VALID"
                ),
                "input_shapes": [(1, 3, 32, 32)],
                "parameters": {
                    "window_shape": (2, 2),
                    "strides": (2, 2),
                    "padding": "VALID",
                },
            }
        ],
    }
