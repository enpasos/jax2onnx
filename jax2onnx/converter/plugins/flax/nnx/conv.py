# file: jax2onnx/converter/plugins/flax/nnx/conv.py

import numpy as np
from jax import core
from jax.extend.core import Primitive
from flax import nnx
from onnx import helper
import contextlib
from typing import TYPE_CHECKING, Tuple, Sequence

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

# Define a new JAX primitive for convolution.
nnx.conv_p = Primitive("nnx.conv")


def get_primitive():
    return nnx.conv_p


def _compute_conv_output_shape(
    x_shape: Tuple[int, ...],
    kernel_shape: Tuple[int, ...],
    strides: Sequence[int] | int,  # Allow strides to be a sequence or an integer
    padding: str,
) -> Tuple[int, ...]:
    """
    Compute the output shape for a 2D convolution.
    Assumes:
      - Input is in NHWC format: (N, H, W, C)
      - Kernel is in HWIO format: (filter_height, filter_width, in_channels, out_channels)
    """
    if isinstance(strides, int):
        strides = (strides, strides)
    N, H, W, _ = x_shape
    filter_height, filter_width, _, out_channels = kernel_shape
    if padding.upper() == "VALID":
        out_H = (H - filter_height) // strides[0] + 1
        out_W = (W - filter_width) // strides[1] + 1
    elif padding.upper() == "SAME":
        # Use ceiling division for SAME padding.
        out_H = -(-H // strides[0])
        out_W = -(-W // strides[1])
    else:
        raise ValueError("Unsupported padding: " + padding)
    return (N, out_H, out_W, out_channels)


def _get_monkey_patch():
    def conv(x, kernel, bias, strides, padding, dilations, dimension_numbers):
        def conv_abstract_eval(
            x, kernel, bias, strides, padding, dilations, dimension_numbers
        ):
            out_shape = _compute_conv_output_shape(
                x.shape, kernel.shape, strides, padding
            )
            return core.ShapedArray(out_shape, x.dtype)

        nnx.conv_p.multiple_results = False
        nnx.conv_p.def_abstract_eval(conv_abstract_eval)
        if bias is None:
            return nnx.conv_p.bind(
                x,
                kernel,
                strides=strides,
                padding=padding,
                dilations=dilations,
                dimension_numbers=dimension_numbers,
            )
        else:
            return nnx.conv_p.bind(
                x,
                kernel,
                bias,
                strides=strides,
                padding=padding,
                dilations=dilations,
                dimension_numbers=dimension_numbers,
            )

    def patched_conv_call(self, x):
        # Extract convolution parameters from the instance.
        strides = self.strides
        padding = self.padding
        dilations = getattr(self, "dilations", (1, 1))
        dimension_numbers = getattr(self, "dimension_numbers", None)
        kernel = self.kernel.value
        bias = (
            self.bias.value if hasattr(self, "bias") and self.bias is not None else None
        )
        return conv(x, kernel, bias, strides, padding, dilations, dimension_numbers)

    return patched_conv_call


@contextlib.contextmanager
def temporary_patch():
    original_call = nnx.Conv.__call__
    nnx.Conv.__call__ = _get_monkey_patch()
    try:
        yield
    finally:
        nnx.Conv.__call__ = original_call


def get_handler(s: "Jaxpr2OnnxConverter"):
    def handle_conv(node_inputs, node_outputs, params):
        # Expected node_inputs: [input, kernel, (optional) bias]
        input_var = node_inputs[0]
        kernel_var = node_inputs[1]
        bias_var = node_inputs[2] if len(node_inputs) > 2 else None

        # Get names from the converter.
        input_name = s.get_name(input_var)
        final_output_name = s.get_name(node_outputs[0])
        bias_name = s.get_name(bias_var) if bias_var is not None else None

        # Pre-Transpose: Convert input from NHWC -> NCHW.
        pre_transpose_name = s.get_unique_name("pre_transpose")
        pre_transpose_node = helper.make_node(
            "Transpose",
            inputs=[input_name],
            outputs=[pre_transpose_name],
            name=s.get_unique_name("transpose_pre"),
            perm=[0, 3, 1, 2],
        )
        s.add_node(pre_transpose_node)
        # Compute the pre-transposed shape.
        jax_input_shape = input_var.aval.shape  # e.g. (B, H, W, C)
        pre_transposed_shape = tuple(
            jax_input_shape[i] for i in [0, 3, 1, 2]
        )  # (B, C, H, W)
        s.add_shape_info(pre_transpose_name, pre_transposed_shape)

        # Convert kernel constant: from HWIO to OIHW.
        kernel_name = s.get_name(kernel_var)
        kernel_const = s.name_to_const[kernel_name]
        transposed_kernel = np.transpose(kernel_const, [3, 2, 0, 1])
        weights_name = s.get_constant_name(transposed_kernel)

        # Determine convolution parameters.
        strides = params.get("strides", (1, 1))
        if isinstance(strides, int):
            strides = (strides, strides)
        padding = params.get("padding", "VALID")
        dilations = params.get("dilations", (1, 1))

        # Create the Conv node. ONNX Conv expects input in NCHW and kernel in OIHW.
        conv_out_name = s.get_unique_name("conv_output")
        if bias_name is not None:
            conv_node = helper.make_node(
                "Conv",
                inputs=[pre_transpose_name, weights_name, bias_name],
                outputs=[conv_out_name],
                name=s.get_unique_name("conv"),
                strides=strides,
                dilations=dilations,
                pads=[0, 0, 0, 0] if padding.upper() == "VALID" else None,
            )
        else:
            conv_node = helper.make_node(
                "Conv",
                inputs=[pre_transpose_name, weights_name],
                outputs=[conv_out_name],
                name=s.get_unique_name("conv"),
                strides=strides,
                dilations=dilations,
                pads=[0, 0, 0, 0] if padding.upper() == "VALID" else None,
            )
        if padding.upper() == "SAME":
            # Compute symmetric padding for height and width.
            # ONNX expects pads in the order: [pad_top, pad_left, pad_bottom, pad_right]
            input_shape = input_var.aval.shape  # (B, H, W, C)
            filter_shape = kernel_const.shape  # (H, W, I, O)
            # Height padding.
            in_h = input_shape[1]
            filt_h = filter_shape[0]
            stride_h = strides[0]
            out_h = -(-in_h // stride_h)  # Ceiling division
            pad_total_h = max((out_h - 1) * stride_h + filt_h - in_h, 0)
            pad_top = pad_total_h // 2
            pad_bottom = pad_total_h - pad_top
            # Width padding.
            in_w = input_shape[2]
            filt_w = filter_shape[1]
            stride_w = strides[1]
            out_w = -(-in_w // stride_w)
            pad_total_w = max((out_w - 1) * stride_w + filt_w - in_w, 0)
            pad_left = pad_total_w // 2
            pad_right = pad_total_w - pad_left
            pads = [pad_top, pad_left, pad_bottom, pad_right]
            conv_node.attribute.append(helper.make_attribute("pads", pads))
        s.add_node(conv_node)
        # Compute the conv node's intermediate output shape (in NCHW):
        # First, get the expected final output shape in JAX (NHWC) using our helper:
        jax_output_shape = _compute_conv_output_shape(
            jax_input_shape, kernel_const.shape, strides, padding
        )
        # Then, compute the intermediate shape by transposing NHWC -> NCHW.
        conv_output_shape_NCHW = tuple(jax_output_shape[i] for i in [0, 3, 1, 2])
        s.add_shape_info(conv_out_name, conv_output_shape_NCHW)

        # Post-Transpose: Convert Conv output from NCHW -> NHWC.
        post_transpose_node = helper.make_node(
            "Transpose",
            inputs=[conv_out_name],
            outputs=[final_output_name],
            name=s.get_unique_name("transpose_post"),
            perm=[0, 2, 3, 1],
        )
        s.add_node(post_transpose_node)
        # The final output shape should match the JAX output shape.
        # s.add_shape_info(final_output_name, jax_output_shape)

    return handle_conv


def get_metadata() -> dict:
    return {
        "jaxpr_primitive": "nnx.conv",
        "jax_doc": "https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv_general_dilated.html",
        "onnx": [
            {
                "component": "Conv",
                "doc": "https://onnx.ai/onnx/operators/onnx__Conv.html",
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
                "testcase": "conv",
                "callable": nnx.Conv(
                    in_features=3,
                    out_features=16,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="SAME",
                    use_bias=True,
                    rngs=nnx.Rngs(0),
                ),
                "input_shapes": [("B", 28, 28, 3)],
            },
            {
                "testcase": "conv_2",
                "callable": nnx.Conv(1, 32, kernel_size=(3, 3), rngs=nnx.Rngs(0)),
                "input_shapes": [(2, 28, 28, 1)],
            },
            {
                "testcase": "conv_3",
                "callable": nnx.Conv(
                    in_features=1,
                    out_features=32,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="SAME",
                    use_bias=True,
                    rngs=nnx.Rngs(0),
                ),
                "input_shapes": [(3, 28, 28, 1)],
            },
            {
                "testcase": "conv_4",
                "callable": nnx.Conv(
                    in_features=32,
                    out_features=64,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    padding="SAME",
                    use_bias=True,
                    rngs=nnx.Rngs(0),
                ),
                "input_shapes": [(3, 28, 28, 32)],
            },
        ],
    }
