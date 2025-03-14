import jax
import numpy as np
from typing import TYPE_CHECKING, Tuple, Sequence, Dict, List
from onnx import helper

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def get_primitive():
    return jax.lax.conv_general_dilated_p


def _compute_conv_output_shape(
    x_shape: Tuple[int, ...],
    kernel_shape: Tuple[int, ...],
    strides: Sequence[int] | int,
    padding: str,
) -> Tuple[int, ...]:
    if isinstance(strides, int):
        strides = (strides, strides)
    N, H, W, _ = x_shape
    filter_height, filter_width, _, out_channels = kernel_shape
    if padding.upper() == "VALID":
        out_H = (H - filter_height) // strides[0] + 1
        out_W = (W - filter_width) // strides[1] + 1
    elif padding.upper() == "SAME":
        out_H = -(-H // strides[0])
        out_W = -(-W // strides[1])
    else:
        raise ValueError("Unsupported padding: " + padding)
    return (N, out_H, out_W, out_channels)


def get_handler(s: "Jaxpr2OnnxConverter"):
    def _handle_conv(node_inputs: List, node_outputs: List, params: Dict):
        input_name = s.get_name(node_inputs[0])
        filter_var = node_inputs[1]
        output_name = s.get_var_name(node_outputs[0])

        dimension_numbers = params["dimension_numbers"]
        window_strides = params["window_strides"]
        padding = params["padding"]

        # Handle dimension numbers.
        lhs_spec, rhs_spec, out_spec = dimension_numbers
        if lhs_spec == (0, 3, 1, 2) and rhs_spec == (3, 2, 0, 1):  # NHWC & HWIO
            input_perm = [0, 3, 1, 2]  # NHWC -> NCHW
            kernel_perm = [3, 2, 0, 1]  # HWIO -> OIHW
            output_perm = [0, 2, 3, 1]  # NCHW -> NHWC
        elif lhs_spec == (0, 1, 2, 3) and rhs_spec == (0, 1, 2, 3):  # NCHW & OIHW
            input_perm = None
            kernel_perm = [0, 1, 2, 3]  # already OIHW
            output_perm = None  # already NCHW
        else:
            raise ValueError(f"Unhandled dimension_numbers: {dimension_numbers}")

        # Transpose input if necessary.
        conv_input = s.get_name(node_inputs[0])
        if input_perm:
            transposed_input = s.get_unique_name("input_transposed")
            s.add_node(
                helper.make_node(
                    "Transpose",
                    inputs=[input_name],
                    outputs=[transposed_input],
                    perm=input_perm,
                    name=s.get_unique_name("Transpose_input"),
                )
            )
            conv_input = transposed_input

        # Handle kernel: constant branch or dynamic branch.
        filter_name = s.get_name(filter_var)
        if filter_name in s.name_to_const:
            kernel_const = s.name_to_const[filter_name]
            kernel_transposed = np.transpose(kernel_const, kernel_perm)
            transposed_kernel_name = s.get_constant_name(kernel_transposed)
            s.name_to_const[transposed_kernel_name] = kernel_transposed
            kernel_shape = kernel_transposed.shape[
                2:
            ]  # OIHW: spatial dims at indices 2,3
        else:
            # Dynamic branch: insert a Transpose node for the kernel.
            transposed_kernel_name = s.get_unique_name("kernel_transposed")
            s.add_node(
                helper.make_node(
                    "Transpose",
                    inputs=[filter_name],
                    outputs=[transposed_kernel_name],
                    perm=kernel_perm,
                    name=s.get_unique_name("Transpose_kernel"),
                )
            )
            # For dynamic kernels: if rhs_spec is HWIO, spatial dims are the first two elements;
            # if rhs_spec is OIHW, they are the last two elements.
            if rhs_spec == (3, 2, 0, 1):
                kernel_shape = filter_var.aval.shape[
                    :2
                ]  # HWIO: (H, W, I, O) → spatial dims H, W
            else:
                kernel_shape = filter_var.aval.shape[2:]

        # Process padding.
        if isinstance(padding, str):
            if padding.upper() == "VALID":
                pads = [0, 0, 0, 0]
            elif padding.upper() == "SAME":
                # Compute SAME padding based on input and kernel shapes.
                # For simplicity, assume symmetric padding.
                if lhs_spec == (0, 3, 1, 2):  # Input in NHWC format
                    H_in, W_in = node_inputs[0].aval.shape[1:3]
                else:  # NCHW
                    H_in, W_in = node_inputs[0].aval.shape[2:4]
                filter_H, filter_W = kernel_shape
                pad_top, pad_bottom = compute_same_pads(
                    H_in, filter_H, window_strides[0]
                )
                pad_left, pad_right = compute_same_pads(
                    W_in, filter_W, window_strides[1]
                )
                pads = [pad_top, pad_left, pad_bottom, pad_right]
            else:
                raise ValueError("Unsupported padding string: " + padding)
        else:
            pads = [pad for pair in padding for pad in pair]

        conv_output = s.get_unique_name("conv_output")
        conv_node = helper.make_node(
            "Conv",
            inputs=[conv_input, transposed_kernel_name],
            outputs=[conv_output],
            kernel_shape=kernel_shape,
            strides=window_strides,
            pads=pads,
            name=s.get_unique_name("Conv"),
        )
        s.add_node(conv_node)

        if output_perm:
            s.add_node(
                helper.make_node(
                    "Transpose",
                    inputs=[conv_output],
                    outputs=[output_name],
                    perm=output_perm,
                    name=s.get_unique_name("Transpose_output"),
                )
            )
        else:
            if conv_output != output_name:
                s.add_node(
                    helper.make_node(
                        "Identity",
                        inputs=[conv_output],
                        outputs=[output_name],
                        name=s.get_unique_name("Identity_output"),
                    )
                )

        s.add_shape_info(output_name, node_outputs[0].aval.shape)

    return _handle_conv


def compute_same_pads(input_size, filter_size, stride):
    out_size = int(np.ceil(float(input_size) / float(stride)))
    pad_total = max((out_size - 1) * stride + filter_size - input_size, 0)
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before
    return pad_before, pad_after


def get_metadata() -> dict:
    return {
        "jaxpr_primitive": "conv",
        "jax_doc": "https://docs.jax.dev/en/latest/_autosummary/jax.lax.conv.html",
        "onnx": [
            {
                "component": "Conv",
                "doc": "https://onnx.ai/onnx/operators/onnx__Conv.html",
            }
        ],
        "since": "v0.2.0",
        "context": "plugins.lax",
        "testcases": [
            {
                "testcase": "conv",  # NCHW & OIHW: no transposition needed.
                "callable": lambda x, y: jax.lax.conv(
                    x, y, window_strides=(1, 1), padding="VALID"
                ),
                "input_shapes": [(1, 2, 3, 3), (1, 2, 2, 2)],
            },
            {
                "testcase": "conv2",  # NHWC & HWIO: transposition required.
                "callable": lambda x, y: jax.lax.conv_general_dilated(
                    x,
                    y,
                    window_strides=(1, 1),
                    padding="VALID",
                    dimension_numbers=("NHWC", "HWIO", "NHWC"),
                ),
                "input_shapes": [(1, 3, 3, 2), (2, 2, 2, 1)],
            },
        ],
    }
