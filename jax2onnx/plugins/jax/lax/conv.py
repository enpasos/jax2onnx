from typing import TYPE_CHECKING, Any, Tuple

import jax
import numpy as np
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


def compute_same_pads(input_size, filter_size, stride):
    out_size = int(np.ceil(float(input_size) / float(stride)))
    pad_total = max((out_size - 1) * stride + filter_size - input_size, 0)
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before
    return pad_before, pad_after


def _get_perm_from_layout(source_layout: str, target_layout: str) -> list[int] | None:
    """Returns the permutation to transpose from source to target layout."""
    if source_layout == target_layout:
        return None
    source_indices = {axis: i for i, axis in enumerate(source_layout)}
    return [source_indices[axis] for axis in target_layout]


def _get_spatial_dims_from_spec(spec: Tuple[int, ...]) -> Tuple[int, int]:
    """Extracts spatial dimension indices from an integer-based layout spec."""
    # This is based on the assumption that H and W are the last two spatial dims
    # in JAX's tuple representation, which holds for standard layouts.
    # Example: NCHW is (0,1,2,3), NHWC is (0,2,3,1). H and W are always at indices > 1.
    return spec[2], spec[3]


def _get_spatial_dims(layout: str) -> list[int]:
    """Returns the indices of spatial dimensions (H, W) in a given layout string."""
    return [i for i, char in enumerate(layout) if char in "HW"]


@register_primitive(
    jaxpr_primitive=jax.lax.conv_general_dilated_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.conv.html",
    onnx=[
        {
            "component": "Conv",
            "doc": "https://onnx.ai/onnx/operators/onnx__Conv.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="conv",
    testcases=[
        {
            "testcase": "conv",  # NCHW & OIHW: no transposition needed.
            "callable": lambda x, y: jax.lax.conv(
                x, y, window_strides=(1, 1), padding="VALID"
            ),
            "input_shapes": [(1, 2, 3, 3), (1, 2, 2, 2)],
            "run_only_f32_variant": True,
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
            "run_only_f32_variant": True,
        },
        # TODO: enable testcases
        # {
        #     "testcase": "conv_general_dilated_nhwc_output",
        #     "callable": lambda x, k: jax.lax.conv_general_dilated(
        #         x, k,
        #         window_strides=(1, 1),
        #         padding='SAME',
        #         dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        #     ),
        #     "input_values": [
        #         np.ones((1, 5, 5, 3), dtype=np.float32),
        #         np.ones((2, 2, 3, 4), dtype=np.float32)
        #     ],
        #     "expected_output_shapes": [(1, 5, 5, 4)],
        #     "run_only_f32_variant": True,
        # },
    ],
)
class ConvGeneralDilatedPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting jax.lax.conv_general_dilated to ONNX.
    """

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        input_name = s.get_name(node_inputs[0])
        filter_var = node_inputs[1]
        output_name = s.get_name(node_outputs[0])

        dimension_numbers = params["dimension_numbers"]
        window_strides = params["window_strides"]
        padding = params["padding"]

        lhs_spec, rhs_spec, out_spec = dimension_numbers
        if lhs_spec == (0, 3, 1, 2) and rhs_spec == (3, 2, 0, 1):
            input_perm = [0, 3, 1, 2]
            kernel_perm = [3, 2, 0, 1]
            output_perm = [0, 2, 3, 1]
        elif lhs_spec == (0, 1, 2, 3) and rhs_spec == (0, 1, 2, 3):
            input_perm = None
            kernel_perm = [0, 1, 2, 3]
            output_perm = None
        else:
            raise ValueError(f"Unhandled dimension_numbers: {dimension_numbers}")

        conv_input = input_name
        if input_perm:
            transposed_input = s.get_unique_name("input_transposed")
            input_shape = node_inputs[0].aval.shape
            transposed_input_shape = tuple(input_shape[i] for i in input_perm)
            s.add_node(
                helper.make_node(
                    "Transpose",
                    inputs=[input_name],
                    outputs=[transposed_input],
                    perm=input_perm,
                    name=s.get_unique_name("Transpose_input"),
                )
            )
            s.add_shape_info(transposed_input, transposed_input_shape)

            conv_input = transposed_input

        filter_name = s.get_name(filter_var)
        if filter_name in s.name_to_const:
            kernel_const = s.name_to_const[filter_name]
            kernel_transposed = np.transpose(kernel_const, kernel_perm)
            transposed_kernel_name = s.get_constant_name(kernel_transposed)
            s.name_to_const[transposed_kernel_name] = kernel_transposed
            kernel_shape = kernel_transposed.shape[2:]
        else:
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
            new_kernel_shape = tuple(
                np.array(filter_var.aval.shape)[kernel_perm].tolist()
            )
            s.add_shape_info(transposed_kernel_name, new_kernel_shape)
            if rhs_spec == (3, 2, 0, 1):
                kernel_shape = filter_var.aval.shape[:2]
            else:
                kernel_shape = filter_var.aval.shape[2:]

        if isinstance(padding, str):
            if padding.upper() == "VALID":
                pads = [0, 0, 0, 0]
            elif padding.upper() == "SAME":
                if lhs_spec == (0, 3, 1, 2):  # NHWC
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

        # ----------  Build Conv attributes ----------
        conv_attrs: dict[str, Any] = {
            "kernel_shape": kernel_shape,
            "strides": window_strides,
        }

        if isinstance(padding, str):
            pad_mode = padding.upper()
            if pad_mode == "SAME":
                # rely on ONNX automatic padding
                conv_attrs["auto_pad"] = "SAME_UPPER"
            elif pad_mode != "VALID":
                raise ValueError(f"Unsupported padding string: {padding}")
        else:
            # tuple-of-pairs → flat list [t, l, b, r]
            conv_attrs["pads"] = pads

        conv_node = helper.make_node(
            "Conv",
            inputs=[conv_input, transposed_kernel_name],
            outputs=[conv_output],
            name=s.get_unique_name("Conv"),
            **conv_attrs,  # ← the *only* attributes we pass
        )
        s.add_node(conv_node)

        if output_perm:
            # The conv_output is in NCHW format. We need to calculate this shape
            # by transposing the final NHWC shape from JAX.
            nhwc_shape = node_outputs[0].aval.shape
            nchw_shape = (nhwc_shape[0], nhwc_shape[3], nhwc_shape[1], nhwc_shape[2])
            s.add_shape_info(conv_output, nchw_shape)

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
            # No output permutation, so the conv_output shape is the final shape.
            s.add_shape_info(conv_output, node_outputs[0].aval.shape)
            if conv_output != output_name:
                # This happens if the name was already taken, just add an Identity
                s.add_node(
                    helper.make_node(
                        "Identity",
                        inputs=[conv_output],
                        outputs=[output_name],
                        name=s.get_unique_name("Identity_output"),
                    )
                )
        s.add_shape_info(output_name, node_outputs[0].aval.shape)
