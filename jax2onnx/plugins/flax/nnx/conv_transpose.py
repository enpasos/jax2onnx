"""
Conv Transpose Plugin for JAX to ONNX conversion.

This plugin enables conversion of flax.nnx.ConvTranspose layers to ONNX format.
It transforms JAX’s conv_transpose operations into an ONNX ConvTranspose operator
with necessary Transpose operations for NHWC to NCHW conversion.
"""

from typing import TYPE_CHECKING

from flax import nnx
from jax import core
from jax.extend.core import Primitive
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

# Define a new primitive for conv transpose.
nnx.conv_transpose_p = Primitive("nnx.conv_transpose")
nnx.conv_transpose_p.multiple_results = False  # Set once at initialization


def _convert_padding(padding):
    if isinstance(padding, str):
        p = padding.upper()
        if p == "VALID":
            return [0, 0, 0, 0]
        elif p == "SAME":
            return [1, 1, 1, 1]
        elif p == "CIRCULAR":
            return [2, 2, 2, 2]
    return padding


@register_primitive(
    jaxpr_primitive=nnx.conv_transpose_p.name,
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/conv_transpose.html",
    onnx=[
        {
            "component": "ConvTranspose",
            "doc": "https://onnx.ai/onnx/operators/onnx__ConvTranspose.html",
        },
    ],
    since="v0.3.0",
    context="primitives.nnx",
    component="conv_transpose",
    testcases=[
        {
            "testcase": "conv_transpose_valid_padding",
            "callable": nnx.ConvTranspose(
                in_features=3,
                out_features=4,
                kernel_size=(3,),
                padding="VALID",
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [(1, 8, 3)],
        },
        {
            "testcase": "conv_transpose_circular_padding",
            "callable": nnx.ConvTranspose(
                in_features=3,
                out_features=4,
                kernel_size=(6, 6),
                strides=(2, 2),
                padding="CIRCULAR",
                transpose_kernel=True,
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [(1, 15, 15, 3)],
        },
    ],
)
class ConvTransposePlugin(PrimitiveLeafPlugin):

    @staticmethod
    def abstract_eval(x, weight, *args, **kwargs):
        pads_raw = kwargs.get("pads", [0, 0, 0, 0])
        is_circular = isinstance(pads_raw, str) and pads_raw.upper() == "CIRCULAR"

        strides = kwargs.get("strides", (1, 1))
        pads = kwargs.get("pads", [0, 0, 0, 0])
        if isinstance(pads, str):
            pads = _convert_padding(pads)
        dilations = kwargs.get("dilations", (1, 1))
        output_padding = kwargs.get("output_padding", (0, 0))

        if len(x.shape) == 3:
            output_shape = ConvTransposePlugin.calculate_output_shape_1d(
                x.shape, weight.shape, strides, pads, dilations, output_padding
            )
        elif len(x.shape) == 4:
            output_shape = ConvTransposePlugin.calculate_output_shape_2d(
                x.shape, weight.shape, strides, pads, dilations, output_padding
            )
            if is_circular:
                # For circular mode, JAX produces an extra singleton dimension after batch.
                output_shape = (output_shape[0], 1) + output_shape[1:]
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        return core.ShapedArray(output_shape, x.dtype)

    @staticmethod
    def calculate_output_shape_2d(
        input_shape, weight_shape, strides, pads, dilations, output_padding
    ):
        batch_size, input_height, input_width, in_channels = input_shape
        kernel_height, kernel_width, _, out_channels = weight_shape

        if isinstance(pads, str):
            p = pads.upper()
            if p == "VALID":
                pads = [0, 0, 0, 0]
            elif p == "SAME":
                pad_along_height = max(
                    (input_height - 1) * strides[0]
                    + dilations[0] * (kernel_height - 1)
                    + 1
                    - input_height,
                    0,
                )
                pad_along_width = max(
                    (input_width - 1) * strides[1]
                    + dilations[1] * (kernel_width - 1)
                    + 1
                    - input_width,
                    0,
                )
                pad_top = pad_along_height // 2
                pad_bottom = pad_along_height - pad_top
                pad_left = pad_along_width // 2
                pad_right = pad_along_width - pad_left
                pads = [pad_top, pad_left, pad_bottom, pad_right]
            elif p == "CIRCULAR":
                pads = [2, 2, 2, 2]
        elif isinstance(pads, int):
            pads = [pads, pads, pads, pads]
        elif isinstance(pads, tuple):
            if len(pads) == 2:
                pads = [pads[0], pads[1], pads[0], pads[1]]
            elif len(pads) == 4:
                pads = list(pads)
            else:
                raise ValueError(
                    "Unsupported pads tuple length for 2D convolution transpose."
                )

        if isinstance(dilations, int):
            dilations = [dilations, dilations]
        if isinstance(output_padding, int):
            output_padding = [output_padding, output_padding]
        if isinstance(strides, int):
            strides = [strides, strides]
        elif strides is None:
            strides = [1, 1]

        output_height = (
            (input_height - 1) * strides[0]
            - pads[0]
            - pads[2]
            + dilations[0] * (kernel_height - 1)
            + output_padding[0]
            + 1
        )
        output_width = (
            (input_width - 1) * strides[1]
            - pads[1]
            - pads[3]
            + dilations[1] * (kernel_width - 1)
            + output_padding[1]
            + 1
        )

        return (batch_size, output_height, output_width, out_channels)

    @staticmethod
    def calculate_output_shape_1d(
        input_shape, weight_shape, strides, pads, dilations, output_padding
    ):
        batch_size, in_channels, seq_len = input_shape
        kernel_size, _, out_channels = weight_shape

        if isinstance(pads, str):
            if pads.upper() == "VALID":
                pads = [0, 0]
            elif pads.upper() == "SAME":
                pad_along_length = max(
                    (seq_len - 1) * strides[0]
                    + dilations[0] * (kernel_size - 1)
                    + 1
                    - seq_len,
                    0,
                )
                pad_l = pad_along_length // 2
                pad_r = pad_along_length - pad_l
                pads = [pad_l, pad_r]
            elif pads.upper() == "CIRCULAR":
                pads = [1, 1]
        elif isinstance(pads, int):
            pads = [pads, pads]
        elif isinstance(pads, tuple):
            if len(pads) == 1:
                pads = [pads[0], pads[0]]
            else:
                pads = [pads[0], pads[1]]

        if isinstance(dilations, int):
            dilations = [dilations]
        if isinstance(output_padding, int):
            output_padding = [output_padding]
        if isinstance(strides, int):
            strides = [strides]
        elif strides is None:
            strides = [1]

        output_length = (
            (seq_len - 1) * strides[0]
            - 2 * pads[0]
            + dilations[0] * (kernel_size - 1)
            + output_padding[0]
            + 1
        )
        return (batch_size, out_channels, output_length)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        input_name = s.get_name(node_inputs[0])
        weight_name = s.get_name(node_inputs[1])
        has_bias = len(node_inputs) > 2
        if has_bias:
            bias_name = s.get_name(node_inputs[2])
        final_output_name = s.get_name(node_outputs[0])

        strides = params.get("strides", (1, 1))
        pads = params.get("pads", [0, 0, 0, 0])
        if isinstance(pads, str):
            pads = _convert_padding(pads)
        dilations = params.get("dilations", (1, 1))
        group = params.get("group", 1)
        output_padding = params.get("output_padding", (0, 0))
        transpose_kernel = params.get("transpose_kernel", False)
        is_circular = (
            isinstance(params.get("pads"), str)
            and params.get("pads").upper() == "CIRCULAR"
        )

        jax_shape = node_inputs[0].aval.shape

        if len(jax_shape) == 4:
            if isinstance(strides, int):
                strides = (strides, strides)

            # Pre-transpose input: NHWC -> NCHW.
            pre_transpose_name = s.get_unique_name("convtrans_pre_transpose")
            pre_transpose_node = helper.make_node(
                "Transpose",
                inputs=[input_name],
                outputs=[pre_transpose_name],
                name=s.get_unique_name("convtrans_pre_transpose_node"),
                perm=[0, 3, 1, 2],
            )
            s.add_node(pre_transpose_node)
            pre_transposed_shape = (
                jax_shape[0],
                jax_shape[3],
                jax_shape[1],
                jax_shape[2],
            )
            s.add_shape_info(
                pre_transpose_name,
                pre_transposed_shape,
                dtype=node_inputs[0].aval.dtype,
            )
            input_to_conv = pre_transpose_name

            # Transpose weight for 2D convolution.
            weight_transpose_name = s.get_unique_name("convtrans_weight_transpose")
            perm = [2, 3, 0, 1] if not transpose_kernel else [3, 2, 0, 1]
            weight_transpose_node = helper.make_node(
                "Transpose",
                inputs=[weight_name],
                outputs=[weight_transpose_name],
                name=s.get_unique_name("convtrans_weight_transpose_node"),
                perm=perm,
            )
            s.add_node(weight_transpose_node)
            weight_shape = node_inputs[1].aval.shape
            transposed_weight_shape = tuple(weight_shape[i] for i in perm)
            s.add_shape_info(
                weight_transpose_name,
                transposed_weight_shape,
                dtype=node_inputs[1].aval.dtype,
            )
            weight_name = weight_transpose_name

            conv_output_name = s.get_unique_name("conv_transpose_output")
            conv_inputs = [input_to_conv, weight_name]
            if has_bias:
                conv_inputs.append(bias_name)

            conv_transpose_node = helper.make_node(
                "ConvTranspose",
                inputs=conv_inputs,
                outputs=[conv_output_name],
                name=s.get_unique_name("conv_transpose"),
                strides=strides,
                pads=pads,
                dilations=dilations,
                group=group,
                output_padding=output_padding,
            )
            s.add_node(conv_transpose_node)

            conv_output_shape = node_outputs[0].aval.shape
            s.add_shape_info(
                conv_output_name, conv_output_shape, dtype=node_outputs[0].aval.dtype
            )

            # Post-transpose output: NCHW -> NHWC.
            post_transpose_node = helper.make_node(
                "Transpose",
                inputs=[conv_output_name],
                outputs=[final_output_name],
                name=s.get_unique_name("convtrans_post_transpose"),
                perm=[0, 2, 3, 1],
            )
            s.add_node(post_transpose_node)

            if is_circular:
                # Insert an Unsqueeze to add the extra singleton dimension.
                unsqueeze_node = helper.make_node(
                    "Unsqueeze",
                    inputs=[final_output_name],
                    outputs=[final_output_name],
                    axes=[1],
                )
                s.add_node(unsqueeze_node)

        elif len(jax_shape) == 3:
            # 1D ConvTranspose branch.
            transpose_name = s.get_unique_name("convtrans_input_transpose")
            transpose_node = helper.make_node(
                "Transpose",
                inputs=[input_name],
                outputs=[transpose_name],
                name=s.get_unique_name("convtrans_input_transpose_node"),
                perm=[0, 2, 1],
            )
            s.add_node(transpose_node)
            transposed_shape = (jax_shape[0], jax_shape[2], jax_shape[1])
            s.add_shape_info(
                transpose_name, transposed_shape, dtype=node_inputs[0].aval.dtype
            )
            input_to_conv = transpose_name

            weight_transpose_name = s.get_unique_name("convtrans_weight_transpose")
            perm = [1, 2, 0] if not transpose_kernel else [2, 1, 0]
            weight_transpose_node = helper.make_node(
                "Transpose",
                inputs=[weight_name],
                outputs=[weight_transpose_name],
                name=s.get_unique_name("convtrans_weight_transpose_node"),
                perm=perm,
            )
            s.add_node(weight_transpose_node)
            weight_shape = node_inputs[1].aval.shape
            transposed_weight_shape = tuple(weight_shape[i] for i in perm)
            s.add_shape_info(
                weight_transpose_name,
                transposed_weight_shape,
                dtype=node_inputs[1].aval.dtype,
            )
            weight_name = weight_transpose_name

            if strides is None:
                strides = (1,)
            elif isinstance(strides, int):
                strides = (strides,)
            if len(strides) == 1:
                strides = (strides[0],)

            conv_output_name = s.get_unique_name("conv_transpose_output")
            conv_inputs = [input_to_conv, weight_name]
            if has_bias:
                conv_inputs.append(bias_name)

            conv_transpose_node = helper.make_node(
                "ConvTranspose",
                inputs=conv_inputs,
                outputs=[conv_output_name],
                name=s.get_unique_name("conv_transpose"),
                strides=strides,
                pads=pads,
                dilations=dilations,
                group=group,
                output_padding=output_padding,
            )
            s.add_node(conv_transpose_node)
            conv_output_shape = node_outputs[0].aval.shape
            s.add_shape_info(
                conv_output_name, conv_output_shape, dtype=node_outputs[0].aval.dtype
            )

            post_transpose_node = helper.make_node(
                "Transpose",
                inputs=[conv_output_name],
                outputs=[final_output_name],
                name=s.get_unique_name("convtrans_output_transpose"),
                perm=[0, 2, 1],
            )
            s.add_node(post_transpose_node)

        else:
            raise ValueError(f"Unsupported input shape for ConvTranspose: {jax_shape}")

    @staticmethod
    def _conv_transpose(
        x,
        weight,
        bias=None,
        strides=(1, 1),
        pads=(0, 0, 0, 0),
        dilations=(1, 1),
        group=1,
        output_padding=(0, 0),
        transpose_kernel=False,
    ):
        nnx.conv_transpose_p.multiple_results = False
        if bias is not None:
            return nnx.conv_transpose_p.bind(
                x,
                weight,
                bias,
                strides=strides,
                pads=pads,
                dilations=dilations,
                group=group,
                output_padding=output_padding,
                transpose_kernel=transpose_kernel,
            )
        else:
            return nnx.conv_transpose_p.bind(
                x,
                weight,
                strides=strides,
                pads=pads,
                dilations=dilations,
                group=group,
                output_padding=output_padding,
                transpose_kernel=transpose_kernel,
            )

    @staticmethod
    def conv_transpose(
        x,
        weight,
        bias,
        strides,
        pads,
        dilations,
        group,
        output_padding,
        transpose_kernel,
    ):
        return ConvTransposePlugin._conv_transpose(
            x,
            weight,
            bias,
            strides,
            pads,
            dilations,
            group,
            output_padding,
            transpose_kernel,
        )

    @staticmethod
    def get_monkey_patch():
        def patched_conv_transpose_call(self, x):
            return ConvTransposePlugin._conv_transpose(
                x,
                self.kernel.value,
                (
                    self.bias.value
                    if hasattr(self, "bias") and self.bias is not None
                    else None
                ),
                strides=self.strides,
                pads=_convert_padding(self.padding),
                dilations=getattr(self, "dilations", (1, 1)),
                group=getattr(self, "group", 1),
                output_padding=getattr(self, "output_padding", (0, 0)),
                transpose_kernel=getattr(self, "transpose_kernel", False),
            )

        return patched_conv_transpose_call

    @staticmethod
    def patch_info():
        return {
            "patch_targets": [nnx.ConvTranspose],
            "patch_function": lambda _: ConvTransposePlugin.get_monkey_patch(),
            "target_attribute": "__call__",
        }


# Register abstract evaluation function.
nnx.conv_transpose_p.def_abstract_eval(ConvTransposePlugin.abstract_eval)
