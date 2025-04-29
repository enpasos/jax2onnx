# file: jax2onnx/plugins/flax/nnx/conv.py

from collections.abc import Sequence
from typing import TYPE_CHECKING, Callable
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from flax import nnx
from jax import core
from jax.extend.core import Primitive, Literal  # Import Literal from new location
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Define the primitive for convolution operations.
nnx.conv_p = Primitive("nnx.conv")
nnx.conv_p.multiple_results = False  # Correctly set at initialization


@register_primitive(
    jaxpr_primitive=nnx.conv_p.name,
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/linear.html#flax.nnx.Conv",
    onnx=[
        {"component": "Conv", "doc": "https://onnx.ai/onnx/operators/onnx__Conv.html"},
        {
            "component": "Transpose",
            "doc": "https://onnx.ai/onnx/operators/onnx__Transpose.html",
        },
    ],
    since="v0.1.0",
    context="primitives.nnx",
    component="conv",
    testcases=[
        {
            "testcase": "conv_basic_bias",
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
            "testcase": "conv_basic_bias_2",
            "callable": nnx.Conv(1, 32, kernel_size=(3, 3), rngs=nnx.Rngs(0)),
            "input_shapes": [(2, 28, 28, 1)],
        },
        {
            "testcase": "conv_basic_bias_3",
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
            "testcase": "conv_stride2_bias",
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
        {
            "testcase": "conv_no_bias",
            "callable": nnx.Conv(
                in_features=3,
                out_features=16,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
                use_bias=False,
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [("B", 28, 28, 3)],
        },
        {
            "testcase": "conv_valid_padding",
            "callable": nnx.Conv(
                in_features=3,
                out_features=8,
                kernel_size=(5, 5),
                strides=(2, 2),
                padding="VALID",
                use_bias=True,
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [(2, 32, 32, 3)],
        },
        {
            "testcase": "conv_stride1",
            "callable": nnx.Conv(
                in_features=3,
                out_features=8,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
                use_bias=True,
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [(2, 16, 16, 3)],
        },
        {
            "testcase": "conv_stride2",
            "callable": nnx.Conv(
                in_features=3,
                out_features=8,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="SAME",
                use_bias=True,
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [(2, 16, 16, 3)],
        },
        {
            "testcase": "conv_different_kernel",
            "callable": nnx.Conv(
                in_features=3,
                out_features=8,
                kernel_size=(1, 5),
                strides=(1, 1),
                padding="SAME",
                use_bias=True,
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [(2, 16, 16, 3)],
        },
        {
            "testcase": "conv_float64",
            "callable": nnx.Conv(
                in_features=3,
                out_features=8,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
                use_bias=True,
                rngs=nnx.Rngs(0),
                dtype=np.float64,
            ),
            "input_shapes": [(2, 16, 16, 3)],
        },
        {
            "testcase": "conv_single_batch",
            "callable": nnx.Conv(
                in_features=3,
                out_features=8,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
                use_bias=True,
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [(1, 16, 16, 3)],
        },
        {
            "testcase": "conv_large_batch",
            "callable": nnx.Conv(
                in_features=3,
                out_features=8,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
                use_bias=True,
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [(32, 16, 16, 3)],
        },
    ],
)
class ConvPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting flax.nnx.Conv to ONNX.

    Converts a Conv operation into an ONNX Conv operator with Transpose
    operations to handle format conversion.
    """

    @staticmethod
    def _compute_conv_output_shape(
        x_shape: tuple[int, ...],
        kernel_shape: tuple[int, ...],
        strides: Sequence[int] | int,
        padding: str,
    ) -> tuple[int, ...]:
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

    # -----------------------------------------------------------------
    #  abstract_eval – delegate to the *original* Conv implementation
    #                 via **jax.eval_shape**
    # -----------------------------------------------------------------
    # Will be filled once, when we monkey–patch nnx.Conv
    _ORIGINAL_CONV_CALL: Callable | None = None

    @staticmethod
    def abstract_eval(
        x: core.ShapedArray,
        kernel: core.ShapedArray,
        bias: core.ShapedArray,
        strides,
        padding,
        dilations,
        dimension_numbers,
    ):
        """Symbolic-shape rule that mirrors the concatenate example."""

        if ConvPlugin._ORIGINAL_CONV_CALL is None:
            raise RuntimeError("Original nnx.Conv.__call__ not captured.")

        # normalise scalars → tuples (Flax default)
        if isinstance(strides, int):
            strides = (strides, strides)
        if isinstance(dilations, int):
            dilations = (dilations, dilations)

        # Shape-dtype specs
        x_spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        k_spec = jax.ShapeDtypeStruct(kernel.shape, kernel.dtype)
        bias_spec = jax.ShapeDtypeStruct(bias.shape, bias.dtype)

        def _helper(xv, kv, bv):
            """
            Call the *original* nnx.Conv.__call__.
            We fabricate a lightweight dummy-instance that carries all
            attributes the method needs (kernel, bias, strides, …).
            """
            if ConvPlugin._ORIGINAL_CONV_CALL is None:  # safety net
                raise RuntimeError("Original nnx.Conv.__call__ missing.")

            # Extract kernel dimensions - kernel is in HWIO format
            _, _, in_features, out_features = kv.shape
            kernel_height, kernel_width = kv.shape[0], kv.shape[1]
            kernel_size = (kernel_height, kernel_width)

            # Define a promote_dtype function that acts like the original one
            def promote_dtype_func(args, dtype=None):
                # If no dtype specified, just return args unchanged
                if dtype is None:
                    return args
                # Otherwise, we would cast all arrays to dtype - but in abstract evaluation,
                # we just return the arrays as is
                return args

            # Import lax function for conv_general_dilated
            from jax import lax

            dummy = SimpleNamespace(
                # parameters exposed as .value like real nnx.Param
                kernel=SimpleNamespace(value=kv),
                bias=None if bv is None else SimpleNamespace(value=bv),
                # required attributes from the Conv initialization
                kernel_size=kernel_size,  # Needed by nnx.Conv.__call__
                in_features=in_features,
                out_features=out_features,
                # hyper-parameters expected by the method
                strides=strides,
                padding=padding,
                dilations=dilations,
                dimension_numbers=dimension_numbers,
                # sane defaults for rarely-used attrs
                feature_group_count=1,
                input_dilation=1,
                kernel_dilation=1,
                use_bias=bv is not None,
                lhs_dilation=None,
                rhs_dilation=None,
                precision=None,
                # Add missing attributes referenced in Conv.__call__ implementation
                mask=None,  # Missing in original implementation
                kernel_shape=(
                    kernel_height,
                    kernel_width,
                    in_features,
                    out_features,
                ),  # HWIO format
                dtype=None,  # Default to None, as in the original
                # Add the promote_dtype function
                promote_dtype=promote_dtype_func,
                # Add the conv_general_dilated function
                conv_general_dilated=lax.conv_general_dilated,
            )

            return ConvPlugin._ORIGINAL_CONV_CALL(dummy, xv)

        out = jax.eval_shape(_helper, x_spec, k_spec, bias_spec)
        out = jax.tree_util.tree_leaves(out)[0]  # single tensor
        return core.ShapedArray(out.shape, out.dtype)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Convert conv operation to ONNX format."""

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

        # ------------------------------------------------------------------ #
        # Kernel: detect once-for-all constants v. runtime tensors
        # ------------------------------------------------------------------ #
        kernel_name = s.get_name(kernel_var)

        # -----------------------------------------------------------------
        #  Detect compile-time constants *robustly*
        #   – Literal            (jax.extend.core.Literal)
        #   – or   name_to_const entry keyed by the variable *name*
        # -----------------------------------------------------------------
        kernel_const = None
        if isinstance(
            kernel_var, Literal
        ):  # Use imported Literal instead of core.Literal
            kernel_const = np.asarray(kernel_var.val)
        else:
            kernel_const = s.name_to_const.get(kernel_name, None)

        if kernel_const is not None:  # ─ constant ─
            #   JAX supplies kernels in HWIO – transpose here to OIHW and
            #   register the **transposed** tensor as an initializer. No
            #   runtime Transpose node is inserted.
            transposed_kernel = np.transpose(kernel_const, (3, 2, 0, 1))  # OIHW
            weights_name = s.builder.get_constant_name(transposed_kernel)
            filter_shape = kernel_const.shape  # HWIO
        else:  # ─ dynamic ─
            weights_name = s.get_unique_name("kernel_OIHW")
            kernel_transpose_node = helper.make_node(
                "Transpose",
                inputs=[kernel_name],
                outputs=[weights_name],
                name=s.get_unique_name("transpose_kernel"),
                perm=[3, 2, 0, 1],  # HWIO → OIHW
            )
            s.add_node(kernel_transpose_node)
            # Shapes are still static (H,W,I,O) → record both layouts
            k_shape = kernel_var.aval.shape  # (H, W, I, O)
            k_OIHW = (k_shape[3], k_shape[2], k_shape[0], k_shape[1])
            s.add_shape_info(weights_name, k_OIHW, kernel_var.aval.dtype)
            filter_shape = k_shape  # (H,W,I,O)

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
        jax_output_shape = ConvPlugin._compute_conv_output_shape(
            jax_input_shape, filter_shape, strides, padding
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

    @staticmethod
    def _conv(x, kernel, bias, strides, padding, dilations, dimension_numbers):
        return nnx.conv_p.bind(
            x,
            kernel,
            bias,
            strides=strides,
            padding=padding,
            dilations=dilations,
            dimension_numbers=dimension_numbers,
        )

    @staticmethod
    def get_monkey_patch(orig_fn: Callable):
        """
        *Store* the original nnx.Conv.__call__ (for abstract_eval)
        and return a patched replacement that routes through the new
        primitive while auto-filling a zero-bias when `use_bias=False`.
        """
        ConvPlugin._ORIGINAL_CONV_CALL = orig_fn  # <── one-time capture

        import jax.numpy as jnp

        def patched_conv_call(self, x):
            # If bias is None, substitute zeros of correct shape and dtype
            if self.bias is not None:
                bias = self.bias.value
            else:
                # Infer out_features from kernel shape (HWIO)
                out_features = self.kernel.value.shape[-1]
                bias = jnp.zeros((out_features,), dtype=self.kernel.value.dtype)
            return ConvPlugin._conv(
                x,
                self.kernel.value,
                bias,
                self.strides,
                self.padding,
                getattr(self, "dilations", (1, 1)),
                getattr(self, "dimension_numbers", None),
            )

        return patched_conv_call

    @staticmethod
    def patch_info():
        return {
            "patch_targets": [nnx.Conv],
            "patch_function": ConvPlugin.get_monkey_patch,  # ← gets `orig_fn`
            "target_attribute": "__call__",
        }


# Register abstract evaluation function.
nnx.conv_p.def_abstract_eval(ConvPlugin.abstract_eval)
