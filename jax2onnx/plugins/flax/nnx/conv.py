# file: jax2onnx/plugins/flax/nnx/conv.py

from collections.abc import Sequence
from typing import TYPE_CHECKING, Callable, Union, Tuple, Any, cast  # Added cast
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from flax import nnx
from jax import core
from jax.extend.core import Primitive, Literal
from onnx import helper
import logging  # Added logging

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

    # For type hinting nnx.Conv if not directly importable for isinstance checks
    # However, direct import of nnx.Conv should be fine.

logger = logging.getLogger("jax2onnx.plugins.flax.nnx.conv")


# Define the primitive for convolution operations.
# Ensure nnx.conv_p is defined or handled appropriately if it's dynamically created
if not hasattr(nnx, "conv_p"):
    nnx.conv_p = Primitive("nnx.conv")  # type: ignore
    nnx.conv_p.multiple_results = False


@register_primitive(
    jaxpr_primitive=nnx.conv_p.name,  # type: ignore
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
            "run_only_f32_variant": True,
            # ADDED: This lambda will be executed by the test generator.
            # It asserts that a "Conv" op exists in the generated graph.
            "post_check_onnx_graph": lambda model: "Conv"
            in {n.op_type for n in model.graph.node},
        },
        {
            "testcase": "conv_basic_bias_2",
            "callable": nnx.Conv(1, 32, kernel_size=(3, 3), rngs=nnx.Rngs(0)),
            "input_shapes": [(2, 28, 28, 1)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda model: "Conv"
            in {n.op_type for n in model.graph.node},
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
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda model: "Conv"
            in {n.op_type for n in model.graph.node},
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
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda model: "Conv"
            in {n.op_type for n in model.graph.node},
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
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda model: "Conv"
            in {n.op_type for n in model.graph.node},
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
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda model: "Conv"
            in {n.op_type for n in model.graph.node},
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
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda model: "Conv"
            in {n.op_type for n in model.graph.node},
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
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda model: "Conv"
            in {n.op_type for n in model.graph.node},
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
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda model: "Conv"
            in {n.op_type for n in model.graph.node},
        },
        {
            "testcase": "conv_float64",  # This test case explicitly initializes Conv with float64 params
            "callable": nnx.Conv(
                in_features=3,
                out_features=8,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
                use_bias=True,
                rngs=nnx.Rngs(0),
                dtype=np.float64,  # Parameters will be float64
            ),
            "input_shapes": [(2, 16, 16, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda model: "Conv"
            in {n.op_type for n in model.graph.node},
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
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda model: "Conv"
            in {n.op_type for n in model.graph.node},
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
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda model: "Conv"
            in {n.op_type for n in model.graph.node},
        },
    ],
)
class ConvPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting flax.nnx.Conv to ONNX.
    """

    _ORIGINAL_CONV_CALL: Callable | None = None

    @staticmethod
    def _compute_conv_output_shape(
        x_shape: tuple[int, ...],
        kernel_shape: tuple[int, ...],
        strides: Union[Sequence[int], int],
        padding: str,
    ) -> tuple[int, ...]:
        if isinstance(strides, int):
            strides_tuple: tuple[int, int] = (strides, strides)
        else:
            # Explicitly create a new tuple with exactly two elements
            strides_list = list(strides)
            strides_tuple = (strides_list[0], strides_list[1])
        N, H, W, _ = x_shape
        filter_height, filter_width, _, out_channels = kernel_shape
        if padding.upper() == "VALID":
            out_H = (H - filter_height) // strides_tuple[0] + 1
            out_W = (W - filter_width) // strides_tuple[1] + 1
        elif padding.upper() == "SAME":
            out_H = -(-H // strides_tuple[0])
            out_W = -(-W // strides_tuple[1])
        else:
            raise ValueError("Unsupported padding: " + padding)
        return (N, out_H, out_W, out_channels)

    @staticmethod
    def abstract_eval(
        x: core.ShapedArray,
        kernel: core.ShapedArray,
        bias: core.ShapedArray,
        use_bias: bool,
        strides: Union[int, Tuple[int, int]],  # Flax uses Sequence[int] | int
        padding: str,  # Flax uses str | Sequence[Tuple[int, int]]
        dilations: Union[int, Tuple[int, int]],  # Flax uses Sequence[int] | int
        dimension_numbers: Any,  # Flax uses str | ConvDimensionNumbers | None
    ):
        if ConvPlugin._ORIGINAL_CONV_CALL is None:
            raise RuntimeError("Original nnx.Conv.__call__ not captured.")

        # Guarantee `Tuple[int, int]` ⇢ silence "tuple[int, …]" complaints
        strides_tuple: tuple[int, int]
        if isinstance(strides, int):
            strides_tuple = (strides, strides)
        else:
            # Extract exactly two elements to ensure tuple[int, int]
            strides_list = list(strides)
            strides_tuple = (strides_list[0], strides_list[1])

        dilations_tuple: tuple[int, int]
        if isinstance(dilations, int):
            dilations_tuple = (dilations, dilations)
        else:
            # Extract exactly two elements to ensure tuple[int, int]
            dilations_list = list(dilations)
            dilations_tuple = (dilations_list[0], dilations_list[1])

        x_spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        k_spec = jax.ShapeDtypeStruct(kernel.shape, kernel.dtype)
        bias_spec = jax.ShapeDtypeStruct(bias.shape, bias.dtype)

        def _helper(xv, kv, bv):
            if ConvPlugin._ORIGINAL_CONV_CALL is None:
                raise RuntimeError("Original nnx.Conv.__call__ missing.")

            _, _, in_features, out_features = kv.shape
            kernel_height, kernel_width = kv.shape[0], kv.shape[1]
            kernel_size_tuple = (kernel_height, kernel_width)

            def promote_dtype_func(*args, dtype=None):  # type: ignore
                # Simplified for abstract_eval: assume dtypes are already correct or handled by JAX
                # In a real scenario, this would cast arrays if dtype is specified.
                # For eval_shape, we primarily care about shapes.
                if len(args) == 1 and isinstance(args[0], (list, tuple)):
                    return args[0]  # Return the sequence of arrays
                return args  # Return single array or sequence

            # Create a dummy instance that mimics nnx.Conv for the original __call__
            dummy_conv_instance = SimpleNamespace(
                kernel=SimpleNamespace(value=kv),
                bias=(
                    SimpleNamespace(value=bv) if bv is not None else None
                ),  # Bias might be None if use_bias is False
                kernel_size=kernel_size_tuple,
                in_features=in_features,
                out_features=out_features,
                strides=strides_tuple,
                padding=padding,  # Original padding string/sequence
                dilations=dilations_tuple,
                dimension_numbers=dimension_numbers,
                feature_group_count=1,  # Assuming default
                input_dilation=1,  # Assuming default, Flax nnx.Conv uses input_dilation not lhs_dilation
                kernel_dilation=dilations_tuple,  # This is rhs_dilation in lax
                use_bias=(
                    bv is not None and use_bias
                ),  # Check if bias tensor exists and use_bias is true
                lhs_dilation=None,
                rhs_dilation=None,
                precision=None,  # Add this missing attribute
                mask=None,  # Assuming no mask
                kernel_shape=kv.shape,  # HWIO format
                dtype=x.dtype,  # Set the dummy's dtype to match the input's dtype for promote_dtype
                param_dtype=kv.dtype,  # Set param_dtype to kernel's dtype
                promote_dtype=promote_dtype_func,  # Use the simplified one
                conv_general_dilated=lax.conv_general_dilated,
                # Flax nnx.Conv.__call__ might reference other attributes not listed here.
            )
            return ConvPlugin._ORIGINAL_CONV_CALL(dummy_conv_instance, xv)

        out_sds = jax.eval_shape(_helper, x_spec, k_spec, bias_spec)
        if isinstance(out_sds, (list, tuple)):  # Should be single tensor for Conv
            out_sds = out_sds[0]
        return core.ShapedArray(out_sds.shape, out_sds.dtype)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        input_var = node_inputs[0]
        kernel_var = node_inputs[1]
        bias_var = (
            node_inputs[2]
            if params.get("use_bias", False) and len(node_inputs) > 2
            else None
        )

        use_float64 = getattr(s.builder, "enable_double_precision", False)
        # Dtype for ONNX constants/initializers if they need to be created/cast
        # Expected JAX dtype for float operations if enable_double_precision is true

        input_name = s.get_name(input_var)
        final_output_name = s.get_name(node_outputs[0])

        # Pre-Transpose: NHWC (JAX) -> NCHW (ONNX)
        pre_transpose_name = s.get_unique_name(f"{input_name}_nchw")
        pre_transpose_node = helper.make_node(
            "Transpose",
            inputs=[input_name],
            outputs=[pre_transpose_name],
            name=s.get_unique_name("transpose_pre"),
            perm=[0, 3, 1, 2],
        )
        s.add_node(pre_transpose_node)
        jax_input_shape = input_var.aval.shape
        s.add_shape_info(
            pre_transpose_name,
            (
                jax_input_shape[0],
                jax_input_shape[3],
                jax_input_shape[1],
                jax_input_shape[2],
            ),
            input_var.aval.dtype,
        )

        # Kernel: HWIO (JAX) -> OIHW (ONNX)
        kernel_name = s.get_name(kernel_var)
        kernel_const_val = s.name_to_const.get(kernel_name, None)
        if isinstance(kernel_var, Literal):  # If kernel is a compile-time literal
            kernel_const_val = np.asarray(kernel_var.val)

        onnx_kernel_name: str
        kernel_hwio_shape = kernel_var.aval.shape

        if kernel_const_val is not None:
            kernel_np = np.asarray(kernel_const_val)
            if use_float64 and jnp.issubdtype(kernel_np.dtype, jnp.floating):
                kernel_np = kernel_np.astype(np.float64)
            elif (
                not use_float64 and kernel_np.dtype == np.float64
            ):  # Downcast if needed
                kernel_np = kernel_np.astype(np.float32)

            transposed_kernel_np = np.transpose(kernel_np, (3, 2, 0, 1))  # OIHW
            onnx_kernel_name = s.builder.get_constant_name(
                transposed_kernel_np
            )  # This should handle dtype
        else:  # Dynamic kernel
            onnx_kernel_name = s.get_unique_name(f"{kernel_name}_oihw")
            kernel_transpose_node = helper.make_node(
                "Transpose",
                inputs=[kernel_name],
                outputs=[onnx_kernel_name],
                name=s.get_unique_name("transpose_kernel"),
                perm=[3, 2, 0, 1],  # HWIO -> OIHW
            )
            s.add_node(kernel_transpose_node)
            s.add_shape_info(
                onnx_kernel_name,
                (
                    kernel_hwio_shape[3],
                    kernel_hwio_shape[2],
                    kernel_hwio_shape[0],
                    kernel_hwio_shape[1],
                ),
                kernel_var.aval.dtype,
            )
            # If kernel_var.aval.dtype is float32 and use_float64 is true, a Cast might be needed on onnx_kernel_name
            if kernel_var.aval.dtype == jnp.float32 and use_float64:
                casted_kernel_name = s.get_unique_name(f"{onnx_kernel_name}_f64")
                s.add_node(
                    helper.make_node(
                        "Cast",
                        inputs=[onnx_kernel_name],
                        outputs=[casted_kernel_name],
                        to=helper.TensorProto.DOUBLE,
                    )
                )
                onnx_kernel_name = casted_kernel_name
                s.add_shape_info(
                    onnx_kernel_name,
                    (
                        kernel_hwio_shape[3],
                        kernel_hwio_shape[2],
                        kernel_hwio_shape[0],
                        kernel_hwio_shape[1],
                    ),
                    jnp.float64,
                )

        # Bias
        onnx_bias_name: str | None = None
        if bias_var is not None:
            bias_name_orig = s.get_name(bias_var)
            bias_const_val = s.name_to_const.get(bias_name_orig, None)
            if isinstance(bias_var, Literal):
                bias_const_val = np.asarray(bias_var.val)

            if bias_const_val is not None:
                bias_np = np.asarray(bias_const_val)
                if use_float64 and jnp.issubdtype(bias_np.dtype, jnp.floating):
                    bias_np = bias_np.astype(np.float64)
                elif not use_float64 and bias_np.dtype == np.float64:
                    bias_np = bias_np.astype(np.float32)
                onnx_bias_name = s.builder.get_constant_name(bias_np)
            else:  # Dynamic bias
                onnx_bias_name = bias_name_orig
                if bias_var.aval.dtype == jnp.float32 and use_float64:
                    casted_bias_name = s.get_unique_name(f"{onnx_bias_name}_f64")
                    s.add_node(
                        helper.make_node(
                            "Cast",
                            inputs=[onnx_bias_name],
                            outputs=[casted_bias_name],
                            to=helper.TensorProto.DOUBLE,
                        )
                    )
                    onnx_bias_name = casted_bias_name
                    s.add_shape_info(onnx_bias_name, bias_var.aval.shape, jnp.float64)

        # Convolution parameters
        strides_param = params.get("strides", (1, 1))
        strides_final = (
            tuple(strides_param)
            if isinstance(strides_param, Sequence)
            else (strides_param, strides_param)
        )

        padding_param = params.get("padding", "VALID")  # This is JAX padding string

        dilations_param = params.get("dilations", (1, 1))
        dilations_final = (
            tuple(dilations_param)
            if isinstance(dilations_param, Sequence)
            else (dilations_param, dilations_param)
        )

        conv_inputs = [pre_transpose_name, onnx_kernel_name]
        if onnx_bias_name:
            conv_inputs.append(onnx_bias_name)

        conv_out_nchw_name = s.get_unique_name("conv_out_nchw")
        conv_attrs: dict[str, Any] = {
            "strides": list(strides_final),
            "dilations": list(dilations_final),
        }

        # Handle ONNX padding attribute based on JAX padding string
        if isinstance(padding_param, str):
            padding_str_upper = padding_param.upper()
            if padding_str_upper == "VALID":
                # cast() keeps mypy quiet – runtime is still a plain str
                conv_attrs["auto_pad"] = cast(Any, "VALID")
            elif padding_str_upper == "SAME":
                conv_attrs["auto_pad"] = cast(Any, "SAME_UPPER")  # Or SAME_LOWER
            else:
                logger.warning(
                    f"ConvPlugin: Received unhandled JAX padding '{padding_param}'. Defaulting to ONNX VALID padding."
                )
                conv_attrs["auto_pad"] = cast(Any, "VALID")
        elif (
            isinstance(padding_param, Sequence)
            and len(padding_param) == len(kernel_hwio_shape) - 2
        ):  # Spatial dims
            # JAX padding: Sequence of (low, high) pairs for each spatial dimension
            # ONNX padding: [x1_begin, x2_begin,...,x1_end, x2_end,...]
            # Example: JAX ((pad_h_low, pad_h_high), (pad_w_low, pad_w_high)) for 2D
            # ONNX [pad_h_low, pad_w_low, pad_h_high, pad_w_high]
            onnx_pads = []
            for i in range(len(padding_param)):  # Iterate spatial dimensions
                onnx_pads.append(padding_param[i][0])  # low pads
            for i in range(len(padding_param)):
                onnx_pads.append(padding_param[i][1])  # high pads
            conv_attrs["pads"] = onnx_pads
        else:
            logger.error(f"ConvPlugin: Unrecognized padding format: {padding_param}")
            # Fallback or raise error
            conv_attrs["auto_pad"] = cast(Any, "VALID")

        conv_node = helper.make_node(
            "Conv",
            inputs=conv_inputs,
            outputs=[conv_out_nchw_name],
            name=s.get_unique_name("conv"),
            **conv_attrs,
        )
        s.add_node(conv_node)

        # Output shape calculation (NHWC for JAX, then transpose to NCHW for ONNX Conv output)
        # The abstract_eval of nnx.conv_p should give the correct JAX output shape.
        # We can rely on the output_var.aval from the jaxpr for the final NHWC shape.
        final_jax_output_shape = node_outputs[0].aval.shape
        final_jax_output_dtype = node_outputs[
            0
        ].aval.dtype  # This should be target_jax_dtype

        # Intermediate NCHW shape for conv_out_nchw_name
        conv_out_nchw_shape = (
            final_jax_output_shape[0],
            final_jax_output_shape[3],
            final_jax_output_shape[1],
            final_jax_output_shape[2],
        )
        s.add_shape_info(
            conv_out_nchw_name, conv_out_nchw_shape, final_jax_output_dtype
        )

        # Post-Transpose: NCHW (ONNX Conv output) -> NHWC (JAX convention)
        post_transpose_node = helper.make_node(
            "Transpose",
            inputs=[conv_out_nchw_name],
            outputs=[final_output_name],
            name=s.get_unique_name("transpose_post"),
            perm=[0, 2, 3, 1],
        )
        s.add_node(post_transpose_node)
        s.add_shape_info(
            final_output_name, final_jax_output_shape, final_jax_output_dtype
        )

    @staticmethod
    def _conv(
        x: jax.Array,
        kernel: jax.Array,
        bias: jax.Array,
        use_bias: bool,
        strides: Union[int, Tuple[int, ...]],
        padding: Union[str, Sequence[Tuple[int, int]]],  # JAX padding can be complex
        dilations: Union[int, Tuple[int, ...]],
        dimension_numbers: Any,
    ):
        # Fix these type errors by explicitly constructing a new tuple with the precise type
        strides_arg: Tuple[int, ...] = (
            (strides, strides) if isinstance(strides, int) else tuple(strides)
        )
        dilations_arg: Tuple[int, ...] = (
            (dilations, dilations) if isinstance(dilations, int) else tuple(dilations)
        )

        # This binds to the nnx.conv_p primitive.
        # The dtypes of x, kernel, bias must match here.
        return nnx.conv_p.bind(  # type: ignore
            x,
            kernel,
            bias,
            use_bias=use_bias,
            strides=strides_arg,  # Pass the correctly typed tuple
            padding=padding,  # Pass JAX padding directly
            dilations=dilations_arg,  # Pass the correctly typed tuple
            dimension_numbers=dimension_numbers,
        )

    @staticmethod
    def get_monkey_patch(orig_fn: Callable):
        ConvPlugin._ORIGINAL_CONV_CALL = orig_fn

        def patched_conv_call(
            self: nnx.Conv, x: jax.Array
        ):  # self is the nnx.Conv instance
            # Determine the target JAX dtype based on JAX's x64 config,
            # which is set by to_onnx(..., enable_double_precision=True)
            target_jax_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

            # Ensure kernel is of target_jax_dtype if it's a float
            current_kernel_val = self.kernel.value
            if (
                jnp.issubdtype(current_kernel_val.dtype, jnp.floating)
                and current_kernel_val.dtype != target_jax_dtype
            ):
                logger.debug(
                    f"ConvPlugin (patched_conv_call): Casting kernel from {current_kernel_val.dtype} to {target_jax_dtype} for primitive binding."
                )
                kernel_to_bind = jnp.asarray(current_kernel_val, dtype=target_jax_dtype)
            else:
                kernel_to_bind = current_kernel_val

            # Ensure bias is of target_jax_dtype if it's a float
            bias_to_bind: jax.Array
            if self.use_bias:
                if (
                    self.bias is not None and self.bias.value is not None
                ):  # Check if bias Param exists and has a value
                    current_bias_val = self.bias.value
                    if (
                        jnp.issubdtype(current_bias_val.dtype, jnp.floating)
                        and current_bias_val.dtype != target_jax_dtype
                    ):
                        logger.debug(
                            f"ConvPlugin (patched_conv_call): Casting bias from {current_bias_val.dtype} to {target_jax_dtype} for primitive binding."
                        )
                        bias_to_bind = jnp.asarray(
                            current_bias_val, dtype=target_jax_dtype
                        )
                    else:
                        bias_to_bind = current_bias_val
                else:  # use_bias is True, but self.bias.value is None (e.g. uninitialized or explicitly set to None)
                    # This case should ideally be handled by nnx.Conv's init logic if bias is expected.
                    # For safety, create a zero bias of the correct type.
                    out_features = kernel_to_bind.shape[-1]
                    logger.warning(
                        f"ConvPlugin (patched_conv_call): use_bias is True but self.bias.value is None. Creating zero bias with dtype {target_jax_dtype}."
                    )
                    bias_to_bind = jnp.zeros((out_features,), dtype=target_jax_dtype)
            else:  # Not using bias (self.use_bias is False)
                # The nnx.conv_p primitive still expects a bias argument.
                # Pass a dummy zero bias of the target_jax_dtype.
                out_features = kernel_to_bind.shape[-1]
                bias_to_bind = jnp.zeros((out_features,), dtype=target_jax_dtype)

            # Also, if the nnx.Conv instance itself has a 'dtype' attribute that influences
            # its internal promote_dtype behavior, we might need to temporarily adjust it.
            # For nnx.Conv, its own `self.dtype` and `self.param_dtype` are primarily for initialization.
            # The key is that the arguments to `lax.conv_general_dilated` (called by original __call__) must match.
            # The `promote_dtype` method within the original `__call__` should now work correctly
            # if `x`, `kernel_to_bind`, and `bias_to_bind` are already consistent or if `self.dtype`
            # (on the dummy instance in abstract_eval, or on the real `self` here) guides it.
            # The critical part is that `kernel_to_bind` and `bias_to_bind` are now aligned with `x`'s potential float64 type.

            return ConvPlugin._conv(
                x,  # x will be float64 if jax_enable_x64 is True
                kernel_to_bind,
                bias_to_bind,
                self.use_bias,
                self.strides,
                self.padding,  # Pass original JAX padding
                getattr(self, "dilations", (1, 1)),
                getattr(
                    self, "dimension_numbers", None
                ),  # Pass original dimension_numbers
            )

        return patched_conv_call

    @staticmethod
    def patch_info():
        return {
            "patch_targets": [nnx.Conv],
            "patch_function": ConvPlugin.get_monkey_patch,
            "target_attribute": "__call__",
        }


# Register abstract evaluation function.
nnx.conv_p.def_abstract_eval(ConvPlugin.abstract_eval)
