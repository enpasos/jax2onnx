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
        {
            "testcase": "conv_1d",
            "callable": nnx.Conv(28, 4, kernel_size=(3,), rngs=nnx.Rngs(0)),
            "input_shapes": [(2, 28, 28)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda model: "Conv"
            in {n.op_type for n in model.graph.node},
        },
        {
            "testcase": "conv_1d_large_kernel",
            "callable": nnx.Conv(
                16, 8, kernel_size=(5,), strides=(2,), rngs=nnx.Rngs(0)
            ),
            "input_shapes": [(4, 32, 16)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda model: "Conv"
            in {n.op_type for n in model.graph.node},
        },
        {
            "testcase": "conv_1d_dilation",
            "callable": nnx.Conv(
                8, 16, kernel_size=(3,), kernel_dilation=(2,), rngs=nnx.Rngs(0)
            ),
            "input_shapes": [(2, 24, 8)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda model: "Conv"
            in {n.op_type for n in model.graph.node},
        },
        {
            "testcase": "conv_1d_stride_dilation",
            "callable": nnx.Conv(
                12,
                6,
                kernel_size=(7,),
                strides=(3,),
                kernel_dilation=(2,),
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [(3, 40, 12)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda model: "Conv"
            in {n.op_type for n in model.graph.node},
        },
        {
            "testcase": "conv_2d_asymmetric_kernel",
            "callable": nnx.Conv(
                4, 8, kernel_size=(2, 5), strides=(1, 2), rngs=nnx.Rngs(0)
            ),
            "input_shapes": [(2, 20, 20, 4)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda model: "Conv"
            in {n.op_type for n in model.graph.node},
        },
        {
            "testcase": "conv_2d_asymmetric_stride",
            "callable": nnx.Conv(
                6, 12, kernel_size=(3, 3), strides=(1, 3), rngs=nnx.Rngs(0)
            ),
            "input_shapes": [(2, 18, 24, 6)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda model: "Conv"
            in {n.op_type for n in model.graph.node},
        },
        {
            "testcase": "conv_2d_asymmetric_dilation",
            "callable": nnx.Conv(
                3, 9, kernel_size=(3, 3), kernel_dilation=(1, 2), rngs=nnx.Rngs(0)
            ),
            "input_shapes": [(2, 16, 16, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda model: "Conv"
            in {n.op_type for n in model.graph.node},
        },
        {
            "testcase": "conv_2d_large_dilation",
            "callable": nnx.Conv(
                8, 16, kernel_size=(3, 3), kernel_dilation=(3, 3), rngs=nnx.Rngs(0)
            ),
            "input_shapes": [(2, 32, 32, 8)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda model: "Conv"
            in {n.op_type for n in model.graph.node},
        },
        {
            "testcase": "conv_2d_large_stride",
            "callable": nnx.Conv(
                4, 8, kernel_size=(5, 5), strides=(4, 4), rngs=nnx.Rngs(0)
            ),
            "input_shapes": [(2, 32, 32, 4)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda model: "Conv"
            in {n.op_type for n in model.graph.node},
        },
        {
            "testcase": "conv_2d_mixed_params",
            "callable": nnx.Conv(
                5,
                10,
                kernel_size=(4, 6),
                strides=(2, 3),
                kernel_dilation=(2, 1),
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [(3, 24, 30, 5)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda model: "Conv"
            in {n.op_type for n in model.graph.node},
        },
        # 3D Convolutions
        {
            "testcase": "conv_3d_basic",
            "callable": nnx.Conv(2, 4, kernel_size=(3, 3, 3), rngs=nnx.Rngs(0)),
            "input_shapes": [(2, 8, 8, 8, 2)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda model: "Conv"
            in {n.op_type for n in model.graph.node},
        },
        {
            "testcase": "conv_3d_stride",
            "callable": nnx.Conv(
                4, 8, kernel_size=(3, 3, 3), strides=(2, 2, 2), rngs=nnx.Rngs(0)
            ),
            "input_shapes": [(2, 16, 16, 16, 4)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda model: "Conv"
            in {n.op_type for n in model.graph.node},
        },
        {
            "testcase": "conv_3d_asymmetric",
            "callable": nnx.Conv(
                3, 6, kernel_size=(2, 3, 4), strides=(1, 2, 1), rngs=nnx.Rngs(0)
            ),
            "input_shapes": [(2, 12, 14, 16, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda model: "Conv"
            in {n.op_type for n in model.graph.node},
        },
        {
            "testcase": "conv_3d_dilation",
            "callable": nnx.Conv(
                2, 4, kernel_size=(3, 3, 3), kernel_dilation=(2, 1, 2), rngs=nnx.Rngs(0)
            ),
            "input_shapes": [(2, 16, 12, 16, 2)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda model: "Conv"
            in {n.op_type for n in model.graph.node},
        },
        {
            "testcase": "conv_2d_small_input",
            "callable": nnx.Conv(1, 4, kernel_size=(2, 2), rngs=nnx.Rngs(0)),
            "input_shapes": [(1, 4, 4, 1)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda model: "Conv"
            in {n.op_type for n in model.graph.node},
        },
        {
            "testcase": "conv_2d_many_channels",
            "callable": nnx.Conv(64, 128, kernel_size=(3, 3), rngs=nnx.Rngs(0)),
            "input_shapes": [(2, 16, 16, 64)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda model: "Conv"
            in {n.op_type for n in model.graph.node},
        },
        {
            "testcase": "conv_1d_wide_input",
            "callable": nnx.Conv(
                8, 16, kernel_size=(7,), strides=(1,), rngs=nnx.Rngs(0)
            ),
            "input_shapes": [(2, 128, 8)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda model: "Conv"
            in {n.op_type for n in model.graph.node},
        },
        {
            "testcase": "conv_2d_kernel_1x1",
            "callable": nnx.Conv(16, 32, kernel_size=(1, 1), rngs=nnx.Rngs(0)),
            "input_shapes": [(4, 14, 14, 16)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda model: "Conv"
            in {n.op_type for n in model.graph.node},
        },
        {
            "testcase": "conv_1d_kernel_1",
            "callable": nnx.Conv(8, 16, kernel_size=(1,), rngs=nnx.Rngs(0)),
            "input_shapes": [(3, 20, 8)],
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
        # Determine number of spatial dimensions from kernel shape
        num_spatial_dims = (
            len(kernel_shape) - 2
        )  # kernel: (spatial..., in_features, out_features)

        # Handle strides for arbitrary dimensions
        if isinstance(strides, int):
            strides_tuple = (strides,) * num_spatial_dims
        else:
            strides_tuple = tuple(strides)

        # Extract batch size and out_channels
        batch_size = x_shape[0]
        out_channels = kernel_shape[-1]

        # Calculate output spatial dimensions
        out_spatial_dims = []
        for i in range(num_spatial_dims):
            spatial_size = x_shape[i + 1]  # Skip batch dimension
            filter_size = kernel_shape[i]
            stride = strides_tuple[i] if i < len(strides_tuple) else 1

            if padding.upper() == "VALID":
                out_size = (spatial_size - filter_size) // stride + 1
            elif padding.upper() == "SAME":
                out_size = -(-spatial_size // stride)  # Ceiling division
            else:
                raise ValueError("Unsupported padding: " + padding)

            out_spatial_dims.append(out_size)

        # Return (batch, spatial..., channels) format
        return (batch_size, *out_spatial_dims, out_channels)

    @staticmethod
    def abstract_eval(
        x: core.ShapedArray,
        kernel: core.ShapedArray,
        bias: core.ShapedArray,
        use_bias: bool,
        strides: Union[int, Tuple[int, ...]],  # Updated to support arbitrary dimensions
        padding: str,
        dilations: Union[
            int, Tuple[int, ...]
        ],  # Updated to support arbitrary dimensions
        dimension_numbers: Any,  # Flax uses str | ConvDimensionNumbers | None
    ):
        if ConvPlugin._ORIGINAL_CONV_CALL is None:
            raise RuntimeError("Original nnx.Conv.__call__ not captured.")

        x_spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        k_spec = jax.ShapeDtypeStruct(kernel.shape, kernel.dtype)
        bias_spec = jax.ShapeDtypeStruct(bias.shape, bias.dtype)

        def _helper(xv, kv, bv):
            if ConvPlugin._ORIGINAL_CONV_CALL is None:
                raise RuntimeError("Original nnx.Conv.__call__ missing.")

            in_features = kv.shape[-2]
            out_features = kv.shape[-1]
            kernel_size_tuple = kv.shape[:-2]

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
                strides=strides,  # Pass the original parameter - let Flax handle broadcasting
                padding=padding,  # Original padding string/sequence
                dimension_numbers=dimension_numbers,
                feature_group_count=1,  # Assuming default
                input_dilation=1,  # Assuming default, Flax nnx.Conv uses input_dilation not lhs_dilation
                kernel_dilation=dilations,  # This is rhs_dilation in lax
                use_bias=(
                    bv is not None and use_bias
                ),  # Check if bias tensor exists and use_bias is true
                lhs_dilation=None,
                rhs_dilation=None,
                precision=None,  # Add this missing attribute
                mask=None,  # Assuming no mask
                kernel_shape=kv.shape,  # Original format (spatial..., in_features, out_features)
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

        # Get shapes and determine dimensionality
        jax_input_shape = input_var.aval.shape
        kernel_shape = kernel_var.aval.shape

        # Determine spatial dimensions
        input_rank = len(jax_input_shape)
        kernel_rank = len(kernel_shape)
        num_spatial_dims = (
            kernel_rank - 2
        )  # kernel: (spatial..., in_features, out_features)

        # Dynamic Pre-Transpose: (batch, spatial..., channels) -> (batch, channels, spatial...)
        # For N-D: [0, input_rank-1, 1, 2, ..., input_rank-2]
        pre_transpose_perm = [0, input_rank - 1] + list(range(1, input_rank - 1))
        pre_transpose_name = s.get_unique_name(f"{input_name}_nchw")
        pre_transpose_node = helper.make_node(
            "Transpose",
            inputs=[input_name],
            outputs=[pre_transpose_name],
            name=s.get_unique_name("transpose_pre"),
            perm=pre_transpose_perm,
        )
        s.add_node(pre_transpose_node)

        # Calculate pre-transpose output shape: (batch, channels, spatial...)
        pre_transpose_shape = [jax_input_shape[0], jax_input_shape[-1]] + list(
            jax_input_shape[1:-1]
        )
        s.add_shape_info(
            pre_transpose_name, tuple(pre_transpose_shape), input_var.aval.dtype
        )

        # Dynamic Kernel Transpose: (spatial..., in_features, out_features) -> (out_features, in_features, spatial...)
        kernel_name = s.get_name(kernel_var)
        kernel_const_val = s.name_to_const.get(kernel_name, None)
        if isinstance(kernel_var, Literal):  # If kernel is a compile-time literal
            kernel_const_val = np.asarray(kernel_var.val)

        onnx_kernel_name: str

        # Kernel: HWIO (Jax) -> OIHW (ONNX)
        # Build kernel transpose permutation: [kernel_rank-1, kernel_rank-2, 0, 1, ..., kernel_rank-3]
        kernel_transpose_perm = [kernel_rank - 1, kernel_rank - 2] + list(
            range(kernel_rank - 2)
        )

        if kernel_const_val is not None:
            kernel_np = np.asarray(kernel_const_val)
            if use_float64 and jnp.issubdtype(kernel_np.dtype, jnp.floating):
                kernel_np = kernel_np.astype(np.float64)
            elif (
                not use_float64 and kernel_np.dtype == np.float64
            ):  # Downcast if needed
                kernel_np = kernel_np.astype(np.float32)

            transposed_kernel_np = np.transpose(kernel_np, kernel_transpose_perm)
            onnx_kernel_name = s.builder.get_constant_name(
                transposed_kernel_np
            )  # This should handle dtype
        else:  # Dynamic kernel
            onnx_kernel_name = s.get_unique_name(f"{kernel_name}_onnx")
            kernel_transpose_node = helper.make_node(
                "Transpose",
                inputs=[kernel_name],
                outputs=[onnx_kernel_name],
                name=s.get_unique_name("transpose_kernel"),
                perm=kernel_transpose_perm,
            )
            s.add_node(kernel_transpose_node)

            # Calculate transposed kernel shape: (out_features, in_features, spatial...)
            transposed_kernel_shape = [kernel_shape[-1], kernel_shape[-2]] + list(
                kernel_shape[:-2]
            )
            s.add_shape_info(
                onnx_kernel_name, tuple(transposed_kernel_shape), kernel_var.aval.dtype
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
                    onnx_kernel_name, tuple(transposed_kernel_shape), jnp.float64
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

        # Convolution parameters - make dimension-agnostic
        strides_param = params.get("strides", 1)
        strides_final = (
            tuple(strides_param)
            if isinstance(strides_param, Sequence)
            else (strides_param,) * num_spatial_dims
        )

        padding_param = params.get("padding", "VALID")  # This is JAX padding string

        dilations_param = params.get("dilations", 1)
        dilations_final = (
            tuple(dilations_param)
            if isinstance(dilations_param, Sequence)
            else (dilations_param,) * num_spatial_dims
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
        # Special handling: ONNX doesn't support SAME padding with dilations > 1
        has_any_dilation = any(d > 1 for d in dilations_final)

        if isinstance(padding_param, str):
            padding_str_upper = padding_param.upper()
            if padding_str_upper == "VALID":
                # cast() keeps mypy quiet – runtime is still a plain str
                conv_attrs["auto_pad"] = cast(Any, "VALID")
            elif padding_str_upper == "SAME":
                if has_any_dilation:
                    # Convert SAME padding to explicit pads when any dilations are used
                    # This is required because ONNX doesn't support auto_pad with dilations > 1
                    onnx_pads = []
                    for i in range(num_spatial_dims):
                        kernel_size = kernel_shape[i]
                        dilation = dilations_final[i]
                        input_size = jax_input_shape[i + 1]  # Skip batch dimension
                        stride = strides_final[i]

                        # Calculate effective kernel size with dilation
                        effective_kernel = kernel_size + (kernel_size - 1) * (
                            dilation - 1
                        )

                        # For SAME padding: pad such that output_size = ceil(input_size / stride)
                        # output_size = (input_size + pad_total - effective_kernel) / stride + 1
                        # Solving for pad_total: pad_total = (output_size - 1) * stride + effective_kernel - input_size
                        output_size = -(-input_size // stride)  # Ceiling division
                        pad_total = (
                            (output_size - 1) * stride + effective_kernel - input_size
                        )

                        # Ensure non-negative padding
                        pad_total = max(0, pad_total)

                        # Distribute padding (prefer padding at beginning for odd totals)
                        pad_begin = pad_total // 2
                        pad_end = pad_total - pad_begin
                        onnx_pads.append(pad_begin)

                    # Add the end pads
                    for i in range(num_spatial_dims):
                        kernel_size = kernel_shape[i]
                        dilation = dilations_final[i]
                        input_size = jax_input_shape[i + 1]  # Skip batch dimension
                        stride = strides_final[i]

                        # Calculate effective kernel size with dilation
                        effective_kernel = kernel_size + (kernel_size - 1) * (
                            dilation - 1
                        )

                        # For SAME padding: pad such that output_size = ceil(input_size / stride)
                        output_size = -(-input_size // stride)  # Ceiling division
                        pad_total = (
                            (output_size - 1) * stride + effective_kernel - input_size
                        )

                        # Ensure non-negative padding
                        pad_total = max(0, pad_total)

                        # Distribute padding (prefer padding at beginning for odd totals)
                        pad_begin = pad_total // 2
                        pad_end = pad_total - pad_begin
                        onnx_pads.append(pad_end)

                    conv_attrs["pads"] = onnx_pads
                    logger.info(
                        f"ConvPlugin: Converted SAME padding to explicit pads {onnx_pads} for input_shape={jax_input_shape}, "
                        f"kernel_shape={kernel_shape}, strides={strides_final}, dilations={dilations_final}."
                    )
                else:
                    conv_attrs["auto_pad"] = cast(Any, "SAME_UPPER")  # Or SAME_LOWER
            else:
                logger.warning(
                    f"ConvPlugin: Received unhandled JAX padding '{padding_param}'. Defaulting to ONNX VALID padding."
                )
                conv_attrs["auto_pad"] = cast(Any, "VALID")
        elif (
            isinstance(padding_param, Sequence)
            and len(padding_param) == num_spatial_dims
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
        final_jax_output_dtype = node_outputs[0].aval.dtype

        # Calculate intermediate ONNX conv output shape: (batch, channels, spatial...)
        output_rank = len(final_jax_output_shape)
        conv_out_nchw_shape = [
            final_jax_output_shape[0],
            final_jax_output_shape[-1],
        ] + list(final_jax_output_shape[1:-1])
        s.add_shape_info(
            conv_out_nchw_name, tuple(conv_out_nchw_shape), final_jax_output_dtype
        )

        # Dynamic Post-Transpose: (batch, channels, spatial...) -> (batch, spatial..., channels)
        # For N-D: [0, 2, 3, ..., output_rank-1, 1]
        post_transpose_perm = [0] + list(range(2, output_rank)) + [1]
        post_transpose_node = helper.make_node(
            "Transpose",
            inputs=[conv_out_nchw_name],
            outputs=[final_output_name],
            name=s.get_unique_name("transpose_post"),
            perm=post_transpose_perm,
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
        # Determine spatial dimensions from kernel shape
        num_spatial_dims = (
            len(kernel.shape) - 2
        )  # kernel: (spatial..., in_features, out_features)

        # Handle strides and dilations for arbitrary dimensions
        if isinstance(strides, int):
            strides_arg: Tuple[int, ...] = (strides,) * num_spatial_dims
        else:
            strides_arg = tuple(strides)

        if isinstance(dilations, int):
            dilations_arg: Tuple[int, ...] = (dilations,) * num_spatial_dims
        else:
            dilations_arg = tuple(dilations)

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
                getattr(self, "kernel_dilation", 1),
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
