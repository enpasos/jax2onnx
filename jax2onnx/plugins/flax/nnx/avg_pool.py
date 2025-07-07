# file: jax2onnx/plugins/flax/nnx/avg_pool.py
from typing import TYPE_CHECKING, Callable, Any

from flax import nnx
import jax
from jax import core
from jax.extend.core import Primitive
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


# Define the avg_pool primitive
# Ensure the name matches what's used in patching/binding
avg_pool_p = Primitive("nnx.avg_pool")  # Primitive name reflecting the patched function
avg_pool_p.multiple_results = False


@register_primitive(
    # Use the actual primitive object
    primitive_obj=avg_pool_p,
    # Factory points to the function being patched
    binding_factory=lambda: nnx.avg_pool,
    # Primitive name used in Jaxpr
    jaxpr_primitive=avg_pool_p.name,
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.avg_pool",
    onnx=[
        {
            "component": "AveragePool",
            "doc": "https://onnx.ai/onnx/operators/onnx__AveragePool.html",
        },
        {
            "component": "Transpose",
            "doc": "https://onnx.ai/onnx/operators/onnx__Transpose.html",
        },
    ],
    since="v0.1.0",
    context="primitives.nnx",
    component="avg_pool",
    testcases=[  # --- Keep existing testcases ---
        {
            "testcase": "avg_pool",
            "callable": lambda x: nnx.avg_pool(
                x, window_shape=(2, 2), strides=(2, 2), padding="VALID"
            ),
            "input_shapes": [("B", 32, 32, 3)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "avg_pool_same_padding",
            "callable": lambda x: nnx.avg_pool(
                x, window_shape=(2, 2), strides=(2, 2), padding="SAME"
            ),
            "input_shapes": [("B", 32, 32, 3)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "avg_pool_default_padding",
            "callable": lambda x: nnx.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            "input_shapes": [("B", 32, 32, 3)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "avg_pool_stride1",
            "callable": lambda x: nnx.avg_pool(
                x, window_shape=(2, 2), strides=(1, 1), padding="VALID"
            ),
            "input_shapes": [("B", 8, 8, 3)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "avg_pool_win3x3_stride2",
            "callable": lambda x: nnx.avg_pool(
                x, window_shape=(3, 3), strides=(2, 2), padding="VALID"
            ),
            "input_shapes": [("B", 10, 10, 1)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "avg_pool_stride_none",
            "callable": lambda x: nnx.avg_pool(
                x, window_shape=(2, 2), strides=None, padding="VALID"
            ),
            "input_shapes": [("B", 8, 8, 3)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "avg_pool_count_include_pad_false",
            "callable": lambda x: nnx.avg_pool(
                x,
                window_shape=(2, 2),
                strides=(2, 2),
                padding="SAME",
                count_include_pad=False,
            ),
            "input_shapes": [("B", 8, 8, 3)],
            "run_only_f32_variant": True,
        },
    ],
)
class AvgPoolPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting flax.nnx.avg_pool to ONNX using jax.eval_shape.
    """

    _ORIG_CALL: Callable[..., Any] | None = None

    # ------------------------------------------------------------
    # abstract_eval – delegate to original call via jax.eval_shape
    # ------------------------------------------------------------
    @staticmethod
    def abstract_eval(
        inputs: core.ShapedArray, *, window_shape, strides, padding, count_include_pad
    ):
        """Use jax.eval_shape on the original nnx.avg_pool."""
        if AvgPoolPlugin._ORIG_CALL is None:
            raise RuntimeError("Original nnx.avg_pool not captured.")

        # --- Correctly handle default stride for abstract eval ---
        actual_strides = strides if strides is not None else (1,) * len(window_shape)

        # Helper function to call the original nnx.avg_pool
        def _helper(in_arg):
            # Pass the actual strides JAX would use
            return AvgPoolPlugin._ORIG_CALL(
                in_arg,
                window_shape=window_shape,
                strides=actual_strides,  # Use corrected strides
                padding=padding,
                count_include_pad=count_include_pad,
            )

        # Build ShapeDtypeStruct spec for the input
        spec_inputs = jax.ShapeDtypeStruct(inputs.shape, inputs.dtype)

        # Evaluate the shape using the original function
        out_spec = jax.eval_shape(_helper, spec_inputs)

        # Return the abstract value (ShapedArray)
        return core.ShapedArray(out_spec.shape, out_spec.dtype)

    # ------------------------------------------------------------
    # ONNX Conversion Logic (to_onnx)
    # ------------------------------------------------------------
    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handles conversion of avg_pool primitive to ONNX format."""
        input_var = node_inputs[0]
        output_var = node_outputs[0]

        input_name = s.get_name(input_var)
        final_output_name = s.get_name(output_var)

        window_shape = params.get("window_shape")
        strides_param = params.get("strides")  # Get original param value
        padding = params.get("padding")
        count_include_pad = params.get("count_include_pad")

        jax_input_aval = input_var.aval
        jax_output_aval = output_var.aval

        # --- Correctly determine strides JAX actually used ---
        # (Must match the logic used in abstract_eval and patching)
        actual_strides = (
            strides_param if strides_param is not None else (1,) * len(window_shape)
        )

        # === Pre-Transpose: NHWC -> NCHW ===
        pre_transpose_name = s.get_unique_name(f"{input_name}_pre_transpose")
        pre_transpose_node = helper.make_node(
            "Transpose",
            inputs=[input_name],
            outputs=[pre_transpose_name],
            name=s.get_unique_name("transpose_pre"),
            perm=[0, 3, 1, 2],  # NHWC -> NCHW
        )
        s.add_node(pre_transpose_node)
        # Calculate NCHW shape symbolically using input aval
        pre_transposed_shape = (
            jax_input_aval.shape[0],  # N
            jax_input_aval.shape[3],  # C
            jax_input_aval.shape[1],  # H
            jax_input_aval.shape[2],  # W
        )
        s.add_shape_info(
            pre_transpose_name, pre_transposed_shape, jax_input_aval.dtype
        )  # Use dtype from aval

        # === AveragePool Node in ONNX (operates in NCHW) ===
        pool_out_name = s.get_unique_name("avg_pool_nchw_output")

        # --- Calculate ONNX Padding (Keep using auto_pad) ---
        if padding.upper() == "SAME":
            onnx_auto_pad = "SAME_UPPER"
        elif padding.upper() == "VALID":
            onnx_auto_pad = "VALID"
        else:
            raise NotImplementedError(f"Unsupported padding: {padding}")

        onnx_count_include_pad = 1 if count_include_pad else 0

        avg_pool_node = helper.make_node(
            "AveragePool",
            inputs=[pre_transpose_name],
            outputs=[pool_out_name],
            name=s.get_unique_name("avg_pool"),
            kernel_shape=window_shape,
            strides=actual_strides,  # Use the corrected strides
            auto_pad=onnx_auto_pad,
            count_include_pad=onnx_count_include_pad,
        )
        s.add_node(avg_pool_node)

        # Calculate expected NCHW output shape using output AVAL
        avgpool_output_shape_nchw = (
            jax_output_aval.shape[0],
            jax_output_aval.shape[3],
            jax_output_aval.shape[1],
            jax_output_aval.shape[2],
        )
        s.add_shape_info(
            pool_out_name, avgpool_output_shape_nchw, jax_output_aval.dtype
        )

        # === Post-Transpose: NCHW -> NHWC ===
        post_transpose_node = helper.make_node(
            "Transpose",
            inputs=[pool_out_name],
            outputs=[final_output_name],
            name=s.get_unique_name("transpose_post"),
            perm=[0, 2, 3, 1],  # NCHW -> NHWC
        )
        s.add_node(post_transpose_node)
        # Register final output shape using the output aval directly (already NHWC)
        s.add_shape_info(
            final_output_name, jax_output_aval.shape, jax_output_aval.dtype
        )

    # ------------------------------------------------------------
    # monkey-patch – capture original & inject primitive binding
    # ------------------------------------------------------------
    @staticmethod
    def _avg_pool_binding(inputs, window_shape, strides, padding, count_include_pad):
        """Binds inputs to the avg_pool primitive."""
        # Primitive binding expects the actual args JAX uses
        return avg_pool_p.bind(
            inputs,
            window_shape=window_shape,
            strides=strides,  # Pass the already-corrected strides
            padding=padding,
            count_include_pad=count_include_pad,
        )

    @staticmethod
    def get_monkey_patch(orig_fn: Callable):
        """Returns the patched function that captures the original and binds the primitive."""
        AvgPoolPlugin._ORIG_CALL = orig_fn

        def patched_avg_pool(
            inputs, window_shape, strides=None, padding="VALID", count_include_pad=True
        ):
            # --- Correct default stride handling ---
            actual_strides = (
                strides if strides is not None else (1,) * len(window_shape)
            )
            # --- End correction ---
            return AvgPoolPlugin._avg_pool_binding(
                inputs, window_shape, actual_strides, padding, count_include_pad
            )

        return patched_avg_pool

    @staticmethod
    def patch_info():
        """Provides patching information for nnx.avg_pool."""
        # (Keep as before)
        return {
            "patch_targets": [nnx],
            "target_attribute": "avg_pool",
            "patch_function": AvgPoolPlugin.get_monkey_patch,
        }


# --- Existing registration ---
avg_pool_p.def_abstract_eval(AvgPoolPlugin.abstract_eval)
# --- End Existing registration ---
