# file: jax2onnx/plugins/flax/nnx/linear_general.py
"""
Linear General Plugin for JAX to ONNX conversion.

This plugin enables conversion of flax.nnx.LinearGeneral layers to ONNX format.
It transforms JAX's linear_general operations (a specialized dot_general for linear layers)
into an ONNX Gemm operator with necessary Reshape operations.

The conversion process involves:
  1. Calculating the output shape and the reshaping parameters.
  2. Providing an abstract evaluation for JAX's tracing system.
  3. Converting the operation to ONNX using Gemm and Reshape nodes.
  4. Monkey-patching LinearGeneral.__call__ to redirect calls to our primitive.
"""

from typing import TYPE_CHECKING, Callable
from types import SimpleNamespace

import jax
import numpy as np
from flax import nnx
from jax import core
from jax.extend.core import Primitive
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Define the primitive for linear_general operations.
nnx.linear_general_p = Primitive("nnx.linear_general")
nnx.linear_general_p.multiple_results = False

# ---------------------------------------------------------
#  We keep a reference to the *unpatched* __call__
# ---------------------------------------------------------
_ORIGINAL_LG_CALL: Callable | None = None


@register_primitive(
    jaxpr_primitive=nnx.linear_general_p.name,
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/linear.html#flax.nnx.LinearGeneral",
    onnx=[
        {"component": "Gemm", "doc": "https://onnx.ai/onnx/operators/onnx__Gemm.html"},
        {
            "component": "Reshape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
        },
    ],
    since="v0.1.0",
    context="primitives.nnx",
    component="linear_general",
    testcases=[
        {
            "testcase": "linear_general",
            "callable": nnx.LinearGeneral(
                in_features=(8, 32),
                out_features=(256,),
                axis=(-2, -1),
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [("B", 4, 8, 32)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "linear_general_2",
            "callable": nnx.LinearGeneral(
                in_features=(30,),
                out_features=(20,),
                axis=(-1,),
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [(3, 30)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "linear_general_3",
            "callable": nnx.LinearGeneral(
                in_features=(256,),
                out_features=(8, 32),
                axis=(-1,),
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [(2, 4, 256)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "linear_general_4",
            "callable": nnx.LinearGeneral(
                in_features=(8, 32),
                out_features=(256,),
                axis=(-2, -1),
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [(2, 4, 8, 32)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "linear_general_abstract_eval_axes",
            "callable": nnx.LinearGeneral(
                in_features=(256,),
                out_features=(8, 32),
                axis=(-1,),
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [(3, 10, 256)],
            "expected_output_shape": (3, 10, 8, 32),
            "run_only_f32_variant": True,
        },
        {
            "testcase": "linear_general_abstract_eval_axes_pair",
            "callable": nnx.LinearGeneral(
                in_features=(8, 32),
                out_features=(256,),
                axis=(-2, -1),
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [(3, 10, 8, 32)],
            "expected_output_shape": (3, 10, 256),
            "run_only_f32_variant": True,
        },
    ],
)
class LinearGeneralPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting flax.nnx.LinearGeneral to ONNX.

    Converts a LinearGeneral operation into a Gemm (matrix multiplication)
    followed by a Reshape to recover the desired output shape.
    """

    @staticmethod
    def _normalize_contracting_dims(dimension_numbers, x_shape, kernel_shape):
        # Unpack and normalize contracting dimensions to positive indices.
        ((lhs_contract, rhs_contract), _) = dimension_numbers
        lhs_contract = [d % len(x_shape) for d in lhs_contract]
        rhs_contract = [d % len(kernel_shape) for d in rhs_contract]
        return lhs_contract, rhs_contract

    @staticmethod
    def _compute_batch_and_kernel_output_dims(
        x_shape, kernel_shape, lhs_contract, rhs_contract
    ):
        # Compute sizes for batch dimensions from input and non-contracted (output) dimensions from kernel.
        x_batch_dims = [i for i in range(len(x_shape)) if i not in lhs_contract]
        x_batch_dims_sizes = [x_shape[i] for i in x_batch_dims]
        kernel_noncontract_dims = [
            i for i in range(len(kernel_shape)) if i not in rhs_contract
        ]
        kernel_out_dims = [kernel_shape[i] for i in kernel_noncontract_dims]
        return x_batch_dims_sizes, kernel_out_dims

    @staticmethod
    def _shape_linear_general(x_shape, kernel_shape, dimension_numbers):
        """Calculate all reshaping parameters for the Gemm transformation."""
        lhs_contract, rhs_contract = LinearGeneralPlugin._normalize_contracting_dims(
            dimension_numbers, x_shape, kernel_shape
        )
        x_batch_dims_sizes, kernel_out_dims = (
            LinearGeneralPlugin._compute_batch_and_kernel_output_dims(
                x_shape, kernel_shape, lhs_contract, rhs_contract
            )
        )

        # Create output shape correctly handling symbolic dimensions
        output_shape = tuple(x_batch_dims_sizes + kernel_out_dims)

        # Handle kernel dimensions safely - these are generally fixed sizes
        kernel_contract_dims = [kernel_shape[i] for i in rhs_contract]
        kernel_contract_size = 1
        for dim in kernel_contract_dims:
            kernel_contract_size *= dim

        kernel_out_size = 1
        for dim in kernel_out_dims:
            kernel_out_size *= dim

        new_kernel_dims_sizes = (kernel_contract_size, kernel_out_size)

        # Handle input dimensions with care for symbolic dimensions
        # Batch dimensions
        has_symbolic_batch = any(
            not isinstance(dim, (int, float)) for dim in x_batch_dims_sizes
        )
        if has_symbolic_batch:
            # If there's a symbolic dimension, preserve it
            for dim in x_batch_dims_sizes:
                if not isinstance(dim, (int, float)):
                    # Use the symbolic dimension as the batch size
                    batch_size = dim
                    break
            else:
                # Fallback if no symbolic dim is found
                batch_size = x_batch_dims_sizes[0] if x_batch_dims_sizes else 1
        else:
            # Safe multiplication for concrete dimensions
            batch_size = 1
            for dim in x_batch_dims_sizes:
                batch_size *= dim

        # Contract dimensions - these are usually concrete
        x_contract_dims = [x_shape[i] for i in lhs_contract]
        contract_size = 1
        for dim in x_contract_dims:
            if isinstance(dim, (int, float)):
                contract_size *= dim
            else:
                # If we have a symbolic contract dimension (unusual),
                # just use the dimension directly
                contract_size = dim
                break

        input_gemm_shape = (batch_size, contract_size)
        output_gemm_shape = (batch_size, new_kernel_dims_sizes[1])

        return {
            "input": x_shape,
            "input_gemm": input_gemm_shape,
            "output_gemm": output_gemm_shape,
            "output": output_shape,
            "new_kernel": new_kernel_dims_sizes,
        }

    # -----------------------------------------------------------------
    #  abstract_eval – call the *original* implementation via
    #                  jax.eval_shape (symbolic-shape safe)
    # -----------------------------------------------------------------
    @staticmethod
    def abstract_eval(x, kernel, bias, dimension_numbers):
        if _ORIGINAL_LG_CALL is None:
            raise RuntimeError("Original LinearGeneral.__call__ not captured.")

        # Build ShapeDtypeStruct specs
        x_spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        k_spec = jax.ShapeDtypeStruct(kernel.shape, kernel.dtype)
        b_spec = (
            jax.ShapeDtypeStruct(bias.shape, bias.dtype) if bias is not None else None
        )

        def _helper(xv, kv, bv):
            """Invoke the original nnx.LinearGeneral.__call__."""
            # Determine output features from kernel shape and dimension numbers
            kernel_shape = kv.shape
            # Figure out which dimensions in kernel are output features
            rhs_contract = dimension_numbers[0][1]  # Right side contracting dims
            output_dims = [i for i in range(len(kernel_shape)) if i not in rhs_contract]
            out_features = tuple(kernel_shape[i] for i in output_dims)

            dummy = SimpleNamespace(
                kernel=SimpleNamespace(value=kv),
                bias=None if bv is None else SimpleNamespace(value=bv),
                dimension_numbers=dimension_numbers,
                # Add additional required attributes
                batch_axis={},  # FrozenDict in real implementation, but empty dict works fine
                axis=dimension_numbers[0][0],  # Extract axis from dimension_numbers
                in_features=tuple(
                    kv.shape[: len(dimension_numbers[0][1])]
                ),  # Extract from kernel shape
                out_features=out_features,  # Add the output features
                # attributes referenced inside the real implementation
                promote_dtype=lambda a, dtype=None: a,
                # Add dtype (set to None like in the real implementation)
                dtype=None,  # The error occurs when accessing self.dtype
                # Add missing dot_general related attributes
                dot_general=None,
                dot_general_cls=None,
                precision=None,
            )
            return _ORIGINAL_LG_CALL(dummy, xv)

        out = jax.eval_shape(_helper, x_spec, k_spec, b_spec)
        out = jax.tree_util.tree_leaves(out)[0]
        return core.ShapedArray(out.shape, out.dtype)

    def to_onnx(
        self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, dimension_params
    ):
        """Convert linear_general operation to ONNX format."""
        input_var, kernel_var, bias_var = node_inputs[:3]
        output_var = node_outputs[0]

        input_name = s.get_name(input_var)
        output_name = s.get_name(output_var)

        # Get kernel and bias - support both constants and literals
        kernel_name = s.get_name(kernel_var)
        kernel_const = None
        if kernel_name in s.name_to_const:
            kernel_const = s.name_to_const[kernel_name]
        elif hasattr(kernel_var, "val"):
            kernel_const = np.asarray(kernel_var.val)

        if kernel_const is None:
            raise ValueError(
                f"Expected kernel to be a constant tensor, got {kernel_var}"
            )

        # Handle bias - it may be None or a constant
        bias_const = None
        if bias_var is not None:
            bias_name = s.get_name(bias_var)
            if bias_name in s.name_to_const:
                bias_const = s.name_to_const[bias_name]
            elif hasattr(bias_var, "val"):
                bias_const = np.asarray(bias_var.val)

        shape_info = LinearGeneralPlugin._shape_linear_general(
            input_var.aval.shape,
            kernel_const.shape,
            dimension_params["dimension_numbers"],
        )
        output_shape = shape_info["output"]
        new_kernel_shape = shape_info["new_kernel"]
        input_gemm_shape = shape_info["input_gemm"]
        output_gemm_shape = shape_info["output_gemm"]

        # Transform and register kernel as constant
        reshaped_kernel = kernel_const.reshape(new_kernel_shape)
        weights_name = s.get_constant_name(reshaped_kernel)

        # Create reshape for input if needed
        target_input_shape = (-1,) + input_gemm_shape[1:]
        if LinearGeneralPlugin._is_noop_reshape(
            input_var.aval.shape, target_input_shape
        ):
            input_reshape_name = input_name
        else:
            input_reshape_name = s.get_unique_name("input_reshape")
            s.add_node(
                helper.make_node(
                    "Reshape",
                    inputs=[
                        input_name,
                        s.get_constant_name(
                            np.array(target_input_shape, dtype=np.int64)
                        ),
                    ],
                    outputs=[input_reshape_name],
                    name=s.get_unique_name("reshape_input"),
                )
            )
            s.add_shape_info(input_reshape_name, input_gemm_shape)

        # Prepare bias: reshape if necessary or create zero bias
        bias_shape = (output_gemm_shape[1],)
        if bias_const is not None:
            if bias_const.shape != bias_shape:
                bias_const = bias_const.reshape(bias_shape)
            bias_name = s.get_constant_name(bias_const)
        else:
            # Create zero bias with appropriate dtype
            zero_bias = np.zeros(bias_shape, dtype=input_var.aval.dtype)
            bias_name = s.get_constant_name(zero_bias)

        # Build ONNX Gemm operation
        gemm_inputs = [input_reshape_name, weights_name, bias_name]
        gemm_output_name = (
            output_name
            if LinearGeneralPlugin._is_noop_reshape(output_gemm_shape, output_shape)
            else s.get_unique_name("gemm_output")
        )
        s.add_node(
            helper.make_node(
                "Gemm",
                inputs=gemm_inputs,
                outputs=[gemm_output_name],
                name=s.get_unique_name("gemm"),
            )
        )
        s.add_shape_info(gemm_output_name, output_gemm_shape)

        # Final reshape if needed
        if gemm_output_name != output_name:
            target_output_shape = [-1] + list(output_shape[1:])
            s.add_node(
                helper.make_node(
                    "Reshape",
                    inputs=[
                        gemm_output_name,
                        s.get_constant_name(
                            np.array(target_output_shape, dtype=np.int64)
                        ),
                    ],
                    outputs=[output_name],
                    name=s.get_unique_name("reshape_output"),
                )
            )

    @staticmethod
    def _linear_general(x, kernel, bias, dimension_numbers):
        nnx.linear_general_p.multiple_results = False
        return nnx.linear_general_p.bind(
            x, kernel, bias, dimension_numbers=dimension_numbers
        )

    @staticmethod
    def linear_general(x, kernel, bias, dimension_numbers):
        """Binding function for linear_general."""
        return LinearGeneralPlugin._linear_general(x, kernel, bias, dimension_numbers)

    @staticmethod
    # -----------------------------------------------------------------
    #  monkey-patch – capture original & redirect to primitive
    # -----------------------------------------------------------------
    def get_monkey_patch(orig_fn: Callable):
        """Capture *orig_fn* and return our replacement."""
        global _ORIGINAL_LG_CALL
        _ORIGINAL_LG_CALL = orig_fn

        def patched_linear_general_call(self, x):
            # --- 🔑 convert potentially‑negative axes to positive indices ----
            rank = max(x.ndim, 1)  # 👈 avoid "modulo 0"
            if isinstance(self.axis, int):
                lhs_contract = (self.axis % rank,)
            else:
                lhs_contract = tuple((a % rank) for a in self.axis)

            contracting_dims = (
                lhs_contract,
                tuple(range(len(self.in_features))),  # rhs_contracting dims
            )
            dimension_numbers = (contracting_dims, ((), ()))
            return LinearGeneralPlugin._linear_general(
                x,
                self.kernel.value,
                self.bias.value if self.bias else None,
                dimension_numbers,
            )

        return patched_linear_general_call

    @staticmethod
    def patch_info():
        """Provides patching information."""
        return {
            "patch_targets": [nnx.LinearGeneral],
            "patch_function": LinearGeneralPlugin.get_monkey_patch,
            "target_attribute": "__call__",
        }

    @staticmethod
    def _is_noop_reshape(original_shape, target_shape):
        """Return True if target_shape is equivalent to original_shape,
        allowing for a dynamic (-1) in the first dimension.
        """
        if len(original_shape) != len(target_shape):
            return False
        # Compare all dimensions except possibly the first.
        return all(a == b for a, b in zip(original_shape[1:], target_shape[1:]))


# Register abstract evaluation function.
nnx.linear_general_p.def_abstract_eval(LinearGeneralPlugin.abstract_eval)
