# jax2onnx/plugins/flax/nnx/linear_general.py

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

from typing import Callable
from types import SimpleNamespace

import jax
import numpy as np
from flax import nnx
from jax import core
from jax.extend.core import Primitive
from onnx import helper

from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter
from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive


# ------------------------------------------------------------------
# Helpers for shape‑info assertions in the “merge symbolic dim” test
# ------------------------------------------------------------------
def _shape_of(coll, name: str):
    """Return tuple of dims (could be int, str or None) for the tensor named `name`."""
    for vi in coll:
        if vi.name == name:
            return tuple(
                (
                    d.dim_param
                    if d.HasField("dim_param") and d.dim_param
                    else (d.dim_value if d.HasField("dim_value") else None)
                )
                for d in vi.type.tensor_type.shape.dim
            )
    raise KeyError(f"Cannot find '{name}' in ValueInfo collection")


def _shape_prefix_of(coll, prefix: str):
    """Return tuple of dims for the first tensor whose name starts with `prefix`."""
    for vi in coll:
        if vi.name.startswith(prefix):
            return tuple(
                (
                    d.dim_param
                    if d.HasField("dim_param") and d.dim_param
                    else (d.dim_value if d.HasField("dim_value") else None)
                )
                for d in vi.type.tensor_type.shape.dim
            )
    raise KeyError(f"No tensor name starting with '{prefix}'")


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
            "testcase": "linear_general_merge_symbolic_dim",
            "callable": nnx.LinearGeneral(
                in_features=(4, 16),  # ⟨4,16⟩ are the contracting dims
                out_features=32,
                axis=(-2, -1),
                rngs=nnx.Rngs(0),
            ),
            # B is symbolic
            "input_shapes": [("B", 8, 4, 16)],
            "run_only_dynamic": True,
            "run_only_f32_variant": True,
            # Validate *all* shape‑infos: input, two intermediates, and output
            "post_check_onnx_graph": lambda m: (
                # var_0  (graph input)  should be B×8×4×16
                _shape_of(m.graph.input, "var_0") == ("B", 8, 4, 16)
                # input_reshape  flatten → ?×64
                and (lambda s: s[0] is None and s[1] == 64)(
                    _shape_prefix_of(m.graph.value_info, "input_reshape")
                )
                # gemm_output     → ?×32
                and (lambda s: s[0] is None and s[1] == 32)(
                    _shape_prefix_of(m.graph.value_info, "gemm_output")
                )
                # var_3  (graph output)  B×8×32
                and _shape_of(m.graph.output, "var_3") == ("B", 8, 32)
            ),
        },
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
        {
            "testcase": "dynamic_batch_and_feature_dims",
            # Use the plugin's static method to bind the primitive
            "callable": lambda x, k, b: LinearGeneralPlugin._linear_general(
                x, k, b, dimension_numbers=(((2,), (0,)), ((), ()))
            ),
            "input_shapes": [("B", "H", 16), (16, 4, 4), (4, 4)],
            "run_only_dynamic": True,
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
        # If a flatten merges   B × 8   we can no longer expose a single
        # symbolic letter → mark it “unknown” so Netron shows “?”.
        # ── Batch‑dimension handling ───────────────────────────────────
        # After flattening (B, 8) →   ?   we do **not** want a *new*
        # symbol – we want a fully anonymous dynamic dim so Netron
        # shows “?” (no dim_param, no dim_value).
        has_symbolic_batch = any(
            not isinstance(d, (int, float)) for d in x_batch_dims_sizes
        )
        if has_symbolic_batch:
            batch_size = (
                x_batch_dims_sizes[0]
                if len(x_batch_dims_sizes) == 1
                and not isinstance(x_batch_dims_sizes[0], (int, float))
                else None  # ← anonymous dynamic dimension
            )
        else:
            batch_size = (
                np.prod(x_batch_dims_sizes, dtype=int) if x_batch_dims_sizes else 1
            )

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

    # ... inside the LinearGeneralPlugin class

    def to_onnx(
        self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, dimension_params
    ):
        """Convert linear_general operation to ONNX format."""
        input_var, kernel_var, bias_var = node_inputs[:3]
        output_var = node_outputs[0]

        input_name = s.get_name(input_var)
        output_name = s.get_name(output_var)
        kernel_name = s.get_name(kernel_var)

        # Handle both constant and variable kernels
        kernel_const = None
        if kernel_name in s.name_to_const:
            kernel_const = s.name_to_const[kernel_name]
            kernel_shape = kernel_const.shape
        elif hasattr(kernel_var, "val"):
            kernel_const = np.asarray(kernel_var.val)
            kernel_shape = kernel_const.shape
        else:
            kernel_shape = kernel_var.aval.shape

        shape_info = LinearGeneralPlugin._shape_linear_general(
            input_var.aval.shape,
            kernel_shape,
            dimension_params["dimension_numbers"],
        )
        output_shape = shape_info["output"]
        new_kernel_shape = shape_info["new_kernel"]
        input_gemm_shape = shape_info["input_gemm"]
        output_gemm_shape = shape_info["output_gemm"]

        # Transform kernel (constant or variable)
        if kernel_const is not None:
            weights_name = s.get_constant_name(kernel_const.reshape(new_kernel_shape))
        else:
            weights_name = s.get_unique_name("reshaped_kernel")
            s.add_node(
                helper.make_node(
                    "Reshape",
                    [
                        kernel_name,
                        s.get_constant_name(np.array(new_kernel_shape, dtype=np.int64)),
                    ],
                    [weights_name],
                )
            )
            s.add_shape_info(weights_name, new_kernel_shape)

        # Reshape input for Gemm
        target_input_shape = (-1,) + input_gemm_shape[1:]
        if len(input_var.aval.shape) > 2 or input_var.aval.shape != input_gemm_shape:
            input_reshape_name = s.get_unique_name("input_reshape")
            s.add_node(
                helper.make_node(
                    "Reshape",
                    [
                        input_name,
                        s.get_constant_name(
                            np.array(target_input_shape, dtype=np.int64)
                        ),
                    ],
                    [input_reshape_name],
                )
            )
            s.add_shape_info(input_reshape_name, input_gemm_shape)
        else:
            input_reshape_name = input_name

        # Prepare bias for Gemm
        bias_shape = (output_gemm_shape[-1],)

        bias_const = None
        is_bias_present = bias_var is not None and s.get_name(bias_var) != "onnx::None"

        if is_bias_present:
            bias_name = s.get_name(bias_var)
            if bias_name in s.name_to_const:
                bias_const = s.name_to_const[bias_name]
            elif hasattr(bias_var, "val"):
                bias_const = np.asarray(bias_var.val)

        if bias_const is not None:
            bias_gemm_name = s.get_constant_name(bias_const.reshape(bias_shape))
        elif is_bias_present:
            bias_gemm_name = s.get_unique_name("bias_for_gemm")
            bias_shape_tensor = s.get_constant_name(
                np.array(bias_shape, dtype=np.int64)
            )
            s.add_node(
                helper.make_node(
                    "Reshape",
                    [s.get_name(bias_var), bias_shape_tensor],
                    [bias_gemm_name],
                )
            )
            s.add_shape_info(bias_gemm_name, bias_shape)
        else:
            zero_bias = np.zeros(bias_shape, dtype=input_var.aval.dtype)
            bias_gemm_name = s.get_constant_name(zero_bias)

        # Build ONNX Gemm operation
        gemm_inputs = [input_reshape_name, weights_name, bias_gemm_name]
        gemm_output_name = (
            output_name
            if tuple(output_gemm_shape) == tuple(output_shape)
            else s.get_unique_name("gemm_output")
        )
        s.add_node(
            helper.make_node("Gemm", inputs=gemm_inputs, outputs=[gemm_output_name])
        )
        s.add_shape_info(gemm_output_name, output_gemm_shape)

        # Final reshape if needed
        if gemm_output_name != output_name:
            if all(isinstance(d, (int, np.integer)) for d in output_shape):
                shape_tensor = s.get_constant_name(
                    np.array(output_shape, dtype=np.int64)
                )
            else:
                (
                    (_, rhs_contract),
                    (_, _),
                ) = dimension_params["dimension_numbers"]
                kernel_rank = len(kernel_shape)
                kernel_out_dims_indices = [
                    i for i in range(kernel_rank) if i not in rhs_contract
                ]
                kernel_out_dims = [kernel_shape[i] for i in kernel_out_dims_indices]
                num_batch_dims = len(output_shape) - len(kernel_out_dims)

                input_shape_tensor = s.get_unique_name("input_shape")
                s.add_node(
                    helper.make_node("Shape", [input_name], [input_shape_tensor])
                )
                s.add_shape_info(
                    input_shape_tensor, (len(input_var.aval.shape),), np.int64
                )

                batch_dims_tensor = s.get_unique_name("batch_dims")
                s.add_node(
                    helper.make_node(
                        "Slice",
                        [
                            input_shape_tensor,
                            s.get_constant_name(np.array([0], dtype=np.int64)),
                            s.get_constant_name(
                                np.array([num_batch_dims], dtype=np.int64)
                            ),
                        ],
                        [batch_dims_tensor],
                    )
                )
                s.add_shape_info(batch_dims_tensor, (num_batch_dims,), np.int64)

                static_dims_tensor = s.get_constant_name(
                    np.array(kernel_out_dims, dtype=np.int64)
                )

                shape_tensor = s.get_unique_name("final_shape")
                s.add_node(
                    helper.make_node(
                        "Concat",
                        [batch_dims_tensor, static_dims_tensor],
                        [shape_tensor],
                        axis=0,
                    )
                )
                s.add_shape_info(shape_tensor, (len(output_shape),), np.int64)

            s.add_node(
                helper.make_node(
                    "Reshape", [gemm_output_name, shape_tensor], [output_name]
                )
            )

        s.add_shape_info(output_name, output_shape)

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


# ------------------------------------------------------------------
# 🔑 Define and register the *concrete implementation* for the primitive.
# This tells JAX how to execute the operation.
# ------------------------------------------------------------------
def _linear_general_impl(x, kernel, bias, dimension_numbers):
    """The actual implementation of the linear_general primitive."""
    y = jax.lax.dot_general(x, kernel, dimension_numbers=dimension_numbers)
    if bias is not None:
        # Add bias if provided
        y += bias
    return y


# Register the implementation function
nnx.linear_general_p.def_impl(_linear_general_impl)
