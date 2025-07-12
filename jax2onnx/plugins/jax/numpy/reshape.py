# file: jax2onnx/plugins/jax/numpy/reshape.py

from collections.abc import Sequence
from typing import TYPE_CHECKING, Callable, Any

from jax import core
from jax import numpy as jnp
import jax
from jax.extend.core import Primitive

# Use the internal name with an alias for convenience
from jax._src.export.shape_poly import _DimExpr as DimExpr

from onnx import helper, TensorProto

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

import numpy as np

# Define the reshape primitive
jnp.reshape_p = Primitive("jnp.reshape")
jnp.reshape_p.multiple_results = False


@register_primitive(
    jaxpr_primitive=jnp.reshape_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.reshape.html",
    onnx=[
        {
            "component": "Reshape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
        }
    ],
    since="v0.1.0",
    context="primitives.jnp",
    component="reshape",
    testcases=[
        {
            "testcase": "reshape_1",
            "callable": lambda a: jnp.reshape(a, (2, 6)),
            "input_shapes": [(3, 4)],
        },
        {
            "testcase": "reshape_2",
            "callable": lambda a: jnp.reshape(a, (-1, 2)),
            "input_shapes": [(3, 4)],
        },
        {
            "testcase": "reshape_3",
            "callable": lambda a: jnp.reshape(a, (2, -1)),
            "input_shapes": [(3, 4)],
        },
        {
            "testcase": "reshape_4",
            "callable": lambda a: jnp.reshape(a, (-1, 4)),
            "input_shapes": [("B", 3, 4)],
        },
        {
            "testcase": "reshape_to_scalar",
            "callable": lambda a: jnp.reshape(a, ()),
            "input_shapes": [(1,)],
        },
        {
            "testcase": "reshape_from_scalar",
            "callable": lambda a: jnp.reshape(a, (1,)),
            "input_shapes": [()],
        },
        {
            "testcase": "reshape_cnn",
            "callable": lambda x: x.reshape(x.shape[0], -1),
            "input_shapes": [("B", 64, 14, 14)],
        },
        {
            "testcase": "reshape_valid_flatten_trailing",
            # Reshape (N, M, K) to (N, M*K)
            # Corrected callable for reshape_valid_flatten_trailing
            "callable": lambda x: jnp.reshape(
                x, (x.shape[0], x.shape[1] * x.shape[2])  # Pass new shape as a tuple
            ),
            "input_shapes": [(201, 1, 5)],
        },
        {
            "testcase": "reshape_with_target_shape_from_symbolic_dim_computation",
            # This tests if jax2onnx correctly handles new_sizes derived from symbolic computations,
            # relevant to the user's directive on `dim_as_value.to_onnx`.
            "callable": lambda x: jnp.reshape(
                x,
                (
                    x.shape[0],  # Use x.shape[0] for the symbolic dimension 'N'
                    x.shape[1]
                    * x.shape[2],  # Use x.shape[1] for 'M', x.shape[2] for 'K'
                ),
            ),
            "input_shapes": [("N", 3, 5)],  # N, M, K
        },
    ],
)
class ReshapePlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting jax.numpy.reshape to ONNX following the eval_shape pattern.
    """

    _ORIG_CALL: Callable[..., Any] | None = None

    # ------------------------------------------------------------
    # abstract_eval – delegate to original call via jax.eval_shape
    # ------------------------------------------------------------
    @staticmethod
    def abstract_eval(a: core.ShapedArray, *, newshape: Sequence[int | DimExpr], **_):
        """Use jax.eval_shape on the original jnp.reshape."""
        if ReshapePlugin._ORIG_CALL is None:
            raise RuntimeError("Original jnp.reshape not captured by ReshapePlugin.")

        def _helper(arr):
            # Need to handle the case where newshape might contain DimExpr directly
            # during abstract eval. JAX's original reshape should handle this.
            return ReshapePlugin._ORIG_CALL(arr, newshape)

        spec_a = jax.ShapeDtypeStruct(a.shape, a.dtype)
        out_spec = jax.eval_shape(_helper, spec_a)
        return core.ShapedArray(out_spec.shape, out_spec.dtype)

    # ------------------------------------------------------------
    # ONNX Conversion Logic (to_onnx)
    # ------------------------------------------------------------
    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handles conversion of the reshape primitive to ONNX format."""
        input_var = node_inputs[0]
        output_var = node_outputs[0]

        if "new_sizes" in params:
            target_shape_param = params["new_sizes"]
        elif "newshape" in params:
            target_shape_param = params["newshape"]
        else:
            raise KeyError(
                "Could not find 'new_sizes' or 'newshape' in reshape parameters."
            )

        input_name = s.get_name(input_var)
        output_name = s.get_name(output_var)
        input_shape = input_var.aval.shape
        output_aval = output_var.aval
        output_shape = output_aval.shape  # Shape calculated by abstract_eval

        # --- Construct the target shape tensor RELIABLY ---
        shape_components = []
        # Ensure target_shape_param is treated as a sequence
        target_shape_seq = (
            target_shape_param
            if isinstance(target_shape_param, Sequence)
            else [target_shape_param]
        )
        # Handle empty tuple explicitly for reshape to scalar
        if (
            not isinstance(target_shape_param, Sequence)
            and target_shape_param is not None
        ):
            target_shape_seq = [target_shape_param]
        elif target_shape_param == ():  # Check for empty tuple specifically
            target_shape_seq = []
        else:
            # Cast to ensure type safety
            target_shape_seq = (
                list(target_shape_param) if target_shape_param is not None else []
            )

        symbolic_dim_map = getattr(
            s.builder, "reverse_map", getattr(s, "reverse_map", {})
        )
        symbol_to_index = {
            sym: idx for idx, sym in enumerate(input_shape) if isinstance(sym, DimExpr)
        }

        shape_of_input_name = None

        # --- Use target_shape_seq for iteration ---
        for dim in target_shape_seq:
            if isinstance(dim, DimExpr):
                axis_index = symbol_to_index.get(dim)
                if axis_index is None:
                    dim_str = symbolic_dim_map.get(dim, str(dim))
                    str_symbol_to_index = {
                        str(sym): idx for idx, sym in enumerate(input_shape)
                    }
                    axis_index = str_symbol_to_index.get(dim_str)
                    if axis_index is None:
                        raise ValueError(
                            f"Cannot map DimExpr {dim} ('{dim_str}') to input axis index in {input_shape}"
                        )

                if shape_of_input_name is None:
                    shape_of_input_name = s.get_unique_name(f"{input_name}_shape")
                    s.add_node(
                        helper.make_node(
                            "Shape",
                            inputs=[input_name],
                            outputs=[shape_of_input_name],
                            name=s.get_unique_name(f"shape_of_{input_name}"),
                        )
                    )
                    s.add_shape_info(shape_of_input_name, (len(input_shape),), np.int64)

                axis_const = s.get_constant_name(np.array(axis_index, dtype=np.int64))
                gathered_dim_scalar_int64 = s.get_unique_name(
                    f"{input_name}_dim{axis_index}_int64"
                )
                s.add_node(
                    helper.make_node(
                        "Gather",
                        inputs=[shape_of_input_name, axis_const],
                        outputs=[gathered_dim_scalar_int64],
                        axis=0,
                        name=s.get_unique_name(f"gather_dim_{axis_index}"),
                    )
                )
                s.add_shape_info(gathered_dim_scalar_int64, (), np.int64)

                # --- REMOVED INTERMEDIATE CASTS ---

                # --- CORRECTED UNSQUEEZE ---
                # Create constant tensor for 'axes' input
                unsqueeze_axes_const = s.get_constant_name(
                    np.array([0], dtype=np.int64)
                )
                unsqueezed_dim_name = s.get_unique_name(
                    f"{gathered_dim_scalar_int64}_unsqueezed"
                )
                # Provide 'axes' as the second input, remove 'axes' attribute
                s.add_node(
                    helper.make_node(
                        "Unsqueeze",
                        inputs=[
                            gathered_dim_scalar_int64,
                            unsqueeze_axes_const,
                        ],  # Data and Axes inputs
                        outputs=[unsqueezed_dim_name],
                        name=s.get_unique_name(
                            f"unsqueeze_{gathered_dim_scalar_int64}"
                        ),
                        # No 'axes' attribute here for opset >= 13
                    )
                )
                # --- END CORRECTED UNSQUEEZE ---
                s.add_shape_info(unsqueezed_dim_name, (1,), np.int64)
                shape_components.append(unsqueezed_dim_name)

            elif isinstance(dim, (int, np.integer)):
                const_name = s.get_constant_name(np.array([int(dim)], dtype=np.int64))
                shape_components.append(const_name)
            else:
                # This case should ideally not be hit if target_shape_seq is correct
                raise TypeError(
                    f"Unsupported dimension type: {type(dim)} in {target_shape_seq}"
                )

        # Concatenate components
        if not shape_components:  # Reshape to scalar case ()
            shape_tensor_name = s.get_constant_name(np.array([], dtype=np.int64))
        elif len(shape_components) == 1:
            shape_tensor_name = shape_components[0]
        else:
            shape_tensor_name = s.get_unique_name(f"{input_name}_target_shape")
            s.add_node(
                helper.make_node(
                    "Concat",
                    inputs=shape_components,
                    outputs=[shape_tensor_name],
                    axis=0,
                    name=s.get_unique_name("concat_target_shape"),
                )
            )
            s.add_shape_info(
                shape_tensor_name, (len(target_shape_seq),), np.int64
            )  # Use length of original sequence

        # --- Data Input/Output Type Handling (Keep previous logic) ---
        input_dtype = input_var.aval.dtype
        input_dtype_enum = s._ensure_onnx_dtype(input_dtype)
        expected_output_dtype = output_aval.dtype
        expected_output_dtype_enum = s._ensure_onnx_dtype(expected_output_dtype)
        reshape_data_input_name = input_name
        reshape_data_input_dtype_enum = input_dtype_enum
        if input_dtype_enum == TensorProto.INT32:
            casted_input_name = s.get_unique_name(f"{input_name}_casted_int64")
            s.add_node(
                helper.make_node(
                    "Cast",
                    inputs=[input_name],
                    outputs=[casted_input_name],
                    to=int(TensorProto.INT64),
                    name=s.get_unique_name("cast_reshape_data_to_int64"),
                )
            )
            s.add_shape_info(casted_input_name, input_shape, np.int64)
            reshape_data_input_name = casted_input_name
            reshape_data_input_dtype_enum = TensorProto.INT64

        # --- Emit ONNX Reshape node ---
        temp_reshape_output_name = s.get_unique_name(f"{input_name}_reshaped")
        reshape_node = helper.make_node(
            "Reshape",
            inputs=[reshape_data_input_name, shape_tensor_name],
            outputs=[temp_reshape_output_name],
            name=s.get_unique_name("reshape_0"),
            allowzero=0,
        )
        s.add_node(reshape_node)
        s.add_shape_info(
            temp_reshape_output_name, output_shape, reshape_data_input_dtype_enum
        )

        # --- Cast Output back to JAX expected type if necessary ---
        if reshape_data_input_dtype_enum != expected_output_dtype_enum:
            cast_node = helper.make_node(
                "Cast",
                inputs=[temp_reshape_output_name],
                outputs=[output_name],
                to=int(expected_output_dtype_enum),
                name=s.get_unique_name("cast_reshape_output_to_jax_expected"),
            )
            s.add_node(cast_node)
            s.add_shape_info(output_name, output_shape, expected_output_dtype_enum)
        else:
            identity_node = helper.make_node(
                "Identity",
                inputs=[temp_reshape_output_name],
                outputs=[output_name],
                name=s.get_unique_name("identity_reshape_output"),
            )
            s.add_node(identity_node)
            s.add_shape_info(output_name, output_shape, expected_output_dtype_enum)

    # ------------------------------------------------------------
    # monkey-patch – capture original & inject primitive binding
    # ------------------------------------------------------------
    @staticmethod
    def _reshape_binding(a, newshape, order="C"):
        """Binds inputs to the reshape primitive."""
        if order != "C":
            raise NotImplementedError("Only C-style reshape is supported.")
        return jnp.reshape_p.bind(a, newshape=newshape)

    @staticmethod
    def get_monkey_patch(orig_fn: Callable):
        """Returns the patched function that captures the original and binds the primitive."""
        ReshapePlugin._ORIG_CALL = orig_fn

        def patched_reshape(a, newshape, order="C"):
            return ReshapePlugin._reshape_binding(a, newshape, order)

        return patched_reshape

    @staticmethod
    def patch_info():
        """Provides patching information for jnp.reshape."""
        return {
            "patch_targets": [jnp],
            "target_attribute": "reshape",
            "patch_function": ReshapePlugin.get_monkey_patch,
        }


# --- Existing registration ---
jnp.reshape_p.def_abstract_eval(ReshapePlugin.abstract_eval)
# --- End Existing registration ---
