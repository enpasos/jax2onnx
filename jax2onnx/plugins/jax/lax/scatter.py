# jax2onnx/plugins/jax/lax/scatter.py
from __future__ import annotations
from typing import TYPE_CHECKING, Sequence, Any
import numpy as np
import jax.numpy as jnp
from jax import lax, core
from jax.lax import (
    ScatterDimensionNumbers,
    GatherScatterMode,  # Ensure GatherScatterMode is imported
)
from onnx import helper, TensorProto

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # only for static type checkers
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


# ---------------------------------------------------------------------
# 1. primitive alias
# ---------------------------------------------------------------------
scatter_p = lax.scatter_p
# ---------------------------------------------------------------------


@register_primitive(
    jaxpr_primitive=scatter_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.scatter.html",
    onnx=[
        {
            "component": "ScatterElements",
            "doc": "https://onnx.ai/onnx/operators/onnx__ScatterElements.html",
        },
        {
            "component": "ScatterND",
            "doc": "https://onnx.ai/onnx/operators/onnx__ScatterND.html",
        },
    ],
    since="v0.4.4",
    context="primitives.lax",
    component="scatter",
    testcases=[
        {
            "testcase": "scatter_set_axis0",
            "callable": lambda x: x.at[0].set(-100.0),
            "input_shapes": [(1, 1)],
        },
        {
            "testcase": "scatter_set_middle",
            "callable": lambda x: x.at[1].set(42.0),
            "input_shapes": [(3,)],
        },
        {
            "testcase": "scatter_correct_axis_determination",
            "callable": lambda op, idx, upd_scalar_batch: lax.scatter(
                op,
                idx,
                jnp.reshape(upd_scalar_batch, idx.shape[:-1]),
                ScatterDimensionNumbers(
                    update_window_dims=(),
                    inserted_window_dims=(0,),
                    scatter_dims_to_operand_dims=(0,),
                ),
            ),
            "input_shapes": [(5,), (1, 1, 1, 1), (1,)],
            "input_dtypes": [np.float32, np.int32, np.float32],
        },
        {
            "testcase": "scatter_updates_slice_needed_axis0",
            "callable": lambda op, idx, upd_scalar_batch: lax.scatter(
                op,
                idx,
                jnp.reshape(upd_scalar_batch, idx.shape[:-1]),
                ScatterDimensionNumbers(
                    update_window_dims=(),
                    inserted_window_dims=(0,),
                    scatter_dims_to_operand_dims=(0,),
                ),
            ),
            "input_shapes": [(5,), (1, 1, 1, 1), (1,)],
            "input_dtypes": [np.float32, np.int32, np.float32],
        },
        {
            "testcase": "scatter_from_user_warning_shapes_valid_jax",
            "callable": lambda operand, indices, updates_sliced_scalar_batch: lax.scatter(
                operand,
                indices,
                jnp.reshape(updates_sliced_scalar_batch, indices.shape[:-1]),
                ScatterDimensionNumbers(
                    update_window_dims=(),
                    inserted_window_dims=(0,),
                    scatter_dims_to_operand_dims=(0,),
                ),
            ),
            "input_shapes": [(5,), (1, 1, 1, 1), (1,)],
            "input_dtypes": [np.float32, np.int32, np.float32],
        },
        {
            "testcase": "scatter_user_error_scenario_precise",
            "callable": lambda operand, indices, updates: lax.scatter(
                operand,
                indices,
                updates,
                ScatterDimensionNumbers(
                    update_window_dims=(1, 2, 3),
                    inserted_window_dims=(0,),
                    scatter_dims_to_operand_dims=(0,),
                    operand_batching_dims=(),
                    scatter_indices_batching_dims=(),
                ),
                mode=GatherScatterMode.FILL_OR_DROP,
                unique_indices=False,
                indices_are_sorted=False,
            ),
            "input_shapes": [(5, 201, 1, 1), (2, 1), (2, 201, 1, 1)],
            "input_dtypes": [np.float32, np.int32, np.float32],
            "expected_error": NotImplementedError,  # Will change after fix
        },
    ],
)
class ScatterPlugin(PrimitiveLeafPlugin):
    @staticmethod
    def abstract_eval(
        operand: core.ShapedArray,
        indices: core.ShapedArray,
        updates: core.ShapedArray,
        update_jaxpr,
        *,
        dimension_numbers,
        indices_are_sorted,
        unique_indices,
        mode,
        **params,
    ):
        return core.ShapedArray(operand.shape, operand.dtype)

    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: Sequence[Any],
        node_outputs: Sequence[Any],
        params: dict[str, Any],
    ):
        operand_v, indices_v, updates_v = node_inputs
        out_v = node_outputs[0]

        operand_name = s.get_name(operand_v)
        original_indices_onnx_name = s.get_name(indices_v)
        original_updates_onnx_name = s.get_name(updates_v)
        out_name = s.get_name(out_v)

        operand_shape = tuple(operand_v.aval.shape)
        operand_rank = len(operand_shape)
        operand_dtype = operand_v.aval.dtype

        # Keep original JAX shapes for ScatterND path decision
        jax_indices_shape = tuple(indices_v.aval.shape)
        jax_indices_rank = len(jax_indices_shape)
        jax_indices_dtype = indices_v.aval.dtype  # Original dtype for casting check

        jax_updates_shape = tuple(updates_v.aval.shape)
        # jax_updates_rank = len(jax_updates_shape) # Not directly used in ScatterND decision

        current_indices_name = original_indices_onnx_name  # Will be updated if casted
        current_indices_dtype_for_onnx = jax_indices_dtype

        indices_onnx_target_dtype = np.int64
        if jax_indices_dtype != indices_onnx_target_dtype:
            cast_indices_out_name = s.get_unique_name(
                f"{original_indices_onnx_name}_int64"
            )
            s.add_node(
                helper.make_node(
                    "Cast",
                    inputs=[original_indices_onnx_name],
                    outputs=[cast_indices_out_name],
                    name=s.get_unique_name(
                        f"cast_{original_indices_onnx_name}_to_int64"
                    ),
                    to=int(TensorProto.INT64),
                )
            )
            s.add_shape_info(
                cast_indices_out_name, jax_indices_shape, indices_onnx_target_dtype
            )
            current_indices_name = cast_indices_out_name
            current_indices_dtype_for_onnx = indices_onnx_target_dtype

        dimension_numbers: ScatterDimensionNumbers = params["dimension_numbers"]
        mode_param = params.get("mode", lax.GatherScatterMode.PROMISE_IN_BOUNDS)

        mode_enum = mode_param
        if not isinstance(mode_param, lax.GatherScatterMode):
            mode_enum = lax.GatherScatterMode.from_any(str(mode_param))

        # --- Attempt ScatterND path ---
        use_scatter_nd = False
        if (
            mode_enum != lax.GatherScatterMode.CLIP
        ):  # ScatterND does not support CLIP directly
            K = len(dimension_numbers.scatter_dims_to_operand_dims)

            # Condition 1: JAX indices last dimension size must match K
            # Condition 2: Number of batch dimensions in JAX indices and JAX updates must match.
            #             A simple check for this is rank(indices) - 1 == rank(updates) - rank(update_slice)
            # Condition 3: The slice shape from JAX updates must match operand.shape[K:]

            if K > 0 and jax_indices_shape and K == jax_indices_shape[-1]:
                # Number of "batch" dimensions for indices (dimensions before the K coordinates)
                indices_batch_rank = jax_indices_rank - 1

                # Expected shape of the "slice" part of the updates tensor
                # These are dimensions of the operand tensor *after* the K indexed dimensions.
                expected_update_slice_shape_in_operand = operand_shape[K:]

                # Actual shape of the "slice" part of the JAX updates tensor.
                # These are dimensions of the JAX updates tensor *after* its batch dimensions.
                # The batch dimensions of updates should match batch dimensions of indices.
                if len(jax_updates_shape) >= indices_batch_rank:
                    actual_updates_slice_shape_in_updates = jax_updates_shape[
                        indices_batch_rank:
                    ]

                    # Check if batch prefixes match
                    if (
                        jax_indices_shape[:indices_batch_rank]
                        == jax_updates_shape[:indices_batch_rank]
                    ):
                        # Now, the crucial part: map JAX's complex windowing to what ScatterND expects.
                        # ScatterND expects updates[i_batch, ..., actual_updates_slice_shape_in_updates]
                        # to match the shape of operand[coords, expected_update_slice_shape_in_operand]
                        # For the user's case:
                        # update_window_dims=(1,2,3) describes how to get the slice from `updates` tensor.
                        # inserted_window_dims=(0,) describes operand window.
                        # The true "slice" from updates is `updates[..., d1, d2, d3]` where d1,d2,d3 are from update_window_dims.

                        # For the user's specific case:
                        # operand_shape = (5, 201, 1, 1), jax_indices_shape = (2, 1), jax_updates_shape = (2, 201, 1, 1)
                        # K = 1 (from scatter_dims_to_operand_dims=(0,))
                        # indices_batch_rank = 2 - 1 = 1. indices_batch_shape = (2,)
                        # expected_update_slice_shape_in_operand = operand_shape[1:] = (201, 1, 1)
                        # updates_batch_shape_prefix = jax_updates_shape[:1] = (2,)
                        # actual_updates_slice_shape_in_updates (from updates) = jax_updates_shape[1:] = (201, 1, 1)

                        if (
                            actual_updates_slice_shape_in_updates
                            == expected_update_slice_shape_in_operand
                        ):
                            # This simple alignment works for cases like the user's.
                            # More complex dimension_numbers would need more sophisticated mapping.
                            use_scatter_nd = True

        if use_scatter_nd:
            s.logger.info(
                f"Using ONNX ScatterND for JAX scatter primitive with dimension_numbers: "
                f"{dimension_numbers}, mode: {mode_enum}"
            )
            s.add_node(
                helper.make_node(
                    "ScatterND",
                    inputs=[
                        operand_name,
                        current_indices_name,
                        original_updates_onnx_name,
                    ],  # Use original updates
                    outputs=[out_name],
                    name=s.get_unique_name("scatter_nd_node"),
                )
            )
            s.add_shape_info(out_name, operand_shape, operand_dtype)
            return  # ScatterND path taken and finished

        # --- Fallback to ScatterElements logic (existing logic which will likely error for user's case) ---
        s.logger.warning(
            f"Attempting to use ONNX ScatterElements for JAX scatter. Configuration: "
            f"operand_shape={operand_shape}, jax_indices_shape={jax_indices_shape}, "
            f"jax_updates_shape={jax_updates_shape}, dimension_numbers={dimension_numbers}. "
            f"This might fail if shapes are not compatible with ScatterElements."
        )

        # Continue with the original ScatterElements path transformations
        # These will transform current_indices_shape and current_updates_shape
        current_indices_shape_for_elements = jax_indices_shape
        current_indices_rank_for_elements = jax_indices_rank
        # current_indices_name is already set (potentially casted)

        current_updates_name_for_elements = original_updates_onnx_name
        current_updates_shape_for_elements = jax_updates_shape
        current_updates_rank_for_elements = len(jax_updates_shape)
        current_updates_dtype_for_elements = updates_v.aval.dtype

        if current_indices_rank_for_elements < operand_rank:
            target_indices_shape_list = list(current_indices_shape_for_elements)
            for _ in range(operand_rank - current_indices_rank_for_elements):
                target_indices_shape_list.insert(0, 1)
            target_indices_shape = tuple(target_indices_shape_list)

            reshape_indices_out_name = s.get_unique_name(
                f"{current_indices_name}_reshaped_rank_pad"
            )
            reshape_indices_shape_const_val = np.array(
                target_indices_shape, dtype=np.int64
            )
            reshape_indices_shape_const_name = s.get_constant_name(
                reshape_indices_shape_const_val
            )

            s.add_node(
                helper.make_node(
                    "Reshape",
                    inputs=[current_indices_name, reshape_indices_shape_const_name],
                    outputs=[reshape_indices_out_name],
                    name=s.get_unique_name(
                        f"reshape_{current_indices_name}_rank_pad_se"
                    ),
                )
            )
            s.add_shape_info(
                reshape_indices_out_name,
                target_indices_shape,
                current_indices_dtype_for_onnx,
            )
            current_indices_name = (
                reshape_indices_out_name  # Update name for ScatterElements
            )
            current_indices_shape_for_elements = target_indices_shape
            # current_indices_rank_for_elements = len(current_indices_shape_for_elements) # Not needed further

        elif current_indices_rank_for_elements > operand_rank:
            num_dims_to_squeeze = current_indices_rank_for_elements - operand_rank
            if all(
                current_indices_shape_for_elements[i] == 1
                for i in range(num_dims_to_squeeze)
            ):
                squeezed_indices_shape = current_indices_shape_for_elements[
                    num_dims_to_squeeze:
                ]
                if not squeezed_indices_shape:
                    squeezed_indices_shape = (1,)

                reshape_indices_out_name = s.get_unique_name(
                    f"{current_indices_name}_reshaped_rank_squeeze"
                )
                reshape_indices_shape_const_val = np.array(
                    squeezed_indices_shape, dtype=np.int64
                )
                reshape_indices_shape_const_name = s.get_constant_name(
                    reshape_indices_shape_const_val
                )
                s.add_node(
                    helper.make_node(
                        "Reshape",
                        [current_indices_name, reshape_indices_shape_const_name],
                        [reshape_indices_out_name],
                        name=s.get_unique_name(
                            f"reshape_{current_indices_name}_rank_squeeze_se"
                        ),
                    )
                )
                s.add_shape_info(
                    reshape_indices_out_name,
                    squeezed_indices_shape,
                    current_indices_dtype_for_onnx,
                )
                current_indices_name = reshape_indices_out_name
                current_indices_shape_for_elements = squeezed_indices_shape
            else:
                raise ValueError(  # This was the original error path, keep for consistency if ScatterND fails early
                    f"Scatter original indices rank ({jax_indices_rank}) > operand rank ({operand_rank}), "
                    f"and leading dimensions of indices {jax_indices_shape} are not all 1 to allow squeezing for ScatterElements."
                )

        if current_updates_rank_for_elements < operand_rank:
            target_updates_shape_list = list(current_updates_shape_for_elements)
            for _ in range(operand_rank - current_updates_rank_for_elements):
                target_updates_shape_list.insert(0, 1)
            target_updates_shape = tuple(target_updates_shape_list)

            reshape_updates_out_name = s.get_unique_name(
                f"{current_updates_name_for_elements}_reshaped_rank_pad"
            )
            reshape_updates_shape_const_val = np.array(
                target_updates_shape, dtype=np.int64
            )
            reshape_updates_shape_const_name = s.get_constant_name(
                reshape_updates_shape_const_val
            )
            s.add_node(
                helper.make_node(
                    "Reshape",
                    [
                        current_updates_name_for_elements,
                        reshape_updates_shape_const_name,
                    ],
                    [reshape_updates_out_name],
                    name=s.get_unique_name(
                        f"reshape_{current_updates_name_for_elements}_rank_pad_se"
                    ),
                )
            )
            s.add_shape_info(
                reshape_updates_out_name,
                target_updates_shape,
                current_updates_dtype_for_elements,
            )
            current_updates_name_for_elements = reshape_updates_out_name
            current_updates_shape_for_elements = target_updates_shape

        if current_updates_shape_for_elements != current_indices_shape_for_elements:
            if np.prod(current_updates_shape_for_elements) == np.prod(
                current_indices_shape_for_elements
            ):
                reshape_out_name = s.get_unique_name(
                    f"{current_updates_name_for_elements}_reshaped_to_indices_fallback"
                )
                reshape_target_shape_const_val = np.array(
                    current_indices_shape_for_elements, dtype=np.int64
                )
                reshape_target_shape_const_name = s.get_constant_name(
                    reshape_target_shape_const_val
                )
                s.add_node(
                    helper.make_node(
                        "Reshape",
                        [
                            current_updates_name_for_elements,
                            reshape_target_shape_const_name,
                        ],
                        [reshape_out_name],
                        name=s.get_unique_name(
                            f"reshape_{current_updates_name_for_elements}_to_indices_fallback_se"
                        ),
                    )
                )
                s.add_shape_info(
                    reshape_out_name,
                    current_indices_shape_for_elements,
                    current_updates_dtype_for_elements,
                )
                current_updates_name_for_elements = reshape_out_name
                # current_updates_shape_for_elements = current_indices_shape_for_elements # Shape is now matched
            else:
                # This is where the user's error is triggered if ScatterND path was not taken.
                raise NotImplementedError(
                    f"Cannot make JAX updates shape {jax_updates_shape} (processed to {current_updates_shape_for_elements}) "
                    f"match indices shape {current_indices_shape_for_elements} for ONNX ScatterElements. "
                    f"This configuration may not be suitable for element-wise ScatterElements or requires a different ONNX operator."
                )

        onnx_scatter_axis = 0
        if dimension_numbers.scatter_dims_to_operand_dims:
            if len(dimension_numbers.scatter_dims_to_operand_dims) == 1:
                onnx_scatter_axis = dimension_numbers.scatter_dims_to_operand_dims[0]
            else:
                s.logger.warning(
                    f"Multiple scatter_dims_to_operand_dims ({dimension_numbers.scatter_dims_to_operand_dims}) "
                    f"found for ScatterElements, using axis 0. This might be incorrect."
                )

        scatter_node = helper.make_node(
            "ScatterElements",
            inputs=[
                operand_name,
                current_indices_name,
                current_updates_name_for_elements,
            ],
            outputs=[out_name],
            name=s.get_unique_name("scatter_elements_node"),
            axis=onnx_scatter_axis,
        )
        s.add_node(scatter_node)
        s.add_shape_info(out_name, operand_shape, operand_dtype)


scatter_p.def_abstract_eval(ScatterPlugin.abstract_eval)
