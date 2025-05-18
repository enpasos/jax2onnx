from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Any
import numpy as np
from jax import numpy as jnp
from jax import lax, core
from jax.lax import ScatterDimensionNumbers, GatherScatterMode
from onnx import helper, TensorProto

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # only for static type checkers
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# ---------------------------------------------------------------------
# 1. primitive alias
# ---------------------------------------------------------------------
scatter_add_p = lax.scatter_add_p
# ---------------------------------------------------------------------


@register_primitive(
    jaxpr_primitive=scatter_add_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scatter_add.html",
    onnx=[
        {
            "component": "ScatterND",
            "doc": "https://onnx.ai/onnx/operators/onnx__ScatterND.html",
            "attributes": ["reduction='add'"],
        }
    ],
    since="v0.5.3",
    context="primitives.lax",
    component="scatter_add",
    testcases=[
        {
            "testcase": "scatter_add_simple_1d",
            "callable": lambda operand, indices, updates: lax.scatter_add(
                operand,
                indices,
                updates,
                dimension_numbers=ScatterDimensionNumbers(
                    update_window_dims=(),
                    inserted_window_dims=(0,),
                    scatter_dims_to_operand_dims=(0,),
                ),
            ),
            "input_values": [
                np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
                np.array([[1], [3]], dtype=np.int32),
                np.array([10.0, 20.0], dtype=np.float32),
            ],
        },
        {
            "testcase": "scatter_add_window_2d_operand_1d_indices",
            "callable": lambda operand, indices, updates: lax.scatter_add(
                operand,  # operand (2,3)
                indices,
                updates,
                dimension_numbers=ScatterDimensionNumbers(
                    update_window_dims=(
                        1,
                    ),  # Describes the (3,) window in updates (dim 1 of updates(1,3))
                    inserted_window_dims=(
                        0,
                    ),  # Operand's dim 0 is "inserted" to align ranks for JAX rule
                    scatter_dims_to_operand_dims=(
                        0,
                    ),  # Indices map to operand's 0-th dim
                ),
            ),
            "input_values": [
                np.array(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
                ),  # operand (2,3)
                np.array(
                    [[0]], dtype=np.int32
                ),  # indices (1,1) -> scatter to operand[0]
                np.array(
                    [[10.0, 20.0, 30.0]], dtype=np.float32
                ),  # updates (1,3) -> add [10,20,30]
            ],
            # Expected output: [[11,22,33],[4,5,6]]
        },
        {
            "testcase": "scatter_add_batch_updates_1d_operand",
            "callable": lambda operand, indices, updates: lax.scatter_add(
                operand,
                indices,
                updates,
                dimension_numbers=ScatterDimensionNumbers(
                    update_window_dims=(),
                    inserted_window_dims=(0,),
                    scatter_dims_to_operand_dims=(0,),
                ),
            ),
            "input_values": [
                np.zeros((5,), dtype=np.float32),
                np.array(
                    [[[0], [1]], [[0], [2]]], dtype=np.int32
                ),  # JAX indices (2,2,1)
                np.array(
                    [[10.0, 20.0], [30.0, 40.0]], dtype=np.float32
                ),  # JAX updates (2,2)
            ],
        },
        {
            "testcase": "scatter_add_windowed_from_original_failure_analogue",
            "callable": lambda operand, indices, updates: lax.scatter_add(  # Or jax.lax.scatter_add
                operand,
                indices,
                updates,
                ScatterDimensionNumbers(  # MODIFIED HERE
                    update_window_dims=(0, 1, 2),
                    inserted_window_dims=(),
                    scatter_dims_to_operand_dims=(0,),
                    operand_batching_dims=(1,),  #  <--- Key change to satisfy rank rule
                ),
            ),
            "input_values": [
                jnp.zeros((5, 201, 1, 1), dtype=jnp.float32),
                jnp.array([[0], [4]], dtype=jnp.int32),
                jnp.arange(2 * 201, dtype=jnp.float32).reshape(2, 201, 1, 1),
            ],
            "input_shapes": [(5, 201, 1, 1), (2, 1), (2, 201, 1, 1)],
            "input_dtypes": [jnp.float32, jnp.int32, jnp.float32],
        },
        # {
        #     "testcase": "scatter_add_windowed_simple",
        #     "callable": lambda operand, indices, updates: jax.lax.scatter_add(
        #         operand,
        #         indices,
        #         updates,
        #         jax.lax.ScatterDimensionNumbers(
        #             update_window_dims=(1, 2),  # Dims of `updates` forming the window
        #             inserted_window_dims=(),  # Dims of `operand` not part of window from `updates`
        #             scatter_dims_to_operand_dims=(0,),  # `indices` map to operand dim 0
        #         ),
        #     ),
        #     "input_values": [
        #         jnp.zeros((5, 3, 2), dtype=jnp.float32),  # operand
        #         jnp.array([[0], [2]], dtype=jnp.int32),  # indices
        #         jnp.ones((2, 3, 2), dtype=jnp.float32),  # updates
        #     ],
        #     "input_shapes": [
        #         (5, 3, 2),  # operand
        #         (2, 1),  # indices
        #         (2, 3, 2),  # updates
        #     ],
        #     "input_dtypes": [jnp.float32, jnp.int32, jnp.float32],
        # },
        # {
        #     "testcase": "scatter_add_windowed_from_original_failure_analogue",
        #     "callable": lambda operand, indices, updates: jax.lax.scatter_add(
        #         operand,
        #         indices,
        #         updates,
        #         jax.lax.ScatterDimensionNumbers(
        #             update_window_dims=(1, 2, 3),
        #             inserted_window_dims=(),
        #             scatter_dims_to_operand_dims=(0,),
        #         ),
        #     ),
        #     "input_values": [
        #         jnp.zeros((5, 201, 1, 1), dtype=jnp.float32),
        #         jnp.array([[0], [4]], dtype=jnp.int32),
        #         jnp.arange(2 * 201, dtype=jnp.float32).reshape(2, 201, 1, 1),
        #     ],
        #     "input_shapes": [
        #         (5, 201, 1, 1), (2, 1), (2, 201, 1, 1)
        #     ],
        #     "input_dtypes": [jnp.float32, jnp.int32, jnp.float32],
        # },
        # {
        #     "testcase": "scatter_add_windowed_inserted_dims_complex",
        #     # Operand: (depth, num_parallel_updates, window_rows, window_cols) e.g. (5, 10, 3, 2)
        #     # Indices: (num_scatters, 1) -> indices into 'depth'
        #     # Updates: (num_scatters, window_rows, window_cols) -> windows to scatter
        #     # We want to update slices of shape (window_rows, window_cols)
        #     # at specific 'depth' indices, across *all* 'num_parallel_updates'.
        #     "callable": lambda operand, indices, updates: jax.lax.scatter_add(
        #         operand, # (5, 10, 3, 2)
        #         indices, # e.g. [[0], [2]] shape (2,1)
        #         updates, # Now expects (2, 3, 2)
        #         jax.lax.ScatterDimensionNumbers(
        #             update_window_dims=(1, 2), # These are dims in `updates` non-batch part (3,2) that define the window shape
        #             inserted_window_dims=(1,), # This is an operand dim (dim 1 of size 10) that is *not* indexed by scatter_dims_to_operand_dims
        #                                         # and is *not* part of the window defined by update_window_dims from JAX updates.
        #                                         # The update is applied across this dimension.
        #             scatter_dims_to_operand_dims=(0,) # `indices` map to operand dim 0 (depth)
        #         ),
        #     ),
        #     "input_values": [
        #         jnp.zeros((5, 10, 3, 2), dtype=jnp.float32),     # operand
        #         jnp.array([[0], [2]], dtype=jnp.int32),          # indices
        #         jnp.ones((2, 3, 2), dtype=jnp.float32),          # CORRECTED updates shape
        #     ],
        #     "input_shapes": [ # (operand_shape, indices_shape, updates_shape)
        #         (5, 10, 3, 2),
        #         (2, 1),
        #         (2, 3, 2)  # CORRECTED updates shape
        #     ],
        #     "input_dtypes": [jnp.float32, jnp.int32, jnp.float32],
        # },
    ],
)
class ScatterAddPlugin(PrimitiveLeafPlugin):
    @staticmethod
    def abstract_eval(
        operand: core.ShapedArray,
        indices: core.ShapedArray,
        updates: core.ShapedArray,
        update_jaxpr,  # type: ignore
        update_consts,  # type: ignore
        *,
        dimension_numbers: ScatterDimensionNumbers,
        indices_are_sorted: bool,
        unique_indices: bool,
        mode: GatherScatterMode | None,
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
        out_name = s.get_name(out_v)

        operand_aval_shape = tuple(operand_v.aval.shape)
        operand_dtype_np = np.dtype(operand_v.aval.dtype)
        operand_rank = len(operand_aval_shape)

        # --- Step 1: Extract Dimension Numbers ---
        dim_numbers: ScatterDimensionNumbers = params["dimension_numbers"]
        jax_inserted_window_dims = dim_numbers.inserted_window_dims
        jax_scatter_dims_to_operand_dims = dim_numbers.scatter_dims_to_operand_dims
        # jax_operand_batching_dims = dim_numbers.operand_batching_dims # Not directly used in this ScatterND mapping logic
        # jax_scatter_indices_batching_dims = dim_numbers.scatter_indices_batching_dims # Not directly used

        # --- Process Indices (Aligns with User Plan Step 5) ---
        current_indices_name = s.get_name(indices_v)
        jax_indices_shape = list(indices_v.aval.shape)
        current_indices_dtype_np = np.dtype(indices_v.aval.dtype)

        if current_indices_dtype_np != np.int64:
            cast_indices_out_name = s.get_unique_name(f"{current_indices_name}_int64")
            s.add_node(
                helper.make_node(
                    "Cast",
                    inputs=[current_indices_name],
                    outputs=[cast_indices_out_name],
                    name=s.get_unique_name(f"cast_{current_indices_name}_to_int64"),
                    to=TensorProto.INT64,
                )
            )
            s.add_shape_info(cast_indices_out_name, tuple(jax_indices_shape), np.int64)
            current_indices_name = cast_indices_out_name
            current_indices_dtype_np = np.int64

        if not jax_indices_shape:  # Should not happen if JAX validation passed
            raise ValueError("JAX indices shape is empty.")
        index_depth = jax_indices_shape[-1]
        indices_batch_dims_shape = jax_indices_shape[:-1]

        num_updates = (
            np.prod(indices_batch_dims_shape).astype(np.int64).item()
            if indices_batch_dims_shape
            else 1
        )
        if any(d == 0 for d in indices_batch_dims_shape):
            num_updates = 0

        # Correct num_updates for scalar indices if original shape was like [K] instead of [1, K]
        if (
            not indices_batch_dims_shape and len(jax_indices_shape) == 1
        ):  # e.g. indices shape [3]
            # This case implies num_updates is 1, and index_depth is jax_indices_shape[0]
            # However, ScatterND expects (num_updates, index_depth).
            # If JAX indices are (K), it means K indices for a single update batch.
            # This situation needs careful mapping or should be caught by JAX rules.
            # For ONNX ScatterND, if indices are (K), it's 1 update of depth K.
            # If JAX indices are (N, K), it's N updates of depth K.
            # The current logic assumes JAX indices are (*batch_dims, index_vector_len)
            # If jax_indices_shape = [K] (rank 1), then indices_batch_dims_shape is empty, num_updates = 1, index_depth = K.
            pass

        onnx_indices_target_shape = (num_updates, index_depth)
        if tuple(jax_indices_shape) != onnx_indices_target_shape:
            reshape_indices_out_name = s.get_unique_name(
                f"{current_indices_name}_reshaped_for_scatternd"
            )
            reshape_indices_shape_const = np.array(
                onnx_indices_target_shape, dtype=np.int64
            )
            reshape_indices_shape_name = s.get_constant_name(
                reshape_indices_shape_const
            )
            s.add_node(
                helper.make_node(
                    "Reshape",
                    inputs=[current_indices_name, reshape_indices_shape_name],
                    outputs=[reshape_indices_out_name],
                    name=s.get_unique_name(
                        f"reshape_{current_indices_name}_for_scatternd"
                    ),
                )
            )
            s.add_shape_info(
                reshape_indices_out_name,
                onnx_indices_target_shape,
                current_indices_dtype_np,
            )
            current_indices_name = reshape_indices_out_name

        # --- Process Updates (Aligns with User Plan Steps 2, 3, 4) ---
        current_updates_name = s.get_name(updates_v)
        jax_updates_shape = list(updates_v.aval.shape)

        # Step 2 (partially): Identify JAX updates batch and window shapes
        # num_batch_dims_in_indices was calculated as len(indices_batch_dims_shape)
        num_jax_updates_batch_dims = len(indices_batch_dims_shape)
        jax_updates_actual_window_shape = tuple(
            jax_updates_shape[num_jax_updates_batch_dims:]
        )

        # Reshape JAX updates to have a flat batch dimension: (num_updates, *jax_updates_actual_window_shape)
        flat_batch_jax_updates_shape = (num_updates, *jax_updates_actual_window_shape)
        if tuple(jax_updates_shape) != flat_batch_jax_updates_shape:
            reshape_flat_batch_updates_out_name = s.get_unique_name(
                f"{current_updates_name}_flat_batch"
            )
            reshape_flat_batch_updates_shape_const = np.array(
                flat_batch_jax_updates_shape, dtype=np.int64
            )
            reshape_flat_batch_updates_shape_name = s.get_constant_name(
                reshape_flat_batch_updates_shape_const
            )
            s.add_node(
                helper.make_node(
                    "Reshape",
                    inputs=[
                        current_updates_name,
                        reshape_flat_batch_updates_shape_name,
                    ],
                    outputs=[reshape_flat_batch_updates_out_name],
                    name=s.get_unique_name(
                        f"reshape_{current_updates_name}_flat_batch"
                    ),
                )
            )
            s.add_shape_info(
                reshape_flat_batch_updates_out_name,
                flat_batch_jax_updates_shape,
                operand_dtype_np,
            )
            current_updates_name = reshape_flat_batch_updates_out_name
            # current_updates_shape = flat_batch_jax_updates_shape # Not needed as var

        # Step 3: Determine the correct ONNX updates tensor shape for ScatterND
        # The window shape for ONNX updates is the shape of the slice of the operand
        # that each individual update writes to.
        # Iterate through operand dims by their role (scattered, inserted, or part of update window)
        # This logic reconstructs the shape of the slice of the operand.
        # JAX's update_window_dims directly gives the shape of the JAX update window,
        # and inserted_window_dims tells us which operand dimensions are broadcast over.

        # The shape of a single slice written by ONNX ScatterND must match the operand dimensions
        # NOT indexed by scatter_dims_to_operand_dims.
        operand_dims_forming_slice = [
            i for i in range(operand_rank) if i not in jax_scatter_dims_to_operand_dims
        ]
        onnx_slice_shape = tuple(
            operand_aval_shape[i] for i in operand_dims_forming_slice
        )

        onnx_updates_target_shape = (num_updates, *onnx_slice_shape)

        # Step 4: Implement explicit broadcasting for ONNX if required (due to inserted_window_dims)
        # The current_updates_name holds data reshaped to (num_updates, *jax_updates_actual_window_shape)
        # If jax_updates_actual_window_shape is different from onnx_slice_shape, broadcasting is needed.
        if flat_batch_jax_updates_shape != onnx_updates_target_shape:
            # This implies inserted_window_dims caused jax_updates_actual_window_shape to be smaller
            # or have rank differences compared to onnx_slice_shape.

            # We need to Unsqueeze jax_updates to match rank of onnx_slice_shape,
            # placing 1s for dimensions that were inserted.

            # Example: operand (D, I, R, C), scatter_dim_op_dim=(0,), inserted_dims_op=(1,)
            # jax_updates_window_actual_shape is (R,C)
            # onnx_slice_shape is (I,R,C)
            # We need to unsqueeze (num_updates, R,C) to (num_updates, 1, R,C)

            # Determine axes to unsqueeze in the (num_updates, *jax_updates_actual_window_shape) tensor
            # to align with (num_updates, *onnx_slice_shape)

            # Simplified approach: If shapes don't match, assume Expand is needed.
            # The Expand operator in ONNX can handle broadcasting from lower rank if dims are 1 or missing.
            # Let's ensure the input to Expand has 1s where inserted_window_dims would be.

            # Construct the shape for unsqueezing:
            # Start with (num_updates, ...). For each dim in onnx_slice_shape:
            # if it corresponds to an inserted_window_dim (relative to operand_dims_forming_slice), shape is 1.
            # otherwise, it's from jax_updates_actual_window_shape.

            temp_unsqueezed_shape_list = [num_updates]
            iter(jax_updates_actual_window_shape)

            # operand_dims_forming_slice gives the original operand indices for onnx_slice_shape
            # jax_inserted_window_dims are original operand indices

            current_jax_window_dim_idx = 0
            for i, op_dim_idx_in_slice in enumerate(operand_dims_forming_slice):
                if op_dim_idx_in_slice in jax_inserted_window_dims:
                    temp_unsqueezed_shape_list.append(1)
                else:
                    if current_jax_window_dim_idx < len(
                        jax_updates_actual_window_shape
                    ):
                        temp_unsqueezed_shape_list.append(
                            jax_updates_actual_window_shape[current_jax_window_dim_idx]
                        )
                        current_jax_window_dim_idx += 1
                    else:
                        # This case should ideally not happen if JAX rules are consistent
                        raise ValueError(
                            f"Mismatch between JAX update window shape and operand slice structure for inserted dims. JAX win: {jax_updates_actual_window_shape}, ONNX slice: {onnx_slice_shape}"
                        )

            temp_unsqueezed_shape = tuple(temp_unsqueezed_shape_list)

            if (
                flat_batch_jax_updates_shape != temp_unsqueezed_shape
            ):  # If unsqueezing is actually needed
                unsqueezed_updates_name = s.get_unique_name(
                    f"{current_updates_name}_unsqueezed"
                )
                s.add_node(
                    helper.make_node(
                        "Reshape",  # Using Reshape to achieve unsqueeze
                        inputs=[
                            current_updates_name,
                            s.get_constant_name(
                                np.array(temp_unsqueezed_shape, dtype=np.int64)
                            ),
                        ],
                        outputs=[unsqueezed_updates_name],
                        name=s.get_unique_name(
                            f"unsqueeze_{current_updates_name}_for_expand"
                        ),
                    )
                )
                s.add_shape_info(
                    unsqueezed_updates_name, temp_unsqueezed_shape, operand_dtype_np
                )
                current_updates_name = unsqueezed_updates_name

            # Now, current_updates_name has shape `temp_unsqueezed_shape`. Expand it to `onnx_updates_target_shape`.
            if temp_unsqueezed_shape != onnx_updates_target_shape:
                expanded_updates_name = s.get_unique_name(
                    f"{current_updates_name}_expanded"
                )
                expand_target_shape_name = s.get_constant_name(
                    np.array(onnx_updates_target_shape, dtype=np.int64)
                )
                s.add_node(
                    helper.make_node(
                        "Expand",
                        inputs=[current_updates_name, expand_target_shape_name],
                        outputs=[expanded_updates_name],
                        name=s.get_unique_name(f"expand_{current_updates_name}"),
                    )
                )
                s.add_shape_info(
                    expanded_updates_name, onnx_updates_target_shape, operand_dtype_np
                )
                current_updates_name = expanded_updates_name
            # Else: no expansion needed after unsqueezing, or no unsqueezing needed.
            # This means current_updates_name (after flat_batch reshape) is already suitable or became suitable after unsqueeze.

        # --- Create ScatterND Node (Aligns with User Plan Step 6) ---
        scatter_nd_node = helper.make_node(
            "ScatterND",
            inputs=[operand_name, current_indices_name, current_updates_name],
            outputs=[out_name],
            name=s.get_unique_name("scatter_add_as_scatter_nd"),
            reduction="add",
        )
        s.add_node(scatter_nd_node)
        s.add_shape_info(out_name, operand_aval_shape, operand_dtype_np)
