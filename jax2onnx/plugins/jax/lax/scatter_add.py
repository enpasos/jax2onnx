# jax2onnx/plugins/jax/lax/scatter_add.py
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Any
import numpy as np
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
    ],
)
class ScatterAddPlugin(PrimitiveLeafPlugin):
    @staticmethod
    def abstract_eval(
        operand: core.ShapedArray,
        indices: core.ShapedArray,
        updates: core.ShapedArray,
        update_jaxpr,
        update_consts,
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

        # --- Handle Indices ---
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

        # Reshape indices for ScatterND: from JAX (*batch_dims, index_vector_len) to ONNX (num_updates, index_depth)
        if not jax_indices_shape:
            raise ValueError("JAX indices shape is empty.")
        index_depth = jax_indices_shape[-1]
        indices_batch_dims_shape = jax_indices_shape[:-1]

        num_updates = (
            np.prod(indices_batch_dims_shape).astype(np.int64)
            if indices_batch_dims_shape
            else 1
        )
        if isinstance(num_updates, np.ndarray):
            num_updates = num_updates.item()
        # Handle case where prod of empty shape is 1, but should be 0 if any dim was 0
        if not indices_batch_dims_shape and jax_indices_shape == [
            0
        ]:  # Special case for single index input that might be empty overall
            num_updates = 0
        elif any(
            d == 0 for d in indices_batch_dims_shape
        ):  # If any batch dim is 0, num_updates is 0
            num_updates = 0

        target_indices_shape_for_onnx = (num_updates, index_depth)

        if tuple(jax_indices_shape) != target_indices_shape_for_onnx:
            reshape_indices_out_name = s.get_unique_name(
                f"{current_indices_name}_reshaped_for_scatternd"
            )
            reshape_indices_shape_const = np.array(
                target_indices_shape_for_onnx, dtype=np.int64
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
                reshape_indices_out_name, target_indices_shape_for_onnx, np.int64
            )
            current_indices_name = reshape_indices_out_name

        # --- Handle Updates ---
        current_updates_name = s.get_name(updates_v)
        jax_updates_shape = list(updates_v.aval.shape)

        # Determine the shape of a single update "slice" from JAX `updates`
        # These are the dimensions of the `updates` tensor excluding its batch dimensions.
        # The number of batch dimensions in `updates` should match those in `indices`.
        num_batch_dims_in_updates = len(
            indices_batch_dims_shape
        )  # Same as indices batch rank

        if len(jax_updates_shape) < num_batch_dims_in_updates:
            raise ValueError(
                f"JAX updates rank {len(jax_updates_shape)} is less than "
                f"number of batch dimensions in JAX indices ({num_batch_dims_in_updates}). Updates shape: {jax_updates_shape}, Indices shape: {jax_indices_shape}"
            )

        jax_updates_window_part_shape = jax_updates_shape[num_batch_dims_in_updates:]
        target_updates_shape_for_onnx = tuple(
            [num_updates] + jax_updates_window_part_shape
        )

        if tuple(jax_updates_shape) != target_updates_shape_for_onnx:
            if np.prod(jax_updates_shape) != np.prod(target_updates_shape_for_onnx):
                raise ValueError(
                    f"Cannot reshape JAX updates from {jax_updates_shape} to target ONNX shape {target_updates_shape_for_onnx}. "
                    f"Product of elements mismatch: {np.prod(jax_updates_shape)} vs {np.prod(target_updates_shape_for_onnx)}."
                )

            reshape_updates_out_name = s.get_unique_name(
                f"{current_updates_name}_reshaped_for_scatternd_final"
            )
            reshape_updates_shape_const = np.array(
                target_updates_shape_for_onnx, dtype=np.int64
            )
            reshape_updates_shape_name = s.get_constant_name(
                reshape_updates_shape_const
            )

            s.add_node(
                helper.make_node(
                    "Reshape",
                    inputs=[current_updates_name, reshape_updates_shape_name],
                    outputs=[reshape_updates_out_name],
                    name=s.get_unique_name(
                        f"reshape_{current_updates_name}_for_scatternd_final"
                    ),
                )
            )
            s.add_shape_info(
                reshape_updates_out_name,
                target_updates_shape_for_onnx,
                operand_dtype_np,
            )
            current_updates_name = reshape_updates_out_name

        # --- Create ScatterND Node ---
        scatter_nd_node = helper.make_node(
            "ScatterND",
            inputs=[operand_name, current_indices_name, current_updates_name],
            outputs=[out_name],
            name=s.get_unique_name("scatter_add_as_scatter_nd"),
            reduction="add",
        )
        s.add_node(scatter_nd_node)
        s.add_shape_info(out_name, operand_aval_shape, operand_dtype_np)
