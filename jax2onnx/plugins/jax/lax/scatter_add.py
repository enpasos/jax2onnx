# file: jax2onnx/plugins/jax/lax/scatter_add.py
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Any
import numpy as np
from jax import lax, core
from jax.lax import ScatterDimensionNumbers, GatherScatterMode
from onnx import helper, TensorProto

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # only for static type checkers
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

import logging

logger = logging.getLogger("jax2onnx.plugins.jax.lax.scatter_add")
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
                operand,
                indices,
                updates,
                dimension_numbers=ScatterDimensionNumbers(
                    update_window_dims=(1,),
                    inserted_window_dims=(0,),
                    scatter_dims_to_operand_dims=(0,),
                ),
            ),
            "input_values": [
                np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
                np.array([[0]], dtype=np.int32),
                np.array([[10.0, 20.0, 30.0]], dtype=np.float32),
            ],
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
                np.array([[[0], [1]], [[0], [2]]], dtype=np.int32),
                np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
            ],
        },
        # {
        #     "testcase": "scatter_add_windowed_from_original_failure_analogue",
        #     "callable": lambda operand, indices, updates: lax.scatter_add(
        #         operand,
        #         indices,
        #         updates,
        #         ScatterDimensionNumbers(
        #             update_window_dims=(0, 1, 2),
        #             inserted_window_dims=(),
        #             scatter_dims_to_operand_dims=(0,),
        #             operand_batching_dims=(1,),
        #             scatter_indices_batching_dims=(0,) # Corrected: Fixes count error, will hit shape error
        #         ),
        #     ),
        #     "input_values": [
        #         jnp.zeros((5, 201, 1, 1), dtype=jnp.float32),
        #         jnp.array([[0], [4]], dtype=jnp.int32),
        #         jnp.arange(2 * 201, dtype=jnp.float32).reshape(2, 201, 1, 1),
        #     ],
        #     "input_shapes": [(5, 201, 1, 1), (2, 1), (2, 201, 1, 1)],
        #     "input_dtypes": [jnp.float32, jnp.int32, jnp.float32],
        # },
        # {
        #     "testcase": "scatter_add_windowed_simple",
        #     "callable": lambda operand, indices, updates: jax.lax.scatter_add(
        #         operand,
        #         indices,
        #         updates,
        #         jax.lax.ScatterDimensionNumbers(
        #             update_window_dims=(0,1),
        #             inserted_window_dims=(),
        #             scatter_dims_to_operand_dims=(0,),
        #             operand_batching_dims=(2,), # Corrected: operand.shape[2] (2)
        #             scatter_indices_batching_dims=(0,) # Corrected: indices.shape[0] (2). Shapes match.
        #         ),
        #     ),
        #     "input_values": [
        #         jnp.zeros((5, 3, 2), dtype=jnp.float32),
        #         jnp.array([[0], [2]], dtype=jnp.int32),
        #         jnp.ones((2, 3, 2), dtype=jnp.float32),
        #     ],
        #     "input_shapes": [(5, 3, 2), (2, 1), (2, 3, 2)],
        #     "input_dtypes": [jnp.float32, jnp.int32, jnp.float32],
        # },
        # {
        #     "testcase": "scatter_add_windowed_inserted_dims_complex",
        #     "callable": lambda operand, indices, updates: jax.lax.scatter_add(
        #         operand,
        #         indices,
        #         updates,
        #         jax.lax.ScatterDimensionNumbers(
        #             update_window_dims=(0,1),      # len 2
        #             inserted_window_dims=(1,),     # len 1
        #             scatter_dims_to_operand_dims=(0,),
        #             operand_batching_dims=(2,),      # len 1. Sum for rank = 2+1+1=4. OK.
        #             scatter_indices_batching_dims=(0,) # len 1. Batch counts OK.
        #                                                # Next JAX error: Batch shapes op.shape[2](3) vs idx.shape[0](2)
        #         ),
        #     ),
        #     "input_values": [
        #         jnp.zeros((5, 10, 3, 2), dtype=jnp.float32),
        #         jnp.array([[0], [2]], dtype=jnp.int32),
        #         jnp.ones((2, 3, 2), dtype=jnp.float32),
        #     ],
        #     "input_shapes": [ (5, 10, 3, 2), (2, 1), (2, 3, 2)],
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
        operand_rank = len(operand_aval_shape)

        dim_numbers: ScatterDimensionNumbers = params["dimension_numbers"]

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

        if not jax_indices_shape:
            raise ValueError("JAX indices shape is empty for scatter_add.")
        index_depth = jax_indices_shape[-1]
        indices_batch_dims_shape = jax_indices_shape[:-1]

        num_updates = (
            np.prod(indices_batch_dims_shape).astype(np.int64).item()
            if indices_batch_dims_shape
            else 1
        )
        if any(d == 0 for d in indices_batch_dims_shape) or (
            not indices_batch_dims_shape and jax_indices_shape == [0]
        ):
            num_updates = 0

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
                reshape_indices_out_name, onnx_indices_target_shape, np.int64
            )
            current_indices_name = reshape_indices_out_name

        current_updates_name = s.get_name(updates_v)
        jax_updates_shape = list(updates_v.aval.shape)

        operand_dims_forming_slice = [
            i
            for i in range(operand_rank)
            if i not in dim_numbers.scatter_dims_to_operand_dims
        ]
        onnx_update_window_target_shape = tuple(
            operand_aval_shape[i] for i in operand_dims_forming_slice
        )
        onnx_updates_target_shape = (num_updates, *onnx_update_window_target_shape)

        num_jax_updates_batch_dims = len(indices_batch_dims_shape)
        jax_updates_actual_window_shape = tuple(
            jax_updates_shape[num_jax_updates_batch_dims:]
        )

        flat_batch_jax_updates_shape = (num_updates, *jax_updates_actual_window_shape)

        if tuple(jax_updates_shape) != flat_batch_jax_updates_shape:
            reshaped_flat_batch_updates_name = s.get_unique_name(
                f"{current_updates_name}_flat_batch"
            )
            s.add_node(
                helper.make_node(
                    "Reshape",
                    inputs=[
                        current_updates_name,
                        s.get_constant_name(
                            np.array(flat_batch_jax_updates_shape, dtype=np.int64)
                        ),
                    ],
                    outputs=[reshaped_flat_batch_updates_name],
                    name=s.get_unique_name(
                        f"reshape_{current_updates_name}_flat_batch"
                    ),
                )
            )
            s.add_shape_info(
                reshaped_flat_batch_updates_name,
                flat_batch_jax_updates_shape,
                operand_dtype_np,
            )
            current_updates_name = reshaped_flat_batch_updates_name

        if flat_batch_jax_updates_shape != onnx_updates_target_shape:
            shape_for_unsqueeze_step_list = [num_updates]
            consumed_jax_window_dims = 0

            for i_op_dim_in_slice, op_dim_in_slice_idx in enumerate(
                operand_dims_forming_slice
            ):
                if op_dim_in_slice_idx in dim_numbers.inserted_window_dims:
                    shape_for_unsqueeze_step_list.append(1)
                else:
                    if consumed_jax_window_dims < len(jax_updates_actual_window_shape):
                        shape_for_unsqueeze_step_list.append(
                            jax_updates_actual_window_shape[consumed_jax_window_dims]
                        )
                        consumed_jax_window_dims += 1
                    else:
                        if (
                            not jax_updates_actual_window_shape
                            and len(onnx_update_window_target_shape) > i_op_dim_in_slice
                            and onnx_update_window_target_shape[i_op_dim_in_slice] == 1
                        ):
                            shape_for_unsqueeze_step_list.append(1)
                        elif (
                            not jax_updates_actual_window_shape
                            and len(onnx_update_window_target_shape) == 0
                        ):
                            pass
                        else:
                            raise NotImplementedError(
                                f"Error in scatter_add: Mismatch in window dimension structure for ScatterND. "
                                f"Operand slice dim {op_dim_in_slice_idx} (part of target ONNX window {onnx_update_window_target_shape}) "
                                f"not covered by JAX updates window {jax_updates_actual_window_shape} "
                                f"and not in inserted_dims {dim_numbers.inserted_window_dims}."
                            )

            if consumed_jax_window_dims != len(jax_updates_actual_window_shape):
                if not (
                    len(jax_updates_actual_window_shape) == 0
                    and consumed_jax_window_dims == 0
                ):
                    raise NotImplementedError(
                        f"Error in scatter_add: Not all JAX update window dimensions {jax_updates_actual_window_shape} were mapped "
                        f"to ONNX slice shape {onnx_update_window_target_shape} "
                        f"with inserted dims {dim_numbers.inserted_window_dims}."
                    )

            shape_for_unsqueeze_step = tuple(shape_for_unsqueeze_step_list)

            if flat_batch_jax_updates_shape != shape_for_unsqueeze_step:
                unsqueezed_updates_name = s.get_unique_name(
                    f"{current_updates_name}_unsqueeze_for_expand"
                )
                s.add_node(
                    helper.make_node(
                        "Reshape",
                        inputs=[
                            current_updates_name,
                            s.get_constant_name(
                                np.array(shape_for_unsqueeze_step, dtype=np.int64)
                            ),
                        ],
                        outputs=[unsqueezed_updates_name],
                        name=s.get_unique_name(f"unsqueeze_{current_updates_name}"),
                    )
                )
                s.add_shape_info(
                    unsqueezed_updates_name, shape_for_unsqueeze_step, operand_dtype_np
                )
                current_updates_name = unsqueezed_updates_name

            if shape_for_unsqueeze_step != onnx_updates_target_shape:
                expanded_updates_name = s.get_unique_name(
                    f"{current_updates_name}_expanded_to_target"
                )
                s.add_node(
                    helper.make_node(
                        "Expand",
                        inputs=[
                            current_updates_name,
                            s.get_constant_name(
                                np.array(onnx_updates_target_shape, dtype=np.int64)
                            ),
                        ],
                        outputs=[expanded_updates_name],
                        name=s.get_unique_name(f"expand_{current_updates_name}"),
                    )
                )
                s.add_shape_info(
                    expanded_updates_name, onnx_updates_target_shape, operand_dtype_np
                )
                current_updates_name = expanded_updates_name

        scatter_nd_node = helper.make_node(
            "ScatterND",
            inputs=[operand_name, current_indices_name, current_updates_name],
            outputs=[out_name],
            name=s.get_unique_name("scatter_add_as_scatter_nd"),
            reduction="add",
        )
        s.add_node(scatter_nd_node)
        s.add_shape_info(out_name, operand_aval_shape, operand_dtype_np)
