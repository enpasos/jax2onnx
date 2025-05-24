# file: jax2onnx/plugins/jax/lax/scatter_add.py
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Any
import numpy as np
from jax import ShapeDtypeStruct, lax, core
from jax.lax import ScatterDimensionNumbers, GatherScatterMode  # Keep for params
from onnx import helper  # TensorProto might not be directly needed here anymore

# Import the new utility function
from .scatter_utils import (
    _are_shapes_equal,
    _ensure_np_dtype,
    _prepare_scatter_inputs_for_onnx,
)

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
    since="v0.5.3",  # Consider if version should be updated due to significant internal change
    context="primitives.lax",
    component="scatter_add",
    testcases=[  # Existing testcases are kept
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
        {
            "testcase": "scatter_add_mismatched_window_dims_from_user_report",
            "callable": lambda operand, indices, updates: lax.scatter_add(
                operand,
                indices,
                updates,
                dimension_numbers=lax.ScatterDimensionNumbers(
                    update_window_dims=(0, 1, 2, 3),
                    inserted_window_dims=(),
                    scatter_dims_to_operand_dims=(1,),
                    operand_batching_dims=(),
                    scatter_indices_batching_dims=(),
                ),
            ),
            "input_values": [
                np.zeros((5, 208, 1, 1), dtype=np.float64),
                np.array([4], dtype=np.int32),
                np.ones((5, 200, 1, 1), dtype=np.float64),
            ],
            "run_only_f64_variant": True,
            "expected_output_shapes": [(5, 208, 1, 1)],
            "expected_output_dtypes": [np.float64],
        },
        {
            "testcase": "scatter_add_mismatched_window_dims_from_user_report2",
            "callable": lambda operand, indices, updates: lax.scatter_add(
                operand,
                indices,
                updates,
                dimension_numbers=lax.ScatterDimensionNumbers(
                    update_window_dims=(0, 1, 2, 3),
                    inserted_window_dims=(),
                    scatter_dims_to_operand_dims=(1,),
                    operand_batching_dims=(),
                    scatter_indices_batching_dims=(),
                ),
            ),
            "input_values": [
                np.zeros((3, 150, 1, 1), dtype=np.float64),
                np.array([7], dtype=np.int32),
                np.ones((3, 140, 1, 1), dtype=np.float64),
            ],
            "run_only_f64_variant": True,
            "expected_output_shapes": [(3, 150, 1, 1)],
            "expected_output_dtypes": [np.float64],
        },
        {
            "testcase": "scatter_add_mismatched_window_dims_from_user_report3",
            "callable": lambda operand, indices, updates: lax.scatter_add(
                operand,
                indices,
                updates,
                dimension_numbers=lax.ScatterDimensionNumbers(
                    update_window_dims=(0, 1, 2, 3),
                    inserted_window_dims=(),
                    scatter_dims_to_operand_dims=(1,),
                    operand_batching_dims=(),
                    scatter_indices_batching_dims=(),
                ),
            ),
            "input_values": [
                np.zeros((8, 50, 1, 1), dtype=np.float64),
                np.array([2], dtype=np.int32),
                np.ones((8, 45, 1, 1), dtype=np.float64),
            ],
            "run_only_f64_variant": True,
            "expected_output_shapes": [(8, 50, 1, 1)],
            "expected_output_dtypes": [np.float64],
        },
        {
            "testcase": "scatter_add_fluids_pattern_updates_5_4_1_1",
            "callable": lambda operand, indices, updates: lax.scatter_add(
                operand,
                indices,
                updates,
                dimension_numbers=lax.ScatterDimensionNumbers(
                    update_window_dims=(0, 1, 2, 3),
                    inserted_window_dims=(),
                    scatter_dims_to_operand_dims=(
                        1,
                    ),  # JAX index targets axis 1 of operand
                    operand_batching_dims=(),
                    scatter_indices_batching_dims=(),
                ),
            ),
            "input_values": [
                # Operand shape (5, 208, 1, 1)
                np.zeros((5, 208, 1, 1), dtype=np.float64),
                # JAX indices: e.g., update starting at column index 0 of axis 1 for all batches
                np.array([0], dtype=np.int32),
                # JAX Updates shape (5, 4, 1, 1)
                np.ones((5, 4, 1, 1), dtype=np.float64),
            ],
            "run_only_f64_variant": True,
            "expected_output_shapes": [(5, 208, 1, 1)],  # Matches operand
            "expected_output_dtypes": [np.float64],
        },
    ],
)
class ScatterAddPlugin(PrimitiveLeafPlugin):
    @staticmethod
    def abstract_eval(
        operand: core.ShapedArray,
        indices: core.ShapedArray,
        updates: core.ShapedArray,
        update_jaxpr,  # Not directly used by scatter_add itself after preparation
        update_consts,  # Not directly used
        *,
        dimension_numbers: ScatterDimensionNumbers,  # type: ignore
        indices_are_sorted: bool,  # type: ignore
        unique_indices: bool,  # type: ignore
        mode: GatherScatterMode | None,  # type: ignore
    ):
        # Output shape and dtype match the operand
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
        out_name = s.get_name(out_v)

        operand_aval = operand_v.aval
        operand_shape = tuple(operand_aval.shape)
        operand_dtype_np = np.dtype(operand_aval.dtype)

        dimension_numbers: ScatterDimensionNumbers = params["dimension_numbers"]

        logger.info(
            f"Preparing inputs for ONNX ScatterND (reduction=add) for JAX scatter_add "
            f"primitive with dimension_numbers: {dimension_numbers}"
        )

        final_operand_name, final_indices_name, final_updates_name = (
            _prepare_scatter_inputs_for_onnx(
                s,
                operand_v,
                indices_v,
                updates_v,
                dimension_numbers,
            )
        )

        # This block was for debugging and ensuring inputs to ScatterND are known by the builder.
        # It should be fine if _prepare_scatter_inputs_for_onnx does its job.
        # The "Output info ... is None" error happens *after* ScatterND node is added.

        reduction_attribute = "add"
        node_attributes = {}
        if s.builder.opset >= 11:
            if s.builder.opset >= 13:
                node_attributes["reduction"] = reduction_attribute
            else:
                raise NotImplementedError(
                    f"ScatterND with reduction='{reduction_attribute}' requires ONNX opset 13+. "
                    f"Current opset: {s.builder.opset}. For scatter_add, this is essential."
                )
        else:
            raise NotImplementedError(
                f"ScatterND requires ONNX opset 11+. Current opset: {s.builder.opset}"
            )

        # Logging before ScatterND node creation
        if final_updates_name in s.shape_env:
            updates_info_for_scatternd_log = s.shape_env[final_updates_name]
            if isinstance(updates_info_for_scatternd_log, ShapeDtypeStruct):
                logger.debug(  # Changed from ERROR to DEBUG
                    f"[ScatterAddPlugin] Shape of updates ('{final_updates_name}') going into ScatterND: {updates_info_for_scatternd_log.shape}"
                )
            else:
                logger.warning(  # Changed from ERROR to WARNING
                    f"[ScatterAddPlugin] Updates info for '{final_updates_name}' (ScatterND input) in shape_env is type {type(updates_info_for_scatternd_log)}."
                )
        else:
            logger.error(  # This is genuinely an error if it happens
                f"[ScatterAddPlugin] Shape of updates ('{final_updates_name}') not found in shape_env before ScatterND creation!"
            )
        logger.debug(  # Changed from ERROR to DEBUG
            f"[ScatterAddPlugin] ScatterND inputs: data='{final_operand_name}', indices='{final_indices_name}', updates='{final_updates_name}'"
        )

        s.add_node(
            helper.make_node(
                "ScatterND",
                inputs=[final_operand_name, final_indices_name, final_updates_name],
                outputs=[out_name],
                name=s.get_unique_name(f"scatter_nd_add_{out_name}"),
                **node_attributes,
            )
        )

        # 3. Register final output shape and dtype
        # Ensure s.shape_env is explicitly populated with a ShapeDtypeStruct
        # operand_shape and operand_dtype_np are defined earlier from operand_v.aval

        current_out_info = s.shape_env.get(out_name)
        if (
            not isinstance(current_out_info, ShapeDtypeStruct)
            or not _are_shapes_equal(current_out_info.shape, operand_shape, s)
            or _ensure_np_dtype(current_out_info.dtype) != operand_dtype_np
        ):

            logger.debug(
                f"Re-registering/ensuring ShapeDtypeStruct for output '{out_name}' in shape_env."
            )
            sds_out = ShapeDtypeStruct(operand_shape, operand_dtype_np)
            s.shape_env[out_name] = sds_out
            # Also call the main add_shape_info, which might do other things like add to graph value_info
            s.add_shape_info(out_name, operand_shape, operand_dtype_np)

        # For debugging, verify after attempting to fix:
        final_output_info = s.shape_env.get(out_name)
        if not isinstance(final_output_info, ShapeDtypeStruct):
            logger.error(
                f"CRITICAL ERROR in {self.__class__.__name__}: Output info for '{out_name}' is type {type(final_output_info)} "
                f"not ShapeDtypeStruct in shape_env even after explicit set."
            )
        elif not (
            _are_shapes_equal(final_output_info.shape, operand_shape, s)
            and _ensure_np_dtype(final_output_info.dtype) == operand_dtype_np
        ):
            logger.error(
                f"CRITICAL ERROR in {self.__class__.__name__}: Output info for '{out_name}' {final_output_info} "
                f"does not match expected operand info ({operand_shape}, {operand_dtype_np})."
            )
        if isinstance(
            final_output_info, ShapeDtypeStruct
        ):  # Use ShapeDtypeStruct from jax, not core
            if not (
                _are_shapes_equal(final_output_info.shape, operand_shape, s)
                and _ensure_np_dtype(final_output_info.dtype) == operand_dtype_np
            ):
                logger.warning(
                    f"[ScatterAddPlugin] Final assertion mismatch for {out_name}. "
                    f"Env: {final_output_info.shape}/{final_output_info.dtype}, "
                    f"Expected: {operand_shape}/{operand_dtype_np}"
                )
                # Forcing re-registration if mismatch is found and was unexpected
                sds_out_corrected = ShapeDtypeStruct(operand_shape, operand_dtype_np)
                s.shape_env[out_name] = sds_out_corrected
                s.add_shape_info(
                    out_name, operand_shape, operand_dtype_np
                )  # Re-call add_shape_info too
        elif final_output_info is not None:
            logger.warning(
                f"[ScatterAddPlugin] Output info for {out_name} in shape_env is type {type(final_output_info)}, not ShapeDtypeStruct."
            )
        else:
            logger.error(
                f"[ScatterAddPlugin] CRITICAL: Output info for {out_name} is None in shape_env AFTER attempting registration."
            )
        # --- END OF MODIFIED SECTION ---
