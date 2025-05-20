# file: jax2onnx/plugins/jax/lax/scatter_add.py
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Any
import numpy as np
from jax import lax, core
from jax.lax import ScatterDimensionNumbers, GatherScatterMode  # Keep for params
from onnx import helper  # TensorProto might not be directly needed here anymore

# Import the new utility function
from .scatter_utils import _prepare_scatter_inputs_for_onnx

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

        # Original operand shape and dtype for final output registration
        operand_aval = operand_v.aval
        operand_shape = tuple(operand_aval.shape)
        operand_dtype_np = np.dtype(operand_aval.dtype)

        dimension_numbers: ScatterDimensionNumbers = params["dimension_numbers"]
        # mode_param = params.get("mode") # JAX `mode` (e.g. CLIP) might need handling
        # if ScatterND reduction doesn't cover it.
        # For 'add', ScatterND handles out-of-bounds by ignoring.

        logger.info(
            f"Preparing inputs for ONNX ScatterND (reduction=add) for JAX scatter_add "
            f"primitive with dimension_numbers: {dimension_numbers}"
        )

        # 1. Call the common utility function to prepare inputs for ScatterND
        final_operand_name, final_indices_name, final_updates_name = (
            _prepare_scatter_inputs_for_onnx(
                s,  # Jaxpr2OnnxConverter instance
                operand_v,
                indices_v,
                updates_v,
                dimension_numbers,  # Pass the dimension_numbers directly
            )
        )

        # --- BEGIN MODIFICATION ---
        # Explicitly ensure shape_info for all inputs to ScatterND is in converter.shape_env,
        # using the builder's metadata as the source of truth.
        input_names_to_ensure = {
            "operand": (final_operand_name, operand_v.aval.dtype),
            "indices": (
                final_indices_name,
                np.int64,
            ),  # Indices are expected to be int64
            "updates": (
                final_updates_name,
                updates_v.aval.dtype,
            ),  # Original dtype of updates
        }

        for role, (name_to_check, expected_np_dtype) in input_names_to_ensure.items():
            try:
                # Query the builder for the canonical shape and ONNX dtype enum
                shape_from_builder, dtype_enum_from_builder = s.builder.get_shape_dtype(
                    name_to_check
                )

                # Ensure this info is in the converter's shape_env.
                # s.add_shape_info can take np.dtype or ONNX dtype enum.
                s.add_shape_info(
                    name_to_check, shape_from_builder, dtype_enum_from_builder
                )

                logger.debug(
                    f"[ScatterAddPlugin] Ensured/updated shape_info for {role} '{name_to_check}': "
                    f"shape={shape_from_builder}, builder_dtype_enum={dtype_enum_from_builder}"
                )
            except ValueError as e:
                # This means the builder doesn't have metadata for this tensor name.
                # This is critical, especially if _prepare_scatter_inputs_for_onnx was supposed to register it.
                # The pytest logs show register_value_info_metadata *is* called for the final updates tensor,
                # so builder.get_shape_dtype should succeed for it.
                logger.error(
                    f"[ScatterAddPlugin] CRITICAL: Could not get shape/dtype for {role} '{name_to_check}' "
                    f"from builder: {e}. Conversion may fail or be incorrect."
                )
                # As a last resort, if it's an input from Jaxpr, its original aval might be used,
                # but for processed tensors (like final_updates_name), this is insufficient.
                if (
                    name_to_check not in s.shape_env
                ):  # If also not in shape_env after this
                    logger.error(
                        f"[ScatterAddPlugin] '{name_to_check}' remains without info in shape_env."
                    )

        # Log details before creating ScatterND
        # This check led to the original error message. It should ideally pass now for final_updates_name.
        if final_updates_name in s.shape_env:
            logger.info(
                f"[ScatterAddPlugin] Shape of updates ('{final_updates_name}') in shape_env before ScatterND creation: "
                f"{s.shape_env[final_updates_name].shape}, dtype: {s.shape_env[final_updates_name].dtype}"
            )
        else:
            logger.error(
                f"[ScatterAddPlugin] Shape of updates ('{final_updates_name}') *still* not found in shape_env "
                f"before ScatterND creation! This indicates a persistent issue."
            )
        # --- END MODIFICATION ---

        # 2. Create the ONNX ScatterND node with reduction="add"
        reduction_attribute = "add"

        node_attributes = {}
        # ScatterND introduced in opset 11. Reduction attribute in opset 13 (for add, mul)
        # and expanded in opset 16 (min, max, none).
        if s.builder.opset >= 11:  # ScatterND exists
            if s.builder.opset >= 13:  # 'add' reduction is available
                node_attributes["reduction"] = reduction_attribute
            # If opset is 11 or 12, 'reduction' attribute is not supported.
            # The default behavior of ScatterND (opset 11-12) is 'none' (update/overwrite).
            # It does NOT default to 'add'. So, direct conversion for 'add' requires opset >= 13.
            else:  # opset 11 or 12
                raise NotImplementedError(
                    f"ScatterND with reduction='{reduction_attribute}' requires ONNX opset 13+. "
                    f"Current opset: {s.builder.opset}. For scatter_add, this is essential."
                )
        else:  # opset < 11
            raise NotImplementedError(
                f"ScatterND requires ONNX opset 11+. Current opset: {s.builder.opset}"
            )

        logger.error(
            f"[ScatterAddPlugin] ScatterND inputs: data='{final_operand_name}', indices='{final_indices_name}', updates='{final_updates_name}'"
        )
        if final_updates_name in s.shape_env:
            logger.error(
                f"[ScatterAddPlugin] Shape of updates ('{final_updates_name}') going into ScatterND: {s.shape_env[final_updates_name].shape}"
            )
        else:
            logger.error(
                f"[ScatterAddPlugin] Shape of updates ('{final_updates_name}') not found in shape_env before ScatterND creation!"
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
        if out_name not in s.shape_env:
            s.add_shape_info(out_name, operand_shape, operand_dtype_np)
        else:
            assert s.shape_env[out_name].shape == operand_shape
            assert s.shape_env[out_name].dtype == operand_dtype_np
