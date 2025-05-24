# file: jax2onnx/plugins/jax/lax/scatter_add.py
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Any
import numpy as np
from jax import (
    lax,
    core,
    ShapeDtypeStruct,
)  # Ensure ShapeDtypeStruct is imported directly from jax
from jax.lax import ScatterDimensionNumbers, GatherScatterMode
from onnx import helper

from .scatter_utils import _prepare_scatter_inputs_for_onnx

# If _ensure_np_dtype and _are_shapes_equal are needed for assertions, they'd also need importing or inlining.

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

import logging

logger = logging.getLogger("jax2onnx.plugins.jax.lax.scatter_add")

scatter_add_p = lax.scatter_add_p


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
                s, operand_v, indices_v, updates_v, dimension_numbers
            )
        )

        # --- Logging Block for Inputs to ScatterND ---
        # Ensure these logging calls robustly check for presence in shape_env and attribute existence
        for role, name_to_check in [
            ("operand", final_operand_name),
            ("indices", final_indices_name),
            ("updates", final_updates_name),
        ]:
            info = s.shape_env.get(name_to_check)
            if info is not None and hasattr(info, "shape") and hasattr(info, "dtype"):
                logger.debug(
                    f"[ScatterAddPlugin] Input '{role}' ('{name_to_check}') for ScatterND: "
                    f"shape={info.shape}, dtype={info.dtype}"
                )
            else:
                if info is None:
                    logger.warning(
                        f"[ScatterAddPlugin] Input '{role}' ('{name_to_check}') for ScatterND: "
                        f"Info is None in shape_env."
                    )
                else:
                    logger.warning(
                        f"[ScatterAddPlugin] Input '{role}' ('{name_to_check}') for ScatterND: "
                        f"Info not a valid ShapeDtypeStruct in shape_env (type: {type(info)})."
                    )
        # --- End Logging Block ---

        reduction_attribute = "add"
        node_attributes = {}
        if s.builder.opset >= 11:
            if s.builder.opset >= 13:
                node_attributes["reduction"] = reduction_attribute
            else:
                raise NotImplementedError(
                    f"ScatterND with reduction='{reduction_attribute}' requires ONNX opset 13+. "
                    f"Current opset: {s.builder.opset}."
                )
        else:
            raise NotImplementedError(
                f"ScatterND requires ONNX opset 11+. Current opset: {s.builder.opset}"
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

        # 3. Register final output shape and dtype robustly
        # operand_shape and operand_dtype_np are for the output, derived from operand_v.aval

        sds_out = ShapeDtypeStruct(operand_shape, operand_dtype_np)
        s.shape_env[out_name] = (
            sds_out  # Explicitly populate s.shape_env with the consistently imported ShapeDtypeStruct
        )
        s.add_shape_info(
            out_name, operand_shape, operand_dtype_np
        )  # Ensure builder's value_info is also updated

        logger.debug(
            f"[{self.__class__.__name__}] Ensured s.shape_env and called add_shape_info for ScatterND output '{out_name}' with {sds_out}"
        )

        # More robust verification using hasattr (duck typing)
        output_info_final_check = s.shape_env.get(out_name)
        if output_info_final_check is None:
            logger.error(
                f"CRITICAL ERROR in {self.__class__.__name__}: Output info for '{out_name}' is None in shape_env AFTER explicit set."
            )
        elif not (
            hasattr(output_info_final_check, "shape")
            and hasattr(output_info_final_check, "dtype")
            and output_info_final_check.shape is not None
            and output_info_final_check.dtype is not None
        ):
            logger.error(
                f"CRITICAL ERROR in {self.__class__.__name__}: Output info for '{out_name}' (type: {type(output_info_final_check)}) "
                f"in shape_env AFTER explicit set is not ShapeDtypeStruct-like (missing .shape or .dtype)."
            )
        else:
            # Only check .shape and .dtype if output_info_final_check is not None and has those attributes
            if not (
                output_info_final_check.shape == operand_shape
                and np.dtype(output_info_final_check.dtype) == operand_dtype_np
            ):
                logger.warning(
                    f"[{self.__class__.__name__}] Final verification mismatch for {out_name}. "
                    f"Env: {output_info_final_check.shape}/{output_info_final_check.dtype}, "
                    f"Expected: {operand_shape}/{operand_dtype_np}. This might be due to symbolic shapes if dynamically resolved."
                )
            else:
                logger.debug(
                    f"[{self.__class__.__name__}] Output info for '{out_name}' verified in shape_env."
                )
