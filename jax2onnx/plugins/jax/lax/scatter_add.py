# file: jax2onnx/plugins/jax/lax/scatter_add.py
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Any
import numpy as np
from jax import (
    lax,
    core,
)
from jax.lax import ScatterDimensionNumbers, GatherScatterMode
from onnx import helper

# Correctly import from the plugin_system module based on your provided files
from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive
from .scatter_utils import _prepare_scatter_inputs_for_onnx
import logging

if TYPE_CHECKING:
    # This is the correct way to type hint the converter
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

logger = logging.getLogger("jax2onnx.plugins.jax.lax.scatter_add")


@register_primitive(
    jaxpr_primitive=lax.scatter_add_p.name,
    # Metadata for documentation is retained
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
        {
            "testcase": "scatter_add_in_cond_float64",
            # This model places the scatter_add call within a lax.cond branch.
            # All arguments are passed through the conditional branches.
            "callable": lambda pred, operand, indices, updates: lax.cond(
                pred,
                # True branch: performs the scatter_add
                lambda op, idx, upd: lax.scatter_add(
                    op,
                    idx,
                    upd,
                    dimension_numbers=lax.ScatterDimensionNumbers(
                        # These dimension numbers are known to trigger complex logic
                        # in the scatter plugin.
                        update_window_dims=(0, 1, 2, 3),
                        inserted_window_dims=(),
                        scatter_dims_to_operand_dims=(1,),
                        operand_batching_dims=(),
                        scatter_indices_batching_dims=(),
                    ),
                ),
                # False branch: returns the operand unchanged.
                lambda op, idx, upd: op,
                # Operands for lax.cond:
                operand,
                indices,
                updates,
            ),
            # Input data with valid shapes for the scatter operation.
            "input_values": [
                np.array(True),  # The predicate for lax.cond
                np.zeros((8, 50, 1, 1), dtype=np.float64),  # operand
                np.array([2], dtype=np.int32),  # indices
                np.ones((8, 45, 1, 1), dtype=np.float64),  # updates
            ],
            "run_only_f64_variant": True,
        },
        # {
        #     "testcase": "scatter_add_batched_indices_strategy_f64",
        #     "callable": lambda op, ind, upd: lax.scatter_add(
        #         op,
        #         ind,
        #         upd,
        #         dimension_numbers=ScatterDimensionNumbers(
        #             update_window_dims=(2,),
        #             inserted_window_dims=(1,),
        #             scatter_dims_to_operand_dims=(0,),
        #         ),
        #     ),
        #     "input_values": [
        #         np.zeros((5, 4), dtype=np.float64),
        #         np.array([[[1]], [[3]]], dtype=np.int64),
        #         np.ones((2, 1, 4), dtype=np.float64) * 9.0,
        #     ],
        #     "run_only_f64_variant": True,
        # },
    ],
)
class ScatterAddPlugin(PrimitiveLeafPlugin):
    """
    ONNX conversion for the lax.scatter_add primitive.
    """

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
        """
        The abstract evaluation of scatter_add returns an array with the same
        shape and dtype as the operand.
        """
        return core.ShapedArray(operand.shape, operand.dtype)

    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: Sequence[Any],  # These are jax.core.Var or jax.core.Literal
        node_outputs: Sequence[Any],
        params: dict[str, Any],
    ):
        """
        Converts the JAX scatter_add operation to an ONNX ScatterND node.
        """
        operand_v, indices_v, updates_v = node_inputs
        out_v = node_outputs[0]
        out_name = s.get_name(out_v)
        operand_aval = operand_v.aval

        dimension_numbers: ScatterDimensionNumbers = params["dimension_numbers"]
        logger.info(
            f"Converting lax.scatter_add with dimension_numbers: {dimension_numbers}"
        )

        # This utility function contains the core fix and returns the final ONNX tensor names
        final_operand_name, final_indices_name, final_updates_name = (
            _prepare_scatter_inputs_for_onnx(
                s, operand_v, indices_v, updates_v, dimension_numbers
            )
        )

        logger.debug(
            "[ScatterAddPlugin] Final inputs to ScatterND: operand=%s, indices=%s, updates=%s",
            final_operand_name,
            final_indices_name,
            final_updates_name,
        )

        # Create the ONNX ScatterND node with the 'add' reduction.
        s.add_node(
            helper.make_node(
                "ScatterND",
                inputs=[final_operand_name, final_indices_name, final_updates_name],
                outputs=[out_name],
                name=s.get_unique_name(f"scatter_nd_add_{out_name}"),
                reduction="add",
            )
        )

        # The output shape of scatter_add is always the same as the operand's shape.
        s.add_shape_info(out_name, operand_aval.shape, operand_aval.dtype)
