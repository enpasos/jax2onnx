# jax2onnx/plugins/jax/lax/scatter.py

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax.scatter_utils import lower_scatter_common
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


@register_primitive(
    jaxpr_primitive=jax.lax.scatter_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.scatter.html",
    onnx=[
        {
            "component": "ScatterND",
            "doc": "https://onnx.ai/onnx/operators/onnx__ScatterND.html",
        }
    ],
    since="v0.4.4",
    context="primitives.lax",
    component="scatter",
    testcases=[
        {
            "testcase": "scatter_set_axis0",
            "callable": lambda x: x.at[0].set(jnp.array(-100.0, dtype=x.dtype)),
            "input_shapes": [(1, 1)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "ScatterND",
                        "inputs": {2: {"const": -100.0}},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "scatter_set_middle",
            "callable": lambda x: x.at[1].set(jnp.array(42.0, dtype=x.dtype)),
            "input_shapes": [(3,)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "ScatterND",
                        "inputs": {2: {"const": 42.0}},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "scatter_set_single",
            "callable": lambda x: x.at[0].set(jnp.array(-1.0, dtype=x.dtype)),
            "input_shapes": [(4,)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "ScatterND",
                        "inputs": {2: {"const": -1.0}},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "scatter_set_vector",
            "callable": lambda x: x.at[jnp.array([1, 3], dtype=jnp.int32)].set(
                jnp.array([10.0, 20.0], dtype=x.dtype)
            ),
            "input_shapes": [(5,)],
            "post_check_onnx_graph": EG(
                ["ScatterND"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "scatter_correct_axis_determination",
            "callable": lambda op, idx, upd_scalar_batch: jax.lax.scatter(
                op,
                idx,
                jnp.reshape(upd_scalar_batch, idx.shape[:-1]),
                jax.lax.ScatterDimensionNumbers(
                    update_window_dims=(),
                    inserted_window_dims=(0,),
                    scatter_dims_to_operand_dims=(0,),
                ),
            ),
            "input_shapes": [(5,), (1, 1, 1, 1), (1,)],
            "input_dtypes": [jnp.float32, jnp.int32, jnp.float32],
            "post_check_onnx_graph": EG(
                ["ScatterND"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "scatter_updates_slice_needed_axis0",
            "callable": lambda op, idx, upd_scalar_batch: jax.lax.scatter(
                op,
                idx,
                jnp.reshape(upd_scalar_batch, idx.shape[:-1]),
                jax.lax.ScatterDimensionNumbers(
                    update_window_dims=(),
                    inserted_window_dims=(0,),
                    scatter_dims_to_operand_dims=(0,),
                ),
            ),
            "input_shapes": [(5,), (1, 1, 1, 1), (1,)],
            "input_dtypes": [jnp.float32, jnp.int32, jnp.float32],
            "post_check_onnx_graph": EG(
                ["ScatterND"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "scatter_from_user_warning_shapes_valid_jax",
            "callable": lambda operand, indices, updates_sliced_scalar_batch: jax.lax.scatter(
                operand,
                indices,
                jnp.reshape(updates_sliced_scalar_batch, indices.shape[:-1]),
                jax.lax.ScatterDimensionNumbers(
                    update_window_dims=(),
                    inserted_window_dims=(0,),
                    scatter_dims_to_operand_dims=(0,),
                ),
            ),
            "input_shapes": [(5,), (1, 1, 1, 1), (1,)],
            "input_dtypes": [jnp.float32, jnp.int32, jnp.float32],
            "post_check_onnx_graph": EG(
                ["ScatterND"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "scatter_user_error_scenario_precise",
            "callable": lambda operand, indices, updates: jax.lax.scatter(
                operand,
                indices,
                updates,
                jax.lax.ScatterDimensionNumbers(
                    update_window_dims=(1, 2, 3),
                    inserted_window_dims=(0,),
                    scatter_dims_to_operand_dims=(0,),
                ),
                mode=jax.lax.GatherScatterMode.FILL_OR_DROP,
                unique_indices=False,
                indices_are_sorted=False,
            ),
            "input_shapes": [(5, 201, 1, 1), (2, 1), (2, 201, 1, 1)],
            "input_dtypes": [jnp.float32, jnp.int32, jnp.float32],
            "post_check_onnx_graph": EG(
                ["ScatterND"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "scatter_window_update_f64",
            "callable": lambda operand, indices, updates: jax.lax.scatter(
                operand=operand,
                scatter_indices=indices,
                updates=updates,
                dimension_numbers=jax.lax.ScatterDimensionNumbers(
                    update_window_dims=(1, 2, 3, 4),
                    inserted_window_dims=(),
                    scatter_dims_to_operand_dims=(1, 2),
                ),
                indices_are_sorted=True,
                unique_indices=True,
                mode=jax.lax.GatherScatterMode.FILL_OR_DROP,
            ),
            "input_values": [
                np.zeros((5, 266, 266, 1), dtype=np.float64),
                np.array([[10, 10]], dtype=np.int32),
                np.ones((1, 5, 256, 256, 1), dtype=np.float64),
            ],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(["ScatterND"], no_unused_inputs=True),
        },
        {
            "testcase": "scatter_window_update_depth3_shapes_ok",
            "callable": lambda operand, indices, updates: jax.lax.scatter(
                operand=operand,
                scatter_indices=indices,
                updates=updates,
                dimension_numbers=jax.lax.ScatterDimensionNumbers(
                    update_window_dims=(1, 2, 3, 4),
                    inserted_window_dims=(),
                    scatter_dims_to_operand_dims=(1, 2),
                ),
                indices_are_sorted=True,
                unique_indices=True,
                mode=jax.lax.GatherScatterMode.FILL_OR_DROP,
            ),
            "input_values": [
                np.zeros((5, 266, 266, 1), dtype=np.float64),
                np.array([[10, 10]], dtype=np.int32),
                np.ones((1, 5, 256, 256, 1), dtype=np.float64),
            ],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(["ScatterND"], no_unused_inputs=True),
        },
        {
            "testcase": "scatter_static_slice_set_f64",
            "callable": lambda operand, indices, updates: operand.at[
                :, 5:261, 5:261, :
            ].set(updates),
            "input_values": [
                np.zeros((5, 266, 266, 1), dtype=np.float64),
                np.array([[5, 5]], dtype=np.int32),
                np.ones((5, 256, 256, 1), dtype=np.float64),
            ],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(["ScatterND"]),
        },
        {
            "testcase": "scatter_depth2_fp64_type_mismatch",
            "callable": lambda: jax.lax.scatter(
                jnp.zeros((2, 3, 4, 5), dtype=jnp.float64),
                jnp.array([[1]], dtype=jnp.int32),
                jnp.ones((1, 2, 3, 4, 5), dtype=jnp.float64),
                dimension_numbers=jax.lax.ScatterDimensionNumbers(
                    update_window_dims=(1, 2, 3, 4),
                    inserted_window_dims=(),
                    scatter_dims_to_operand_dims=(1,),
                ),
            ),
            "input_shapes": [],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(["ScatterND"], no_unused_inputs=True),
        },
        {
            "testcase": "scatter_clip_2d_window_at_edge",
            "callable": lambda: jax.lax.scatter(
                jnp.array(np.arange(5, dtype=np.float32).reshape(1, 5)),
                jnp.array([[4]], dtype=jnp.int32),
                jnp.array([[[9.0, 8.0]]], dtype=jnp.float32),
                dimension_numbers=jax.lax.ScatterDimensionNumbers(
                    update_window_dims=(1, 2),
                    inserted_window_dims=(),
                    scatter_dims_to_operand_dims=(1,),
                ),
                mode=jax.lax.GatherScatterMode.CLIP,
            ),
            "input_shapes": [],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(["ScatterND"], no_unused_inputs=True),
        },
        {
            "testcase": "scatter_simple_2d_window_out_of_bounds",
            "callable": lambda: jax.lax.scatter(
                jnp.zeros((5, 5), dtype=jnp.float32),
                jnp.array([[4]], dtype=jnp.int32),
                jnp.ones((1, 5, 2), dtype=jnp.float32),
                dimension_numbers=jax.lax.ScatterDimensionNumbers(
                    update_window_dims=(1, 2),
                    inserted_window_dims=(),
                    scatter_dims_to_operand_dims=(1,),
                ),
            ),
            "input_shapes": [],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(["ScatterND"], no_unused_inputs=True),
        },
        {
            "testcase": "scatter_depth2_mixed_dtypes_fp_mismatch_f64",
            "callable": lambda: jax.lax.scatter(
                jnp.zeros((2, 3, 4, 5), dtype=jnp.float64),
                jnp.array([[0, 1], [1, 2]], dtype=jnp.int32),
                jnp.ones((2, 4, 5), dtype=jnp.float64),
                dimension_numbers=jax.lax.ScatterDimensionNumbers(
                    update_window_dims=(1, 2),
                    inserted_window_dims=(0, 1),
                    scatter_dims_to_operand_dims=(0, 1),
                ),
            ),
            "input_shapes": [],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(["ScatterND"], no_unused_inputs=True),
        },
        {
            "testcase": "scatter_depth2_mixed_dtypes_fp_mismatch",
            "callable": lambda: jax.lax.scatter(
                jnp.zeros((2, 3, 4, 5), dtype=jnp.float64),
                jnp.array([[0, 1], [1, 2]], dtype=jnp.int32),
                jnp.ones((2, 4, 5), dtype=jnp.float32),
                dimension_numbers=jax.lax.ScatterDimensionNumbers(
                    update_window_dims=(1, 2),
                    inserted_window_dims=(0, 1),
                    scatter_dims_to_operand_dims=(0, 1),
                ),
            ),
            "input_shapes": [],
            "run_only_f32_variant": True,
            # not an option: "skip_numeric_validation": True,
            "post_check_onnx_graph": EG(["ScatterND"], no_unused_inputs=True),
        },
    ],
)
class ScatterPlugin(PrimitiveLeafPlugin):
    """IR-first lowering for ``lax.scatter`` (element-wise variant)."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        lower_scatter_common(ctx, eqn, reduction="none")
