# jax2onnx/plugins/jax/lax/scatter_add.py

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

# Ensure cond plugin metadata dependencies are registered.
from jax2onnx.plugins.jax.lax import cond as _cond_plugin  # noqa: F401

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax.scatter_utils import lower_scatter_common
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


@register_primitive(
    jaxpr_primitive=jax.lax.scatter_add_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.scatter_add.html",
    onnx=[
        {
            "component": "ScatterND(reduction='add')",
            "doc": "https://onnx.ai/onnx/operators/onnx__ScatterND.html",
        }
    ],
    since="v0.5.3",
    context="primitives.lax",
    component="scatter_add",
    testcases=[
        {
            "testcase": "scatter_add_vector",
            "callable": lambda x: x.at[jnp.array([0, 2], dtype=jnp.int32)].add(
                jnp.array([1.5, -2.0], dtype=x.dtype)
            ),
            "input_shapes": [(4,)],
            "post_check_onnx_graph": EG(
                ["ScatterND:4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "scatter_add_scalar",
            "callable": lambda x: x.at[3].add(jnp.array(5.0, dtype=x.dtype)),
            "input_shapes": [(6,)],
            "post_check_onnx_graph": EG(
                [{"path": "ScatterND:6", "inputs": {2: {"const": 5.0}}}],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "scatter_add_simple_1d",
            "callable": lambda operand, indices, updates: jax.lax.scatter_add(
                operand,
                indices,
                updates,
                jax.lax.ScatterDimensionNumbers(
                    update_window_dims=(),
                    inserted_window_dims=(0,),
                    scatter_dims_to_operand_dims=(0,),
                ),
            ),
            "input_shapes": [(5,), (2, 1), (2,)],
            "input_dtypes": [jnp.float32, jnp.int32, jnp.float32],
            "post_check_onnx_graph": EG(
                ["ScatterND:5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "scatter_add_batch_updates_1d_operand",
            "callable": lambda operand, indices, updates: jax.lax.scatter_add(
                operand,
                indices,
                updates,
                jax.lax.ScatterDimensionNumbers(
                    update_window_dims=(),
                    inserted_window_dims=(0,),
                    scatter_dims_to_operand_dims=(0,),
                ),
            ),
            "input_shapes": [(5,), (2, 2, 1), (2, 2)],
            "input_dtypes": [jnp.float32, jnp.int32, jnp.float32],
            "post_check_onnx_graph": EG(
                ["ScatterND:5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "scatter_add_window_2d_operand_1d_indices",
            "callable": lambda operand, indices, updates: jax.lax.scatter_add(
                operand,
                indices,
                updates,
                jax.lax.ScatterDimensionNumbers(
                    update_window_dims=(1,),
                    inserted_window_dims=(0,),
                    scatter_dims_to_operand_dims=(0,),
                ),
            ),
            "input_values": [
                jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float32),
                jnp.array([[0]], dtype=jnp.int32),
                jnp.array([[10.0, 20.0, 30.0]], dtype=jnp.float32),
            ],
            "post_check_onnx_graph": EG(
                ["ScatterND:2x3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "scatter_add_mismatched_window_dims_from_user_report",
            "callable": lambda operand, indices, updates: jax.lax.scatter_add(
                operand,
                indices,
                updates,
                jax.lax.ScatterDimensionNumbers(
                    update_window_dims=(0, 1, 2, 3),
                    inserted_window_dims=(),
                    scatter_dims_to_operand_dims=(1,),
                ),
            ),
            "input_values": [
                jnp.zeros((5, 208, 1, 1), dtype=jnp.float64),
                jnp.array([4], dtype=jnp.int32),
                jnp.ones((5, 200, 1, 1), dtype=jnp.float64),
            ],
            "run_only_f64_variant": True,
            "expected_output_shapes": [(5, 208, 1, 1)],
            "expected_output_dtypes": [jnp.float64],
            "post_check_onnx_graph": EG(
                ["ScatterND:5x208x1x1"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "scatter_add_mismatched_window_dims_from_user_report2",
            "callable": lambda operand, indices, updates: jax.lax.scatter_add(
                operand,
                indices,
                updates,
                jax.lax.ScatterDimensionNumbers(
                    update_window_dims=(0, 1, 2, 3),
                    inserted_window_dims=(),
                    scatter_dims_to_operand_dims=(1,),
                ),
            ),
            "input_values": [
                jnp.zeros((3, 150, 1, 1), dtype=jnp.float64),
                jnp.array([7], dtype=jnp.int32),
                jnp.ones((3, 140, 1, 1), dtype=jnp.float64),
            ],
            "run_only_f64_variant": True,
            "expected_output_shapes": [(3, 150, 1, 1)],
            "expected_output_dtypes": [jnp.float64],
            "post_check_onnx_graph": EG(
                ["ScatterND:3x150x1x1"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "scatter_add_mismatched_window_dims_from_user_report3",
            "callable": lambda operand, indices, updates: jax.lax.scatter_add(
                operand,
                indices,
                updates,
                jax.lax.ScatterDimensionNumbers(
                    update_window_dims=(0, 1, 2, 3),
                    inserted_window_dims=(),
                    scatter_dims_to_operand_dims=(1,),
                ),
            ),
            "input_values": [
                jnp.zeros((8, 50, 1, 1), dtype=jnp.float64),
                jnp.array([2], dtype=jnp.int32),
                jnp.ones((8, 45, 1, 1), dtype=jnp.float64),
            ],
            "run_only_f64_variant": True,
            "expected_output_shapes": [(8, 50, 1, 1)],
            "expected_output_dtypes": [jnp.float64],
            "post_check_onnx_graph": EG(
                ["ScatterND:8x50x1x1"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "scatter_add_fluids_pattern_updates_5_4_1_1",
            "callable": lambda operand, indices, updates: jax.lax.scatter_add(
                operand,
                indices,
                updates,
                jax.lax.ScatterDimensionNumbers(
                    update_window_dims=(0, 1, 2, 3),
                    inserted_window_dims=(),
                    scatter_dims_to_operand_dims=(1,),
                ),
            ),
            "input_values": [
                jnp.zeros((5, 208, 1, 1), dtype=jnp.float64),
                jnp.array([0], dtype=jnp.int32),
                jnp.ones((5, 4, 1, 1), dtype=jnp.float64),
            ],
            "run_only_f64_variant": True,
            "expected_output_shapes": [(5, 208, 1, 1)],
            "expected_output_dtypes": [jnp.float64],
            "post_check_onnx_graph": EG(
                ["ScatterND:5x208x1x1"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "scatter_add_in_cond_float64",
            "callable": lambda pred, operand, indices, updates: jax.lax.cond(
                pred,
                lambda op, idx, upd: jax.lax.scatter_add(
                    op,
                    idx,
                    upd,
                    jax.lax.ScatterDimensionNumbers(
                        update_window_dims=(0, 1, 2, 3),
                        inserted_window_dims=(),
                        scatter_dims_to_operand_dims=(1,),
                    ),
                ),
                lambda op, idx, upd: op,
                operand,
                indices,
                updates,
            ),
            "input_values": [
                jnp.array(True),
                jnp.zeros((8, 50, 1, 1), dtype=jnp.float64),
                jnp.array([2], dtype=jnp.int32),
                jnp.ones((8, 45, 1, 1), dtype=jnp.float64),
            ],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                ["If:8x50x1x1"],
            ),
        },
        {
            "testcase": "scatter_add_fp64_dtype_mismatch",
            "callable": lambda: jax.lax.scatter_add(
                jnp.zeros((4, 3), dtype=jnp.float64),
                jnp.array([[0, 0], [2, 1]], dtype=jnp.int32),
                jnp.ones((2,), dtype=jnp.float64),
                jax.lax.ScatterDimensionNumbers(
                    update_window_dims=(),
                    inserted_window_dims=(0, 1),
                    scatter_dims_to_operand_dims=(0, 1),
                ),
            ),
            "run_only_f64_variant": True,
            "post_check_onnx_graph": lambda m: (
                __import__("onnx").checker.check_model(m) or True
            ),
        },
        {
            "testcase": "scatter_add_depth2_depth2_helper_regression",
            "callable": lambda: jax.lax.scatter_add(
                jnp.zeros((2, 3, 4, 5), dtype=jnp.float64),
                jnp.array([[0, 1], [1, 2]], dtype=jnp.int32),
                jnp.ones((2, 4, 5), dtype=jnp.float64),
                jax.lax.ScatterDimensionNumbers(
                    update_window_dims=(1, 2),
                    inserted_window_dims=(0, 1),
                    scatter_dims_to_operand_dims=(0, 1),
                ),
            ),
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                ["Cast:2x2 -> Reshape:?x2 -> ScatterND:2x3x4x5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "scatter_depth2_fp64_type_mismatch",
            "callable": lambda: jax.lax.scatter(
                jnp.zeros((2, 3, 4, 5), dtype=jnp.float64),
                jnp.array([[1]], dtype=jnp.int32),
                jnp.ones((1, 2, 3, 4, 5), dtype=jnp.float64),
                jax.lax.ScatterDimensionNumbers(
                    update_window_dims=(1, 2, 3, 4),
                    inserted_window_dims=(),
                    scatter_dims_to_operand_dims=(1,),
                ),
            ),
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                ["ScatterND:2x3x4x5"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class ScatterAddPlugin(PrimitiveLeafPlugin):
    """IR-first lowering for ``lax.scatter_add`` (element-wise variant)."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        lower_scatter_common(ctx, eqn, reduction="add")
