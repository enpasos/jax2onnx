# jax2onnx/plugins/jax/lax/scatter_mul.py

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

# Ensure cond plugin is registered when scatter_mul metadata uses it.
from jax2onnx.plugins.jax.lax import cond as _cond_plugin  # noqa: F401

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax.scatter_utils import lower_scatter_common
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


@register_primitive(
    jaxpr_primitive=jax.lax.scatter_mul_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.scatter_mul.html",
    onnx=[
        {
            "component": "ScatterND(reduction='mul')",
            "doc": "https://onnx.ai/onnx/operators/onnx__ScatterND.html",
        }
    ],
    since="v0.6.4",
    context="primitives.lax",
    component="scatter_mul",
    testcases=[
        {
            "testcase": "scatter_mul_simple_1d",
            "callable": lambda operand, indices, updates: jax.lax.scatter_mul(
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
            "testcase": "scatter_mul_batch_updates_1d_operand",
            "callable": lambda operand, indices, updates: jax.lax.scatter_mul(
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
            "testcase": "scatter_mul_window_2d_operand_1d_indices",
            "callable": lambda operand, indices, updates: jax.lax.scatter_mul(
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
            "testcase": "scatter_mul_mismatched_window_dims_from_user_report",
            "callable": lambda operand, indices, updates: jax.lax.scatter_mul(
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
                jnp.ones((5, 208, 1, 1), dtype=jnp.float64),
                jnp.array([4], dtype=jnp.int32),
                jnp.full((5, 200, 1, 1), 2.0, dtype=jnp.float64),
            ],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                ["ScatterND:5x208x1x1"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "scatter_mul_mismatched_window_dims_from_user_report2",
            "callable": lambda operand, indices, updates: jax.lax.scatter_mul(
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
                jnp.ones((3, 150, 1, 1), dtype=jnp.float64),
                jnp.array([7], dtype=jnp.int32),
                jnp.full((3, 140, 1, 1), 2.0, dtype=jnp.float64),
            ],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                ["ScatterND:3x150x1x1"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "scatter_mul_mismatched_window_dims_from_user_report3",
            "callable": lambda operand, indices, updates: jax.lax.scatter_mul(
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
                jnp.ones((8, 50, 1, 1), dtype=jnp.float64),
                jnp.array([2], dtype=jnp.int32),
                jnp.full((8, 45, 1, 1), 2.0, dtype=jnp.float64),
            ],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                ["ScatterND:8x50x1x1"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "scatter_mul_fluids_pattern_updates_5_4_1_1",
            "callable": lambda operand, indices, updates: jax.lax.scatter_mul(
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
                jnp.ones((5, 208, 1, 1), dtype=jnp.float64),
                jnp.array([0], dtype=jnp.int32),
                jnp.full((5, 4, 1, 1), 2.0, dtype=jnp.float64),
            ],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                ["ScatterND:5x208x1x1"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "scatter_mul_in_cond_float64",
            "callable": lambda pred, operand, indices, updates: jax.lax.cond(
                pred,
                lambda op, idx, upd: jax.lax.scatter_mul(
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
                jnp.ones((8, 50, 1, 1), dtype=jnp.float64),
                jnp.array([2], dtype=jnp.int32),
                jnp.full((8, 45, 1, 1), 2.0, dtype=jnp.float64),
            ],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                ["If:8x50x1x1"],
            ),
        },
    ],
)
class ScatterMulPlugin(PrimitiveLeafPlugin):
    """IR-first lowering for ``lax.scatter_mul`` (element-wise variant)."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        lower_scatter_common(ctx, eqn, reduction="mul")
