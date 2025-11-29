# jax2onnx/plugins/jax/lax/scatter_max.py

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax.scatter_utils import lower_scatter_common
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


@register_primitive(
    jaxpr_primitive=jax.lax.scatter_max_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.scatter_max.html",
    onnx=[
        {
            "component": "ScatterND(reduction='max')",
            "doc": "https://onnx.ai/onnx/operators/onnx__ScatterND.html",
        }
    ],
    since="v0.7.5",
    context="primitives.lax",
    component="scatter_max",
    testcases=[
        {
            "testcase": "scatter_max_simple_1d",
            "callable": lambda operand, indices, updates: jax.lax.scatter_max(
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
            "testcase": "scatter_max_batch_updates_1d_operand",
            "callable": lambda operand, indices, updates: jax.lax.scatter_max(
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
            "testcase": "scatter_max_window_2d_operand_1d_indices",
            "callable": lambda operand, indices, updates: jax.lax.scatter_max(
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
            "testcase": "scatter_max_fp64_dtype_path_check",
            "callable": lambda: jax.lax.scatter_max(
                jnp.zeros((4, 3), dtype=jnp.float64),
                jnp.array([[0, 0], [2, 1]], dtype=jnp.int32),
                jnp.array([9.0, 8.0], dtype=jnp.float64),
                jax.lax.ScatterDimensionNumbers(
                    update_window_dims=(),
                    inserted_window_dims=(0, 1),
                    scatter_dims_to_operand_dims=(0, 1),
                ),
            ),
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                ["Cast:2x2 -> Reshape:?x2 -> ScatterND:4x3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "scatter_max_depth2_helper_regression_fp64",
            "callable": lambda: jax.lax.scatter_max(
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
    ],
)
class ScatterMaxPlugin(PrimitiveLeafPlugin):
    """IR-first lowering for ``lax.scatter_max`` (element-wise variant)."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        lower_scatter_common(ctx, eqn, reduction="max")
