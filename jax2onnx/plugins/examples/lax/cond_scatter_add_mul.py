# jax2onnx/plugins/examples/lax/cond_scatter_add_mul.py

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import register_example


def cond_scatter_add_mul_f64(
    operand: jax.Array,
    scatter_indices: jax.Array,
    updates_for_add: jax.Array,
    updates_for_mul: jax.Array,
):
    """Scatter add vs mul inside a conditional; regression for scatter lowering."""
    dimension_numbers = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )

    branch_if_true = jax.lax.scatter_add(
        operand, scatter_indices, updates_for_add, dimension_numbers
    )
    branch_if_false = jax.lax.scatter_mul(
        operand, scatter_indices, updates_for_mul, dimension_numbers
    )

    condition = jnp.sum(operand) > 0.0
    final_output = jnp.where(condition, branch_if_true, branch_if_false)
    return (condition, final_output)


register_example(
    component="cond_scatter_add_mul",
    description="Scatter add/mul inside conditional branches (converter).",
    since="v0.8.0",
    context="examples.lax",
    children=["jax.lax.scatter_add", "jax.lax.scatter_mul", "jnp.where"],
    testcases=[
        {
            "testcase": "cond_scatter_add_mul_f64_a",
            "callable": cond_scatter_add_mul_f64,
            "input_values": [
                np.ones((6,), dtype=np.float64),
                np.array([[0], [4]], dtype=np.int64),
                np.full((2,), 2.0, dtype=np.float64),
                np.full((2,), 3.0, dtype=np.float64),
            ],
            "expected_output_shapes": [(), (6,)],
            "expected_output_dtypes": [jnp.bool_, jnp.float64],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 0.0}},
                        "path": "ReduceSum -> Greater",
                    },
                    "ReduceSum -> Greater -> Where:6",
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "cond_scatter_add_mul_f64_b",
            "callable": cond_scatter_add_mul_f64,
            "input_values": [
                np.arange(10, dtype=np.float64),
                np.array([[1], [7]], dtype=np.int64),
                np.full((2,), 5.0, dtype=np.float64),
                np.full((2,), 7.0, dtype=np.float64),
            ],
            "expected_output_shapes": [(), (10,)],
            "expected_output_dtypes": [jnp.bool_, jnp.float64],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 0.0}},
                        "path": "ReduceSum -> Greater",
                    },
                    "ReduceSum -> Greater -> Where:10",
                ],
                no_unused_inputs=True,
            ),
        },
    ],
)
