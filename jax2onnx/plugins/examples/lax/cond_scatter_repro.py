# jax2onnx/plugins/examples/lax/cond_scatter_repro.py

from __future__ import annotations

import numpy as np

import jax.numpy as jnp
from jax import lax

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import register_example


def model_with_cond_and_scatter():
    """Reproducer where lax.cond branches capture local scatter operands."""
    base_vals = np.arange(0.0, 2 * 4 * 1 * 1, 1.0, dtype=np.float64).reshape(2, 4, 1, 1)
    original_operand_val = jnp.asarray(base_vals, dtype=jnp.float64)

    raw_updates_data_val = jnp.ones((1, 4, 1, 1), dtype=jnp.float64) * 100.0
    reshaped_updates_for_slices_val = jnp.reshape(raw_updates_data_val, (1, 4, 1, 1))

    indices_for_axis_0_val = jnp.array([1])

    predicate = jnp.array(True)

    branch_operands = (
        original_operand_val,
        indices_for_axis_0_val,
        reshaped_updates_for_slices_val,
    )

    def true_branch_takes_tuple(operands_tuple):
        op, idx, upd = operands_tuple
        return op.at[idx].set(upd)

    def false_branch_takes_tuple(operands_tuple):
        op, _, _ = operands_tuple
        return op + 1.0

    scattered_result = lax.cond(
        predicate, true_branch_takes_tuple, false_branch_takes_tuple, branch_operands
    )

    some_int_value = jnp.array(42, dtype=jnp.int64)
    reshaped_int_value = jnp.reshape(some_int_value, ())

    return scattered_result, reshaped_int_value


register_example(
    component="cond_scatter_repro",
    description="Reproduces a bug where lax.cond subgraphs do not inherit parent initializers.",
    since="v0.6.4",
    context="examples.lax",
    children=[],
    testcases=[
        {
            "testcase": "cond_scatter_repro_f64",
            "callable": model_with_cond_and_scatter,
            "input_shapes": [],
            "input_dtypes": [],
            "expected_output_shapes": [(2, 4, 1, 1), ()],
            "expected_output_dtypes": [jnp.float64, jnp.int64],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                [
                    "Cast -> If:2x4x1x1",
                ],
                no_unused_inputs=True,
            ),
        },
    ],
)
