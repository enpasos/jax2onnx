from __future__ import annotations

import numpy as np
import jax
from jax.lax import ScatterDimensionNumbers, GatherScatterMode

from jax2onnx.plugin_system import register_example


def scatter_window_function(operand, indices, updates):
    """
    Depth-3 window-scatter (H×W patch) with implicit batch.
    Regression for an earlier ONNX conversion failure when using
    JAX double precision and FILL_OR_DROP mode.
    """
    dnums = ScatterDimensionNumbers(
        update_window_dims=(1, 2, 3, 4),
        inserted_window_dims=(),
        scatter_dims_to_operand_dims=(1, 2),
    )
    return jax.lax.scatter(
        operand,
        indices,
        updates,
        dimension_numbers=dnums,
        indices_are_sorted=True,
        unique_indices=True,
        mode=GatherScatterMode.FILL_OR_DROP,
    )


register_example(
    component="scatter_window",
    description=(
        "Window-scatter (H×W patch) with implicit batch (depth-3 path). "
        "Exercises GatherScatterMode.FILL_OR_DROP and double precision. "
        "Regression of a prior conversion failure."
    ),
    since="v0.7.4",
    context="examples.lax",
    children=[],
    testcases=[
        {
            "testcase": "scatter_window_update_f64_example",
            "callable": lambda operand, indices, updates: scatter_window_function(
                operand, indices, updates
            ),
            # Use concrete inputs to exactly match the repro:
            "input_values": [
                np.zeros((5, 266, 266, 1), dtype=np.float64),  # operand
                np.array([[10, 10]], dtype=np.int32),  # indices (1,2)
                np.ones((1, 5, 256, 256, 1), dtype=np.float64),  # updates
            ],
            # Output is the updated operand; shape/dtype remain the same
            "expected_output_shapes": [(5, 266, 266, 1)],
            "expected_output_dtypes": [np.float64],
            # Run only the f64 variant to align with the original reproducer
            "run_only_f64_variant": True,
        },
    ],
)
