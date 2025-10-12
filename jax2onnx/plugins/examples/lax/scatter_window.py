# jax2onnx/plugins/examples/lax/scatter_window.py

from __future__ import annotations

import numpy as np
import jax
from jax.lax import GatherScatterMode, ScatterDimensionNumbers

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import register_example


def scatter_window_function(operand, indices, updates):
    """Depth-3 window-scatter (H×W patch) regression for converter."""
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
            "callable": scatter_window_function,
            "input_values": [
                np.zeros((5, 266, 266, 1), dtype=np.float64),
                np.array([[10, 10]], dtype=np.int32),
                np.ones((1, 5, 256, 256, 1), dtype=np.float64),
            ],
            "expected_output_shapes": [(5, 266, 266, 1)],
            "expected_output_dtypes": [np.float64],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                ["ScatterND:5x266x266x1"],
                no_unused_inputs=True,
            ),
        },
    ],
)
