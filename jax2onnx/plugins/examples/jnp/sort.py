# jax2onnx/plugins/examples/jnp/sort.py

from __future__ import annotations

import jax.numpy as jnp

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import register_example


def _sort_model(x: jnp.ndarray) -> jnp.ndarray:
    upper = jnp.sort(x[5:])
    lower = jnp.sort(x[:5])
    return upper * lower


register_example(
    component="sort_test",
    description="sort_test: demonstrates jnp.sort on slices of an input array.",
    since="v0.9.0",
    context="examples.jnp",
    children=[],
    testcases=[
        {
            "testcase": "sort_test_basic",
            "callable": _sort_model,
            "input_shapes": [(10,)],
            "input_dtypes": [jnp.float32],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Slice:5 -> TopK:5 -> Identity:5 -> Mul:5"],
                no_unused_inputs=True,
            ),
        },
    ],
)
