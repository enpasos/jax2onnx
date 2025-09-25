from __future__ import annotations

import jax.numpy as jnp

from jax2onnx.plugins2.plugin_system import register_example


def _sort_model(x: jnp.ndarray) -> jnp.ndarray:
    upper = jnp.sort(x[5:])
    lower = jnp.sort(x[:5])
    return upper * lower


register_example(
    component="sort_test",
    description="sort_test: demonstrates jnp.sort on slices of an input array.",
    since="v0.9.0",
    context="examples2.jnp",
    children=[],
    testcases=[
        {
            "testcase": "sort_test_basic",
            "callable": _sort_model,
            "input_shapes": [(10,)],
            "input_dtypes": [jnp.float32],
            "run_only_f32_variant": True,
            "use_onnx_ir": True,
        },
    ],
)
