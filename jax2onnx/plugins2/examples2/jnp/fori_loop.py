from __future__ import annotations

import jax
import jax.numpy as jnp

from jax2onnx.plugins2.plugin_system import register_example


def _model(x: jax.Array) -> tuple[jax.Array, int]:
    steps = 5

    def body(index: int, args: tuple[jax.Array, int]) -> tuple[jax.Array, int]:
        value, counter = args
        value = value + 0.1 * value**2
        counter = counter + 1
        return value, counter

    return jax.lax.fori_loop(0, steps, body, (x, 0))


register_example(
    component="fori_loop_test",
    description="fori_loop_test: demonstrates jax.lax.fori_loop with a simple loop.",
    since="v0.9.0",
    context="examples2.jnp",
    children=[],
    testcases=[
        {
            "testcase": "fori_loop_test",
            "callable": _model,
            "input_shapes": [(2,)],
            "input_dtypes": [jnp.float32],
            "expected_output_shapes": [(2,), ()],
            "run_only_f32_variant": True,
            "use_onnx_ir": True,
        },
        {
            "testcase": "fori_loop_test_f64",
            "callable": _model,
            "input_shapes": [(3,)],
            "input_dtypes": [jnp.float64],
            "expected_output_shapes": [(3,), ()],
            "run_only_f64_variant": True,
            "use_onnx_ir": True,
        },
    ],
)
