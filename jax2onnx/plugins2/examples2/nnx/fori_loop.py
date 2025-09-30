from __future__ import annotations

import jax

from jax2onnx.plugins2.plugin_system import register_example


def _model(x: jax.Array) -> jax.Array:
    steps = 5

    def body(index: int, state: tuple[jax.Array, int]) -> tuple[jax.Array, int]:
        value, counter = state
        value = value + 0.1 * value**2
        counter = counter + 1
        return value, counter

    result, _ = jax.lax.fori_loop(0, steps, body, (x, 0))
    return result


register_example(
    component="ForiLoop",
    description="fori_loop example using nnx-compatible primitives (converter2).",
    since="v0.5.1",
    context="examples2.nnx",
    children=["jax.lax.fori_loop"],
    testcases=[
        {
            "testcase": "fori_loop_counter",
            "callable": _model,
            "input_shapes": [(1,)],
            "expected_output_shapes": [(1,)],
        },
    ],
)
