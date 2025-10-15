# jax2onnx/plugins/examples/nnx/fori_loop.py

from __future__ import annotations

import jax

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import register_example


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
    description="fori_loop example using nnx-compatible primitives (converter).",
    since="v0.5.1",
    context="examples.nnx",
    children=["jax.lax.fori_loop"],
    testcases=[
        {
            "testcase": "fori_loop_counter",
            "callable": _model,
            "input_shapes": [(1,)],
            "expected_output_shapes": [(1,)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {
                            0: {"const": 5.0},
                            1: {"const_bool": True},
                            3: {"const": 0.0},
                        },
                        "path": "Loop:1",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
    ],
)
