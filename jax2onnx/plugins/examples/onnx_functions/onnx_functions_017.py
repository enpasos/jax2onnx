# jax2onnx/plugins/examples/onnx_functions/onnx_functions_017.py

from __future__ import annotations

import jax.numpy as jnp
from flax import nnx

from jax2onnx.plugins._post_check_onnx_graph import expect_graph
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    onnx_function,
    register_example,
)


@onnx_function(unique=True)
def square_shift(x: jnp.ndarray, *, shift: float) -> jnp.ndarray:
    """Shift the input and square the result."""
    return jnp.square(x + shift)


class DoubleSquareShift(nnx.Module):
    """Apply the unique function twice with the same parameters."""

    def __init__(self, shift: float):
        self.shift = shift

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        first = square_shift(x, shift=self.shift)
        return square_shift(first, shift=self.shift)


register_example(
    component="onnx_functions_017",
    description="Demonstrates @onnx_function(unique=True) reuse across call sites.",
    since="v0.10.0",
    context="examples.onnx_functions",
    children=["square_shift"],
    testcases=[
        {
            "testcase": "017_unique_function_reuse",
            "callable": construct_and_call(DoubleSquareShift, shift=0.25),
            "input_shapes": [("N", 4)],
            "expected_number_of_function_instances": 1,
            "run_only_f32_variant": True,
            "post_check_onnx_graph": expect_graph(
                ["square_shift_1:Nx4", "square_shift_2:Nx4"],
                symbols={"N": None},
                no_unused_inputs=True,
            ),
        },
    ],
)
