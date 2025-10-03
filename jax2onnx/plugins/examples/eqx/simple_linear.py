from __future__ import annotations

import equinox as eqx
import jax

from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    register_example,
    with_prng_key,
)


class Linear(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, in_size: int, out_size: int, key: jax.Array):
        wkey, bkey = jax.random.split(key)
        self.weight = jax.random.normal(wkey, (out_size, in_size))
        self.bias = jax.random.normal(bkey, (out_size,))

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.weight @ x + self.bias


register_example(
    component="SimpleLinearExample",
    description="A simple linear layer example using Equinox (converter).",
    source="https://github.com/patrick-kidger/equinox",
    since="v0.8.0",
    context="examples.eqx",
    children=["eqx.nn.Linear"],
    testcases=[
        {
            "testcase": "simple_linear",
            "callable": construct_and_call(
                lambda key: jax.vmap(Linear(30, 3, key=key)),
                key=with_prng_key(0),
            ),
            "input_shapes": [(12, 30)],
        },
        {
            "testcase": "nn_linear",
            "callable": construct_and_call(
                lambda key: jax.vmap(
                    eqx.nn.Linear(in_features=30, out_features=3, key=key)
                ),
                key=with_prng_key(1),
            ),
            "input_shapes": [(12, 30)],
        },
    ],
)
