from __future__ import annotations

import equinox as eqx
import jax

from jax2onnx.plugins2.plugin_system import register_example


class Linear(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, in_size: int, out_size: int, key: jax.Array):
        wkey, bkey = jax.random.split(key)
        self.weight = jax.random.normal(wkey, (out_size, in_size))
        self.bias = jax.random.normal(bkey, (out_size,))

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.weight @ x + self.bias


# Create the modules once so weight init is outside the traced callable.
_batched_linear = jax.vmap(Linear(30, 3, key=jax.random.PRNGKey(0)))
_batched_nn_linear = jax.vmap(
    eqx.nn.Linear(in_features=30, out_features=3, key=jax.random.PRNGKey(1))
)


register_example(
    component="SimpleLinearExample",
    description="A simple linear layer example using Equinox (converter2).",
    source="https://github.com/patrick-kidger/equinox",
    since="v0.8.0",
    context="examples2.eqx",
    children=["eqx.nn.Linear"],
    testcases=[
        {
            "testcase": "simple_linear",
            "callable": _batched_linear,
            "input_shapes": [(12, 30)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "nn_linear",
            "callable": _batched_nn_linear,
            "input_shapes": [(12, 30)],
            "use_onnx_ir": True,
        },
    ],
)
