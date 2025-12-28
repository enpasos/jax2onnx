# jax2onnx/plugins/examples/linen/dropout.py

from __future__ import annotations

from flax import linen as nn

from jax2onnx.plugins.flax.test_utils import linen_to_nnx
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    register_example,
    with_requested_dtype,
    with_rng_seed,
)


class SimpleDropout(nn.Module):
    rate: float = 0.1

    @nn.compact
    def __call__(self, x):
        return nn.Dropout(rate=self.rate, deterministic=True)(x)


register_example(
    component="LinenDropout",
    description="A simple Flax Linen Dropout layer (deterministic).",
    source="",
    since="v0.11.0",
    context="examples.linen",
    children=["flax.linen.Dropout"],
    testcases=[
        {
            "testcase": "simple_linen_dropout",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=SimpleDropout,
                input_shape=(1, 10),
                dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(1, 10)],
        },
    ],
)
