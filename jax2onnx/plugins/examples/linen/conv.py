# jax2onnx/plugins/examples/linen/conv.py

from __future__ import annotations

from flax import linen as nn

from jax2onnx.plugins.flax.test_utils import linen_to_nnx
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    register_example,
    with_requested_dtype,
    with_rng_seed,
)


class SimpleConv(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        return nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            kernel_init=nn.initializers.ones,
            bias_init=nn.initializers.zeros,
        )(x)


register_example(
    component="LinenConv",
    description="A simple Flax Linen Conv layer.",
    source="",
    since="v0.11.0",
    context="examples.linen",
    children=["flax.linen.Conv"],
    testcases=[
        {
            "testcase": "simple_linen_conv",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=SimpleConv,
                input_shape=(1, 28, 28, 3),
                dtype=with_requested_dtype(),
                features=8,
                rngs=with_rng_seed(0),
            ),
            "run_only_f32_variant": True,
            "input_shapes": [(1, 28, 28, 3)],
        },
    ],
)
