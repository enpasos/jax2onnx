# jax2onnx/plugins/examples/nnx/autoencoder.py

from __future__ import annotations

import jax
from flax import nnx

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    register_example,
    with_rng_seed,
)


def Encoder(rngs: nnx.Rngs) -> nnx.Linear:
    return nnx.Linear(2, 10, rngs=rngs)


def Decoder(rngs: nnx.Rngs) -> nnx.Linear:
    return nnx.Linear(10, 2, rngs=rngs)


class AutoEncoder(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.encoder = Encoder(rngs)
        self.decoder = Decoder(rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.decoder(self.encoder(x))

    def encode(self, x: jax.Array) -> jax.Array:
        return self.encoder(x)


register_example(
    component="AutoEncoder",
    description="A simple autoencoder example (converter pipeline).",
    source="https://github.com/google/flax/blob/main/README.md",
    since="v0.2.0",
    context="examples.nnx",
    children=["Encoder", "Decoder"],
    testcases=[
        {
            "testcase": "simple_autoencoder",
            "callable": construct_and_call(AutoEncoder, rngs=with_rng_seed(0)),
            "input_shapes": [(1, 2)],
            "expected_output_shapes": [(1, 2)],
            "post_check_onnx_graph": EG(
                ["Gemm:1x10 -> Gemm:1x2"],
                no_unused_inputs=True,
            ),
        }
    ],
)
