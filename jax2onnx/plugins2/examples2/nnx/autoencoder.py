from __future__ import annotations

import jax
from flax import nnx

from jax2onnx.plugins2.plugin_system import register_example


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


_model = AutoEncoder(rngs=nnx.Rngs(0))


register_example(
    component="AutoEncoder",
    description="A simple autoencoder example (converter2 pipeline).",
    source="https://github.com/google/flax/blob/main/README.md",
    since="v0.2.0",
    context="examples2.nnx",
    children=["Encoder", "Decoder"],
    testcases=[
        {
            "testcase": "simple_autoencoder",
            "callable": _model,
            "input_shapes": [(1, 2)],
            "expected_output_shapes": [(1, 2)],
            "use_onnx_ir": True,
        }
    ],
)
