# file: jax2onnx/examples/autoencoder.py
import jax
from flax import nnx

from jax2onnx.plugin_system import register_example, construct_and_call


def Encoder(rngs):
    return nnx.Linear(2, 10, rngs=rngs)


def Decoder(rngs):
    return nnx.Linear(10, 2, rngs=rngs)


class AutoEncoder(nnx.Module):
    def __init__(self, rngs):
        self.encoder = Encoder(rngs)
        self.decoder = Decoder(rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.decoder(self.encoder(x))

    def encode(self, x: jax.Array) -> jax.Array:
        return self.encoder(x)


# ✅ build the model when the testcase actually runs
def _run_autoencoder(x):
    model = AutoEncoder(rngs=nnx.Rngs(0))
    return model(x)

register_example(
    component="AutoEncoder",
    description="A simple autoencoder example.",
    source="https://github.com/google/flax/blob/main/README.md",
    since="v0.2.0",
    context="examples.nnx",
    children=["Encoder", "Decoder"],
    testcases=[
        {
            "testcase": "simple_autoencoder",
            # Late-construct the module at call time (no RNG/params at import).
            "callable": construct_and_call(AutoEncoder, rngs=nnx.Rngs(0)),
            "input_shapes": [(1, 2)],
        }
    ],
)
