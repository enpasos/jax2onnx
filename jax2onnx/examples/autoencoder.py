# file: jax2onnx/examples/autoencoder.py
from flax import nnx
import jax


# from https://github.com/google/flax/blob/main/README.md
def Encoder(rngs):
    return nnx.Linear(2, 10, rngs=rngs)


def Decoder(rngs):
    return nnx.Linear(10, 2, rngs=rngs)


class AutoEncoder(nnx.Module):
    def __init__(self, rngs):
        self.encoder = Encoder(rngs)
        self.decoder = Decoder(rngs)

    def __call__(self, x) -> jax.Array:
        return self.decoder(self.encoder(x))

    def encode(self, x) -> jax.Array:
        return self.encoder(x)


def get_metadata() -> dict:
    """Return test parameters for verifying the ONNX conversion of the AutoEncoder model."""
    return {
        "component": "AutoEncoder",
        "description": "A simple autoencoder example.",
        "since": "v0.2.0",
        "context": "examples.nnx",
        "children": ["Encoder", "Decoder"],
        "testcases": [
            {
                "testcase": "autoencoder",
                "callable": AutoEncoder(rngs=nnx.Rngs(0)),
                "input_shapes": [(1, 2)],
            }
        ],
    }
