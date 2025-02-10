# file: tests/examples/mlp.py


from flax import nnx
import jax
import jax.numpy as jnp
from jax2onnx.typing_helpers import PartialWithOnnx, Supports2Onnx


class MLP(nnx.Module):
    """A Multi-Layer Perceptron (MLP) with BatchNorm, Dropout, and GELU activation."""

    def __init__(self, din: int, dmid: int, dout: int, *, rngs=nnx.Rngs(0)):
        """Initializes the MLP model with linear layers, batch normalization, dropout, and activation."""
        self.layers: list[Supports2Onnx] = [
            nnx.Linear(din, dmid, rngs=rngs),
            nnx.BatchNorm(dmid, rngs=rngs),
            nnx.Dropout(rate=0.1, rngs=rngs),
            PartialWithOnnx(jax.nn.gelu, approximate=False),
            nnx.Linear(dmid, dout, rngs=rngs),
        ]

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """Defines the forward pass of the MLP."""
        for layer in self.layers:
            if isinstance(layer, nnx.Dropout):
                x = layer(x, deterministic=deterministic)
            else:
                x = layer(x)
        return x

    def to_onnx(self, z, **params):
        """Defines the ONNX export logic for the MLP model."""
        for layer in self.layers:
            z = layer.to_onnx(z, **params)
        return z


def get_test_params():
    """
    Defines test parameters for verifying the ONNX conversion of the MLP model.

    Returns:
        list: A list of test cases for the MLP model.
    """
    return [
        {
            "testcase": "mlp",
            "model": MLP(din=30, dmid=20, dout=10, rngs=nnx.Rngs(17)),
            "input_shapes": [(1, 30)],
        }
    ]
