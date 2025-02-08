# file: tests/examples/mlp.py
from flax import nnx
import jax
import jax.numpy as jnp


class MLP(nnx.Module):
    """A Multi-Layer Perceptron (MLP) with BatchNorm, Dropout, and GELU activation."""

    def __init__(self, din: int, dmid: int, dout: int, *, rngs=nnx.Rngs(0)):
        """Initializes the MLP model with linear layers, batch normalization, dropout, and activation."""
        self.linear1 = nnx.Linear(din, dmid, rngs=rngs)
        self.batch_norm = nnx.BatchNorm(dmid, rngs=rngs)
        self.dropout = nnx.Dropout(rate=0.1, rngs=rngs)
        self.activation = jax.nn.gelu
        self.linear2 = nnx.Linear(dmid, dout, rngs=rngs)

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """Defines the forward pass of the MLP."""
        x = self.linear1(x)
        x = self.batch_norm(x)
        x = self.dropout(x, deterministic=deterministic)
        x = self.activation(x)
        return self.linear2(x)

    def to_onnx(self, z, parameters=None):
        """Defines the ONNX export logic for the MLP model."""
        for layer in [
            self.linear1,
            self.batch_norm,
            self.dropout,
            self.activation,
            self.linear2,
        ]:
            z = layer.to_onnx(z, parameters)
        return z


def get_test_params():
    """
    Defines test parameters for verifying the ONNX conversion of the MLP model.

    Returns:
        list: A list of test cases for the MLP model.
    """
    return [
        {
            "model_name": "mlp",
            "model": MLP(din=30, dmid=20, dout=10, rngs=nnx.Rngs(17)),
            "input_shapes": [(1, 30)],
        }
    ]
