# file: tests/examples/mnist_cnn.py

from flax import nnx
import jax
import jax.numpy as jnp
import onnx.helper as oh
from functools import partial
from jax2onnx.to_onnx import Z


class MLP(nnx.Module):
    def __init__(self, in_features, out_features, *, rngs=nnx.Rngs(0)):
        self.layer = nnx.Linear(in_features, out_features, rngs=rngs)
        self.activation = jax.nn.relu

    def __call__(self, x):
        x = self.layer(x)
        x = self.activation(x)
        return x

    def to_onnx(self, z, parameters=None):
        z = self.layer.to_onnx(z)
        z = self.activation.to_onnx(z)
        return z


def get_test_params():
    """
    Test parameters for the MLP model.

    Returns:
        list: A list of dictionaries, each defining a test case.
    """
    return [
        {
            "model_name": "mlp",
            "model":  MLP(30, 10, rngs=nnx.Rngs(0)),
            "input_shapes": [(1, 30)]
        }
    ]
