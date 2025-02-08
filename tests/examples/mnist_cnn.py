# file: tests/examples/mnist_cnn.py
from functools import partial
import jax
import jax.numpy as jnp
from flax import nnx


class MNIST_CNN(nnx.Module):
    """A simple CNN model."""

    def __init__(self, *, rngs: nnx.Rngs):
        """Initializes the CNN model with convolutional and linear layers."""
        self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
        self.linear1 = nnx.Linear(3136, 256, rngs=rngs)
        self.linear2 = nnx.Linear(256, 10, rngs=rngs)
        self.act = jax.nn.relu
        self.reshape = lambda x: x.reshape(x.shape[0], -1)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Defines the forward pass of the model."""
        x = self.avg_pool(self.act(self.conv1(x)))
        x = self.avg_pool(self.act(self.conv2(x)))
        x = self.reshape(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x

    def to_onnx(self, z, parameters=None):
        """Defines the ONNX export logic for the CNN model."""
        z = self.conv1.to_onnx(z)
        z = self.act.to_onnx(z)
        z = nnx.avg_pool.to_onnx(z, {"window_shape": (2, 2), "strides": (2, 2)})

        z = self.conv2.to_onnx(z)
        z = self.act.to_onnx(z)
        z = nnx.avg_pool.to_onnx(z, {"window_shape": (2, 2), "strides": (2, 2)})

        reshape_params = {
            "shape": (z.shapes[0][0], 3136),
            "pre_transpose": [(0, 2, 3, 1)],
        }
        z = jax.numpy.reshape.to_onnx(z, reshape_params)

        z = self.linear1.to_onnx(z)
        z = self.act.to_onnx(z)
        z = self.linear2.to_onnx(z)

        return z


def get_test_params():
    """Define test parameters for the MNIST CNN."""
    return [
        {
            "model_name": "mnist_cnn",
            "model": MNIST_CNN(rngs=nnx.Rngs(0)),
            "input_shapes": [(1, 28, 28, 1)],  # Updated for (N, H, W, C) as used in JAX
            "export": {"pre_transpose": [(0, 3, 1, 2)]},
        }
    ]
