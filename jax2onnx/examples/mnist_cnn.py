# file: tests/examples/mnist_cnn.py

import jax
from flax import nnx
from jax2onnx.typing_helpers import PartialWithOnnx, Supports2Onnx
from typing import Any
import jax.numpy as jnp

import jax2onnx.plugins  # noqa: F401
from jax2onnx.convert import Z


class ReshapeWithOnnx(Supports2Onnx):

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Reshapes input using the provided shape function."""
        return x.reshape(x.shape[0], -1)

    def to_onnx(self, z: Z, **params: Any) -> Z:
        """ONNX conversion function for Reshape."""
        flatten_size = z.shapes[0][1] * z.shapes[0][2] * z.shapes[0][3]
        reshape_params = {
            "shape": (-1, flatten_size),
            "pre_transpose": [(0, 2, 3, 1)],
        }
        return jax.numpy.reshape.to_onnx(z, **reshape_params)


class MNIST_CNN(nnx.Module):
    """A simple CNN model."""

    def __init__(self, *, rngs: nnx.Rngs):
        """Initializes the CNN model with convolutional and linear layers."""

        self.layers: list[Supports2Onnx] = [
            nnx.Conv(1, 32, kernel_size=(3, 3), padding="SAME", rngs=rngs),
            nnx.relu,
            PartialWithOnnx(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2)),
            nnx.Conv(32, 64, kernel_size=(3, 3), padding="SAME", rngs=rngs),
            nnx.relu,
            PartialWithOnnx(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2)),
            ReshapeWithOnnx(),
            nnx.Linear(3136, 256, rngs=rngs),
            nnx.relu,
            nnx.Linear(256, 10, rngs=rngs),
            nnx.log_softmax,
        ]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Defines the forward pass of the model."""
        for layer in self.layers:
            x = layer(x)
        return x

    def to_onnx(self, z, **params):
        """Defines the ONNX export logic for the CNN model."""
        for layer in self.layers:
            z = layer.to_onnx(z, **params)
        return z


def get_test_params():
    """Defines test parameters for the MNIST CNN."""
    return [
        {
            "component": "CNN",
            "description": "A MNIST CNN model with convolutional and linear layers.",
            "children": [
                "flax.nnx.Conv",
                "flax.nnx.Linear",
                "flax.nnx.relu",
                "flax.nnx.avg_pool",
                "flax.nnx.reshape",
                "flax.nnx.log_softmax",
            ],
            "since": "v0.1.0",
            "testcases": [
                {
                    "testcase": "mnist_cnn",
                    "component": MNIST_CNN(rngs=nnx.Rngs(0)),
                    "input_shapes": [(1, 28, 28, 1)],  # (N, H, W, C) format for JAX
                    "params": {
                        "pre_transpose": [(0, 3, 1, 2)],  # Convert JAX → ONNX
                        # "post_transpose": [(0, 2, 3, 1)],  # Convert ONNX → JAX
                    },
                }
            ],
        }
    ]
