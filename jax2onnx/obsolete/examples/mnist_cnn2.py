# file: tests/examples/mnist_cnn2.py
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from obsolete.convert import Z

import jax2onnx.plugins  # noqa: F401
from obsolete.typing_helpers import PartialWithOnnx, Supports2Onnx


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


class CNN(nnx.Module):
    """A CNN model with convolutional, layer norm, pooling, dropout, and linear layers."""

    def __init__(self, *, rngs: nnx.Rngs):
        """Initializes the CNN model with a structured layers list."""

        params = {"window_shape": (2, 2), "strides": (2, 2), "padding": "VALID"}

        self.layers: list[Supports2Onnx] = [
            nnx.Conv(1, 64, kernel_size=(3, 3), padding="SAME", rngs=rngs),
            nnx.LayerNorm(
                num_features=64 * 28 * 28,
                reduction_axes=(1, 2, 3),
                feature_axes=(1, 2, 3),
                rngs=rngs,
            ),
            nnx.relu,
            nnx.Conv(64, 64, kernel_size=(3, 3), padding="SAME", rngs=rngs),
            nnx.LayerNorm(
                num_features=64 * 28 * 28,
                reduction_axes=(1, 2, 3),
                feature_axes=(1, 2, 3),
                rngs=rngs,
            ),
            nnx.relu,
            PartialWithOnnx(nnx.max_pool, **params),
            nnx.Dropout(rate=0.5, rngs=rngs),
            nnx.Conv(64, 128, kernel_size=(3, 3), padding="SAME", rngs=rngs),
            nnx.LayerNorm(
                num_features=128 * 14 * 14,
                reduction_axes=(1, 2, 3),
                feature_axes=(1, 2, 3),
                rngs=rngs,
            ),
            nnx.relu,
            nnx.Conv(128, 128, kernel_size=(3, 3), padding="SAME", rngs=rngs),
            nnx.LayerNorm(
                num_features=128 * 14 * 14,
                reduction_axes=(1, 2, 3),
                feature_axes=(1, 2, 3),
                rngs=rngs,
            ),
            nnx.relu,
            PartialWithOnnx(nnx.max_pool, **params),
            nnx.Dropout(rate=0.5, rngs=rngs),
            ReshapeWithOnnx(),
            nnx.Linear(6272, 512, rngs=rngs),
            nnx.LayerNorm(num_features=512, rngs=rngs),
            nnx.relu,
            nnx.Dropout(rate=0.25, rngs=rngs),
            nnx.Linear(512, 1024, rngs=rngs),
            nnx.LayerNorm(num_features=1024, rngs=rngs),
            nnx.relu,
            nnx.Dropout(rate=0.5, rngs=rngs),
            nnx.Linear(1024, 10, rngs=rngs),
            nnx.log_softmax,
        ]

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """Defines the forward pass of the model using `self.layers`."""
        for layer in self.layers:
            if isinstance(layer, nnx.Dropout):
                x = layer(x, deterministic=deterministic)
            else:
                x = layer(x)
        return x

    def to_onnx(self, z, **params):
        """Defines the ONNX export logic for the CNN model using `self.layers`."""

        for layer in self.layers:
            z = layer.to_onnx(z, **params)
        return z


def get_test_params():
    """Define test parameters for the CNN."""
    return [
        {
            "component": "CNN",
            "description": "A MNIST CNN model with convolutional, layer norm, pooling, dropout, and linear layers.",
            "children": [
                "flax.nnx.Conv",
                "flax.nnx.Linear",
                "flax.nnx.relu",
                "flax.nnx.avg_pool",
                "flax.nnx.reshape",
                "flax.nnx.log_softmax",
                "flax.nnx.LayerNorm",
                "flax.nnx.Dropout",
                "flax.nnx.max_pool",
            ],
            "since": "v0.1.0",
            "testcases": [
                {
                    "testcase": "mnist_cnn_2",
                    "component": CNN(rngs=nnx.Rngs(0)),
                    "input_shapes": [(1, 28, 28, 1)],
                    "params": {"pre_transpose": [(0, 3, 1, 2)]},
                }
            ],
        }
    ]
