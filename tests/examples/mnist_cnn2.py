# file: tests/examples/mnist_cnn2.py

import jax
import jax.numpy as jnp
from flax import nnx
from jax2onnx.typing_helpers import PartialWithOnnx


class ReshapeWithOnnx:
    """Wrapper class for reshape function with ONNX export support."""

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x.reshape(x.shape[0], -1)

    def to_onnx(self, z, **params):
        flatten_size = z.shapes[0][1] * z.shapes[0][2] * z.shapes[0][3]
        reshape_params = {
            "shape": (z.shapes[0][0], flatten_size),
            "pre_transpose": [(0, 2, 3, 1)],
        }
        return jax.numpy.reshape.to_onnx(z, **reshape_params)


class CNN(nnx.Module):
    """A CNN model with the specified architecture."""

    def __init__(self, *, rngs: nnx.Rngs):
        """Initializes the CNN model with convolutional, layer normalization, pooling, dropout, and linear layers."""
        self.conv1 = nnx.Conv(1, 64, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.ln1 = nnx.LayerNorm(
            num_features=64 * 28 * 28,
            reduction_axes=(1, 2, 3),
            feature_axes=(1, 2, 3),
            rngs=rngs,
        )
        self.conv2 = nnx.Conv(64, 64, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.ln2 = nnx.LayerNorm(
            num_features=64 * 28 * 28,
            reduction_axes=(1, 2, 3),
            feature_axes=(1, 2, 3),
            rngs=rngs,
        )

        params = {"window_shape": (2, 2), "strides": (2, 2), "padding": "VALID"}
        self.pool1 = PartialWithOnnx(nnx.max_pool, **params)
        self.dropout1 = nnx.Dropout(rate=0.5, rngs=rngs)

        self.conv3 = nnx.Conv(64, 128, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.ln3 = nnx.LayerNorm(
            num_features=128 * 14 * 14,
            reduction_axes=(1, 2, 3),
            feature_axes=(1, 2, 3),
            rngs=rngs,
        )
        self.conv4 = nnx.Conv(128, 128, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.ln4 = nnx.LayerNorm(
            num_features=128 * 14 * 14,
            reduction_axes=(1, 2, 3),
            feature_axes=(1, 2, 3),
            rngs=rngs,
        )

        self.pool2 = PartialWithOnnx(nnx.max_pool, **params)
        self.dropout2 = nnx.Dropout(rate=0.5, rngs=rngs)

        self.linear1 = nnx.Linear(6272, 512, rngs=rngs)
        self.ln5 = nnx.LayerNorm(num_features=512, rngs=rngs)
        self.dropout3 = nnx.Dropout(rate=0.25, rngs=rngs)
        self.linear2 = nnx.Linear(512, 1024, rngs=rngs)
        self.ln6 = nnx.LayerNorm(num_features=1024, rngs=rngs)
        self.dropout4 = nnx.Dropout(rate=0.5, rngs=rngs)
        self.linear3 = nnx.Linear(1024, 10, rngs=rngs)

        self.reshape = ReshapeWithOnnx()
        self.relu = jax.nn.relu

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """Defines the forward pass of the model."""
        x = self.relu(self.ln1(self.conv1(x)))
        x = self.relu(self.ln2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x, deterministic=deterministic)

        x = self.relu(self.ln3(self.conv3(x)))
        x = self.relu(self.ln4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x, deterministic=deterministic)

        x = self.reshape(x)
        x = self.relu(self.ln5(self.linear1(x)))
        x = self.dropout3(x, deterministic=deterministic)
        x = self.relu(self.ln6(self.linear2(x)))
        x = self.dropout4(x, deterministic=deterministic)
        x = nnx.log_softmax(self.linear3(x))
        return x

    def to_onnx(self, z, **params):
        """Defines the ONNX export logic for the CNN model."""

        # Extract parameters specific to transpose operations
        transpose_params = {
            k: v for k, v in params.items() if k in ["pre_transpose", "post_transpose"]
        }

        # General parameters (excluding transpose-specific ones)
        general_params = {k: v for k, v in params.items() if k not in transpose_params}

        for layer in [
            self.conv1,
            self.ln1,
            jax.nn.relu,
            self.conv2,
            self.ln2,
            jax.nn.relu,
            self.pool1,
            self.dropout1,
            self.conv3,
            self.ln3,
            jax.nn.relu,
            self.conv4,
            self.ln4,
            jax.nn.relu,
            self.pool2,
            self.dropout2,
            self.reshape,
            self.linear1,
            self.ln5,
            jax.nn.relu,
            self.dropout3,
            self.linear2,
            self.ln6,
            jax.nn.relu,
            self.dropout4,
            self.linear3,
            jax.nn.log_softmax,
        ]:

            # Ensure the layer has a `to_onnx` method before calling it
            if hasattr(layer, "to_onnx"):
                # Pass full parameters for layers that support them
                if isinstance(
                    layer, (nnx.Conv, ReshapeWithOnnx)
                ):  # Layers that use `pre_transpose`
                    z = layer.to_onnx(z, **params)
                else:
                    z = layer.to_onnx(z, **general_params)  # Pass only general params

        return z


def get_test_params():
    """Define test parameters for the CNN."""
    return [
        {
            "testcase": "mnist_cnn_2",
            "model": CNN(rngs=nnx.Rngs(0)),
            "input_shapes": [(1, 28, 28, 1)],
            "params": {"pre_transpose": [(0, 3, 1, 2)]},
        }
    ]
