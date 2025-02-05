# file: tests/examples/mnist_cnn2.py

from functools import partial
import jax
import jax.numpy as jnp
import onnx.helper as oh
from flax import nnx
from jax2onnx.to_onnx import Z


class CNN(nnx.Module):
    """A CNN model with the specified architecture."""

    def __init__(self, *, rngs: nnx.Rngs):
        """Initializes the CNN model with convolutional, layer normalization, pooling, dropout, and linear layers."""
        self.conv1 = nnx.Conv(1, 64, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.ln1 = nnx.LayerNorm(num_features=64 * 28 * 28, reduction_axes=(1, 2, 3), feature_axes=(1, 2, 3), rngs=rngs)
        self.conv2 = nnx.Conv(64, 64, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.ln2 = nnx.LayerNorm(num_features=64 * 28 * 28, reduction_axes=(1, 2, 3), feature_axes=(1, 2, 3), rngs=rngs)

        params = {"window_shape": (2, 2), "strides":(2, 2), "padding": "VALID"}
        self.pool1 = partial(nnx.max_pool, **params)
        self.pool1.to_onnx = lambda z, _: nnx.max_pool.to_onnx(z, params)

        self.dropout1 = nnx.Dropout(rate=0.5, rngs=rngs)

        self.conv3 = nnx.Conv(64, 128, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.ln3 = nnx.LayerNorm(num_features=128 * 14 * 14, reduction_axes=(1, 2, 3), feature_axes=(1, 2, 3), rngs=rngs)
        self.conv4 = nnx.Conv(128, 128, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.ln4 = nnx.LayerNorm(num_features=128 * 14 * 14, reduction_axes=(1, 2, 3), feature_axes=(1, 2, 3), rngs=rngs)

        params = {"window_shape": (2, 2), "strides":(2, 2), "padding": "VALID"}
        self.pool2 = partial(nnx.max_pool, **params)
        self.pool2.to_onnx = lambda z, _: nnx.max_pool.to_onnx(z, params)

        self.dropout2 = nnx.Dropout(rate=0.5, rngs=rngs)

        self.linear1 = nnx.Linear(6272, 512, rngs=rngs)
        self.ln5 = nnx.LayerNorm(num_features=512, rngs=rngs)
        self.dropout3 = nnx.Dropout(rate=0.25, rngs=rngs)
        self.linear2 = nnx.Linear(512, 1024, rngs=rngs)
        self.ln6 = nnx.LayerNorm(num_features=1024, rngs=rngs)
        self.dropout4 = nnx.Dropout(rate=0.5, rngs=rngs)
        self.linear3 = nnx.Linear(1024, 10, rngs=rngs)


        self.reshape = lambda x: x.reshape(x.shape[0], -1)

        def reshape_to_onnx(z, _):
            flatten_size = z.shapes[0][1] * z.shapes[0][2] * z.shapes[0][3]
            reshape_params = {"shape": (z.shapes[0][0], flatten_size),
                              "pre_transpose": [(0, 2, 3, 1)]}
            return jax.numpy.reshape.to_onnx(z, reshape_params)


        self.reshape.to_onnx = reshape_to_onnx

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


    def to_onnx(self, z, parameters=None):
        """Defines the ONNX export logic for the CNN model."""
        for layer in [
            self.conv1, self.ln1, self.relu,
            self.conv2, self.ln2, self.relu, self.pool1, self.dropout1,
            self.conv3, self.ln3, self.relu,
            self.conv4, self.ln4, self.relu, self.pool2, self.dropout2,
            self.reshape,
            self.linear1, self.ln5, self.relu, self.dropout3,
            self.linear2, self.ln6, self.relu, self.dropout4,
            self.linear3, jax.nn.log_softmax
        ]:
            z = layer.to_onnx(z, parameters)
        return z


def get_test_params():
    """Define test parameters for the CNN."""
    return [
        {
            "model_name": "mnist_cnn_2",
            "model":  CNN(rngs=nnx.Rngs(0)),
            "input_shapes": [(1, 28, 28, 1)],
            "export": {"pre_transpose": [(0, 3, 1, 2)]},
        }
    ]
