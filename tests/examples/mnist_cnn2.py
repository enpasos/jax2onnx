# file: tests/examples/mnist_cnn2.py

from functools import partial
import jax
import jax.numpy as jnp
import onnx.helper as oh
from flax import nnx


class CNN(nnx.Module):
    """A CNN model with the specified architecture."""


    def __init__(self, *, rngs: nnx.Rngs):
        """Initializes the CNN model with convolutional, layer normalization, pooling, dropout, and linear layers."""
        self.conv1 = nnx.Conv(1, 64, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.ln1 = nnx.LayerNorm(num_features=64*28*28, reduction_axes = (1, 2, 3), feature_axes=(1, 2, 3),  rngs=rngs)
        self.conv2 = nnx.Conv(64, 64, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.ln2 = nnx.LayerNorm(num_features=64*28*28, reduction_axes = (1, 2, 3), feature_axes=(1, 2, 3),  rngs=rngs)
        self.pool1 = partial(nnx.max_pool, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        self.dropout1 = nnx.Dropout(rate=0.5, rngs=rngs)

        self.conv3 = nnx.Conv(64, 128, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.ln3 = nnx.LayerNorm(num_features=128*14*14, reduction_axes = (1, 2, 3), feature_axes=(1, 2, 3),  rngs=rngs)
        self.conv4 = nnx.Conv(128, 128, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.ln4 = nnx.LayerNorm(num_features=128*14*14, reduction_axes = (1, 2, 3), feature_axes=(1, 2, 3),  rngs=rngs)
        self.pool2 = partial(nnx.max_pool, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        self.dropout2 = nnx.Dropout(rate=0.5, rngs=rngs)

        self.linear1 = nnx.Linear(6272, 512, rngs=rngs)
        self.ln5 = nnx.LayerNorm(num_features=512,   rngs=rngs)
        self.dropout3 = nnx.Dropout(rate=0.25, rngs=rngs)
        self.linear2 = nnx.Linear(512, 1024, rngs=rngs)
        self.ln6 = nnx.LayerNorm(num_features=1024 ,    rngs=rngs)
        self.dropout4 = nnx.Dropout(rate=0.5, rngs=rngs)
        self.linear3 = nnx.Linear(1024, 10, rngs=rngs)
        self.reshape = lambda x: x.reshape(x.shape[0], -1)
        self.relu = nnx.relu

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """Defines the forward pass of the model."""
        x = nnx.relu(self.ln1(self.conv1(x)))
        x = nnx.relu(self.ln2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x, deterministic=deterministic)

        x = nnx.relu(self.ln3(self.conv3(x)))
        x = nnx.relu(self.ln4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x, deterministic=deterministic)

        x = self.reshape(x)
        x = nnx.relu(self.ln5(self.linear1(x)))
        x = self.dropout3(x, deterministic=deterministic)
        x = nnx.relu(self.ln6(self.linear2(x)))
        x = self.dropout4(x, deterministic=deterministic)
        x = nnx.log_softmax(self.linear3(x))
        return x

    def to_onnx(self, xs, names, onnx_graph, parameters=None):
        """Defines the ONNX export logic for the CNN model."""
        # Conv1 + LN + ReLU + Conv2 + LN + ReLU + Pool + Dropout
        xs, names = self.conv1.to_onnx(xs, names, onnx_graph)
        xs, names = self.ln1.to_onnx(xs, names, onnx_graph )
        xs, names = jax.nn.relu.to_onnx(self.relu, xs, names, onnx_graph, parameters)
        xs, names = self.conv2.to_onnx(xs, names, onnx_graph)
        xs, names = self.ln2.to_onnx(xs, names, onnx_graph )
        xs, names = jax.nn.relu.to_onnx(self.relu, xs, names, onnx_graph, parameters)
        xs, names = nnx.max_pool.to_onnx(self.pool1, xs, names, onnx_graph, parameters)
        xs, names = self.dropout1.to_onnx(xs, names, onnx_graph, parameters)

        # Conv3 + LN + ReLU + Conv4 + LN + ReLU + Pool + Dropout
        xs, names = self.conv3.to_onnx(xs, names, onnx_graph)
        xs, names = self.ln3.to_onnx(xs, names, onnx_graph )
        xs, names = jax.nn.relu.to_onnx(self.relu, xs, names, onnx_graph, parameters)
        xs, names = self.conv4.to_onnx(xs, names, onnx_graph)
        xs, names = self.ln4.to_onnx(xs, names, onnx_graph )
        xs, names = jax.nn.relu.to_onnx(self.relu, xs, names, onnx_graph, parameters)
        xs, names = nnx.max_pool.to_onnx(self.pool2, xs, names, onnx_graph, parameters)
        xs, names = self.dropout2.to_onnx(xs, names, onnx_graph, parameters)

        # # Compute flatten size dynamically
        flatten_size = xs[0][1] * xs[0][2] * xs[0][3]

        # Reshape
        reshape_params = {"shape": (xs[0][0], flatten_size),
                          "pre_transpose": [(0, 2, 3, 1)]}
        xs, names = jax.numpy.reshape.to_onnx(self.reshape, xs,  names, onnx_graph, reshape_params)


        # Linear + LN + ReLU + Dropout + Linear + LN + ReLU + Dropout + Linear
        xs, names = self.linear1.to_onnx(xs, names, onnx_graph)
        xs, names = self.ln5.to_onnx(xs, names, onnx_graph )
        xs, names = jax.nn.relu.to_onnx(self.relu, xs, names, onnx_graph, parameters)
        xs, names = self.dropout3.to_onnx(xs, names, onnx_graph, parameters)
        xs, names = self.linear2.to_onnx(xs, names, onnx_graph)
        xs, names = self.ln6.to_onnx(xs, names, onnx_graph )
        xs, names = jax.nn.relu.to_onnx(self.relu, xs, names, onnx_graph, parameters)
        xs, names = self.dropout4.to_onnx(xs, names, onnx_graph, parameters)
        xs, names = self.linear3.to_onnx(xs, names, onnx_graph)
        xs, names = jax.nn.log_softmax.to_onnx(nnx.log_softmax, xs, names, onnx_graph)

        return xs, names


def get_test_params():
    """Define test parameters for the CNN."""
    return [
        {
            "model_name": "mnist_cnn_2",
            "model": lambda: CNN(rngs=nnx.Rngs(0)),
            "input_shapes": [(1, 28, 28, 1)],
            "to_onnx": CNN.to_onnx,
            "export": {
                "pre_transpose": [(0, 3, 1, 2)],

            }
        }
    ]
