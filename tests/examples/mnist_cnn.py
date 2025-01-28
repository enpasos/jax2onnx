# file: tests/examples/mnist_cnn.py
from functools import partial
import jax
import jax.numpy as jnp
import onnx.helper as oh
from flax import nnx


class MNIST_CNN(nnx.Module):
    """A simple CNN model."""

    def __init__(self, *, rngs: nnx.Rngs):
        """Initializes the CNN model with convolutional and linear layers."""
        self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), padding='SAME', rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), padding='SAME', rngs=rngs)
        self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
        self.linear1 = nnx.Linear(3136, 256, rngs=rngs)
        self.linear2 = nnx.Linear(256, 10, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Defines the forward pass of the model."""
        x = self.avg_pool(nnx.relu(self.conv1(x)))
        x = self.avg_pool(nnx.relu(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)  # Flatten to (B, 3136)
        x =  self.linear1(x)
        x = nnx.relu(x)
        x = self.linear2(x)
        return x

    def build_onnx_node(self, xs, names, onnx_graph, parameters=None):
        """Defines the ONNX export logic for the CNN model."""
        # Conv1 + ReLU + AvgPool
        xs,  names = self.conv1.build_onnx_node(xs,  names, onnx_graph)
        xs,  names = jax.nn.relu.build_onnx_node(xs,  names, onnx_graph, parameters)
        xs,  names = nnx.avg_pool.build_onnx_node(xs,  names, onnx_graph, parameters)

        # Conv2 + ReLU + AvgPool
        xs,  names = self.conv2.build_onnx_node(xs,  names, onnx_graph)
        xs, names = jax.nn.relu.build_onnx_node(xs,  names, onnx_graph, parameters)
        xs,  names = nnx.avg_pool.build_onnx_node(xs,  names, onnx_graph, parameters)

        # Reshape
        reshape_params = {"shape": (xs[0].shape[0], 3136),
                          "apply_pre_transpose": True,  # Enable pre-transposition
                          "pre_transpose_perm": [0, 2, 3, 1],  # Custom NCHW → NHWC transposition
                          }
        xs, names = jax.numpy.reshape.build_onnx_node(xs,  names, onnx_graph, reshape_params)
        #
        # Linear1 + ReLU
        xs,  names = self.linear1.build_onnx_node(xs, names, onnx_graph)
        xs,  names = jax.nn.relu.build_onnx_node(xs,  names, onnx_graph, parameters)
        #
        # Linear2
        xs, names = self.linear2.build_onnx_node(xs,  names, onnx_graph)

        return xs,  names




def get_test_params():
    """Define test parameters for the MNIST CNN."""
    return [
        {
            "model_name": "mnist_cnn",
            "model": lambda: MNIST_CNN(rngs=nnx.Rngs(0)),
            "input_shapes": [(1, 28, 28, 1)],  # Updated for (N, H, W, C) as used in JAX
            "build_onnx_node": lambda jax_inputs, input_names, onnx_graph, parameters=None: (
                MNIST_CNN(rngs=nnx.Rngs(0)).build_onnx_node(jax_inputs, input_names, onnx_graph, parameters)
            ),
        }
    ]
