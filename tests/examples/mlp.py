# file: tests/examples/mnist_cnn.py
from flax import nnx
import jax
import jax.numpy as jnp
import onnx.helper as oh
from functools import partial

class MLP(nnx.Module):
    def __init__(self, in_features, out_features, *, rngs=nnx.Rngs(0)):
        self.layer = nnx.Linear(in_features, out_features, rngs=rngs)
        self.activation = partial(jax.nn.relu)

    def __call__(self, x):
        x = self.layer(x)
        x = self.activation(x)
        return x

    def build_onnx(self, xs,  names, onnx_graph, parameters=None):
        # Generate the ONNX node for the Linear layer
        xs, names = self.layer.build_onnx(xs, names, onnx_graph)
        xs, names = jax.nn.relu.build_onnx(self.activation, xs, names, onnx_graph, parameters)


        return xs, names



def get_test_params():
    """
    Test parameters for the MLP model.
    """
    return [
        {
            "model_name": "mlp",
            "model": lambda: MLP(30, 10, rngs=nnx.Rngs(0)),
            "input_shapes": [(1, 30)],
            "build_onnx": lambda jax_inputs, input_names, onnx_graph, parameters=None: (
                MLP(30, 10, rngs=nnx.Rngs(0)).build_onnx(jax_inputs, input_names, onnx_graph, parameters)
            )
        }
    ]
