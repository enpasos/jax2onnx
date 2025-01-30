# file: tests/examples/mnist_cnn.py
from flax import nnx
import jax
import jax.numpy as jnp
import onnx.helper as oh


class MLP(nnx.Module):
    def __init__(self, in_features, out_features, *, rngs=nnx.Rngs(0)):
        self.layer = nnx.Linear(in_features, out_features, rngs=rngs)

    def __call__(self, x):
        x = self.layer(x)
        x = jax.nn.relu(x)
        return x

    def build_onnx_node(self, xs, input_names, onnx_graph, parameters=None):
        # Generate the ONNX node for the Linear layer
        xs, linear_output_names = self.layer.build_onnx_node(xs, input_names, onnx_graph)

        # Add ReLU activation node
        xs, relu_output_names = jax.nn.relu.build_onnx_node(xs, linear_output_names, onnx_graph, parameters)


        return xs, relu_output_names



def get_test_params():
    """
    Test parameters for the MLP model.
    """
    return [
        {
            "model_name": "mlp",
            "model": lambda: MLP(30, 10, rngs=nnx.Rngs(0)),
            "input_shapes": [(1, 30)],
            "build_onnx_node": lambda jax_inputs, input_names, onnx_graph, parameters=None: (
                MLP(30, 10, rngs=nnx.Rngs(0)).build_onnx_node(jax_inputs, input_names, onnx_graph, parameters)
            )
        }
    ]
