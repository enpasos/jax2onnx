# file: tests/examples/mlp.py
from flax import nnx
import jax

class MLP(nnx.Module):

    def __init__(self, in_features, out_features, *, rngs=nnx.Rngs(0)):
        self.layer = nnx.Linear(in_features, out_features, rngs=rngs)

    def __call__(self, x):
        x = self.layer(x)
        x = jax.nn.relu(x)
        return x

    def build_onnx_node(self, jax_inputs, input_names, onnx_graph, parameters=None):
        output_name = self.layer.build_onnx_node(jax_inputs, input_names, onnx_graph)
        return jax.nn.relu.build_onnx_node(self.layer(jax_inputs), output_name, onnx_graph, parameters)


def get_test_params():
    # Return a list of dictionaries
    return [
        {
            "model_name": "mlp",
            "model": lambda: MLP(30, 10),
            "input_shapes": [(1, 30)],
            "build_onnx_node": None,
         }
    ]
