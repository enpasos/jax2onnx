from flax import nnx
import jax

class MLP(nnx.Module):
    features: list

    def __init__(self, in_features, out_features, *, rngs=nnx.Rngs(0)):
        self.layer = nnx.Linear(in_features, out_features, rngs=rngs)

    def __call__(self, x):
        x = self.layer(x)
        x = jax.nn.relu(x)
        return x

    def build_onnx_node(self, example_input, input_name, nodes, parameters, counter):
        output_name = self.layer.build_onnx_node(example_input, input_name, nodes, parameters, counter)
        return jax.nn.relu.build_onnx_node(self.layer(example_input), output_name, nodes, parameters, counter)



def get_test_params():
    return {
        "model_name": "mlp",
        "model": lambda: MLP(30, 10),
        "input_shape": (1, 30),
        "build_onnx_node": None,
    }
