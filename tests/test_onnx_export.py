import jax
import numpy as np
import pytest
import onnxruntime as ort
from flax import nnx
from jax2onnx.onnx_export import export_to_onnx



def test_linear():
    seed = 0
    in_features = 5
    out_features = 3

    linear = nnx.Linear(in_features, out_features, rngs=nnx.Rngs(seed))

    input_shape = (1, in_features)

    # Generate random input
    jax_input = jax.random.normal(jax.random.PRNGKey(seed), input_shape)
    jax_output = linear(jax_input)

    export_to_onnx(linear, jax_input, "output/linear_model.onnx")

    session = ort.InferenceSession("output/linear_model.onnx")
    onnx_input = np.array(jax_input)
    onnx_output = session.run(None, {"input": onnx_input})

    assert np.allclose(jax_output, onnx_output[0], atol=1e-6)

def test_relu():
    seed = 0
    input_shape = (1, 10)

    # Generate random input
    jax_input = jax.random.normal(jax.random.PRNGKey(seed), input_shape)
    jax_output = jax.nn.relu(jax_input)

    # Mock model for relu
    class ReluModel(nnx.Module):
        def __call__(self, x):
            return jax.nn.relu(x)

        def build_onnx_node(self, example_input, input_name, nodes, parameters, counter):
            return nnx.Module.build_relu_onnx_node(example_input, input_name, nodes, parameters, counter)

    relu_model = ReluModel()

    export_to_onnx(relu_model, jax_input, "output/relu_model.onnx")

    session = ort.InferenceSession("output/relu_model.onnx")
    onnx_input = np.array(jax_input)
    onnx_output = session.run(None, {"input": onnx_input})

    assert np.allclose(jax_output, onnx_output[0], atol=1e-6)

def test_mlp():
    seed = 0
    in_features = 30
    out_features = 10

    class MLP(nnx.Module):
        features: list

        def __init__(self, in_features, out_features, *, rngs: nnx.Rngs):
            self.layer = nnx.Linear(in_features, out_features, rngs=rngs)

        def __call__(self, x):
            x = self.layer(x)
            x = jax.nn.relu(x)
            return x

        def build_onnx_node(self, example_input, input_name, nodes, parameters, counter):
            output_name = self.layer.build_onnx_node(example_input, input_name, nodes, parameters, counter)
            return nnx.Module.build_relu_onnx_node(self.layer(example_input), output_name, nodes, parameters, counter)

    mlp = MLP(in_features, out_features, rngs=nnx.Rngs(seed))

    input_shape = (1, in_features)

    # Generate random input
    jax_input = jax.random.normal(jax.random.PRNGKey(seed), input_shape)
    jax_output = mlp(jax_input)

    export_to_onnx(mlp, jax_input, "output/mlp_model.onnx")

    session = ort.InferenceSession("output/mlp_model.onnx")
    onnx_input = np.array(jax_input)
    onnx_output = session.run(None, {"input": onnx_input})

    assert np.allclose(jax_output, onnx_output[0], atol=1e-6)

if __name__ == "__main__":
    pytest.main()
