# tests/test_onnx_export.py
import pytest
import jax
import numpy as np
import onnxruntime as ort
from flax import nnx
from jax2onnx.onnx_export import export_to_onnx

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


@pytest.mark.parametrize("model_name, model, input_shape, build_onnx_node", [
    ("linear", lambda: nnx.Linear(5, 3, rngs=nnx.Rngs(0)), (1, 5), None),
    ("mlp", lambda: MLP(30, 10), (1, 30), None),
    ("relu", lambda: lambda x: jax.nn.relu(x), (1, 10), lambda example_input, input_name, nodes, parameters, counter: jax.nn.relu.build_onnx_node(example_input, input_name, nodes, parameters, counter)),
])
def test_onnx_export(model_name, model, input_shape, build_onnx_node):
    seed = 0
    rng = jax.random.PRNGKey(seed)
    output_path = f"output/{model_name}_model.onnx"
    example_input = jax.random.normal(rng, input_shape)

    model_instance = model()
    expected_output = model_instance(example_input)

    export_to_onnx(model_instance, example_input, output_path=output_path, build_onnx_node=build_onnx_node)

    ort_session = ort.InferenceSession(output_path)
    onnx_input = {ort_session.get_inputs()[0].name: np.array(example_input)}
    onnx_output = ort_session.run(None, onnx_input)[0]

    np.testing.assert_allclose(expected_output, onnx_output, rtol=1e-3, atol=1e-5)
    print("Test passed!")
