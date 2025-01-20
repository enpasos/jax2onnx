import pytest
import jax
import numpy as np
import onnxruntime as ort
import importlib
import pkgutil
from jax2onnx.onnx_export import export_to_onnx
import os

def load_test_params():
    params = []

    # Load plugins
    package = 'jax2onnx.plugins'
    plugins_path = os.path.join(os.path.dirname(__file__), '../jax2onnx/plugins')
    for _, name, _ in pkgutil.iter_modules([plugins_path]):
        module = importlib.import_module(f'{package}.{name}')
        print(f"Loading test params from plugin: {name}")
        params.append(module.get_test_params())

    # Load examples
    examples_package = 'tests.examples'
    examples_path = os.path.join(os.path.dirname(__file__), 'examples')
    for _, name, _ in pkgutil.iter_modules([examples_path]):
        module = importlib.import_module(f'{examples_package}.{name}')
        if hasattr(module, 'get_test_params'):
            print(f"Loading test params from example: {name}")
            params.append(module.get_test_params())

    return params


@pytest.mark.parametrize("test_params", load_test_params())
def test_onnx_export(test_params):
    model = test_params["model"]
    input_shape = test_params["input_shape"]

    seed = 0
    rng = jax.random.PRNGKey(seed)
    example_input = jax.random.normal(rng, input_shape)

    model_instance = model()
    expected_output = model_instance(example_input)

    model_path = f"output/{test_params['model_name']}_model.onnx"

    export_to_onnx(model_instance, example_input, output_path=model_path, build_onnx_node=test_params["build_onnx_node"])

    ort_session = ort.InferenceSession(model_path)
    onnx_input = {ort_session.get_inputs()[0].name: np.array(example_input)}
    onnx_output = ort_session.run(None, onnx_input)[0]

    np.testing.assert_allclose(expected_output, onnx_output, rtol=1e-3, atol=1e-5)
    print(f"Test for {test_params['model_name']} passed!")
