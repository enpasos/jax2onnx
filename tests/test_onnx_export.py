import pytest
import jax
import numpy as np
import onnxruntime as ort
import importlib
import pkgutil
import os

from jax2onnx.onnx_export import export_to_onnx, jax_shape_to_onnx_shape, onnx_shape_to_jax_shape, transpose_to_onnx, transpose_to_jax

def load_test_params():
    params = []

    # Load plugins
    package = 'jax2onnx.plugins'
    plugins_path = os.path.join(os.path.dirname(__file__), '../jax2onnx/plugins')
    for _, name, _ in pkgutil.iter_modules([plugins_path]):
        module = importlib.import_module(f'{package}.{name}')
        if hasattr(module, 'get_test_params'):
            print(f"Loading test params from plugin: {name}")
            plugin_params = module.get_test_params()
            if isinstance(plugin_params, list) and all(isinstance(p, dict) for p in plugin_params):
                for param in plugin_params:
                    param.setdefault("test_name", param.get("model_name", "Unnamed"))
                params.extend(plugin_params)
            else:
                raise ValueError(f"Plugin {name} must return a list of dictionaries.")

    # Load examples
    examples_package = 'tests.examples'
    examples_path = os.path.join(os.path.dirname(__file__), 'examples')
    for _, name, _ in pkgutil.iter_modules([examples_path]):
        module = importlib.import_module(f'{examples_package}.{name}')
        if hasattr(module, 'get_test_params'):
            print(f"Loading test params from example: {name}")
            example_params = module.get_test_params()
            if isinstance(example_params, list) and all(isinstance(p, dict) for p in example_params):
                for param in example_params:
                    param.setdefault("test_name", param.get("model_name", "Unnamed"))
                params.extend(example_params)
            else:
                raise ValueError(f"Example {name} must return a list of dictionaries.")

    # Wrap params with pytest.param to set custom test names
    return [
        pytest.param(param, id=param["test_name"])
        for param in params
        # filter only conv
        # if param["model_name"] in ["batchnorm"]
    ]

@pytest.mark.parametrize("test_params", load_test_params())
def test_onnx_export(test_params):
    model = test_params["model"]
    input_shape = test_params["input_shape"]

    seed = 0
    rng = jax.random.PRNGKey(seed)
    example_input = jax.random.normal(rng, input_shape)


    onnx_input =    transpose_to_onnx(example_input)

    model_instance = model()

    # if model_instance has function eval, call it
    if hasattr (model_instance, "eval"):
        model_instance.eval()

    expected_output = np.array(model_instance(example_input))

    model_path = f"output/{test_params['model_name']}_model.onnx"

    os.makedirs("output", exist_ok=True)
    export_to_onnx(model_instance, example_input, output_path=model_path, build_onnx_node=test_params["build_onnx_node"])

    ort_session = ort.InferenceSession(model_path)

    onnx_input = {ort_session.get_inputs()[0].name: np.array(onnx_input)}
    onnx_output = ort_session.run(None, onnx_input)[0]

    np.testing.assert_allclose(expected_output, transpose_to_jax(onnx_output ), rtol=1e-3, atol=1e-5)
    print(f"Test for {test_params['model_name']} passed!")


