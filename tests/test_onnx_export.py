# file: tests/test_onnx_export.py
import pytest
import jax
import jax.numpy as jnp
import onnxruntime as ort
import importlib
import pkgutil
import os
import numpy as np

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

            # Überprüfen, ob plugin_params eine Liste oder ein Dictionary ist
            if isinstance(plugin_params, list):
                if all(isinstance(p, dict) for p in plugin_params):
                    for param in plugin_params:
                        param.setdefault("test_name", param.get("model_name", "Unnamed"))
                    params.extend(plugin_params)
                else:
                    raise ValueError(f"Plugin {name} must return a list of dictionaries.")
            elif isinstance(plugin_params, dict):
                plugin_params.setdefault("test_name", plugin_params.get("model_name", "Unnamed"))
                params.append(plugin_params)  # Anhängen des Dictionaries an die Liste
            else:
                raise ValueError(f"Plugin {name} must return a list or a dictionary.")

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
        # if param["model_name"] in ["linear"]
    ]

@pytest.mark.parametrize("test_params", load_test_params())
def test_onnx_export(test_params):
    model = test_params["model"]
    input_shapes = test_params["input_shapes"]  # Beachten Sie den Plural!
    seed = 0
    rng = jax.random.PRNGKey(seed)

    # Beispiel-Eingaben generieren
    jax_inputs = [jax.random.normal(rng, shape) for shape in input_shapes]

    # ONNX-Eingaben erstellen
    onnx_inputs = [transpose_to_onnx(input) for input in jax_inputs]

    model_instance = model()
    if hasattr(model_instance, "eval"):
        model_instance.eval()

    # Modell mit mehreren Eingaben aufrufen
    expected_output = model_instance(*jax_inputs)  # Beachten Sie den Stern!

    model_path = f"output/{test_params['model_name']}_model.onnx"
    os.makedirs("output", exist_ok=True)
    export_to_onnx(model_instance, jax_inputs, output_path=model_path, build_onnx_node=test_params["build_onnx_node"])  # Übergeben Sie nur den ersten Eingang für die Form
    ort_session = ort.InferenceSession(model_path)

    # ONNX-Eingabe-Dictionary erstellen
    # onnx_input = {ort_session.get_inputs()[i].name: jnp.array(onnx_input) for i, onnx_input in enumerate(onnx_inputs)}

    # onnx_output = ort_session.run(None, onnx_input)

    #onnx_input = {ort_session.get_inputs()[0].name: jnp.array(onnx_inputs[0])}
    # onnx_output = ort_session.run(None, onnx_input)[0]

    onnx_input = {ort_session.get_inputs()[0].name: np.array(onnx_inputs[0])}
    onnx_output = ort_session.run(None, onnx_input)[0]

    np.testing.assert_allclose(expected_output, transpose_to_jax(onnx_output), rtol=1e-3, atol=1e-5)
    print(f"Test for {test_params['model_name']} passed!")
