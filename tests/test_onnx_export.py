# file: tests/test_onnx_export.py
import pytest
import jax
import jax.numpy as jnp
import onnxruntime as ort
import importlib
import pkgutil
import os
import numpy as np

from jax2onnx.onnx_export import export_to_onnx, transpose_to_onnx, transpose_to_jax, OnnxGraph

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

            if isinstance(plugin_params, list):
                if all(isinstance(p, dict) for p in plugin_params):
                    for param in plugin_params:
                        param.setdefault("test_name", param.get("model_name", "Unnamed"))
                    params.extend(plugin_params)
                else:
                    raise ValueError(f"Plugin {name} must return a list of dictionaries.")
            elif isinstance(plugin_params, dict):
                plugin_params.setdefault("test_name", plugin_params.get("model_name", "Unnamed"))
                params.append(plugin_params)
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
        # if param["model_name"] in ["reshape2"]
    ]

@pytest.mark.parametrize("test_params", load_test_params())
def test_onnx_export(test_params):
    model = test_params["model"]
    input_shapes = test_params["input_shapes"]  # Note the plural!
    parameters = test_params.get("parameters", {})  # Get parameters from the test case
    seed = 0
    rng = jax.random.PRNGKey(seed)

    # Generate JAX inputs
    jax_inputs = [jax.random.normal(rng, shape) for shape in input_shapes]

    # Transpose inputs to ONNX format
    onnx_inputs = [transpose_to_onnx(input) for input in jax_inputs]

    # Initialize model
    model_instance = model()
    if hasattr(model_instance, "eval"):
        model_instance.eval()

    # Compute expected JAX output
    #expected_output = model_instance(*jax_inputs)

    # Export the model to ONNX
    model_file_name = f"{test_params['model_name']}_model.onnx"
    model_path = f"output/{model_file_name}"
    os.makedirs("output", exist_ok=True)
    expected_outputs =  export_to_onnx(
        model_file_name,
        model_instance,
        jax_inputs,
        output_path=model_path,
        build_onnx_node=test_params["build_onnx_node"],
        parameters=parameters,
    )

    # Load the ONNX model
    ort_session = ort.InferenceSession(model_path)

    # Create ONNX input dictionary
    onnx_inputs_dict = {
        ort_session.get_inputs()[i].name: np.array(onnx_input)
        for i, onnx_input in enumerate(onnx_inputs)
    }

    # Compute ONNX output
    onnx_output = ort_session.run(None, onnx_inputs_dict)[0]

    # Transpose ONNX output back to JAX format
    onnx_output_jax = transpose_to_jax(onnx_output)

    # Assert the results
    np.testing.assert_allclose(
        expected_outputs[0],
        onnx_output_jax,
        rtol=1e-3,
        atol=1e-5
    )
    print(f"Test for {test_params['model_name']} passed!")
