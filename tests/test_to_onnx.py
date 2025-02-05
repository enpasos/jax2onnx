# file: tests/test_to_onnx.py
import pytest
import jax
import jax.numpy as jnp
import onnxruntime as ort
import importlib
import pkgutil
import os
import numpy as np

from jax2onnx.to_onnx import to_onnx,  OnnxGraph
from transpose_utils import onnx_to_jax_axes


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
        # if param["model_name"] in [  "mnist_cnn" ]
    ]

@pytest.mark.parametrize("test_params", load_test_params())
def test_onnx_export(test_params):

    jax_model = test_params.get("model", None)
    if hasattr(jax_model, "eval"):
        jax_model.eval()

    input_shapes = test_params["input_shapes"]  # Note the plural!
    export_params = test_params.get("export", {})  # Get export_params from the test case
    seed = 0
    rng = jax.random.PRNGKey(seed)

    # Generate JAX inputs
    inputs = [jax.random.normal(rng, shape) for shape in input_shapes]


    # Export the jax_model to ONNX
    onnx_model_file_name = f"{test_params['model_name']}_model.onnx"
    model_path = f"output/{onnx_model_file_name}"
    os.makedirs("output", exist_ok=True)

    to_onnx_function = test_params.get("to_onnx", None)
    z = to_onnx(
        onnx_model_file_name,
        jax_model,
        input_shapes,
        output_path=model_path,
        # Provide a default function or None if no conversion is needed
        to_onnx=to_onnx_function,
        parameters=export_params,
    )

    # Load the ONNX jax_model
    ort_session = ort.InferenceSession(model_path)


    # Create ONNX input dictionary
    onnx_inputs_dict = {
        ort_session.get_inputs()[i].name: np.array(onnx_input)
        for i, onnx_input in enumerate(inputs)
    }

    # Compute ONNX output
    onnx_outputs = ort_session.run(None, onnx_inputs_dict)

    # Compute JAX output
    # for now one input, one output
    # call model or function
    if (jax_model is not None) and callable(jax_model):
        expected_outputs = [jax_model(*inputs)]
    # else if to_onnx_function is not None and is a function
    elif (to_onnx_function is not None and callable(to_onnx_function)):
        expected_outputs = [z.jax_function(*inputs)]
    else:
        raise ValueError("No model or to_onnx function provided")

    # Assert the results
    for i in range(len(expected_outputs)):
        np.testing.assert_allclose(
            onnx_outputs[i],
            expected_outputs[i],
            rtol=1e-3,
            atol=1e-5
        )
    print(f"Test for {test_params['model_name']} passed!")
