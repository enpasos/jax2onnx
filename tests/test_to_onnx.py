# file: tests/test_to_onnx.py
import flax.nnx
import pytest
import jax
import onnxruntime as ort
import importlib
import pkgutil
import os
import numpy as np

from jax2onnx.to_onnx import to_onnx
from jax2onnx.typing_helpers import supports_onnx


def load_test_params():
    params = []

    # Load plugins
    package = "jax2onnx.plugins"
    plugins_path = os.path.join(os.path.dirname(__file__), "../jax2onnx/plugins")

    for _, name, _ in pkgutil.iter_modules([plugins_path]):
        module = importlib.import_module(f"{package}.{name}")
        if hasattr(module, "get_test_params"):
            print(f"Loading test params from plugin: {name}")
            plugin_params = module.get_test_params()

            if isinstance(plugin_params, list):
                if all(isinstance(p, dict) for p in plugin_params):
                    for param in plugin_params:
                        param.setdefault("test_name", param.get("testcase", "Unnamed"))
                    params.extend(plugin_params)
                else:
                    raise ValueError(
                        f"Plugin {name} must return a list of dictionaries."
                    )
            elif isinstance(plugin_params, dict):
                plugin_params.setdefault(
                    "test_name", plugin_params.get("testcase", "Unnamed")
                )
                params.append(plugin_params)
            else:
                raise ValueError(f"Plugin {name} must return a list or a dictionary.")

    # Load examples
    examples_package = "tests.examples"
    examples_path = os.path.join(os.path.dirname(__file__), "examples")
    for _, name, _ in pkgutil.iter_modules([examples_path]):
        module = importlib.import_module(f"{examples_package}.{name}")
        if hasattr(module, "get_test_params"):
            print(f"Loading test params from example: {name}")
            example_params = module.get_test_params()
            if isinstance(example_params, list) and all(
                isinstance(p, dict) for p in example_params
            ):
                for param in example_params:
                    param.setdefault("test_name", param.get("testcase", "Unnamed"))
                params.extend(example_params)
            else:
                raise ValueError(f"Example {name} must return a list of dictionaries.")

    # Wrap params with pytest.param to set custom test names
    return [
        pytest.param(param, id=param["test_name"])
        for param in params
        # filter only conv
        # if param["testcase"] in [ "einsum_attention" ]
    ]


@pytest.mark.parametrize("test_params", load_test_params())
def test_onnx_export(test_params):

    component = test_params.get("component", None)

    if not supports_onnx(component):
        raise TypeError(
            f"Component {type(component).__name__} does not support ONNX export."
        )

    if hasattr(component, "eval"):
        component.eval()

    input_shapes = test_params["input_shapes"]  # Note the plural!
    params = test_params.get("params", {})  # Get params from the test case
    seed = 0
    rng = jax.random.PRNGKey(seed)

    # Generate JAX inputs
    inputs = [jax.random.normal(rng, shape) for shape in input_shapes]

    # Export the jax_model to ONNX
    onnx_model_file_name = f"{test_params['testcase']}_model.onnx"
    model_path = f"output/{onnx_model_file_name}"
    os.makedirs("output", exist_ok=True)

    z = to_onnx(
        onnx_model_file_name,
        component,
        input_shapes,
        output_path=model_path,
        params=params,
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
    # if component is instance of flax.nnx.Module

    if isinstance(component, flax.nnx.Module):
        expected_outputs = [component(*inputs)]
    # else if to_onnx_function is not None and is a function
    elif z.jax_function is not None and callable(z.jax_function):
        expected_outputs = [z.jax_function(*inputs)]
    else:
        raise ValueError("Can not call JAX function or module")

    # Assert the results
    for i in range(len(expected_outputs)):
        np.testing.assert_allclose(
            onnx_outputs[i], expected_outputs[i], rtol=1e-2, atol=1e-4
        )
    print(f"Test for {test_params['testcase']} passed!")
