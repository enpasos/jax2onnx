# file: tests/test_to_onnx.py

import pytest
import jax
import onnxruntime as ort
import importlib
import os
import numpy as np
from flax import nnx
from jax2onnx.to_onnx import to_onnx


def load_test_params():
    params = []

    # Load plugins
    plugins_path = os.path.join(os.path.dirname(__file__), "../jax2onnx/plugins")
    examples_path = os.path.join(
        os.path.dirname(__file__), "../jax2onnx/examples"
    )  # ✅ Add examples path

    def load_tests_from_directory(base_path):
        """Helper function to load testcases from a given directory."""
        for dirpath, _, filenames in os.walk(base_path):
            for filename in filenames:
                if filename.endswith(".py") and filename != "__init__.py":
                    module_path = os.path.join(dirpath, filename)
                    module_name = module_path.replace("/", ".").replace(".py", "")

                    spec = importlib.util.spec_from_file_location(
                        module_name, module_path
                    )
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    if hasattr(module, "get_test_params"):
                        test_params = module.get_test_params()
                        if isinstance(test_params, list):
                            for entry in test_params:
                                # Unpack multiple testcases per component
                                if "testcases" in entry:
                                    for test in entry["testcases"]:
                                        test["jax_component"] = entry[
                                            "jax_component"
                                        ]  # Add reference
                                        params.append(test)

    # ✅ Load tests from plugins and examples
    load_tests_from_directory(plugins_path)
    load_tests_from_directory(examples_path)

    # Wrap params with pytest.param to set custom test names
    return [pytest.param(param, id=param["testcase"]) for param in params]


@pytest.mark.parametrize("test_params", load_test_params())
def test_onnx_export(test_params):

    component = test_params.get("component", None)

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

    if isinstance(component, nnx.Module):
        expected_outputs = [component(*inputs)]
    elif z.jax_function is not None and callable(z.jax_function):
        expected_outputs = [z.jax_function(*inputs)]
    else:
        raise ValueError("Cannot call JAX function or module")

    # Assert the results
    for i in range(len(expected_outputs)):
        np.testing.assert_allclose(
            onnx_outputs[i], expected_outputs[i], rtol=1e-2, atol=1e-4
        )
    print(f"Test for {test_params['testcase']} passed!")
