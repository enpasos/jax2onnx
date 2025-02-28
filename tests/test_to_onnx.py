# file: tests/test_to_onnx.py

import pytest
import jax
import onnxruntime as ort
import importlib.util
import os
import numpy as np
from flax import nnx
from jax2onnx import to_onnx


def load_test_params() -> list:
    """Load test parameters from plugins and examples."""
    params = []

    # Load plugins and examples
    base_paths = {
        "Plugin": os.path.join(os.path.dirname(__file__), "../jax2onnx/plugins"),
        "Example": os.path.join(os.path.dirname(__file__), "../jax2onnx/examples"),
    }

    def load_tests_from_directory(base_path: str, source_name: str) -> None:
        """Helper function to load test cases from plugins/examples directories."""
        for dirpath, _, filenames in os.walk(base_path):
            for filename in filenames:
                if filename.endswith(".py") and filename != "__init__.py":
                    module_path = os.path.join(dirpath, filename)
                    module_name = module_path.replace("/", ".").replace(".py", "")
                    spec = importlib.util.spec_from_file_location(
                        module_name, module_path
                    )
                    if spec is None:  # Check if spec is None
                        raise ImportError(
                            f"Could not find module specification for {module_name} at {module_path}"
                        )
                    if spec.loader is None:  # Check if spec.loader is None
                        raise ImportError(
                            f"Could not find loader for {module_name} at {module_path}"
                        )
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    if hasattr(module, "get_test_params"):
                        test_params = module.get_test_params()
                        if isinstance(test_params, list):
                            for entry in test_params:
                                if "testcases" in entry:
                                    for test in entry["testcases"]:
                                        test["source"] = (
                                            source_name  # Track whether it's from plugins/examples
                                        )
                                        test["jax_component"] = entry.get(
                                            "jax_component", entry.get("component")
                                        )
                                        params.append(test)

    # ✅ Load tests from plugins and examples
    for source, path in base_paths.items():
        load_tests_from_directory(path, source)

    # Add parameter combinations and filter to only one testcase: "conv_3x3_1_internal_True_dynamic_False"
    new_params = []
    for param in params:
        # Change filter to check if "conv_3x3_1" is in the testcase name
        # if not any(keyword in param.get("testcase", "") for keyword in
        #            [ "gather" ]):
        #     continue

        # Check if we should skip generating dynamic batch dim testcases
        skip_dynamic = param.get("generate_derived_batch_dim_testcases") is False

        # Generate combinations based on flags
        for internal in [True, False]:
            for dynamic in [False, True]:
                # Skip dynamic batch dimension cases if specified
                if skip_dynamic and dynamic:
                    continue

                new_param = param.copy()
                new_param["internal_shape_info"] = internal
                new_param["dynamic_batch_dim"] = dynamic
                new_param["testcase"] = (
                    f"{param['testcase']}_{'1' if internal else '0'}{'1' if dynamic else '0'}"
                )

                if dynamic:
                    if "batch_input_shapes" in param:
                        new_param["input_shapes"] = param["batch_input_shapes"]
                    else:
                        new_param["input_shapes"] = [
                            ["B"] + list(shape)[1:]
                            for shape in new_param["input_shapes"]
                        ]

                new_params.append(
                    pytest.param(
                        new_param, id=f"{new_param['testcase']} ({new_param['source']})"
                    )
                )
    return new_params


@pytest.mark.parametrize("test_params", load_test_params())
def test_onnx_export(test_params: dict) -> None:
    """Test the ONNX export functionality."""
    component = test_params.get("component", None)

    if hasattr(component, "eval"):
        component.eval()

    input_shapes = test_params["input_shapes"]  # Note the plural!
    params = test_params.get("params", {})  # Get params from the test case
    internal_shape_info = test_params.get("internal_shape_info", True)

    seed = 0
    rng = jax.random.PRNGKey(seed)

    # Export the jax_model to ONNX
    onnx_model_file_name = f"{test_params['testcase']}.onnx"
    model_path = f"docs/onnx/{onnx_model_file_name}"
    os.makedirs("docs/onnx", exist_ok=True)

    z = to_onnx(
        onnx_model_file_name,
        component,
        input_shapes,
        output_path=model_path,
        params=params,
        internal_shape_info=internal_shape_info,
    )

    # Load the ONNX jax_model
    ort_session = ort.InferenceSession(model_path)

    # Process all shapes to replace 'B' with concrete value, regardless of dynamic_batch_dim flag
    processed_input_shapes = []
    for shape in input_shapes:
        # Check if shape contains 'B' and replace it with 2
        if any(isinstance(dim, str) and dim == "B" for dim in shape):
            processed_shape = [2 if dim == "B" else dim for dim in shape]
            processed_input_shapes.append(processed_shape)
        else:
            processed_input_shapes.append(shape)

    # Generate JAX inputs using processed shapes
    inputs = [jax.random.normal(rng, shape) for shape in processed_input_shapes]

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
