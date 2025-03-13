# tests/test_examples.py
import os
import importlib.util
import pytest
import jax
import numpy as np
from jax2onnx import save_onnx, allclose
from typing import Any, Dict, List


def load_example_metadata() -> list:
    """Walk through the examples directory and load metadata from each example.

    Each example should define a get_metadata() function that returns a dictionary
    with a "testcases" key (a list of dicts), each representing one testcase.
    """
    metadata_list = []
    examples_dir = os.path.join(os.path.dirname(__file__), "../jax2onnx/examples")
    for root, _, files in os.walk(examples_dir):
        for file in files:
            # Skip __init__.py
            if file.endswith(".py") and file not in ["__init__.py"]:
                module_path = os.path.join(root, file)
                # Create a module name based on the file path.
                module_name = module_path.replace(os.sep, ".").replace(".py", "")
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                if spec is None or spec.loader is None:
                    continue  # Could not load this module, skip it.
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, "get_metadata"):
                    md = module.get_metadata()
                    md = [md] if not isinstance(md, list) else md
                    for entry in md:
                        if "testcases" in entry and isinstance(
                            entry["testcases"], list
                        ):
                            for testcase in entry["testcases"]:
                                if isinstance(testcase, dict):
                                    testcase["source"] = module_name
                                    # Pass the context along (e.g. "plugins.nnx")
                                    testcase["context"] = entry.get(
                                        "context", "default"
                                    )
                                    metadata_list.append(testcase)
    return metadata_list


def generate_test_params(metadata_entry: dict) -> list:
    """
    Given a metadata entry, inspect its "input_shapes" attribute (expected to be a list
    of tuples). If any tuple contains the string 'B', generate two variants:
      - One where input_shapes is unchanged, with '_dynamic' added to the testcase name.
      - One where each occurrence of 'B' is replaced with the concrete value 3, with '_concrete' added.
    If no symbolic 'B' is found, return the original testcase.
    """
    params = []
    base = metadata_entry.copy()
    input_shapes = base.get("input_shapes", [])
    if isinstance(input_shapes, list) and all(
        isinstance(t, (tuple, list)) for t in input_shapes
    ):
        has_symbolic = any(
            any(isinstance(dim, str) and dim == "B" for dim in tup)
            for tup in input_shapes
        )
        if has_symbolic:
            dynamic_variant = base.copy()
            dynamic_variant["testcase"] = f"{base.get('testcase', 'unknown')}_dynamic"
            concrete_variant = base.copy()
            concrete_variant["input_shapes"] = [
                tuple(
                    3 if (isinstance(dim, str) and dim == "B") else dim for dim in tup
                )
                for tup in input_shapes
            ]
            concrete_variant["testcase"] = f"{base.get('testcase', 'unknown')}_concrete"
            params.extend([dynamic_variant, concrete_variant])
        else:
            params.append(base)
    else:
        params.append(base)
    return params


def load_all_example_test_params() -> List[Dict[str, Any]]:
    all_params = []
    for md in load_example_metadata():
        for param in generate_test_params(md):
            all_params.append(param)
    return all_params


def get_tests_by_context():
    tests_by_context = {}
    for param in load_all_example_test_params():
        context = param.get("context", "default")
        tests_by_context.setdefault(context, []).append(param)
    return tests_by_context


# Dynamically create a test class for each context.
for context, tests in get_tests_by_context().items():
    # For a cleaner name, use just the last part of the context (e.g. "example" or "mycontext")
    cls_name = f"Test_{context.split('.')[-1]}"
    attrs = {}

    # Create a test function for each test parameter.
    for test_param in tests:
        # Create a safe test name based on the testcase name.
        test_name = f"test_{test_param.get('testcase', 'unknown')}"
        test_name = test_name.replace(" ", "_").replace(".", "_")

        def make_test_function(tp):
            def test_func(self, tp=tp):
                callable_fn = tp.get("callable")
                if callable_fn is None:
                    pytest.skip("No callable component found in metadata.")
                if hasattr(callable_fn, "eval"):
                    callable_fn.eval()

                input_shapes = tp.get("input_shapes", [])
                testcase_name = tp.get("testcase", "unknown")

                # Determine the ONNX model file path.
                onnx_model_file = f"{testcase_name}.onnx"

                context_path = tp.get("context", "default").split(".")
                model_path = os.path.join(
                    "docs", "onnx", *context_path, onnx_model_file
                )
                os.makedirs(os.path.dirname(model_path), exist_ok=True)

                # Convert the JAX component to ONNX.
                save_onnx(
                    callable_fn,
                    input_shapes,
                    model_path,
                    include_intermediate_shapes=True,
                )

                seed = 1001
                rng = jax.random.PRNGKey(seed)

                # Execute test using input data.
                if testcase_name.endswith("_dynamic"):
                    for concrete_value in [2, 3]:
                        processed_input_shapes = [
                            [
                                (
                                    concrete_value
                                    if isinstance(dim, str) and dim == "B"
                                    else dim
                                )
                                for dim in shape
                            ]
                            for shape in input_shapes
                        ]
                        xs = [
                            jax.random.normal(rng, tuple(shape))
                            for shape in processed_input_shapes
                        ]
                        np.testing.assert_(allclose(callable_fn, model_path, *xs))
                else:
                    xs = [jax.random.normal(rng, shape) for shape in input_shapes]
                    np.testing.assert_(allclose(callable_fn, model_path, *xs))
                print(f"Test for {testcase_name} passed!")

            return test_func

        attrs[test_name] = make_test_function(test_param)

    globals()[cls_name] = type(cls_name, (object,), attrs)
