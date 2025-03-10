# file: tests/test_examples.py

import os
import pytest
import jax
import numpy as np
import onnxruntime as ort
from jax2onnx import save_onnx, allclose


import os
import importlib.util
from typing import Any, Dict, List


def load_example_metadata() -> list:
    """Walk through the examples directory and load metadata from each example.

    Each example should define a get_metadata() function that returns a dictionary
    with a "testcases" key (a list of dicts), each representing one testcase.
    """
    metadata_list = []
    plugins_dir = os.path.join(os.path.dirname(__file__), "../jax2onnx/examples")

    for root, _, files in os.walk(plugins_dir):
        for file in files:
            # Skip __init__.py and registry/interface files
            if file.endswith(".py") and file not in ["__init__.py"]:
                module_path = os.path.join(root, file)
                # Create a module name based on the file path
                module_name = module_path.replace(os.sep, ".").replace(".py", "")

                spec = importlib.util.spec_from_file_location(module_name, module_path)
                if spec is None or spec.loader is None:
                    continue  # Could not load this module, skip it
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                if hasattr(module, "get_metadata"):
                    md = module.get_metadata()
                    if isinstance(md, dict):
                        # If metadata contains a "testcases" key, flatten them.
                        if "testcases" in md and isinstance(md["testcases"], list):
                            for testcase in md["testcases"]:
                                if isinstance(testcase, dict):
                                    testcase["source"] = module_name
                                    metadata_list.append(testcase)
    return metadata_list


# def load_example_metadata() -> List[Dict[str, Any]]:
#     """
#     Walk through the jax2onnx/examples directory and load metadata from each example module.
#     Each module should define a get_test_params() function returning a list of metadata dicts.
#     """
#     metadata_list = []
#     examples_dir = os.path.join(os.path.dirname(__file__), "../jax2onnx/examples")
#     for root, _, files in os.walk(examples_dir):
#         for file in files:
#             if file.endswith(".py") and file != "__init__.py":
#                 module_path = os.path.join(root, file)
#                 module_name = module_path.replace(os.sep, ".").rstrip(".py")
#                 spec = importlib.util.spec_from_file_location(module_name, module_path)
#                 if spec is None or spec.loader is None:
#                     continue
#                 module = importlib.util.module_from_spec(spec)
#                 spec.loader.exec_module(module)
#                 if hasattr(module, "get_test_params"):
#                     md = module.get_test_params()
#                     if isinstance(md, list):
#                         for entry in md:
#                             entry["source"] = module_name
#                             metadata_list.append(entry)
#     return metadata_list


def generate_example_test_params(
    metadata_entry: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Given an example metadata entry (a dict), extract the individual testcases.
    If the entry has a "testcases" key, each of its elements becomes a test parameter,
    and we copy in (or inherit) the parent's "component" (if not already defined).
    """
    params = []
    if "testcases" in metadata_entry and isinstance(metadata_entry["testcases"], list):
        for tc in metadata_entry["testcases"]:
            new_tc = tc.copy()
            new_tc["source"] = metadata_entry.get("source", "example")
            # Inherit parent's "callable" if the test case doesn't provide one.
            if "callable" not in new_tc and "callable" in metadata_entry:
                new_tc["callable"] = metadata_entry["callable"]
            params.append(new_tc)
    else:
        params.append(metadata_entry)
    return params


def load_all_example_test_params() -> List[Dict[str, Any]]:
    all_params = []
    for md in load_example_metadata():
        # Flatten the metadata into variants based on input_shapes.
        for param in generate_test_params(md):
            # Wrap in pytest.param to create a unique id based on the testcase and source.
            all_params.append(
                pytest.param(
                    param,
                    id=f"{param.get('testcase', 'unknown')} ({param.get('source', 'plugin')})",
                )
            )
    return all_params


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

    # Verify input_shapes is a list of tuples (or lists)
    if isinstance(input_shapes, list) and all(
        isinstance(t, (tuple, list)) for t in input_shapes
    ):
        # Check if any tuple contains the string 'B'
        has_symbolic = any(
            any(isinstance(dim, str) and dim == "B" for dim in tup)
            for tup in input_shapes
        )
        if has_symbolic:
            # Create dynamic variant: keep input_shapes as is
            dynamic_variant = base.copy()
            dynamic_variant["testcase"] = f"{base.get('testcase', 'unknown')}_dynamic"

            # Create concrete variant: replace each 'B' with the int 3
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


@pytest.mark.parametrize("test_params", load_all_example_test_params())
def test_example_onnx_export(test_params: dict) -> None:
    """
    For each example test case (from jax2onnx/examples), convert the JAX component to ONNX
    and verify that its output matches the JAX output.
    """
    # The example metadata should provide a 'callable' key.
    callable_fn = test_params.get("callable")
    if callable_fn is None:
        pytest.skip("No callable component found in metadata.")

    # If the component has an eval() method, call it.
    if callable_fn is not None and hasattr(callable_fn, "eval"):
        callable_fn.eval()

    input_shapes = test_params.get("input_shapes", [])
    testcase_name = test_params.get("testcase", "unknown")

    # Build concrete input arguments: replace any dynamic marker 'B' with a concrete value (e.g., 3).
    example_args = []
    for shape in input_shapes:
        concrete_shape = tuple(
            3 if isinstance(dim, str) and dim == "B" else dim for dim in shape
        )
        example_args.append(jax.numpy.zeros(concrete_shape))

    onnx_model_file = f"{testcase_name}.onnx"
    model_path = os.path.join("docs", "onnx", onnx_model_file)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Convert the JAX component to ONNX.
    save_onnx(callable_fn, input_shapes, model_path, include_intermediate_shapes=True)

    seed = 1001
    rng = jax.random.PRNGKey(seed)

    if testcase_name.endswith("_dynamic"):
        # Process input shapes: replace any symbolic dims (like 'B') with a concrete value (e.g., 2)
        # for dummy input generation
        for concrete_value in [2, 3]:
            processed_input_shapes = []
            for shape in input_shapes:
                processed_shape = [
                    concrete_value if isinstance(dim, str) and dim == "B" else dim
                    for dim in shape
                ]
                processed_input_shapes.append(processed_shape)

            xs = [jax.random.normal(rng, shape) for shape in processed_input_shapes]
            np.testing.assert_(allclose(callable, model_path, *xs))

    else:
        xs = [jax.random.normal(rng, shape) for shape in input_shapes]
        np.testing.assert_(allclose(callable, model_path, *xs))

    print(f"Test for {testcase_name} passed!")
