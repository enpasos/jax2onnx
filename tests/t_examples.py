# tests/test_examples.py
import os
import importlib.util
import pytest
import jax
import numpy as np
from jax2onnx import save_onnx, allclose
from typing import Any, Dict, List, Tuple


def load_example_metadata() -> list:
    """Walk through the examples directory and load metadata from each example.

    Each example module should define a get_metadata() function that returns
    a dictionary (or list of dictionaries) with a "testcases" key.
    """
    metadata_list = []
    examples_dir = os.path.join(os.path.dirname(__file__), "../jax2onnx/examples")
    for root, _, files in os.walk(examples_dir):
        for file in files:
            if file.endswith(".py") and file not in ["__init__.py"]:
                module_path = os.path.join(root, file)
                module_name = module_path.replace(os.sep, ".").replace(".py", "")
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, "get_metadata"):
                    md = module.get_metadata()
                    if not isinstance(md, list):
                        md = [md]
                    for entry in md:
                        if "testcases" in entry and isinstance(
                            entry["testcases"], list
                        ):
                            for testcase in entry["testcases"]:
                                if isinstance(testcase, dict):
                                    testcase["source"] = module_name
                                    testcase["context"] = entry.get(
                                        "context", "default"
                                    )
                                    testcase["component"] = entry.get(
                                        "component", "default"
                                    )
                                    metadata_list.append(testcase)
    return metadata_list


def generate_test_params(metadata_entry: dict) -> list:
    """
    For each metadata entry, if any input_shape tuple contains the symbolic 'B',
    generate two variants:
      - One keeping the symbolic 'B' (suffix "_dynamic")
      - One where 'B' is replaced with a concrete value (here 3, suffix "_concrete")
    Otherwise, return the original entry.
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


def get_tests_by_context_and_component() -> Dict:
    grouping: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for param in load_all_example_test_params():
        context = param.get("context", "default")
        component = param.get("component", "default")
        grouping.setdefault((context, component), []).append(param)
    return grouping


def make_test_function(tp):
    def test_func(self):
        callable_fn = tp.get("callable")
        if callable_fn is None:
            pytest.skip("No callable component found in metadata.")
        if hasattr(callable_fn, "eval"):
            callable_fn.eval()
        input_shapes = tp.get("input_shapes", [])
        rng = jax.random.PRNGKey(1001)
        testcase_name = tp.get("testcase", "unknown")
        context_path = tp.get("context", "default").split(".")
        model_path = os.path.join(
            "docs", "onnx", *context_path, f"{testcase_name}.onnx"
        )
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        save_onnx(
            callable_fn, input_shapes, model_path, include_intermediate_shapes=True
        )
        xs = [jax.random.normal(rng, tuple(shape)) for shape in input_shapes]
        assert allclose(callable_fn, model_path, *xs)
        print(f"Test for {testcase_name} passed!")

    return test_func


# Generate test classes grouped by context and component.
tests_by_group = get_tests_by_context_and_component()
for (context, component), tests in tests_by_group.items():
    class_name = f"Test_{context.replace('.', '_')}_{component}"
    attrs = {}
    for tp in tests:
        test_name = f"test_{tp.get('testcase', 'unknown')}"
        test_name = test_name.replace(" ", "_").replace(".", "_")
        attrs[test_name] = make_test_function(tp)
    globals()[class_name] = type(class_name, (object,), attrs)
