# file: tests/test_plugins.py
# This file is a pytest test suite that dynamically generates test classes
# based on the metadata of the plugins. The metadata is loaded from the
# plugins and then used to generate test cases. The test cases are then run
# using the allclose function from the jax2onnx package. The test cases are
# grouped by context and component, and a test class is created for each
# component. The test class is then dynamically created with test functions
# for each test case. The test functions use the allclose function to compare
# the outputs of the JAX model and the ONNX model. The test functions are then
# run using pytest. The test results are then printed to the console.


import os
import importlib.util
import pytest
import jax
from jax2onnx import save_onnx, allclose


def get_plugin_from_source(source: str) -> str:
    return source.split(".")[-1]


def load_plugin_metadata() -> list:
    metadata_list = []
    plugins_dir = os.path.join(
        os.path.dirname(__file__), "../jax2onnx/converter/plugins"
    )
    for root, _, files in os.walk(plugins_dir):
        for file in files:
            if file.endswith(".py") and file not in [
                "__init__.py",
                "plugin_interface.py",
                "plugin_registry.py",
                "plugin_registry_static.py",
            ]:
                module_path = os.path.join(root, file)
                module_name = module_path.replace(os.sep, ".").replace(".py", "")
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    if hasattr(module, "get_metadata"):
                        md = module.get_metadata()
                        if isinstance(md, dict) and "testcases" in md:
                            for testcase in md["testcases"]:
                                testcase["source"] = module_name
                                testcase["context"] = md.get("context", "default")
                                metadata_list.append(testcase)
    return metadata_list


def generate_test_params(metadata_entry: dict) -> list:
    params = []
    base = metadata_entry.copy()
    input_shapes = base.get("input_shapes", [])
    has_symbolic = any(
        isinstance(dim, str) and dim == "B" for shape in input_shapes for dim in shape
    )
    if has_symbolic:
        dynamic = base.copy()
        dynamic["testcase"] += "_dynamic"
        concrete = base.copy()
        concrete["input_shapes"] = [
            tuple(3 if (isinstance(dim, str) and dim == "B") else dim for dim in shape)
            for shape in input_shapes
        ]
        concrete["testcase"] += "_concrete"
        params.extend([dynamic, concrete])
    else:
        params.append(base)
    return params


def load_all_test_params() -> list:
    all_params = []
    for md in load_plugin_metadata():
        all_params.extend(generate_test_params(md))
    return all_params


def get_tests_by_context_and_plugin():
    grouping = {}
    for param in load_all_test_params():
        context = param.get("context", "default")
        plugin = get_plugin_from_source(param.get("source", "default"))
        grouping.setdefault((context, plugin), []).append(param)
    return grouping


def make_test_function(tp):
    def test_func(self):
        callable_obj = tp.get("callable")
        if callable_obj is None:
            pytest.skip("No callable component defined")
        if hasattr(callable_obj, "eval"):
            callable_obj.eval()
        input_shapes = tp.get("input_shapes", [])
        rng = jax.random.PRNGKey(1001)
        testcase_name = tp.get("testcase", "unknown")
        context_path = tp.get("context", "default").split(".")
        model_path = os.path.join(
            "docs", "onnx", *context_path, f"{testcase_name}.onnx"
        )
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        save_onnx(
            callable_obj, input_shapes, model_path, include_intermediate_shapes=True
        )

        if any(
            isinstance(dim, str) and dim == "B"
            for shape in input_shapes
            for dim in shape
        ):
            for concrete_value in [2, 3]:  # Example concrete values
                processed_input_shapes = [
                    tuple(
                        concrete_value if (isinstance(dim, str) and dim == "B") else dim
                        for dim in shape
                    )
                    for shape in input_shapes
                ]
                xs = [
                    jax.random.normal(rng, tuple(shape))
                    for shape in processed_input_shapes
                ]
                assert allclose(callable_obj, model_path, *xs)
        else:
            xs = [jax.random.normal(rng, tuple(shape)) for shape in input_shapes]
            assert allclose(callable_obj, model_path, *xs)

    return test_func


# Generate test classes
for (context, plugin), tests in get_tests_by_context_and_plugin().items():
    class_name = f"Test_{context.replace('.', '_')}_{plugin}"
    attrs = {}
    for tp in tests:
        test_name = f"test_{tp['testcase'].replace('.', '_').replace(' ', '_')}"
        attrs[test_name] = make_test_function(tp)
    test_class = type(class_name, (object,), attrs)  # Inherit directly from object
    globals()[class_name] = test_class
