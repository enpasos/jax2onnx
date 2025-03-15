# file: tests/t_generator.py

import os
import shutil
import importlib.util
import jax
from jax2onnx import save_onnx, allclose
import pytest

TESTS_DIR = os.path.dirname(__file__)
PLUGINS_DIR = os.path.join(TESTS_DIR, "../jax2onnx/converter/plugins")
GENERATED_TESTS_DIR = os.path.join(TESTS_DIR, "plugins")


def clean_generated_tests_dir():
    """Delete and recreate the plugins test directory."""
    if os.path.exists(GENERATED_TESTS_DIR):
        shutil.rmtree(GENERATED_TESTS_DIR)
    os.makedirs(GENERATED_TESTS_DIR, exist_ok=True)


def load_plugin_metadata():
    """Load metadata from plugin source files."""
    metadata = []
    for root, _, files in os.walk(PLUGINS_DIR):
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
                                metadata.append(testcase)
    return metadata


def generate_test_params(metadata_entry):
    """Generate dynamic/concrete test variants if needed."""
    params = []
    input_shapes = metadata_entry.get("input_shapes", [])
    symbolic = any("B" in shape for shape in input_shapes)
    if symbolic:
        dynamic = metadata_entry.copy()
        dynamic["testcase"] += "_dynamic"
        concrete = metadata_entry.copy()
        concrete["testcase"] += "_concrete"
        concrete["input_shapes"] = [
            tuple(3 if dim == "B" else dim for dim in shape) for shape in input_shapes
        ]
        params.extend([dynamic, concrete])
    else:
        params.append(metadata_entry)
    return params


def load_all_test_params():
    """Load and expand all plugin test cases."""
    params = []
    for md in load_plugin_metadata():
        params.extend(generate_test_params(md))
    return params


def organize_by_context_and_plugin():
    """Group tests by context and plugin."""
    grouping = {}
    for param in load_all_test_params():
        context = param.get("context", "default")
        plugin = param["source"].split(".")[-1]
        grouping.setdefault(context, {}).setdefault(plugin, []).append(param)
    return grouping


def make_test_function(tp):
    def test_func(self):
        callable_obj = tp["callable"]
        input_shapes = tp["input_shapes"]
        testcase_name = tp["testcase"]
        rng = jax.random.PRNGKey(1001)
        context_path = tp["context"].split(".")
        model_path = os.path.join(
            "docs", "onnx", *context_path, f"{testcase_name}.onnx"
        )
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        save_onnx(
            callable_obj, input_shapes, model_path, include_intermediate_shapes=True
        )

        if any("B" in shape for shape in input_shapes):
            for B in [2, 3]:
                processed_shapes = [
                    tuple(B if dim == "B" else dim for dim in shape)
                    for shape in input_shapes
                ]
                xs = [jax.random.normal(rng, shape) for shape in processed_shapes]
                assert allclose(callable_obj, model_path, *xs)
        else:
            xs = [jax.random.normal(rng, shape) for shape in input_shapes]
            assert allclose(callable_obj, model_path, *xs)

    return test_func


def generate_test(context: str, plugin: str, namespace: dict):
    """Generate and register test classes into the provided namespace."""
    all_tests = organize_by_context_and_plugin()
    testcases = all_tests.get(context, {}).get(plugin, [])

    class_name = f"Test_{context.replace('.', '_')}_{plugin}"
    attrs = {}
    for tp in testcases:
        test_name = f"test_{tp['testcase'].replace('.', '_').replace(' ', '_')}"
        attrs[test_name] = make_test_function(tp)

    test_class = type(class_name, (object,), attrs)
    namespace[class_name] = test_class  # Inject into caller's namespace


def create_minimal_test_file(context: str, plugins):
    """Generate minimal test file for each context."""
    filename = os.path.join(GENERATED_TESTS_DIR, f"test_{context.split('.')[-1]}.py")
    with open(filename, "w") as f:
        f.write("from tests.t_generator import generate_test\n\n")
        f.write("def generate_tests():\n")
        for plugin in plugins:
            f.write(f"    generate_test({repr(context)}, {repr(plugin)}, globals())\n")
        f.write("\ngenerate_tests()\n")
    print(f"Generated minimal test file: {filename}")


def generate_all_minimal_test_files():
    """Clean and generate minimal test files."""
    clean_generated_tests_dir()
    grouped_tests = organize_by_context_and_plugin()
    for context, plugins_dict in grouped_tests.items():
        create_minimal_test_file(context, plugins_dict.keys())


if __name__ == "__main__":
    generate_all_minimal_test_files()
