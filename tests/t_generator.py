# file: tests/t_generator.py

import os
import shutil
import importlib.util
from typing import Any, Dict, List, Tuple
import jax
from jax2onnx import save_onnx, allclose

# Define base directories.
TESTS_DIR = os.path.dirname(__file__)
PLUGINS_DIR = os.path.join(TESTS_DIR, "../jax2onnx/converter/plugins")
EXAMPLES_DIR = os.path.join(TESTS_DIR, "../jax2onnx/examples")
GENERATED_PLUGINS_TESTS_DIR = os.path.join(TESTS_DIR, "plugins")
GENERATED_EXAMPLES_TESTS_DIR = os.path.join(TESTS_DIR, "examples")

# --- Cleaning and Setup ---


def clean_generated_dir(directory: str):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)
    # Create an empty __init__.py so that the directory is a package.
    init_path = os.path.join(directory, "__init__.py")
    with open(init_path, "w"):
        pass


def clean_generated_test_dirs():
    clean_generated_dir(GENERATED_PLUGINS_TESTS_DIR)
    clean_generated_dir(GENERATED_EXAMPLES_TESTS_DIR)


# --- Metadata Loading ---


def load_metadata_from_dir(directory: str, exclude_files=None) -> List[Dict[str, Any]]:
    exclude_files = exclude_files or ["__init__.py"]
    metadata_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py") and file not in exclude_files:
                module_path = os.path.join(root, file)
                module_name = module_path.replace(os.sep, ".").replace(".py", "")
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    if hasattr(module, "get_metadata"):
                        md = module.get_metadata()
                        md = md if isinstance(md, list) else [md]
                        for entry in md:
                            testcases = entry.get("testcases", [])
                            for testcase in testcases:
                                testcase["source"] = module_name
                                testcase["context"] = entry.get("context", "default")
                                testcase["component"] = entry.get(
                                    "component", file[:-3]
                                )
                                metadata_list.append(testcase)
    return metadata_list


def load_plugin_metadata() -> List[Dict[str, Any]]:
    return load_metadata_from_dir(
        PLUGINS_DIR,
        [
            "__init__.py",
            "plugin_interface.py",
            "plugin_registry.py",
            "plugin_registry_static.py",
        ],
    )


def load_example_metadata() -> List[Dict[str, Any]]:
    return load_metadata_from_dir(EXAMPLES_DIR)


# --- Test Param Generation ---


def generate_test_params(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    input_shapes = entry.get("input_shapes", [])
    has_symbolic = any("B" in shape for shape in input_shapes)
    if has_symbolic:
        dynamic = entry.copy()
        dynamic["testcase"] += "_dynamic"
        concrete = entry.copy()
        concrete["testcase"] += "_concrete"
        concrete["input_shapes"] = [
            tuple(3 if dim == "B" else dim for dim in shape) for shape in input_shapes
        ]
        return [dynamic, concrete]
    return [entry]


def load_all_test_params() -> List[Dict[str, Any]]:
    metadata = load_plugin_metadata() + load_example_metadata()
    params = []
    for md in metadata:
        params.extend(generate_test_params(md))
    return params


# --- Organizing Tests ---


def organize_tests_by_context_and_component():
    plugins_group: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    examples_group: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for param in load_all_test_params():
        source = param["source"]
        context = param.get("context", "default")
        component = param.get("component", "default")
        if "converter.plugins" in source:
            plugins_group.setdefault((context, component), []).append(param)
        else:
            examples_group.setdefault((context, component), []).append(param)
    return plugins_group, examples_group


# --- Test Function Creation ---


def make_test_function(tp: Dict[str, Any]):
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

        if hasattr(callable_obj, "eval"):
            callable_obj.eval()

        save_onnx(
            callable_obj, input_shapes, model_path, include_intermediate_shapes=True
        )

        if any("B" in shape for shape in input_shapes):
            for B in [2, 3]:
                processed_shapes = [
                    tuple(B if dim == "B" else dim for dim in s) for s in input_shapes
                ]
                xs = [jax.random.normal(rng, shape) for shape in processed_shapes]
                assert allclose(callable_obj, model_path, *xs)
        else:
            xs = [jax.random.normal(rng, shape) for shape in input_shapes]
            assert allclose(callable_obj, model_path, *xs)

    return test_func


# --- Test Class Registration ---


def generate_test_class(context: str, component: str, namespace: dict):
    # Retrieve both plugins and examples groupings.
    plugins_group, examples_group = organize_tests_by_context_and_component()
    testcases = plugins_group.get((context, component))
    if testcases is None:
        testcases = examples_group.get((context, component), [])
    class_name = f"Test_{context.replace('.', '_')}_{component}"
    attrs = {}
    for tp in testcases:
        test_name = f"test_{tp['testcase'].replace('.', '_').replace(' ', '_')}"
        attrs[test_name] = make_test_function(tp)
    namespace[class_name] = type(class_name, (object,), attrs)


# --- Minimal Test File Generation ---


def create_minimal_test_file(directory: str, context: str, components: List[str]):
    filename = os.path.join(directory, f"test_{context.split('.')[-1]}.py")
    with open(filename, "w") as f:
        f.write("from tests.t_generator import generate_test_class\n\n")
        # f.write("def generate_tests():\n")
        for component in components:
            f.write(
                f"generate_test_class({repr(context)}, {repr(component)}, globals())\n"
            )
        # f.write("\ngenerate_tests()\n")
    print(f"Generated minimal test file: {filename}")


# --- Main Generation ---


def generate_all_tests():
    clean_generated_test_dirs()
    plugins_group, examples_group = organize_tests_by_context_and_component()

    # Generate plugin test files.
    plugin_context_components: Dict[str, List[str]] = {}
    for context, component in plugins_group.keys():
        plugin_context_components.setdefault(context, []).append(component)
    for context, components in plugin_context_components.items():
        create_minimal_test_file(GENERATED_PLUGINS_TESTS_DIR, context, components)

    # Generate example test files.
    example_context_components: Dict[str, List[str]] = {}
    for context, component in examples_group.keys():
        example_context_components.setdefault(context, []).append(component)
    for context, components in example_context_components.items():
        create_minimal_test_file(GENERATED_EXAMPLES_TESTS_DIR, context, components)


if __name__ == "__main__":
    generate_all_tests()
