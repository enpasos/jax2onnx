# file: tests/t_generator.py

import os
import shutil
from typing import Any, Dict, List, Tuple
import jax
from jax2onnx import save_onnx, allclose
from jax2onnx.plugin_system import (
    PLUGIN_REGISTRY,
    import_all_plugins,
)

# Define base directories.
TESTS_DIR = os.path.dirname(__file__)
PLUGINS_DIR = os.path.join(TESTS_DIR, "../jax2onnx/plugins")


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
    clean_generated_dir(os.path.join(TESTS_DIR, "primitives"))
    clean_generated_dir(os.path.join(TESTS_DIR, "examples"))


# --- Metadata Loading ---


def extract_from_metadata(mds) -> List[Dict[str, Any]]:
    metadata_list = []
    for entry in mds:
        testcases = entry.get("testcases", [])
        for testcase in testcases:
            testcase["context"] = entry.get("context", "default")
            testcase["component"] = entry.get(
                "jaxpr_primitive", entry.get("component", "default")
            )
            testcase["jax_doc"] = entry.get("jax_doc", "")
            testcase["onnx"] = entry.get("onnx", "")
            testcase["source"] = entry.get("source", "")
            testcase["since"] = entry.get("since", "")
            testcase["description"] = entry.get("description", "")
            testcase["children"] = entry.get("children", [])
            metadata_list.append(testcase)
    return metadata_list


def load_metadata_from_plugins() -> List[Dict[str, Any]]:
    """Helper function to load metadata from the new plugin system."""
    import_all_plugins()  # Automatically imports everything once

    return [
        {**plugin.metadata, "jaxpr_primitive": name}
        for name, plugin in PLUGIN_REGISTRY.items()
        if hasattr(plugin, "metadata")
    ]


def load_plugin_metadata() -> List[Dict[str, Any]]:
    """Loads metadata from the new plugin system."""
    md = load_metadata_from_plugins()
    return extract_from_metadata(md)


# --- Test Param Generation ---


def generate_test_params(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    input_shapes = entry.get("input_shapes", [])
    if any("B" in shape for shape in input_shapes):
        dynamic = entry.copy()
        dynamic["testcase"] += "_dynamic"
        concrete = entry.copy()
        concrete["input_shapes"] = [
            tuple(3 if dim == "B" else dim for dim in shape) for shape in input_shapes
        ]
        return [dynamic, concrete]
    return [entry]


# --- Organizing Tests (and caching groupings) ---


def organize_tests_by_context_and_component_from_params(
    params: List[Dict[str, Any]],
) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    grouping: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for param in params:
        context = param.get("context", "default")
        component = param.get("component", "default")
        grouping.setdefault((context, component), []).append(param)
    return grouping


_GLOBAL_PLUGIN_GROUPING = None


def get_plugin_grouping(reset=False) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    global _GLOBAL_PLUGIN_GROUPING
    if reset:
        _GLOBAL_PLUGIN_GROUPING = None

    if _GLOBAL_PLUGIN_GROUPING is None:
        plugin_params = []
        for md in load_plugin_metadata():
            plugin_params.extend(generate_test_params(md))
        _GLOBAL_PLUGIN_GROUPING = organize_tests_by_context_and_component_from_params(
            plugin_params
        )
    return _GLOBAL_PLUGIN_GROUPING


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
            callable_obj,
            input_shapes,
            model_path,
            include_intermediate_shapes=True,
            opset=21,
        )

        def generate_inputs(shapes, B=None):
            return [
                jax.random.normal(rng, tuple(B if dim == "B" else dim for dim in shape))
                for shape in shapes
            ]

        if any("B" in shape for shape in input_shapes):
            for B in [2, 3]:
                xs = generate_inputs(input_shapes, B)
                assert allclose(callable_obj, model_path, *xs)
        else:
            xs = generate_inputs(input_shapes)
            assert allclose(callable_obj, model_path, *xs)

    return test_func


# --- Test Class Registration ---


def generate_test_class(context: str, component: str, namespace: dict):
    testcases = get_plugin_grouping().get((context, component), [])
    class_name = f"Test_{component}"
    attrs = {
        f"test_{tp['testcase'].replace('.', '_').replace(' ', '_')}": make_test_function(
            tp
        )
        for tp in testcases
    }
    namespace[class_name] = type(class_name, (object,), attrs)


# --- Minimal Test File Generation ---


def create_minimal_test_file(directory: str, context: str, components: List[str]):
    folder = context.split(".")[0]
    directory = os.path.join(directory, folder)
    os.makedirs(directory, exist_ok=True)

    init_path = os.path.join(directory, "__init__.py")
    with open(init_path, "w"):
        pass

    filename = os.path.join(directory, f"test_{context.split('.')[-1]}.py")
    with open(filename, "w") as f:
        f.write("from tests.t_generator import generate_test_class\n\n")
        for component in components:
            f.write(
                f"generate_test_class({repr(context)}, {repr(component)}, globals())\n"
            )
    print(f"Generated minimal test file: {filename}")


def create_minimal_test_files(
    grouping: Dict[Tuple[str, str], List[Dict[str, Any]]], directory: str
):
    """Helper function to create minimal test files based on the grouping."""
    context_components: Dict[str, List[str]] = {}
    for context, component in grouping.keys():
        context_components.setdefault(context, []).append(component)
    for context, components in context_components.items():
        create_minimal_test_file(directory, context, components)


# --- Main Generation ---


def generate_all_tests():
    clean_generated_test_dirs()
    create_minimal_test_files(get_plugin_grouping(True), TESTS_DIR)


if __name__ == "__main__":
    generate_all_tests()
