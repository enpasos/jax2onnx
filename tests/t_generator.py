# file: tests/t_generator.py

import os
import shutil
import importlib.util
from typing import Any, Dict, List, Tuple
import jax
from jax2onnx import save_onnx, allclose

# Define base directories.
TESTS_DIR = os.path.dirname(__file__)
# PLUGINS_DIR = os.path.join(TESTS_DIR, "../jax2onnx/converter/plugins")
# EXAMPLES_DIR = os.path.join(TESTS_DIR, "../jax2onnx/examples")
# GENERATED_PLUGINS_TESTS_DIR = os.path.join(TESTS_DIR, "primitives")
# GENERATED_EXAMPLES_TESTS_DIR = os.path.join(TESTS_DIR, "examples")

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


def load_metadata_only_from_dir(
    directory: str, exclude_files=None
) -> List[Dict[str, Any]]:
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
                        # if isinstance(md, list) then add all elements in the list to metadata_list
                        # else add the single element to metadata_list
                        md = md if isinstance(md, list) else [md]
                        metadata_list.extend(md)

    return metadata_list


def extract_from_metadata(mds) -> List[Dict[str, Any]]:
    metadata_list = []
    for entry in mds:
        testcases = entry.get("testcases", [])
        if "component" in entry and entry["component"] == "MLP":
            print("MLP")
        for testcase in testcases:
            # testcase["source"] = module_name
            testcase["context"] = entry.get("context", "default")
            # if exist entry.jaxpr_primitive then use it as component
            # if testcase["component"] is MLP then print MLP

            if "jaxpr_primitive" in entry:
                testcase["component"] = entry["jaxpr_primitive"]
            else:
                testcase["component"] = entry.get("component", "default")

            testcase["jax_doc"] = entry.get("jax_doc", "")
            testcase["onnx"] = entry.get("onnx", "")
            testcase["source"] = entry.get("source", "")
            testcase["since"] = entry.get("since", "")
            testcase["description"] = entry.get("description", "")
            testcase["children"] = entry.get("children", [])
            metadata_list.append(testcase)
    return metadata_list


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
                                testcase["jax_doc"] = entry.get("jax_doc", "")
                                testcase["onnx"] = entry.get("onnx", "")
                                testcase["since"] = entry.get("since", "")
                                testcase["description"] = entry.get("description", "")
                                testcase["children"] = entry.get("children", [])
                                metadata_list.append(testcase)
    return metadata_list


NEW_PLUGINS_DIR = os.path.join(TESTS_DIR, "../jax2onnx/plugins")

from jax2onnx.plugin_system import (
    PLUGIN_REGISTRY,
    import_all_plugins,
)  # Import new registry


def load_plugin_metadata() -> List[Dict[str, Any]]:
    """Loads metadata from both the old and new plugin systems."""
    # old_mds = load_metadata_only_from_dir(
    #     PLUGINS_DIR,
    #     [
    #         "__init__.py",
    #         "plugin_interface.py",
    #         "plugin_registry.py",
    #         "plugin_registry_static.py",
    #     ],
    # )

    # Extract metadata from the new plugin system
    import_all_plugins()  # Automatically imports everything once

    new_md = []
    for name, plugin in PLUGIN_REGISTRY.items():
        if hasattr(plugin, "metadata"):
            plugin.metadata["jaxpr_primitive"] = name
            new_md.append(plugin.metadata)
            # metadata_entry = plugin.metadata.copy()
            # metadata_entry["component"] = name
            # metadata_entry["context"] = metadata_entry.get("context", "plugins.nnx")
            # new_plugins.append(metadata_entry)

    return extract_from_metadata(new_md)  # + new_md  # Merge old and new metadata


EXAMPLES_DIR = os.path.join(TESTS_DIR, "../jax2onnx/examples")


def load_example_metadata() -> List[Dict[str, Any]]:
    """Loads metadata from both the old and new example systems."""
    old_examples = load_metadata_from_dir(EXAMPLES_DIR)  # Load old-style metadata

    # Extract metadata from the new plugin system (if any examples exist there)
    new_examples = []
    for name, plugin in PLUGIN_REGISTRY.items():
        if plugin.metadata.get("context", "").startswith("examples."):
            metadata_entry = plugin.metadata.copy()
            metadata_entry["component"] = name  # Ensure component name
            metadata_entry["context"] = metadata_entry.get("context", "examples")
            new_examples.append(metadata_entry)

    return old_examples + new_examples  # Merge old and new metadata


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
_GLOBAL_EXAMPLE_GROUPING = None


def get_plugin_grouping() -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    global _GLOBAL_PLUGIN_GROUPING
    if _GLOBAL_PLUGIN_GROUPING is None:
        plugin_params = []
        for md in load_plugin_metadata():
            plugin_params.extend(generate_test_params(md))
        _GLOBAL_PLUGIN_GROUPING = organize_tests_by_context_and_component_from_params(
            plugin_params
        )
    return _GLOBAL_PLUGIN_GROUPING


def get_example_grouping() -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    global _GLOBAL_EXAMPLE_GROUPING
    if _GLOBAL_EXAMPLE_GROUPING is None:
        example_params = []
        for md in load_example_metadata():
            example_params.extend(generate_test_params(md))
        _GLOBAL_EXAMPLE_GROUPING = organize_tests_by_context_and_component_from_params(
            example_params
        )
    return _GLOBAL_EXAMPLE_GROUPING


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
    # Select grouping based on context prefix.
    if component == "linear_general":
        print("linear_general")

    if context.startswith("primitives"):
        grouping = get_plugin_grouping()
    else:
        grouping = get_example_grouping()
    testcases = grouping.get((context, component), [])
    class_name = f"Test_{context.replace('.', '_')}_{component}"
    attrs = {}
    for tp in testcases:
        test_name = f"test_{tp['testcase'].replace('.', '_').replace(' ', '_')}"
        attrs[test_name] = make_test_function(tp)
    namespace[class_name] = type(class_name, (object,), attrs)


# --- Minimal Test File Generation ---


def create_minimal_test_file(directory: str, context: str, components: List[str]):

    folder = context.split(".")[0]
    # add folder to directory
    directory = os.path.join(directory, folder)
    os.makedirs(directory, exist_ok=True)
    # Create an empty __init__.py so that the directory is a package.
    init_path = os.path.join(directory, "__init__.py")

    filename = os.path.join(directory, f"test_{context.split('.')[-1]}.py")
    with open(filename, "w") as f:
        f.write("from tests.t_generator import generate_test_class\n\n")
        for component in components:
            f.write(
                f"generate_test_class({repr(context)}, {repr(component)}, globals())\n"
            )
    print(f"Generated minimal test file: {filename}")


# --- Main Generation ---


def generate_all_tests():
    clean_generated_test_dirs()
    plugin_grouping = get_plugin_grouping()
    # example_grouping = get_example_grouping()

    # For plugins: group by context.
    plugin_context_components: Dict[str, List[str]] = {}
    for context, component in plugin_grouping.keys():
        plugin_context_components.setdefault(context, []).append(component)
    for context, components in plugin_context_components.items():
        create_minimal_test_file(TESTS_DIR, context, components)

    # For examples: group by context.
    # example_context_components: Dict[str, List[str]] = {}
    # for context, component in example_grouping.keys():
    #     example_context_components.setdefault(context, []).append(component)
    # for context, components in example_context_components.items():
    #     create_minimal_test_file(GENERATED_EXAMPLES_TESTS_DIR, context, components)


if __name__ == "__main__":
    generate_all_tests()
