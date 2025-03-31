# file: tests/t_generator.py

import os
import shutil
from typing import Any, Dict, List, Tuple
import jax
import jax.numpy as jnp
import onnx  # <<< Add onnx import

# === Add imports for to_onnx ===
from jax2onnx import allclose

# Assuming to_onnx is accessible via jax2onnx or directly from user_interface
# If needed, adjust the import path, e.g.:
from jax2onnx.converter.user_interface import to_onnx, save_onnx

# ===============================
from jax2onnx.plugin_system import (
    PLUGIN_REGISTRY,
    import_all_plugins,
)

# Define base directories.
TESTS_DIR = os.path.dirname(__file__)
PLUGINS_DIR = os.path.join(TESTS_DIR, "../jax2onnx/plugins")


# --- Cleaning and Setup (no changes needed) ---
def clean_generated_dir(directory: str):
    # ... (keep existing code) ...
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)
    # Create an empty __init__.py so that the directory is a package.
    init_path = os.path.join(directory, "__init__.py")
    with open(init_path, "w"):
        pass


def clean_generated_test_dirs():
    # ... (keep existing code) ...
    clean_generated_dir(os.path.join(TESTS_DIR, "primitives"))
    clean_generated_dir(os.path.join(TESTS_DIR, "examples"))


# --- Metadata Loading (no changes needed) ---
def extract_from_metadata(mds) -> List[Dict[str, Any]]:
    # ... (keep existing code) ...
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
    # ... (keep existing code) ...
    import_all_plugins()
    return [
        {**plugin.metadata, "jaxpr_primitive": name}
        for name, plugin in PLUGIN_REGISTRY.items()
        if hasattr(plugin, "metadata")
    ]


def load_plugin_metadata() -> List[Dict[str, Any]]:
    # ... (keep existing code) ...
    md = load_metadata_from_plugins()
    return extract_from_metadata(md)


# --- Test Param Generation (no changes needed) ---
def generate_test_params(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    # ... (keep existing code) ...
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


# --- Organizing Tests (no changes needed) ---
def organize_tests_by_context_and_component_from_params(
    params: List[Dict[str, Any]],
) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    # ... (keep existing code) ...
    grouping: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for param in params:
        context = param.get("context", "default")
        component = param.get("component", "default")
        grouping.setdefault((context, component), []).append(param)
    return grouping


_GLOBAL_PLUGIN_GROUPING = None


def get_plugin_grouping(reset=False) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    # ... (keep existing code) ...
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


# --- Test Function Creation (MODIFIED) ---
def make_test_function(tp: Dict[str, Any]):
    def test_func(self):
        callable_obj = tp["callable"]
        input_shapes = tp["input_shapes"]
        testcase_name = tp["testcase"]
        # === Get expected function count ===
        expected_num_funcs = tp.get("expected_number_of_function_instances")
        # ===================================
        rng = jax.random.PRNGKey(1001)
        context_path = tp["context"].split(".")
        # Use a consistent opset, e.g., 21
        opset_version = 21
        model_path = os.path.join(
            "docs", "onnx", *context_path, f"{testcase_name}.onnx"
        )
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        if hasattr(callable_obj, "eval"):
            callable_obj.eval()  # Set to eval mode if applicable

        # === Conditionally generate model and check function count ===
        if expected_num_funcs is not None:
            print(
                f"\nGenerating model for {testcase_name} (expecting {expected_num_funcs} functions)..."
            )
            # Generate the model object first
            onnx_model = to_onnx(
                callable_obj,
                input_shapes,
                model_name=testcase_name,  # Use testcase name for model name
                opset=opset_version,
            )
            # Assert the function count
            num_found_funcs = len(onnx_model.functions)
            assert (
                num_found_funcs == expected_num_funcs
            ), f"Test '{testcase_name}': Expected {expected_num_funcs} functions, found {num_found_funcs} in generated model."
            print(f"-> Found expected {num_found_funcs} functions.")
            # Save the model
            onnx.save_model(onnx_model, model_path)
            print(f"   Model saved to: {model_path}")
        else:
            # Original behavior if no expectation is set
            print(f"\nGenerating model for {testcase_name}...")
            save_onnx(
                callable_obj,
                input_shapes,
                model_path,
                opset=opset_version,
            )
            print(f"   Model saved to: {model_path}")
        # ===========================================================

        # --- Numerical Check ---
        def generate_inputs(shapes, B=None):
            # Generate dynamic inputs if 'B' is present
            actual_shapes = [
                (
                    tuple(B if dim == "B" else dim for dim in shape)
                    if B is not None
                    else shape
                )
                for shape in shapes
            ]
            return [
                jax.random.normal(
                    rng, shape=s, dtype=jnp.float32
                )  # Assuming float32 default
                for s in actual_shapes
            ]

        if any("B" in shape for shape in input_shapes):
            print("Running numerical checks for dynamic batch sizes [2, 3]...")
            for B in [2, 3]:
                print(f"  Batch size B={B}")
                xs = generate_inputs(input_shapes, B=B)
                assert allclose(
                    callable_obj, model_path, *xs
                ), f"Numerical check failed for B={B}"
                print(f"  Numerical check passed for B={B}.")
        else:
            print("Running numerical check for static shape...")
            xs = generate_inputs(input_shapes)
            assert allclose(
                callable_obj, model_path, *xs
            ), "Numerical check failed for static shape."
            print("  Numerical check passed for static shape.")
        # --- End Numerical Check ---

    # Set a descriptive name for the generated test function
    test_func.__name__ = f"test_{tp['testcase'].replace('.', '_').replace(' ', '_')}"
    return test_func


# --- Test Class Registration (no changes needed) ---
def generate_test_class(context: str, component: str, namespace: dict):
    # ... (keep existing code) ...
    testcases = get_plugin_grouping().get((context, component), [])
    class_name = f"Test_{component.replace('.', '_')}"  # Sanitize class name
    attrs = {
        f"test_{tp['testcase'].replace('.', '_').replace(' ', '_')}": make_test_function(
            tp
        )
        for tp in testcases
    }
    # Prevent creating empty test classes
    if attrs:
        namespace[class_name] = type(class_name, (object,), attrs)


# --- Minimal Test File Generation (no changes needed) ---
def create_minimal_test_file(directory: str, context: str, components: List[str]):
    # ... (keep existing code) ...
    folder = context.split(".")[0]
    directory = os.path.join(directory, folder)
    os.makedirs(directory, exist_ok=True)
    init_path = os.path.join(directory, "__init__.py")
    with open(init_path, "w"):
        pass  # Ensure __init__.py exists
    filename = os.path.join(directory, f"test_{context.split('.')[-1]}.py")
    with open(filename, "w") as f:
        f.write("from tests.t_generator import generate_test_class\n\n")
        for component in components:
            # Sanitize component name for class generation if needed
            component.replace(".", "_")
            f.write(
                f"generate_test_class({repr(context)}, {repr(component)}, globals())\n"
            )
    print(f"Generated minimal test file: {filename}")


def create_minimal_test_files(
    grouping: Dict[Tuple[str, str], List[Dict[str, Any]]], directory: str
):
    # ... (keep existing code) ...
    context_components: Dict[str, List[str]] = {}
    for context, component in grouping.keys():
        context_components.setdefault(context, []).append(component)
    for context, components in context_components.items():
        create_minimal_test_file(directory, context, components)


# --- Main Generation (no changes needed) ---
def generate_all_tests():
    # ... (keep existing code) ...
    clean_generated_test_dirs()
    create_minimal_test_files(get_plugin_grouping(True), TESTS_DIR)


if __name__ == "__main__":
    generate_all_tests()
