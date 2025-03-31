# file: tests/t_generator.py

import os
import shutil
from typing import Any, Dict, List, Tuple
import jax
import onnx  # <<< Add onnx import

# === Add imports for to_onnx and save_onnx ===
from jax2onnx import allclose

# Assuming to_onnx/save_onnx are accessible via jax2onnx or directly from user_interface
# Adjust import if necessary, e.g.:
from jax2onnx.converter.user_interface import to_onnx, save_onnx

# =================================================
from jax2onnx.plugin_system import (
    PLUGIN_REGISTRY,
    import_all_plugins,
)
import jax.numpy as jnp  # <<< Add jnp import for generate_inputs

# Define base directories.
TESTS_DIR = os.path.dirname(__file__)
PLUGINS_DIR = os.path.join(TESTS_DIR, "../jax2onnx/plugins")


# --- Cleaning and Setup ---
def clean_generated_dir(directory: str):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)
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
            # === Combine testcase with parent metadata ===
            combined_testcase = entry.copy()
            combined_testcase.pop("testcases", None)
            combined_testcase.update(testcase)

            # Ensure necessary keys exist
            combined_testcase.setdefault("context", "default")
            combined_testcase.setdefault("component", entry.get("component", "default"))
            combined_testcase.setdefault(
                "jaxpr_primitive", entry.get("jaxpr_primitive")
            )
            combined_testcase.setdefault("jax_doc", "")
            combined_testcase.setdefault("onnx", "")
            combined_testcase.setdefault("source", "")
            combined_testcase.setdefault("since", "")
            combined_testcase.setdefault("description", "")
            combined_testcase.setdefault("children", [])
            metadata_list.append(combined_testcase)
            # ============================================
    return metadata_list


def load_metadata_from_plugins() -> List[Dict[str, Any]]:
    import_all_plugins()
    return [
        {**plugin.metadata, "jaxpr_primitive": name}
        for name, plugin in PLUGIN_REGISTRY.items()
        if hasattr(plugin, "metadata")
    ]


def load_plugin_metadata() -> List[Dict[str, Any]]:
    md = load_metadata_from_plugins()
    return extract_from_metadata(md)


# --- Test Param Generation ---
def generate_test_params(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "callable" not in entry:
        return []  # Skip if no callable

    input_shapes = entry.get("input_shapes", [])
    # Ensure input_shapes is a list/tuple before checking for 'B'
    if isinstance(input_shapes, (list, tuple)) and any(
        "B" in shape for shape in input_shapes
    ):
        dynamic = entry.copy()
        dynamic["testcase"] += "_dynamic"
        concrete = entry.copy()
        concrete["input_shapes"] = [
            # Ensure shape is iterable before processing 'B'
            tuple(
                3 if dim == "B" else dim
                for dim in (s if isinstance(s, (list, tuple)) else (s,))
            )
            for s in input_shapes
        ]
        return [dynamic, concrete]
    return [entry]


# --- Organizing Tests ---
def organize_tests_by_context_and_component_from_params(
    params: List[Dict[str, Any]],
) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    grouping: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for param in params:
        context = param.get("context", "default")
        component_name = param.get("component", "default").replace(".", "_")  # Sanitize
        grouping.setdefault((context, component_name), []).append(param)
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


# --- Test Function Creation (MODIFIED) ---
def make_test_function(tp: Dict[str, Any]):
    test_case_name_safe = tp["testcase"].replace(".", "_").replace(" ", "_")
    func_name = f"test_{test_case_name_safe}"

    def test_func(self):
        callable_obj = tp["callable"]
        input_shapes = tp["input_shapes"]
        testcase_name = tp["testcase"]
        # === Get expected function count ===
        expected_num_funcs = tp.get("expected_number_of_function_instances")
        # ===================================
        rng = jax.random.PRNGKey(1001)
        context_path = tp["context"].split(".")
        opset_version = 21
        model_path = os.path.join(
            "docs", "onnx", *context_path, f"{testcase_name}.onnx"
        )
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        if hasattr(callable_obj, "eval"):
            callable_obj.eval()

        # === Conditionally generate model and check function count ===
        if expected_num_funcs is not None:
            print(
                f"\nGenerating model for {testcase_name} (expecting {expected_num_funcs} functions)..."
            )
            # Generate the model object first using to_onnx
            onnx_model = to_onnx(
                callable_obj,
                input_shapes,
                model_name=testcase_name,
                opset=opset_version,
            )
            # Assert the function count
            num_found_funcs = len({f.name for f in onnx_model.functions})

            assert (
                num_found_funcs == expected_num_funcs
            ), f"Test '{testcase_name}': Expected {expected_num_funcs} functions, found {num_found_funcs} in generated model."
            print(f"-> Found expected {num_found_funcs} functions.")
            # Save the model manually
            onnx.save_model(onnx_model, model_path)
            print(f"   Model saved to: {model_path}")
        else:
            # Original behavior if no expectation is set
            print(f"\nGenerating model for {testcase_name}...")
            save_onnx(
                callable_obj,
                input_shapes,
                model_path,
                model_name=testcase_name,
                opset=opset_version,
            )
            print(f"   Model saved to: {model_path}")
        # ===========================================================

        # --- Numerical Check ---
        def generate_inputs(shapes, B=None):
            actual_shapes = []
            # Ensure shapes is iterable
            if not isinstance(shapes, (list, tuple)):
                shapes = [shapes]

            for shape in shapes:
                current_shape = shape if isinstance(shape, (list, tuple)) else (shape,)
                actual_shapes.append(
                    tuple(B if dim == "B" else dim for dim in current_shape)
                    if B is not None
                    else current_shape
                )
            # Default dtype to float32
            return [
                jax.random.normal(rng, shape=s, dtype=jnp.float32)
                for s in actual_shapes
            ]

        if isinstance(input_shapes, (list, tuple)) and any(
            "B" in shape for shape in input_shapes
        ):
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

    test_func.__name__ = func_name
    return test_func


# --- Test Class Registration ---
def generate_test_class(context: str, component: str, namespace: dict):
    # Component name might contain '.', replace for valid class name
    class_name_suffix = component.replace(".", "_")
    class_name = f"Test_{class_name_suffix}"

    # Retrieve test cases using the original context and component key
    testcases = get_plugin_grouping().get(
        (context, component), []
    )  # Use original component name for lookup

    attrs = {}
    for tp in testcases:
        test_name_safe = tp["testcase"].replace(".", "_").replace(" ", "_")
        attrs[f"test_{test_name_safe}"] = make_test_function(tp)

    if attrs:
        namespace[class_name] = type(class_name, (object,), attrs)


# --- Minimal Test File Generation ---
def create_minimal_test_file(directory: str, context: str, components: List[str]):
    folder_parts = context.split(".")
    folder_name = folder_parts[0]
    sub_folder_name = folder_parts[-1] if len(folder_parts) > 1 else folder_name
    target_dir = os.path.join(directory, folder_name)
    os.makedirs(target_dir, exist_ok=True)
    init_path = os.path.join(target_dir, "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w"):
            pass

    filename = os.path.join(target_dir, f"test_{sub_folder_name}.py")
    mode = "a" if os.path.exists(filename) else "w"
    if mode == "w":
        print(f"Generating new test file: {filename}")
    else:
        print(f"Appending to existing test file: {filename}")

    with open(filename, mode) as f:
        if mode == "w":
            f.write("# Auto-generated by tests/t_generator.py\n")
            f.write("from tests.t_generator import generate_test_class\n\n")
        for component in components:
            f.write(f"# Tests for {context}.{component}\n")
            # Use original component name for generating class call
            f.write(
                f"generate_test_class({repr(context)}, {repr(component)}, globals())\n\n"
            )


def create_minimal_test_files(
    grouping: Dict[Tuple[str, str], List[Dict[str, Any]]], directory: str
):
    """Creates minimal test files, grouping components by context."""
    context_components: Dict[str, List[str]] = {}
    for context, component_key in grouping.keys():  # component_key might be sanitized
        # Need to map back or store original component name if sanitized key is used
        # Assuming the component name used in grouping keys is sufficient for now
        # Or retrieve original component name from grouping[key][0]['component']
        original_component_name = grouping[(context, component_key)][0]["component"]
        context_components.setdefault(context, []).append(original_component_name)

    unique_context_components = {
        ctx: sorted(list(set(comps))) for ctx, comps in context_components.items()
    }

    for context, components in unique_context_components.items():
        create_minimal_test_file(directory, context, components)


# --- Main Generation ---
def generate_all_tests():
    clean_generated_test_dirs()
    create_minimal_test_files(get_plugin_grouping(True), TESTS_DIR)


if __name__ == "__main__":
    generate_all_tests()
