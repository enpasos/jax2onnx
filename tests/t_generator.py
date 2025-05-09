# file: tests/t_generator.py

import os
import shutil
from typing import Any, List, Dict, Tuple, Sequence, Union
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import logging

# from logging_config import configure_logging

from jax2onnx import allclose
from jax2onnx.converter.user_interface import to_onnx
from jax2onnx.plugin_system import (
    PLUGIN_REGISTRY,
    import_all_plugins,
)

# Define base directories.
TESTS_DIR = os.path.dirname(__file__)
PLUGINS_DIR = os.path.join(TESTS_DIR, "../jax2onnx/plugins")

logger = logging.getLogger("jax2onnx.tests.t_generator")


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
def extract_from_metadata(mds) -> list[dict[str, Any]]:
    metadata_list = []

    for entry in mds:
        testcases = entry.get("testcases", [])
        for testcase in testcases:
            testcase["context"] = entry.get("context", "default")
            testcase["component"] = entry.get("component", "default")
            testcase["jax_doc"] = entry.get("jax_doc", "")
            testcase["onnx"] = entry.get("onnx", "")
            testcase["source"] = entry.get("source", "")
            testcase["since"] = entry.get("since", "")
            testcase["description"] = entry.get("description", "")
            testcase["children"] = entry.get("children", [])
            metadata_list.append(testcase)
    return metadata_list


def load_metadata_from_plugins() -> list[dict[str, Any]]:
    import_all_plugins()
    return [
        {**plugin.metadata, "jaxpr_primitive": name}
        for name, plugin in PLUGIN_REGISTRY.items()
        if hasattr(plugin, "metadata")
    ]


def load_plugin_metadata() -> list[dict[str, Any]]:
    md = load_metadata_from_plugins()
    return extract_from_metadata(md)


# --- Test Param Generation ---
def generate_test_params(entry: dict[str, Any]) -> list[dict[str, Any]]:
    if "callable" not in entry:
        return []  # Skip if no callable

    input_shapes = entry.get("input_shapes", [])
    if isinstance(input_shapes, (list, tuple)) and any(
        "B" in shape for shape in input_shapes
    ):
        dynamic = entry.copy()
        dynamic["testcase"] += "_dynamic"
        concrete = entry.copy()
        concrete["input_shapes"] = [
            tuple(
                3 if dim == "B" else dim
                for dim in (s if isinstance(s, (list, tuple)) else (s,))
            )
            for s in input_shapes
        ]
        if "expected_output_shapes" in entry:
            concrete["expected_output_shapes"] = [
                tuple(
                    3 if dim == "B" else dim
                    for dim in (s if isinstance(s, (list, tuple)) else (s,))
                )
                for s in entry["expected_output_shapes"]
            ]
        return [dynamic, concrete]
    return [entry]


# --- Organizing Tests ---
def organize_tests_by_context_and_component_from_params(
    params: list[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouping: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for param in params:
        context = param.get("context", "default")
        component_name = param.get("component", "default").replace(".", "_")
        grouping.setdefault((context, component_name), []).append(param)
    return grouping


_GLOBAL_PLUGIN_GROUPING = None


def get_plugin_grouping(reset=False) -> dict[tuple[str, str], list[dict[str, Any]]]:
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


def make_test_function(tp: dict[str, Any]):
    test_case_name_safe = tp["testcase"].replace(".", "_").replace(" ", "_")
    func_name = f"test_{test_case_name_safe}"

    def test_func(self=None):
        # configure_logging() # Call once if needed

        callable_obj = tp["callable"]
        input_values_from_testcase = tp.get("input_values")
        input_shapes_from_testcase = tp.get("input_shapes")

        processed_input_specs_for_to_onnx: List[Any]

        if input_shapes_from_testcase is not None:
            processed_input_specs_for_to_onnx = input_shapes_from_testcase
            logger.info(
                f"Test '{tp['testcase']}': Using explicit input_shapes from testcase: {processed_input_specs_for_to_onnx}"
            )
        elif input_values_from_testcase is not None:
            processed_input_specs_for_to_onnx = [
                jax.ShapeDtypeStruct(np.asarray(val).shape, np.asarray(val).dtype)
                for val in input_values_from_testcase
            ]
            logger.info(
                f"Test '{tp['testcase']}': Inferred ShapeDtypeStructs for to_onnx from input_values: {processed_input_specs_for_to_onnx}"
            )
        else:
            sig = inspect.signature(callable_obj)
            if not sig.parameters:
                processed_input_specs_for_to_onnx = []
                logger.info(
                    f"Test '{tp['testcase']}': Callable takes no arguments. Using empty input_specs for to_onnx."
                )
            else:
                raise ValueError(
                    f"Testcase '{tp['testcase']}' (for a callable that expects arguments) "
                    "must provide 'input_shapes' (for symbolic/typed tracing) or 'input_values' (for concrete tracing)."
                )

        input_params_from_testcase = tp.get("input_params", {})
        testcase_name = tp["testcase"]
        expected_num_funcs = tp.get("expected_number_of_function_instances")
        expected_output_shapes_from_testcase = tp.get("expected_output_shapes")
        onnx_input_names_from_testcase = tp.get("input_names")

        context_path = tp.get("context", "default.unknown").split(".")
        opset_version = tp.get("opset_version", 21)
        model_path = os.path.join(
            "docs", "onnx", *context_path, f"{testcase_name}.onnx"
        )
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        logger.info(
            f"Converting '{testcase_name}' to ONNX with input_specs: {processed_input_specs_for_to_onnx}"
        )
        try:
            onnx_model = to_onnx(
                callable_obj,
                processed_input_specs_for_to_onnx,
                input_params=input_params_from_testcase,
                model_name=testcase_name,
                opset=opset_version,
            )
        except Exception as e:
            logger.error(
                f"Failed during to_onnx conversion for '{testcase_name}': {e}",
                exc_info=True,
            )
            raise

        onnx.save_model(onnx_model, model_path)
        logger.info(f"Model saved to: {model_path}")

        # --- Numerical Validation ---
        if input_values_from_testcase:
            logger.info(f"Running numerical check for '{testcase_name}'...")
            xs_for_num_check = [np.asarray(val) for val in input_values_from_testcase]
            passed_numerical, validation_message = allclose(
                callable_obj, model_path, xs_for_num_check, input_params_from_testcase
            )
            assert (
                passed_numerical
            ), f"Numerical check failed for {testcase_name}: {validation_message}"
            logger.info(f"Numerical check passed for {testcase_name}.")
        else:
            logger.info(
                f"No input_values provided for '{testcase_name}', skipping numerical validation."
            )

        # --- Function Count Check ---
        if expected_num_funcs is not None:
            num_found_funcs = len(list(onnx_model.functions))
            assert (
                num_found_funcs == expected_num_funcs
            ), f"Test '{testcase_name}': Expected {expected_num_funcs} ONNX functions, found {num_found_funcs}."
            logger.info(f"Found expected {num_found_funcs} ONNX functions.")

        # --- Refined Shape Checking Logic ---
        if expected_output_shapes_from_testcase:
            logger.info(f"== Checking output shapes for '{testcase_name}' ==")

            onnx_graph_structural_shapes = []
            for output_node_info in onnx_model.graph.output:
                dims = []
                for d_proto in output_node_info.type.tensor_type.shape.dim:
                    if d_proto.HasField("dim_param") and d_proto.dim_param:
                        dims.append(d_proto.dim_param)
                    elif d_proto.HasField("dim_value"):
                        dims.append(d_proto.dim_value)
                    else:
                        dims.append(None)
                logger.info(
                    f"ONNX Graph Output: {output_node_info.name}  Defined Structural Shape: {tuple(dims)}"
                )
                onnx_graph_structural_shapes.append(tuple(dims))

            runtime_onnx_output_shapes = []
            expected_concrete_jax_shapes = []

            if input_values_from_testcase:
                try:
                    onnx_outputs_numerical = _run_onnx_model_for_shape_check(
                        onnx_model,
                        [np.asarray(v) for v in input_values_from_testcase],
                        onnx_input_names_from_testcase,
                    )
                    runtime_onnx_output_shapes = [
                        tuple(out.shape) for out in onnx_outputs_numerical
                    ]
                    logger.info(
                        f"Shape Check ({testcase_name}): ONNX runtime actual output shapes: {runtime_onnx_output_shapes}"
                    )

                    jax_callable_inputs = [
                        jnp.asarray(val) for val in input_values_from_testcase
                    ]
                    jax_fn_outputs = callable_obj(*jax_callable_inputs)
                    if not isinstance(jax_fn_outputs, (list, tuple)):
                        jax_fn_outputs = [jax_fn_outputs]
                    expected_concrete_jax_shapes = [
                        tuple(out.shape) for out in jax_fn_outputs
                    ]
                    logger.info(
                        f"Shape Check ({testcase_name}): JAX callable expected concrete shapes: {expected_concrete_jax_shapes}"
                    )

                except Exception as e:
                    logger.error(
                        f"Shape Check ({testcase_name}): Error during runtime execution for shape comparison: {e}",
                        exc_info=True,
                    )

            if len(onnx_graph_structural_shapes) != len(
                expected_output_shapes_from_testcase
            ):
                raise AssertionError(
                    f"[❌] '{testcase_name}': Output count mismatch for shape assertion. "
                    f"Testcase expects {len(expected_output_shapes_from_testcase)} output shapes, "
                    f"ONNX graph defines {len(onnx_graph_structural_shapes)}."
                )

            final_assert_messages = []
            all_checks_passed = True

            for i, expected_shape_spec_from_testcase in enumerate(
                expected_output_shapes_from_testcase
            ):
                onnx_graph_shape = onnx_graph_structural_shapes[i]

                if len(expected_shape_spec_from_testcase) != len(onnx_graph_shape):
                    all_checks_passed = False
                    final_assert_messages.append(
                        f"Output {i} for '{testcase_name}': Rank mismatch. "
                        f"Expected rank {len(expected_shape_spec_from_testcase)} (from testcase: {expected_shape_spec_from_testcase}), "
                        f"ONNX graph rank {len(onnx_graph_shape)} (from ONNX: {onnx_graph_shape})."
                    )
                    continue

                graph_dim_match = True
                for j, expected_dim_representation in enumerate(
                    expected_shape_spec_from_testcase
                ):
                    onnx_graph_actual_dim = onnx_graph_shape[j]
                    if isinstance(expected_dim_representation, str):
                        if onnx_graph_actual_dim != expected_dim_representation:
                            if not (
                                expected_dim_representation.startswith("dynamic_")
                                and isinstance(onnx_graph_actual_dim, str)
                                and onnx_graph_actual_dim.startswith("dynamic_")
                            ):
                                graph_dim_match = False
                                break
                    elif isinstance(expected_dim_representation, int):
                        if onnx_graph_actual_dim != expected_dim_representation:
                            graph_dim_match = False
                            break

                if not graph_dim_match:
                    all_checks_passed = False
                    final_assert_messages.append(
                        f"Output {i} for '{testcase_name}' (ONNX Graph Structure): Mismatch. "
                        f"Expected from testcase: {expected_shape_spec_from_testcase}, "
                        f"Actual ONNX Graph: {onnx_graph_shape}."
                    )

                if input_values_from_testcase:
                    if i < len(runtime_onnx_output_shapes) and i < len(
                        expected_concrete_jax_shapes
                    ):
                        actual_runtime_shape = runtime_onnx_output_shapes[i]
                        authoritative_jax_runtime_shape = expected_concrete_jax_shapes[
                            i
                        ]
                        if actual_runtime_shape != authoritative_jax_runtime_shape:
                            all_checks_passed = False
                            final_assert_messages.append(
                                f"Output {i} for '{testcase_name}' (Runtime Shape): Mismatch. "
                                f"Expected (from JAX execution): {authoritative_jax_runtime_shape}, "
                                f"Actual ONNX runtime: {actual_runtime_shape}."
                            )
                    elif i < len(expected_concrete_jax_shapes):
                        all_checks_passed = False
                        final_assert_messages.append(
                            f"Output {i} for '{testcase_name}' (Runtime): JAX expected shape {expected_concrete_jax_shapes[i]} but ONNX runtime output missing or produced fewer outputs."
                        )

            if not all_checks_passed:
                raise AssertionError(
                    f"[❌] Test '{testcase_name}': Output shape checks failed.\n"
                    + "\n".join(final_assert_messages)
                )
            logger.info(
                f"-> '{testcase_name}': Output shapes checks passed (graph structure and runtime)."
            )

    test_func.__name__ = func_name
    setattr(test_func, "_testcase_params", tp)
    return test_func


# Helper function to run ONNX model for shape check
def _run_onnx_model_for_shape_check(
    model_proto: onnx.ModelProto,
    input_values_list: Sequence[np.ndarray],
    input_names_list_from_testcase: Sequence[str] | None = None,
) -> List[np.ndarray]:
    import onnxruntime

    sess_options = onnxruntime.SessionOptions()
    providers = ["CPUExecutionProvider"]
    model_bytes = model_proto.SerializeToString()
    try:
        ort_session = onnxruntime.InferenceSession(
            model_bytes, sess_options, providers=providers
        )
    except Exception as e:
        logger.error(
            f"Failed to create ONNX InferenceSession for model '{model_proto.graph.name}': {e}"
        )
        raise

    model_graph_input_names = [inp.name for inp in ort_session.get_inputs()]
    initializers = {init.name for init in model_proto.graph.initializer}
    runnable_model_input_names = [
        name for name in model_graph_input_names if name not in initializers
    ]

    if input_names_list_from_testcase:
        current_input_names_for_onnx = list(input_names_list_from_testcase)
    else:
        current_input_names_for_onnx = runnable_model_input_names

    if len(current_input_names_for_onnx) != len(input_values_list):
        error_msg = (
            f"Mismatch in number of ONNX input names to use ({len(current_input_names_for_onnx)}: {current_input_names_for_onnx}) "
            f"and provided input_values ({len(input_values_list)}). Model expects {len(runnable_model_input_names)} runnable inputs: {runnable_model_input_names}."
        )
        logger.error(error_msg)
        if (
            len(runnable_model_input_names) == len(input_values_list)
            and not input_names_list_from_testcase
        ):
            logger.warning(
                f"Using inferred runnable_model_input_names for ONNX session: {runnable_model_input_names}"
            )
            current_input_names_for_onnx = runnable_model_input_names
        else:
            raise ValueError(
                error_msg
                + " Please define 'input_names' in testcase if order/names are ambiguous."
            )

    inputs_dict = {
        name: val for name, val in zip(current_input_names_for_onnx, input_values_list)
    }
    try:
        outputs = ort_session.run(None, inputs_dict)
    except Exception as e:
        logger.error(
            f"ONNX runtime error during model execution for '{model_proto.graph.name}': {e}"
        )
        logger.error(
            f"Inputs provided to ONNX runtime: { {k: (v.shape, v.dtype) for k,v in inputs_dict.items()} }"
        )
        raise
    return outputs


# --- Test Class Registration ---
def generate_test_class(context: str, component: str, namespace: dict):
    # Component name might contain '.', replace for valid class name
    class_name_suffix = component.replace(".", "_")
    class_name = f"Test_{class_name_suffix}"

    # Retrieve test cases using the original context and component key
    testcases = get_plugin_grouping().get(
        (context, component), []
    )  # Use original component name for generating class call

    attrs = {}
    for tp in testcases:
        test_name_safe = tp["testcase"].replace(".", "_").replace(" ", "_")
        attrs[f"test_{test_name_safe}"] = make_test_function(tp)

    if attrs:
        namespace[class_name] = type(class_name, (object,), attrs)


# --- Minimal Test File Generation ---
def create_minimal_test_file(directory: str, context: str, components: list[str]):
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
    grouping: dict[tuple[str, str], list[dict[str, Any]]], directory: str
):
    """Creates minimal test files, grouping components by context."""
    context_components: dict[str, list[str]] = {}
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
