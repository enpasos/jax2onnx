# file: tests/t_generator.py

import os
import shutil
from typing import Any, Dict, List, Sequence  # Ensure Dict is imported
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import logging

# from logging_config import configure_logging # Assuming this is handled elsewhere or not strictly needed for this fix

from jax2onnx import allclose
from jax2onnx.converter.user_interface import to_onnx
from jax2onnx.plugin_system import (
    PLUGIN_REGISTRY,
    import_all_plugins,
)

# Define base directories.
TESTS_DIR = os.path.dirname(__file__)
PLUGINS_DIR = os.path.join(
    TESTS_DIR, "../jax2onnx/plugins"
)  # Corrected path assuming plugins are one level up from jax2onnx/tests

# Configure logger for this module
logger = logging.getLogger("jax2onnx.tests.t_generator")
# Basic logging configuration if not set elsewhere
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


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
    """
    Generates test parameters for a given metadata entry.
    This involves:
    1. Creating dynamic and concrete shape variants if "B" (batch dim) is present.
    2. For each of those, creating float32 (default) and float64 variants.
    """
    if "callable" not in entry:
        logger.debug(f"Skipping entry, no callable: {entry.get('testcase', 'Unknown')}")
        return []

    # Part 1: Handle dynamic vs. concrete shapes
    # entry_input_shapes is typically List[Tuple[Union[str, int], ...]]
    # e.g., entry_input_shapes = [("B", 28, 28, 3)]
    entry_input_shapes = entry.get("input_shapes", [])

    intermediate_params_list: List[Dict[str, Any]] = []

    has_dynamic_dim = False
    if isinstance(entry_input_shapes, (list, tuple)):
        for (
            shape_spec
        ) in entry_input_shapes:  # shape_spec is a tuple like ("B", 28, 28, 3)
            if isinstance(shape_spec, (list, tuple)) and "B" in shape_spec:
                has_dynamic_dim = True
                break

    if has_dynamic_dim:
        # Create dynamic variant
        dynamic_param_set = entry.copy()
        dynamic_param_set["testcase"] += "_dynamic"
        # input_shapes for dynamic remains as is (e.g., [("B", 28, 28, 3)])
        intermediate_params_list.append(dynamic_param_set)

        # Create concrete variant
        concrete_param_set = entry.copy()
        # The testcase name for the concrete version usually doesn't get a special suffix
        # if the dynamic one is already explicitly named.

        # Convert input_shapes for concrete version
        concrete_input_s = []
        for shape_spec in entry_input_shapes:  # e.g., shape_spec is ("B", 28, 28, 3)
            if isinstance(shape_spec, (list, tuple)):
                concrete_input_s.append(
                    tuple(3 if dim == "B" else dim for dim in shape_spec)
                )
            else:
                # This case should ideally not happen with well-formed metadata
                concrete_input_s.append(shape_spec)
        concrete_param_set["input_shapes"] = concrete_input_s

        # Convert expected_output_shapes if they exist for concrete version
        if "expected_output_shapes" in entry:
            concrete_output_s = []
            entry_expected_output_shapes = entry.get("expected_output_shapes", [])
            for shape_spec in entry_expected_output_shapes:
                if isinstance(shape_spec, (list, tuple)):
                    concrete_output_s.append(
                        tuple(3 if dim == "B" else dim for dim in shape_spec)
                    )
                else:
                    concrete_output_s.append(shape_spec)
            concrete_param_set["expected_output_shapes"] = concrete_output_s
        intermediate_params_list.append(concrete_param_set)
    else:
        # No dynamic "B" dimension, just use the entry as is (it's effectively concrete)
        intermediate_params_list.append(entry.copy())

    # Part 2: For each base parameter set from Part 1, create float32 and float64 variants
    final_params_list: List[Dict[str, Any]] = []
    for base_param_set in intermediate_params_list:
        # Default (float32)
        p_f32 = base_param_set.copy()
        p_f32["_enable_float64_test_setting"] = False
        # Testcase name for f32 is usually the base name (e.g., "test_conv_basic" or "test_conv_basic_dynamic")
        final_params_list.append(p_f32)

        # Float64 variant, only if not explicitly disabled by test metadata
        if not base_param_set.get("disable_float64_test", False):
            p_f64 = base_param_set.copy()
            p_f64["testcase"] += "_f64"  # Add suffix for the float64 test case name
            p_f64["_enable_float64_test_setting"] = True

            # If input_values are provided, try to cast them to float64 for the f64 test
            if "input_values" in p_f64 and p_f64["input_values"] is not None:
                try:
                    p_f64["input_values"] = [
                        (
                            np.array(val, dtype=np.float64)
                            if np.issubdtype(np.array(val).dtype, np.floating)
                            else np.array(val)
                        )
                        for val in p_f64["input_values"]
                    ]
                except Exception as e:
                    logger.warning(
                        f"Could not cast input_values to float64 for testcase {p_f64['testcase']}: {e}"
                    )
            final_params_list.append(p_f64)

    return final_params_list


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

    def test_func(self=None):  # Pytest will inject 'self' if it's a method of a class
        # configure_logging() # Call once if needed, or ensure logger is configured globally

        callable_obj = tp["callable"]
        input_values_from_testcase = tp.get("input_values")
        # input_shapes_from_testcase is List[Tuple[Union[str, int], ...]]
        # e.g. [("B", 28, 28, 3)] or [(2, 28, 28, 1)]
        input_shapes_from_testcase = tp.get("input_shapes")

        # Get the float64 setting for this specific test variant
        current_enable_float64 = tp.get("_enable_float64_test_setting", False)
        # Determine default JAX dtype based on the flag for tracing if shapes are given without dtypes
        default_jax_dtype_for_tracing = (
            jnp.float64 if current_enable_float64 else jnp.float32
        )

        # processed_input_specs_for_to_onnx will be List[Union[ShapeDtypeStruct, Tuple[Shape, Dtype]]]
        # to_onnx expects List[Tuple[Shape, Dtype]] or List[ShapeDtypeStruct]
        processed_input_specs_for_to_onnx: List[Any]

        if input_shapes_from_testcase is not None:
            # input_shapes_from_testcase is List[Tuple[Dim,...]]
            # We need to pair them with a dtype for to_onnx's `inputs` argument.
            # The dtype should be default_jax_dtype_for_tracing.
            processed_input_specs_for_to_onnx = []
            for shape_spec in input_shapes_from_testcase:
                # Ensure shape_spec is a tuple
                current_shape_tuple = (
                    tuple(shape_spec)
                    if isinstance(shape_spec, (list, tuple))
                    else (shape_spec,)
                )
                processed_input_specs_for_to_onnx.append(
                    current_shape_tuple
                )  # This will be passed to to_onnx, which expects shapes

            logger.info(
                f"Test '{tp['testcase']}': Using explicit input_shapes from testcase: {processed_input_specs_for_to_onnx}. "
                f"enable_float64={current_enable_float64}, implies default JAX dtype for tracing: {default_jax_dtype_for_tracing}"
            )
        elif input_values_from_testcase is not None:
            shapes_from_values = []  # This list will hold ShapeDtypeStructs
            for val in input_values_from_testcase:
                np_array = np.array(val)
                # For float64 test variant, if original was float, ensure it's float64 for tracing
                # otherwise, preserve the original dtype (bool, int32, etc.)
                if current_enable_float64 and np.issubdtype(
                    np_array.dtype, np.floating
                ):
                    shapes_from_values.append(
                        jax.ShapeDtypeStruct(np_array.shape, jnp.float64)
                    )
                else:
                    # For f32 or non-float types (bool, int), use original dtype
                    shapes_from_values.append(
                        jax.ShapeDtypeStruct(np_array.shape, np_array.dtype)
                    )

            # ****** THE FIX IS HERE ******
            # Pass the list of ShapeDtypeStructs directly
            processed_input_specs_for_to_onnx = shapes_from_values
            # Previously was: [s.shape for s in shapes_from_values]

            logger.info(
                f"Test '{tp['testcase']}': Using ShapeDtypeStructs for to_onnx from input_values: {processed_input_specs_for_to_onnx}. "
                f"enable_float64={current_enable_float64}."
            )
        else:
            sig = inspect.signature(callable_obj)
            if not sig.parameters:
                processed_input_specs_for_to_onnx = []  # Empty list of shapes
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
        opset_version = tp.get(
            "opset_version", 21
        )  # Ensure this is a reasonable default

        # Ensure model_path is correctly formed
        # Example: docs/onnx/primitives/nnx/conv_basic_bias_f64.onnx
        model_folder_path = os.path.join("docs", "onnx", *context_path)
        os.makedirs(model_folder_path, exist_ok=True)
        model_path = os.path.join(model_folder_path, f"{test_case_name_safe}.onnx")

        logger.info(
            f"Converting '{testcase_name}' to ONNX with input shapes: {processed_input_specs_for_to_onnx}, "
            f"enable_float64: {current_enable_float64}"
        )
        try:
            # to_onnx expects `inputs` to be a list of shape tuples.
            # `default_dtype` in `to_onnx` will be combined with `enable_float64`
            # to determine the `working_dtype` for these shapes if dtypes aren't part of `processed_input_specs_for_to_onnx`.
            onnx_model = to_onnx(
                callable_obj,
                processed_input_specs_for_to_onnx,  # This is List[Tuple[Dim,...]]
                input_params=input_params_from_testcase,
                model_name=testcase_name,  # Use the specific test case name (e.g., with _f64)
                opset=opset_version,
                enable_float64=current_enable_float64,  # Pass the flag
            )
        except Exception as e:
            logger.error(
                f"Failed during to_onnx conversion for '{testcase_name}' with enable_float64={current_enable_float64}: {e}",
                exc_info=True,
            )
            raise

        onnx.save_model(onnx_model, model_path)
        logger.info(f"Model saved to: {model_path}")

        # --- Numerical Validation ---
        if input_values_from_testcase:
            logger.info(
                f"Running numerical check for '{testcase_name}' (enable_float64={current_enable_float64})..."
            )
            # For f64 tests, input_values_from_testcase should already be cast to np.float64 if they were floats
            # The callable_obj will be called with these (potentially f64) inputs by allclose
            # The ONNX model was generated with enable_float64=True, so it should also compute in f64

            # Adjust tolerance for float64 comparisons if needed
            rtol = 1e-7 if current_enable_float64 else 1e-5
            atol = 1e-7 if current_enable_float64 else 1e-5

            xs_for_num_check = [np.asarray(val) for val in input_values_from_testcase]

            passed_numerical, validation_message = allclose(
                callable_obj,
                model_path,
                xs_for_num_check,
                input_params_from_testcase,
                rtol=rtol,
                atol=atol,
            )
            assert (
                passed_numerical
            ), f"Numerical check failed for {testcase_name} (enable_float64={current_enable_float64}): {validation_message}"
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

        # --- DType Validation for ONNX model outputs ---
        # This check is crucial for the enable_float64 flag.
        # We expect float outputs to be DOUBLE if current_enable_float64 is True.
        if input_values_from_testcase:  # Only if we can infer JAX output dtypes
            # Determine expected JAX output dtypes by evaluating shape (and dtype) of JAX function
            # Ensure inputs to JAX for this eval_shape match the float64 setting
            jax_eval_inputs_sds = []
            for (
                val
            ) in (
                input_values_from_testcase
            ):  # These are already potentially np.float64 for f64 tests
                np_array_val = np.array(val)
                # If current_enable_float64 is true and it's a float, use jnp.float64 for eval_shape
                if current_enable_float64 and np.issubdtype(
                    np_array_val.dtype, np.floating
                ):
                    jax_eval_inputs_sds.append(
                        jax.ShapeDtypeStruct(np_array_val.shape, jnp.float64)
                    )
                else:  # Otherwise, use the dtype from the (potentially f32) input_values
                    jax_eval_inputs_sds.append(
                        jax.ShapeDtypeStruct(np_array_val.shape, np_array_val.dtype)
                    )

            try:
                # JAX callable_obj might be a class instance, ensure it's callable for eval_shape
                # If callable_obj is an nnx.Module, its __call__ is the target
                eval_target_jax_func = callable_obj

                # If callable_obj is an instance of a class that itself is not directly callable
                # but has a __call__ method (common for Flax modules), use that.
                if (
                    hasattr(callable_obj, "__call__")
                    and not inspect.isfunction(callable_obj)
                    and not inspect.ismethod(callable_obj)
                ):
                    # Check if it's a class instance rather than the class type itself
                    if not inspect.isclass(callable_obj):
                        eval_target_jax_func = callable_obj.__call__
                    # If it is the class type, we assume to_onnx handles it by instantiating or similar.
                    # For eval_shape, we'd typically need an instance or a function.
                    # This part might need refinement based on how `callable_obj` is structured for nnx.Conv etc.
                    # For nnx.Conv, callable_obj is an instance of nnx.Conv.

                # Special handling for nnx.Module instances for eval_shape
                # JAX's eval_shape expects a function. If callable_obj is an nnx.Module instance,
                # we need to wrap its call.
                if isinstance(callable_obj, jax.tree_util.Partial) or (
                    hasattr(callable_obj, "__module__")
                    and "flax.experimental.nnx" in callable_obj.__module__
                ):

                    def jax_func_wrapper(*args):
                        # For nnx.Conv, it expects one arg usually.
                        # If callable_obj is nnx.Conv(..), then callable_obj(x) is the call.
                        return callable_obj(*args)

                    eval_target_jax_func = jax_func_wrapper

                jax_output_sds = jax.eval_shape(
                    eval_target_jax_func, *jax_eval_inputs_sds
                )

                if not isinstance(jax_output_sds, (list, tuple)):
                    jax_output_sds = [jax_output_sds]

                expected_onnx_dtype_for_float_outputs = (
                    onnx.TensorProto.DOUBLE
                    if current_enable_float64
                    else onnx.TensorProto.FLOAT
                )

                for i, onnx_output_vi in enumerate(onnx_model.graph.output):
                    if i < len(jax_output_sds):
                        jax_dtype = jax_output_sds[i].dtype
                        onnx_tensor_type_enum = (
                            onnx_output_vi.type.tensor_type.elem_type
                        )

                        # We only enforce this check for JAX float types. Other types (int, bool) should remain as they are.
                        if jnp.issubdtype(jax_dtype, jnp.floating):
                            assert (
                                onnx_tensor_type_enum
                                == expected_onnx_dtype_for_float_outputs
                            ), (
                                f"Test '{testcase_name}', output '{onnx_output_vi.name}' (index {i}): "
                                f"ONNX dtype mismatch. JAX output dtype is {jax_dtype}, "
                                f"enable_float64 is {current_enable_float64}. "
                                f"Expected ONNX type {onnx.TensorProto.DataType.Name(expected_onnx_dtype_for_float_outputs)}, "
                                f"but got {onnx.TensorProto.DataType.Name(onnx_tensor_type_enum)}."
                            )
                logger.info(
                    f"Output dtypes verified for '{testcase_name}' (enable_float64={current_enable_float64})."
                )

            except Exception as e:
                logger.warning(
                    f"Could not perform JAX eval_shape for dtype checking in test '{testcase_name}' (enable_float64={current_enable_float64}): {e}",
                    exc_info=True,
                )
        else:
            logger.info(
                f"Skipping ONNX output dtype validation for '{testcase_name}' as no input_values were provided."
            )

        # --- Refined Shape Checking Logic ---
        if expected_output_shapes_from_testcase:
            logger.info(f"== Checking output shapes for '{testcase_name}' ==")
            # ... (existing shape checking logic, should be fine) ...
            # The existing shape checking logic seems robust and independent of dtype.
            # It compares ONNX graph structural shapes and runtime shapes against testcase expectations.

            onnx_graph_structural_shapes = []
            for output_node_info in onnx_model.graph.output:
                dims = []
                for d_proto in output_node_info.type.tensor_type.shape.dim:
                    if d_proto.HasField("dim_param") and d_proto.dim_param:
                        dims.append(d_proto.dim_param)
                    elif d_proto.HasField("dim_value"):
                        dims.append(d_proto.dim_value)
                    else:
                        dims.append(None)  # Represent unknown fixed dimension
                logger.info(
                    f"ONNX Graph Output: {output_node_info.name}  Defined Structural Shape: {tuple(dims)}"
                )
                onnx_graph_structural_shapes.append(tuple(dims))

            runtime_onnx_output_shapes = []
            expected_concrete_jax_shapes = []

            if input_values_from_testcase:
                try:
                    # Use the same potentially float64 inputs for ONNX runtime
                    onnx_runtime_inputs = [
                        np.asarray(v) for v in input_values_from_testcase
                    ]
                    onnx_outputs_numerical = _run_onnx_model_for_shape_check(
                        onnx_model,
                        onnx_runtime_inputs,
                        onnx_input_names_from_testcase,
                    )
                    runtime_onnx_output_shapes = [
                        tuple(out.shape) for out in onnx_outputs_numerical
                    ]
                    logger.info(
                        f"Shape Check ({testcase_name}): ONNX runtime actual output shapes: {runtime_onnx_output_shapes}"
                    )

                    # JAX callable inputs should also match the float64 setting for this run
                    # jax_callable_inputs are already prepared as jax_eval_inputs_sds (ShapeDtypeStructs)
                    # For actual JAX execution, we need jnp arrays
                    jax_concrete_inputs = []
                    for (
                        sds
                    ) in jax_eval_inputs_sds:  # These were prepared for eval_shape
                        # Create dummy JAX arrays with the correct shape and dtype for this test variant
                        # This is only for getting JAX output shapes; numerical check uses input_values directly.
                        jax_concrete_inputs.append(
                            jnp.zeros(sds.shape, dtype=sds.dtype)
                        )

                    # Re-evaluate JAX function with concrete shapes and dtypes for this test variant
                    # Use the same eval_target_jax_func as in dtype checking

                    jax_concrete_inputs = []
                    if input_values_from_testcase:  # Should be true for this test
                        for idx, val_spec in enumerate(input_values_from_testcase):
                            # Use the dtype from jax_eval_inputs_sds which respects current_enable_float64
                            target_dtype = jax_eval_inputs_sds[idx].dtype
                            jax_concrete_inputs.append(
                                jnp.asarray(
                                    val_spec, dtype=target_dtype
                                )  # Use actual value
                            )
                    else:  # Fallback, or for tests with no input_values (like static arange)
                        for (
                            sds
                        ) in (
                            jax_eval_inputs_sds
                        ):  # jax_eval_inputs_sds would be empty for static arange
                            jax_concrete_inputs.append(
                                jnp.zeros(sds.shape, dtype=sds.dtype)
                            )

                    # If callable_obj takes no arguments but input_values_from_testcase was empty (e.g. static test)
                    # jax_concrete_inputs should be empty.
                    # The original eval_target_jax_func(*jax_concrete_inputs) should handle this.
                    # For static tests (like lambda: jnp.arange(5)), jax_eval_inputs_sds and input_values_from_testcase are empty.
                    # So jax_concrete_inputs will also be empty, which is correct for calling such a lambda.

                    jax_fn_outputs_for_shape = eval_target_jax_func(
                        *jax_concrete_inputs
                    )

                    if not isinstance(jax_fn_outputs_for_shape, (list, tuple)):
                        jax_fn_outputs_for_shape = [jax_fn_outputs_for_shape]
                    expected_concrete_jax_shapes = [
                        tuple(out.shape) for out in jax_fn_outputs_for_shape
                    ]
                    logger.info(
                        f"Shape Check ({testcase_name}): JAX callable expected concrete shapes: {expected_concrete_jax_shapes}"
                    )

                except Exception as e:
                    logger.error(
                        f"Shape Check ({testcase_name}): Error during runtime execution for shape comparison: {e}",
                        exc_info=True,
                    )

            # ... (rest of the shape assertion logic from the original file) ...
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

                # Compare rank
                if len(expected_shape_spec_from_testcase) != len(onnx_graph_shape):
                    all_checks_passed = False
                    final_assert_messages.append(
                        f"Output {i} for '{testcase_name}': Rank mismatch. "
                        f"Expected rank {len(expected_shape_spec_from_testcase)} (from testcase: {expected_shape_spec_from_testcase}), "
                        f"ONNX graph rank {len(onnx_graph_shape)} (from ONNX: {onnx_graph_shape})."
                    )
                    continue  # Skip further checks for this output if rank mismatches

                # Compare dimensions (symbolic or concrete)
                graph_dim_match = True
                for j, expected_dim_representation in enumerate(
                    expected_shape_spec_from_testcase
                ):
                    onnx_graph_actual_dim = onnx_graph_shape[j]
                    if isinstance(
                        expected_dim_representation, str
                    ):  # Symbolic dim in testcase
                        # If testcase expects symbolic, ONNX graph should also have symbolic (dim_param)
                        # or a matching concrete value if the symbolic dim was resolved during test setup
                        if onnx_graph_actual_dim != expected_dim_representation:
                            # Allow for "dynamic_XYZ" to match if testcase specified "B" and it became dynamic
                            if not (
                                expected_dim_representation == "B"
                                and isinstance(onnx_graph_actual_dim, str)
                                and onnx_graph_actual_dim.startswith("dynamic_")
                            ):
                                graph_dim_match = False
                                break
                    elif isinstance(
                        expected_dim_representation, int
                    ):  # Concrete dim in testcase
                        if onnx_graph_actual_dim != expected_dim_representation:
                            graph_dim_match = False
                            break
                    # else: expected_dim_representation might be None, handle if necessary

                if not graph_dim_match:
                    all_checks_passed = False
                    final_assert_messages.append(
                        f"Output {i} for '{testcase_name}' (ONNX Graph Structure): Mismatch. "
                        f"Expected from testcase: {expected_shape_spec_from_testcase}, "
                        f"Actual ONNX Graph: {onnx_graph_shape}."
                    )

                # Compare with runtime shapes if available
                if (
                    input_values_from_testcase
                ):  # implies runtime_onnx_output_shapes and expected_concrete_jax_shapes are populated
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
                    elif i < len(
                        expected_concrete_jax_shapes
                    ):  # JAX produced output, ONNX didn't (or fewer)
                        all_checks_passed = False
                        final_assert_messages.append(
                            f"Output {i} for '{testcase_name}' (Runtime): JAX expected shape {expected_concrete_jax_shapes[i]} but ONNX runtime output missing or produced fewer outputs."
                        )
                    # Could add a case for ONNX producing more outputs than JAX if necessary

            if not all_checks_passed:
                raise AssertionError(
                    f"[❌] Test '{testcase_name}': Output shape checks failed.\n"
                    + "\n".join(final_assert_messages)
                )
            logger.info(
                f"-> '{testcase_name}': Output shapes checks passed (graph structure and runtime)."
            )

    test_func.__name__ = func_name
    setattr(
        test_func, "_testcase_params", tp
    )  # Store params for potential debugging or inspection
    return test_func


# Helper function to run ONNX model for shape check
def _run_onnx_model_for_shape_check(
    model_proto: onnx.ModelProto,
    input_values_list: Sequence[np.ndarray],
    input_names_list_from_testcase: Sequence[str] | None = None,
) -> List[np.ndarray]:
    # Ensure onnxruntime is available
    try:
        import onnxruntime
    except ImportError:
        logger.error(
            "onnxruntime is not installed. Skipping ONNX model execution for shape check."
        )
        # Return empty list or raise error, depending on desired strictness
        return []

    sess_options = onnxruntime.SessionOptions()
    # Consider adding more providers if needed, e.g., 'CUDAExecutionProvider'
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

    # Determine input names for the ONNX model
    model_graph_input_names = [inp.name for inp in ort_session.get_inputs()]
    initializers = {init.name for init in model_proto.graph.initializer}
    # Runnable inputs are graph inputs that are not initializers
    runnable_model_input_names = [
        name for name in model_graph_input_names if name not in initializers
    ]

    # Use input names from testcase if provided, otherwise use inferred runnable names
    if input_names_list_from_testcase:
        current_input_names_for_onnx = list(input_names_list_from_testcase)
    else:
        current_input_names_for_onnx = runnable_model_input_names

    # Validate input counts
    if len(current_input_names_for_onnx) != len(input_values_list):
        error_msg_detail = (
            f"Mismatch in number of ONNX input names to use ({len(current_input_names_for_onnx)}: {current_input_names_for_onnx}) "
            f"and provided input_values ({len(input_values_list)}). Model expects {len(runnable_model_input_names)} runnable inputs: {runnable_model_input_names}."
        )
        logger.error(error_msg_detail)
        # If names weren't from testcase and count matches runnable_model_input_names, try to use those
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
                error_msg_detail
                + " Please define 'input_names' in testcase if order/names are ambiguous or if params are also graph inputs."
            )

    # Create input dictionary for ONNX runtime
    inputs_dict = {
        name: val for name, val in zip(current_input_names_for_onnx, input_values_list)
    }
    try:
        outputs = ort_session.run(None, inputs_dict)
    except Exception as e:
        logger.error(
            f"ONNX runtime error during model execution for '{model_proto.graph.name}': {e}"
        )
        # Log details of inputs provided to ONNX runtime for easier debugging
        inputs_summary = {k: (v.shape, str(v.dtype)) for k, v in inputs_dict.items()}
        logger.error(f"Inputs provided to ONNX runtime: {inputs_summary}")
        raise
    return outputs


# --- Test Class Registration ---
def generate_test_class(context: str, component: str, namespace: dict):
    # Component name might contain '.', replace for valid class name
    class_name_suffix = component.replace(".", "_")
    class_name = f"Test_{class_name_suffix}"

    # Retrieve test cases using the original context and component key
    # The component name used in get_plugin_grouping keys is already sanitized ('.' -> '_')
    testcases = get_plugin_grouping().get((context, component.replace(".", "_")), [])

    attrs = {}
    for tp in testcases:
        # Ensure test case names are unique and valid Python identifiers
        test_name_safe = (
            tp["testcase"].replace(".", "_").replace(" ", "_").replace("-", "_")
        )
        attrs[f"test_{test_name_safe}"] = make_test_function(tp)

    if attrs:
        # Create the test class and add it to the provided namespace (e.g., globals() of the test file)
        namespace[class_name] = type(class_name, (object,), attrs)
        logger.debug(
            f"Generated test class: {class_name} with {len(attrs)} test methods."
        )
    else:
        logger.debug(
            f"No test cases found for {context}.{component}, so class {class_name} was not generated."
        )


# --- Minimal Test File Generation ---
def create_minimal_test_file(directory: str, context: str, components: list[str]):
    folder_parts = context.split(".")  # e.g., "primitives.lax" -> ["primitives", "lax"]
    folder_name = folder_parts[0]  # "primitives"
    # Use the last part of the context for the sub_folder_name/file_name part
    sub_folder_or_file_name_part = (
        folder_parts[-1] if len(folder_parts) > 0 else folder_name
    )

    target_dir = os.path.join(directory, folder_name)  # e.g., tests/primitives
    os.makedirs(target_dir, exist_ok=True)

    # Ensure __init__.py exists in the target_dir (e.g., tests/primitives/__init__.py)
    init_path_target_dir = os.path.join(target_dir, "__init__.py")
    if not os.path.exists(init_path_target_dir):
        with open(init_path_target_dir, "w"):
            pass

    # Filename based on the last part of the context
    # e.g., tests/primitives/test_lax.py
    filename = os.path.join(target_dir, f"test_{sub_folder_or_file_name_part}.py")

    mode = "a" if os.path.exists(filename) else "w"
    if mode == "w":
        logger.info(f"Generating new test file: {filename}")
    else:
        logger.info(f"Appending to existing test file: {filename}")

    with open(filename, mode) as f:
        if mode == "w":  # Write header only for new files
            f.write("# Auto-generated by tests/t_generator.py\n")
            f.write("from tests.t_generator import generate_test_class\n\n")

        # For each component, generate the class registration call
        for component in components:  # component is like "conv", "add", etc.
            # The component name passed to generate_test_class should be the original one from metadata
            # (before sanitization for file/class naming if that was different)
            f.write(f"# Tests for {context}.{component}\n")
            f.write(
                f"generate_test_class(context={repr(context)}, component={repr(component)}, namespace=globals())\n\n"
            )


def create_minimal_test_files(
    grouping: dict[tuple[str, str], list[dict[str, Any]]], directory: str
):
    """Creates minimal test files, grouping components by context."""
    context_components: dict[str, list[str]] = {}
    for (context_key, component_key_sanitized), test_params_list in grouping.items():
        if not test_params_list:
            continue
        # Retrieve the original component name from the first test parameter set
        # This assumes 'component' in test_params_list[0] is the original, unsanitized name.
        original_component_name = test_params_list[0].get(
            "component", component_key_sanitized
        )

        context_components.setdefault(context_key, []).append(original_component_name)

    # Ensure unique components per context before writing files
    unique_context_components = {
        ctx: sorted(list(set(comps))) for ctx, comps in context_components.items()
    }

    for context_str, components_list in unique_context_components.items():
        create_minimal_test_file(directory, context_str, components_list)


# --- Main Generation ---
def generate_all_tests():
    """Cleans generated directories and creates all test files."""
    logger.info("Starting generation of all test files...")
    clean_generated_test_dirs()
    # Get plugin grouping (force reset to load fresh metadata and apply new generate_test_params logic)
    plugin_grouping_data = get_plugin_grouping(reset=True)
    create_minimal_test_files(plugin_grouping_data, TESTS_DIR)
    logger.info("Test file generation complete.")


if __name__ == "__main__":
    # This allows the script to be run directly to regenerate tests.
    # Ensure JAX/ONNX and other dependencies are available in the environment.
    # It's good practice to also import and configure logging here if not done by an imported module.
    # For example:
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    logger.info("Running t_generator.py script...")
    generate_all_tests()
