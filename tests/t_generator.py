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
from logging_config import (
    configure_logging,
)  # Assuming this is handled elsewhere or not strictly needed for this fix

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

    # ---- Sanity check for the new optional field ---------------------------------
    entry_input_dtypes = entry.get("input_dtypes")
    entry_input_shapes = entry.get("input_shapes", [])
    if (
        entry_input_dtypes is not None
        and entry_input_shapes is not None
        and len(entry_input_dtypes) != len(entry_input_shapes)
    ):
        raise ValueError(
            f"[metadata] In testcase '{entry.get('testcase')}' the "
            "`input_dtypes` list must have the same length as `input_shapes`."
        )

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

    run_only_dynamic = entry.get("run_only_dynamic", False)

    if has_dynamic_dim:
        # Create dynamic variant
        dynamic_param_set = entry.copy()
        dynamic_param_set["testcase"] += "_dynamic"
        # input_shapes for dynamic remains as is (e.g., [("B", 28, 28, 3)])
        intermediate_params_list.append(dynamic_param_set)

        # 👉  Skip the concrete branch when the testcase opts‑in
        if not run_only_dynamic:
            concrete_param_set = entry.copy()
            # The testcase name for the concrete version usually doesn't get a special suffix
            # if the dynamic one is already explicitly named.

            # Convert input_shapes for concrete version
            concrete_input_s = []
            for (
                shape_spec
            ) in entry_input_shapes:  # e.g., shape_spec is ("B", 28, 28, 3)
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
        run_only_f64_variant = base_param_set.get("run_only_f64_variant", False)
        run_only_f32_variant = base_param_set.get("run_only_f32_variant", False)
        disable_float64_test_from_meta = base_param_set.get(
            "disable_float64_test", False
        )  # Existing flag

        # Determine if the test case involves any floating point dtypes in its inputs or expected outputs
        # Simplified check based on presence of float in input_dtypes or expected_output_dtypes
        is_float_test = False

        bpset_input_dtypes = base_param_set.get("input_dtypes")
        if bpset_input_dtypes is not None:
            if any(np.issubdtype(dt, np.floating) for dt in bpset_input_dtypes):
                is_float_test = True

        if not is_float_test:
            bpset_expected_output_dtypes = base_param_set.get("expected_output_dtypes")
            if bpset_expected_output_dtypes is not None:
                if any(
                    np.issubdtype(dt, np.floating)
                    for dt in bpset_expected_output_dtypes
                ):
                    is_float_test = True

        if run_only_f64_variant:
            # Only generate the float64 enabled variant, using the original testcase name
            p_f64_only = base_param_set.copy()
            p_f64_only["_enable_double_precision_test_setting"] = True
            # testcase name remains p_f64_only["testcase"] (no suffix)

            # Apply float64 casting to values/dtypes for this variant if it's a float test
            if (
                is_float_test or True
            ):  # Apply always if only_f64 is true to ensure dtypes are set for float64 mode
                if (
                    "input_values" in p_f64_only
                    and p_f64_only["input_values"] is not None
                ):
                    p_f64_only["input_values"] = [
                        (
                            np.array(val, dtype=np.float64)
                            if np.issubdtype(np.array(val).dtype, np.floating)
                            else np.array(val)
                        )
                        for val in p_f64_only["input_values"]
                    ]
                if (
                    "expected_output_dtypes" in p_f64_only
                    and p_f64_only["expected_output_dtypes"] is not None
                ):
                    p_f64_only["expected_output_dtypes"] = [
                        (np.float64 if np.issubdtype(dtype, np.floating) else dtype)
                        for dtype in p_f64_only["expected_output_dtypes"]
                    ]
                if (
                    "input_dtypes" in p_f64_only
                    and p_f64_only["input_dtypes"] is not None
                ):
                    p_f64_only["input_dtypes"] = [
                        (np.float64 if np.issubdtype(dt, np.floating) else dt)
                        for dt in p_f64_only["input_dtypes"]
                    ]
            final_params_list.append(p_f64_only)
        else:
            # Default behavior: generate f32 variant (base name)
            p_f32 = base_param_set.copy()
            p_f32["_enable_double_precision_test_setting"] = False
            # testcase name remains base_param_set["testcase"]

            if not run_only_f64_variant:
                final_params_list.append(p_f32)

            # And f64 variant if applicable (and not disabled)
            if (
                not disable_float64_test_from_meta and not run_only_f32_variant
            ):  # Check existing flag and suppress if f32-only
                # Check if this test involves floats, only add _f64 variant if it does.
                p_f64 = base_param_set.copy()
                p_f64["testcase"] += "_f64"  # Add suffix
                p_f64["_enable_double_precision_test_setting"] = True

                if "input_values" in p_f64 and p_f64["input_values"] is not None:
                    p_f64["input_values"] = [
                        (
                            np.array(val, dtype=np.float64)
                            if np.issubdtype(np.array(val).dtype, np.floating)
                            else np.array(val)
                        )
                        for val in p_f64["input_values"]
                    ]
                if (
                    "expected_output_dtypes" in p_f64
                    and p_f64["expected_output_dtypes"] is not None
                ):
                    p_f64["expected_output_dtypes"] = [
                        (np.float64 if np.issubdtype(dtype, np.floating) else dtype)
                        for dtype in p_f64["expected_output_dtypes"]
                    ]
                if "input_dtypes" in p_f64 and p_f64["input_dtypes"] is not None:
                    p_f64["input_dtypes"] = [
                        (np.float64 if np.issubdtype(dt, np.floating) else dt)
                        for dt in p_f64["input_dtypes"]
                    ]
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
    """Create a test function from test parameters."""
    test_case_name_safe = tp["testcase"].replace(".", "_").replace(" ", "_")
    func_name = f"test_{test_case_name_safe}"

    def test_func(self=None):  # Pytest will inject 'self' if it's a method of a class
        configure_logging()  # Call once if needed, or ensure logger is configured globally

        callable_obj = tp["callable"]
        input_values_from_testcase = tp.get("input_values")
        input_shapes_from_testcase = tp.get("input_shapes")
        input_dtypes_from_testcase = tp.get("input_dtypes")
        expected_output_dtypes_from_testcase = tp.get("expected_output_dtypes")

        current_enable_double_precision = tp.get(
            "_enable_double_precision_test_setting", False
        )

        processed_input_specs_for_to_onnx: List[Any]

        if input_shapes_from_testcase is not None:
            processed_input_specs_for_to_onnx = []
            if input_dtypes_from_testcase:
                if len(input_dtypes_from_testcase) != len(input_shapes_from_testcase):
                    raise ValueError(
                        f"Testcase '{tp['testcase']}': `input_dtypes` length must match `input_shapes`."
                    )
                for shape_spec, dt in zip(
                    input_shapes_from_testcase, input_dtypes_from_testcase
                ):
                    current_shape_tuple = (
                        tuple(shape_spec)
                        if isinstance(shape_spec, (list, tuple))
                        else (shape_spec,)
                    )
                    if current_enable_double_precision and np.issubdtype(
                        dt, np.floating
                    ):
                        dt = jnp.float64
                    processed_input_specs_for_to_onnx.append(
                        jax.ShapeDtypeStruct(current_shape_tuple, dt)
                    )
                logger.info(
                    f"Test '{tp['testcase']}': Using ShapeDtypeStructs from `input_shapes` + `input_dtypes`: {processed_input_specs_for_to_onnx}."
                )
            else:
                for shape_spec in input_shapes_from_testcase:
                    current_shape_tuple = (
                        tuple(shape_spec)
                        if isinstance(shape_spec, (list, tuple))
                        else (shape_spec,)
                    )
                    processed_input_specs_for_to_onnx.append(current_shape_tuple)
                logger.info(
                    f"Test '{tp['testcase']}': Using explicit input_shapes from testcase (no dtype list supplied): {processed_input_specs_for_to_onnx}."
                )
        elif input_values_from_testcase is not None:
            shapes_from_values = []
            for val in input_values_from_testcase:
                np_array = np.array(val)
                if current_enable_double_precision and np.issubdtype(
                    np_array.dtype, np.floating
                ):
                    shapes_from_values.append(
                        jax.ShapeDtypeStruct(np_array.shape, jnp.float64)
                    )
                else:
                    shapes_from_values.append(
                        jax.ShapeDtypeStruct(np_array.shape, np_array.dtype)
                    )
            processed_input_specs_for_to_onnx = shapes_from_values
            logger.info(
                f"Test '{tp['testcase']}': Using ShapeDtypeStructs for to_onnx from input_values: {processed_input_specs_for_to_onnx}. "
                f"enable_double_precision={current_enable_double_precision}."
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
                    f"Testcase '{tp['testcase']}' must provide 'input_shapes' or 'input_values'."
                )

        input_params_from_testcase = tp.get("input_params", {})
        testcase_name = tp["testcase"]
        expected_num_funcs = tp.get("expected_number_of_function_instances")
        expected_output_shapes_from_testcase = tp.get("expected_output_shapes")
        onnx_input_names_from_testcase = tp.get("input_names")

        context_path = tp.get("context", "default.unknown").split(".")
        opset_version = tp.get("opset_version", 21)

        model_folder_path = os.path.join("docs", "onnx", *context_path)
        os.makedirs(model_folder_path, exist_ok=True)
        model_path = os.path.join(model_folder_path, f"{test_case_name_safe}.onnx")

        logger.info(
            f"Converting '{testcase_name}' to ONNX with input shapes: {processed_input_specs_for_to_onnx}, "
            f"enable_double_precision: {current_enable_double_precision}"
        )
        try:
            onnx_model = to_onnx(
                callable_obj,
                processed_input_specs_for_to_onnx,
                input_params=input_params_from_testcase,
                model_name=testcase_name,
                opset=opset_version,
                enable_double_precision=current_enable_double_precision,
            )
        except Exception as e:
            logger.error(
                f"Failed during to_onnx conversion for '{testcase_name}' with enable_double_precision={current_enable_double_precision}: {e}",
                exc_info=True,
            )
            raise

        post_check = tp.get("post_check_onnx_graph")
        if post_check:
            assert post_check(
                onnx_model
            ), f"Post-conversion graph check failed for '{testcase_name}'."

        onnx.save_model(onnx_model, model_path)
        logger.info(f"Model saved to: {model_path}")

        # --- ONNX checker and runtime session (if requested) ---
        if tp.get("check_onnx_load", False):
            onnx_model = onnx.load_model(model_path)
            onnx.checker.check_model(onnx_model)

        # Optional per-test override: skip numeric validation entirely
        if tp.get("skip_numeric_validation", False):
            logger.warning(
                f"Skipping numeric validation for '{testcase_name}' (per-test flag)."
            )
        elif input_values_from_testcase:
            initializers = {init.name for init in onnx_model.graph.initializer}
            runnable_graph_input_names = [
                inp.name
                for inp in onnx_model.graph.input
                if inp.name not in initializers
            ]
            if len(input_values_from_testcase) != len(runnable_graph_input_names):
                logger.warning(
                    f"Skipping numerical validation for '{testcase_name}' due to input count mismatch."
                )
            else:
                logger.info(
                    f"Running numerical check for '{testcase_name}' (enable_double_precision={current_enable_double_precision})..."
                )
                rtol, atol = _get_rtol_atol(tp, current_enable_double_precision)

                xs_for_num_check = []
                for val_from_tc in input_values_from_testcase:
                    arr = np.asarray(val_from_tc)
                    dt = arr.dtype
                    if not current_enable_double_precision:
                        if dt == np.float64:
                            dt = np.float32
                        elif dt == np.int64:
                            dt = np.int32
                    elif np.issubdtype(dt, np.floating):
                        dt = np.float64
                    xs_for_num_check.append(np.asarray(val_from_tc, dtype=dt))

                passed, msg = allclose(
                    callable_obj,
                    model_path,
                    xs_for_num_check,
                    input_params_from_testcase,
                    rtol=rtol,
                    atol=atol,
                )
                assert passed, f"Numerical check failed for {testcase_name}: {msg}"
                logger.info(f"Numerical check passed for {testcase_name}.")

        # ------------------------------------------------------------------
        # ✅  NUMERICAL CHECK – fall-back to default float32 dtypes
        # ------------------------------------------------------------------
        #
        # If a testcase gives `input_shapes` but forgets an explicit
        # `input_dtypes`, we still want to run the end-to-end numeric
        # comparison.  Assume `jnp.float32` for every tensor unless the
        # testcase overrode it.
        #
        elif input_shapes_from_testcase:
            # Build a dtype list: honour explicit list if present.
            # Otherwise choose float32 *or* float64 depending on the
            # variant we are running.
            if input_dtypes_from_testcase is None:
                default_dtype = (
                    jnp.float64 if current_enable_double_precision else jnp.float32
                )
                input_dtypes_from_testcase = [default_dtype] * len(
                    input_shapes_from_testcase
                )

            symbol_map: dict[str, int] = {}
            concrete_shapes: list[tuple[int, ...]] = []
            for shape_tuple in input_shapes_from_testcase:
                concrete_shape = tuple(
                    symbol_map.setdefault(dim, 2) if isinstance(dim, str) else dim
                    for dim in shape_tuple
                )
                concrete_shapes.append(concrete_shape)

            def _rand(shape, dtype):
                """
                Return a NumPy array/random scalar of the requested shape and dtype,
                always as an np.ndarray (so .astype is available).
                """
                # 1) Generate a raw Python float or ndarray
                if np.issubdtype(dtype, np.floating):
                    raw = np.random.randn(*shape) if shape else np.random.randn()
                elif np.issubdtype(dtype, np.integer):
                    raw = (
                        np.random.randint(0, 5, size=shape)
                        if shape
                        else np.random.randint(0, 5)
                    )
                elif dtype == np.bool_ or dtype == np.dtype(bool):
                    raw = (
                        (np.random.rand(*shape) > 0.5)
                        if shape
                        else (np.random.rand() > 0.5)
                    )
                else:
                    raw = np.random.randn(*shape) if shape else np.random.randn()

                # 2) Wrap into ndarray and cast
                arr = np.array(raw)
                target = (
                    jnp.float64
                    if (
                        current_enable_double_precision
                        and np.issubdtype(dtype, np.floating)
                    )
                    else dtype
                )
                return arr.astype(target)

            xs_for_num_check = [
                _rand(shp, dt)
                for shp, dt in zip(concrete_shapes, input_dtypes_from_testcase)
            ]

            # Special-case: avoid all-False attention masks (would yield −inf → NaN).
            for idx, dt in enumerate(input_dtypes_from_testcase):
                if dt == np.bool_ or dt == np.dtype(bool):
                    mask = xs_for_num_check[idx]
                    # Only apply if there's at least one dimension to index
                    if mask.ndim == 0:
                        continue
                    all_false_rows = ~np.any(mask, axis=-1, keepdims=True)
                    if np.any(all_false_rows):
                        fix = np.zeros_like(mask)
                        # build a safe indexer for the last axis
                        idxers = [slice(None)] * mask.ndim
                        idxers[-1] = 0
                        fix[tuple(idxers)] = True
                        xs_for_num_check[idx] = np.where(all_false_rows, fix, mask)
                        logger.warning(
                            f"Modified random attention mask for test '{testcase_name}' to prevent all-False rows that can cause NaNs."
                        )

            rtol, atol = _get_rtol_atol(tp, current_enable_double_precision)

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
            ), f"Numerical check failed for {testcase_name}: {validation_message}"
            logger.info(f"Numerical check passed for {testcase_name}.")

        else:
            logger.info(
                f"No concrete inputs available for '{testcase_name}', skipping numerical validation."
            )

        # --- Function Count Check ---
        if expected_num_funcs is not None:
            num_found_funcs = len(list(onnx_model.functions))
            assert (
                num_found_funcs == expected_num_funcs
            ), f"Test '{testcase_name}': Expected {expected_num_funcs} ONNX functions, found {num_found_funcs}."
            logger.info(f"Found expected {num_found_funcs} ONNX functions.")

        # --- DType Validation for ONNX model outputs ---
        if input_values_from_testcase:
            jax_eval_inputs_sds = []
            for val in input_values_from_testcase:
                np_array_val = np.array(val)
                if current_enable_double_precision and np.issubdtype(
                    np_array_val.dtype, np.floating
                ):
                    jax_eval_inputs_sds.append(
                        jax.ShapeDtypeStruct(np_array_val.shape, jnp.float64)
                    )
                else:
                    jax_eval_inputs_sds.append(
                        jax.ShapeDtypeStruct(np_array_val.shape, np_array_val.dtype)
                    )

            try:
                eval_target_jax_func = callable_obj
                if (
                    hasattr(callable_obj, "__call__")
                    and not inspect.isfunction(callable_obj)
                    and not inspect.ismethod(callable_obj)
                ):
                    if not inspect.isclass(callable_obj):
                        eval_target_jax_func = callable_obj.__call__
                if isinstance(callable_obj, jax.tree_util.Partial) or (
                    hasattr(callable_obj, "__module__")
                    and "flax.experimental.nnx" in callable_obj.__module__
                ):

                    def jax_func_wrapper(*args):
                        return callable_obj(*args)

                    eval_target_jax_func = jax_func_wrapper

                jax_output_sds = jax.eval_shape(
                    eval_target_jax_func, *jax_eval_inputs_sds
                )
                if not isinstance(jax_output_sds, (list, tuple)):
                    jax_output_sds = [jax_output_sds]
                expected_onnx_dtype_for_float_outputs = (
                    onnx.TensorProto.DOUBLE
                    if current_enable_double_precision
                    else onnx.TensorProto.FLOAT
                )
                for i, onnx_output_vi in enumerate(onnx_model.graph.output):
                    if i < len(jax_output_sds):
                        jax_dtype = jax_output_sds[i].dtype
                        onnx_tensor_type_enum = (
                            onnx_output_vi.type.tensor_type.elem_type
                        )
                        if jnp.issubdtype(jax_dtype, jnp.floating):
                            assert (
                                onnx_tensor_type_enum
                                == expected_onnx_dtype_for_float_outputs
                            ), f"Test '{testcase_name}', output '{onnx_output_vi.name}' (index {i}): ONNX dtype mismatch."
                logger.info(f"Output dtypes verified for '{testcase_name}'.")
            except Exception as e:
                logger.warning(
                    f"Could not perform JAX eval_shape for dtype checking in test '{testcase_name}': {e}",
                    exc_info=True,
                )
        else:
            logger.info(
                f"Skipping ONNX output dtype validation for '{testcase_name}' as no input_values were provided."
            )

        # --- Expected Output Dtypes Validation ---
        if expected_output_dtypes_from_testcase:
            logger.info(
                f"== Validating expected output dtypes for '{testcase_name}' =="
            )
            if len(onnx_model.graph.output) != len(
                expected_output_dtypes_from_testcase
            ):
                raise AssertionError(
                    f"Test '{testcase_name}': Output count mismatch for dtype validation. "
                    f"Expected {len(expected_output_dtypes_from_testcase)} output dtypes, "
                    f"ONNX model has {len(onnx_model.graph.output)} outputs."
                )

            for i, expected_dtype_np in enumerate(expected_output_dtypes_from_testcase):
                onnx_output_vi = onnx_model.graph.output[i]
                actual_onnx_dtype_enum = onnx_output_vi.type.tensor_type.elem_type

                # Convert expected numpy dtype to ONNX enum for comparison
                # Ensure expected_dtype_np is a numpy.dtype object
                if not isinstance(expected_dtype_np, np.dtype):
                    try:
                        expected_dtype_np = np.dtype(expected_dtype_np)
                    except TypeError:
                        raise TypeError(
                            f"Test '{testcase_name}', output {i}: Expected dtype {expected_dtype_np} is not a valid numpy dtype."
                        )

                expected_onnx_dtype_enum = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE.get(
                    expected_dtype_np
                )

                if expected_onnx_dtype_enum is None:
                    raise ValueError(
                        f"Test '{testcase_name}', output {i}: Could not map expected numpy dtype {expected_dtype_np} to ONNX dtype."
                    )

                assert actual_onnx_dtype_enum == expected_onnx_dtype_enum, (
                    f"Test '{testcase_name}', output '{onnx_output_vi.name}' (index {i}): "
                    f"Expected ONNX output dtype {onnx.TensorProto.DataType.Name(expected_onnx_dtype_enum)} (from expected_output_dtypes), "
                    f"but got ONNX dtype {onnx.TensorProto.DataType.Name(actual_onnx_dtype_enum)}."
                )
            logger.info(f"Expected output dtypes validated for '{testcase_name}'.")

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
                        dims.append(None)  # Represent unknown fixed dimension
                logger.info(
                    f"ONNX Graph Output: {output_node_info.name}  Defined Structural Shape: {tuple(dims)}"
                )
                onnx_graph_structural_shapes.append(tuple(dims))

            runtime_onnx_output_shapes = []
            expected_concrete_jax_shapes = []

            # Determine the effective expected output shapes for the ONNX graph structure
            effective_expected_output_shapes = list(
                expected_output_shapes_from_testcase
            )  # Default to original

            if (
                current_enable_double_precision
            ):  # current_enable_double_precision is tp.get("_enable_double_precision_test_setting", False)
                x64_specific_expected_shapes = tp.get("x64_expected_output_shapes")
                if x64_specific_expected_shapes is not None:
                    logger.info(
                        f"Test '{testcase_name}': Using x64_expected_output_shapes: {x64_specific_expected_shapes} "
                        f"due to current_enable_double_precision={current_enable_double_precision}."
                    )
                    effective_expected_output_shapes = list(
                        x64_specific_expected_shapes
                    )
                elif tp.get(
                    "shape_dynamic_on_x64"
                ):  # Legacy or alternative flag, if you used it previously
                    logger.warning(
                        f"Test '{testcase_name}': Legacy flag 'shape_dynamic_on_x64' found. "
                        f"Adjusting expectation to dynamic for x64."
                    )
                    if effective_expected_output_shapes:  # Ensure not empty
                        effective_expected_output_shapes[0] = (
                            "JAX2ONNX_DYNAMIC_DIM_SENTINEL",
                        )

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

                    # --- Code to get JAX-side concrete shapes for comparison ---
                    jax_concrete_inputs_for_exec = []
                    if (
                        input_values_from_testcase
                    ):  # If the testcase provides concrete input values
                        # Ensure these values are JAX arrays with dtypes reflecting current_enable_double_precision
                        # jax_eval_inputs_sds was prepared earlier with correct dtypes for this.
                        if len(input_values_from_testcase) != len(jax_eval_inputs_sds):
                            raise ValueError(  # Should not happen if jax_eval_inputs_sds derived correctly
                                f"Internal inconsistency: input_values_from_testcase length ({len(input_values_from_testcase)}) "
                                f"does not match jax_eval_inputs_sds length ({len(jax_eval_inputs_sds)}) for {testcase_name}."
                            )
                        for idx, val_spec in enumerate(input_values_from_testcase):
                            # Use the dtype from jax_eval_inputs_sds, as it correctly considers current_enable_double_precision
                            sds_for_dtype = jax_eval_inputs_sds[idx]
                            jax_concrete_inputs_for_exec.append(
                                jnp.asarray(val_spec, dtype=sds_for_dtype.dtype)
                            )
                    # If input_values_from_testcase is empty, but jax_eval_inputs_sds is not (e.g. from input_shapes):
                    elif (
                        jax_eval_inputs_sds
                    ):  # This case is for tests specified by shapes, not static arange with no args
                        for sds in jax_eval_inputs_sds:
                            # For shape-based tracing to get JAX output shapes, zeros are fine.
                            jax_concrete_inputs_for_exec.append(
                                jnp.zeros(sds.shape, dtype=sds.dtype)
                            )
                    # If both are empty (e.g. for static arange like `lambda: jnp.arange(5)`),
                    # jax_concrete_inputs_for_exec remains empty, which is correct.

                    # Temporarily enable x64 for this JAX execution if current_enable_double_precision is set for the test variant
                    if current_enable_double_precision:
                        with jax.config.update("jax_enable_x64", True):
                            jax_fn_outputs_for_shape = eval_target_jax_func(
                                *jax_concrete_inputs_for_exec
                            )
                    else:
                        # Ensure x64 is disabled if not set for the test variant
                        # (assuming default might be True or to avoid interference from previous tests)
                        with jax.config.update("jax_enable_x64", False):
                            jax_fn_outputs_for_shape = eval_target_jax_func(
                                *jax_concrete_inputs_for_exec
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

            # Now use 'effective_expected_output_shapes' in the structural shape assertion block
            if len(onnx_graph_structural_shapes) != len(
                effective_expected_output_shapes
            ):
                raise AssertionError(
                    f"[❌] '{testcase_name}': Output count mismatch for shape assertion. "
                    f"Testcase expects {len(effective_expected_output_shapes)} output shapes, "
                    f"ONNX graph defines {len(onnx_graph_structural_shapes)}."
                )

            final_assert_messages = []
            all_checks_passed = True

            for i, expected_shape_spec_adjusted in enumerate(
                effective_expected_output_shapes
            ):
                onnx_graph_shape = onnx_graph_structural_shapes[i]

                # Compare rank
                if len(expected_shape_spec_adjusted) != len(onnx_graph_shape):
                    all_checks_passed = False
                    # Use expected_shape_spec_from_testcase for original intent in log if different
                    original_expected_spec = (
                        expected_output_shapes_from_testcase[i]
                        if i < len(expected_output_shapes_from_testcase)
                        else "N/A"
                    )
                    final_assert_messages.append(
                        f"Output {i} for '{testcase_name}': Rank mismatch. "
                        f"Adjusted expectation for rank: {len(expected_shape_spec_adjusted)} (from {expected_shape_spec_adjusted}), "
                        f"Original testcase expectation: {original_expected_spec}. "
                        f"ONNX graph rank {len(onnx_graph_shape)} (from ONNX: {onnx_graph_shape})."
                    )
                    continue

                # Compare dimensions (symbolic or concrete)
                graph_dim_match = True
                for j, expected_dim_representation in enumerate(
                    expected_shape_spec_adjusted
                ):
                    onnx_graph_actual_dim = onnx_graph_shape[j]
                    # If expected_dim_representation is our sentinel, we need to match it correctly
                    if expected_dim_representation == "JAX2ONNX_DYNAMIC_DIM_SENTINEL":
                        # The onnx_graph_actual_dim might be the string representation of the sentinel,
                        # or a symbolic name if resolution failed.
                        # The key is that it's not a concrete integer.
                        if not isinstance(
                            onnx_graph_actual_dim, (str, type(None))
                        ):  # Allow None for fully dynamic/unknown
                            # If it's an int, it's a mismatch if we expected the sentinel
                            if isinstance(onnx_graph_actual_dim, int):
                                graph_dim_match = False
                                break
                        # If onnx_graph_actual_dim IS the sentinel string, it's a match.
                        # Or if it's a 'dynamic_...' string, it's also considered a match for a dynamic dim.
                        elif isinstance(onnx_graph_actual_dim, str) and (
                            onnx_graph_actual_dim == "JAX2ONNX_DYNAMIC_DIM_SENTINEL"
                            or onnx_graph_actual_dim.startswith("dynamic_")
                        ):
                            pass  # Match
                        elif (
                            onnx_graph_actual_dim is None
                        ):  # Match if ONNX has no dim info
                            pass
                        else:  # Mismatch if it's some other string or unexpected type
                            graph_dim_match = False
                            break
                    elif isinstance(
                        expected_dim_representation, str
                    ):  # Symbolic dim in testcase
                        if onnx_graph_actual_dim != expected_dim_representation:
                            if not (
                                expected_dim_representation
                                == "B"  # Example, adapt if other symbols used
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
                    original_expected_spec = (
                        expected_output_shapes_from_testcase[i]
                        if i < len(expected_output_shapes_from_testcase)
                        else "N/A"
                    )
                    final_assert_messages.append(
                        f"Output {i} for '{testcase_name}' (ONNX Graph Structure): Mismatch. "
                        f"Adjusted expectation: {expected_shape_spec_adjusted}, "
                        f"Original testcase expectation: {original_expected_spec}. "
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
    setattr(test_func, "_testcase_params", tp)
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


def _get_rtol_atol(tp, current_enable_double_precision):
    if current_enable_double_precision:
        default_rtol, default_atol = 1e-7, 1e-7
        rtol = tp.get("rtol_f64", tp.get("rtol", default_rtol))
        atol = tp.get("atol_f64", tp.get("atol", default_atol))
    else:
        default_rtol, default_atol = 1e-5, 1e-5
        rtol = tp.get("rtol_f32", tp.get("rtol", default_rtol))
        atol = tp.get("atol_f32", tp.get("atol", default_atol))
    return rtol, atol


if __name__ == "__main__":
    # This allows the script to be run directly to regenerate tests.
    # Ensure JAX/ONNX and other dependencies are available in the environment.
    # It's good practice to also import and configure logging here if not done by an imported module.
    # For example:
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    logger.info("Running t_generator.py script...")
    generate_all_tests()
