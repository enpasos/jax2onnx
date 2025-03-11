# jax2onnx/tests/test_plugins.py
import os
import importlib.util
import pytest
import jax
import numpy as np
from jax2onnx import save_onnx, allclose


def load_plugin_metadata() -> list:
    # (Your original implementation here)
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
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, "get_metadata"):
                    md = module.get_metadata()
                    if isinstance(md, dict):
                        if "testcases" in md and isinstance(md["testcases"], list):
                            for testcase in md["testcases"]:
                                if isinstance(testcase, dict):
                                    testcase["source"] = module_name
                                    # Pass the context along (e.g. "plugins.nnx")
                                    testcase["context"] = md.get("context", "default")
                                    metadata_list.append(testcase)
    return metadata_list


def generate_test_params(metadata_entry: dict) -> list:
    params = []
    base = metadata_entry.copy()
    input_shapes = base.get("input_shapes", [])
    if isinstance(input_shapes, list) and all(
        isinstance(t, (tuple, list)) for t in input_shapes
    ):
        has_symbolic = any(
            any(isinstance(dim, str) and dim == "B" for dim in tup)
            for tup in input_shapes
        )
        if has_symbolic:
            dynamic_variant = base.copy()
            dynamic_variant["testcase"] = f"{base.get('testcase', 'unknown')}_dynamic"
            concrete_variant = base.copy()
            concrete_variant["input_shapes"] = [
                tuple(
                    3 if (isinstance(dim, str) and dim == "B") else dim for dim in tup
                )
                for tup in input_shapes
            ]
            concrete_variant["testcase"] = f"{base.get('testcase', 'unknown')}_concrete"
            params.extend([dynamic_variant, concrete_variant])
        else:
            params.append(base)
    else:
        params.append(base)
    return params


def load_all_test_params() -> list:
    all_params = []
    for md in load_plugin_metadata():
        for param in generate_test_params(md):
            all_params.append(param)
    return all_params


# Group tests by context.
def get_tests_by_context():
    tests_by_context = {}
    for param in load_all_test_params():
        context = param.get("context", "default")
        tests_by_context.setdefault(context, []).append(param)
    return tests_by_context


# Dynamically create a test class for each context.
for context, tests in get_tests_by_context().items():
    # For a cleaner name, use just the last part of the context (e.g. "nnx")
    cls_name = f"Test_{context.split('.')[-1]}"
    attrs = {}

    # Create a test function for each test parameter.
    for test_param in tests:
        # Create a safe test name based on the testcase name.
        test_name = f"test_{test_param.get('testcase', 'unknown')}"
        # Ensure unique names if needed.
        test_name = test_name.replace(" ", "_").replace(".", "_")

        def make_test_function(tp):
            def test_func(self, tp=tp):
                callable_obj = tp.get("callable", None)
                if callable_obj is None:
                    pytest.skip("No callable defined in metadata.")
                if hasattr(callable_obj, "eval"):
                    callable_obj.eval()

                input_shapes = tp.get("input_shapes", [])
                rng = jax.random.PRNGKey(1001)

                # Determine a valid model path for the ONNX model.
                testcase_name = tp.get("testcase", "unknown")
                onnx_model_file_name = f"{testcase_name}.onnx"
                model_path = os.path.join("docs", "onnx", onnx_model_file_name)
                os.makedirs(os.path.dirname(model_path), exist_ok=True)

                # Convert the JAX model to an ONNX model.
                save_onnx(
                    callable_obj,
                    input_shapes,
                    model_path,
                    include_intermediate_shapes=True,
                )

                # Execute the test based on input shapes.
                if testcase_name.endswith("_dynamic"):
                    for concrete_value in [2, 3]:
                        processed_input_shapes = [
                            [
                                (
                                    concrete_value
                                    if isinstance(dim, str) and dim == "B"
                                    else dim
                                )
                                for dim in shape
                            ]
                            for shape in input_shapes
                        ]
                        xs = [
                            jax.random.normal(rng, tuple(shape))
                            for shape in processed_input_shapes
                        ]
                        np.testing.assert_(allclose(callable_obj, model_path, *xs))
                else:
                    xs = [jax.random.normal(rng, shape) for shape in input_shapes]
                    np.testing.assert_(allclose(callable_obj, model_path, *xs))
                print(f"Test for {testcase_name} passed!")

            return test_func

        # Bind the test function to the class attributes.
        attrs[test_name] = make_test_function(test_param)

    # Dynamically create the class and add it to the module's globals.
    globals()[cls_name] = type(cls_name, (object,), attrs)
