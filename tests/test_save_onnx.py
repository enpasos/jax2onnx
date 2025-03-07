# jax2onnx/tests/test_save_onnx.py

import os
import importlib.util
import pytest
import jax
import onnxruntime as ort
import numpy as np
from jax2onnx import save_onnx, allclose


def load_plugin_metadata() -> list:
    """Walk through the plugins directory and load metadata from each plugin.

    Each plugin should define a get_metadata() function that returns a dictionary
    with a "testcases" key (a list of dicts), each representing one testcase.
    """
    metadata_list = []
    plugins_dir = os.path.join(
        os.path.dirname(__file__), "../jax2onnx/converter/plugins"
    )

    for root, _, files in os.walk(plugins_dir):
        for file in files:
            # Skip __init__.py and registry/interface files
            if file.endswith(".py") and file not in [
                "__init__.py",
                "plugin_interface.py",
                "plugin_registry.py",
                "plugin_registry_static.py",
            ]:
                module_path = os.path.join(root, file)
                # Create a module name based on the file path
                module_name = module_path.replace(os.sep, ".").replace(".py", "")

                spec = importlib.util.spec_from_file_location(module_name, module_path)
                if spec is None or spec.loader is None:
                    continue  # Could not load this module, skip it
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                if hasattr(module, "get_metadata"):
                    md = module.get_metadata()
                    if isinstance(md, dict):
                        # If metadata contains a "testcases" key, flatten them.
                        if "testcases" in md and isinstance(md["testcases"], list):
                            for testcase in md["testcases"]:
                                if isinstance(testcase, dict):
                                    testcase["source"] = module_name
                                    metadata_list.append(testcase)
    return metadata_list


def generate_test_params(metadata_entry: dict) -> list:
    """
    Given a metadata entry, inspect its "input_shapes" attribute (expected to be a list
    of tuples). If any tuple contains the string 'B', generate two variants:
      - One where input_shapes is unchanged, with '_dynamic' added to the testcase name.
      - One where each occurrence of 'B' is replaced with the concrete value 3, with '_concrete' added.
    If no symbolic 'B' is found, return the original testcase.
    """
    params = []
    base = metadata_entry.copy()
    input_shapes = base.get("input_shapes", [])

    # Verify input_shapes is a list of tuples (or lists)
    if isinstance(input_shapes, list) and all(
        isinstance(t, (tuple, list)) for t in input_shapes
    ):
        # Check if any tuple contains the string 'B'
        has_symbolic = any(
            any(isinstance(dim, str) and dim == "B" for dim in tup)
            for tup in input_shapes
        )
        if has_symbolic:
            # Create dynamic variant: keep input_shapes as is
            dynamic_variant = base.copy()
            dynamic_variant["testcase"] = f"{base.get('testcase', 'unknown')}_dynamic"

            # Create concrete variant: replace each 'B' with the int 3
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
    """Load and expand metadata from all plugins into a list of test parameters."""
    all_params = []
    for md in load_plugin_metadata():
        # Flatten the metadata into variants based on input_shapes.
        for param in generate_test_params(md):
            # Wrap in pytest.param to create a unique id based on the testcase and source.
            all_params.append(
                pytest.param(
                    param,
                    id=f"{param.get('testcase', 'unknown')} ({param.get('source', 'plugin')})",
                )
            )
    return all_params


@pytest.mark.parametrize("test_params", load_all_test_params())
def test_onnx_export_from_metadata(test_params: dict) -> None:
    """Generate a test case based on the metadata."""
    callable = test_params.get("callable", None)
    if callable is None:
        pytest.skip("No callable (nnx.module or function) defined in metadata.")

    # Prepare the component if needed (e.g., call eval() for modules)
    if hasattr(callable, "eval"):
        callable.eval()

    input_shapes = test_params.get("input_shapes", [])
    params = test_params.get("params", {})
    internal_shape_info = test_params.get("internal_shape_info", True)
    testcase_name = test_params.get("testcase", "unknown")

    # Set up a reproducible random seed
    seed = 1001
    rng = jax.random.PRNGKey(seed)

    # Determine output file name and model path (e.g., in a docs folder)
    onnx_model_file_name = f"{testcase_name}.onnx"
    model_path = os.path.join("docs", "onnx", onnx_model_file_name)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Convert the JAX model to an ONNX model
    save_onnx(callable, input_shapes, model_path, include_intermediate_shapes=True)

    # Load the exported ONNX model
    ort_session = ort.InferenceSession(model_path)

    # Process input shapes: replace any symbolic dims (like 'B') with a concrete value (e.g., 2)
    # for dummy input generation
    processed_input_shapes = []
    for shape in input_shapes:
        processed_shape = [
            2 if isinstance(dim, str) and dim == "B" else dim for dim in shape
        ]
        processed_input_shapes.append(processed_shape)

    # Create dummy inputs using JAX random data
    xs = [jax.random.normal(rng, shape) for shape in processed_input_shapes]

    # Use the provided helper to check outputs
    np.testing.assert_(allclose(callable, model_path, *xs))

    print(f"Test for {testcase_name} passed!")

    # def test_example():
    # seed = 1001

    # fn = nnx.LinearGeneral(
    #     in_features=(8, 32), out_features=(256,), axis=(-2, -1), rngs=nnx.Rngs(seed)
    # )

    # dir = "docs/onnx"
    # os.makedirs(dir, exist_ok=True)

    # model_path = dir + "/example5.onnx"

    # save_onnx(fn, [("B", 4, 8, 32)], model_path, include_intermediate_shapes=True)

    # rng = jax.random.PRNGKey(seed)
    # example_batch_size = 2
    # x = jax.random.normal(rng, (example_batch_size, 4, 8, 32))

    # # Verify outputs match
    # assert allclose(fn, model_path, x)
