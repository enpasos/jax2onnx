import jax
import jax.numpy as jnp
import numpy as np
import onnx
import pytest
from onnx import TensorProto

from jax2onnx import to_onnx, allclose

# A constant defined in the global scope, with a float32 dtype.
# This simulates a constant captured from a closure, like the jxf_buffers
# in the user's example.
CAPTURED_CONSTANT_F32 = np.array([1.5, 2.5, 3.5], dtype=np.float32)


def function_with_captured_constant(x):
    """A JAX function that adds a captured f32 constant to its input."""
    # When this function is traced, CAPTURED_CONSTANT_F32 becomes a jaxpr constant.
    return x + jnp.array(CAPTURED_CONSTANT_F32)


@pytest.mark.order(-1)  # run *after* the models have been produced
def test_f64_promotion_for_captured_constants():
    """
    Tests that constants captured from a function's closure are correctly
    promoted to float64 when `enable_double_precision=True`.
    This replicates the scenario where a function with no inputs captures
    buffers from its environment.
    """
    # --- JAX Execution (with x64 enabled for baseline) ---
    jax.config.update("jax_enable_x64", True)

    input_f64 = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    # The JAX execution should promote the f32 constant to f64 for the addition.
    jax_result = function_with_captured_constant(input_f64)

    assert (
        jax_result.dtype == jnp.float64
    ), f"JAX output dtype should be float64, but got {jax_result.dtype}"

    # --- ONNX Conversion with enable_double_precision=True ---
    onnx_model = to_onnx(
        function_with_captured_constant,
        [jax.ShapeDtypeStruct(input_f64.shape, input_f64.dtype)],
        model_name="f64_constant_promotion_test",
        enable_double_precision=True,
    )

    # --- Validation ---
    # 1. Check the initializer dtype in the ONNX graph
    found_initializer = False
    for initializer in onnx_model.graph.initializer:
        # Check if this initializer corresponds to our constant
        # Note: The name is generated, so we check the value.
        values = onnx.numpy_helper.to_array(initializer)
        if np.allclose(values, CAPTURED_CONSTANT_F32.astype(np.float64)):
            assert initializer.data_type == TensorProto.DOUBLE, (
                f"Initializer '{initializer.name}' should have been promoted to DOUBLE (11), "
                f"but has dtype {TensorProto.DataType.Name(initializer.data_type)} ({initializer.data_type})."
            )
            found_initializer = True
            break

    assert (
        found_initializer
    ), "Could not find the initializer corresponding to the captured constant."

    # 2. Numerical check using the test generator's `allclose`
    model_path = "f64_constant_promotion_test.onnx"
    with open(model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    passed_numerical, validation_message = allclose(
        function_with_captured_constant,
        model_path,
        [input_f64],
        {},
        rtol=1e-7,
        atol=1e-7,
    )

    assert passed_numerical, f"Numerical check failed: {validation_message}"


# Another test for a function with NO inputs, to be even closer to the user's case
def function_with_only_captured_constants():
    """A JAX function that works only on captured constants."""
    # FIX: Multiply by a float64 constant to ensure promotion
    return jnp.array(CAPTURED_CONSTANT_F32) * np.float64(2.0)


@pytest.mark.order(-1)  # run *after* the models have been produced
def test_f64_promotion_for_function_with_no_inputs():
    """
    Tests that constants are promoted correctly for a function with no inputs,
    which is very similar to the user's `do_integration_step_fn`.
    """
    # --- JAX Execution (with x64 enabled for baseline) ---
    jax.config.update("jax_enable_x64", True)
    jax_result = function_with_only_captured_constants()
    assert jax_result.dtype == jnp.float64

    # --- ONNX Conversion ---
    onnx_model = to_onnx(
        function_with_only_captured_constants,
        [],  # No inputs
        model_name="f64_no_inputs_test",
        enable_double_precision=True,
    )

    # --- Validation ---
    # 1. Check initializer dtype
    assert len(onnx_model.graph.initializer) > 0, "Model should have initializers"
    initializer = onnx_model.graph.initializer[0]
    assert (
        initializer.data_type == TensorProto.DOUBLE
    ), f"Initializer should be DOUBLE, but got {TensorProto.DataType.Name(initializer.data_type)}"

    # 2. Numerical Check
    model_path = "f64_no_inputs_test.onnx"
    with open(model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    passed_numerical, validation_message = allclose(
        function_with_only_captured_constants,
        model_path,
        [],  # No inputs for runtime either
        {},
        rtol=1e-7,
        atol=1e-7,
    )

    assert (
        passed_numerical
    ), f"Numerical check failed for no-input function: {validation_message}"
