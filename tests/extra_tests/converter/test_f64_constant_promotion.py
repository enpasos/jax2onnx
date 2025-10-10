# tests/extra_tests/converter/test_f64_constant_promotion.py

from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto

import jax
import jax.numpy as jnp

from jax2onnx.user_interface import to_onnx
from jax2onnx import allclose

CAPTURED_CONSTANT_F32 = np.array([1.5, 2.5, 3.5], dtype=np.float32)


def _function_with_captured_constant(x):
    return x + jnp.array(CAPTURED_CONSTANT_F32)


def _function_with_only_captured_constants():
    return jnp.array(CAPTURED_CONSTANT_F32) * np.float64(2.0)


@pytest.mark.parametrize("enable_x64", [True])
def test_f64_promotion_for_captured_constants_ir(tmp_path, enable_x64):
    original_flag = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", enable_x64)
    try:
        input_f64 = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        jax_result = _function_with_captured_constant(input_f64)
        assert jax_result.dtype == jnp.float64

        onnx_model = to_onnx(
            _function_with_captured_constant,
            [jax.ShapeDtypeStruct(input_f64.shape, input_f64.dtype)],
            model_name="f64_constant_promotion_test_ir",
            enable_double_precision=True,
        )

        found_initializer = False
        for initializer in onnx_model.graph.initializer:
            values = onnx.numpy_helper.to_array(initializer)
            if np.allclose(values, CAPTURED_CONSTANT_F32.astype(np.float64)):
                assert initializer.data_type == TensorProto.DOUBLE
                found_initializer = True
                break
        assert found_initializer, "Captured constant not found in initializers"

        model_path = tmp_path / "f64_constant_promotion_test_ir.onnx"
        model_path.write_bytes(onnx_model.SerializeToString())

        passed, msg = allclose(
            _function_with_captured_constant,
            str(model_path),
            [input_f64],
            {},
            rtol=1e-7,
            atol=1e-7,
        )
        assert passed, msg
    finally:
        jax.config.update("jax_enable_x64", original_flag)


@pytest.mark.parametrize("enable_x64", [True])
def test_f64_promotion_for_function_with_no_inputs_ir(tmp_path, enable_x64):
    original_flag = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", enable_x64)
    try:
        jax_result = _function_with_only_captured_constants()
        assert jax_result.dtype == jnp.float64

        onnx_model = to_onnx(
            _function_with_only_captured_constants,
            [],
            model_name="f64_no_inputs_test_ir",
            enable_double_precision=True,
        )
        assert onnx_model.graph.initializer
        initializer = onnx_model.graph.initializer[0]
        assert initializer.data_type == TensorProto.DOUBLE

        model_path = tmp_path / "f64_no_inputs_test_ir.onnx"
        model_path.write_bytes(onnx_model.SerializeToString())

        passed, msg = allclose(
            _function_with_only_captured_constants,
            str(model_path),
            [],
            {},
            rtol=1e-7,
            atol=1e-7,
        )
        assert passed, msg
    finally:
        jax.config.update("jax_enable_x64", original_flag)
