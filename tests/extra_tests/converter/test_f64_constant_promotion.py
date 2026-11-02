# tests/extra_tests/converter/test_f64_constant_promotion.py

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
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


def _with_observable_constant(function):
    def observed(*args):
        return function(*args), jnp.array(CAPTURED_CONSTANT_F32)

    return observed


def _check_source_and_runtime(function, model, inputs):
    args = [jnp.asarray(value, dtype=value.dtype) for value in inputs]
    reference = function(*args)
    leaves = jax.tree.leaves(reference)
    compiled = jax.jit(function)
    for result in (compiled(*args), compiled(*args)):
        assert jax.tree.structure(result) == jax.tree.structure(reference)
        for actual, expected in zip(jax.tree.leaves(result), leaves):
            assert actual.dtype == expected.dtype
            np.testing.assert_array_equal(actual, expected)
    onnx.checker.check_model(model, full_check=True)
    for level in (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    ):
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.graph_optimization_level = level
        session = ort.InferenceSession(
            model.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        actual = session.run(
            None,
            {meta.name: value for meta, value in zip(session.get_inputs(), inputs)},
        )
        assert len(actual) == len(leaves)
        for value, expected in zip(actual, leaves):
            assert value.dtype == expected.dtype
            assert value.shape == expected.shape
            np.testing.assert_allclose(value, expected, rtol=1e-7, atol=1e-7)


def _assert_observable_constant(model):
    # A required FP64 consumer may fold its local conversion. An independently
    # observable FP32 output must still retain the original payload and type.
    assert model.graph.output[1].type.tensor_type.elem_type == TensorProto.FLOAT
    payloads = [onnx.numpy_helper.to_array(value) for value in model.graph.initializer]
    assert any(
        value.dtype == CAPTURED_CONSTANT_F32.dtype
        and value.shape == CAPTURED_CONSTANT_F32.shape
        and value.tobytes() == CAPTURED_CONSTANT_F32.tobytes()
        for value in payloads
    )


@pytest.mark.parametrize("observe_constant", [False, True])
@pytest.mark.parametrize("enable_x64", [True])
def test_f64_consumers_preserve_captured_constant_precision(
    tmp_path, enable_x64, observe_constant
):
    original_flag = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", enable_x64)
    try:
        input_f64 = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        function = (
            _with_observable_constant(_function_with_captured_constant)
            if observe_constant
            else _function_with_captured_constant
        )
        jax_result = function(input_f64)
        assert jax.tree.leaves(jax_result)[0].dtype == jnp.float64

        onnx_model = to_onnx(
            function,
            [jax.ShapeDtypeStruct(input_f64.shape, input_f64.dtype)],
            model_name="f64_constant_promotion_test_ir",
        )

        assert (
            onnx_model.graph.output[0].type.tensor_type.elem_type == TensorProto.DOUBLE
        )
        if observe_constant:
            _assert_observable_constant(onnx_model)
        _check_source_and_runtime(function, onnx_model, [input_f64])

        model_path = tmp_path / "f64_constant_promotion_test_ir.onnx"
        model_path.write_bytes(onnx_model.SerializeToString())

        passed, msg = allclose(
            function,
            str(model_path),
            [input_f64],
            {},
            rtol=1e-7,
            atol=1e-7,
        )
        assert passed, msg
    finally:
        jax.config.update("jax_enable_x64", original_flag)


@pytest.mark.parametrize("observe_constant", [False, True])
@pytest.mark.parametrize("enable_x64", [True])
def test_f64_constant_arithmetic_retains_typed_payload_without_inputs(
    tmp_path, enable_x64, observe_constant
):
    original_flag = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", enable_x64)
    try:
        function = (
            _with_observable_constant(_function_with_only_captured_constants)
            if observe_constant
            else _function_with_only_captured_constants
        )
        jax_result = function()
        assert jax.tree.leaves(jax_result)[0].dtype == jnp.float64

        onnx_model = to_onnx(
            function,
            [],
            model_name="f64_no_inputs_test_ir",
        )
        assert onnx_model.graph.initializer
        assert (
            onnx_model.graph.output[0].type.tensor_type.elem_type == TensorProto.DOUBLE
        )
        if observe_constant:
            _assert_observable_constant(onnx_model)
        _check_source_and_runtime(function, onnx_model, [])

        model_path = tmp_path / "f64_no_inputs_test_ir.onnx"
        model_path.write_bytes(onnx_model.SerializeToString())

        passed, msg = allclose(
            function,
            str(model_path),
            [],
            {},
            rtol=1e-7,
            atol=1e-7,
        )
        assert passed, msg
    finally:
        jax.config.update("jax_enable_x64", original_flag)
