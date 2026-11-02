# tests/extra_tests/test_conj_precision.py

"""Conjugation keeps source complex types and signed imaginary zeros."""

import jax
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from jax2onnx import to_onnx
from tests.extra_tests.test_precision_control_flow import _assert_exact


def _packed(value):
    value = np.asarray(value)
    return np.stack((value.real, value.imag), axis=-1)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_lax_conj_real_and_complex_source_precision(dtype):
    x = np.asarray([[-3.0, -0.0], [0.0, np.inf], [-np.inf, np.nan]], dtype=dtype)
    is_complex = np.issubdtype(dtype, np.complexfloating)
    if is_complex:
        x.imag = np.asarray([[0.0, -0.0], [2.0, -2.0], [0.0, -0.0]])

    def source(value):
        return jax.lax.conj(value)

    with jax.enable_x64():
        native_input = jax.numpy.asarray(x)
        reference = np.asarray(source(native_input))
        expected_dtype = np.complex128 if x.real.dtype == np.float64 else np.complex64
        assert reference.dtype == expected_dtype
        _assert_exact(reference.real, x.real)
        _assert_exact(reference.imag, -x.imag if is_complex else np.zeros_like(x))
        compiled = jax.jit(source)
        _assert_exact(_packed(compiled(native_input)), _packed(reference))
        _assert_exact(_packed(compiled(native_input)), _packed(reference))
        model = to_onnx(source, [jax.ShapeDtypeStruct(("B", 2), dtype)])

    onnx.checker.check_model(model, full_check=True)
    feed = _packed(x) if is_complex else x
    for optimization in (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    ):
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.graph_optimization_level = optimization
        session = ort.InferenceSession(
            model.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        (actual,) = session.run(None, {session.get_inputs()[0].name: feed})
        _assert_exact(actual, _packed(reference))
