# tests/extra_tests/framework/test_polynomial_lowering.py

"""Regression coverage for the native polynomial primitive added in JAX 0.12."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import onnxruntime as ort
import pytest

from jax2onnx import to_onnx
from jax2onnx._compat.jax import native_polynomial_lowering


def _polynomial(x, coefficients, *, unroll=None):
    if native_polynomial_lowering() is not None:
        return jax.lax.polynomial(x, coefficients, unroll=unroll)
    # Exercise the existing public polynomial path on older supported JAX.
    # Only the capability check below claims coverage of the new primitive.
    broadcast = jnp.broadcast_arrays(
        *(jnp.asarray(c, dtype=x.dtype) for c in coefficients)
    )
    return jnp.polyval(
        jnp.stack(tuple(reversed(broadcast))),
        x,
        unroll=1 if unroll is None else unroll,
    )


def test_polynomial_native_capability_matches_installed_jax():
    lowering = native_polynomial_lowering()
    if lowering is None:
        assert "polynomial_p" not in vars(jax.lax)
        assert "polynomial" not in vars(jax.lax)
    else:
        assert callable(lowering)
        assert vars(jax.lax)["polynomial_p"].name == "polynomial"


def _run(fn, inputs, *, dtype):
    values = [np.asarray(value, dtype=dtype) for value in inputs]
    with jax.enable_x64(np.dtype(dtype) == np.dtype(np.float64)):
        expected = np.asarray(fn(*(jnp.asarray(value) for value in values)))
        model = to_onnx(
            fn,
            inputs=values,
            enable_double_precision=np.dtype(dtype) == np.dtype(np.float64),
        )
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    actual = session.run(
        None,
        {
            argument.name: value
            for argument, value in zip(session.get_inputs(), values, strict=True)
        },
    )[0]
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    tolerance = 1e-12 if np.dtype(dtype) == np.dtype(np.float64) else 2e-6
    np.testing.assert_allclose(
        actual, expected, rtol=tolerance, atol=tolerance, equal_nan=True
    )
    return actual, model


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_polynomial_ascending_coefficients_and_broadcast(dtype):
    actual, model = _run(
        lambda x, c0, c1, c2: _polynomial(x, [c0, c1, c2]),
        [
            np.asarray([[3.0], [-2.0]]),
            np.asarray([[1.0, -1.0]]),
            np.asarray(5.0),
            np.asarray(2.0),
        ],
        dtype=dtype,
    )
    np.testing.assert_array_equal(actual[:, 0], np.asarray([34.0, -1.0], dtype=dtype))
    if native_polynomial_lowering() is not None:
        assert sum(node.op_type == "Mul" for node in model.graph.node) == 2
        assert sum(node.op_type == "Add" for node in model.graph.node) == 2


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("unroll", [None, 1])
def test_polynomial_scan_preserves_initial_zero_multiply(dtype, unroll):
    coefficients = [
        np.asarray(value)
        for value in [1.0, 0.25, -0.5, 0.125, -0.25, 0.5, 0.125, -0.125, 0.25]
    ]
    _, model = _run(
        lambda x, *cs: _polynomial(x, cs, unroll=unroll),
        [np.asarray([-0.0, 0.25, -0.75, 1.0, np.inf, -np.inf, np.nan]), *coefficients],
        dtype=dtype,
    )
    assert any(node.op_type == "Loop" for node in model.graph.node)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_two_polynomial_calls_do_not_share_var_bindings(dtype):
    def fn(x, y):
        a = _polynomial(x, [1.0, 5.0, 2.0])
        b = _polynomial(y, [-3.0, 0.25, -0.5])
        return a + b

    _run(fn, [np.asarray([3.0, -0.5]), np.asarray([-2.0, 0.75])], dtype=dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sinc_noninteger_arguments(dtype):
    _run(
        jnp.sinc, [np.asarray([-2.25, -0.1, -0.0, 0.0, 0.125, 1.5, 13.25])], dtype=dtype
    )


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_polynomial_preserves_low_precision_tensor_metadata(dtype):
    import onnx_ir as ir

    model = to_onnx(
        lambda x: _polynomial(x, [1.0, 5.0, 2.0]),
        inputs=[jax.ShapeDtypeStruct((3,), dtype)],
        return_mode="ir",
        enable_double_precision=False,
    )
    expected = ir.DataType.FLOAT16 if dtype is jnp.float16 else ir.DataType.BFLOAT16
    assert model.graph.inputs[0].dtype == expected
    assert model.graph.outputs[0].dtype == expected
