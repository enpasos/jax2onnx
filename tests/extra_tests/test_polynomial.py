# tests/extra_tests/test_polynomial.py

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest

from jax2onnx import to_onnx


@pytest.fixture(autouse=True)
def _enable_jax_x64() -> Iterator[None]:
    with jax.enable_x64():
        yield


def _session(
    fn: Callable[..., jax.Array],
    inputs: Sequence[np.ndarray],
) -> ort.InferenceSession:
    model = to_onnx(fn, list(inputs))
    onnx.checker.check_model(model, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def _assert_parity(
    session: ort.InferenceSession,
    fn: Callable[..., jax.Array],
    inputs: Sequence[np.ndarray],
) -> None:
    expected = np.asarray(fn(*(jnp.asarray(value) for value in inputs)))
    assert len(session.get_inputs()) == len(inputs)
    (actual,) = session.run(
        None,
        {
            meta.name: np.asarray(value)
            for meta, value in zip(session.get_inputs(), inputs)
        },
    )
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    tolerance = 2e-3 if actual.dtype == np.float16 else 1e-6
    np.testing.assert_allclose(
        actual, expected, rtol=tolerance, atol=tolerance, equal_nan=True
    )
    zeros = expected == 0
    np.testing.assert_array_equal(
        np.signbit(actual[zeros]), np.signbit(expected[zeros])
    )


@pytest.mark.parametrize(
    ("dtype", "enable_x64"),
    [
        (np.float16, False),
        (np.float16, True),
        (np.float32, False),
        (np.float64, True),
        (np.int32, False),
    ],
)
@pytest.mark.parametrize("shape", [(), (2, 3)])
def test_polyval_dynamic_coefficients_preserve_shape_and_dtype(
    dtype: type, enable_x64: bool, shape: tuple[int, ...]
) -> None:
    with jax.enable_x64(enable_x64):

        def fn(coeffs: jax.Array, x: jax.Array) -> jax.Array:
            return jnp.polyval(coeffs, x)

        coeffs = np.array([2, -3, 1], dtype=dtype)
        x = np.full(shape, 2, dtype=dtype)
        session = _session(fn, [coeffs, x])
        _assert_parity(session, fn, [coeffs, x])
        # Reuse the exported graph with different coefficients and evaluation points.
        _assert_parity(session, fn, [coeffs + 1, x - 1])


@pytest.mark.parametrize("enable_x64", [False, True])
def test_polyval_long_float16_coefficients(enable_x64: bool) -> None:
    with jax.enable_x64(enable_x64):

        def fn(coeffs: jax.Array, x: jax.Array) -> jax.Array:
            return jnp.polyval(coeffs, x)

        coeffs = np.zeros(9, dtype=np.float16)
        coeffs[-3:] = [2, -3, 1]
        x = np.full((2, 3), 2, dtype=np.float16)
        session = _session(fn, [coeffs, x])
        _assert_parity(session, fn, [coeffs, x])
        _assert_parity(session, fn, [coeffs + 1, x - 1])


def test_polyval_broadcasts_coefficient_arrays() -> None:
    def fn(coeffs: jax.Array, x: jax.Array) -> jax.Array:
        return jnp.polyval(coeffs, x)

    coeffs = np.arange(6, dtype=np.float32).reshape(3, 2, 1) / 4
    x = np.array([[-1.5, 0.0, 2.0]], dtype=np.float32)
    session = _session(fn, [coeffs, x])
    _assert_parity(session, fn, [coeffs, x])


@pytest.mark.parametrize(
    ("coefficient_count", "unroll"), [(3, None), (9, None), (3, 1), (9, 16)]
)
def test_polyval_recurrence_preserves_nonfinite_and_signed_zero_semantics(
    coefficient_count: int, unroll: int | None
) -> None:
    def fn(coeffs: jax.Array, x: jax.Array) -> jax.Array:
        if unroll is None:
            return jnp.polyval(coeffs, x)
        return jnp.polyval(coeffs, x, unroll=unroll)

    coeffs = np.zeros(coefficient_count, dtype=np.float32)
    coeffs[0] = 1
    x = np.array([np.nan, -np.inf, np.inf, -0.0, 0.0, -1.0, 1.0, 2.0], dtype=np.float32)
    session = _session(fn, [coeffs, x])
    _assert_parity(session, fn, [coeffs, x])
    _assert_parity(session, fn, [np.full_like(coeffs, -0.0), x])


def test_sinc_polynomial_branch_near_zero() -> None:
    def fn(x: jax.Array) -> jax.Array:
        return jnp.sinc(x)

    x = np.array([-0.1, -1e-4, -0.0, 0.0, 1e-4, 0.1], dtype=np.float32)
    session = _session(fn, [x])
    _assert_parity(session, fn, [x])
