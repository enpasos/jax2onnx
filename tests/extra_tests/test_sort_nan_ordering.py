# tests/extra_tests/test_sort_nan_ordering.py

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from jax2onnx import to_onnx


def _session(
    fn: Callable[..., Any],
    arrays: Sequence[np.ndarray],
    *,
    opset: int,
    optimization: str,
    inputs: list[Any] | None = None,
) -> ort.InferenceSession:
    model = to_onnx(
        fn,
        inputs or [jax.ShapeDtypeStruct(array.shape, array.dtype) for array in arrays],
        input_names=[f"input_{index}" for index in range(len(arrays))],
        opset=opset,
    )
    onnx.checker.check_model(model, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        if optimization == "basic"
        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def _run(session: ort.InferenceSession, *arrays: np.ndarray) -> list[np.ndarray]:
    return session.run(
        None, {f"input_{index}": array for index, array in enumerate(arrays)}
    )


def _assert_exact(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    # Include signed zero and quiet NaN payload/sign bits in the oracle.
    assert actual.tobytes(order="C") == expected.tobytes(order="C")


def _sort(frontend: str, x: jax.Array, axis: int = -1) -> jax.Array:
    if frontend == "numpy":
        return jnp.sort(x, axis=axis)
    return jax.lax.sort(x, dimension=axis, is_stable=True)


@pytest.mark.parametrize("frontend", ["numpy", "lax"])
@pytest.mark.parametrize(
    "dtype,opset,optimization",
    [
        (np.float16, 11, "basic"),
        (np.float32, 12, "disabled"),
        (np.float64, 23, "basic"),
    ],
    ids=["fp16-op11-basic", "fp32-op12-disabled", "fp64-current-basic"],
)
def test_sort_places_nan_after_positive_infinity_for_both_input_orders(
    frontend: str, dtype: type, opset: int, optimization: str
) -> None:
    with jax.enable_x64(dtype == np.float64):
        cases = [
            np.asarray(values, dtype=dtype)
            for values in (
                [np.nan, np.inf, -np.inf, 3, np.nan, np.inf],
                [np.inf, np.nan, -np.inf, np.inf, 3, np.nan],
                [np.nan, np.inf, 1, 1, np.inf, np.nan],
                [np.nan] * 6,
            )
        ]

        def fn(x: jax.Array) -> jax.Array:
            return _sort(frontend, x)

        session = _session(fn, cases[:1], opset=opset, optimization=optimization)
        for array in cases:
            expected = np.asarray(fn(jnp.asarray(array)))
            actual = _run(session, array)[0]
            _assert_exact(actual, expected)
            non_nan_count = np.count_nonzero(~np.isnan(array))
            assert not np.any(np.isnan(actual[:non_nan_count]))
            assert np.all(np.isnan(actual[non_nan_count:]))


def _payload_values(dtype: type) -> np.ndarray:
    if dtype == np.float16:
        bits, uint_dtype = [
            0x7E11,
            0xFC00,
            0,
            0x8000,
            0x7C00,
            0xFE22,
            0x8000,
            0,
            0x3C00,
            0xBC00,
        ], np.uint16
    elif dtype == np.float32:
        bits, uint_dtype = [
            0x7FC01234,
            0xFF800000,
            0,
            0x80000000,
            0x7F800000,
            0xFFC05678,
            0x80000000,
            0,
            0x3F800000,
            0xBF800000,
        ], np.uint32
    else:
        bits, uint_dtype = [
            0x7FF8000000001234,
            0xFFF0000000000000,
            0,
            0x8000000000000000,
            0x7FF0000000000000,
            0xFFF8000000005678,
            0x8000000000000000,
            0,
            0x3FF0000000000000,
            0xBFF0000000000000,
        ], np.uint64
    return np.asarray(bits, dtype=uint_dtype).view(dtype)


@pytest.mark.parametrize("frontend", ["numpy", "lax"])
@pytest.mark.parametrize(
    "dtype,opset,optimization",
    [
        (np.float16, 12, "disabled"),
        (np.float32, 23, "basic"),
        (np.float64, 11, "disabled"),
    ],
    ids=["fp16-op12-disabled", "fp32-current-basic", "fp64-op11-disabled"],
)
def test_stable_sort_preserves_nan_payloads_and_signed_zero_reciprocals(
    frontend: str, dtype: type, opset: int, optimization: str
) -> None:
    # Cover low-precision storage with caller-owned x64 enabled too.
    with jax.enable_x64():
        array = _payload_values(dtype)

        def fn(x: jax.Array) -> tuple[jax.Array, jax.Array]:
            sorted_values = _sort(frontend, x)
            return sorted_values, jnp.reciprocal(sorted_values)

        session = _session(fn, [array], opset=opset, optimization=optimization)
        sorted_values, reciprocals = _run(session, array)
        # The explicit stable permutation is independent of exporter/JAX code.
        expected = array[[1, 9, 2, 3, 6, 7, 8, 4, 0, 5]]
        _assert_exact(sorted_values, expected)
        _assert_exact(sorted_values, np.asarray(fn(jnp.asarray(array))[0]))
        zero_positions = expected == 0
        assert np.all(np.isinf(reciprocals[zero_positions]))
        np.testing.assert_array_equal(
            np.signbit(reciprocals[zero_positions]),
            np.signbit(expected[zero_positions]),
        )
        # NaNs are equal sorting keys; their distinct payloads keep input order.
        all_nan = np.tile(array[[0, 5]], 5)
        _assert_exact(_run(session, all_nan)[0], all_nan)


@pytest.mark.parametrize(
    "kind,opset,optimization",
    [
        ("axis0", 11, "disabled"),
        ("negative-axis", 12, "basic"),
        ("vmap", 23, "basic"),
        ("symbolic-batch", 23, "disabled"),
    ],
)
def test_nan_last_sort_preserves_axis_and_batch_behavior(
    kind: str, opset: int, optimization: str
) -> None:
    row = np.asarray([np.nan, np.inf, -np.inf, 2, np.inf, np.nan], dtype=np.float32)
    batch = np.stack([row, row[::-1], np.roll(row, 2)])
    if kind == "axis0":
        array = batch.T.copy()

        def fn(x: jax.Array) -> jax.Array:
            return jnp.sort(x, axis=0)

    elif kind == "negative-axis":
        array = np.stack([batch.T, np.roll(batch.T, 1, axis=0)])

        def fn(x: jax.Array) -> jax.Array:
            return jax.lax.sort(x, dimension=-2, is_stable=True)

    elif kind == "vmap":
        array = batch

        def fn(x: jax.Array) -> jax.Array:
            return jax.vmap(lambda y: jnp.sort(y))(x)

    else:
        array = batch

        def fn(x: jax.Array) -> jax.Array:
            return jnp.sort(x, axis=-1)

    inputs = [("B", 6)] if kind == "symbolic-batch" else None
    session = _session(
        fn, [array], opset=opset, optimization=optimization, inputs=inputs
    )
    for current in [array[:1], array] if kind == "symbolic-batch" else [array]:
        _assert_exact(_run(session, current)[0], np.asarray(fn(jnp.asarray(current))))


@pytest.mark.parametrize("opset,optimization", [(11, "disabled"), (23, "basic")])
def test_lax_sort_keys_and_distinct_payloads_keep_stable_ties(
    opset: int, optimization: str
) -> None:
    keys = np.asarray(
        [np.nan, np.inf, 2, 2, np.nan, np.inf, -0.0, 0.0, 2], dtype=np.float32
    )
    payload = np.arange(keys.size, dtype=np.int32)

    def fn(k: jax.Array, v: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jax.lax.sort((k, v), dimension=0, is_stable=True, num_keys=1)

    session = _session(fn, [keys, payload], opset=opset, optimization=optimization)
    expected_order = [6, 7, 2, 3, 8, 1, 5, 0, 4]
    actual = _run(session, keys, payload)
    _assert_exact(actual[0], keys[expected_order])
    _assert_exact(actual[1], payload[expected_order])
    for result, expected in zip(
        actual, fn(jnp.asarray(keys), jnp.asarray(payload)), strict=True
    ):
        _assert_exact(result, np.asarray(expected))


@pytest.mark.parametrize("opset,optimization", [(12, "basic"), (23, "disabled")])
def test_lax_sort_lexicographic_nan_keys_preserve_payload_associations(
    opset: int, optimization: str
) -> None:
    primary = np.asarray(
        [np.nan, np.inf, np.nan, np.inf, 1, 1, 0.0, -0.0, np.nan, 1], dtype=np.float32
    )
    secondary = np.asarray(
        [np.nan, np.nan, -np.inf, np.inf, 2, 2, 1, 1, np.nan, np.inf], dtype=np.float32
    )
    payload = np.arange(10, dtype=np.int32)

    def fn(
        a: jax.Array, b: jax.Array, p: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return jax.lax.sort((a, b, p), dimension=-1, num_keys=2, is_stable=True)

    session = _session(
        fn, [primary, secondary, payload], opset=opset, optimization=optimization
    )
    expected_order = [6, 7, 4, 5, 9, 3, 1, 2, 0, 8]
    actual = _run(session, primary, secondary, payload)
    for result, original in zip(actual, (primary, secondary, payload), strict=True):
        _assert_exact(result, original[expected_order])
    for result, expected in zip(
        actual,
        fn(jnp.asarray(primary), jnp.asarray(secondary), jnp.asarray(payload)),
        strict=True,
    ):
        _assert_exact(result, np.asarray(expected))


def test_argsort_nan_infinity_and_signed_zero_indices_are_stable() -> None:
    array = np.asarray([np.nan, np.inf, 0.0, -0.0, np.nan, np.inf], dtype=np.float32)

    def fn(x: jax.Array) -> jax.Array:
        return jnp.argsort(x, stable=True)

    # This consumer's iota path needs a newer Squeeze schema; direct sort tests
    # above cover opsets 11/12 independently of that path.
    session = _session(fn, [array], opset=23, optimization="basic")
    actual = _run(session, array)[0]
    np.testing.assert_array_equal(
        actual, np.asarray([2, 3, 1, 5, 0, 4], dtype=actual.dtype)
    )
    _assert_exact(actual, np.asarray(fn(jnp.asarray(array))))


@pytest.mark.parametrize("frontend,dtype", [("numpy", np.int32), ("lax", np.bool_)])
def test_integer_and_boolean_sort_paths_keep_their_dtypes(
    frontend: str, dtype: type
) -> None:
    array = np.asarray([1, 0, 1, 0, 0, 1], dtype=dtype)

    def fn(x: jax.Array) -> jax.Array:
        return _sort(frontend, x)

    session = _session(fn, [array], opset=23, optimization="basic")
    _assert_exact(_run(session, array)[0], np.asarray(fn(jnp.asarray(array))))
