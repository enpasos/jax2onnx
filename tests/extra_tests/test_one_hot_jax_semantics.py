# tests/extra_tests/test_one_hot_jax_semantics.py

"""jax.nn.one_hot exports the comparison that defines it in JAX.

JAX computes ``(expand_dims(x, axis) == iota(x.dtype)).astype(dtype)``. ONNX
OneHot is a different function: it counts an index in ``[-depth, -1]`` from the
end and truncates a non-integer index, where JAX returns a row of zeros.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest

from jax2onnx import onnx_function, to_onnx


@contextmanager
def _x64(enabled: bool) -> Iterator[None]:
    previous = bool(jax.config.read("jax_enable_x64"))
    jax.config.update("jax_enable_x64", enabled)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _op_types(model: onnx.ModelProto) -> list[str]:
    ops = [node.op_type for node in model.graph.node]
    for function in model.functions:
        ops.extend(node.op_type for node in function.node)
    return ops


def _run(model: onnx.ModelProto, *inputs: np.ndarray) -> np.ndarray:
    onnx.checker.check_model(model, full_check=True)
    assert "OneHot" not in _op_types(model)
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    feeds = {
        meta.name: value
        for meta, value in zip(session.get_inputs(), inputs, strict=True)
    }
    (output,) = session.run(None, feeds)
    return np.asarray(output)


def _assert_matches_jax(
    fn: Callable[[jax.Array], jax.Array], indices: np.ndarray
) -> np.ndarray:
    expected = np.asarray(fn(jnp.asarray(indices)))
    got = _run(to_onnx(fn, [jnp.asarray(indices)]), indices)
    assert got.dtype == expected.dtype
    np.testing.assert_array_equal(got, expected)
    return got


# Indices inside the class range, on both of its edges, and outside of it.
_INTEGERS = [-7, -6, -1, 0, 2, 5, 6, 7]
_FLOATS = [0.0, 1.0, 1.5, -0.5, 2.0, -1.0, 5.0, 6.0, np.nan, np.inf, -np.inf]

_INDEX_CASES: dict[str, tuple[Any, list[float], bool]] = {
    # name: (index dtype, values, needs x64)
    "int8": (np.int8, _INTEGERS, False),
    "int16": (np.int16, _INTEGERS, False),
    "int32": (np.int32, _INTEGERS, False),
    "int64": (np.int64, _INTEGERS, True),
    "uint8": (np.uint8, [0, 2, 5, 6, 7, 250], False),
    "uint32": (np.uint32, [0, 2, 5, 6, 7, 2**32 - 1], False),
    "float16": (np.float16, _FLOATS, False),
    "float32": (np.float32, _FLOATS, False),
    "float64": (np.float64, _FLOATS, True),
}


@pytest.mark.parametrize("x64", [False, True], ids=["x32", "x64"])
@pytest.mark.parametrize("name", _INDEX_CASES)
def test_rows_match_jax_for_every_index_dtype(name: str, x64: bool) -> None:
    dtype, values, needs_x64 = _INDEX_CASES[name]
    if needs_x64 and not x64:
        pytest.skip(f"{name} indices need x64")
    indices = np.asarray(values, dtype=dtype)

    with _x64(x64):
        got = _assert_matches_jax(lambda x: jax.nn.one_hot(x, 6), indices)

    # The default output dtype is JAX's default float dtype.
    assert got.dtype == (np.float64 if x64 else np.float32)
    in_range = np.isin(indices, np.arange(6))
    np.testing.assert_array_equal(got.sum(axis=-1), in_range.astype(got.dtype))


def test_negative_index_is_a_row_of_zeros() -> None:
    # ONNX OneHot would select the last class for -1.
    got = _assert_matches_jax(
        lambda x: jax.nn.one_hot(x, 3), np.array([-1, 2, -3], dtype=np.int32)
    )

    np.testing.assert_array_equal(got, [[0, 0, 0], [0, 0, 1], [0, 0, 0]])


@pytest.mark.parametrize(
    ("dtype", "num_classes", "indices", "selected"),
    [
        (np.uint8, 300, [0, 43, 255], [[0, 0], [0, 256], [1, 43], [1, 299], [2, 255]]),
        (np.int8, 200, [-128, -57, 127], [[0, 128], [1, 199], [2, 127]]),
    ],
    ids=["uint8", "int8"],
)
def test_class_positions_wrap_in_a_narrow_index_dtype_like_jax(
    dtype: Any, num_classes: int, indices: list[int], selected: list[list[int]]
) -> None:
    # JAX builds the class positions in the dtype of the indices. With more
    # classes than that dtype can count, the positions wrap: a uint8 index
    # selects two classes, and a negative int8 index selects one.
    got = _assert_matches_jax(
        lambda x: jax.nn.one_hot(x, num_classes), np.asarray(indices, dtype=dtype)
    )

    assert np.argwhere(got).tolist() == selected


@pytest.mark.parametrize(
    "dtype", [jnp.bool_, jnp.int8, jnp.int32, jnp.float16, jnp.float32]
)
def test_requested_output_dtype(dtype: Any) -> None:
    got = _assert_matches_jax(
        lambda x: jax.nn.one_hot(x, 4, dtype=dtype),
        np.array([3, -1, 0], dtype=np.int32),
    )

    assert got.dtype == np.dtype(dtype)


@pytest.mark.parametrize("x64", [False, True], ids=["x32", "x64"])
def test_requested_output_dtype_follows_the_x64_setting(x64: bool) -> None:
    indices = np.array([1, -1], dtype=np.int32)

    with _x64(x64):
        got = _assert_matches_jax(
            lambda x: jax.nn.one_hot(x, 3, dtype=jnp.float64), indices
        )

    assert got.dtype == (np.float64 if x64 else np.float32)


@pytest.mark.parametrize("axis", [0, 1, 2, -1, -2, -3])
def test_axis(axis: int) -> None:
    indices = np.array([[0, 3, -1], [2, 4, 1]], dtype=np.int32)

    _assert_matches_jax(lambda x: jax.nn.one_hot(x, 4, axis=axis), indices)


def test_scalar_index_and_empty_class_range() -> None:
    _assert_matches_jax(lambda x: jax.nn.one_hot(x, 4), np.asarray(2, dtype=np.int32))
    _assert_matches_jax(lambda x: jax.nn.one_hot(x, 0), np.array([1], dtype=np.int32))


def test_symbolic_batch_dimension() -> None:
    def fn(x: jax.Array) -> jax.Array:
        return jax.nn.one_hot(x, 5)

    model = to_onnx(fn, [jax.ShapeDtypeStruct(("B", 3), jnp.int32)])
    for batch in (1, 4):
        indices = np.arange(batch * 3, dtype=np.int32).reshape(batch, 3) - 2
        np.testing.assert_array_equal(
            _run(model, indices), np.asarray(fn(jnp.asarray(indices)))
        )


def test_boolean_indices_are_rejected_like_in_jax() -> None:
    indices = jnp.asarray([True, False])

    with pytest.raises(TypeError, match="bool"):
        to_onnx(lambda x: jax.nn.one_hot(x, 2), [indices])


@onnx_function
def _labels_to_rows(x: jax.Array) -> jax.Array:
    return jax.nn.one_hot(x, 4, dtype=jnp.float32)


def test_inside_an_onnx_function() -> None:
    # A Function body has no initializers; the class positions are a Constant.
    def fn(x: jax.Array) -> jax.Array:
        return _labels_to_rows(x) * 2.0

    indices = np.array([3, -1, 0], dtype=np.int32)
    model = to_onnx(fn, [jnp.asarray(indices)])

    (function,) = model.functions
    assert [node.op_type for node in function.node].count("Equal") == 1
    np.testing.assert_array_equal(
        _run(model, indices), np.asarray(fn(jnp.asarray(indices)))
    )
