# tests/extra_tests/test_creation_precision_contract.py

"""Array creation retains source dtype choices and local JAX precision scopes."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import pytest

from tests.extra_tests.test_precision_contract import check


def test_explicit_fp32_creation_stays_fp32_inside_x64() -> None:
    def source():
        return (
            jnp.arange(0.0, 5.0, dtype=jnp.float32),
            jnp.linspace(0.0, 8.0, num=5, dtype=jnp.float32),
            jnp.arange(np.float32(0), np.float32(5)),
        )

    with jax.enable_x64():
        model, reference = check(source, [])
    np.testing.assert_array_equal(reference[0], np.arange(5, dtype=np.float32))
    np.testing.assert_array_equal(reference[1], np.arange(0, 10, 2, dtype=np.float32))
    np.testing.assert_array_equal(reference[2], np.arange(5, dtype=np.float32))
    assert all(
        output.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
        for output in model.graph.output
    )


@pytest.mark.parametrize("operation", ["arange", "linspace"])
@pytest.mark.parametrize("local_x64", [False, True])
def test_creation_uses_local_source_scope(operation: str, local_x64: bool) -> None:
    def source():
        with jax.enable_x64(local_x64):
            if operation == "arange":
                return jnp.arange(0.0, 5.0)
            return jnp.linspace(0.0, 8.0, num=5)

    with jax.enable_x64(not local_x64):
        _, reference = check(source, [])
        assert jax.config.jax_enable_x64 == (not local_x64)
    assert reference.dtype == (np.float64 if local_x64 else np.float32)


def test_linspace_strong_int32_bounds_choose_native_fp32_inside_x64() -> None:
    with jax.enable_x64():
        _, reference = check(
            lambda start, stop: jnp.linspace(start, stop, num=5),
            [np.asarray(0, np.int32), np.asarray(8, np.int32)],
        )
    assert reference.dtype == np.float32
    np.testing.assert_array_equal(reference, np.arange(0, 10, 2, dtype=np.float32))


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_integer_linspace_keeps_native_work_precision_in_local_x64(dtype) -> None:
    def source():
        with jax.enable_x64():
            return jnp.linspace(2**24, 2**24 + 2, num=3, dtype=dtype)

    with jax.enable_x64(False):
        _, reference = check(source, [])
    middle = 2**24 if dtype is np.int32 else 2**24 + 1
    np.testing.assert_array_equal(
        reference, np.asarray([2**24, middle, 2**24 + 2], dtype=dtype)
    )


def test_integer_linspace_preserves_native_flooring_of_negative_fractions() -> None:
    with jax.enable_x64():
        _, reference = check(lambda: jnp.linspace(-2, 1, num=5, dtype=jnp.int32), [])
    np.testing.assert_array_equal(reference, np.asarray([-2, -2, -1, 0, 1], np.int32))


def test_boolean_linspace_uses_native_conversion() -> None:
    with jax.enable_x64():
        _, reference = check(lambda: jnp.linspace(-1, 1, num=3, dtype=jnp.bool_), [])
    np.testing.assert_array_equal(reference, np.asarray([True, False, True], np.bool_))
