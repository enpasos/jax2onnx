# tests/extra_tests/test_complex_cast_precision_contract.py

"""Complex conversion changes component values without native complex ONNX ops."""

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import pytest

from tests.extra_tests.test_precision_contract import check


def _assert_no_native_complex_cast(model):
    for node in model.graph.node:
        if node.op_type == "Cast":
            assert all(
                attribute.i
                not in (onnx.TensorProto.COMPLEX64, onnx.TensorProto.COMPLEX128)
                for attribute in node.attribute
                if attribute.name == "to"
            )


@pytest.mark.parametrize(
    "dtype,values",
    [
        (np.float32, [4097, -3]),
        (np.int64, [2**40 + 1, -(2**40 + 3)]),
    ],
    ids=["real-length-two", "wide-int64"],
)
def test_real_to_complex128_preserves_components_and_shared_original(dtype, values):
    def source(value):
        converted = value.astype(jnp.complex128)
        return jnp.real(converted), jnp.imag(converted), value

    inputs = [np.asarray(values, dtype)]
    with jax.enable_x64():
        model, reference = check(source, inputs)
    assert reference[0].dtype == reference[1].dtype == np.float64
    np.testing.assert_array_equal(reference[0], inputs[0].astype(np.float64))
    np.testing.assert_array_equal(reference[1], np.zeros((2,), np.float64))
    assert reference[2].dtype == dtype
    np.testing.assert_array_equal(reference[2], inputs[0])
    _assert_no_native_complex_cast(model)


def test_complex128_to_complex64_rounds_components_and_preserves_original():
    def source(real, imag):
        original = jax.lax.complex(real, imag)
        converted = original.astype(jnp.complex64)
        return (
            jnp.real(converted),
            jnp.imag(converted),
            jnp.real(original),
            jnp.imag(original),
        )

    inputs = [
        np.asarray([1 + 2**-24, 2**24 + 1], np.float64),
        np.asarray([-1 - 2**-24, -(2**24 + 3)], np.float64),
    ]
    with jax.enable_x64():
        model, reference = check(source, inputs)
    for actual, original in zip(reference[:2], inputs, strict=True):
        assert actual.dtype == np.float32
        np.testing.assert_array_equal(actual, original.astype(np.float32))
    for actual, original in zip(reference[2:], inputs, strict=True):
        assert actual.dtype == np.float64
        np.testing.assert_array_equal(actual, original)
    _assert_no_native_complex_cast(model)


@pytest.mark.parametrize(
    "dtype,expected",
    [
        (np.float64, [200, -200, 127.5, -128.5]),
        (np.int8, [127, -128, 127, -128]),
    ],
)
def test_complex_to_real_projects_before_cast_and_preserves_original(dtype, expected):
    def source(real, imag):
        original = jax.lax.complex(real, imag)
        converted = original.astype(dtype)
        return converted, jnp.real(original), jnp.imag(original)

    inputs = [
        np.asarray([200, -200, 127.5, -128.5], np.float32),
        np.asarray([3, -4, 5, -6], np.float32),
    ]
    with jax.enable_x64():
        model, reference = check(source, inputs)
    assert reference[0].dtype == dtype
    np.testing.assert_array_equal(reference[0], np.asarray(expected, dtype))
    for actual, original in zip(reference[1:], inputs, strict=True):
        assert actual.dtype == np.float32
        np.testing.assert_array_equal(actual, original)
    _assert_no_native_complex_cast(model)
