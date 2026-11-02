# tests/extra_tests/test_complex_exp_precision_contract.py

"""Finite complex exponentials retain packed source-width components."""

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import pytest

from tests.extra_tests.test_precision_contract import check


def _exp(api, value):
    return jax.lax.exp(value) if api == "lax" else jnp.exp(value)


@pytest.mark.parametrize("api", ["lax", "jnp"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_finite_complex_exp_and_rope_rows_preserve_shared_original(api, dtype):
    def source(real, imag):
        original = jax.lax.complex(real, imag)
        result = _exp(api, original)
        return (
            jnp.real(result),
            jnp.imag(result),
            jnp.real(original),
            jnp.imag(original),
        )

    inputs = [
        np.asarray([[0, 0, 0], [-0.5, 0.25, 0.75]], dtype),
        np.asarray([[0, 0.25, -0.75], [0.125, -0.5, 1]], dtype),
    ]
    tolerance = 1e-14 if dtype is np.float64 else 1e-5
    with jax.enable_x64():
        model, reference = check(source, inputs, rtol=tolerance, atol=0)
    assert reference[0].dtype == reference[1].dtype == dtype
    for actual, original in zip(reference[2:], inputs, strict=True):
        assert actual.dtype == dtype
        np.testing.assert_array_equal(actual, original)
    for value in (*model.graph.input, *model.graph.value_info, *model.graph.output):
        assert value.type.tensor_type.elem_type not in (
            onnx.TensorProto.COMPLEX64,
            onnx.TensorProto.COMPLEX128,
        )
    for node in model.graph.node:
        if node.op_type == "Cast":
            assert all(
                attribute.i
                not in (onnx.TensorProto.COMPLEX64, onnx.TensorProto.COMPLEX128)
                for attribute in node.attribute
                if attribute.name == "to"
            )


@pytest.mark.parametrize("api", ["lax", "jnp"])
def test_complex128_exp_tiny_real_and_signed_zero_exact_bits(api):
    def source(real, imag):
        original = jax.lax.complex(real, imag)
        result = _exp(api, original)
        return (
            jnp.real(result),
            jnp.imag(result),
            jnp.real(original),
            jnp.imag(original),
        )

    inputs = [
        np.asarray([2**-25, 2**-25], np.float64),
        np.asarray([0.0, -0.0], np.float64),
    ]
    with jax.enable_x64():
        _, reference = check(source, inputs)
    # exp(2**-25) rounds to 1 + 2**-25 + 2**-51 in binary64.
    # A FLOAT computation would lose the increment above one.
    np.testing.assert_array_equal(
        reference[0].view(np.uint64),
        np.asarray([0x3FF0000008000002, 0x3FF0000008000002], np.uint64),
    )
    np.testing.assert_array_equal(
        reference[1].view(np.uint64), np.asarray([0, 0], np.uint64)
    )
    for actual, original in zip(reference[2:], inputs, strict=True):
        assert actual.dtype == np.float64
        np.testing.assert_array_equal(actual.view(np.uint64), original.view(np.uint64))
