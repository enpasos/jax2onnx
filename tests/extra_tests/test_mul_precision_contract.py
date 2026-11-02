# tests/extra_tests/test_mul_precision_contract.py

"""Mul converts its operands to the source result dtype before arithmetic."""

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import pytest

from tests.extra_tests.test_precision_contract import check


def _out_dtype_available(record_property):
    if "out_dtype" in inspect.signature(jax.lax.mul).parameters:
        return True
    # Minimum JAX lacks this API. Verify that boundary explicitly; it supplies
    # no numerical reference for these requests. Ordinary Mul remains tested.
    inputs = np.ones(1, np.float32)
    with pytest.raises(TypeError, match="unexpected keyword argument 'out_dtype'"):
        jax.lax.mul(inputs, inputs, out_dtype=np.float64)
    record_property("native_execution", "API_UNAVAILABLE")
    record_property("native_numerical_qualification", "UNVERIFIED")
    return False


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_ordinary_mul_retains_source_dtype_and_shared_operands(dtype):
    def source(lhs, rhs):
        return jax.lax.mul(lhs, rhs), lhs, rhs

    inputs = [np.asarray([1.5, -2], dtype), np.asarray([2, 0.5], dtype)]
    with jax.enable_x64():
        _, reference = check(source, inputs)
    np.testing.assert_array_equal(reference[0], np.asarray([3, -1], dtype))
    np.testing.assert_array_equal(reference[1], inputs[0])
    np.testing.assert_array_equal(reference[2], inputs[1])


@pytest.mark.parametrize(
    "input_dtype,result_dtype,value,expected",
    [
        (np.float32, np.float64, 4097, 16785409),
        (np.float64, np.float32, 1 + 2**-24, 1),
        (np.float32, np.float16, 1 + 2**-11, 1),
    ],
    ids=["wider-operands", "round-operands-before-product", "half-result"],
)
def test_mul_out_dtype_prepares_both_operands(
    input_dtype, result_dtype, value, expected, record_property
):
    if not _out_dtype_available(record_property):
        return

    def source(lhs, rhs):
        return jax.lax.mul(lhs, rhs, out_dtype=result_dtype), lhs, rhs

    inputs = [np.asarray([value], input_dtype), np.asarray([value], input_dtype)]
    with jax.enable_x64():
        model, reference = check(source, inputs)
    assert reference[0].dtype == result_dtype
    np.testing.assert_array_equal(reference[0], np.asarray([expected], result_dtype))
    np.testing.assert_array_equal(reference[1], inputs[0])
    np.testing.assert_array_equal(reference[2], inputs[1])

    types = {
        value.name: value.type.tensor_type.elem_type
        for value in (
            *model.graph.input,
            *model.graph.value_info,
            *model.graph.output,
        )
    }
    work_type = (
        onnx.TensorProto.DOUBLE
        if result_dtype is np.float64
        else onnx.TensorProto.FLOAT
    )
    (multiplication,) = [node for node in model.graph.node if node.op_type == "Mul"]
    assert all(types[name] == work_type for name in multiplication.input)
    assert types[multiplication.output[0]] == work_type


def test_mul_complex_out_dtype_preserves_wide_product_and_shared_operand(
    record_property,
):
    if not _out_dtype_available(record_property):
        return

    def source(real, imag):
        original = jax.lax.complex(real, imag)
        product = jax.lax.mul(original, original, out_dtype=jnp.complex128)
        return (
            jnp.real(product),
            jnp.imag(product),
            jnp.real(original),
            jnp.imag(original),
            real,
        )

    inputs = [np.asarray([4097], np.float32), np.asarray([2], np.float32)]
    with jax.enable_x64():
        model, reference = check(source, inputs)
    assert reference[0].dtype == reference[1].dtype == np.float64
    np.testing.assert_array_equal(reference[0], np.asarray([16785405], np.float64))
    np.testing.assert_array_equal(reference[1], np.asarray([16388], np.float64))
    for actual, expected in zip(reference[2:], [*inputs, inputs[0]], strict=True):
        assert actual.dtype == np.float32
        np.testing.assert_array_equal(actual, expected)
    types = {
        value.name: value.type.tensor_type.elem_type
        for value in (*model.graph.value_info, *model.graph.output)
    }
    multiplications = [node for node in model.graph.node if node.op_type == "Mul"]
    assert len(multiplications) == 4
    assert all(
        types[node.output[0]] == onnx.TensorProto.DOUBLE for node in multiplications
    )


def test_mul_integer_out_dtype_prepares_wide_operands(record_property):
    if not _out_dtype_available(record_property):
        return

    def source(lhs, rhs):
        return jax.lax.mul(lhs, rhs, out_dtype=jnp.int64), lhs

    inputs = [np.asarray([65537], np.int32), np.asarray([65537], np.int32)]
    with jax.enable_x64():
        _, reference = check(source, inputs)
    assert reference[0].dtype == np.int64
    np.testing.assert_array_equal(reference[0], np.asarray([4295098369], np.int64))
    np.testing.assert_array_equal(reference[1], inputs[0])


@pytest.mark.parametrize("result_dtype", [np.float64, np.int64])
def test_mul_complex_to_real_converts_operands_before_multiplication(
    result_dtype, record_property
):
    if not _out_dtype_available(record_property):
        return

    def source(real, imag):
        original = jax.lax.complex(real, imag)
        product = jax.lax.mul(original, original, out_dtype=result_dtype)
        return product, jnp.real(original), jnp.imag(original)

    inputs = [np.asarray([4097], np.float32), np.asarray([2], np.float32)]
    with jax.enable_x64():
        _, reference = check(source, inputs)
    assert reference[0].dtype == result_dtype
    # Both operands become real 4097 first. Projecting (4097+2j)**2 would
    # instead produce 16785405, and computing in FP32 would lose the final 1.
    np.testing.assert_array_equal(reference[0], np.asarray([16785409], result_dtype))
    for actual, expected in zip(reference[1:], inputs, strict=True):
        assert actual.dtype == np.float32
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "result_dtype,expected",
    [(np.int8, [-2, 0, -2, 0]), (np.uint8, [144, 0, 254, 0])],
)
def test_mul_narrow_integer_out_dtype_saturates_float_operands(
    result_dtype, expected, record_property
):
    if not _out_dtype_available(record_property):
        return

    def source(lhs, rhs):
        return jax.lax.mul(lhs, rhs, out_dtype=result_dtype), lhs, rhs

    inputs = [
        np.asarray([200, -200, 127.5, -128.5], np.float32),
        np.full((4,), 2, np.float32),
    ]
    with jax.enable_x64():
        _, reference = check(source, inputs)
    assert reference[0].dtype == result_dtype
    np.testing.assert_array_equal(reference[0], np.asarray(expected, result_dtype))
    for actual, original in zip(reference[1:], inputs, strict=True):
        assert actual.dtype == np.float32
        np.testing.assert_array_equal(actual, original)


def test_mul_complex_out_dtype_preserves_broadcast_shape_and_shared_operands(
    record_property,
):
    if not _out_dtype_available(record_property):
        return

    def source(left_real, left_imag, right_real, right_imag):
        left = jax.lax.complex(left_real, left_imag)
        right = jax.lax.complex(right_real, right_imag)
        product = jax.lax.mul(left, right, out_dtype=jnp.complex128)
        return (
            jnp.real(product),
            jnp.imag(product),
            jnp.real(left),
            jnp.imag(left),
            jnp.real(right),
            jnp.imag(right),
        )

    inputs = [
        np.asarray([[4097], [2]], np.float32),
        np.asarray([[2], [-1]], np.float32),
        np.asarray([[4097, 1, -2]], np.float32),
        np.asarray([[2, 3, 1]], np.float32),
    ]
    with jax.enable_x64():
        model, reference = check(source, inputs)
    expected = (inputs[0].astype(np.float64) + 1j * inputs[1].astype(np.float64)) * (
        inputs[2].astype(np.float64) + 1j * inputs[3].astype(np.float64)
    )
    for actual, wanted in zip(
        reference[:2], (expected.real, expected.imag), strict=True
    ):
        assert actual.dtype == np.float64
        assert actual.shape == (2, 3)
        np.testing.assert_array_equal(actual, wanted)
    for actual, original in zip(reference[2:], inputs, strict=True):
        assert actual.dtype == np.float32
        np.testing.assert_array_equal(actual, original)
    shapes = {
        value.name: tuple(dim.dim_value for dim in value.type.tensor_type.shape.dim)
        for value in (*model.graph.value_info, *model.graph.output)
    }
    products = [node for node in model.graph.node if node.op_type == "Mul"]
    assert len(products) == 4
    assert all(shapes[node.output[0]] == (2, 3) for node in products)


@pytest.mark.parametrize(
    "input_dtype,values,factors,expected",
    [
        (np.float32, [4097, -3], [4097, 2], [16785409, -6]),
        (
            np.int64,
            [2**40 + 1, -(2**40 + 3)],
            [1, 1],
            [2**40 + 1, -(2**40 + 3)],
        ),
    ],
    ids=["real-length-two", "wide-int64"],
)
def test_mul_known_real_operands_pack_by_source_dtype(
    input_dtype, values, factors, expected, record_property
):
    if not _out_dtype_available(record_property):
        return

    def source(lhs, rhs):
        product = jax.lax.mul(lhs, rhs, out_dtype=jnp.complex128)
        return jnp.real(product), jnp.imag(product), lhs, rhs

    inputs = [np.asarray(values, input_dtype), np.asarray(factors, input_dtype)]
    with jax.enable_x64():
        model, reference = check(source, inputs)
    assert reference[0].dtype == reference[1].dtype == np.float64
    np.testing.assert_array_equal(reference[0], np.asarray(expected, np.float64))
    np.testing.assert_array_equal(reference[1], np.zeros((2,), np.float64))
    for actual, original in zip(reference[2:], inputs, strict=True):
        assert actual.dtype == input_dtype
        np.testing.assert_array_equal(actual, original)
    types = {
        value.name: value.type.tensor_type.elem_type
        for value in (*model.graph.value_info, *model.graph.output)
    }
    products = [node for node in model.graph.node if node.op_type == "Mul"]
    assert len(products) == 4
    assert all(types[node.output[0]] == onnx.TensorProto.DOUBLE for node in products)
