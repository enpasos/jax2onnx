import pytest
import numpy as np
import jax.numpy as jnp
from jax2onnx.transpose_utils import jax_shape_to_onnx_shape, onnx_shape_to_jax_shape, transpose_to_onnx, transpose_to_jax


@pytest.mark.parametrize("jax_shape,expected_onnx_shape", [
    ((1, 64, 64, 3), (1, 3, 64, 64)),  # Typical 4D tensor
    ((1, 10), (1, 10)),  # 2D tensor
    ((5, 32, 32, 16), (5, 16, 32, 32)),  # Larger 4D tensor
    ((2, 2), (2, 2))  # Edge case with 2D shape
])
def test_jax_shape_to_onnx_shape(jax_shape, expected_onnx_shape):
    assert jax_shape_to_onnx_shape(jax_shape) == expected_onnx_shape


@pytest.mark.parametrize("onnx_shape,expected_jax_shape", [
    ((1, 3, 64, 64), (1, 64, 64, 3)),  # Typical 4D tensor
    ((1, 10), (1, 10)),  # 2D tensor
    ((5, 16, 32, 32), (5, 32, 32, 16)),  # Larger 4D tensor
    ((2, 2), (2, 2))  # Edge case with 2D shape
])
def test_onnx_shape_to_jax_shape(onnx_shape, expected_jax_shape):
    assert onnx_shape_to_jax_shape(onnx_shape) == expected_jax_shape


@pytest.mark.parametrize("array,expected_transposed_shape", [
    (jnp.ones((1, 64, 64, 3)), (1, 3, 64, 64)),  # Typical 4D tensor
    (jnp.ones((1, 10)), (1, 10)),  # 2D tensor
    (jnp.ones((5, 32, 32, 16)), (5, 16, 32, 32)),  # Larger 4D tensor
    (jnp.ones((2, 2)), (2, 2))  # Edge case with 2D shape
])
def test_transpose_to_onnx(array, expected_transposed_shape):
    transposed = transpose_to_onnx(array)
    assert transposed.shape == expected_transposed_shape


@pytest.mark.parametrize("array,expected_transposed_shape", [
    (jnp.ones((1, 3, 64, 64)), (1, 64, 64, 3)),  # Typical 4D tensor
    (jnp.ones((1, 10)), (1, 10)),  # 2D tensor
    (jnp.ones((5, 16, 32, 32)), (5, 32, 32, 16)),  # Larger 4D tensor
    (jnp.ones((2, 2)), (2, 2))  # Edge case with 2D shape
])
def test_transpose_to_jax(array, expected_transposed_shape):
    transposed = transpose_to_jax(array)
    assert transposed.shape == expected_transposed_shape
