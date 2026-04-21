# tests/extra_tests/converter/test_exact_dtype_casts.py

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir

from jax2onnx.user_interface import to_onnx


def _convert_to_float32(x: jax.Array) -> jax.Array:
    return jax.lax.convert_element_type(x, jnp.float32)


def _bitcast_to_float32(x: jax.Array) -> jax.Array:
    return jax.lax.bitcast_convert_type(x, new_dtype=jnp.float32)


def _zeros_float32() -> jax.Array:
    return jnp.zeros((2,), dtype=jnp.float32)


def _ones_float32() -> jax.Array:
    return jnp.ones((2,), dtype=jnp.float32)


def _eye_float32() -> jax.Array:
    return jnp.eye(2, dtype=jnp.float32)


def _full_float32() -> jax.Array:
    return jnp.full((2,), 1.5, dtype=jnp.float32)


def _linspace_float32() -> jax.Array:
    return jnp.linspace(0.0, 1.0, num=3, dtype=jnp.float32)


def _arange_float32() -> jax.Array:
    return jnp.arange(1.0, 2.0, 0.25, dtype=jnp.float32)


def test_convert_element_type_preserves_explicit_float32_target_under_x64() -> None:
    model = to_onnx(
        _convert_to_float32,
        [jax.ShapeDtypeStruct((2,), jnp.float64)],
        return_mode="ir",
        enable_double_precision=True,
    )

    assert model.graph.outputs[0].type is not None
    assert model.graph.outputs[0].type.dtype == ir.DataType.FLOAT


def test_zeros_preserves_explicit_float32_dtype_under_x64() -> None:
    model = to_onnx(
        _zeros_float32,
        [],
        return_mode="ir",
        enable_double_precision=True,
    )

    assert model.graph.outputs[0].type is not None
    assert model.graph.outputs[0].type.dtype == ir.DataType.FLOAT


def test_ones_preserves_explicit_float32_dtype_under_x64() -> None:
    model = to_onnx(
        _ones_float32,
        [],
        return_mode="ir",
        enable_double_precision=True,
    )

    assert model.graph.outputs[0].type is not None
    assert model.graph.outputs[0].type.dtype == ir.DataType.FLOAT


def test_eye_preserves_explicit_float32_dtype_under_x64() -> None:
    model = to_onnx(
        _eye_float32,
        [],
        return_mode="ir",
        enable_double_precision=True,
    )

    assert model.graph.outputs[0].type is not None
    assert model.graph.outputs[0].type.dtype == ir.DataType.FLOAT


def test_full_preserves_explicit_float32_dtype_under_x64() -> None:
    model = to_onnx(
        _full_float32,
        [],
        return_mode="ir",
        enable_double_precision=True,
    )

    assert model.graph.outputs[0].type is not None
    assert model.graph.outputs[0].type.dtype == ir.DataType.FLOAT


def test_linspace_preserves_explicit_float32_dtype_under_x64() -> None:
    model = to_onnx(
        _linspace_float32,
        [],
        return_mode="ir",
        enable_double_precision=True,
    )

    assert model.graph.outputs[0].type is not None
    assert model.graph.outputs[0].type.dtype == ir.DataType.FLOAT


def test_arange_preserves_explicit_float32_dtype_under_x64() -> None:
    model = to_onnx(
        _arange_float32,
        [],
        return_mode="ir",
        enable_double_precision=True,
    )

    assert model.graph.outputs[0].type is not None
    assert model.graph.outputs[0].type.dtype == ir.DataType.FLOAT


def test_bitcast_convert_type_preserves_explicit_float32_target_under_x64() -> None:
    model = to_onnx(
        _bitcast_to_float32,
        [jax.ShapeDtypeStruct((2,), np.uint32)],
        return_mode="ir",
        enable_double_precision=True,
    )

    assert model.graph.outputs[0].type is not None
    assert model.graph.outputs[0].type.dtype == ir.DataType.FLOAT
