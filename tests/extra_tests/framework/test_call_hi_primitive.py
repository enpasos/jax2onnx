# tests/extra_tests/framework/test_call_hi_primitive.py

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import pytest

from jax2onnx import allclose, to_onnx


def test_nested_nonzero_reuses_captured_constants(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    thresholds = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)

    @jax.jit
    def select(x: jax.Array) -> tuple[jax.Array, ...]:
        return jnp.nonzero(x > thresholds, size=4, fill_value=(-2, -3))

    @jax.jit
    def nested(x: jax.Array) -> tuple[tuple[jax.Array, ...], ...]:
        return select(x), select(x + 1)

    def fn(x: jax.Array) -> tuple[tuple[jax.Array, ...], ...]:
        return *nested(x), select(x - 1)

    inputs = [np.array([[0, 3, 1], [8, 4, 6]], dtype=np.int32)]
    model_path = tmp_path / "nested_nonzero.onnx"
    to_onnx(fn, inputs=inputs, return_mode="file", output_path=model_path)

    onnx.checker.check_model(onnx.load(model_path), full_check=True)
    matches, message = allclose(fn, str(model_path), inputs)
    assert matches, message


def test_argwhere_in_cond_preserves_multiple_coordinates(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    mask = np.array([[True, False, True], [False, True, False]])

    def fn(x: jax.Array) -> jax.Array:
        return jax.lax.cond(
            jnp.sum(x) > 0,
            lambda y: jnp.argwhere((y > 0) & mask, size=4, fill_value=-1),
            lambda y: jnp.argwhere((y < 0) & mask, size=4, fill_value=-1),
            x,
        )

    inputs = [np.array([[1, -2, 3], [4, 5, -6]], dtype=np.int32)]
    model_path = tmp_path / "cond_argwhere.onnx"
    to_onnx(fn, inputs=inputs, return_mode="file", output_path=model_path)

    onnx.checker.check_model(onnx.load(model_path), full_check=True)
    for values in (inputs, [-inputs[0]]):
        matches, message = allclose(fn, str(model_path), values)
        assert matches, message
