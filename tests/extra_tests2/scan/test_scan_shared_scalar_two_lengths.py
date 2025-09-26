from __future__ import annotations

import pathlib

import jax.numpy as jnp
from jax import lax
import onnx
import pytest

from jax2onnx.user_interface import to_onnx


def _two_scans_shared_scalar():
    xs5 = jnp.arange(5, dtype=jnp.float32)
    xs100 = jnp.arange(100, dtype=jnp.float32)
    dt = jnp.asarray(0.1, dtype=jnp.float32)

    def body(c, x):
        return (c + x + dt, c)

    _, y5 = lax.scan(body, 0.0, xs5)
    _, y100 = lax.scan(body, 0.0, xs100)
    return y5, y100


def test_shared_scalar_two_lengths(tmp_path: pathlib.Path):
    ort = pytest.importorskip("onnxruntime")

    path = tmp_path / "two_scans_shared_scalar.onnx"
    model = to_onnx(
        _two_scans_shared_scalar,
        inputs=[],
        model_name="two_scans_shared_scalar",
        opset=21,
        use_onnx_ir=True,
    )
    onnx.save_model(model, path)
    ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
