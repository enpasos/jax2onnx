# tests/extra_tests/test_cos_precision.py

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest

from jax2onnx import to_onnx


def _op_types(model: onnx.ModelProto) -> set[str]:
    return {node.op_type for node in model.graph.node}


def _run(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    onnx.checker.check_model(model, full_check=True)
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (actual,) = session.run(None, {session.get_inputs()[0].name: x})
    result: np.ndarray = np.asarray(actual)
    return result


@pytest.mark.parametrize("dtype", [np.float32, np.int32], ids=["float32", "int32"])
def test_jnp_cos_emits_cos_below_float64(dtype: type) -> None:
    x: np.ndarray = np.arange(6).astype(dtype)

    model = to_onnx(lambda v: jnp.cos(v), [x])

    assert "Cos" in _op_types(model)
    assert "Sin" not in _op_types(model)
    np.testing.assert_allclose(_run(model, x), np.cos(x.astype(np.float64)), atol=1e-6)


def test_jnp_cos_float64_uses_sin_for_onnxruntime() -> None:
    x = np.linspace(-4.0, 4.0, 9)

    model = to_onnx(lambda v: jnp.cos(v), [x], enable_double_precision=True)

    assert {"Add", "Sin"} <= _op_types(model)
    assert "Cos" not in _op_types(model)
    np.testing.assert_allclose(_run(model, x), np.cos(x), rtol=0, atol=1e-12)


def test_jnp_cos_is_accurate_for_rotary_angles() -> None:
    # RoPE-style angle table; cos(x) as sin(x + pi/2) loses ~2.6e-5 here.
    positions: np.ndarray = np.arange(1024, dtype=np.float32)[:, None]
    frequencies = (1.0 / 10_000 ** (np.arange(0, 64, 2, dtype=np.float32) / 64))[None]
    angles = (positions * frequencies).astype(np.float32)

    model = to_onnx(lambda v: jnp.cos(v), [angles])
    expected = np.cos(angles.astype(np.float64))

    np.testing.assert_allclose(_run(model, angles), expected, rtol=0, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(jax.jit(jnp.cos)(angles)), expected, rtol=0, atol=1e-6
    )
