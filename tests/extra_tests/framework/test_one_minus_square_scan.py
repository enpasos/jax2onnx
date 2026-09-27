# tests/extra_tests/framework/test_one_minus_square_scan.py

"""Regression for JAX's one_minus_square inside a float32 scan body."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest

from jax2onnx import to_onnx


@pytest.mark.skipif(
    not hasattr(jax.lax, "one_minus_square_p"),
    reason="This JAX version does not emit one_minus_square",
)
def test_one_minus_square_preserves_float32_in_double_precision_scan() -> None:
    def fn(x: jax.Array) -> jax.Array:
        seed = x.astype(jnp.float32)[0]

        def body(carry: jax.Array, unused: None) -> tuple[jax.Array, jax.Array]:
            del unused
            return jax.lax.one_minus_square(carry), carry

        result, _ = jax.lax.scan(body, seed, None, length=2)
        return result

    x = np.asarray([0.3], dtype=np.float64)
    model = to_onnx(fn, [x], enable_double_precision=True)
    onnx.checker.check_model(model, full_check=True)

    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    actual = session.run(None, {session.get_inputs()[0].name: x})[0]
    expected = np.asarray(fn(jnp.asarray(x, dtype=jnp.float32)))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
