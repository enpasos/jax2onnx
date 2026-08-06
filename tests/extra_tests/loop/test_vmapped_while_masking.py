from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import onnxruntime as ort

import onnx
from jax2onnx import to_onnx


def test_vmapped_while_masks_finished_examples() -> None:
    def loop(x: jax.Array) -> jax.Array:
        return jax.lax.while_loop(
            lambda value: jnp.mean(value) < 3.0,
            lambda value: value + 1.0,
            x,
        )

    fn = jax.vmap(loop)
    x = jnp.asarray([[1.0, 1.0], [2.5, 2.5]], dtype=jnp.float32)

    model = to_onnx(fn, [x], opset=18)
    onnx.checker.check_model(model)
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    actual = session.run(None, {model.graph.input[0].name: np.asarray(x)})[0]
    expected = np.asarray(fn(x))

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual[0], np.asarray([3.0, 3.0]))
    np.testing.assert_allclose(actual[1], np.asarray([3.5, 3.5]))
