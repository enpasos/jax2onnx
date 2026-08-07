# tests/extra_tests/test_scatter_broadcast_isolation.py

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import onnxruntime as ort

import onnx
from jax2onnx import to_onnx


def test_scatter_window_shape_does_not_affect_later_softmax() -> None:
    def one(
        operand: jax.Array, updates: jax.Array, scores: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        scattered = operand.at[:3].add(updates[:, None, :])
        probabilities = jax.nn.softmax(scores, axis=-1)
        return scattered, probabilities

    fn = jax.vmap(one)
    inputs = [
        jnp.arange(5 * 4 * 8, dtype=jnp.float32).reshape(1, 5, 4, 8) / 100,
        jnp.arange(3 * 8, dtype=jnp.float32).reshape(1, 3, 8) / 10,
        jnp.arange(1 * 2 * 3, dtype=jnp.float32).reshape(1, 1, 2, 3),
    ]

    model = to_onnx(fn, inputs, opset=18)
    onnx.checker.check_model(model)
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    feeds = {
        value_info.name: np.asarray(value)
        for value_info, value in zip(model.graph.input, inputs)
    }
    actual = session.run(None, feeds)
    expected = fn(*inputs)
    for actual_leaf, expected_leaf in zip(actual, expected):
        np.testing.assert_allclose(
            actual_leaf, np.asarray(expected_leaf), rtol=1e-5, atol=1e-5
        )
