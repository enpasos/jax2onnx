from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import onnxruntime as ort

import onnx
from jax2onnx import to_onnx


def _assert_ort_parity(fn: Callable[..., object], inputs: Sequence[jax.Array]) -> None:
    model = to_onnx(fn, list(inputs), opset=18)
    onnx.checker.check_model(model)
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    feeds = {
        value_info.name: np.asarray(value)
        for value_info, value in zip(model.graph.input, inputs)
    }
    actual = session.run(None, feeds)
    expected = jax.tree_util.tree_leaves(fn(*inputs))
    assert len(actual) == len(expected)
    for actual_leaf, expected_leaf in zip(actual, expected):
        np.testing.assert_allclose(actual_leaf, np.asarray(expected_leaf))


def test_vmapped_split_uses_logical_axis() -> None:
    fn = jax.vmap(lambda x: jnp.split(x, [1, 3], axis=0))
    x = jnp.arange(2 * 4 * 3, dtype=jnp.float32).reshape(2, 4, 3)

    _assert_ort_parity(fn, [x])


def test_vmapped_split_normalizes_negative_logical_axis() -> None:
    fn = jax.vmap(lambda x: jnp.split(x, [1, 3], axis=-1))
    x = jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4)

    _assert_ort_parity(fn, [x])


def test_split_preserves_zero_length_outputs() -> None:
    def fn(x: jax.Array) -> list[jax.Array]:
        return jnp.split(x, [0, 2, 2, 4], axis=0)

    x = jnp.arange(4 * 3, dtype=jnp.float32).reshape(4, 3)

    _assert_ort_parity(fn, [x])
