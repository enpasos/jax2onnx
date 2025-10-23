# tests/extra_tests/test_vmap_batching_regression.py

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax2onnx.user_interface import to_onnx


_NN_CASES = [
    ("relu", lambda x: jax.nn.relu(x)),
    ("sigmoid", lambda x: jax.nn.sigmoid(x)),
    ("softplus", lambda x: jax.nn.softplus(x)),
    ("leaky_relu", lambda x: jax.nn.leaky_relu(x, negative_slope=0.2)),
    ("celu", lambda x: jax.nn.celu(x, alpha=0.3)),
]


@pytest.mark.parametrize(("name", "op"), _NN_CASES, ids=[case[0] for case in _NN_CASES])
def test_vmap_unary_nn_primitives(name: str, op):
    data = jnp.linspace(-1.0, 1.0, num=12, dtype=jnp.float32).reshape(3, 4)

    def model(x):
        return jax.vmap(op)(x)

    result = to_onnx(
        model,
        [jax.ShapeDtypeStruct(data.shape, data.dtype)],
        model_name=f"vmap_nn_{name}",
        return_mode="ir",
    )
    assert result is not None


def test_vmap_jnp_stack_all_mapped():
    data = np.arange(12, dtype=np.float32).reshape(3, 4)

    def model(x):
        def inner(row):
            return jnp.stack((row, row + 1.0), axis=1)

        return jax.vmap(inner)(x)

    result = to_onnx(
        model,
        [jax.ShapeDtypeStruct(data.shape, data.dtype)],
        model_name="vmap_stack_mapped",
        return_mode="ir",
    )
    assert result is not None


def test_vmap_jnp_stack_mixed_mapped():
    data = np.arange(6, dtype=np.float32).reshape(3, 2)
    const = jnp.array([1.0, -1.0], dtype=jnp.float32)

    def model(x):
        def inner(row):
            return jnp.stack((row, const), axis=0)

        return jax.vmap(inner)(x)

    result = to_onnx(
        model,
        [jax.ShapeDtypeStruct(data.shape, data.dtype)],
        model_name="vmap_stack_mixed",
        return_mode="ir",
    )
    assert result is not None
