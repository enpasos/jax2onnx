# tests/extra_tests/framework/test_public_api_compat.py

from __future__ import annotations

import numpy as np

from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.plugins.plugin_system import import_all_plugins


def test_jnp_clip_plugin_world_accepts_legacy_and_current_keywords() -> None:
    import jax.numpy as jnp

    import_all_plugins()

    x = np.array([-1.0, 0.0, 2.0], dtype=np.float32)
    expected = np.clip(x, -0.5, 0.25)

    with _activate_plugin_worlds():
        got_legacy = np.asarray(jnp.clip(x, a_min=-0.5, a_max=0.25))
        got_current = np.asarray(jnp.clip(x, min=-0.5, max=0.25))

    np.testing.assert_allclose(got_legacy, expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(got_current, expected, rtol=0.0, atol=0.0)


def test_jax_nn_dpa_plugin_world_accepts_bias_argument() -> None:
    import jax.nn as jax_nn

    import_all_plugins()

    q = np.arange(8, dtype=np.float32).reshape((1, 2, 1, 4))
    k = np.arange(12, dtype=np.float32).reshape((1, 3, 1, 4))
    v = (np.arange(12, dtype=np.float32) + 10.0).reshape((1, 3, 1, 4))
    bias = np.linspace(-0.5, 0.5, num=6, dtype=np.float32).reshape((1, 1, 2, 3))

    expected = np.asarray(jax_nn.dot_product_attention(q, k, v, bias=bias))

    with _activate_plugin_worlds():
        got = np.asarray(jax_nn.dot_product_attention(q, k, v, bias=bias))

    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_jax_nn_dpa_plugin_world_accepts_scale_argument() -> None:
    import jax.nn as jax_nn

    import_all_plugins()

    q = np.arange(8, dtype=np.float32).reshape((1, 2, 1, 4))
    k = np.arange(12, dtype=np.float32).reshape((1, 3, 1, 4))
    v = (np.arange(12, dtype=np.float32) + 10.0).reshape((1, 3, 1, 4))

    expected = np.asarray(jax_nn.dot_product_attention(q, k, v, scale=0.5))

    with _activate_plugin_worlds():
        got = np.asarray(jax_nn.dot_product_attention(q, k, v, scale=0.5))

    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_jax_nn_dpa_plugin_world_accepts_local_window_size_argument() -> None:
    import jax.nn as jax_nn

    import_all_plugins()

    q = np.arange(64, dtype=np.float32).reshape((1, 16, 1, 4))
    k = (np.arange(64, dtype=np.float32) - 3.0).reshape((1, 16, 1, 4))
    v = (np.arange(64, dtype=np.float32) + 10.0).reshape((1, 16, 1, 4))

    expected = np.asarray(
        jax_nn.dot_product_attention(q, k, v, local_window_size=(1, 1))
    )

    with _activate_plugin_worlds():
        got = np.asarray(
            jax_nn.dot_product_attention(q, k, v, local_window_size=(1, 1))
        )

    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
