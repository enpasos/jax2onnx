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


def test_jax_nn_dpa_plugin_world_accepts_grouped_query_attention() -> None:
    import jax.nn as jax_nn

    import_all_plugins()

    q = np.linspace(-1.0, 1.0, num=32, dtype=np.float32).reshape((1, 2, 4, 4))
    k = np.linspace(-0.5, 0.5, num=24, dtype=np.float32).reshape((1, 3, 2, 4))
    v = np.linspace(0.25, 1.25, num=24, dtype=np.float32).reshape((1, 3, 2, 4))

    expected = np.asarray(jax_nn.dot_product_attention(q, k, v))

    with _activate_plugin_worlds():
        got = np.asarray(jax_nn.dot_product_attention(q, k, v))

    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_jax_nn_dpa_plugin_world_accepts_unbatched_tnh_inputs() -> None:
    import jax.nn as jax_nn

    import_all_plugins()

    q = np.linspace(-1.0, 1.0, num=4 * 2 * 8, dtype=np.float32).reshape((4, 2, 8))
    k = np.linspace(-0.5, 0.5, num=5 * 2 * 8, dtype=np.float32).reshape((5, 2, 8))
    v = np.linspace(0.25, 1.25, num=5 * 2 * 8, dtype=np.float32).reshape((5, 2, 8))

    expected = np.asarray(jax_nn.dot_product_attention(q, k, v))

    with _activate_plugin_worlds():
        got = np.asarray(jax_nn.dot_product_attention(q, k, v))

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


def test_jax_nn_dpa_plugin_world_accepts_default_return_residual() -> None:
    import jax.nn as jax_nn

    import_all_plugins()

    q = np.arange(8, dtype=np.float32).reshape((1, 2, 1, 4))
    k = np.arange(12, dtype=np.float32).reshape((1, 3, 1, 4))
    v = (np.arange(12, dtype=np.float32) + 10.0).reshape((1, 3, 1, 4))

    expected = np.asarray(jax_nn.dot_product_attention(q, k, v, return_residual=False))

    with _activate_plugin_worlds():
        got = np.asarray(jax_nn.dot_product_attention(q, k, v, return_residual=False))

    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
