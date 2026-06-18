# tests/extra_tests/framework/test_jax_compat.py

from __future__ import annotations

import jax.numpy as jnp

from jax2onnx.plugins.jax import _jax_compat as compat


def test_jax_compat_exports_core_types() -> None:
    prim = compat.Primitive("jax2onnx_test_compat")
    aval = compat.ShapedArray((2, 3), jnp.float32)

    assert prim.name == "jax2onnx_test_compat"
    assert aval.shape == (2, 3)
    assert aval.dtype == jnp.float32


def test_jax_compat_exposes_not_mapped_alias() -> None:
    assert compat.ensure_batching_not_mapped_attr() is compat.NOT_MAPPED
    assert compat.batching.not_mapped is compat.NOT_MAPPED
