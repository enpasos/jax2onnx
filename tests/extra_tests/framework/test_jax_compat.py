# tests/extra_tests/framework/test_jax_compat.py

from __future__ import annotations

import jax.numpy as jnp
from pytest import MonkeyPatch

from jax2onnx._compat import jax as compat


def test_jax_compat_exports_core_types() -> None:
    prim = compat.Primitive("jax2onnx_test_compat")
    aval = compat.ShapedArray((2, 3), jnp.float32)

    assert prim.name == "jax2onnx_test_compat"
    assert aval.shape == (2, 3)
    assert aval.dtype == jnp.float32
    assert isinstance(compat.Literal, type)


def test_jax_compat_literal_falls_back_to_jax_core(
    monkeypatch: MonkeyPatch,
) -> None:
    class FallbackLiteral:
        pass

    monkeypatch.delattr(compat.jax_core_ext, "Literal", raising=False)
    monkeypatch.setattr(compat.jax_core, "Literal", FallbackLiteral, raising=False)

    assert compat._resolve_literal_type() is FallbackLiteral


def test_jax_compat_exposes_not_mapped_alias() -> None:
    assert compat.ensure_batching_not_mapped_attr() is compat.NOT_MAPPED
    assert compat.batching.not_mapped is compat.NOT_MAPPED


def test_jax_compat_exposes_shape_equality_helper() -> None:
    assert compat.definitely_equal_shape((2, 3), (2, 3))
    assert not compat.definitely_equal_shape((2, 3), (2, 4))
