# tests/extra_tests/framework/test_jax_compat.py

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import pytest
from pytest import MonkeyPatch

from jax2onnx._compat import jax as compat


def test_jax_compat_exports_core_types() -> None:
    prim = compat.Primitive("jax2onnx_test_compat")
    aval = compat.ShapedArray((2, 3), jnp.float32)

    assert prim.name == "jax2onnx_test_compat"
    assert aval.shape == (2, 3)
    assert aval.dtype == jnp.float32
    assert isinstance(compat.Literal, type)


def test_jax_compat_literal_falls_back_to_jax_src_core(
    monkeypatch: MonkeyPatch,
) -> None:
    from jax._src import core as jax_core_src

    monkeypatch.delattr(compat.jax_core_ext, "Literal", raising=False)
    monkeypatch.delattr(compat.jax_core, "Literal", raising=False)

    assert compat._resolve_literal_type() is jax_core_src.Literal


def test_jax_compat_exposes_not_mapped_alias() -> None:
    assert compat.ensure_batching_not_mapped_attr() is compat.NOT_MAPPED
    assert compat.batching.not_mapped is compat.NOT_MAPPED


def test_jax_compat_exposes_shape_equality_helper() -> None:
    assert compat.definitely_equal_shape((2, 3), (2, 3))
    assert not compat.definitely_equal_shape((2, 3), (2, 4))


def test_jax_compat_concrete_or_error_returns_static_value() -> None:
    assert compat.concrete_or_error(int, 7, "must be static") == 7


def test_fresh_var_like_uses_current_jax_constructor() -> None:
    aval = compat.ShapedArray((2, 3), jnp.float32)
    original = compat.Var(aval)

    fresh = compat.fresh_var_like(original)

    assert isinstance(fresh, compat.Var)
    assert fresh is not original
    assert fresh.aval is aval


def test_fresh_var_like_preserves_legacy_quantization_metadata(
    monkeypatch: MonkeyPatch,
) -> None:
    class LegacyVar:
        def __init__(
            self,
            aval: Any,
            initial_qdd: Any = None,
            final_qdd: Any = None,
        ) -> None:
            self.aval = aval
            self.initial_qdd = initial_qdd
            self.final_qdd = final_qdd

    original = LegacyVar("aval", initial_qdd="initial", final_qdd="final")
    monkeypatch.setattr(compat, "Var", LegacyVar)

    fresh = compat.fresh_var_like(original)

    assert fresh is not original
    assert fresh.aval == "aval"
    assert fresh.initial_qdd == "initial"
    assert fresh.final_qdd == "final"


def _scan_arity_of(fn: Any, *args: Any) -> tuple[int, int, int]:
    closed = jax.make_jaxpr(fn)(*args)
    for eqn in closed.jaxpr.eqns:
        if eqn.primitive.name == "scan":
            inner = eqn.params["jaxpr"]
            return compat.scan_arity(eqn.params, len(inner.jaxpr.invars))
    raise AssertionError("no scan equation found")


def test_scan_arity_without_xs() -> None:
    fn = lambda x: jax.lax.scan(  # noqa: E731
        lambda carry, _: (carry + 1, carry), x, xs=None, length=5
    )[1]
    assert _scan_arity_of(fn, jnp.float32(0.0)) == (0, 1, 0)


def test_scan_arity_with_consts_carry_and_xs() -> None:
    const = jnp.ones((2,))

    def fn(init: Any, xs: Any) -> Any:
        def body(carry: Any, x: Any) -> Any:
            a, b = carry
            return (a + const + x, b + 1.0), a.sum()

        return jax.lax.scan(body, init, xs)

    arity = _scan_arity_of(fn, (jnp.ones((2,)), jnp.float32(0.0)), jnp.ones((4, 2)))
    assert arity == (1, 2, 1)


def test_scan_arity_falls_back_to_legacy_params() -> None:
    # JAX <0.11 exposes integer counts instead of ``ft_in``.
    params = {"num_consts": 2, "num_carry": 1}
    assert compat.scan_arity(params, 5) == (2, 1, 2)


def test_scan_arity_rejects_inconsistent_arity() -> None:
    with pytest.raises(ValueError, match="Inconsistent Scan arity"):
        compat.scan_arity({"num_consts": 2, "num_carry": 1}, 2)
