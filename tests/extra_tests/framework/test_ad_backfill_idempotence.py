# tests/extra_tests/framework/test_ad_backfill_idempotence.py

from __future__ import annotations

from jax.extend.core import Primitive
from jax.interpreters import ad

from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.plugins.jax._autodiff_utils import (
    backfill_missing_transpose_rules,
    register_fallback_jvp_rule,
)
from jax2onnx.plugins.jax.numpy.transpose import JnpTransposePlugin


def _new_temp_identity_primitive(name: str) -> Primitive:
    prim = Primitive(name)
    prim.multiple_results = False

    @prim.def_impl
    def _impl(x):
        return x

    return prim


def test_backfill_is_idempotent_and_reentrant():
    prim = _new_temp_identity_primitive("test.ad_backfill_idempotence")
    register_fallback_jvp_rule(prim, prim.impl, register_transpose=False)
    ad.primitive_transposes.pop(prim, None)
    allowlist = {prim.name}

    try:
        first = backfill_missing_transpose_rules([prim], allowlist=allowlist)
        assert first.installed == 1
        installed_rule = ad.primitive_transposes[prim]

        second = backfill_missing_transpose_rules([prim], allowlist=allowlist)
        assert second.installed == 0
        assert ad.primitive_transposes[prim] is installed_rule
    finally:
        ad.primitive_jvps.pop(prim, None)
        ad.primitive_transposes.pop(prim, None)


def test_backfill_respects_disable_env(monkeypatch):
    prim = _new_temp_identity_primitive("test.ad_backfill_disable_env")
    register_fallback_jvp_rule(prim, prim.impl, register_transpose=False)
    ad.primitive_transposes.pop(prim, None)
    allowlist = {prim.name}

    try:
        monkeypatch.setenv("JAX2ONNX_DISABLE_AD_BACKFILL", "1")
        stats = backfill_missing_transpose_rules([prim], allowlist=allowlist)
        assert stats.disabled
        assert prim not in ad.primitive_transposes
    finally:
        ad.primitive_jvps.pop(prim, None)
        ad.primitive_transposes.pop(prim, None)


def test_backfill_skips_non_allowlisted_primitive_and_reports_stats():
    prim = _new_temp_identity_primitive("test.ad_backfill_not_allowlisted")
    register_fallback_jvp_rule(prim, prim.impl, register_transpose=False)
    ad.primitive_transposes.pop(prim, None)

    try:
        stats = backfill_missing_transpose_rules([prim])
        assert stats.scanned == 1
        assert stats.missing_transpose == 1
        assert stats.installed == 0
        assert stats.skipped_not_allowlisted == 1
        assert prim not in ad.primitive_transposes
    finally:
        ad.primitive_jvps.pop(prim, None)
        ad.primitive_transposes.pop(prim, None)


def test_activation_backfill_path_is_reentrant_for_allowlisted_primitive():
    prim = JnpTransposePlugin._PRIM

    with _activate_plugin_worlds():
        first = ad.primitive_transposes[prim]

    with _activate_plugin_worlds():
        second = ad.primitive_transposes[prim]

    assert first is second
