# tests/extra_tests/framework/test_ad_rule_no_override.py

from __future__ import annotations

from jax.interpreters import ad

from jax2onnx.plugins.jax._autodiff_utils import (
    backfill_missing_transpose_rules,
    register_transpose_via_linear_transpose,
)
from jax2onnx.plugins.jax.numpy.sum import JnpSumPlugin, _sum_impl


def test_generic_transpose_registration_does_not_override_existing_rule():
    original_rule = ad.primitive_transposes[JnpSumPlugin._PRIM]

    changed = register_transpose_via_linear_transpose(
        JnpSumPlugin._PRIM,
        _sum_impl,
        override=False,
    )

    assert not changed
    assert ad.primitive_transposes[JnpSumPlugin._PRIM] is original_rule


def test_backfill_does_not_override_existing_rule():
    original_rule = ad.primitive_transposes[JnpSumPlugin._PRIM]
    stats = backfill_missing_transpose_rules(
        [JnpSumPlugin._PRIM],
        allowlist={JnpSumPlugin._PRIM.name},
    )

    assert stats.installed == 0
    assert ad.primitive_transposes[JnpSumPlugin._PRIM] is original_rule
