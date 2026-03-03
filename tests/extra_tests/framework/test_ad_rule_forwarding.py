# tests/extra_tests/framework/test_ad_rule_forwarding.py

from __future__ import annotations

import pytest
from jax import lax
from jax.extend.core import Primitive
from jax.interpreters import ad

from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.plugins.jax import _autodiff_utils as ad_utils
from jax2onnx.plugins.jax._autodiff_utils import (
    get_original_rule_forwarding_allowlist,
    get_original_rule_forwarding_blocklist,
    register_original_rule_forwarding,
)
from jax2onnx.plugins.jax.numpy.add import JnpAddPlugin
from jax2onnx.plugins.jax.numpy.concatenate import JnpConcatenatePlugin
from jax2onnx.plugins.jax.numpy.moveaxis import JnpMoveaxisPlugin
from jax2onnx.plugins.jax.numpy.reshape import JnpReshapePlugin
from jax2onnx.plugins.jax.numpy.select import JnpSelectPlugin
from jax2onnx.plugins.jax.numpy.split import JnpSplitPlugin
from jax2onnx.plugins.jax.numpy.stack import JnpStackPlugin
from jax2onnx.plugins.jax.numpy.squeeze import JnpSqueezePlugin
from jax2onnx.plugins.jax.numpy.take import JnpTakePlugin
from jax2onnx.plugins.jax.numpy.tile import JnpTilePlugin
from jax2onnx.plugins.jax.numpy.transpose import JnpTransposePlugin
from jax2onnx.plugins.jax.numpy.where import JnpWherePlugin
from jax2onnx.plugins.plugin_system import import_all_plugins


def test_add_forwarding_mapping_is_allowlisted() -> None:
    assert ("add", "jax.numpy.add") in get_original_rule_forwarding_allowlist()
    assert (
        "concatenate",
        "jax.numpy.concatenate",
    ) in get_original_rule_forwarding_allowlist()
    assert (
        "transpose",
        "jax.numpy.moveaxis",
    ) in get_original_rule_forwarding_allowlist()
    assert ("reshape", "jax.numpy.reshape") in get_original_rule_forwarding_allowlist()
    assert ("split", "jax.numpy.split") in get_original_rule_forwarding_allowlist()
    assert ("squeeze", "jax.numpy.squeeze") in get_original_rule_forwarding_allowlist()
    assert ("tile", "jax.numpy.tile") in get_original_rule_forwarding_allowlist()
    assert (
        "transpose",
        "jax.numpy.transpose",
    ) in get_original_rule_forwarding_allowlist()


def test_forwarding_allowlist_matches_curated_snapshot() -> None:
    assert get_original_rule_forwarding_allowlist() == frozenset(
        {
            ("add", "jax.numpy.add"),
            ("concatenate", "jax.numpy.concatenate"),
            ("reshape", "jax.numpy.reshape"),
            ("split", "jax.numpy.split"),
            ("squeeze", "jax.numpy.squeeze"),
            ("tile", "jax.numpy.tile"),
            ("transpose", "jax.numpy.moveaxis"),
            ("transpose", "jax.numpy.transpose"),
        }
    )


def test_known_non_1_to_1_mappings_are_not_allowlisted() -> None:
    allowlist = get_original_rule_forwarding_allowlist()
    blocked_pairs = get_original_rule_forwarding_blocklist()
    assert blocked_pairs == frozenset(
        {
            ("concatenate", "jax.numpy.stack"),
            ("gather", "jax.numpy.take"),
            ("select_n", "jax.numpy.select"),
            ("select_n", "jax.numpy.where"),
        }
    )
    assert blocked_pairs.isdisjoint(allowlist)


def test_forwarding_policy_overlap_guard_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allowlist = set(get_original_rule_forwarding_allowlist())
    blocklist = set(get_original_rule_forwarding_blocklist())
    overlap_pair = ("add", "jax.numpy.add")
    blocklist.add(overlap_pair)

    monkeypatch.setattr(ad_utils, "_ORIGINAL_RULE_FORWARDING_ALLOWLIST", allowlist)
    monkeypatch.setattr(ad_utils, "_ORIGINAL_RULE_FORWARDING_BLOCKLIST", blocklist)

    with pytest.raises(RuntimeError, match="allowlist/blocklist overlap"):
        ad_utils._validate_forwarding_policy_sets()


def test_register_original_rule_forwarding_respects_override_flag() -> None:
    orig_prim = Primitive("test.original_rule_forwarding.orig")
    new_prim = Primitive("test.original_rule_forwarding.new")

    @orig_prim.def_impl
    def _orig_impl(x):
        return x

    @new_prim.def_impl
    def _new_impl(x):
        return x

    def _orig_jvp(primals, tangents, **_):
        return primals[0], tangents[0]

    def _orig_transpose(ct, *args, **_):
        return tuple(
            ct if isinstance(arg, ad.UndefinedPrimal) else None for arg in args
        )

    def _existing_jvp(primals, tangents, **_):
        return primals[0], tangents[0]

    def _existing_transpose(ct, *args, **_):
        return tuple(
            ct if isinstance(arg, ad.UndefinedPrimal) else None for arg in args
        )

    ad.primitive_jvps[orig_prim] = _orig_jvp
    ad.primitive_transposes[orig_prim] = _orig_transpose
    ad.primitive_jvps[new_prim] = _existing_jvp
    ad.primitive_transposes[new_prim] = _existing_transpose

    try:
        allowlist = {(orig_prim.name, new_prim.name)}

        register_original_rule_forwarding(
            orig_prim=orig_prim,
            new_prim=new_prim,
            allowlist=allowlist,
            override=False,
            forward_batching=False,
        )
        assert ad.primitive_jvps[new_prim] is _existing_jvp
        assert ad.primitive_transposes[new_prim] is _existing_transpose

        register_original_rule_forwarding(
            orig_prim=orig_prim,
            new_prim=new_prim,
            allowlist=allowlist,
            override=True,
            forward_batching=False,
        )
        assert ad.primitive_jvps[new_prim] is _orig_jvp
        assert ad.primitive_transposes[new_prim] is _orig_transpose
    finally:
        ad.primitive_jvps.pop(orig_prim, None)
        ad.primitive_transposes.pop(orig_prim, None)
        ad.primitive_jvps.pop(new_prim, None)
        ad.primitive_transposes.pop(new_prim, None)


def test_known_non_forwardable_primitives_keep_non_forwarded_ad_rules() -> None:
    import_all_plugins()
    with _activate_plugin_worlds():
        assert (
            ad.primitive_jvps[JnpWherePlugin._PRIM]
            is not ad.primitive_jvps[lax.select_n_p]
        )
        assert (
            ad.primitive_transposes[JnpWherePlugin._PRIM]
            is not ad.primitive_transposes[lax.select_n_p]
        )

        assert (
            ad.primitive_jvps[JnpSelectPlugin._PRIM]
            is not ad.primitive_jvps[lax.select_n_p]
        )
        assert (
            ad.primitive_transposes[JnpSelectPlugin._PRIM]
            is not ad.primitive_transposes[lax.select_n_p]
        )

        assert (
            ad.primitive_jvps[JnpTakePlugin._PRIM]
            is not ad.primitive_jvps[lax.gather_p]
        )
        assert (
            ad.primitive_transposes[JnpTakePlugin._PRIM]
            is not ad.primitive_transposes[lax.gather_p]
        )

        assert (
            ad.primitive_jvps[JnpStackPlugin._PRIM]
            is not ad.primitive_jvps[lax.concatenate_p]
        )
        assert (
            ad.primitive_transposes[JnpStackPlugin._PRIM]
            is not ad.primitive_transposes[lax.concatenate_p]
        )


def test_add_uses_forwarded_original_ad_rules() -> None:
    import_all_plugins()
    with _activate_plugin_worlds():
        assert ad.primitive_jvps[JnpAddPlugin._PRIM] is ad.primitive_jvps[lax.add_p]
        assert (
            ad.primitive_transposes[JnpAddPlugin._PRIM]
            is ad.primitive_transposes[lax.add_p]
        )


def test_transpose_uses_forwarded_original_ad_rules() -> None:
    import_all_plugins()
    with _activate_plugin_worlds():
        assert (
            ad.primitive_jvps[JnpTransposePlugin._PRIM]
            is ad.primitive_jvps[lax.transpose_p]
        )
        assert (
            ad.primitive_transposes[JnpTransposePlugin._PRIM]
            is ad.primitive_transposes[lax.transpose_p]
        )


def test_squeeze_uses_forwarded_original_ad_rules() -> None:
    import_all_plugins()
    with _activate_plugin_worlds():
        assert (
            ad.primitive_jvps[JnpSqueezePlugin._PRIM]
            is ad.primitive_jvps[lax.squeeze_p]
        )
        assert (
            ad.primitive_transposes[JnpSqueezePlugin._PRIM]
            is ad.primitive_transposes[lax.squeeze_p]
        )


def test_concatenate_uses_forwarded_original_ad_rules() -> None:
    import_all_plugins()
    with _activate_plugin_worlds():
        assert (
            ad.primitive_jvps[JnpConcatenatePlugin._PRIM]
            is ad.primitive_jvps[lax.concatenate_p]
        )
        assert (
            ad.primitive_transposes[JnpConcatenatePlugin._PRIM]
            is ad.primitive_transposes[lax.concatenate_p]
        )


def test_reshape_uses_forwarded_original_ad_rules() -> None:
    import_all_plugins()
    with _activate_plugin_worlds():
        assert (
            ad.primitive_jvps[JnpReshapePlugin._PRIM]
            is ad.primitive_jvps[lax.reshape_p]
        )
        assert (
            ad.primitive_transposes[JnpReshapePlugin._PRIM]
            is ad.primitive_transposes[lax.reshape_p]
        )


def test_split_uses_forwarded_original_ad_rules() -> None:
    import_all_plugins()
    with _activate_plugin_worlds():
        assert ad.primitive_jvps[JnpSplitPlugin._PRIM] is ad.primitive_jvps[lax.split_p]
        assert (
            ad.primitive_transposes[JnpSplitPlugin._PRIM]
            is ad.primitive_transposes[lax.split_p]
        )


def test_moveaxis_uses_forwarded_original_ad_rules() -> None:
    import_all_plugins()
    with _activate_plugin_worlds():
        assert (
            ad.primitive_jvps[JnpMoveaxisPlugin._PRIM]
            is ad.primitive_jvps[lax.transpose_p]
        )
        assert (
            ad.primitive_transposes[JnpMoveaxisPlugin._PRIM]
            is ad.primitive_transposes[lax.transpose_p]
        )


def test_tile_uses_forwarded_original_ad_rules() -> None:
    import_all_plugins()
    with _activate_plugin_worlds():
        assert ad.primitive_jvps[JnpTilePlugin._PRIM] is ad.primitive_jvps[lax.tile_p]
        assert (
            ad.primitive_transposes[JnpTilePlugin._PRIM]
            is ad.primitive_transposes[lax.tile_p]
        )
