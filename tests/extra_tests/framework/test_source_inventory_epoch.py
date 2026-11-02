# tests/extra_tests/framework/test_source_inventory_epoch.py

"""Current membership and historical evidence retain distinct, complete epochs."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from scripts._source_inventory_epoch import (
    validate_replay_manifest,
    verify_epoch_mapping,
)
from scripts.source_precision_inventory import (
    CURRENT_BASELINE,
    CURRENT_EPOCH_MAPPING,
    DEFAULT_BASELINE,
    HISTORICAL_BASELINE,
)


@pytest.fixture(scope="module")
def epoch():
    return (
        HISTORICAL_BASELINE.read_bytes(),
        CURRENT_BASELINE.read_bytes(),
        json.loads(CURRENT_EPOCH_MAPPING.read_text()),
    )


@pytest.mark.parametrize(
    "manifest,expected",
    (
        (None, "1" * 64),
        ([], "1" * 64),
        ({"rows": {}}, "1" * 64),
        ({"inventory_sha256": "2" * 64}, "1" * 64),
        ({"inventory_sha256": None}, "1" * 64),
        ({"inventory_sha256": "G" * 64}, "1" * 64),
        ({"inventory_sha256": "1" * 63}, "1" * 64),
        ({"inventory_sha256": "1" * 64}, None),
        ({"inventory_sha256": "1" * 64}, "not-a-digest"),
    ),
)
def test_replay_manifest_requires_an_independent_complete_inventory_binding(
    manifest, expected
):
    with pytest.raises(ValueError):
        validate_replay_manifest(manifest, expected)


def test_replay_manifest_accepts_original_bytes_and_explicit_synthetic_empty():
    manifest = json.loads(
        HISTORICAL_BASELINE.with_name("source_replay_adapters.json").read_text()
    )
    before = copy.deepcopy(manifest)
    expected = hashlib.sha256(HISTORICAL_BASELINE.read_bytes()).hexdigest()
    validate_replay_manifest(manifest, expected)
    validate_replay_manifest({}, None)
    assert manifest == before


def test_complete_current_epoch_preserves_original_bytes_and_all_row_links(epoch):
    historical, current, mapping = epoch
    assert hashlib.sha256(historical).hexdigest() == (
        "b875cf3a8169c2919c425eb9d5f11fda9396020945d7ffc4164fb748e153474c"
    )
    adapters = HISTORICAL_BASELINE.with_name("source_replay_adapters.json")
    assert hashlib.sha256(adapters.read_bytes()).hexdigest() == (
        "528be47750063a18d04aee78ce3078b79de867585eda6137c797f0cc803a2042"
    )
    verify_epoch_mapping(historical, current, mapping)
    assert len(json.loads(historical)["cases"]) == len(mapping["rows"]) == 1852
    assert len(json.loads(current)["cases"]) == 1856
    assert len(mapping["added_rows"]) == 4
    assert mapping["reviewed_relocations"] == sorted(
        [
            [
                ["plugin::jax.nn.dot_product_attention", old],
                ["plugin::jax.nn.dot_product_attention", old + 2],
            ]
            for old in range(20, 25)
        ]
        + [
            [["plugin::jax.numpy.where", old], ["plugin::jax.numpy.where", old + 1]]
            for old in (14, 15)
        ]
    )
    old, new = json.loads(historical), json.loads(current)
    assert new["manual_provenance"]["anchor_manual_test_functions"] == 752
    assert new["manual_provenance"]["existing_manual_test_functions"] == 752
    assert new["manual_provenance"]["removed_since_anchor"] == []
    anchors = set(map(tuple, mapping["manual"]["historical"])) - set(
        map(tuple, old["manual_provenance"]["migration_additions"])
    )
    assert len(anchors) == 752
    assert anchors <= set(map(tuple, mapping["manual"]["current"]))
    assert len(mapping["opaque"]["historical"]) == 63
    assert len(mapping["opaque"]["current"]) == 56
    assert len(mapping["metadata_free"]["current"]) == 104
    assert {
        row["registration"][0]: row["raw_case_rows"]
        for row in mapping["registrations"]["added_obligations"]
    } == {
        "plugin::call_hi_primitive": 0,
        "plugin::platform_index": 0,
        "plugin::polynomial": 0,
    }
    assert {
        family: len(record["baseline"]["members"])
        for family, record in mapping["optional"]["current"].items()
    } == {"maxtext": 45, "maxdiffusion": 5}
    variant = mapping["observed_contract_variants"][0]
    assert variant["current"]["origin"] == "example::examples.lax::remat2"
    assert variant["current"]["ordinal"] == 0
    assert variant["current"]["contract_sha256"] == (
        "f2df5d3538ccd546c7a4e44f138e12e2dd29379268c5c404624937831d0fe10e"
    )
    assert variant["observed_contract_sha256"] == (
        "81af003c9bdc7e396a46c63faa9b32f390a640e3869d37de294d60b5f4a23bf3"
    )
    older = mapping["metadata_observations"]["older"]
    assert older["inventory_sha256"] == (
        "facb35036f9eb50657160865e1f78d4b11982523a8a1f0028a47d217fcf10948"
    )
    assert older["receipt_sha256"] == (
        "6a5033e2f16f003e6d5e7299c43fd2d5d8797eb7173ed950bf2695b0689733f2"
    )
    assert older["source_epoch"]["head"] == "b18abaa24009221fe25861aebe004ee40e6a8363"
    assert older["source_epoch"]["files_sha256"] == (
        "af252a5111b6bdf965ad5880321c5ddfdb695d65e3ca115ab66f885bd5997a68"
    )
    assert older["source_held"] is older["environment_held"] is True


def test_inventory_default_is_current_and_executor_keeps_its_historical_adapter():
    from scripts import source_precision_probe as probe

    assert DEFAULT_BASELINE == CURRENT_BASELINE
    assert probe.DEFAULT_BASELINE == HISTORICAL_BASELINE
    assert CURRENT_BASELINE != HISTORICAL_BASELINE


@pytest.mark.parametrize(
    "origin,ordinal",
    (
        ("plugin::bitcast_convert_type", 0),
        ("plugin::jax.nn.dot_product_attention", 20),
        ("plugin::jax.numpy.where", 14),
        ("plugin::jax.numpy.all", 1),
        ("plugin::jax.numpy.any", 1),
    ),
)
def test_changed_live_rows_cannot_execute_through_a_historical_plan(
    epoch, origin, ordinal
):
    from scripts import source_precision_probe as probe
    from scripts.source_precision_inventory import raw_registrations

    plan = probe.build_probe_plan(json.loads(epoch[0]))
    row = next(
        item for item in plan.rows if (item.origin, item.ordinal) == (origin, ordinal)
    )
    registrations = {name: metadata for name, _, metadata in raw_registrations()[0]}
    with pytest.raises(probe.SourceContractDrift):
        probe._resolve_case(row, registrations)


@pytest.mark.parametrize(
    "fault",
    (
        "wrong-manifest",
        "lost-original-row",
        "duplicate-row",
        "same-name-linear-swap",
        "old-bitcast-request",
        "lost-new-row",
        "wrong-relocation",
        "hidden-changed-contract",
        "lost-zero-case-registration",
        "lost-original-manual",
        "lost-opaque",
        "lost-optional",
        "lost-conditional",
        "transported-acceptance",
        "lost-variant",
        "forged-older-closure",
        "forged-older-closure-and-digest",
    ),
)
def test_documentary_mapping_rejects_missing_or_forged_obligations(epoch, fault):
    historical, current, original = epoch
    mapping = copy.deepcopy(original)
    if fault == "wrong-manifest":
        current += b" "
    elif fault == "lost-original-row":
        mapping["rows"].pop()
    elif fault == "duplicate-row":
        mapping["rows"].append(copy.deepcopy(mapping["rows"][0]))
    elif fault == "same-name-linear-swap":
        links = [
            row
            for row in mapping["rows"]
            if row["historical"]["origin"] == "plugin::nnx.linear"
            and row["historical"]["testcase"] == "linear_high_rank_no_bias"
        ]
        assert len(links) == 2
        links[0]["current"], links[1]["current"] = (
            links[1]["current"],
            links[0]["current"],
        )
        links[0]["relocated"] = links[1]["relocated"] = True
    elif fault == "old-bitcast-request":
        row = next(
            item
            for item in mapping["rows"]
            if item["historical"]["origin"] == "plugin::bitcast_convert_type"
        )
        row["historical"]["opset"] = 26
    elif fault == "lost-new-row":
        mapping["added_rows"].pop()
    elif fault == "wrong-relocation":
        next(row for row in mapping["rows"] if row["relocated"])["relocated"] = False
    elif fault == "hidden-changed-contract":
        next(row for row in mapping["rows"] if row["changed_contract_fields"])[
            "changed_contract_fields"
        ] = []
    elif fault == "lost-zero-case-registration":
        mapping["registrations"]["current"] = [
            row
            for row in mapping["registrations"]["current"]
            if row[0] != "plugin::platform_index"
        ]
    elif fault == "lost-original-manual":
        mapping["manual"]["current"].pop(0)
    elif fault == "lost-opaque":
        mapping["opaque"]["historical"].pop()
    elif fault == "lost-optional":
        mapping["optional"]["current"]["maxtext"]["baseline"]["members"].pop()
    elif fault == "lost-conditional":
        mapping["conditional"]["current"].pop("plugin::one_minus_square")
    elif fault == "transported-acceptance":
        mapping["evidence_transport"] = True
    elif fault == "lost-variant":
        mapping["observed_contract_variants"] = []
    else:
        mapping["observed_contract_variants"][0]["observed_contract"]["callable"] = {}
        if fault == "forged-older-closure-and-digest":
            from scripts._source_inventory_epoch import digest

            variant = mapping["observed_contract_variants"][0]
            variant["observed_contract_sha256"] = digest(variant["observed_contract"])
    with pytest.raises(ValueError):
        verify_epoch_mapping(historical, current, mapping)


def test_current_keepdims_epoch_changes_only_two_callbacks_and_four_derived_recipes(
    epoch,
):
    from scripts.source_precision_inventory import legacy_preparation_obligations

    _, current, mapping = epoch
    assert (
        hashlib.sha256(current).hexdigest()
        == "bb0b210e5425fed30ba99e494f1fa1738e3660cdb368a74779bf285baff03b67"
    )
    child = json.loads(current)
    assert len(child["contracts"]) == 1721 and len(child["cases"]) == 1856
    parent = copy.deepcopy(child)
    parent["contracts"] = parent["contracts"][:1719]
    epochs = (
        (634, 614, 1719, "plugin::jax.numpy.all"),
        (645, 625, 1720, "plugin::jax.numpy.any"),
    )
    for case_index, old_index, new_index, origin in epochs:
        assert child["cases"][case_index][5] == new_index
        assert child["registrations"][child["cases"][case_index][0]][0] == origin
        parent["cases"][case_index][5] = old_index
        previous, revised = child["contracts"][old_index], child["contracts"][new_index]
        assert previous != revised
        assert {
            key: value
            for key, value in previous.items()
            if key != "post_check_onnx_graph"
        } == {
            key: value
            for key, value in revised.items()
            if key != "post_check_onnx_graph"
        }
        link = next(
            row
            for row in mapping["rows"]
            if row["current"]["origin"] == origin and row["current"]["ordinal"] == 1
        )
        assert link["changed_contract_fields"] == ["post_check_onnx_graph"]
        assert link["status"] == "AUDIT_PENDING"
    # Canonical object fingerprint of retained raw c0ccc9e5; its original compact bytes differ.
    parent_canonical_payload = (
        json.dumps(parent, indent=2, sort_keys=True) + "\n"
    ).encode()
    assert (
        hashlib.sha256(parent_canonical_payload).hexdigest()
        == "f51f2ec6390dba7373c1e0e68b5b8c4bce5473fcb70c4557f487178a66425a8f"
    )
    expected = copy.deepcopy(legacy_preparation_obligations(parent))
    changed = []
    for item in expected:
        identity = tuple(item["raw_identity"])
        if identity not in {(origin, 1) for _, _, _, origin in epochs}:
            continue
        _, old_index, new_index, _ = next(
            row for row in epochs if row[3] == identity[0]
        )
        recipe = item["recipe"]
        assert recipe["original_contract"] == child["contracts"][old_index]
        assert (
            item["recipe_sha256"]
            == hashlib.sha256(
                json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        )
        recipe["raw_contract_sha256"] = hashlib.sha256(
            json.dumps(
                child["contracts"][new_index], sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        recipe["original_contract"] = copy.deepcopy(recipe["original_contract"])
        recipe["original_contract"]["post_check_onnx_graph"] = copy.deepcopy(
            child["contracts"][new_index]["post_check_onnx_graph"]
        )
        item["recipe_sha256"] = hashlib.sha256(
            json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        assert item["status"] == "unverified" and item["original_evidence"] is None
        changed.append(item["obligation_id"])
    assert len(changed) == 4 and set(changed) == {
        "legacy-preparation:plugin::jax.numpy."
        + name
        + ":1:original-shapes:legacy-"
        + dtype
        + "-prepared"
        for name in ("all", "any")
        for dtype in ("f32", "f64")
    }
    assert legacy_preparation_obligations(child) == expected
    assert (
        mapping["evidence_transport"] is False
        and mapping["numerical_acceptance"] == "unverified"
    )
