# tests/extra_tests/framework/test_source_inventory_epoch.py

"""Current membership and historical evidence retain distinct, complete epochs."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from scripts._source_inventory_epoch import verify_epoch_mapping
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
