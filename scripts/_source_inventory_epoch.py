# scripts/_source_inventory_epoch.py

"""Pure verification of documentary inventory links; no execution evidence moves."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def rows(inventory: Mapping[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    result = {}
    for registration, ordinal, name, features, opset, contract in inventory["cases"]:
        identity = (inventory["registrations"][registration][0], ordinal)
        if identity in result:
            raise ValueError("Duplicate inventory origin/ordinal")
        result[identity] = {
            "origin": identity[0],
            "ordinal": ordinal,
            "testcase": name,
            "feature_bits": features,
            "opset": opset,
            "contract_sha256": digest(inventory["contracts"][contract]),
        }
    return result


def manual(inventory: Mapping[str, Any]) -> list[list[str]]:
    return sorted([path, name] for path, names in inventory["manual"] for name in names)


def changed_fields(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[str]:
    return sorted(
        key
        for key in left.keys() | right.keys()
        if key not in left or key not in right or left[key] != right[key]
    )


def verify_epoch_mapping(
    historical_payload: bytes,
    current_payload: bytes,
    mapping: Mapping[str, Any],
) -> None:
    """Check complete identities and contracts, without changing drift/admission rules."""

    def require(condition: bool, reason: str) -> None:
        if not condition:
            raise ValueError(reason)

    require(
        mapping.get("schema") == "source-inventory-epoch-v1", "Epoch schema differs"
    )
    require(
        mapping.get("status") == "AUDIT_PENDING"
        and mapping.get("numerical_acceptance") == "unverified"
        and mapping.get("evidence_transport") is False,
        "Documentary mapping cannot transport acceptance",
    )
    for label, payload in (
        ("historical", historical_payload),
        ("current", current_payload),
    ):
        require(
            mapping[label + "_sha256"] == hashlib.sha256(payload).hexdigest(),
            "Inventory epoch bytes differ: " + label,
        )
    old, new = json.loads(historical_payload), json.loads(current_payload)
    old_rows, new_rows = rows(old), rows(new)
    old_contracts = {
        (old["registrations"][r[0]][0], r[1]): old["contracts"][r[5]]
        for r in old["cases"]
    }
    new_contracts = {
        (new["registrations"][r[0]][0], r[1]): new["contracts"][r[5]]
        for r in new["cases"]
    }
    seen_old, seen_new, relocated = set(), set(), []
    for link in mapping["rows"]:
        left, right = link["historical"], link["current"]
        a, b = (left["origin"], left["ordinal"]), (
            right["origin"],
            right["ordinal"],
        )
        require(
            all(
                type(record[key]) is int
                for record in (left, right)
                for key in ("ordinal", "feature_bits", "opset")
            ),
            "Row integers differ",
        )
        require(a not in seen_old and b not in seen_new, "Duplicate epoch row link")
        require(
            left == old_rows.get(a) and right == new_rows.get(b), "Row contract differs"
        )
        require(
            left["origin"] == right["origin"] and left["testcase"] == right["testcase"],
            "Historical row identity disappeared",
        )
        fields = changed_fields(old_contracts[a], new_contracts[b])
        require(link["changed_contract_fields"] == fields, "Changed fields differ")
        require(link["relocated"] is (a != b), "Ordinal relocation differs")
        if a != b:
            relocated.append([list(a), list(b)])
        require(link["status"] == "AUDIT_PENDING", "Row claims acceptance")
        seen_old.add(a)
        seen_new.add(b)
    require(seen_old == old_rows.keys(), "Historical row coverage incomplete")
    require(
        sorted(relocated) == mapping["reviewed_relocations"],
        "Unreviewed ordinal relocation",
    )
    additions = []
    for row in mapping["added_rows"]:
        key = (row["origin"], row["ordinal"])
        require(type(row["ordinal"]) is int, "Added ordinal differs")
        require(key not in seen_new and key not in additions, "Duplicate added row")
        require(row == new_rows.get(key), "Added row contract differs")
        additions.append(key)
    require(
        seen_new | set(additions) == new_rows.keys(), "Current row coverage incomplete"
    )
    require(
        mapping["registrations"]["historical"] == old["registrations"]
        and mapping["registrations"]["current"] == new["registrations"],
        "Registry identities differ",
    )
    require(
        set(r[0] for r in old["registrations"])
        <= set(r[0] for r in new["registrations"]),
        "Historical registration disappeared",
    )
    historical_origins = {row[0] for row in old["registrations"]}
    require(
        mapping["registrations"]["added_obligations"]
        == [
            {
                "registration": row,
                "raw_case_rows": sum(key[0] == row[0] for key in new_rows),
                "status": "AUDIT_PENDING",
            }
            for row in new["registrations"]
            if row[0] not in historical_origins
        ],
        "Added registry obligations differ",
    )
    for label, inventory in (("historical", old), ("current", new)):
        require(
            mapping["metadata_free"][label] == inventory["registry_without_metadata"],
            "Metadata-free obligations differ",
        )
        require(
            mapping["manual"][label] == manual(inventory), "Manual identities differ"
        )
        require(
            mapping["opaque"][label] == inventory["unresolved_recipes"],
            "Opaque obligations differ",
        )
        require(
            mapping["optional"][label] == inventory["optional"],
            "Optional provenance differs",
        )
        require(
            mapping["conditional"][label] == inventory["conditional_registrations"],
            "Conditional obligations differ",
        )
    require(
        set(map(tuple, manual(old))) <= set(map(tuple, manual(new))),
        "Historical manual identity disappeared",
    )
    require(
        mapping["manual"]["added_identities"]
        == sorted(
            list(identity)
            for identity in set(map(tuple, manual(new))) - set(map(tuple, manual(old)))
        ),
        "Added manual identities differ",
    )
    for family, old_optional in old["optional"].items():
        require(
            old_optional["baseline"]["members"]
            == new["optional"][family]["baseline"]["members"],
            "Optional membership changed",
        )
    variants = mapping.get("observed_contract_variants", ())
    require(len(variants) == 1, "Observed older variant missing or duplicated")
    for variant in variants:
        key = (variant["current"]["origin"], variant["current"]["ordinal"])
        require(key == ("example::examples.lax::remat2", 0), "Variant identity differs")
        require(variant["current"] == new_rows.get(key), "Variant row differs")
        require(
            variant["observed_contract"] == old_contracts[key]
            and variant["observed_contract_sha256"] == digest(old_contracts[key]),
            "Observed variant contract differs",
        )
        require(
            variant["status"] == "canonical-mismatch"
            and variant["source_audit"] == "AUDIT_PENDING"
            and variant["numerical_acceptance"] == "unverified",
            "Variant claims acceptance",
        )
