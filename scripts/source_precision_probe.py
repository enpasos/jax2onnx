# scripts/source_precision_probe.py

"""Schedule bounded SOURCE migration probes without shrinking the raw denominator.

The default is planning only. Explicit prepare/product batches retain every other
cell as scheduled. Product probes check artifacts without runtime acceptance;
missing reviewed numerical criteria remain unresolved. Diagnostic export is not
enabled by this initial runner.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Literal

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.source_precision_inventory import (  # noqa: E402
    DEFAULT_BASELINE,
    canonical_contract,
    raw_registrations,
)
from jax2onnx.precision import CheckStatus, SampleCriteria  # noqa: E402
from tests.extra_tests.converter.source_precision_support import (  # noqa: E402
    SourceProvenance,
    prepare_source_case,
    raw_source_scenarios,
    run_source_case,
)


SCENARIOS = ("raw-x64-off", "raw-x64-on")


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _number(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, Mapping) and "scalar_dtype" in value:
        dtype = np.dtype(value["dtype_str"])
        if dtype.kind not in "fiu":
            raise ValueError("A tolerance must be a real scalar")
        array: np.ndarray[Any, Any] = np.frombuffer(
            bytes.fromhex(value["bytes_hex"]), dtype=dtype
        )
        if len(array) == 1:
            return float(array[0])
    raise ValueError("Unsupported explicit tolerance recipe")


def _frozen_criteria(contract: Mapping[str, Any]) -> tuple[SampleCriteria | None, str]:
    if "rtol" not in contract or "atol" not in contract:
        return (
            None,
            "A reviewed source criterion needs both explicit rtol and atol; legacy variant fields and defaults are not source criteria.",
        )
    try:
        return (
            SampleCriteria(
                "tolerant",
                rtol=_number(contract["rtol"]),
                atol=_number(contract["atol"]),
            ),
            "",
        )
    except (TypeError, ValueError, KeyError) as exc:
        return None, f"Unresolved explicit source criterion: {exc}"


@dataclass(frozen=True)
class ProbeRow:
    origin: str
    ordinal: int
    display: tuple[str, str, str]
    opset: int
    contract_sha256: str
    criteria: SampleCriteria | None
    criteria_pending: str
    skip_numeric_validation: bool
    skip_reason: str
    unresolved_recipes: tuple[str, ...]


@dataclass(frozen=True)
class ProbeCell:
    origin: str
    ordinal: int
    scenario: str
    stage: str = "scheduled"
    category: str = "audit-pending"
    source_reference: CheckStatus = CheckStatus.UNVERIFIED
    schema: CheckStatus = CheckStatus.UNVERIFIED
    lowering_audit: CheckStatus = CheckStatus.UNVERIFIED
    runtime: CheckStatus = CheckStatus.UNVERIFIED
    numerics: CheckStatus = CheckStatus.UNVERIFIED
    detail: str = ""
    artifact_sha256: str | None = None

    @property
    def identity(self) -> tuple[str, int, str]:
        return self.origin, self.ordinal, self.scenario


@dataclass(frozen=True)
class SourceProbePlan:
    baseline_sha256: str
    rows: tuple[ProbeRow, ...]
    cells: tuple[ProbeCell, ...]
    optional_json: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "baseline_sha256": self.baseline_sha256,
            "scope": "Bounded migration observations; no full-source or deployment acceptance.",
            "scenario_policy": {
                "names": SCENARIOS,
                "factory": "instantiate original recipe unchanged",
                "caller_x64_enabled": [False, True],
                "default_matmul_precision": None,
                "shape_only_default_dtype": "float32",
                "input_seed": "sha256(origin + NUL + ordinal), first 8 bytes big-endian; identical across scenarios",
                "criteria": "Freeze explicit raw rtol+atol; otherwise unresolved. Product stage does not execute runtime samples.",
            },
            "rows": [asdict(row) for row in self.rows],
            "cells": [asdict(cell) for cell in self.cells],
            "optional": json.loads(self.optional_json),
            "counts": {
                "raw_rows": len(self.rows),
                "cells": len(self.cells),
                "stages": dict(
                    sorted(Counter(cell.stage for cell in self.cells).items())
                ),
                "source_criteria_pending_rows": sum(
                    row.criteria is None for row in self.rows
                ),
                "opaque_recipe_rows": sum(
                    bool(row.unresolved_recipes) for row in self.rows
                ),
            },
        }


def build_probe_plan(
    inventory: Mapping[str, Any], *, baseline_sha256: str | None = None
) -> SourceProbePlan:
    """Freeze all provenance/scenario identities and criteria before any outcomes."""
    if inventory.get("schema_version") != 1:
        raise ValueError("Unsupported SOURCE inventory schema")
    unresolved = dict(inventory.get("unresolved_recipes", []))
    rows = []
    cells = []
    seen = set()
    for index, entry in enumerate(inventory["cases"]):
        registration, ordinal, testcase, _, opset, contract_index = entry
        origin, _, context, component = inventory["registrations"][registration]
        if (origin, ordinal) in seen:
            raise ValueError("Duplicate raw provenance identity")
        seen.add((origin, ordinal))
        contract = inventory["contracts"][contract_index]
        criteria, pending = _frozen_criteria(contract)
        skipped = bool(contract.get("skip_numeric_validation"))
        skip_reason = (
            str(
                contract.get(
                    "skip_numeric_validation_reason",
                    "Raw metadata skips numerical validation; no reason supplied.",
                )
            )
            if skipped
            else ""
        )
        row = ProbeRow(
            origin,
            ordinal,
            (context, component, testcase),
            opset,
            _fingerprint(contract),
            criteria,
            pending,
            skipped,
            skip_reason,
            tuple(unresolved.get(index, ())),
        )
        rows.append(row)
        for scenario in SCENARIOS:
            cells.append(
                ProbeCell(
                    origin,
                    ordinal,
                    scenario,
                    stage="recipe-pending" if row.unresolved_recipes else "scheduled",
                    detail="; ".join(row.unresolved_recipes),
                )
            )
    return SourceProbePlan(
        baseline_sha256 or _fingerprint(inventory),
        tuple(rows),
        tuple(cells),
        json.dumps(inventory.get("optional", {}), sort_keys=True),
    )


def _resolve_case(
    row: ProbeRow, registrations: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    registered = registrations.get(row.origin)
    if registered is None:
        raise ValueError(
            "Baseline registration is unavailable in this environment; its cells remain pending."
        )
    cases = registered.get("testcases", [])
    if row.ordinal >= len(cases):
        raise ValueError("Raw baseline ordinal is unavailable; review inventory drift.")
    raw = cases[row.ordinal]
    unresolved: list[str] = []
    contract = canonical_contract(
        {
            key: value
            for key, value in raw.items()
            if key not in {"testcase", "description"}
        },
        unresolved=unresolved,
    )
    display = (
        str(registered.get("context", "default")),
        str(registered.get("component", "default")),
        str(raw.get("testcase", "unnamed")),
    )
    if _fingerprint(contract) != row.contract_sha256 or display != row.display:
        raise ValueError(
            "Raw source contract changed after the plan was frozen; review and rebuild the plan."
        )
    return {**raw, "context": display[0], "component": display[1]}


def execute_probe_plan(
    plan: SourceProbePlan,
    registrations: Sequence[tuple[str, str, Mapping[str, Any]]],
    *,
    stage: Literal["prepare", "product"],
    limit: int,
    origins: frozenset[str] = frozenset(),
    scenario: str | None = None,
) -> SourceProbePlan:
    """Run a bounded batch; unselected cells and known failures stay explicit."""
    if limit <= 0:
        raise ValueError("A probe batch needs a positive explicit limit")
    if stage not in {"prepare", "product"}:
        raise ValueError("Unsupported probe stage")
    if scenario is not None and scenario not in SCENARIOS:
        raise ValueError("Unknown raw scenario")
    if origins - {row.origin for row in plan.rows}:
        raise ValueError("Origin selectors must name registrations in the frozen plan")
    registered = {origin: metadata for origin, _, metadata in registrations}
    if len(registered) != len(registrations):
        raise ValueError("Ambiguous raw registration provenance")
    rows = {(row.origin, row.ordinal): row for row in plan.rows}
    cells = []
    selected = 0
    for cell in plan.cells:
        if (
            selected >= limit
            or (origins and cell.origin not in origins)
            or (scenario is not None and cell.scenario != scenario)
            or cell.stage not in {"scheduled", "recipe-pending"}
        ):
            cells.append(cell)
            continue
        selected += 1
        row = rows[(cell.origin, cell.ordinal)]
        if row.unresolved_recipes:
            cells.append(cell)
            continue
        try:
            metadata = _resolve_case(row, registered)
            scenarios = raw_source_scenarios(
                metadata,
                SourceProvenance(row.origin, row.ordinal),
                criteria=row.criteria,
            )
            declared = scenarios[SCENARIOS.index(cell.scenario)]
        except (ValueError, TypeError, IndexError) as exc:
            cells.append(replace(cell, stage="preparation-pending", detail=str(exc)))
            continue
        if stage == "prepare":
            try:
                prepare_source_case(metadata, declared)
                outcome = replace(
                    cell,
                    stage="source-prepared",
                    source_reference=CheckStatus.VALIDATED,
                )
            except Exception as exc:
                outcome = replace(
                    cell,
                    stage="source-preparation",
                    category="migration-gap",
                    source_reference=CheckStatus.FAILED,
                    detail=str(exc),
                )
        else:
            # This checkpoint intentionally gathers no runtime/numerical evidence.
            result = run_source_case(metadata, declared, optimization=None)
            outcome = replace(
                cell,
                stage=result.stage,
                category=result.category,
                source_reference=(
                    CheckStatus.VALIDATED
                    if result.prepared is not None
                    else CheckStatus.FAILED
                ),
                schema=(
                    CheckStatus.VALIDATED
                    if result.model is not None
                    else CheckStatus.UNVERIFIED
                ),
                lowering_audit=(
                    result.audit.status
                    if result.prepared is not None
                    else CheckStatus.UNVERIFIED
                ),
                detail=result.audit.detail,
                artifact_sha256=(
                    result.validation.artifact_sha256
                    if result.validation is not None
                    else None
                ),
            )
        cells.append(outcome)
    return replace(plan, cells=tuple(cells))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--stage", choices=("plan", "prepare", "product"), default="plan"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--origin", action="append", default=[])
    parser.add_argument("--scenario", choices=SCENARIOS)
    args = parser.parse_args(argv)
    if args.stage != "plan" and (args.limit is None or args.limit <= 0):
        parser.error(
            "Execution needs an explicit positive --limit; default planning executes no models."
        )
    payload = args.baseline.read_bytes()
    plan = build_probe_plan(
        json.loads(payload), baseline_sha256=hashlib.sha256(payload).hexdigest()
    )
    if args.stage != "plan":
        registrations, _ = raw_registrations()
        plan = execute_probe_plan(
            plan,
            registrations,
            stage=args.stage,
            limit=args.limit,
            origins=frozenset(args.origin),
            scenario=args.scenario,
        )
    result = plan.to_dict()
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["counts"], sort_keys=True))
    print("No full-source or deployment acceptance is implied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
