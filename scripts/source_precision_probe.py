# scripts/source_precision_probe.py

"""Schedule bounded SOURCE migration probes without shrinking the raw denominator.

The default is planning only. Explicit prepare/product batches retain every other
cell as scheduled. Product probes check artifacts without runtime acceptance;
missing reviewed numerical criteria remain unresolved. Diagnostic export uses
the explicit tooling validator and shared facade; its lowering audit stays open.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any, Final, Literal

import numpy as np


REPOSITORY_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.source_precision_inventory import (  # noqa: E402
    DEFAULT_BASELINE,
    canonical_contract,
    raw_registrations,
)
from jax2onnx.precision import (  # noqa: E402
    CheckStatus,
    SampleCriteria,
    PrecisionReport,
    get_precision_report,
)
from tests.extra_tests.converter.source_precision_support import (  # noqa: E402
    SourceProvenance,
    SourceSettings,
    SourceValueEvidence,
    PRESERVE_FACTORY_DTYPE,
    _properties,
    value_evidence,
    prepare_source_case,
    raw_source_scenarios,
    run_source_case,
)
from tests.extra_tests.converter.source_precision_diagnostics import (  # noqa: E402
    DiagnosticValidator,
    StructuralScope,
)


SCENARIOS: Final[tuple[str, str]] = ("raw-x64-off", "raw-x64-on")
FIRST_DIAGNOSTIC_CASES: Final[tuple[SourceProvenance, ...]] = tuple(
    SourceProvenance(f"plugin::{name}", 0)
    for name in (
        "add",
        "convert_element_type",
        "sin",
        "reshape",
        "transpose",
        "gather",
        "reduce_sum",
        "dot_general",
        "conv_general_dilated",
        "jit",
        "pjit",
        "lax.fori_loop",
        "while",
        "scan",
        "complex",
        "real",
        "imag",
        "fft",
    )
)


@dataclass(frozen=True)
class ProbeEnvironment:
    python: str
    implementation: str
    architecture: str
    system: str
    versions: tuple[tuple[str, str], ...]

    @classmethod
    def observe(cls) -> ProbeEnvironment:
        names = (
            "jax2onnx",
            "jax",
            "jaxlib",
            "numpy",
            "flax",
            "onnx",
            "onnx-ir",
            "onnxruntime",
        )
        return cls(
            platform.python_version(),
            platform.python_implementation(),
            platform.machine(),
            platform.system(),
            tuple((name, importlib.metadata.version(name)) for name in names),
        )


@dataclass(frozen=True)
class ProbeStageEvidence:
    cell_identity: tuple[str, int, str]
    requested_stage: str
    row_contract_sha256: str
    scenario_sha256: str
    environment: ProbeEnvironment
    requested_x64: bool
    requested_matmul_precision: str | None
    actual_settings: SourceSettings | None
    export_settings: SourceSettings | None
    original_trace_json: str | None
    original_trace_sha256: str | None
    supplied_inputs: tuple[SourceValueEvidence, ...]
    inputs: tuple[SourceValueEvidence, ...]
    references: tuple[SourceValueEvidence, ...]
    source_reference: CheckStatus
    structure: CheckStatus
    schema: CheckStatus
    artifact: CheckStatus
    lowering_audit: CheckStatus
    runtime: CheckStatus
    numerics: CheckStatus
    stage: str
    category: str
    detail: str
    artifact_sha256: str | None
    export_report: PrecisionReport | None
    diagnostic_scopes: tuple[StructuralScope, ...]
    previous_sha256: str | None

    @property
    def sha256(self) -> str:
        return _fingerprint(asdict(self))


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _trace_pending(value: Any) -> bool:
    if isinstance(value, Mapping):
        return (
            bool(value.get("unresolved_parameters"))
            or "unresolved_dimension" in value
            or "unresolved" in value
            or any(_trace_pending(item) for item in value.values())
        )
    if isinstance(value, (list, tuple)):
        return any(_trace_pending(item) for item in value)
    return False


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
    artifact: CheckStatus = CheckStatus.UNVERIFIED
    lowering_audit: CheckStatus = CheckStatus.UNVERIFIED
    runtime: CheckStatus = CheckStatus.UNVERIFIED
    numerics: CheckStatus = CheckStatus.UNVERIFIED
    detail: str = ""
    artifact_sha256: str | None = None
    history: tuple[ProbeStageEvidence, ...] = ()

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
            "schema_version": 2,
            "baseline_sha256": self.baseline_sha256,
            "scope": "Bounded migration observations; no full-source or deployment acceptance.",
            "scenario_policy": {
                "names": SCENARIOS,
                "factory": "instantiate original recipe unchanged",
                "caller_x64_enabled": [False, True],
                "default_matmul_precision": None,
                "shape_only_default_dtype": "float32",
                "input_seed": "sha256(origin + NUL + ordinal), first 8 bytes big-endian; identical across scenarios",
                "criteria": "Freeze explicit raw rtol+atol; otherwise unresolved. Product and diagnostic artifact stages do not execute runtime samples.",
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


def preflight_cases(
    plan: SourceProbePlan,
    registrations: Sequence[tuple[str, str, Mapping[str, Any]]],
    cases: frozenset[SourceProvenance],
) -> None:
    """Verify every exact requested identity/contract before any model execution."""
    registered = {origin: metadata for origin, _, metadata in registrations}
    if len(registered) != len(registrations):
        raise ValueError("Ambiguous raw registration provenance")
    rows = {(row.origin, row.ordinal): row for row in plan.rows}
    for case in sorted(cases, key=lambda item: (item.origin, item.ordinal)):
        row = rows.get((case.origin, case.ordinal))
        if row is None:
            raise ValueError(f"Requested case is absent from the baseline: {case}")
        _resolve_case(row, registered)


def execute_probe_plan(
    plan: SourceProbePlan,
    registrations: Sequence[tuple[str, str, Mapping[str, Any]]],
    *,
    stage: Literal["prepare", "product", "diagnostic"],
    limit: int,
    origins: frozenset[str] = frozenset(),
    scenario: str | None = None,
    cases: frozenset[SourceProvenance] = frozenset(),
) -> SourceProbePlan:
    """Run a bounded batch; unselected cells and known failures stay explicit."""
    if limit <= 0:
        raise ValueError("A probe batch needs a positive explicit limit")
    if stage not in {"prepare", "product", "diagnostic"}:
        raise ValueError("Unsupported probe stage")
    if scenario is not None and scenario not in SCENARIOS:
        raise ValueError("Unknown raw scenario")
    if origins - {row.origin for row in plan.rows}:
        raise ValueError("Origin selectors must name registrations in the frozen plan")
    registered = {origin: metadata for origin, _, metadata in registrations}
    if len(registered) != len(registrations):
        raise ValueError("Ambiguous raw registration provenance")
    if cases:
        preflight_cases(plan, registrations, cases)
    rows = {(row.origin, row.ordinal): row for row in plan.rows}
    cells = []
    selected = 0
    for cell in plan.cells:
        if (
            selected >= limit
            or (origins and cell.origin not in origins)
            or (scenario is not None and cell.scenario != scenario)
            or (cases and SourceProvenance(cell.origin, cell.ordinal) not in cases)
            or cell.stage
            not in (
                {"scheduled", "recipe-pending"}
                if stage == "prepare"
                else {"scheduled", "recipe-pending", "source-prepared"}
            )
            or any(
                CheckStatus.FAILED
                in (
                    record.source_reference,
                    record.structure,
                    record.schema,
                    record.artifact,
                    record.lowering_audit,
                )
                for record in cell.history
            )
        ):
            cells.append(cell)
            continue
        selected += 1
        row = rows[(cell.origin, cell.ordinal)]
        if row.unresolved_recipes:
            cells.append(cell)
            continue
        environment = ProbeEnvironment.observe()
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
        supplied = tuple(
            value_evidence(
                value,
                _properties(
                    value
                    if hasattr(value, "dtype") and hasattr(value, "shape")
                    else np.asarray(value)
                ),
            )
            for value in declared.input_values or ()
        )
        scenario_sha = _fingerprint(
            {
                "name": declared.name,
                "x64": declared.caller_x64_enabled,
                "matmul": declared.default_matmul_precision,
                "factory": (
                    "preserve"
                    if declared.model_dtype is PRESERVE_FACTORY_DTYPE
                    else np.dtype(declared.model_dtype).name
                ),
                "specs": [
                    asdict(_properties(spec)) for spec in declared.input_specs or ()
                ],
                "supplied": [asdict(value) for value in supplied],
                "generation_provenance": [row.origin, row.ordinal],
                "criteria": asdict(row.criteria) if row.criteria is not None else None,
            }
        )
        prepared = None
        export_settings = None
        export_report = None
        observed_settings: list[SourceSettings] = []
        validator = DiagnosticValidator() if stage == "diagnostic" else None
        try:
            prepared = prepare_source_case(
                metadata, declared, settings_observer=observed_settings.append
            )
            outcome = replace(
                cell, stage="source-prepared", source_reference=CheckStatus.VALIDATED
            )
        except Exception as exc:
            outcome = replace(
                cell,
                stage="source-preparation",
                category=getattr(exc, "category", "migration-gap"),
                source_reference=CheckStatus.FAILED,
                detail=str(exc),
            )
        drift = bool(
            prepared is not None
            and cell.history
            and (
                cell.history[-1].environment != environment
                or cell.history[-1].scenario_sha256 != scenario_sha
                or cell.history[-1].inputs != prepared.input_evidence
                or cell.history[-1].references != prepared.reference_evidence
                or cell.history[-1].original_trace_json != prepared.original_trace_json
                or cell.history[-1].actual_settings != prepared.settings
            )
        )
        if drift:
            outcome = replace(
                outcome,
                stage="source-drift",
                category="migration-gap",
                source_reference=CheckStatus.FAILED,
                detail="Original typed trace/input/reference/environment changed after preparation; prior evidence is retained.",
            )
        elif (
            prepared is not None
            and stage != "prepare"
            and _trace_pending(json.loads(prepared.original_trace_json))
        ):
            outcome = replace(
                outcome,
                stage="trace-lineage-pending",
                category="audit-pending",
                detail="Original trace contains unresolved parameter lineage; no diagnostic export was attempted.",
            )
        elif prepared is not None and stage != "prepare":
            # This checkpoint intentionally gathers no runtime/numerical evidence.
            result = run_source_case(
                metadata,
                declared,
                optimization=None,
                diagnostic_jaxpr_validator=validator,
                _prepared=prepared,
            )
            export_settings = result.export_settings
            export_report = (
                get_precision_report(result.model) if result.model is not None else None
            )
            outcome = replace(
                cell,
                stage=result.stage,
                category=result.category,
                source_reference=(
                    CheckStatus.VALIDATED
                    if result.prepared is not None
                    else CheckStatus.FAILED
                ),
                schema=result.schema,
                artifact=(
                    CheckStatus.VALIDATED
                    if result.model is not None
                    else (
                        CheckStatus.FAILED
                        if result.audit.status is CheckStatus.FAILED
                        else CheckStatus.UNVERIFIED
                    )
                ),
                lowering_audit=(
                    CheckStatus.UNVERIFIED
                    if stage == "diagnostic"
                    else (
                        result.audit.status
                        if result.prepared is not None
                        else CheckStatus.UNVERIFIED
                    )
                ),
                detail=result.audit.detail,
                artifact_sha256=(
                    result.validation.artifact_sha256
                    if result.validation is not None
                    else None
                ),
            )
        structure = CheckStatus.UNVERIFIED
        if validator is not None:
            if validator.failure is not None:
                structure = (
                    CheckStatus.FAILED
                    if validator.failure.category == "lowering-defect"
                    else CheckStatus.UNVERIFIED
                )
            elif validator.scopes:
                structure = CheckStatus.VALIDATED
        evidence = ProbeStageEvidence(
            cell.identity,
            stage,
            row.contract_sha256,
            scenario_sha,
            environment,
            declared.caller_x64_enabled,
            declared.default_matmul_precision,
            (
                prepared.settings
                if prepared is not None
                else (observed_settings[-1] if observed_settings else None)
            ),
            export_settings,
            prepared.original_trace_json if prepared is not None else None,
            (
                hashlib.sha256(prepared.original_trace_json.encode()).hexdigest()
                if prepared is not None
                else None
            ),
            supplied,
            prepared.input_evidence if prepared is not None else (),
            prepared.reference_evidence if prepared is not None else (),
            outcome.source_reference,
            structure,
            outcome.schema,
            outcome.artifact,
            outcome.lowering_audit,
            outcome.runtime,
            outcome.numerics,
            outcome.stage,
            outcome.category,
            outcome.detail,
            outcome.artifact_sha256,
            export_report,
            tuple(validator.scopes) if validator is not None else (),
            cell.history[-1].sha256 if cell.history else None,
        )
        cells.append(replace(outcome, history=(*cell.history, evidence)))
    return replace(plan, cells=tuple(cells))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--stage", choices=("plan", "prepare", "product", "diagnostic"), default="plan"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--origin", action="append", default=[])
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Exact raw origin::ordinal, e.g. plugin::add::0",
    )
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
            cases=frozenset(
                SourceProvenance(origin, int(ordinal))
                for origin, ordinal in (value.rsplit("::", 1) for value in args.case)
            ),
        )
    result = plan.to_dict()
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["counts"], sort_keys=True))
    print("No full-source or deployment acceptance is implied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
