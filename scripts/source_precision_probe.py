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
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import subprocess
from unittest.mock import patch
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
from tests.extra_tests.converter.source_precision_corpus_oracles import (  # noqa: E402
    CORPUS_REPLAY_CASES,
    DOT_COUNTERPARTS,
    GATHER_COUNTERPARTS,
    HISTORICAL_BASELINE_SHA256,
    CorpusStructureEvidence,
    NormalizationCounterpart,
    counterpart_for,
    selector_manifest,
    criterion_proposal_ledger,
)

SCENARIOS: Final[tuple[str, str]] = ("raw-x64-off", "raw-x64-on")
CORPUS_CONTROL_REVISION: Final = "source-static-add-ordinary-dot-v8"
# Prior control acceptance is historical evidence, never a current admission.
HISTORICAL_CORPUS_CONTROL_REVISION_V7: Final = "source-static-add-ordinary-dot-v7"
HISTORICAL_CORPUS_DOT_DESCRIPTOR_V7: Final = {
    "counterpart_identifier": "static-dot-source-counterpart-v3:plugin::dot_general:0",
    "counterpart_sha256": "0ca2b0c8aebbf923bb083fa6b658e4f18c5d4c04bbc2265645c4056e36c31307",
}
HISTORICAL_CORPUS_CONTROL_REVISION_V6: Final = "source-static-add-ordinary-dot-v6"
HISTORICAL_CORPUS_DOT_DESCRIPTOR_V6: Final = {
    "counterpart_identifier": "static-dot-source-counterpart-v2:plugin::dot_general:0",
    "counterpart_sha256": "de011545a97be9d6aecc1bdf16df0a452665a84f485058ea2b5ce23cd6b89d71",
}
_CONTROL_CONTRACTS: Final = {
    "plugin::jax.numpy.add": "39f16c5258b9354829fd94c8825deb0e8a389215d0be1fcced026de8ae82a922",
    "plugin::dot_general": "6957d2d4640c79d15f46c7e930fcbd67467e3b278929165cf8619e6f88c7f8d4",
}
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
    corpus_structure: CorpusStructureEvidence | None = None
    raw_callback_status: str = "not-executed"
    replay_capture: Mapping[str, Any] | None = None

    @property
    def sha256(self) -> str:
        payload = asdict(self)
        if self.replay_capture is None:
            # Historical evidence hashes predate the general replay payload.
            payload.pop("replay_capture")
        return _fingerprint(payload)


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


class SourceContractDrift(ValueError):
    """A complete frozen source/recipe/artifact contract was contradicted."""

    category = "source-contract-drift"


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
        raise SourceContractDrift(
            "Raw baseline ordinal is unavailable; review inventory drift."
        )
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
        raise SourceContractDrift(
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
    _corpus_replay: bool = False,
    raw_replay: bool = False,
    replay_adapters: Mapping[str, Any] | None = None,
    evidence_directory: Path | None = None,
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
    if raw_replay:
        if _corpus_replay or stage == "diagnostic":
            raise ValueError("General raw replay is an ordinary product/prepare lane")
        return _execute_raw_probe_plan(
            plan,
            registrations,
            stage=stage,
            limit=limit,
            origins=origins,
            scenario=scenario,
            cases=cases,
            adapters=replay_adapters or {},
            evidence_directory=evidence_directory,
        )
    registered = {origin: metadata for origin, _, metadata in registrations}
    if len(registered) != len(registrations):
        raise ValueError("Ambiguous raw registration provenance")
    if cases:
        preflight_cases(plan, registrations, cases)
    if _corpus_replay:
        if (
            cases != corpus_replay_cases(plan)
            or limit != 26
            or scenario is not None
            or origins
            or stage not in {"prepare", "diagnostic"}
        ):
            raise ValueError(
                "Corpus replay requires exactly the fixed 26 diagnostic cells"
            )
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
        finite_counterpart = counterpart_for(
            row.origin, row.ordinal, row.contract_sha256
        )
        normalization = (
            finite_counterpart
            if type(finite_counterpart) is NormalizationCounterpart
            else None
        )
        if row.unresolved_recipes and normalization is None:
            cells.append(cell)
            continue
        environment = ProbeEnvironment.observe()
        try:
            metadata = _resolve_case(row, registered)
            if normalization is not None:
                # This exact finite counterpart binds the original registered
                # Eqx object/parameters without rewriting its opaque raw recipe.
                # Other opaque rows retain their original pending status.
                normalization.check_metadata(metadata)
            scenarios = raw_source_scenarios(
                metadata,
                SourceProvenance(row.origin, row.ordinal),
                criteria=row.criteria,
            )
            declared = scenarios[SCENARIOS.index(cell.scenario)]
            if normalization is not None and any(
                "B" in shape for shape in metadata.get("input_shapes", ())
            ):
                declared = replace(
                    declared,
                    concrete_shapes=tuple(
                        tuple(2 if dim == "B" else dim for dim in shape)
                        for shape in metadata["input_shapes"]
                    ),
                )
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
                **(
                    {"concrete_shapes": declared.concrete_shapes}
                    if normalization is not None
                    else {}
                ),
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
        corpus_structure = None
        raw_callback_status = "not-executed"
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
                _corpus_counterpart=(
                    counterpart_for(row.origin, row.ordinal, row.contract_sha256)
                    if _corpus_replay
                    else normalization
                ),
            )
            corpus_structure = result.corpus_structure
            raw_callback_status = result.raw_callback_status
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
            corpus_structure,
            raw_callback_status,
        )
        cells.append(replace(outcome, history=(*cell.history, evidence)))
    return replace(plan, cells=tuple(cells))


def _raw_metadata_checks(prepared: Any, result: Any, capture: Any) -> dict[str, Any]:
    """Observe original assertion requests without changing product admission.

    Known native/metadata contradictions fail. An unsupported assertion recipe
    remains pending, and observer failures cannot abort the ordinary raw batch or
    demote an earlier known contradiction. Original expectations are not inferred
    from the produced artifact.
    """
    metadata = prepared.metadata if prepared is not None else {}
    checks: dict[str, Any] = {}
    capture.records["metadata_assertions"] = checks

    def add(name: str, status: CheckStatus, detail: str) -> None:
        checks[name] = {"status": status.value, "detail": detail}

    for name in ("expected_output_shapes", "expected_output_dtypes"):
        if name not in metadata or metadata[name] is None:
            continue
        try:
            expected = metadata[name]
            if not isinstance(expected, (list, tuple)):
                add(
                    name,
                    CheckStatus.UNVERIFIED,
                    "Original expected-output container needs a reviewed adapter",
                )
                continue
            actual = prepared.traced_outputs
            if len(expected) != len(actual):
                add(
                    name,
                    CheckStatus.FAILED,
                    "Original expected output arity differs from native trace",
                )
                continue
            if name == "expected_output_dtypes":
                if any(dtype is None for dtype in expected):
                    add(
                        name,
                        CheckStatus.UNVERIFIED,
                        "Incomplete original expected dtype does not select a default",
                    )
                    continue
                matched = all(
                    np.dtype(dtype).name == properties.dtype
                    for dtype, properties in zip(expected, actual, strict=True)
                )
                add(
                    name,
                    CheckStatus.VALIDATED if matched else CheckStatus.FAILED,
                    "Original output dtypes compared with native trace before tolerance",
                )
            else:
                if any(not isinstance(shape, (list, tuple)) for shape in expected):
                    add(
                        name,
                        CheckStatus.UNVERIFIED,
                        "Original expected shape recipe needs a reviewed container adapter",
                    )
                    continue
                if any(
                    not isinstance(dim, (int, np.integer))
                    for shape in expected
                    for dim in shape
                ) or any(
                    not isinstance(dim, (int, np.integer))
                    for properties in actual
                    for dim in properties.shape
                ):
                    add(
                        name,
                        CheckStatus.UNVERIFIED,
                        "Original symbolic expected shape needs exact scope reconciliation",
                    )
                    continue
                matched = all(
                    tuple(shape) == tuple(properties.shape)
                    for shape, properties in zip(expected, actual, strict=True)
                )
                add(
                    name,
                    CheckStatus.VALIDATED if matched else CheckStatus.FAILED,
                    "Original static output shapes compared with native trace before tolerance",
                )
        except Exception as exc:
            add(
                name,
                CheckStatus.UNVERIFIED,
                f"Original expected-output observer needs a reviewed adapter: {type(exc).__name__}: {exc}",
            )

    pending = {
        "expected_number_of_function_instances": "Original function-instance count needs shared function ownership evidence",
        "check_onnx_load": "Original reload request has not been executed",
        "check_onnx_shape_inference": "Original shape-inference request has not been executed",
        "check_deployment_readiness_report": "Original deployment report request has not been executed",
        "post_check_onnx_runtime": "Original runtime callback is retained and needs captured target outputs",
        "x64_expected_output_shapes": "Original x64-specific expectation needs explicit counterpart review",
    }
    for name, detail in pending.items():
        if (
            name in metadata
            and metadata[name] is not None
            and metadata[name] is not False
        ):
            add(name, CheckStatus.UNVERIFIED, detail)
    for name in metadata:
        if (
            isinstance(name, str)
            and name.startswith(("expected_", "check_", "post_check_"))
            and name not in checks
            and name != "post_check_onnx_graph"
        ):
            if metadata[name] is not None and metadata[name] is not False:
                add(
                    name,
                    CheckStatus.UNVERIFIED,
                    "Original assertion has no SOURCE handler",
                )
    runtime = capture.records.get("metadata_runtime", {})
    if "runtime_input_values" in metadata:
        add(
            "runtime_input_values",
            CheckStatus(runtime.get("alternate_inputs_status", "unverified")),
            "Original alternate samples must retain bytes, dtype and dynamic controls",
        )
    add(
        "runtime_binding",
        CheckStatus(runtime.get("binding_status", "unverified")),
        runtime.get(
            "binding_detail", "Runtime source/artifact binding has not been observed"
        ),
    )
    add(
        "runtime_reference",
        CheckStatus(runtime.get("reference_status", "unverified")),
        runtime.get(
            "reference_detail",
            "Original runtime reference obligations have not been observed",
        ),
    )
    if metadata.get("export_mode", "standard") == "web":
        add(
            "onnxruntime_web",
            CheckStatus.UNVERIFIED,
            "CPU evidence does not qualify the requested web target",
        )
    for name in ("record_primitive_calls_file", "output_path"):
        if metadata.get(name) is not None:
            add(
                name,
                CheckStatus.UNVERIFIED,
                "Original destination retained; requested facade contract needs qualification",
            )
    if prepared is not None and prepared.scenario.require_reviewed_criteria:
        if prepared.criteria is None or metadata.get("skip_numeric_validation"):
            add(
                "reviewed_criteria",
                CheckStatus.UNVERIFIED,
                "Missing/partial/skipped original criterion does not authorize numerics",
            )
        else:
            add(
                "reviewed_criteria",
                CheckStatus.VALIDATED,
                "Complete explicit raw source criterion retained",
            )
    return checks


def _raw_runtime_status(result: Any) -> CheckStatus:
    """Only an actual validator observation can establish execution status."""
    if result is None or result.validation is None:
        return CheckStatus.UNVERIFIED
    checks = result.validation.checks + tuple(
        check for sample in result.validation.samples for check in sample.checks
    )
    runtime = [
        check.status for check in checks if check.category == "runtime-execution"
    ]
    if CheckStatus.FAILED in runtime:
        return CheckStatus.FAILED
    return (
        CheckStatus.VALIDATED
        if runtime and all(status is CheckStatus.VALIDATED for status in runtime)
        else CheckStatus.UNVERIFIED
    )


def _raw_metadata_runtime(
    prepared: Any,
    result: Any,
    capture: Any,
    checks: Mapping[str, Any],
    *,
    adapter: Mapping[str, Any],
    optimization: Literal["disabled", "all"],
) -> Any:
    """Reuse retained source/reference/artifact and the existing sample validator.

    Reference and binding failures are independent from actual target execution.
    Every original-source invocation is fenced before a subsequent invocation or
    target execution. A later binding failure retains actual validator evidence.
    """
    import jax
    from jax2onnx.precision import _artifact_bytes, validate_source_samples
    from tests.extra_tests.converter.source_precision_support import (
        _source_sample_from_reference,
        prepared_source_reference,
        prepare_source_reference,
        source_configuration,
    )

    observation = {
        "status": "unverified",
        "alternate_inputs_status": "unverified",
        "binding_status": "unverified",
        "reference_status": "unverified",
        "reference_role": "original-prepared-eager",
        "optimization": optimization,
        "criteria": (
            asdict(prepared.criteria)
            if prepared is not None and prepared.criteria
            else None
        ),
    }
    capture.records["metadata_runtime"] = observation
    blockers = []
    if prepared is None or result is None or result.model is None:
        blockers.append("No retained prepared source and returned artifact")
    elif prepared.criteria is None or prepared.metadata.get("skip_numeric_validation"):
        blockers.append("No complete applicable reviewed raw criterion")
    else:
        if (
            result.audit.status is not CheckStatus.VALIDATED
            or result.schema is not CheckStatus.VALIDATED
        ):
            blockers.append("Raw product audit/schema is not validated")
        if (
            capture.readiness is not CheckStatus.VALIDATED
            or capture.source_structure_status() is not CheckStatus.VALIDATED
        ):
            blockers.append("Exact source/artifact/physical mapping is not validated")
        if result.raw_callback_status not in {"absent", "passed"}:
            blockers.append("Original graph callback has not passed")
        if any(
            prepared.metadata.get(name)
            for name in ("inputs_as_nchw", "outputs_as_nchw")
        ):
            blockers.append(
                "Original layout needs an explicit native/physical runtime adapter"
            )
        if adapter.get("samples", [])[1:]:
            blockers.append(
                "Additional symbolic binding target samples are not yet reconciled"
            )
        if any(value["status"] == "failed" for value in checks.values()):
            blockers.append(
                "Original expected metadata contradicts source before tolerance"
            )
        if any(
            name in checks and checks[name]["status"] != "validated"
            for name in (
                "expected_output_shapes",
                "expected_output_dtypes",
                "x64_expected_output_shapes",
            )
        ):
            blockers.append("Original expected output contract needs review")
    if blockers:
        observation["detail"] = "; ".join(blockers)
        return result
    phase = "binding"
    observed_result = result
    try:
        capture.require_stable_target("before-metadata-runtime")
        audited_sha = capture.records["artifact"]["sha256"]
        if _artifact_bytes(result.model)[1] != audited_sha:
            raise SourceContractDrift(
                "Returned artifact differs before metadata runtime"
            )
        phase = "reference"
        references = [prepared_source_reference(prepared)]
        alternate = prepared.metadata.get("runtime_input_values")
        if alternate is not None:
            if not isinstance(alternate, (list, tuple)) or not alternate:
                raise ValueError(
                    "Original runtime input samples need a nonempty reviewed container"
                )
            for index, values in enumerate(alternate):
                label = f"metadata-runtime-reference-{index}"
                with capture.execution_context(label):
                    reference = prepare_source_reference(
                        prepared,
                        label=f"original-runtime-input-values-{index}",
                        input_values=values,
                    )
                capture.require_stable_target(label + "-after-eager")
                references.append(reference)
                capture.arrays(label + "-inputs", reference.arguments)
                capture.arrays(label + "-eager", reference.reference_outputs)
                positional = reference.arguments[: len(prepared.input_specs)]
                named = dict(
                    zip(
                        prepared.named_specs,
                        reference.arguments[len(prepared.input_specs) :],
                        strict=True,
                    )
                )
                with source_configuration(prepared.scenario):
                    if SourceSettings.observe() != prepared.settings:
                        raise SourceContractDrift(
                            "Alternate fresh-JIT settings differ from preparation"
                        )
                    fresh = capture.invoke(
                        label + "-fresh-jit",
                        lambda args=positional, kwargs=named: jax.device_get(
                            jax.jit(prepared.fn)(*args, **kwargs)
                        ),
                    )
                    capture.require_stable_target(label + "-after-fresh-jit")
                    if label + "-fresh-jit" not in capture.native_values:
                        raise ValueError(
                            "Original alternate fresh-JIT reference did not return"
                        )
                    outputs, tree = jax.tree_util.tree_flatten(fresh)
                    if str(tree) != prepared.output_tree or len(outputs) != len(
                        reference.reference_outputs
                    ):
                        raise SourceContractDrift(
                            "Alternate fresh-JIT native output tree/arity differs"
                        )
                    if any(
                        np.asarray(value).dtype != expected.dtype
                        or np.asarray(value).shape != expected.shape
                        for value, expected in zip(
                            outputs, reference.reference_outputs, strict=True
                        )
                    ):
                        raise SourceContractDrift(
                            "Alternate fresh-JIT native dtype/shape differs before tolerance"
                        )
                    capture.arrays(label + "-fresh-jit", outputs)
                    if SourceSettings.observe() != prepared.settings:
                        raise SourceContractDrift(
                            "Alternate fresh-JIT changed caller settings"
                        )
        observation["reference_status"] = "validated"
        observation["reference_detail"] = (
            "Original eager and fresh-JIT source evidence retained before target execution"
        )
        # Materialization may reject an unsupported mapping. It is a source
        # reference/preflight outcome, not proof that ORT ran and failed.
        samples = [
            _source_sample_from_reference(prepared, result.model, reference)
            for reference in references
        ]
        phase = "binding"
        capture.require_stable_target("immediately-before-metadata-validator")
        capture.finalize_source_structure(result.model)
        if (
            capture.readiness is not CheckStatus.VALIDATED
            or capture.source_structure_status() is not CheckStatus.VALIDATED
        ):
            raise ValueError(
                "Source/artifact readiness changed before the sample validator"
            )
        if _artifact_bytes(result.model)[1] != audited_sha:
            raise SourceContractDrift(
                "Audited artifact differs immediately before target execution"
            )
        observation["binding_status"] = "validated"
        phase = "validator"
        validation = validate_source_samples(
            result.model, samples=samples, optimization=optimization
        )
        # Retain the actual validator result before any later source/artifact
        # fence. Later failure must not erase successful or failed execution.
        observed_result = replace(result, validation=validation)
        observation.update(
            {
                "status": validation.status.value,
                "artifact_sha256": validation.artifact_sha256,
                "target": (
                    asdict(validation.target) if validation.target is not None else None
                ),
                "checks": [asdict(item) for item in validation.checks],
                "samples": [asdict(item) for item in validation.samples],
                "alternate_inputs_status": (
                    "validated"
                    if alternate is not None
                    and len(validation.samples) == len(references)
                    and _raw_runtime_status(observed_result) is CheckStatus.VALIDATED
                    else "unverified"
                ),
            }
        )
        phase = "binding"
        if (
            validation.artifact_sha256 != audited_sha
            or _artifact_bytes(result.model)[1] != audited_sha
        ):
            raise SourceContractDrift(
                "Metadata validator changed its audited artifact epoch"
            )
        capture.require_stable_target("after-metadata-runtime")
        capture.finalize_source_structure(result.model)
        if (
            capture.readiness is not CheckStatus.VALIDATED
            or capture.source_structure_status() is not CheckStatus.VALIDATED
        ):
            raise ValueError("Metadata runtime invalidated the source/artifact fence")
        observation["binding_detail"] = (
            "Same retained target and audited artifact fenced before/after target execution"
        )
        return observed_result
    except Exception as exc:
        capture.failure("metadata-runtime-" + phase, exc)
        status = "failed" if isinstance(exc, SourceContractDrift) else "unverified"
        observation.update(failure_phase=phase, detail=str(exc))
        if phase in {"binding", "reference"}:
            observation[phase + "_status"] = status
            observation[phase + "_detail"] = str(exc)
        else:
            observation["validator_observation"] = (
                "No returned validator evidence; runtime remains unverified"
            )
        return observed_result


def execute_raw_probe_cell(
    row: ProbeRow,
    cell: ProbeCell,
    registrations: Sequence[tuple[str, str, Mapping[str, Any]]],
    *,
    stage: Literal["prepare", "product"],
    adapters: Mapping[str, Any],
    evidence_directory: Path,
    runtime_optimization: Literal["disabled", "all"] | None = None,
) -> ProbeCell:
    """Execute one unchanged raw recipe; both CLI and metadata use this owner.

    Runtime is an explicitly requested test contract and reuses the same retained
    target and artifact. It is never enabled by ordinary raw product observation.
    """
    if stage not in {"prepare", "product"} or cell.scenario not in SCENARIOS:
        raise ValueError("Unknown raw source stage or scenario")
    if (row.origin, row.ordinal) != (cell.origin, cell.ordinal):
        raise ValueError("Raw cell and row provenance differ")
    if cell.stage not in {"scheduled", "recipe-pending", "source-prepared"}:
        raise ValueError("Raw cell already has a terminal observation")
    if any(
        CheckStatus.FAILED
        in (
            item.source_reference,
            item.schema,
            item.artifact,
            item.lowering_audit,
            item.structure,
            item.runtime,
            item.numerics,
        )
        for item in cell.history
    ):
        raise ValueError("Prior failed source evidence cannot be overwritten")
    from tests.extra_tests.converter.source_precision_raw_replay import (
        RawReplayCapture,
        prepare_raw_source_case,
        replay_adapter,
        run_raw_source_case,
        structured_source_error,
    )

    registered = {origin: metadata for origin, _, metadata in registrations}
    if len(registered) != len(registrations):
        raise ValueError("Ambiguous raw registration provenance")
    adapter: Mapping[str, Any] = {}
    environment = ProbeEnvironment.observe()
    capture = RawReplayCapture(evidence_directory / stage / _fingerprint(cell.identity))
    prepared = None
    result = None
    declared = None
    scenario_sha = _fingerprint(
        {
            "identity": cell.identity,
            "contract": row.contract_sha256,
            "adapter": adapters.get("rows", {}).get(f"{row.origin}::{row.ordinal}", {}),
            "criteria": asdict(row.criteria) if row.criteria else None,
        }
    )
    observed_settings: list[SourceSettings] = []
    outcome = cell
    try:
        metadata = _resolve_case(row, registered)
        adapter = replay_adapter(row, adapters)
        declared = raw_source_scenarios(
            metadata,
            SourceProvenance(row.origin, row.ordinal),
            criteria=row.criteria,
            preserve_supplied_properties=True,
        )[SCENARIOS.index(cell.scenario)]
        if declared.input_values is not None:
            capture.arrays("supplied-inputs", declared.input_values)
        capture.document("adapter", adapter)
        scenario_sha = _fingerprint(
            {
                "identity": cell.identity,
                "contract": row.contract_sha256,
                "adapter": adapter,
                "criteria": asdict(row.criteria) if row.criteria else None,
            }
        )
        prepared = prepare_raw_source_case(
            metadata,
            declared,
            adapter=adapter,
            capture=capture,
            settings_observer=observed_settings.append,
        )
        outcome = replace(
            cell, stage="source-prepared", source_reference=capture.readiness
        )
        if cell.history and (
            cell.history[-1].environment != environment
            or cell.history[-1].scenario_sha256 != scenario_sha
            or cell.history[-1].inputs != prepared.input_evidence
            or cell.history[-1].references != prepared.reference_evidence
            or cell.history[-1].original_trace_json != prepared.original_trace_json
            or cell.history[-1].actual_settings != prepared.settings
        ):
            outcome = replace(
                outcome,
                stage="source-drift",
                category="migration-gap",
                source_reference=CheckStatus.FAILED,
                detail="Original source inputs/reference/trace/settings changed; prior history retained.",
            )
        elif stage == "product":
            # Ordinary validator remains active; no diagnostic adapter or ORT.
            result = run_raw_source_case(prepared, capture=capture)
            report = (
                get_precision_report(result.model) if result.model is not None else None
            )
            lowering = (
                next(
                    (
                        check.status
                        for check in report.checks
                        if check.category == "lowering"
                    ),
                    CheckStatus.UNVERIFIED,
                )
                if report
                else (
                    capture.records.get("product_audit_status", CheckStatus.UNVERIFIED)
                )
            )
            outcome = replace(
                outcome,
                stage=result.stage,
                category=result.category,
                schema=result.schema,
                artifact=(
                    CheckStatus.VALIDATED
                    if capture.records.get("artifact", {}).get("status") == "captured"
                    else CheckStatus.UNVERIFIED
                ),
                lowering_audit=lowering,
                detail=result.audit.detail,
                artifact_sha256=capture.records.get("artifact", {}).get("sha256"),
            )
    except Exception as exc:
        capture.failure("preparation", exc)
        capture.records["source_error"] = structured_source_error(exc)
        unavailable = cell.origin not in registered
        category = (
            "environment-unavailable"
            if unavailable
            else getattr(exc, "category", "replay-lane-gap")
        )
        lane_gap = category in {"replay-lane-gap", "environment-unavailable"}
        outcome = replace(
            cell,
            stage="replay-adapter-pending" if lane_gap else "source-preparation",
            category=category,
            source_reference=(
                CheckStatus.UNVERIFIED if lane_gap else CheckStatus.FAILED
            ),
            detail=str(exc),
        )
    report = (
        get_precision_report(result.model)
        if result is not None and result.model is not None
        else None
    )
    try:
        supplied_evidence = (
            tuple(
                value_evidence(
                    value,
                    _properties(
                        value
                        if hasattr(value, "dtype") and hasattr(value, "shape")
                        else np.asarray(value)
                    ),
                )
                for value in (declared.input_values or ())
            )
            if declared is not None
            else ()
        )
    except Exception as exc:
        capture.gap("supplied-input-evidence", exc)
        supplied_evidence = ()
    try:
        checks = _raw_metadata_checks(prepared, result, capture)
    except Exception as exc:
        checks = dict(capture.records.get("metadata_assertions", {}))
        checks["metadata_assertion_observer"] = {
            "status": "unverified",
            "detail": f"Unsupported original assertion observer: {type(exc).__name__}: {exc}",
        }
    if stage == "product" and runtime_optimization is not None:
        result = _raw_metadata_runtime(
            prepared,
            result,
            capture,
            checks,
            adapter=adapter,
            optimization=runtime_optimization,
        )
        try:
            checks = _raw_metadata_checks(prepared, result, capture)
        except Exception as exc:
            checks.setdefault(
                "metadata_assertion_observer",
                {
                    "status": "unverified",
                    "detail": f"Unsupported original assertion observer: {type(exc).__name__}: {exc}",
                },
            )
    capture.records["metadata_assertions"] = checks
    if (
        capture.readiness is CheckStatus.UNVERIFIED
        and capture.gaps
        and outcome.source_reference is not CheckStatus.FAILED
    ):
        outcome = replace(
            outcome,
            source_reference=CheckStatus.UNVERIFIED,
            stage="replay-capture-pending",
            category="replay-lane-gap",
            detail="; ".join(
                gap["role"] + ": " + gap["detail"] for gap in capture.gaps
            ),
        )
    evidence = ProbeStageEvidence(
        cell.identity,
        stage,
        row.contract_sha256,
        scenario_sha,
        environment,
        cell.scenario == "raw-x64-on",
        None,
        (
            prepared.settings
            if prepared
            else observed_settings[-1] if observed_settings else None
        ),
        result.export_settings if result else None,
        prepared.original_trace_json if prepared else None,
        (
            hashlib.sha256(prepared.original_trace_json.encode()).hexdigest()
            if prepared
            else None
        ),
        supplied_evidence,
        prepared.input_evidence if prepared else (),
        prepared.reference_evidence if prepared else (),
        outcome.source_reference,
        capture.source_structure_status(),
        outcome.schema,
        outcome.artifact,
        outcome.lowering_audit,
        _raw_runtime_status(result),
        result.numerical_status if result else CheckStatus.UNVERIFIED,
        outcome.stage,
        outcome.category,
        outcome.detail,
        outcome.artifact_sha256,
        report,
        (),
        cell.history[-1].sha256 if cell.history else None,
        raw_callback_status=(result.raw_callback_status if result else "not-executed"),
        replay_capture=dict(capture.records),
    )
    return replace(
        outcome,
        runtime=_raw_runtime_status(result),
        numerics=result.numerical_status if result else CheckStatus.UNVERIFIED,
        history=(*cell.history, evidence),
    )


def _execute_raw_probe_plan(
    plan: SourceProbePlan,
    registrations: Sequence[tuple[str, str, Mapping[str, Any]]],
    *,
    stage: Literal["prepare", "product"],
    limit: int,
    origins: frozenset[str],
    scenario: str | None,
    cases: frozenset[SourceProvenance],
    adapters: Mapping[str, Any],
    evidence_directory: Path | None,
) -> SourceProbePlan:
    """Complete-denominator native/product observations, retaining every cell."""
    import tempfile

    registered = {origin: metadata for origin, _, metadata in registrations}
    if len(registered) != len(registrations):
        raise ValueError("Ambiguous raw registration provenance")
    rows = {(row.origin, row.ordinal): row for row in plan.rows}
    if any((case.origin, case.ordinal) not in rows for case in cases):
        raise ValueError("Requested exact case is absent from the frozen plan")
    directory = evidence_directory
    if directory is None:
        directory = Path(tempfile.mkdtemp(prefix="jax2onnx-raw-replay-"))
    else:
        directory.mkdir(parents=True, exist_ok=True)
    selected = 0
    cells = []
    for cell in plan.cells:
        if (
            selected >= limit
            or (origins and cell.origin not in origins)
            or (scenario is not None and cell.scenario != scenario)
            or (cases and SourceProvenance(cell.origin, cell.ordinal) not in cases)
            or cell.stage not in {"scheduled", "recipe-pending", "source-prepared"}
            or any(
                CheckStatus.FAILED
                in (
                    item.source_reference,
                    item.schema,
                    item.artifact,
                    item.lowering_audit,
                    item.structure,
                    item.runtime,
                    item.numerics,
                )
                for item in cell.history
            )
        ):
            cells.append(cell)
            continue
        selected += 1
        cells.append(
            execute_raw_probe_cell(
                rows[(cell.origin, cell.ordinal)],
                cell,
                registrations,
                stage=stage,
                adapters=adapters,
                evidence_directory=directory,
            )
        )
    return replace(plan, cells=tuple(cells))


_CORPUS_CODE_FILES: Final = (
    "tests/extra_tests/converter/source_precision_support.py",
    "scripts/source_precision_probe.py",
    "tests/extra_tests/converter/source_precision_corpus_oracles.py",
    "tests/extra_tests/converter/test_source_precision_corpus_replay.py",
)
_CORPUS_IMPORTS: Final = (
    "scripts.source_precision_probe",
    "tests.extra_tests.converter.source_precision_support",
    "tests.extra_tests.converter.source_precision_corpus_oracles",
    "tests.extra_tests.converter.source_precision_diagnostics",
    "jax2onnx.plugins.jax.lax.dot_general",
    "jax2onnx.plugins.jax.lax.gather",
    "jax2onnx.plugins.jax.numpy.matmul",
    "jax2onnx.plugins.jax.numpy.add",
    "jax2onnx.user_interface",
    "jax2onnx.converter.conversion_api",
    "jax2onnx.precision",
)
_PRIOR_CORPUS_HASHES: Final = {
    "3.12": frozenset(
        {
            "b5ae12dc4cd745dfa72989f09455316841d9487a793757725e4b0c6e2d9451d9",
            "c1d07feb4ac540645ba3f44247770ae581bf16a8876540c4d367d3e1b89bbdeb",
        }
    ),
    "3.11": frozenset(
        {
            "d61317d5ef72fec347cc4911362b2a0b034039a696447ae9f2dfeb7aa5916940",
            "10d2c3fe6491130c8d38230930221c6bfef4a02981a025c325a7cecae1067cc8",
        }
    ),
}
_STABLE_SOURCE_FIELDS: Final = (
    "row_contract_sha256",
    "scenario_sha256",
    "environment",
    "requested_x64",
    "requested_matmul_precision",
    "actual_settings",
    "original_trace_json",
    "original_trace_sha256",
    "supplied_inputs",
    "inputs",
    "references",
)


def corpus_replay_cases(plan: SourceProbePlan) -> frozenset[SourceProvenance]:
    """Resolve only the approved13 unchanged rows, never an origin-wide selector."""
    if (
        plan.baseline_sha256 != HISTORICAL_BASELINE_SHA256
        or len(plan.rows) != 1802
        or len(plan.cells) != 3604
    ):
        raise ValueError(
            "Corpus revision requires the unchanged historical raw denominator"
        )
    rows = {(row.origin, row.ordinal): row for row in plan.rows}
    for case in CORPUS_REPLAY_CASES:
        row = rows.get((case.origin, case.ordinal))
        if (
            row is None
            or row.contract_sha256 != case.contract_sha256
            or row.display[-1] != case.testcase
            or row.criteria is not None
            or row.unresolved_recipes
        ):
            raise ValueError(
                "Fixed corpus selector/contract or unresolved criteria changed"
            )
    return frozenset(
        SourceProvenance(case.origin, case.ordinal) for case in CORPUS_REPLAY_CASES
    )


def _corpus_code_snapshot() -> dict[str, Any]:
    code = {
        name: hashlib.sha256((REPOSITORY_ROOT / name).read_bytes()).hexdigest()
        for name in _CORPUS_CODE_FILES
    }
    imported = {}
    for name in _CORPUS_IMPORTS:
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(REPOSITORY_ROOT):
            raise ValueError(f"Replay imported another checkout: {name}: {path}")
        imported[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return {"owned_files": code, "imported_modules": imported}


def _corpus_prior_comparisons(
    plan: SourceProbePlan, histories: Sequence[Mapping[str, Any]], *, exported: bool
) -> list[dict[str, Any]]:
    current = {
        tuple(cell.identity): json.loads(json.dumps(asdict(cell)))
        for cell in plan.cells
        if cell.history
    }
    fields = _STABLE_SOURCE_FIELDS + (("export_settings",) if exported else ())
    comparisons = []
    for prior in histories:
        for old in prior["cells"]:
            key = (old["origin"], old["ordinal"], old["scenario"])
            if key not in current or not old["history"]:
                continue
            old_record, new_record = old["history"][-1], current[key]["history"][-1]
            differences = [
                name for name in fields if old_record[name] != new_record[name]
            ]
            comparisons.append(
                {
                    "identity": key,
                    "prior_evidence_sha256": _fingerprint(old_record),
                    "current_evidence_sha256": _fingerprint(new_record),
                    "compared_fields": fields,
                    "drift_fields": differences,
                }
            )
    return comparisons


def _corpus_product_controls(
    plan: SourceProbePlan,
    registrations: Sequence[tuple[str, str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    registered = {origin: metadata for origin, _, metadata in registrations}
    controls = []
    for origin in ("plugin::jax.numpy.add", "plugin::dot_general"):
        row = next(
            row for row in plan.rows if row.origin == origin and row.ordinal == 0
        )
        if row.contract_sha256 != _CONTROL_CONTRACTS[origin]:
            raise ValueError("The fixed audited admission control contract changed")
        counterpart = counterpart_for(origin, 0, row.contract_sha256)
        if origin == "plugin::dot_general" and counterpart != DOT_COUNTERPARTS[0]:
            raise ValueError(
                "The fixed audited dot control needs its reviewed counterpart"
            )
        metadata = _resolve_case(row, registered)
        for scenario in raw_source_scenarios(metadata, SourceProvenance(origin, 0)):
            result = run_source_case(
                metadata,
                scenario,
                optimization=None,
                _corpus_counterpart=counterpart,
            )
            report = (
                get_precision_report(result.model) if result.model is not None else None
            )
            checks = (
                {check.category: check.status for check in report.checks}
                if report
                else {}
            )
            controls.append(
                {
                    "control_revision": CORPUS_CONTROL_REVISION,
                    "origin": origin,
                    "ordinal": 0,
                    "scenario": scenario.name,
                    "contract_sha256": row.contract_sha256,
                    "stage": result.stage,
                    "audit": asdict(result.audit),
                    "schema": result.schema,
                    "export_report": asdict(report) if report is not None else None,
                    "runtime": checks.get("runtime-execution", CheckStatus.UNVERIFIED),
                    "raw_callback_status": result.raw_callback_status,
                    "corpus_structure": (
                        asdict(result.corpus_structure)
                        if result.corpus_structure is not None
                        else None
                    ),
                    "artifact_sha256": (
                        result.validation.artifact_sha256
                        if result.validation is not None
                        else None
                    ),
                    "numerics": result.numerical_status,
                }
            )
    return controls


def _corpus_controls_admitted(controls: Sequence[Mapping[str, Any]]) -> bool:
    """Require all four fixed audited controls, including the retained raw failure."""
    expected = [
        (origin, 0, scenario)
        for origin in ("plugin::jax.numpy.add", "plugin::dot_general")
        for scenario in SCENARIOS
    ]
    if [
        (c.get("origin"), c.get("ordinal"), c.get("scenario")) for c in controls
    ] != expected:
        return False
    for control in controls:
        report = control.get("export_report")
        digest = control.get("artifact_sha256")
        if (
            control.get("control_revision") != CORPUS_CONTROL_REVISION
            or control.get("contract_sha256") != _CONTROL_CONTRACTS[control["origin"]]
            or control.get("stage") != "artifact-checked"
            or control.get("audit", {}).get("status") != CheckStatus.VALIDATED
            or control.get("schema") != CheckStatus.VALIDATED
            or control.get("runtime") != CheckStatus.UNVERIFIED
            or control.get("numerics") != CheckStatus.UNVERIFIED
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or not isinstance(report, Mapping)
        ):
            return False
        checks = report.get("checks", ())
        if (
            report.get("profile") != "source-static-v1"
            or report.get("opset") != 23
            or report.get("schema_version") != 1
            or len(checks) != 4
            or {check["category"]: check["status"] for check in checks}
            != {
                "representation": CheckStatus.VALIDATED,
                "lowering": CheckStatus.VALIDATED,
                "runtime-execution": CheckStatus.UNVERIFIED,
                "numerical-acceptance": CheckStatus.UNVERIFIED,
            }
        ):
            return False
        structure = control.get("corpus_structure")
        if control["origin"] == "plugin::jax.numpy.add":
            if structure is not None or control.get("raw_callback_status") != "passed":
                return False
        elif (
            not isinstance(structure, Mapping)
            or structure.get("counterpart_identifier")
            != "static-dot-source-counterpart-v3:plugin::dot_general:0"
            or structure.get("counterpart_sha256")
            # The descriptor also binds the reviewed whole oracle module.
            != "c390e6d63e3902f436e69c8527e50d9cc550181eb451bb72d56792dbcff0db93"
            or structure.get("source_structure") != CheckStatus.VALIDATED
            or structure.get("raw_callback_status") != "failed"
            or control.get("raw_callback_status") != "failed"
        ):
            return False
    return True


def execute_corpus_revision(
    inventory_blob: bytes,
    registrations: Sequence[tuple[str, str, Mapping[str, Any]]],
    *,
    prior_paths: Sequence[Path],
    preparation_output: Path,
) -> dict[str, Any]:
    """A separate dated revision supplement; original failed histories are immutable."""
    baseline_sha = hashlib.sha256(inventory_blob).hexdigest()
    plan = build_probe_plan(json.loads(inventory_blob), baseline_sha256=baseline_sha)
    cases = corpus_replay_cases(plan)
    preflight_cases(plan, registrations, cases)
    if preparation_output.exists():
        raise ValueError("Retain the frozen preparation; choose a new output path")
    prior_bytes = {str(path.resolve()): path.read_bytes() for path in prior_paths}
    expected = _PRIOR_CORPUS_HASHES.get(
        f"{sys.version_info.major}.{sys.version_info.minor}"
    )
    if (
        len(prior_bytes) != 2
        or {hashlib.sha256(blob).hexdigest() for blob in prior_bytes.values()}
        != expected
    ):
        raise ValueError(
            "Replay needs both unchanged same-stack original36 and two-dot files"
        )
    histories = [json.loads(blob) for blob in prior_bytes.values()]
    if any(item["baseline_sha256"] != baseline_sha for item in histories):
        raise ValueError("Prior observations use another raw baseline")
    unselected = tuple(
        cell
        for cell in plan.cells
        if SourceProvenance(cell.origin, cell.ordinal) not in cases
    )
    if (
        len(unselected) != 3578
        or sum(cell.stage == "recipe-pending" for cell in unselected) != 126
    ):
        raise ValueError("The unselected raw denominator changed")
    code = _corpus_code_snapshot()
    ledger = criterion_proposal_ledger()
    configuration = {
        "schema_version": 1,
        "kind": "separate-static-corpus-revision-supplement",
        "control_revision": CORPUS_CONTROL_REVISION,
        "date": "2026-10-18",
        "checkout": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
        ).strip(),
        "checkout_root": str(REPOSITORY_ROOT),
        "baseline_sha256": baseline_sha,
        "selector": selector_manifest(),
        "counterparts": [
            item.descriptor() for item in (*DOT_COUNTERPARTS, *GATHER_COUNTERPARTS)
        ],
        "criterion_proposal": ledger,
        "criterion_proposal_sha256": _fingerprint(ledger),
        "code": code,
        "prior_files": {
            name: hashlib.sha256(blob).hexdigest() for name, blob in prior_bytes.items()
        },
        "unselected": {
            "count": 3578,
            "recipe_pending": 126,
            "sha256": _fingerprint([asdict(cell) for cell in unselected]),
        },
        "runtime_policy": "Artifact-only; numerical proposal unadopted; no samples or ORT sessions",
        "symbolic_rows_unresolved": 310,
        "nchw_rows_unresolved": 7,
    }
    attempts: list[str] = []
    controls_before: list[dict[str, Any]] = []
    controls_after: list[dict[str, Any]] = []
    caller = SourceSettings.observe()

    def deny_runtime(*args: Any, **kwargs: Any) -> Any:
        attempts.append("InferenceSession attempted")
        raise AssertionError("Corpus artifact revision forbids ORT sessions")

    with patch("onnxruntime.InferenceSession", side_effect=deny_runtime):
        prepared = execute_probe_plan(
            plan,
            registrations,
            stage="prepare",
            limit=26,
            cases=cases,
            _corpus_replay=True,
        )
        comparisons = _corpus_prior_comparisons(prepared, histories, exported=False)
        frozen = {
            **configuration,
            "preparation": [
                asdict(cell)
                for cell in prepared.cells
                if SourceProvenance(cell.origin, cell.ordinal) in cases
            ],
            "prior_source_comparisons": comparisons,
        }
        preparation_output.write_text(
            json.dumps(frozen, indent=2, sort_keys=True) + "\n"
        )
        if any(item["drift_fields"] for item in comparisons):
            result = prepared
            state = "blocked-source-drift-before-export"
        else:
            controls_before = _corpus_product_controls(plan, registrations)
            result = execute_probe_plan(
                prepared,
                registrations,
                stage="diagnostic",
                limit=26,
                cases=cases,
                _corpus_replay=True,
            )
            controls_after = _corpus_product_controls(plan, registrations)
            state = "observed-no-numerical-acceptance"
    final_unselected = tuple(
        cell
        for cell in result.cells
        if SourceProvenance(cell.origin, cell.ordinal) not in cases
    )
    final_comparisons = _corpus_prior_comparisons(result, histories, exported=True)
    unchanged_history = all(
        Path(name).read_bytes() == blob for name, blob in prior_bytes.items()
    )
    invariants = {
        "unselected_3578_unchanged": final_unselected == unselected,
        "126_recipe_pending_unchanged": sum(
            cell.stage == "recipe-pending" for cell in final_unselected
        )
        == 126,
        "prior_file_bytes_unchanged": unchanged_history,
        "code_unchanged": _corpus_code_snapshot() == code,
        "caller_settings_restored": SourceSettings.observe() == caller,
        "zero_ort_attempts": not attempts,
        "all_selected_criteria_unresolved": all(
            row.criteria is None
            for row in result.rows
            if SourceProvenance(row.origin, row.ordinal) in cases
        ),
        "lowering_runtime_numerics_unverified": all(
            cell.lowering_audit
            is cell.runtime
            is cell.numerics
            is CheckStatus.UNVERIFIED
            for cell in result.cells
            if SourceProvenance(cell.origin, cell.ordinal) in cases
        ),
        "same_stack_source_reference_unchanged": not any(
            item["drift_fields"] for item in final_comparisons
        ),
        "default_admission_controls_unchanged": bool(controls_before)
        and controls_before == controls_after,
        "audited_add_keeps_default_admission": _corpus_controls_admitted(controls_after)
        and all(
            item["audit"]["status"] is CheckStatus.VALIDATED
            for item in controls_after
            if item["origin"] == "plugin::jax.numpy.add"
        ),
        "audited_ordinary_matrix_keeps_default_admission": _corpus_controls_admitted(
            controls_after
        ),
    }
    return {
        **configuration,
        "status": state,
        "preparation_file": str(preparation_output),
        "preparation_sha256": hashlib.sha256(
            preparation_output.read_bytes()
        ).hexdigest(),
        "prior_source_comparisons": final_comparisons,
        "selected": [
            asdict(cell)
            for cell in result.cells
            if SourceProvenance(cell.origin, cell.ordinal) in cases
        ],
        "invariants": invariants,
        "ort_session_attempts": attempts,
        "controls_before": controls_before,
        "controls_after": controls_after,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--corpus-replay", action="store_true")
    parser.add_argument("--preparation-output", type=Path)
    parser.add_argument("--history", type=Path, action="append", default=[])
    parser.add_argument(
        "--stage", choices=("plan", "prepare", "product", "diagnostic"), default="plan"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--origin", action="append", default=[])
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--evidence-directory", type=Path)
    parser.add_argument(
        "--replay-adapters",
        type=Path,
        default=REPOSITORY_ROOT / "internal_docs/precision/source_replay_adapters.json",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Exact raw origin::ordinal, e.g. plugin::add::0",
    )
    args = parser.parse_args(argv)
    if args.corpus_replay:
        if (
            args.limit != 26
            or args.stage != "plan"
            or args.origin
            or args.case
            or args.scenario
            or args.evidence_directory is not None
            or args.output is None
            or args.preparation_output is None
        ):
            parser.error(
                "Corpus revision needs --limit26, --output, --preparation-output and both --history files; no stage/origin/case/scenario override"
            )
        if args.output.resolve() == args.preparation_output.resolve():
            parser.error(
                "--output and --preparation-output need distinct resolved paths"
            )
        if args.output.exists():
            parser.error("Retain prior supplement bytes; choose a new output path")
        payload = subprocess.check_output(
            [
                "git",
                "show",
                "158ce625:internal_docs/precision/source_migration_inventory.json",
            ],
            cwd=REPOSITORY_ROOT,
        )
        registrations, _ = raw_registrations()
        result = execute_corpus_revision(
            payload,
            registrations,
            prior_paths=args.history,
            preparation_output=args.preparation_output,
        )
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "selected_cells": len(result["selected"]),
                    "invariants": result["invariants"],
                },
                sort_keys=True,
            )
        )
        return 0 if all(result["invariants"].values()) else 1
    if args.preparation_output is not None or args.history:
        parser.error("Corpus supplement options require --corpus-replay")
    if args.stage != "plan" and (args.limit is None or args.limit <= 0):
        parser.error(
            "Execution needs an explicit positive --limit; default planning executes no models."
        )
    if args.stage != "plan" and args.output is not None and args.output.exists():
        parser.error("Retain previous batch bytes; choose a fresh --output path")
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
            raw_replay=args.stage in {"prepare", "product"},
            replay_adapters=(
                json.loads(args.replay_adapters.read_text())
                if args.stage in {"prepare", "product"}
                else None
            ),
            evidence_directory=args.evidence_directory,
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
