# scripts/source_precision_numerical_probe.py

"""Prepare the fixed approved FP32 counterpart; --execute needs freeze acceptance.

Default preparation constructs no runtime sessions. The artifact-only probe and
all original source criteria/histories remain unchanged.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
from unittest.mock import patch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import source_precision_probe as artifact_probe  # noqa: E402
from scripts.source_precision_inventory import raw_registrations  # noqa: E402
from tests.extra_tests.converter import (  # noqa: E402
    source_precision_corpus_numerics as numerics,
)
from tests.extra_tests.converter.source_precision_corpus_oracles import (  # noqa: E402
    HISTORICAL_BASELINE_SHA256,
    fingerprint,
)
from tests.extra_tests.converter.source_precision_support import (  # noqa: E402
    SourceProvenance,
    SourceSettings,
    _global_source_configuration,
)

_NEW_FILES = (
    "tests/extra_tests/converter/source_precision_corpus_numerics.py",
    "scripts/source_precision_numerical_probe.py",
    "tests/extra_tests/converter/test_source_precision_corpus_numerics.py",
)
_PROTECTED_FILES = (
    "jax2onnx/converter/source_profile.py",
    "jax2onnx/plugins/jax/numpy/_common.py",
    "jax2onnx/plugins/jax/lax/min.py",
    "jax2onnx/plugins/jax/lax/max.py",
    "jax2onnx/plugins/jax/lax/_min_max_utils.py",
)


def code_snapshot() -> dict[str, Any]:
    existing = artifact_probe._corpus_code_snapshot()
    imported = {}
    for name in (
        "tests.extra_tests.converter.source_precision_corpus_numerics",
        "onnxruntime.capi.onnxruntime_pybind11_state",
    ):
        path = Path(importlib.import_module(name).__file__).resolve()
        if name.startswith("tests.") and not path.is_relative_to(REPOSITORY_ROOT):
            raise numerics.NumericalPreflightError(
                "Numerical helper imported another checkout"
            )
        imported[name] = dict(path=str(path), sha256=numerics.sha256(path.read_bytes()))
    return dict(
        artifact_code=existing,
        numerical_files={
            p: numerics.sha256((REPOSITORY_ROOT / p).read_bytes()) for p in _NEW_FILES
        },
        protected_files={
            p: numerics.sha256((REPOSITORY_ROOT / p).read_bytes())
            for p in _PROTECTED_FILES
        },
        numerical_imports=imported,
    )


def fresh_paths(output: Path, preparation: Path) -> Path:
    payloads = preparation.with_name(preparation.name + ".payloads")
    paths = [p.resolve() for p in (output, preparation, payloads)]
    if len(set(paths)) != 3 or any(
        a in b.parents for a in paths for b in paths if a != b
    ):
        raise numerics.NumericalPreflightError(
            "Output, preparation and payload directory need distinct resolved paths"
        )
    if any(p.exists() or p.is_symlink() for p in (output, preparation, payloads)):
        raise numerics.NumericalPreflightError(
            "Retain existing evidence; all output paths must be new"
        )
    if not output.parent.is_dir() or not preparation.parent.is_dir():
        raise numerics.NumericalPreflightError("Output parents must already exist")
    return payloads


def validate_accepted_cells(
    approval: Any, record: Any, label: str
) -> list[dict[str, Any]]:
    cases = [asdict(case) for case in numerics.NUMERICAL_CASES]
    if (
        len(cases) != 12
        or approval["cases"] != cases
        or approval["criterion_version"] != numerics.CRITERION_VERSION
    ):
        raise numerics.NumericalPreflightError(
            "Approved numerical selector/version changed"
        )
    expected = approval["accepted_main_artifact_records"][label]["cells"]
    selected = [
        c
        for c in record["selected"]
        if (c["origin"], c["ordinal"]) in {(v["origin"], v["ordinal"]) for v in cases}
    ]
    if (
        len(selected) != 24
        or len({(c["origin"], c["ordinal"], c["scenario"]) for c in selected}) != 24
        or len(expected) != 24
    ):
        raise numerics.NumericalPreflightError(
            "Require all twenty-four original off/on cells"
        )
    for cell, bound in zip(selected, expected, strict=True):
        old = cell["history"][-1]
        for key, value in bound.items():
            actual = (
                old[key]
                if key in {"scenario_sha256", "original_trace_sha256"}
                else cell[key]
            )
            if actual != value:
                raise numerics.NumericalPreflightError(
                    "Approved artifact/cell/trace binding changed"
                )
        case = next(
            c
            for c in cases
            if (c["origin"], c["ordinal"]) == (cell["origin"], cell["ordinal"])
        )
        if (
            old["row_contract_sha256"] != case["contract_sha256"]
            or any(
                cell[k] != "validated"
                for k in ("schema", "artifact", "source_reference")
            )
            or any(
                cell[k] != "unverified"
                for k in ("lowering_audit", "runtime", "numerics")
            )
        ):
            raise numerics.NumericalPreflightError(
                "Accepted artifact or original audit statuses changed"
            )
        if cell["scenario"] not in artifact_probe.SCENARIOS:
            raise numerics.NumericalPreflightError("Unknown raw scenario")
    return selected


def load_approved_context(approval_path: Path, accepted_path: Path) -> dict[str, Any]:
    approval_blob, accepted_blob = (
        approval_path.read_bytes(),
        accepted_path.read_bytes(),
    )
    if numerics.sha256(approval_blob) != numerics.APPROVAL_SHA256:
        raise numerics.NumericalPreflightError(
            "Missing or changed coordinator criterion approval"
        )
    approval = json.loads(approval_blob)
    label = f"py{sys.version_info.major}{sys.version_info.minor}"
    if label not in approval["accepted_main_artifact_records"]:
        raise numerics.NumericalPreflightError(
            "No accepted record for this interpreter stack"
        )
    expected = approval["accepted_main_artifact_records"][label]
    if numerics.sha256(accepted_blob) != expected["sha256"]:
        raise numerics.NumericalPreflightError(
            "Accepted actual-main artifact file digest changed"
        )
    record = json.loads(accepted_blob)
    if (
        record["baseline_sha256"] != HISTORICAL_BASELINE_SHA256
        or not all(record["invariants"].values())
        or record["ort_session_attempts"]
    ):
        raise numerics.NumericalPreflightError(
            "Accepted artifact provenance/invariants changed"
        )
    cells = validate_accepted_cells(approval, record, label)
    code = code_snapshot()
    current = code["artifact_code"]
    if current["owned_files"] != record["code"]["owned_files"]:
        raise numerics.NumericalPreflightError(
            "Accepted artifact tooling bytes changed"
        )
    proof = {}
    for name, old in record["code"]["imported_modules"].items():
        relative = Path(old["path"]).relative_to(record["checkout_root"])
        now = current["imported_modules"][name]
        if now["sha256"] != old["sha256"]:
            raise numerics.NumericalPreflightError(
                "Accepted source module bytes changed: " + name
            )
        proof[str(relative)] = old["sha256"]
    proof.update(record["code"]["owned_files"])
    proof.update(code["protected_files"])
    for path, digest in proof.items():
        committed = subprocess.check_output(
            ["git", "show", numerics.COMMITTED_SOURCE_LINEAGE + ":" + path],
            cwd=REPOSITORY_ROOT,
        )
        if numerics.sha256(committed) != digest:
            raise numerics.NumericalPreflightError(
                "Accepted recorded source bytes do not match the declared committed lineage"
            )
    historical = {
        str(approval_path.resolve()): numerics.sha256(approval_blob),
        str(accepted_path.resolve()): numerics.sha256(accepted_blob),
        record["preparation_file"]: record["preparation_sha256"],
        **record["prior_files"],
    }
    for path, digest in historical.items():
        if numerics.sha256(Path(path).read_bytes()) != digest:
            raise numerics.NumericalPreflightError("Historical evidence bytes changed")
    return dict(
        approval=approval,
        accepted=record,
        selected=cells,
        code=code,
        historical_files=historical,
        lineage=dict(
            accepted_recorded_checkout=record["checkout"],
            accepted_recorded_checkout_root=record["checkout_root"],
            accepted_code_now_committed_as=numerics.COMMITTED_SOURCE_LINEAGE,
            current_checkout=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
            ).strip(),
            current_checkout_root=str(REPOSITORY_ROOT),
            source_hashes=proof,
            proof="Compare source bytes, not equality of the old checkout label and the later commit",
        ),
    )


def prepare_packet(
    context: Any, payloads: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inventory_blob = subprocess.check_output(
        [
            "git",
            "show",
            "158ce625:internal_docs/precision/source_migration_inventory.json",
        ],
        cwd=REPOSITORY_ROOT,
    )
    if numerics.sha256(inventory_blob) != HISTORICAL_BASELINE_SHA256:
        raise numerics.NumericalPreflightError("Historical baseline bytes changed")
    plan = artifact_probe.build_probe_plan(
        json.loads(inventory_blob), baseline_sha256=HISTORICAL_BASELINE_SHA256
    )
    registered, _ = raw_registrations()
    registrations = {origin: metadata for origin, _, metadata in registered}
    if len(registrations) != len(registered):
        raise numerics.NumericalPreflightError("Ambiguous raw registrations")
    cases = frozenset(
        SourceProvenance(c.origin, c.ordinal) for c in numerics.NUMERICAL_CASES
    )
    contract_errors = {}
    for case in cases:
        try:
            artifact_probe.preflight_cases(plan, registered, frozenset({case}))
        except Exception as exc:
            contract_errors[(case.origin, case.ordinal)] = (
                f"{type(exc).__name__}: {exc}"
            )
    rows = {(r.origin, r.ordinal): r for r in plan.rows}
    unselected = [
        c for c in plan.cells if SourceProvenance(c.origin, c.ordinal) not in cases
    ]
    if (
        len(unselected) != 3580
        or sum(c.stage == "recipe-pending" for c in unselected) != 126
    ):
        raise numerics.NumericalPreflightError(
            "Raw denominator/unselected recipes changed"
        )
    payloads.mkdir()
    prepared, pending = [], []
    preparation_attempts = []

    def deny_runtime(*args: Any, **kwargs: Any) -> Any:
        preparation_attempts.append(
            "ORT constructor attempted before preparation freeze"
        )
        raise AssertionError("Numerical preparation must finish before any ORT session")

    with patch("onnxruntime.InferenceSession", side_effect=deny_runtime):
        for index, cell in enumerate(context["selected"]):
            try:
                if (cell["origin"], cell["ordinal"]) in contract_errors:
                    raise numerics.NumericalPreflightError(
                        contract_errors[(cell["origin"], cell["ordinal"])]
                    )
                row = rows[(cell["origin"], cell["ordinal"])]
                metadata = artifact_probe._resolve_case(row, registrations)
                prepared.append(
                    numerics.prepare_numerical_cell(
                        row, metadata, cell, payloads / f"cell-{index:02d}"
                    )
                )
            except Exception as exc:
                pending.append(
                    dict(
                        identity=[cell["origin"], cell["ordinal"], cell["scenario"]],
                        state="preflight-pending",
                        detail=f"{type(exc).__name__}: {exc}",
                        accepted_cell_sha256=fingerprint(cell),
                        runtime="unverified",
                        numerics="unverified",
                        lowering_audit="unverified",
                    )
                )
    if preparation_attempts:
        raise numerics.NumericalPreflightError(
            "ORT was attempted during source/artifact preparation"
        )
    return prepared, pending


def run_packet(
    *,
    approval: Path,
    accepted: Path,
    output: Path,
    preparation: Path,
    execute: bool = False,
) -> dict[str, Any]:
    payload_path = fresh_paths(output, preparation)
    context = load_approved_context(approval, accepted)
    before, global_before = SourceSettings.observe(), _global_source_configuration()
    cells, pending = prepare_packet(context, payload_path)
    if (
        len(cells) + len(pending) != 24
        or context["code"] != code_snapshot()
        or (SourceSettings.observe(), _global_source_configuration())
        != (before, global_before)
    ):
        raise numerics.NumericalPreflightError(
            "Preparation changed source code/settings or dropped selected cells"
        )
    frozen = dict(
        kind="separate-approved-fp32-numerical-counterpart",
        criterion=numerics.criterion_descriptor(),
        approval_sha256=numerics.APPROVAL_SHA256,
        approval=context["approval"],
        accepted_main_artifact_sha256=numerics.sha256(accepted.read_bytes()),
        accepted_main_artifact_path=str(accepted.resolve()),
        baseline_sha256=HISTORICAL_BASELINE_SHA256,
        source_lineage=context["lineage"],
        code=context["code"],
        environment=asdict(artifact_probe.ProbeEnvironment.observe()),
        historical_files=context["historical_files"],
        cells=cells,
        pending=pending,
        execution_plan=dict(
            cells=24,
            targets=list(numerics.TARGET_MODES),
            runs_per_cell_target=1,
            maximum_runs=48,
            execute_requested=execute,
            preparation_ort_attempts=0,
            all_or_nothing_preflight=True,
            provider="CPUExecutionProvider",
            intra_op_threads=1,
            inter_op_threads=1,
            enable_mem_pattern=False,
            disable_fallback=True,
            execution_mode="ORT_SEQUENTIAL",
        ),
        unselected=dict(
            count=3580,
            recipe_pending=126,
            symbolic_rows_unresolved=310,
            nchw_rows_unresolved=7,
        ),
        original_criteria=None,
        lowering_audit="unverified",
    )
    frozen_file = numerics.write_new(
        preparation, (json.dumps(frozen, indent=2, sort_keys=True) + "\n").encode()
    )
    executions = []
    if execute and not pending:
        payloads = [numerics.load_frozen_payloads(cell) for cell in cells]
        for cell, arrays in zip(cells, payloads, strict=True):
            for mode in numerics.TARGET_MODES:
                executions.append(
                    numerics.execute_frozen_cell(
                        cell, mode, preparation, frozen_file["sha256"], payloads=arrays
                    )
                )
    unchanged = all(
        numerics.sha256(Path(p).read_bytes()) == digest
        for p, digest in context["historical_files"].items()
    )
    invariants = dict(
        historical_files_unchanged=unchanged,
        code_unchanged=context["code"] == code_snapshot(),
        caller_settings_restored=(
            SourceSettings.observe(),
            _global_source_configuration(),
        )
        == (before, global_before),
        preparation_bytes_unchanged=numerics.sha256(preparation.read_bytes())
        == frozen_file["sha256"],
        all_selected_retained=len(cells) + len(pending) == 24,
        general_lowering_audit_unverified=all(
            e["lowering_audit"] == "unverified" for e in executions
        ),
        bounded_one_run=all(e["run_attempts"] <= 1 for e in executions),
    )
    status = (
        "preflight-pending"
        if pending
        else (
            "prepared-no-runtime"
            if not execute
            else (
                "validated-scoped-samples"
                if len(executions) == 48
                and all(
                    e["runtime"] == e["numerics"] == "validated" for e in executions
                )
                else "failed-scoped-samples"
            )
        )
    )
    result = dict(
        kind=frozen["kind"],
        status=status,
        preparation=frozen_file,
        criterion_version=numerics.CRITERION_VERSION,
        approval_sha256=numerics.APPROVAL_SHA256,
        executions=executions,
        pending=pending,
        invariants=invariants,
        source_lineage=context["lineage"],
        scope="Only these original feeds, two actual references, frozen artifacts and CPU targets; general lowering remains unverified",
    )
    numerics.write_new(
        output, (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    )
    return result


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--accepted-artifact", type=Path, required=True)
    parser.add_argument(
        "--criterion-version", choices=(numerics.CRITERION_VERSION,), required=True
    )
    parser.add_argument("--limit", type=int, choices=(24,), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preparation-output", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run only after independent implementation/code-freeze acceptance",
    )
    args = parser.parse_args(argv)
    try:
        fresh_paths(args.output, args.preparation_output)
        result = run_packet(
            approval=args.approval,
            accepted=args.accepted_artifact,
            output=args.output,
            preparation=args.preparation_output,
            execute=args.execute,
        )
    except (numerics.NumericalPreflightError, OSError, KeyError, ValueError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            dict(
                status=result["status"],
                executions=len(result["executions"]),
                pending=len(result["pending"]),
                invariants=result["invariants"],
            ),
            sort_keys=True,
        )
    )
    return (
        0
        if not result["pending"]
        and all(result["invariants"].values())
        and result["status"] in {"prepared-no-runtime", "validated-scoped-samples"}
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
