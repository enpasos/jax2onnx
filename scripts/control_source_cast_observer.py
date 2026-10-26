#!/usr/bin/env python3
# scripts/control_source_cast_observer.py

"""Run one supplemental observer-disabled control of a proved original failure.

The canonical failure remains authoritative. This fresh child reruns the native
source and ordinary assertions, including the complete assigned predecessor
history. It neither retries the canonical suite nor supplies replacement words.
"""

import argparse
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
import math
import os
from pathlib import Path
import sys
from typing import Any
import xml.etree.ElementTree as ET


_ROOT = Path(__file__).resolve().parents[1]
_SCHEMA = "source-cast-observer-off-v1"
_REQUEST_FIELDS = (
    "kind",
    "assigned",
    "full_preselection_ids",
    "collection_skips",
    "pytest_args",
)
_CONTEXT_FIELDS = {
    "status",
    "python",
    "executable",
    "prefix",
    "base_prefix",
    "soabi",
    "byteorder",
    "checkout",
    "cpu",
    "environment",
    "source_files",
    "sys_profile_is_none",
    "threading_profile_is_none",
    "loaded_jax_config",
    "errors",
}
_PROCESS_FIELDS = (
    "python",
    "executable",
    "prefix",
    "base_prefix",
    "soabi",
    "byteorder",
    "checkout",
    "cpu",
    "source_files",
)


def _load_script(name: str) -> Any:
    path = Path(__file__).with_name(name + ".py")
    spec = spec_from_file_location("_cast_control_" + name, path)
    if spec is None or spec.loader is None:
        raise ValueError("diagnostic module loader unavailable")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _wire(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()


def _context(value: Any) -> dict[str, Any]:
    _require(
        type(value) is dict and set(value) == _CONTEXT_FIELDS,
        "worker context missing or has unknown fields",
    )
    _require(
        value["status"] == "OBSERVED" and value["errors"] == [],
        "worker context is unverified",
    )
    _require(
        value["sys_profile_is_none"] is True
        and value["threading_profile_is_none"] is True,
        "worker has a preexisting profile; no profile is cleared",
    )
    _require(
        type(value["environment"]) is dict
        and all(
            type(key) is str and (item is None or type(item) is str)
            for key, item in value["environment"].items()
        ),
        "worker startup environment is invalid",
    )
    return value


def _phase_events(helper: Any, path: Path) -> list[dict[str, Any]]:
    return [
        event
        for line in helper._read(path).splitlines()
        for event in (helper._loads(line),)
        if event.get("kind") == "phase"
    ]


def _original(helper: Any, runner: Any, receipts: Path) -> dict[str, Any] | None:
    witness = helper._preflight(receipts)
    if witness is None:
        return None
    folder = witness["folder"]
    request_path, run_path = folder / "request.json", receipts / "run.json"
    paths = [*witness["evidence_paths"], request_path, run_path]
    before = helper._snapshot(paths)
    receipt = helper._json(folder / "receipt.json")
    request, run = helper._json(request_path), helper._json(run_path)
    _require(set(request) == set(_REQUEST_FIELDS), "original request fields differ")
    request_bytes = helper._read(request_path)
    _require(request_bytes == _wire(request), "original request wire format differs")
    _require(
        request["kind"] == "batch"
        and request["assigned"] == receipt["assigned_ids"] == receipt["selected_ids"]
        and request["full_preselection_ids"] == receipt["full_preselection_ids"]
        and request["collection_skips"] == receipt["collection_skips"]
        and request["pytest_args"] == receipt["pytest_args"],
        "original request and worker receipt disagree",
    )
    assigned, full = request["assigned"], request["full_preselection_ids"]
    _require(
        1 <= len(assigned) <= 500
        and len(set(assigned)) == len(assigned)
        and len(set(full)) == len(full)
        and assigned.count(helper.NODE_ID) == full.count(helper.NODE_ID) == 1,
        "original assignment is incomplete or ambiguous",
    )
    _require(
        [item["nodeid"] for item in receipt["items"]] == assigned,
        "original selected item records disagree",
    )
    _require(
        receipt["supplemental"] is False
        and receipt["cast_observer_enabled"] is True
        and receipt["native_source_recomputed"] is False,
        "parent witness is not a canonical observer-enabled run",
    )
    contexts = {
        key: _context(receipt[key])
        for key in (
            "startup_context",
            "collection_context",
            "target_context_before",
            "target_context_after",
        )
    }
    _require(
        all(witness["replay_source_matches_original"].values()),
        "current source differs from the original call",
    )
    startup = contexts["startup_context"]
    observed = helper._json(folder / helper.OBSERVATION)
    _require(
        type(observed["runtime_environment"]) is dict
        and all(
            contexts[phase]["environment"].get(name) == value
            for phase in ("target_context_before", "target_context_after")
            for name, value in observed["runtime_environment"].items()
        ),
        "original observer and worker target environments disagree",
    )
    _require(
        set(startup["environment"])
        == set(
            runner._worker_environment_names(
                {
                    name: value
                    for name, value in startup["environment"].items()
                    if value is not None
                }
            )
        ),
        "original startup environment name set differs",
    )
    _require(
        all(
            all(context[key] == startup[key] for key in _PROCESS_FIELDS)
            for context in contexts.values()
        ),
        "original process/source context changed between phases",
    )
    current = runner._worker_context()
    _require(
        current["status"] == "OBSERVED" and not current["errors"],
        "current controller process provenance is unverified",
    )
    _require(
        all(current[key] == startup[key] for key in _PROCESS_FIELDS),
        "current source, checkout, interpreter or CPU differs from the parent",
    )
    _require(
        run["status"] == "FAIL"
        and run["nodeids"] == full
        and run["checked_out_commit"] == startup["checkout"]["head"]
        and run["interpreter"] == startup["executable"]
        and run["pytest_addopts"] == (startup["environment"]["PYTEST_ADDOPTS"] or ""),
        "original run identity/startup proof differs",
    )
    matching = [
        index for index, batch in enumerate(run["batches"]) if batch == assigned
    ]
    _require(
        len(matching) == 1 and folder.name == f"batch-{matching[0] + 1:04d}",
        "failed assignment is not the original run's exact batch",
    )
    processes = [
        process
        for process in run["processes"]
        if process["kind"] == "batch" and process["folder"] == folder.name
    ]
    _require(
        len(processes) == 1
        and processes[0]["returncode"] == 1
        and type(processes[0]["returncode"]) is int,
        "original failed worker process proof differs",
    )
    target_index = assigned.index(helper.NODE_ID)
    prefix = assigned[:target_index]
    _require(
        list(receipt["phases"]) == assigned[: target_index + 1],
        "original execution did not stop at its first target failure",
    )
    _require(
        all(
            all(
                phase["outcome"] in ("passed", "skipped")
                for phase in receipt["phases"][node]
            )
            for node in prefix
        ),
        "original predecessor history contains another failure",
    )
    environment = dict(os.environ)
    managed = set(runner._worker_environment_names(environment)) | set(
        startup["environment"]
    )
    for name in managed:
        environment.pop(name, None)
    for name, value in startup["environment"].items():
        if value is not None:
            environment[name] = value
    descriptor = {
        "schema": _SCHEMA,
        "nodeid": helper.NODE_ID,
        "original_request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "original_receipt_sha256": before[str(folder / "receipt.json")],
        "original_observer_sha256": before[str(folder / helper.OBSERVATION)],
        "observer_enabled": False,
    }
    _require(
        helper._snapshot(paths) == before,
        "original proof changed during control preflight",
    )
    return {
        "witness": witness,
        "request": request,
        "receipt": receipt,
        "contexts": contexts,
        "paths": paths,
        "before": before,
        "environment": environment,
        "descriptor": descriptor,
        "target_index": target_index,
        "prefix": prefix,
        "phase_events": _phase_events(helper, folder / "events.jsonl"),
    }


def _classify(
    helper: Any,
    runner: Any,
    proof: dict[str, Any],
    folder: Path,
    returncode: int,
    result: dict[str, Any],
) -> None:
    receipt = helper._json(folder / "receipt.json")
    request = helper._json(folder / "request.json")
    assigned = proof["request"]["assigned"]
    _require(
        set(request) == {*_REQUEST_FIELDS, "diagnostic_control"}
        and all(request[key] == proof["request"][key] for key in _REQUEST_FIELDS)
        and request["diagnostic_control"] == proof["descriptor"],
        "control did not preserve the exact original request",
    )
    verification = receipt["diagnostic_control"]
    _require(
        set(verification)
        == {
            *proof["descriptor"],
            "request_binding_verified_by_worker",
            "evidence_sha_scope",
        }
        and all(
            verification[key] == value for key, value in proof["descriptor"].items()
        )
        and verification["request_binding_verified_by_worker"] is True
        and verification["evidence_sha_scope"]
        == "Coordinator-verified references; worker verifies request binding only.",
        "control request binding/provenance scope differs",
    )
    _require(
        receipt["supplemental"] is True
        and receipt["native_source_recomputed"] is True
        and receipt["cast_observer_enabled"] is False,
        "control worker did not prove observer-disabled mode",
    )
    _require(
        not (folder / helper.OBSERVATION).exists()
        and not (folder / helper.MODEL).exists(),
        "disabled observer unexpectedly produced ordinary observer evidence",
    )
    _require(
        receipt["full_collection_checked"] is True
        and receipt["collection_errors"] == []
        and receipt["full_preselection_ids"]
        == proof["request"]["full_preselection_ids"]
        and receipt["collection_skips"] == proof["request"]["collection_skips"]
        and receipt["assigned_ids"] == receipt["selected_ids"] == assigned
        and [item["nodeid"] for item in receipt["items"]] == assigned,
        "control full collection/import ordering or assignment drifted",
    )
    _require(
        helper.NODE_ID in receipt["phases"],
        "control target was not reached after its predecessor history",
    )
    for key, original in proof["contexts"].items():
        _require(
            _context(receipt[key]) == original, "control context differs at " + key
        )
    phases = receipt["phases"]
    executed = list(phases)
    _require(
        executed == assigned[: len(executed)]
        and receipt["completed"] == runner._completed(phases) == executed,
        "control execution is incomplete or out of order",
    )
    events = _phase_events(helper, folder / "events.jsonl")
    expected = [
        {"kind": "phase", "nodeid": node, **phase}
        for node, reports in phases.items()
        for phase in reports
    ]
    _require(events == expected, "control phase events disagree with its receipt")
    prefix = proof["prefix"]
    _require(
        all(
            node in phases and phases[node] == proof["receipt"]["phases"][node]
            for node in prefix
        ),
        "control predecessor outcomes differ",
    )
    _require(
        [event for event in events if event["nodeid"] in prefix]
        == [event for event in proof["phase_events"] if event["nodeid"] in prefix],
        "control predecessor event order differs",
    )
    target = phases.get(helper.NODE_ID, [])
    result["native_source_recomputed"] = any(
        phase["when"] == "call" for phase in target
    )
    _require(
        [phase["when"] for phase in target] == ["setup", "call", "teardown"]
        and target[0]["outcome"] == target[2]["outcome"] == "passed",
        "control target call is missing or incomplete",
    )
    _require(
        type(returncode) is int
        and returncode == receipt["exitstatus"]
        and type(receipt["exitstatus"]) is int
        and returncode in (0, 1)
        and not receipt.get("interrupted_signal"),
        "control worker interrupted or exited unexpectedly",
    )
    failures = [
        (node, phase)
        for node, reports in phases.items()
        for phase in reports
        if phase["outcome"] == "failed"
    ]
    junit = ET.fromstring(helper._read(folder / "junit.xml"))
    cases = list(junit.iter("testcase"))
    matching_cases = [
        case
        for case in cases
        if case.attrib.get("classname")
        == "tests.extra_tests.converter.test_source_conversion_semantics"
        and case.attrib.get("name") == helper.NODE_ID.split("::")[1]
    ]
    _require(
        len(cases) == len(executed) + len(proof["request"]["collection_skips"])
        and len(matching_cases) == 1,
        "control JUnit case accounting differs",
    )
    target_case = matching_cases[0]
    _require(
        target_case.find("error") is None
        and target_case.find("skipped") is None
        and (target_case.find("failure") is not None)
        == (target[1]["outcome"] == "failed"),
        "control target JUnit outcome differs",
    )
    result.update(
        prefix_matched=True,
        observer_disabled_verified=True,
        target_phases=target,
        assigned_count=len(assigned),
        executed_count=len(executed),
        target_index=proof["target_index"],
        suffix_assigned_count=len(assigned) - proof["target_index"] - 1,
        suffix_executed_count=max(0, len(executed) - proof["target_index"] - 1),
        full_batch_success=returncode == 0 and executed == assigned and not failures,
        control_receipt_sha256=hashlib.sha256(
            helper._read(folder / "receipt.json")
        ).hexdigest(),
    )
    if target[1]["outcome"] == "failed":
        _require(
            returncode == 1
            and len(failures) == 1
            and failures[0][0] == helper.NODE_ID
            and target[1] == proof["receipt"]["phases"][helper.NODE_ID][1]
            and executed == assigned[: proof["target_index"] + 1],
            "control failure is not the same first bitwise failure",
        )
        result.update(
            status="OBSERVER_NOT_NECESSARY",
            conclusion="The active observer is not necessary for this reproduced original bitwise failure; no raw native words were observed in this control.",
        )
    else:
        _require(target[1]["outcome"] == "passed", "control target was skipped")
        _require(
            (returncode == 0 and executed == assigned and not failures)
            or (
                returncode == 1
                and len(failures) == 1
                and assigned.index(failures[0][0]) > proof["target_index"]
                and failures[0][0] == executed[-1]
            ),
            "control suffix outcome does not match ordinary fail-fast execution",
        )
        result.update(
            status="TARGET_PASSED_ASSOCIATION_ONLY",
            suffix_failures=[{"nodeid": node, **phase} for node, phase in failures],
            conclusion="The unprofiled target passed. Association alone does not establish observer causality; process/history nondeterminism remains possible. Full-batch success is recorded separately.",
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=240)
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout_seconds) or not 0 < args.timeout_seconds <= 240:
        parser.error("diagnostic timeout must be positive and at most 240 seconds")
    root = args.output_dir or args.receipts_dir / "diagnostics" / _SCHEMA
    try:
        resolved, receipts = root.resolve(), args.receipts_dir.resolve()
        _require(
            not root.exists()
            and not root.is_symlink()
            and not resolved.is_relative_to(_ROOT),
            "existing/colliding diagnostic output directory",
        )
        if resolved.is_relative_to(receipts):
            _require(
                resolved.relative_to(receipts).parts[:1] == ("diagnostics",),
                "diagnostic output may not enter the canonical batch namespace",
            )
        root.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError) as error:
        print("UNVERIFIED: " + str(error), file=sys.stderr)
        return 1
    helper = _load_script("replay_source_cast_failure")
    runner = _load_script("run_pytest_batches")
    result: dict[str, Any] = {
        "status": "UNVERIFIED",
        "schema": _SCHEMA,
        "scope": "Supplemental fresh-child observer-disabled control; the original canonical failure remains authoritative.",
        "native_source_recomputed": False,
        "native_source_recomputation_requested": True,
        "raw_native_words_captured": False,
        "numerical_repair_claimed": False,
        "child_invocations": 0,
        "processes": [],
    }
    proof = None
    folder = None
    sources_before = None
    source_paths = [
        *(_ROOT / name for name in helper._SOURCE_FILES),
        Path(__file__).resolve(),
        Path(helper.__file__).resolve(),
        _ROOT / ".github/workflows/ci.yml",
    ]
    try:
        proof = _original(helper, runner, args.receipts_dir)
        if proof is None:
            result.update(
                status="NOT_APPLICABLE",
                reason="No current original exact failed cast witness.",
            )
            helper._write(root / "run.json", result)
            return 0
        sources_before = helper._snapshot(source_paths)
        result.update(
            original_batch=proof["witness"]["folder"].name,
            original_proof_hashes=proof["before"],
            original_contexts=proof["contexts"],
            descriptor=proof["descriptor"],
            original_source_before=sources_before,
            assigned_count=len(proof["request"]["assigned"]),
            full_collection_count=len(proof["request"]["full_preselection_ids"]),
            target_index=proof["target_index"],
            child_timeout_seconds=args.timeout_seconds,
        )
        request = {**proof["request"], "diagnostic_control": proof["descriptor"]}
        folder = root / proof["witness"]["folder"].name
        result["child_invocations"] = 1
        helper._write(root / "run.json", result)
        returncode = runner._child(
            folder,
            request,
            result,
            root,
            environment=proof["environment"],
            timeout_seconds=args.timeout_seconds,
        )
        result["worker_returncode"] = returncode
        _classify(helper, runner, proof, folder, returncode, result)
    except BaseException as error:
        result.update(
            status="UNVERIFIED",
            error_type=type(error).__module__ + "." + type(error).__qualname__,
            reason=str(error),
        )
    finally:
        if proof is not None:
            try:
                if folder is not None and (folder / "receipt.json").is_file():
                    partial = helper._json(folder / "receipt.json")
                    result["native_source_recomputed"] = (
                        partial.get("native_source_recomputed") is True
                    )
                    result["control_contexts"] = {
                        key: partial[key] for key in proof["contexts"] if key in partial
                    }
                result["original_evidence_fence_held"] = (
                    helper._snapshot(proof["paths"]) == proof["before"]
                )
                result["source_fence_held"] = (
                    sources_before is not None
                    and helper._snapshot(source_paths) == sources_before
                )
                if (
                    not result["original_evidence_fence_held"]
                    or not result["source_fence_held"]
                ):
                    result.update(
                        status="UNVERIFIED",
                        reason="Original evidence/source changed during the control.",
                    )
            except Exception as error:
                result.update(
                    status="UNVERIFIED", fence_error_type=type(error).__name__
                )
        helper._write(root / "run.json", result)
    return (
        0
        if result["status"]
        in ("OBSERVER_NOT_NECESSARY", "TARGET_PASSED_ASSOCIATION_ONLY")
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
