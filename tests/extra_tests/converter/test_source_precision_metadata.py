# tests/extra_tests/converter/test_source_precision_metadata.py

"""Bounded real SOURCE metadata execution and complete, non-acceptance accounting."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
import subprocess
import sys
import types
from importlib import import_module
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import onnxruntime
import pytest

from jax2onnx import user_interface
from jax2onnx._compat import jax as compat
from jax2onnx.precision import CheckStatus
from jax2onnx.plugins.plugin_system import construct_and_call
from scripts import source_precision_inventory as inventory_owner
from scripts import source_precision_probe as probe
from tests import t_generator


def _original_scope():
    payload = probe.DEFAULT_BASELINE.read_bytes()
    inventory = json.loads(payload)
    plan = probe.build_probe_plan(
        inventory, baseline_sha256=hashlib.sha256(payload).hexdigest()
    )
    registrations, _ = inventory_owner.raw_registrations()
    adapters = json.loads(
        (
            probe.REPOSITORY_ROOT
            / "internal_docs/precision/source_replay_adapters.json"
        ).read_text()
    )
    return inventory, plan, registrations, adapters


# Only two exact caller cells of one real row enter ordinary pytest collection in this
# first cut. Registry discovery supplies metadata; descriptors/function creation
# do not instantiate, trace, export or execute it. All other raw cells remain in
# the unchanged complete plan and accounting report.
_ORIGINAL = _original_scope()
_EPOCH = t_generator.source_metadata_epoch()
_SELECTED = (
    ("plugin::jax.numpy.add", 0, "raw-x64-off"),
    ("plugin::jax.numpy.add", 0, "raw-x64-on"),
)
_DESCRIPTORS = t_generator.source_metadata_cases(
    _ORIGINAL[1],
    _ORIGINAL[0],
    _ORIGINAL[2],
    selection=_SELECTED,
    epoch=_EPOCH,
    adapters=_ORIGINAL[3],
)


class TestRegisteredSourceMetadata:
    """Real generated entrypoints assert export evidence; numerics stay pending."""


for _descriptor in _DESCRIPTORS:
    _method = t_generator.make_source_test_function(
        _descriptor,
        _ORIGINAL[1],
        _ORIGINAL[2],
        adapters=_ORIGINAL[3],
        evidence_directory=None,
        stage="product",
        expected_statuses={
            "source_reference": "validated",
            "schema": "validated",
            "artifact": "validated",
            "lowering_audit": "validated",
            "structure": "validated",
            "raw_callback": "validated",
            "runtime": "unverified",
            "numerics": "unverified",
        },
    )
    _method.__module__ = __name__
    setattr(TestRegisteredSourceMetadata, _descriptor.selector, _method)
del _descriptor, _method


def freeze_source_metadata_acceptance(state):
    """Freeze the full current denominator before any selected-case admission."""
    payload = inventory_owner.CURRENT_BASELINE.read_bytes()
    inventory = json.loads(payload)
    plan = probe.build_probe_plan(
        inventory, baseline_sha256=hashlib.sha256(payload).hexdigest()
    )
    epoch = t_generator.source_metadata_epoch()
    observation_error = None
    try:
        observed = inventory_owner.collect_inventory()
    except Exception as exc:
        observed = None
        observation_error = f"{type(exc).__name__}: {exc}"
    try:
        ledger = inventory_owner.reconcile_source_metadata(
            inventory,
            observed,
            plan,
            baseline_sha256=plan.baseline_sha256,
            epoch=asdict(epoch),
            planned=(),
            retain_inventory_errors=True,
            unavailable=inventory_owner.unavailable_registrations(),
        )
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        observation_error = (
            f"Unusable genuine inventory observation: {type(exc).__name__}: {exc}"
        )
        observed = None
        ledger = inventory_owner.reconcile_source_metadata(
            inventory,
            None,
            plan,
            baseline_sha256=plan.baseline_sha256,
            epoch=asdict(epoch),
            planned=(),
            retain_inventory_errors=True,
            unavailable=inventory_owner.unavailable_registrations(),
        )
    if observation_error:
        ledger["inventory_errors"].append(observation_error)
    ledger["inventory_observation_phase"] = "before-selection-admission"
    state.update(
        inventory=inventory,
        observed_inventory=observed,
        plan=plan,
        epoch=epoch,
        frozen_ledger=ledger,
        baseline_payload=payload,
        baseline_sha256=plan.baseline_sha256,
        descriptors=(),
        identities=(),
        input_bindings={
            str(inventory_owner.CURRENT_BASELINE.resolve()): plan.baseline_sha256
        },
    )


def configure_source_metadata_acceptance(config, owner, state):
    """Install only explicit raw cells in this existing pytest collection owner."""
    from tests.extra_tests.converter.source_precision_raw_replay import (
        _conditional_discrete_criteria_choice,
    )

    if state["configured"]:
        return
    evidence_directory = state["report"].parent / (state["report"].name + ".evidence")
    if evidence_directory.exists():
        raise ValueError("SOURCE acceptance evidence directory already exists")
    baseline = Path(
        config.getoption("source_metadata_baseline") or inventory_owner.CURRENT_BASELINE
    )
    payload = baseline.read_bytes()
    if payload != state["baseline_payload"]:
        raise ValueError(
            "SOURCE acceptance baseline must have exact CURRENT_BASELINE bytes"
        )
    state["input_bindings"][str(baseline.resolve())] = hashlib.sha256(
        payload
    ).hexdigest()
    selection = []
    for choice in config.getoption("source_metadata_case"):
        try:
            origin, ordinal, scenario = choice.rsplit("::", 2)
            identity = (origin, int(ordinal), scenario)
        except (ValueError, TypeError) as exc:
            raise pytest.UsageError(
                "SOURCE case must be origin::ordinal::scenario"
            ) from exc
        if (
            not origin
            or str(identity[1]) != ordinal
            or scenario not in {"raw-x64-off", "raw-x64-on"}
        ):
            raise pytest.UsageError(
                "SOURCE case needs an exact raw identity and caller context"
            )
        selection.append(identity)
    registrations, _ = inventory_owner.raw_registrations()
    if selection:
        inventory_owner.source_metadata_selection_preflight(
            state["inventory"],
            state["observed_inventory"],
            state["plan"],
            registrations,
            selection,
        )
    adapters = {}
    if selection:
        adapter_path = Path(
            config.getoption("source_metadata_adapters")
            or probe.REPOSITORY_ROOT
            / "internal_docs/precision/source_replay_adapters.json"
        )
        adapter_payload = adapter_path.read_bytes()
        adapters = json.loads(adapter_payload)
        state["input_bindings"][str(adapter_path.resolve())] = hashlib.sha256(
            adapter_payload
        ).hexdigest()
    descriptors = (
        t_generator.source_metadata_cases(
            state["plan"],
            state["inventory"],
            registrations,
            selection=selection,
            epoch=state["epoch"],
            adapters=adapters,
        )
        if selection
        else ()
    )
    for name, value in list(vars(owner).items()):
        if getattr(value, "_source_metadata_case", None) is not None:
            delattr(owner, name)
    for descriptor in descriptors:
        method = t_generator.make_source_test_function(
            descriptor,
            state["plan"],
            registrations,
            adapters=adapters,
            evidence_directory=evidence_directory,
            stage="product",
            runtime_optimization=(
                "both"
                if _conditional_discrete_criteria_choice(
                    json.loads(descriptor.adapter_json)
                )
                else "all"
            ),
            expected_statuses={
                "source_reference": "validated",
                "schema": "validated",
                "artifact": "validated",
                "lowering_audit": "validated",
                "structure": "validated",
                "raw_callback": "validated",
            },
        )
        method.__module__ = __name__
        setattr(owner, descriptor.selector, method)
    state.update(configured=True, descriptors=descriptors, identities=tuple(selection))


def finish_source_metadata_acceptance(session, state, exitstatus):
    """Keep the frozen ledger even when actual admission/collection is invalid."""
    collected, outcomes, evidence, report = (), [], [], None
    try:
        if not state["configured"]:
            raise ValueError(
                state.get(
                    "admission_error", "Metadata owner admission did not complete"
                )
            )
        collected = t_generator.source_metadata_collection_records(state["items"])
        for item in state["items"]:
            outcome = getattr(item.obj, "_source_outcome", None)
            cell = getattr(item.obj, "_source_cell", None)
            if outcome is not None:
                outcomes.append({**outcome, "nodeid": item.nodeid})
                evidence.append(cell.history[-1])
        if t_generator.source_metadata_epoch() != state["epoch"]:
            raise ValueError("SOURCE acceptance source/environment epoch changed")
        if any(
            hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest
            for path, digest in state["input_bindings"].items()
        ):
            raise ValueError("SOURCE acceptance baseline/adapter input bytes changed")
        terminal = inventory_owner.source_metadata_pytest_statuses(
            collected, state["reports"]
        )
        expected = set(state["identities"])
        actual = {tuple(record["identity"]) for record in collected}
        collection_ok = (
            state["collection_finished"]
            and all(report.passed for report in state["collection_reports"])
            and expected == actual
            and len(collected) == len(expected)
        )
        run_finished = exitstatus in {
            pytest.ExitCode.OK,
            pytest.ExitCode.TESTS_FAILED,
            pytest.ExitCode.NO_TESTS_COLLECTED,
        }
        outcomes_ok = (
            collection_ok
            and run_finished
            and all(
                terminal.get(record["nodeid"], {}).get("terminal", False)
                for record in collected
            )
        )
        report = inventory_owner.reconcile_source_metadata(
            state["inventory"],
            inventory_owner.collect_inventory(),
            state["plan"],
            baseline_sha256=state["baseline_sha256"],
            epoch=asdict(state["epoch"]),
            planned=[descriptor.record() for descriptor in state["descriptors"]],
            collected=collected,
            outcomes=outcomes,
            stage_evidence=evidence,
            pytest_reports=state["reports"],
            collection_complete=collection_ok,
            outcomes_complete=outcomes_ok,
            unavailable=inventory_owner.unavailable_registrations(),
            retain_inventory_errors=True,
        )
        initial_errors = state["frozen_ledger"]["inventory_errors"]
        report["inventory_errors"] = list(
            dict.fromkeys([*initial_errors, *report["inventory_errors"]])
        )
        report["initial_observed_inventory_sha256"] = state["frozen_ledger"][
            "inventory_mismatch_evidence"
        ]["observed_snapshot_sha256"]
        if report["inventory_errors"]:
            report["complete_migration_acceptance"] = "error"
            report["required_exit_status"] = 1
        if not collection_ok:
            raise ValueError(
                "Actual SOURCE collection failed or omitted selected identities"
            )
    except Exception as exc:
        report = copy.deepcopy(report if report is not None else state["frozen_ledger"])
        report["complete_migration_acceptance"] = "error"
        report["required_exit_status"] = 1
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["unreconciled_actual_collection"] = list(collected)
        report["unreconciled_stage_outcomes"] = outcomes
    report["pytest_collection_owner"] = state["collection_owner"]
    report["requested_pytest_args"] = state["requested_pytest_args"]
    report["requested_case_choices"] = list(
        session.config.getoption("source_metadata_case")
    )
    report["actual_collection_reports"] = [
        {
            "nodeid": item.nodeid,
            "outcome": item.outcome,
            "longrepr": str(item.longrepr) if item.failed else None,
        }
        for item in state["collection_reports"]
    ]
    report["actual_pytest_reports"] = [
        {"nodeid": item.nodeid, "when": item.when, "outcome": item.outcome}
        for item in state["reports"]
    ]
    report["original_pytest_exitstatus"] = int(exitstatus)
    state["report"].parent.mkdir(parents=True, exist_ok=True)
    with state["report"].open("x") as stream:
        json.dump(report, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
    if report["required_exit_status"] and exitstatus in {
        pytest.ExitCode.OK,
        pytest.ExitCode.NO_TESTS_COLLECTED,
    }:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


@pytest.fixture(scope="module")
def original_scope():
    return _ORIGINAL


def _synthetic(case, *, aliases=("plugin::synthetic",), without=(), manual=()):
    registrations = [
        (
            origin,
            "synthetic.owner",
            {
                "context": "primitives.synthetic",
                "component": "same component",
                "testcases": [case],
            },
        )
        for origin in aliases
    ]
    inventory = inventory_owner.build_inventory(
        registrations,
        without,
        manual=manual,
        optional={},
    )
    return inventory, probe.build_probe_plan(inventory), registrations


def _descriptors(scope, *, selection=None):
    inventory, plan, registrations = scope[:3]
    return t_generator.source_metadata_cases(
        plan,
        inventory,
        registrations,
        selection=selection or [cell.identity for cell in plan.cells],
        epoch=_EPOCH,
        adapters=scope[3] if len(scope) > 3 else {},
    )


def _function(
    scope, descriptor, tmp_path, *, expected=None, runtime=None, stage="product"
):
    return t_generator.make_source_test_function(
        descriptor,
        scope[1],
        scope[2],
        adapters=scope[3] if len(scope) > 3 else {},
        evidence_directory=tmp_path,
        stage=stage,
        expected_statuses=expected or {"source_reference": "validated"},
        runtime_optimization=runtime,
    )


def _protocol_scope(scope):
    row = scope[1].rows[0]
    adapters = {
        "inventory_sha256": scope[1].baseline_sha256,
        "rows": {
            f"{row.origin}::{row.ordinal}": {
                "contract_sha256": row.contract_sha256,
                "native_protocol": {
                    "mode": "stateless-three-role-v1",
                    "reviewed": True,
                },
            }
        },
    }
    return (*scope[:3], adapters)


def _retained_stage_record(function):
    record = function._source_cell.history[-1]
    capture = record.replay_capture
    anchor = capture.get("artifact", capture.get("eager"))
    path = Path(anchor["path"]).parent / "stage-evidence.json"
    retained = json.loads(path.read_bytes())
    canonical = json.dumps(retained, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical).hexdigest() == record.sha256
    assert retained == json.loads(json.dumps(asdict(record)))
    assert "stage-evidence" not in retained["replay_capture"]
    return retained


def _collect_record(descriptor):
    return {
        **descriptor.record(),
        "nodeid": f"tests/example.py::TestSource::{descriptor.selector}",
    }


def _report_fixture(nodeid, *, setup="passed", call="passed", teardown="passed"):
    stages = [("setup", setup), ("teardown", teardown)]
    if call is not None:
        stages.insert(1, ("call", call))
    return [
        pytest.TestReport(
            nodeid=nodeid,
            location=("tests/example.py", 0, "TestSource"),
            keywords={},
            outcome=status,
            longrepr=None,
            when=when,
        )
        for when, status in stages
    ]


def _account(scope, descriptors, **kwargs):
    inventory, plan, _ = scope[:3]
    return inventory_owner.reconcile_source_metadata(
        inventory,
        kwargs.pop("observed_inventory", inventory),
        plan,
        baseline_sha256=plan.baseline_sha256,
        epoch=asdict(_EPOCH),
        planned=[item.record() for item in descriptors],
        **kwargs,
    )


def _forbid_legacy_and_native(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail(
            "SOURCE metadata collection must not invoke legacy or native execution"
        )

    monkeypatch.setattr(t_generator, "generate_test_params", forbidden)
    monkeypatch.setattr(t_generator, "allclose", forbidden)
    monkeypatch.setattr(t_generator, "to_onnx", forbidden)
    monkeypatch.setattr(type(construct_and_call(lambda: None)), "with_dtype", forbidden)
    monkeypatch.setattr(
        type(construct_and_call(lambda: None)), "instantiate", forbidden
    )
    monkeypatch.setattr(jax, "make_jaxpr", forbidden)
    monkeypatch.setattr(jax, "jit", forbidden)
    monkeypatch.setattr(user_interface, "_to_onnx", forbidden)
    monkeypatch.setattr(onnxruntime, "InferenceSession", forbidden)


def test_collection_keeps_original_recipe_bytes_and_does_not_execute(
    monkeypatch, tmp_path
):
    def constructor(*, dtype):
        pytest.fail("Collection invoked the original constructor")

    recipe = construct_and_call(constructor, dtype=np.float32)
    values = np.asarray([-0.0, np.nan], dtype=np.float32)
    scope = _synthetic(
        {
            "testcase": "same.name",
            "callable": recipe,
            "input_values": [values],
            "input_shapes": [(2,)],
        }
    )
    snapshot = values.tobytes()
    _forbid_legacy_and_native(monkeypatch)
    descriptors = _descriptors(scope)
    functions = [_function(scope, descriptor, tmp_path) for descriptor in descriptors]
    assert len(functions) == 2
    assert values.tobytes() == snapshot
    assert scope[2][0][2]["testcases"][0]["callable"] is recipe
    assert all(not hasattr(function, "_source_outcome") for function in functions)
    assert scope[1].rows[0].criteria is None


def test_sanitization_collisions_and_aliases_cannot_overwrite_methods():
    first = {"testcase": "same.name", "callable": lambda x: x, "input_shapes": [(1,)]}
    scope = _synthetic(first, aliases=("plugin::a.b", "plugin::a_b"))
    descriptors = _descriptors(scope)
    assert len({item.selector for item in descriptors}) == 4
    namespace = {item.selector: item for item in descriptors}
    assert len(namespace) == 4
    assert {item.identity[0] for item in namespace.values()} == {
        "plugin::a.b",
        "plugin::a_b",
    }
    with pytest.raises(ValueError, match="Duplicate"):
        _descriptors(scope, selection=[descriptors[0].identity] * 2)
    with pytest.raises(ValueError, match="explicit bounded"):
        t_generator.source_metadata_cases(
            scope[1], scope[0], scope[2], selection=[], epoch=_EPOCH, adapters={}
        )


def test_actual_collected_ids_are_read_from_pytest_items(request, original_scope):
    records = t_generator.source_metadata_collection_records(request.session.items)
    selected = [record for record in records if record["identity"] in _SELECTED]
    # These are actual pytest items, not manually planned IDs. A narrower -k or
    # node selection may intentionally exclude the bounded generated methods.
    actual_ids = {item.nodeid for item in request.session.items}
    assert all(record["nodeid"] in actual_ids for record in selected)
    report = _account(original_scope, _DESCRIPTORS, collected=selected)
    assert report["counts"]["planned_source_selectors"] == 2
    assert report["counts"]["actual_source_collected_pytest_ids"] == len(selected)
    assert report["counts"]["terminal_source_stage_outcomes"] == 0
    assert report["counts"]["validated_raw_metadata_capabilities"] == 0


def test_acceptance_keeps_complete_current_denominator_with_two_selected_cells():
    payload = inventory_owner.CURRENT_BASELINE.read_bytes()
    inventory = json.loads(payload)
    scope = (
        inventory,
        probe.build_probe_plan(
            inventory, baseline_sha256=hashlib.sha256(payload).hexdigest()
        ),
        _ORIGINAL[2],
    )
    descriptors = _descriptors(scope, selection=_SELECTED)
    report = _account(scope, descriptors, collection_complete=False)
    assert report["counts"]["raw_rows"] == 1856
    assert report["counts"]["raw_caller_cells"] == 3712
    assert report["counts"]["criterion_unresolved_raw_rows"] == 1838
    assert report["counts"]["planned_source_selectors"] == 2
    raw = [item for item in report["obligations"] if item["kind"] == "raw-cell"]
    assert len(raw) == 3712
    assert all(item["numerical_acceptance"] == "unverified" for item in raw)
    assert {item["kind"] for item in report["obligations"]} >= {
        "raw-cell",
        "metadata-free",
        "original-manual",
        "legacy-preparation",
    }
    assert report["raw_metadata_acceptance"] == "unverified"
    assert report["complete_migration_acceptance"] == "unverified"
    assert report["required_exit_status"] == 1


def test_criteria_pass_does_not_invent_missing_scientific_checks():
    scope = _synthetic(
        {
            "testcase": "criterion",
            "callable": lambda x: x,
            "input_shapes": [(1,)],
            "rtol": 0.0,
            "atol": 0.0,
        }
    )
    evidence = {
        "source_reference": "validated",
        "numerics": "validated",
        "detail": "criteria passed",
        "artifact_sha256": "a" * 64,
        "replay_capture": {
            "metadata_runtime": {
                "optimization": "all",
                "status": "validated",
                "binding_status": "validated",
                "artifact_sha256": "a" * 64,
                "target": {"optimization": "ORT_ENABLE_ALL"},
                "samples": [
                    {
                        "references": [{"dtype": "float32", "shape": [1]}],
                        "outputs": [{"dtype": "float32", "shape": [1]}],
                        "checks": [
                            {"category": "runtime-execution", "status": "validated"}
                        ],
                    }
                ],
            }
        },
    }
    checks = inventory_owner.source_metadata_numerical_checks(
        scope[1].rows[0], evidence
    )
    assert checks["declared_criteria_pair"]["status"] == "validated"
    assert checks["reviewed_criteria"]["status"] == "unverified"
    assert checks["criteria_comparison"]["status"] == "validated"
    assert checks["dtype_shape_output_count"]["status"] == "validated"
    assert checks["ort_all"]["status"] == "validated"
    for name in (
        "native_three_role_agreement",
        "ort_disabled",
        "same_artifact_and_feeds_both_modes",
        "component_nan_masks",
        "zero_signs",
        "operation_rounding",
        "required_internal_fp64",
    ):
        assert checks[name]["status"] == "unverified"
    crossed = copy.deepcopy(evidence)
    crossed["replay_capture"]["metadata_runtime"]["artifact_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="artifact binding"):
        inventory_owner.source_metadata_numerical_checks(scope[1].rows[0], crossed)
    absent = copy.deepcopy(evidence)
    absent["replay_capture"]["metadata_runtime"]["samples"][0]["checks"] = []
    assert (
        inventory_owner.source_metadata_numerical_checks(scope[1].rows[0], absent)[
            "dtype_shape_output_count"
        ]["status"]
        == "unverified"
    )


def _acceptance_subprocess(tmp_path, *, case=None, phase=None, admission=None):
    """A real pytest subprocess, never a direct-call collection/outcome claim."""
    root = Path(__file__).resolve().parents[3]
    report = tmp_path / "acceptance.json"
    forbidden = tmp_path / "test_must_not_run.py"
    forbidden.write_text("raise AssertionError('non-owner model module collected')\n")
    args = [
        sys.executable,
        "-B",
        "-m",
        "pytest",
        "-q",
        str(Path(__file__).resolve()),
        str(forbidden),
        "--source-metadata-acceptance=" + str(report),
        "-o",
        "cache_dir=" + str(tmp_path / "pytest-cache"),
        "--basetemp=" + str(tmp_path / "pytest"),
    ]
    if case is not None:
        args.append("--source-metadata-case=" + case)
        if admission == "historical-adapters":
            adapter_path = root / "internal_docs/precision/source_replay_adapters.json"
        else:
            adapter_path = tmp_path / "adapters.json"
            adapter_path.write_text(
                json.dumps(
                    {
                        "inventory_sha256": hashlib.sha256(
                            inventory_owner.CURRENT_BASELINE.read_bytes()
                        ).hexdigest(),
                        "rows": {},
                    }
                )
            )
        args.append("--source-metadata-adapters=" + str(adapter_path))
    if admission == "historical-baseline":
        args.append(
            "--source-metadata-baseline=" + str(inventory_owner.HISTORICAL_BASELINE)
        )
    elif admission == "existing-evidence":
        (tmp_path / "acceptance.json.evidence").mkdir()
    if phase == "deselected":
        args += ["-k", "no_such_selected_source_case"]
    elif phase == "collection":
        plugin = tmp_path / "metadata_phase_failure.py"
        plugin.write_text(
            "def pytest_pycollect_makeitem(collector, name, obj):\n"
            "    if name == 'TestRegisteredSourceMetadata':\n"
            "        raise RuntimeError('genuine selected collection failure')\n"
        )
        args += ["-p", "metadata_phase_failure"]
    elif phase is not None:
        plugin = tmp_path / "metadata_phase_failure.py"
        plugin.write_text(
            "def pytest_runtest_" + phase + "(item):\n"
            "    if getattr(getattr(item, 'obj', None), '_source_metadata_case', None) is not None:\n"
            "        raise RuntimeError('genuine selected " + phase + " failure')\n"
        )
        args += ["-p", "metadata_phase_failure"]
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join((str(tmp_path), str(root))),
    }
    result = subprocess.run(
        args, cwd=root, env=env, capture_output=True, text=True, timeout=120
    )
    (tmp_path / "pytest-output.txt").write_text(result.stdout + result.stderr)
    return result, json.loads(report.read_text())


def test_optin_no_explicit_cases_accounts_only_and_executes_no_items(tmp_path):
    result, report = _acceptance_subprocess(tmp_path)
    assert result.returncode == 1
    assert "non-owner model module collected" not in result.stdout + result.stderr
    assert Path(report["pytest_collection_owner"]).resolve() == Path(__file__).resolve()
    assert report["counts"]["raw_caller_cells"] == 3712
    assert report["counts"]["planned_source_selectors"] == 0
    assert report["counts"]["actual_source_collected_pytest_ids"] == 0
    assert report["counts"]["observed_source_stage_bodies"] == 0
    assert report["raw_metadata_acceptance"] == "unverified"
    assert report["complete_migration_acceptance"] == (
        "error" if report["inventory_errors"] else "unverified"
    )
    assert report["original_pytest_exitstatus"] == 5
    assert not (tmp_path / "acceptance.json.evidence").exists()


@pytest.mark.parametrize("phase", [None, "setup", "teardown"])
def test_optin_real_pytest_phases_cannot_become_complete_acceptance(tmp_path, phase):
    result, report = _acceptance_subprocess(
        tmp_path, case="plugin::jax.numpy.add::0::raw-x64-off", phase=phase
    )
    assert result.returncode == 1
    assert report["counts"]["raw_caller_cells"] == 3712
    assert report["counts"]["actual_source_collected_pytest_ids"] == 1
    assert report["counts"]["actual_source_pytest_terminal_outcomes"] == 1
    raw = next(
        item
        for item in report["obligations"]
        if item.get("identity") == ["plugin::jax.numpy.add", 0, "raw-x64-off"]
    )
    if phase is None:
        assert raw["pytest_outcome"] == "passed"
        assert raw["stage_contract_passed"] is True
        assert raw["numerical_acceptance"] == "unverified"
    else:
        assert raw["pytest_outcome"] == "failed"
        assert raw["numerical_acceptance"] == "failed"
    assert report["complete_migration_acceptance"] != "validated"


def test_optin_historical_adapter_error_keeps_the_full_current_ledger(tmp_path):
    result, report = _acceptance_subprocess(
        tmp_path,
        case="plugin::jax.numpy.add::0::raw-x64-off",
        admission="historical-adapters",
    )
    assert result.returncode != 0
    assert report["complete_migration_acceptance"] == "error"
    assert report["counts"]["raw_caller_cells"] == 3712
    assert (
        len([item for item in report["obligations"] if item["kind"] == "raw-cell"])
        == 3712
    )
    assert {item["kind"] for item in report["obligations"]} >= {
        "original-manual",
        "legacy-preparation",
        "metadata-free",
    }
    assert report["original_pytest_exitstatus"] == result.returncode


@pytest.mark.parametrize("admission", ["historical-baseline", "existing-evidence"])
def test_optin_admission_requires_current_bytes_and_fresh_evidence(tmp_path, admission):
    options = {"source_metadata_baseline": None, "source_metadata_case": []}
    if admission == "historical-baseline":
        options["source_metadata_baseline"] = str(inventory_owner.HISTORICAL_BASELINE)
    else:
        (tmp_path / "acceptance.json.evidence").mkdir()
    state = {
        "configured": False,
        "frozen_ledger": {"inventory_errors": []},
        "report": tmp_path / "acceptance.json",
        "input_bindings": {},
        "baseline_payload": inventory_owner.CURRENT_BASELINE.read_bytes(),
    }
    config = types.SimpleNamespace(getoption=lambda name: options.get(name))
    with pytest.raises(ValueError, match="CURRENT_BASELINE|evidence directory"):
        configure_source_metadata_acceptance(config, type("EmptyOwner", (), {}), state)
    assert state["configured"] is False


def test_error_ledger_retains_denominator_without_a_fabricated_observation():
    scope = _synthetic(
        {"testcase": "retained", "callable": lambda x: x, "input_shapes": [(1,)]},
        without=[("no_metadata", "owner")],
        manual=[("tests/original.py", ["test_original"])],
    )
    drifted = copy.deepcopy(scope[0])
    drifted["cases"] = []
    drifted["manual"] = []
    for observed in (None, drifted):
        report = _account(
            scope, [], observed_inventory=observed, retain_inventory_errors=True
        )
        assert report["counts"]["raw_caller_cells"] == 2
        assert report["counts"]["original_manual_anchors"] == 1
        assert report["inventory_errors"]
        assert report["current_inventory_observed"] is (observed is not None)
        assert report["required_exit_status"] == 1
        assert (
            len([item for item in report["obligations"] if item["kind"] == "raw-cell"])
            == 2
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "owner",
        "context",
        "component",
        "feature",
        "opset",
        "contract",
        "live-owner",
        "live-contract",
        "unobserved",
        "malformed",
        "header",
    ],
)
def test_selected_diagnostic_preflight_rejects_drift_before_native(
    mutation, monkeypatch
):
    def factory():
        pytest.fail("Selected preflight invoked the original factory")

    scope = _synthetic(
        {
            "testcase": "bound",
            "callable": construct_and_call(factory),
            "input_shapes": [(1,)],
        }
    )
    inventory, plan, registrations = scope
    observed = copy.deepcopy(inventory)
    registrations = [
        (
            origin,
            owner,
            {**metadata, "testcases": [dict(case) for case in metadata["testcases"]]},
        )
        for origin, owner, metadata in registrations
    ]
    if mutation in {"owner", "context", "component"}:
        row = list(observed["registrations"][0])
        row[{"owner": 1, "context": 2, "component": 3}[mutation]] = "changed"
        observed["registrations"][0] = row
    elif mutation in {"feature", "opset"}:
        row = list(observed["cases"][0])
        row[3 if mutation == "feature" else 4] += 1
        observed["cases"][0] = row
    elif mutation == "contract":
        observed["contracts"][0]["input_shapes"] = [{"tuple": [2]}]
    elif mutation == "live-owner":
        origin, _, metadata = registrations[0]
        registrations[0] = (origin, "changed", metadata)
    elif mutation == "live-contract":
        registrations[0][2]["testcases"][0]["input_shapes"] = [(2,)]
    elif mutation == "unobserved":
        observed = None
    elif mutation == "header":
        observed["schema_version"] = 2
    else:
        observed = {"registrations": [], "cases": []}
    _forbid_legacy_and_native(monkeypatch)
    with pytest.raises(ValueError):
        inventory_owner.source_metadata_selection_preflight(
            inventory, observed, plan, registrations, [plan.cells[0].identity]
        )


def test_unrelated_drift_keeps_global_error_but_not_selected_diagnostic_block(
    monkeypatch,
):
    scope = _synthetic(
        {"testcase": "bound", "callable": lambda x: x, "input_shapes": [(1,)]},
        aliases=("plugin::selected", "plugin::unrelated"),
    )
    inventory, plan, registrations = scope
    observed = copy.deepcopy(inventory)
    changed = list(observed["cases"][1])
    observed["contracts"].append(
        {**observed["contracts"][changed[5]], "input_shapes": [{"tuple": [2]}]}
    )
    changed[5] = len(observed["contracts"]) - 1
    observed["cases"][1] = changed
    identity = ("plugin::selected", 0, "raw-x64-off")
    _forbid_legacy_and_native(monkeypatch)
    inventory_owner.source_metadata_selection_preflight(
        inventory, observed, plan, registrations, [identity]
    )
    report = _account(
        scope,
        _descriptors(scope, selection=[identity]),
        observed_inventory=observed,
        retain_inventory_errors=True,
    )
    assert report["inventory_errors"]
    assert report["complete_migration_acceptance"] == "error"
    assert report["counts"]["raw_caller_cells"] == 4
    mismatch = report["inventory_mismatch_evidence"]
    assert mismatch["observed_snapshot_sha256"] == t_generator._source_metadata_digest(
        observed
    )
    row = mismatch["row_mismatches"][0]
    assert row["identity"] == ("plugin::unrelated", 0)
    assert row["expected"]["contract"] == inventory["contracts"][0]
    assert row["observed"]["contract"] == observed["contracts"][-1]


def test_selected_optional_source_drift_stops_before_native(monkeypatch):
    scope = _synthetic(
        {"testcase": "bound", "callable": lambda x: x, "input_shapes": [(1,)]}
    )
    inventory, plan, registrations = scope
    identity = plan.cells[0].identity
    inventory["optional"] = {
        "maxtext": {
            "baseline": {
                "members": [identity[:2]],
                "registration_source_sha256": "a" * 64,
            }
        }
    }
    observed = copy.deepcopy(inventory)
    observed["optional"]["maxtext"]["registration_source_sha256"] = "b" * 64
    _forbid_legacy_and_native(monkeypatch)
    with pytest.raises(ValueError, match="optional registration source"):
        inventory_owner.source_metadata_selection_preflight(
            inventory, observed, plan, registrations, [identity]
        )


@pytest.mark.parametrize("phase", ["deselected", "collection"])
def test_optin_missing_items_and_genuine_collection_errors_keep_the_ledger(
    tmp_path, phase
):
    result, report = _acceptance_subprocess(
        tmp_path, case="plugin::jax.numpy.add::0::raw-x64-off", phase=phase
    )
    assert result.returncode != 0
    assert report["counts"]["raw_caller_cells"] == 3712
    assert report["collection_complete"] is False
    assert report["outcomes_complete"] is False
    assert report["complete_migration_acceptance"] == "error"
    if phase == "collection":
        assert report["original_pytest_exitstatus"] == result.returncode == 2
        assert any(
            item["outcome"] == "failed" for item in report["actual_collection_reports"]
        )


def test_direct_raw_replay_and_metadata_path_have_same_real_evidence(
    original_scope, tmp_path, monkeypatch
):
    inventory, plan, registrations, adapters = original_scope
    identity = ("plugin::jax.numpy.add", 0, "raw-x64-on")
    descriptor = _descriptors(original_scope, selection=[identity])[0]
    original_executor = probe.execute_raw_probe_cell
    calls = []

    def observed_executor(*args, **kwargs):
        calls.append(args[1].identity)
        return original_executor(*args, **kwargs)

    monkeypatch.setattr(probe, "execute_raw_probe_cell", observed_executor)
    direct = probe.execute_probe_plan(
        plan,
        registrations,
        stage="product",
        raw_replay=True,
        limit=1,
        scenario=identity[2],
        cases=frozenset({probe.SourceProvenance(*identity[:2])}),
        replay_adapters=adapters,
        evidence_directory=tmp_path / "direct",
    )
    function = t_generator.make_source_test_function(
        descriptor,
        plan,
        registrations,
        stage="product",
        adapters=adapters,
        evidence_directory=tmp_path / "metadata",
        expected_statuses={
            "source_reference": "validated",
            "schema": "validated",
            "structure": "validated",
            "numerics": "unverified",
        },
    )
    function()
    metadata_cell = function._source_cell
    direct_cell = next(cell for cell in direct.cells if cell.identity == identity)
    left, right = direct_cell.history[-1], metadata_cell.history[-1]
    assert calls == [identity, identity]
    for name in (
        "row_contract_sha256",
        "scenario_sha256",
        "environment",
        "actual_settings",
        "export_settings",
        "original_trace_json",
        "original_trace_sha256",
        "supplied_inputs",
        "inputs",
        "references",
        "artifact_sha256",
        "export_report",
        "source_reference",
        "structure",
        "schema",
        "artifact",
        "lowering_audit",
        "runtime",
        "numerics",
        "raw_callback_status",
    ):
        assert getattr(left, name) == getattr(right, name), name
    assert left.raw_callback_status == right.raw_callback_status == "passed"
    assert function._source_outcome["capability_status"] == "unverified"
    assert [cell for cell in direct.cells if cell.identity != identity] == [
        cell for cell in plan.cells if cell.identity != identity
    ]


@pytest.mark.parametrize("changed_payload", [0x00000000, 0x7FC00001])
def test_supplied_signed_zero_or_nan_storage_drift_fails_before_native(
    changed_payload,
    monkeypatch,
    tmp_path,
):
    values = np.asarray([-0.0, np.nan], dtype=np.float32)
    scope = _synthetic(
        {
            "testcase": "storage",
            "callable": lambda x: x,
            "input_values": [values],
            "input_shapes": [(2,)],
        }
    )
    descriptor = _descriptors(scope)[1]
    values.view(np.uint32)[0 if changed_payload == 0 else 1] = changed_payload
    function = _function(
        scope, descriptor, tmp_path, expected={"source_reference": "failed"}
    )
    _forbid_legacy_and_native(monkeypatch)
    function()
    assert function._source_outcome["capability_status"] == "failed"
    assert "contract changed" in function._source_cell.detail
    assert function._source_cell.history[-1].raw_callback_status == "not-executed"


def test_stale_ordinal_and_epoch_are_rejected_without_factory(monkeypatch, tmp_path):
    scope = _synthetic(
        {"testcase": "stale", "callable": lambda x: x, "input_shapes": [(1,)]}
    )
    descriptor = _descriptors(scope)[1]
    _forbid_legacy_and_native(monkeypatch)
    with pytest.raises(ValueError, match="frozen row/cell"):
        _function(
            scope,
            replace(
                descriptor, identity=(descriptor.identity[0], 4, descriptor.identity[2])
            ),
            tmp_path,
        )
    function = _function(scope, descriptor, tmp_path)
    monkeypatch.setattr(
        t_generator,
        "source_metadata_epoch",
        lambda: replace(_EPOCH, source_sha256="0" * 64),
    )
    with pytest.raises(AssertionError, match="epoch changed before"):
        function()
    assert not tmp_path.exists() or not list(tmp_path.iterdir())


def test_original_factory_constructs_once_inside_declared_context_without_widening(
    tmp_path, monkeypatch
):
    calls = []

    def constructor(*, dtype):
        calls.append((dtype, bool(jax.config.x64_enabled)))
        return lambda x: x + jnp.asarray(1.0, dtype=jnp.float32)

    recipe = construct_and_call(constructor, dtype=np.float32)
    supplied = np.asarray([1.25, -0.0], dtype=np.float32)
    scope = _synthetic(
        {
            "testcase": "factory",
            "callable": recipe,
            "input_shapes": [(2,)],
            "input_values": [supplied],
        }
    )
    descriptor = _descriptors(scope)[1]
    previous = bool(jax.config.x64_enabled)

    def forbidden(*args, **kwargs):
        pytest.fail(
            "SOURCE must not generate variants, coerce factory dtypes, or invoke legacy allclose"
        )

    monkeypatch.setattr(t_generator, "generate_test_params", forbidden)
    monkeypatch.setattr(t_generator, "allclose", forbidden)
    monkeypatch.setattr(type(recipe), "with_dtype", forbidden)
    function = _function(scope, descriptor, tmp_path, stage="prepare")
    function()
    assert calls == [(np.float32, True)]
    assert bool(jax.config.x64_enabled) == previous
    record = function._source_cell.history[-1]
    assert (
        record.inputs[0].logical.dtype
        == record.references[0].logical.dtype
        == "float32"
    )
    assert record.supplied_inputs[0].logical_sha256 == record.inputs[0].logical_sha256
    assert record.requested_x64
    assert record.runtime is record.numerics is CheckStatus.UNVERIFIED


def test_constructor_exception_restores_context_and_records_original_failure(tmp_path):
    def constructor():
        raise RuntimeError("original-constructor-failure")

    scope = _synthetic(
        {
            "testcase": "exception",
            "callable": construct_and_call(constructor),
            "input_shapes": [(1,)],
        }
    )
    previous = bool(jax.config.x64_enabled)
    function = _function(
        scope,
        _descriptors(scope)[1],
        tmp_path,
        stage="prepare",
        expected={"source_reference": "unverified"},
    )
    function()
    assert bool(jax.config.x64_enabled) == previous
    assert "original-constructor-failure" in function._source_cell.detail
    assert function._source_outcome["capability_status"] == "unverified"


@pytest.mark.parametrize("weak", [False, True])
def test_strong_and_weak_inputs_survive_metadata_preparation(weak, tmp_path):
    supplied = jnp.asarray(1.25) if weak else np.asarray(1.25, dtype=np.float32)
    scope = _synthetic(
        {
            "testcase": "weakness",
            "callable": lambda x: x,
            "input_values": [supplied],
            "input_shapes": [()],
        }
    )
    function = _function(scope, _descriptors(scope)[1], tmp_path, stage="prepare")
    function()
    assert function._source_cell.history[-1].inputs[0].logical.weak_type is weak


def test_named_dynamic_parameters_are_not_specialized_or_widened(tmp_path):
    def fn(x, *, scale):
        return x * scale

    scale = np.asarray(2.0, dtype=np.float32)
    scope = _synthetic(
        {
            "testcase": "named",
            "callable": fn,
            "input_shapes": [(2,)],
            "input_dtypes": [np.float32],
            "input_params": {"scale": scale},
        }
    )
    function = _function(scope, _descriptors(scope)[1], tmp_path, stage="prepare")
    function()
    record = function._source_cell.history[-1]
    assert len(record.inputs) == 2
    assert [value.logical.dtype for value in record.inputs] == ["float32", "float32"]
    assert "mul" in record.original_trace_json


def test_nonleading_symbolic_raw_selection_keeps_alternate_bindings(
    original_scope, tmp_path
):
    identity = ("plugin::jax.numpy.squeeze", 7, "raw-x64-on")
    descriptor = _descriptors(original_scope, selection=[identity])[0]
    function = t_generator.make_source_test_function(
        descriptor,
        original_scope[1],
        original_scope[2],
        adapters=original_scope[3],
        evidence_directory=tmp_path,
        stage="prepare",
        expected_statuses={"source_reference": "validated"},
    )
    function()
    record = function._source_cell.history[-1]
    capture = record.replay_capture
    bindings = json.loads(Path(capture["sample-bindings"]["path"]).read_text())
    assert bindings["primary"] == {"B": 2}
    assert bindings["additional_native_samples"] == [{"B": 5}]
    assert capture["native-sample-1-eager"]["status"] == "captured"
    assert capture["native-sample-1-fresh-jit"]["status"] == "captured"
    assert function._source_outcome["capability_status"] == "unverified"


@pytest.mark.parametrize(
    "criterion",
    [{}, {"rtol": 0.0}, {"rtol": 0.0, "atol": 0.0, "skip_numeric_validation": True}],
)
def test_missing_partial_and_skipped_criteria_never_become_exact(
    criterion, tmp_path, monkeypatch
):
    scope = _synthetic(
        {
            "testcase": "criteria",
            "callable": lambda x: x,
            "input_shapes": [(1,)],
            **criterion,
        }
    )
    function = _function(
        scope,
        _descriptors(scope)[1],
        tmp_path,
        runtime="disabled",
        expected={
            "source_reference": "validated",
            "numerics": "unverified",
            "reviewed_criteria": "unverified",
        },
    )

    def forbidden(*args, **kwargs):
        pytest.fail(
            "Missing/partial/skipped criterion must not execute a target comparison"
        )

    monkeypatch.setattr(onnxruntime, "InferenceSession", forbidden)
    function()
    assert function._source_outcome["capability_status"] == "unverified"
    assert function._source_cell.history[-1].numerics is CheckStatus.UNVERIFIED


def test_unhandled_requests_remain_pending_even_after_returned_artifact(tmp_path):
    def runtime_callback(*args, **kwargs):
        pytest.fail("Unimplemented original callback must be retained, not replaced")

    scope = _synthetic(
        {
            "testcase": "requests",
            "callable": lambda x: x,
            "input_shapes": [(1,)],
            "expected_number_of_function_instances": 0,
            "check_onnx_load": True,
            "check_onnx_shape_inference": True,
            "check_deployment_readiness_report": True,
            "check_future_source_contract": True,
            "post_check_onnx_runtime": runtime_callback,
            "runtime_input_values": [[np.asarray([2.0], dtype=np.float32)]],
        }
    )
    pending = {
        name: "unverified"
        for name in (
            "expected_number_of_function_instances",
            "check_onnx_load",
            "check_onnx_shape_inference",
            "check_deployment_readiness_report",
            "post_check_onnx_runtime",
            "runtime_input_values",
            "check_future_source_contract",
        )
    }
    function = _function(
        scope,
        _descriptors(scope)[1],
        tmp_path,
        expected={"artifact": "validated", **pending},
    )
    function()
    assert function._source_outcome["stage_contract_passed"]
    assert function._source_outcome["capability_status"] == "unverified"


def test_complete_explicit_criterion_uses_same_artifact_and_retained_target(
    tmp_path, monkeypatch
):
    def fn(x):
        return x + jnp.asarray(1, dtype=jnp.float32)

    scope = _synthetic(
        {
            "testcase": "criteria-bound",
            "callable": fn,
            "input_values": [np.asarray([0.25, -1.0], dtype=np.float32)],
            "input_shapes": [(2,)],
            "rtol": 0.0,
            "atol": 0.0,
            "expected_output_shapes": [(2,)],
            "expected_output_dtypes": [np.float32],
            "runtime_input_values": [[np.asarray([2.0, -2.0], dtype=np.float32)]],
        }
    )
    scope = _protocol_scope(scope)
    from tests.extra_tests.converter import source_precision_raw_replay as raw

    exported, wrappers, results = [], [], []
    original_export = raw._export_prepared
    original_sample = raw.RawReplayCapture.native_sample
    original_runtime = probe._raw_metadata_runtime_protocol

    def export_once(*args, **kwargs):
        exported.append(args[0].fn)
        return original_export(*args, **kwargs)

    def observe_sample(self, *args, **kwargs):
        value = original_sample(self, *args, **kwargs)
        label = kwargs["label"]
        role = (label + "-" if label else "") + "fresh-jit-wrapper"
        wrappers.append(
            (self.native_values[role], self.native_samples[label]["wrapper_calls"])
        )
        return value

    def observe_runtime(*args, **kwargs):
        value = original_runtime(*args, **kwargs)
        results.append(value)
        return value

    monkeypatch.setattr(raw, "_export_prepared", export_once)
    monkeypatch.setattr(raw.RawReplayCapture, "native_sample", observe_sample)
    monkeypatch.setattr(probe, "_raw_metadata_runtime_protocol", observe_runtime)
    function = _function(
        scope,
        _descriptors(scope)[1],
        tmp_path,
        runtime="both",
        expected={
            "structure": "validated",
            "runtime": "validated",
            "numerics": "validated",
            "expected_output_shapes": "validated",
            "expected_output_dtypes": "validated",
            "runtime_input_values": "validated",
        },
    )
    function()
    outcome = function._source_outcome
    retained_record = _retained_stage_record(function)
    runtime = function._source_cell.history[-1].replay_capture["metadata_runtime"]
    assert retained_record["replay_capture"]["metadata_runtime"] == json.loads(
        json.dumps(runtime)
    )
    assert runtime["artifact_sha256"] == outcome["artifact_sha256"]
    assert (
        runtime["reference_role"] == "original-eager-first-warm-same-wrapper-per-sample"
    )
    assert len(exported) == 1 and exported[0] is fn
    assert len(wrappers) == 2 and wrappers[0][0] is not wrappers[1][0]
    assert [calls for _, calls in wrappers] == [2, 2]
    for sample in (
        function._source_cell.history[-1]
        .replay_capture["native_protocol"]["samples"]
        .values()
    ):
        assert (
            sample["roles"]["fresh-jit"]["jit_wrapper_identity"]
            == sample["roles"]["warm-jit"]["jit_wrapper_identity"]
            == sample["wrapper_identity"]
        )
        assert sample["roles"]["eager"]["jit_wrapper_identity"] is None
    assert len(results) == 1 and len(results[0].validations) == 2
    assert results[0].validation is results[0].validations[0]
    assert results[0].validations[0].target.optimization == "ORT_DISABLE_ALL"
    assert results[0].validations[1].target.optimization == "ORT_ENABLE_ALL"
    saved_artifact = Path(
        function._source_cell.history[-1].replay_capture["artifact"]["path"]
    )
    assert all(report._artifact == saved_artifact for report in results[0].validations)
    assert [sample.sample_sha256 for sample in results[0].validations[0].samples] == [
        sample.sample_sha256 for sample in results[0].validations[1].samples
    ]
    assert runtime["criteria"] == {
        "mode": "tolerant",
        "rtol": 0.0,
        "atol": 0.0,
        "equal_nan": True,
    }
    assert set(runtime["modes"]) == {"disabled", "all"}
    assert runtime["artifact_sha256"] == runtime["loaded_artifact_sha256"]
    for mode, record in runtime["modes"].items():
        assert len(record["samples"]) == len(record["output_captures"]) == 2
        assert record["target"]["providers"] == ("CPUExecutionProvider",)
        assert record["sample_identities"] == runtime["sample_identities"]
        assert record["artifact_sha256"] == outcome["artifact_sha256"]
        for index in range(2):
            saved = function._source_cell.history[-1].replay_capture[
                f"metadata-ort-{mode}-{index}"
            ]
            assert saved["status"] == "captured" and Path(saved["path"]).is_file()
    assert outcome["capability_status"] == "validated"
    scientific = inventory_owner.source_metadata_numerical_checks(
        scope[1].rows[0], asdict(function._source_cell.history[-1])
    )
    assert scientific["declared_criteria_pair"]["status"] == "validated"
    assert scientific["reviewed_criteria"]["status"] == "unverified"
    assert scientific["ort_disabled"]["status"] == "validated"
    assert scientific["dtype_shape_output_count"]["status"] == "validated"
    assert scientific["criteria_comparison"]["status"] == "validated"
    for name in (
        "native_three_role_agreement",
        "ort_all",
        "same_artifact_and_feeds_both_modes",
        "component_nan_masks",
        "zero_signs",
    ):
        assert scientific[name]["status"] == "validated"
    for name in ("reviewed_criteria", "operation_rounding", "required_internal_fp64"):
        assert scientific[name]["status"] == "unverified"


@pytest.mark.parametrize(
    "wrong",
    [{"expected_output_shapes": [(1, 2)]}, {"expected_output_dtypes": [np.float64]}],
)
def test_wrong_original_expected_contract_is_rejected_before_tolerance(
    wrong, tmp_path, monkeypatch
):
    scope = _synthetic(
        {
            "testcase": "wrong-output",
            "callable": lambda x: x,
            "input_shapes": [(2,)],
            "rtol": 1e9,
            "atol": 1e9,
            **wrong,
        }
    )
    name = next(iter(wrong))
    function = _function(
        scope,
        _descriptors(scope)[1],
        tmp_path,
        runtime="disabled",
        expected={name: "failed", "numerics": "unverified"},
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Wrong output count/dtype/shape must fail before tolerance")

    monkeypatch.setattr(onnxruntime, "InferenceSession", forbidden)
    function()
    assert function._source_outcome["capability_status"] == "failed"


def test_accounting_retains_all_frozen_obligations_without_claiming_collection(
    original_scope,
):
    report = _account(original_scope, _DESCRIPTORS)
    counts = report["counts"]
    assert counts["raw_rows"] == 1852 and counts["raw_caller_cells"] == 3704
    assert counts["metadata_free_registrations"] == 104
    assert (
        counts["optional_raw_rows"] == 50 and counts["original_manual_anchors"] == 752
    )
    assert counts["legacy_preparation_recipes"] > 0
    assert counts["actual_legacy_generated_variants"] is None
    assert counts["planned_source_selectors"] == 2
    assert (
        counts["actual_source_collected_pytest_ids"]
        == counts["terminal_source_stage_outcomes"]
        == 0
    )
    assert counts["linked_positive_prepared_counterparts"] == 0
    assert all(
        item["baseline_classification"] == "unclassified"
        for item in report["obligations"]
    )
    prepared = [
        item for item in report["obligations"] if item["kind"] == "legacy-preparation"
    ]
    assert any(
        item["recipe"]["dtype_preparation"] == "legacy-f64-prepared"
        for item in prepared
    )
    assert all(item["source_counterpart"] is None for item in prepared)


def test_absent_optional_rows_remain_pending_in_whole_plan(original_scope):
    from tests.extra_tests.framework.test_source_precision_inventory import (
        _omit_optional_cases,
    )

    actual = _omit_optional_cases(original_scope[0], unavailable=True)
    report = _account(original_scope, _DESCRIPTORS, observed_inventory=actual)
    pending = [
        item
        for item in report["obligations"]
        if item["kind"] == "raw-cell" and item["optional"]
    ]
    assert len(pending) == 100
    assert all(
        item["availability"] == "unavailable-retained"
        and item["capability_status"] == "unverified"
        for item in pending
    )
    assert len(report["retained_missing_cells"]) == 50


def test_completed_collection_rejects_missing_duplicate_or_orphan_ids():
    scope = _synthetic(
        {"testcase": "account", "callable": lambda x: x, "input_shapes": [(1,)]}
    )
    descriptors = _descriptors(scope)
    with pytest.raises(ValueError, match="missing selected"):
        _account(scope, descriptors, collection_complete=True)
    receipt = _collect_record(descriptors[0])
    with pytest.raises(ValueError, match="Duplicate"):
        _account(scope, descriptors, collected=[receipt, receipt])
    with pytest.raises(ValueError, match="Orphan actual"):
        _account(scope, descriptors[:1], collected=[_collect_record(descriptors[1])])
    with pytest.raises(ValueError, match="missing actual terminal"):
        _account(
            scope,
            descriptors[:1],
            collected=[receipt],
            collection_complete=True,
            outcomes_complete=True,
        )


def test_terminal_epoch_reconciliation_rejects_artifact_environment_and_orphan_results(
    tmp_path,
):
    scope = _synthetic(
        {"testcase": "result", "callable": lambda x: x, "input_shapes": [(1,)]}
    )
    descriptor = _descriptors(scope)[1]
    function = _function(
        scope, descriptor, tmp_path, expected={"numerics": "unverified"}
    )
    function()
    receipt = _collect_record(descriptor)
    outcome = {**function._source_outcome, "nodeid": receipt["nodeid"]}
    evidence = [function._source_cell.history[-1]]
    reports = _report_fixture(receipt["nodeid"])
    report = _account(
        scope,
        [descriptor],
        collected=[receipt],
        outcomes=[outcome],
        stage_evidence=evidence,
        pytest_reports=reports,
        collection_complete=True,
        outcomes_complete=True,
    )
    assert report["counts"]["passing_source_stage_contracts"] == 1
    assert report["counts"]["validated_raw_metadata_capabilities"] == 0
    for field in ("artifact_sha256", "evidence_sha256"):
        changed = {**outcome, field: "0" * 64}
        with pytest.raises(ValueError, match="epoch drift"):
            _account(
                scope,
                [descriptor],
                collected=[receipt],
                outcomes=[changed],
                stage_evidence=evidence,
            )
    # Scenario is part of the frozen collection descriptor and is rejected by
    # that earlier guard; artifact/evidence mutations above reach the epoch seal.
    changed = {**outcome, "scenario_sha256": "0" * 64}
    with pytest.raises(
        ValueError, match="^Changed outcome frozen adapter/sample scenario$"
    ):
        _account(
            scope,
            [descriptor],
            collected=[receipt],
            outcomes=[changed],
            stage_evidence=evidence,
        )
    changed = copy.deepcopy(outcome)
    changed["epoch"]["environment_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="epoch"):
        _account(
            scope,
            [descriptor],
            collected=[receipt],
            outcomes=[changed],
            stage_evidence=evidence,
        )
    with pytest.raises(ValueError, match="Orphan"):
        _account(scope, [descriptor], outcomes=[outcome], stage_evidence=evidence)
    with pytest.raises(ValueError, match="retained actual stage evidence"):
        _account(scope, [descriptor], collected=[receipt], outcomes=[outcome])

    # Matching summary changes plus the old identity/hash envelope must not turn
    # retained unverified numerical evidence into an accepted capability.
    promoted = copy.deepcopy(outcome)
    for assertion in promoted["assertions"].values():
        assertion["status"] = "validated"
    promoted["capability_status"] = "validated"
    promoted["stage_contract"] = {
        name: "validated" for name in outcome["stage_contract"]
    }
    promoted["stage_contract_passed"] = True
    with pytest.raises(ValueError, match="summary differs from retained"):
        _account(
            scope,
            [descriptor],
            collected=[receipt],
            outcomes=[promoted],
            stage_evidence=evidence,
            pytest_reports=reports,
        )

    # A source body is useful evidence, but absent setup/call/teardown reports
    # cannot count as an actual successful pytest capability or terminal test.
    body_only = _account(
        scope,
        [descriptor],
        collected=[receipt],
        outcomes=[outcome],
        stage_evidence=evidence,
    )
    assert body_only["counts"]["observed_source_stage_bodies"] == 1
    assert body_only["counts"]["actual_source_pytest_terminal_outcomes"] == 0
    assert body_only["counts"]["passing_source_stage_contracts"] == 0


def test_pytest_setup_teardown_and_missing_reports_are_independent_of_body(tmp_path):
    scope = _synthetic(
        {"testcase": "terminal", "callable": lambda x: x, "input_shapes": [(1,)]}
    )
    descriptor = _descriptors(scope)[1]
    receipt = _collect_record(descriptor)
    failed_setup = _report_fixture(receipt["nodeid"], setup="failed", call=None)
    report = _account(
        scope,
        [descriptor],
        collected=[receipt],
        pytest_reports=failed_setup,
        collection_complete=True,
        outcomes_complete=True,
    )
    assert report["pytest_outcomes"][receipt["nodeid"]]["status"] == "failed"
    assert report["counts"]["actual_source_pytest_terminal_outcomes"] == 1
    assert report["counts"]["observed_source_stage_bodies"] == 0
    function = _function(scope, descriptor, tmp_path)
    function()
    body = {**function._source_outcome, "nodeid": receipt["nodeid"]}
    evidence = [function._source_cell.history[-1]]
    failed_teardown = _report_fixture(receipt["nodeid"], teardown="failed")
    report = _account(
        scope,
        [descriptor],
        collected=[receipt],
        outcomes=[body],
        stage_evidence=evidence,
        pytest_reports=failed_teardown,
        collection_complete=True,
        outcomes_complete=True,
    )
    assert report["pytest_outcomes"][receipt["nodeid"]]["status"] == "failed"
    assert report["counts"]["passing_source_stage_contracts"] == 0
    assert report["counts"]["observed_source_stage_bodies"] == 1
    with pytest.raises(ValueError, match="body evidence contradicts actual setup"):
        _account(
            scope,
            [descriptor],
            collected=[receipt],
            outcomes=[body],
            stage_evidence=evidence,
            pytest_reports=failed_setup,
        )
    with pytest.raises(ValueError, match="missing actual terminal"):
        _account(
            scope,
            [descriptor],
            collected=[receipt],
            outcomes=[body],
            stage_evidence=evidence,
            pytest_reports=failed_teardown[:-1],
            collection_complete=True,
            outcomes_complete=True,
        )


def test_new_passing_variants_cannot_hide_removed_raw_row_or_manual_anchor():
    scope = _synthetic(
        {"testcase": "denominator", "callable": lambda x: x, "input_shapes": [(1,)]},
        without=[("no_metadata", "owner")],
        manual=[("tests/original.py", ["test_original"])],
    )
    descriptors = _descriptors(scope)
    actual = copy.deepcopy(scope[0])
    actual["cases"] = []
    with pytest.raises(ValueError, match="inventory drift"):
        _account(scope, descriptors, observed_inventory=actual)
    actual = copy.deepcopy(scope[0])
    actual["manual"] = [("tests/new.py", ["test_added_pass_1", "test_added_pass_2"])]
    with pytest.raises(ValueError, match="Original manual anchors"):
        _account(scope, descriptors, observed_inventory=actual)
    report = _account(scope, descriptors)
    assert report["counts"]["linked_positive_prepared_counterparts"] == 0


def test_original_classification_needs_original_evidence_and_does_not_close_fp64():
    scope = _synthetic(
        {"testcase": "positive", "callable": lambda x: x, "input_shapes": [(1,)]}
    )
    descriptors = _descriptors(scope)
    report = _account(scope, descriptors)
    fp64 = next(
        item
        for item in report["obligations"]
        if item["kind"] == "legacy-preparation"
        and item["recipe"]["dtype_preparation"] == "legacy-f64-prepared"
    )
    positive = {
        "obligation_id": fp64["obligation_id"],
        "classification": "positive",
        "original_evidence": "original-frozen-positive-receipt",
    }
    classified = _account(scope, descriptors, original_capabilities=[positive])
    item = next(
        item
        for item in classified["obligations"]
        if item["obligation_id"] == fp64["obligation_id"]
    )
    assert (
        item["baseline_classification"] == "positive" and item["status"] == "unverified"
    )
    assert item["source_counterpart"] is None
    with pytest.raises(ValueError, match="original evidence"):
        _account(
            scope,
            descriptors,
            original_capabilities=[{**positive, "original_evidence": None}],
        )


def test_fake_dataclass_cannot_replace_actual_stage_evidence():
    @dataclass(frozen=True)
    class FakeStage:
        sha256: str

    scope = _synthetic(
        {"testcase": "authority", "callable": lambda x: x, "input_shapes": [(1,)]}
    )
    with pytest.raises(ValueError, match="retained ProbeStageEvidence"):
        _account(scope, _descriptors(scope), stage_evidence=[FakeStage("0" * 64)])


def test_metadata_rejects_crossed_inventory_even_when_the_selected_row_matches(
    monkeypatch, tmp_path
):
    scope = _synthetic(
        {"testcase": "binding", "callable": lambda x: x, "input_shapes": [(1,)]}
    )
    row = scope[1].rows[0]
    adapters = {
        "inventory_sha256": scope[1].baseline_sha256,
        "rows": {
            f"{row.origin}::{row.ordinal}": {"contract_sha256": row.contract_sha256}
        },
    }
    descriptor = _descriptors((*scope, adapters))[0]
    _forbid_legacy_and_native(monkeypatch)
    crossed_plan = replace(scope[1], baseline_sha256="0" * 64)
    with pytest.raises(ValueError):
        t_generator.source_metadata_cases(
            crossed_plan,
            scope[0],
            scope[2],
            selection=[descriptor.identity],
            epoch=_EPOCH,
            adapters=adapters,
        )
    del adapters["inventory_sha256"]
    with pytest.raises(ValueError):
        t_generator.make_source_test_function(
            descriptor,
            scope[1],
            scope[2],
            adapters=adapters,
            evidence_directory=tmp_path,
            stage="prepare",
            expected_statuses={"source_reference": "validated"},
        )


@pytest.mark.parametrize("mutation", ["strip-header", "clear-manifest"])
def test_metadata_header_mutation_stops_before_execution(
    mutation, monkeypatch, tmp_path
):
    scope = _synthetic(
        {"testcase": "binding", "callable": lambda x: x, "input_shapes": [(1,)]}
    )
    adapters = {"inventory_sha256": scope[1].baseline_sha256, "rows": {}}
    descriptor = _descriptors((*scope, adapters))[0]
    function = t_generator.make_source_test_function(
        descriptor,
        scope[1],
        scope[2],
        adapters=adapters,
        evidence_directory=tmp_path / "not-created",
        stage="prepare",
        expected_statuses={"source_reference": "validated"},
    )
    if mutation == "strip-header":
        del adapters["inventory_sha256"]
    else:
        adapters.clear()

    def forbidden(*args, **kwargs):
        pytest.fail("Mutated inventory binding reached observation or execution")

    _forbid_legacy_and_native(monkeypatch)
    monkeypatch.setattr(t_generator, "source_metadata_epoch", forbidden)
    monkeypatch.setattr(probe, "execute_raw_probe_cell", forbidden)
    with pytest.raises((ValueError, AssertionError), match="inventory"):
        function()
    assert not (tmp_path / "not-created").exists()
    assert not hasattr(function, "_source_cell")
    assert not hasattr(function, "_source_outcome")


@pytest.mark.parametrize("mutation", ["clear-original", "replace-with-empty"])
def test_metadata_bound_descriptor_cannot_become_synthetic_before_function_creation(
    mutation, monkeypatch, tmp_path
):
    def constructor():
        pytest.fail("Descriptor binding validation invoked a factory")

    scope = _synthetic(
        {
            "testcase": "binding",
            "callable": construct_and_call(constructor),
            "input_shapes": [(1,)],
        }
    )
    adapters = {"inventory_sha256": scope[1].baseline_sha256, "rows": {}}
    descriptor = _descriptors((*scope, adapters))[0]
    assert json.loads(descriptor.adapter_json) == {}
    if mutation == "clear-original":
        adapters.clear()
        supplied = adapters
    else:
        supplied = {}
    _forbid_legacy_and_native(monkeypatch)
    with pytest.raises((ValueError, AssertionError), match="inventory"):
        t_generator.make_source_test_function(
            descriptor,
            scope[1],
            scope[2],
            adapters=supplied,
            evidence_directory=tmp_path / "not-created",
            stage="prepare",
            expected_statuses={"source_reference": "validated"},
        )
    assert not (tmp_path / "not-created").exists()


@pytest.mark.parametrize(
    "mode", ["synthetic-empty", "bound-absent-row", "bound-selected-row"]
)
def test_metadata_forwards_validated_inventory_binding_and_frozen_row(
    mode, monkeypatch, tmp_path
):
    class ReachedExecutor(Exception):
        pass

    scope = _synthetic(
        {"testcase": "binding", "callable": lambda x: x, "input_shapes": [(1,)]}
    )
    row = scope[1].rows[0]
    key = f"{row.origin}::{row.ordinal}"
    selected = (
        {"contract_sha256": row.contract_sha256} if mode == "bound-selected-row" else {}
    )
    adapters = (
        {
            "inventory_sha256": scope[1].baseline_sha256,
            "rows": {key: selected} if mode == "bound-selected-row" else {},
        }
        if mode != "synthetic-empty"
        else {}
    )
    descriptor = _descriptors((*scope, adapters))[0]
    function = t_generator.make_source_test_function(
        descriptor,
        scope[1],
        scope[2],
        adapters=adapters,
        evidence_directory=tmp_path,
        stage="prepare",
        expected_statuses={"source_reference": "validated"},
    )

    def execute(received_row, cell, registrations, **kwargs):
        assert received_row == row and cell.identity == descriptor.identity
        assert registrations == scope[2]
        assert kwargs["_inventory_sha256"] == scope[1].baseline_sha256
        assert kwargs["adapters"] == (
            {"inventory_sha256": scope[1].baseline_sha256, "rows": {key: selected}}
            if mode != "synthetic-empty"
            else {}
        )
        raise ReachedExecutor

    _forbid_legacy_and_native(monkeypatch)
    monkeypatch.setattr(probe, "execute_raw_probe_cell", execute)
    with pytest.raises(ReachedExecutor):
        function()


def test_metadata_private_inventory_binding_preserves_record_schema_and_accounting(
    monkeypatch,
):
    scope = _synthetic(
        {"testcase": "binding", "callable": lambda x: x, "input_shapes": [(1,)]}
    )
    synthetic = _descriptors(scope)[0]
    bound = _descriptors(
        (*scope, {"inventory_sha256": scope[1].baseline_sha256, "rows": {}})
    )[0]
    assert synthetic._inventory_sha256 is None
    assert bound._inventory_sha256 == scope[1].baseline_sha256
    public_fields = asdict(bound)
    assert public_fields.pop("_inventory_sha256") == scope[1].baseline_sha256
    expected_record = {
        **public_fields,
        "descriptor_sha256": t_generator._source_metadata_digest(public_fields),
    }
    assert bound.record() == synthetic.record() == expected_record
    assert bound.adapter_sha256 == synthetic.adapter_sha256
    assert bound.scenario_sha256 == synthetic.scenario_sha256
    _forbid_legacy_and_native(monkeypatch)
    report = _account(scope, [bound])
    assert report["counts"]["planned_source_selectors"] == 1
    assert report["counts"]["terminal_source_stage_outcomes"] == 0


def test_metadata_source_epoch_changes_with_checker_bytes_only(monkeypatch, tmp_path):
    paths = (
        "jax2onnx/__init__.py",
        "scripts/_source_inventory_epoch.py",
        "scripts/source_precision_probe.py",
        "scripts/source_precision_inventory.py",
        "tests/t_generator.py",
        "tests/conftest.py",
        "tests/extra_tests/converter/test_source_precision_metadata.py",
        "tests/extra_tests/converter/source_precision_support.py",
        "tests/extra_tests/converter/source_precision_raw_replay.py",
        "tests/extra_tests/converter/source_precision_diagnostics.py",
    )
    for relative in paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# source epoch data fixture: {relative}\n")
    environment = probe.ProbeEnvironment.observe()
    monkeypatch.setattr(probe.ProbeEnvironment, "observe", lambda: environment)
    monkeypatch.setattr(probe, "REPOSITORY_ROOT", tmp_path)
    _forbid_legacy_and_native(monkeypatch)
    before_bytes = {relative: (tmp_path / relative).read_bytes() for relative in paths}
    before = t_generator.source_metadata_epoch()
    (tmp_path / "scripts/_source_inventory_epoch.py").write_text(
        "# changed checker bytes in this disposable data fixture only\n"
    )
    after = t_generator.source_metadata_epoch()
    changed = {
        relative
        for relative in paths
        if (tmp_path / relative).read_bytes() != before_bytes[relative]
    }
    assert changed == {"scripts/_source_inventory_epoch.py"}
    assert before.source_sha256 != after.source_sha256
    assert before.environment_sha256 == after.environment_sha256


def test_adapter_sample_recipe_is_frozen_at_collection(monkeypatch, tmp_path):
    scope = _synthetic(
        {"testcase": "adapter", "callable": lambda x: x, "input_shapes": [("B",)]}
    )
    row = scope[1].rows[0]
    adapters = {
        "inventory_sha256": scope[1].baseline_sha256,
        "rows": {
            f"{row.origin}::{row.ordinal}": {
                "contract_sha256": row.contract_sha256,
                "samples": [{"B": 2}, {"B": 5}],
            }
        },
    }
    extended = (*scope, adapters)
    descriptor = _descriptors(extended)[1]
    function = t_generator.make_source_test_function(
        descriptor,
        scope[1],
        scope[2],
        stage="prepare",
        adapters=adapters,
        evidence_directory=tmp_path,
        expected_statuses={"source_reference": "validated"},
    )
    adapters["rows"][f"{row.origin}::{row.ordinal}"]["samples"][0]["B"] = 3
    _forbid_legacy_and_native(monkeypatch)
    with pytest.raises(AssertionError, match="adapter/sample recipe changed"):
        function()
    assert not hasattr(function, "_source_cell") and not hasattr(
        function, "_source_outcome"
    )


def test_repeated_early_failure_clears_old_successful_body_evidence(
    monkeypatch, tmp_path
):
    scope = _synthetic(
        {"testcase": "repeat", "callable": lambda x: x, "input_shapes": [(1,)]}
    )
    function = _function(scope, _descriptors(scope)[1], tmp_path, stage="prepare")
    function()
    assert function._source_outcome["stage_body_complete"]
    monkeypatch.setattr(
        t_generator,
        "source_metadata_epoch",
        lambda: replace(_EPOCH, source_sha256="0" * 64),
    )
    with pytest.raises(AssertionError, match="epoch changed before"):
        function()
    assert not hasattr(function, "_source_cell") and not hasattr(
        function, "_source_outcome"
    )


@pytest.mark.parametrize("shapes", [[("B", 2)], [["B", 2]], (("B", 2),), (["B", 2],)])
def test_legacy_recipe_accounting_decodes_both_shape_container_forms(shapes):
    scope = _synthetic(
        {"testcase": "containers", "callable": lambda x: x, "input_shapes": shapes}
    )
    obligations = inventory_owner.legacy_preparation_obligations(scope[0])
    assert {item["recipe"]["shape_preparation"] for item in obligations} == {
        "dynamic-B",
        "concrete-B=3",
    }
    assert len(obligations) == 4
    assert all(
        item["source_counterpart"] is None
        and item["baseline_classification"] == "unclassified"
        for item in obligations
    )


@pytest.mark.parametrize(
    "expectation",
    [
        {"expected_output_shapes": 5},
        {"expected_output_shapes": [None]},
        {"expected_output_dtypes": ["not-a-dtype"]},
        {"expected_output_dtypes": [None]},
    ],
)
def test_malformed_original_expected_observer_does_not_abort_raw_batch(
    expectation, tmp_path
):
    scope = _synthetic(
        {
            "testcase": "malformed",
            "callable": lambda x: x,
            "input_shapes": [(2,)],
            **expectation,
        }
    )
    result = probe.execute_probe_plan(
        scope[1],
        scope[2],
        stage="product",
        raw_replay=True,
        limit=2,
        replay_adapters={},
        evidence_directory=tmp_path,
    )
    assert len(result.cells) == 2 and all(
        len(cell.history) == 1 for cell in result.cells
    )
    for cell in result.cells:
        record = cell.history[-1]
        assert (
            record.source_reference
            is record.schema
            is record.artifact
            is CheckStatus.VALIDATED
        )
        assert (
            record.replay_capture["metadata_assertions"][next(iter(expectation))][
                "status"
            ]
            == "unverified"
        )


def _numeric_scope():
    return _synthetic(
        {
            "testcase": "runtime-fences",
            "callable": lambda x: x,
            "input_shapes": [(2,)],
            "rtol": 0.0,
            "atol": 0.0,
            "runtime_input_values": [[np.asarray([3.0, -2.0], dtype=np.float32)]],
        }
    )


def test_pre_runtime_binding_contradiction_does_not_claim_ort_failure(
    monkeypatch, tmp_path
):
    from jax2onnx import precision

    scope = _numeric_scope()
    original = probe._raw_metadata_runtime

    def bad_binding(*args, **kwargs):
        with monkeypatch.context() as scoped:
            scoped.setattr(precision, "_artifact_bytes", lambda model: (b"", "0" * 64))
            return original(*args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("Pre-runtime artifact contradiction must not create an ORT session")

    monkeypatch.setattr(probe, "_raw_metadata_runtime", bad_binding)
    monkeypatch.setattr(onnxruntime, "InferenceSession", forbidden)
    function = _function(
        scope,
        _descriptors(scope)[1],
        tmp_path,
        runtime="disabled",
        expected={
            "runtime_binding": "failed",
            "runtime": "unverified",
            "numerics": "unverified",
        },
    )
    function()
    observation = function._source_cell.history[-1].replay_capture["metadata_runtime"]
    assert observation["failure_phase"] == "binding"
    assert function._source_outcome["capability_status"] == "failed"


@pytest.mark.parametrize(
    "stop_role",
    [
        "metadata-runtime-reference-0-after-eager",
        "metadata-runtime-reference-0-after-fresh-jit",
        "immediately-before-metadata-validator",
    ],
)
def test_target_drift_stops_before_next_native_or_runtime_call(
    stop_role, monkeypatch, tmp_path
):
    from tests.extra_tests.converter import source_precision_raw_replay as raw
    from tests.extra_tests.converter.source_precision_support import (
        SourcePreparationError,
    )

    original = raw.RawReplayCapture.require_stable_target
    seen = []

    def stop(self, role):
        seen.append(role)
        if role == stop_role:
            error = SourcePreparationError(
                "injected changed target", category="replay-lane-gap"
            )
            self.gap(role, error)
            raise error
        return original(self, role)

    def forbidden(*args, **kwargs):
        pytest.fail("Known target drift must stop before ORT")

    monkeypatch.setattr(raw.RawReplayCapture, "require_stable_target", stop)
    monkeypatch.setattr(onnxruntime, "InferenceSession", forbidden)
    scope = _numeric_scope()
    function = _function(
        scope,
        _descriptors(scope)[1],
        tmp_path,
        runtime="disabled",
        expected={
            "source_reference": "unverified",
            "runtime": "unverified",
            "numerics": "unverified",
        },
    )
    function()
    capture = function._source_cell.history[-1].replay_capture
    assert stop_role in seen
    if stop_role.endswith("-after-eager"):
        assert (
            "metadata-runtime-reference-0-fresh-jit" not in capture["native_outcomes"]
        )
    assert function._source_outcome["capability_status"] == "unverified"


def test_post_runtime_fence_failure_retains_actual_validator_evidence(
    monkeypatch, tmp_path
):
    from tests.extra_tests.converter import source_precision_raw_replay as raw
    from tests.extra_tests.converter.source_precision_support import (
        SourcePreparationError,
    )

    original = raw.RawReplayCapture.require_stable_target

    def stop_after_runtime(self, role):
        if role == "after-metadata-runtime":
            error = SourcePreparationError(
                "injected post-runtime target gap", category="replay-lane-gap"
            )
            self.gap(role, error)
            raise error
        return original(self, role)

    monkeypatch.setattr(
        raw.RawReplayCapture, "require_stable_target", stop_after_runtime
    )
    scope = _numeric_scope()
    function = _function(
        scope,
        _descriptors(scope)[1],
        tmp_path,
        runtime="disabled",
        expected={
            "source_reference": "unverified",
            "runtime_binding": "unverified",
            "runtime": "validated",
            "numerics": "validated",
        },
    )
    function()
    record = function._source_cell.history[-1]
    observation = record.replay_capture["metadata_runtime"]
    assert observation["failure_phase"] == "binding"
    assert observation["target"]["providers"] == ("CPUExecutionProvider",)
    assert len(observation["samples"]) == 2
    assert function._source_outcome["capability_status"] == "unverified"


@pytest.mark.parametrize("dtype", [np.bool_, np.int32, np.int64, np.complex64])
def test_discrete_and_complex_metadata_preparation_preserves_logical_physical_contract(
    dtype, tmp_path
):
    supplied = np.asarray([1, 0], dtype=dtype)
    scope = _synthetic(
        {
            "testcase": "encoded",
            "callable": lambda x: x,
            "input_values": [supplied],
            "input_shapes": [(2,)],
        }
    )
    function = _function(scope, _descriptors(scope)[1], tmp_path, stage="prepare")
    function()
    evidence = function._source_cell.history[-1].inputs[0]
    assert evidence.logical.dtype == np.dtype(dtype).name
    if dtype is np.complex64:
        assert evidence.physical_dtype == "float32" and evidence.physical_shape == (
            2,
            2,
        )
    else:
        assert evidence.physical_dtype == np.dtype(
            dtype
        ).name and evidence.physical_shape == (2,)


def test_repeated_symbol_uses_one_native_scope_and_original_alternate_samples(tmp_path):
    scope = _synthetic(
        {
            "testcase": "repeated-symbol",
            "callable": lambda x: x,
            "input_shapes": [("B", 2, "B")],
        }
    )
    row = scope[1].rows[0]
    adapters = {
        "inventory_sha256": scope[1].baseline_sha256,
        "rows": {
            f"{row.origin}::{row.ordinal}": {
                "contract_sha256": row.contract_sha256,
                "samples": [{"B": 2}, {"B": 4}],
            }
        },
    }
    descriptor = _descriptors((*scope, adapters))[1]
    function = t_generator.make_source_test_function(
        descriptor,
        scope[1],
        scope[2],
        adapters=adapters,
        evidence_directory=tmp_path,
        stage="prepare",
        expected_statuses={"source_reference": "validated"},
    )
    function()
    record = function._source_cell.history[-1]
    assert record.inputs[0].logical.shape == ("B", 2, "B")
    capture = record.replay_capture
    assert capture["inputs"]["properties"][0]["shape"] == [2, 2, 2]
    assert capture["native-sample-1-inputs"]["properties"][0]["shape"] == [4, 2, 4]


def _original_native_add():
    target = compat._native_numpy.add
    assert target is compat._native_ufuncs.add and target is jnp.add
    assert type(target) is compat._NATIVE_CAPTURE_UFUNC_TYPE
    assert compat.selected_native_capture(target) is not None
    return target


def _same_identity_ufunc(target, *, cls=None):
    props = vars(target)["_ufunc__static_props"]
    constructor = type(target) if cls is None else cls
    return constructor(
        props["func"],
        props["nin"],
        props["nout"],
        name=target.__name__,
        nargs=props["nargs"],
        identity=props["identity"],
        call=props["call"],
        reduce=props["reduce"],
        accumulate=props["accumulate"],
        at=props["at"],
        reduceat=props["reduceat"],
    )


def _original_native_bitwise_and():
    target = compat._native_numpy.bitwise_and
    assert target is compat._native_ufuncs.bitwise_and and target is jnp.bitwise_and
    assert type(target) is compat._NATIVE_CAPTURE_UFUNC_TYPE
    assert compat.selected_native_capture(target) is not None
    return target


def test_native_bitwise_and_snapshot_retains_all_fields_and_reducer_alias():
    target = _original_native_bitwise_and()
    selected = compat.selected_native_capture(target)
    nodes = selected["state"]["nodes"]
    assert set(nodes[str(id(target))]["members"]) == set(vars(type(target)))
    assert set(vars(target)) == compat._NATIVE_CAPTURE_UFUNC_FIELDS
    props = vars(target)["_ufunc__static_props"]
    assert set(props) == compat._NATIVE_CAPTURE_STATIC_FIELDS
    assert (props["nin"], props["nout"], props["nargs"], props["identity"]) == (
        2,
        1,
        2,
        -1,
    )
    assert props["func"] is props["call"]
    assert props["accumulate"] is props["at"] is props["reduceat"] is None
    reducer = props["reduce"]
    function = vars(reducer)["_fun"]
    owner = sys.modules[function.__module__]
    assert vars(owner) is function.__globals__
    assert vars(owner)["_reduce_bitwise_and"] is reducer
    for value in (props, props["func"], reducer, function, vars(reducer)["_jit_info"]):
        assert str(id(value)) in nodes
    assert any(
        module is owner and name == "_reduce_bitwise_and" and original is reducer
        for module, name, original in compat._NATIVE_CAPTURE_BINDINGS[id(target)]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "func",
        "nin",
        "nout",
        "nargs",
        "identity",
        "call",
        "reduce",
        "accumulate",
        "at",
        "reduceat",
        "equal-dict",
        "instance-field",
        "private-field",
        "class-field",
        "public-binding",
        "defining-binding",
        "reducer-alias",
        "reducer-code",
        "reducer-config",
        "reducer-cache",
    ],
)
def test_native_bitwise_and_state_mutations_stop_before_invocation(
    mutation, monkeypatch, tmp_path
):
    from tests.extra_tests.converter import source_precision_raw_replay as raw

    target = _original_native_bitwise_and()
    props = vars(target)["_ufunc__static_props"]
    capture = raw.RawReplayCapture(tmp_path / mutation)
    capture.bind_target(target)
    invoked = []

    def replacement(*args, **kwargs):
        invoked.append(True)
        pytest.fail("Changed bitwise owner state must reject before invocation")

    expected = "selected native callable behavior/state changed"
    if mutation in compat._NATIVE_CAPTURE_STATIC_FIELDS:
        values = {"nin": 3, "nout": 2, "nargs": 3, "identity": 0}
        monkeypatch.setitem(props, mutation, values.get(mutation, replacement))
    elif mutation == "equal-dict":
        monkeypatch.setattr(target, "_ufunc__static_props", dict(props))
    elif mutation == "instance-field":
        monkeypatch.setattr(target, "unreviewed_field", None, raising=False)
        expected = "native ufunc instance/static schema changed"
    elif mutation == "private-field":
        monkeypatch.setitem(props, "unreviewed_field", None)
        expected = "native ufunc instance/static schema changed"
    elif mutation == "class-field":
        monkeypatch.setattr(type(target), "unreviewed_field", None, raising=False)
        expected = "native capture class schema changed"
    elif mutation in {"public-binding", "defining-binding"}:
        owner = (
            compat._native_numpy
            if mutation == "public-binding"
            else compat._native_ufuncs
        )
        foreign = _same_identity_ufunc(target)
        assert (
            foreign == target
            and hash(foreign) == hash(target)
            and foreign is not target
        )
        monkeypatch.setattr(owner, "bitwise_and", foreign)
        expected = "selected native public/defining binding changed"
    else:
        reducer = props["reduce"]
        function = vars(reducer)["_fun"]
        if mutation == "reducer-alias":
            monkeypatch.setattr(
                sys.modules[function.__module__], "_reduce_bitwise_and", replacement
            )
            expected = "selected native implementation binding changed"
        elif mutation == "reducer-code":
            monkeypatch.setattr(function, "__code__", function.__code__.replace())
        elif mutation == "reducer-config":
            info = vars(reducer)["_jit_info"]
            monkeypatch.setitem(
                vars(reducer),
                "_jit_info",
                info._replace(keep_unused=not info.keep_unused),
            )
        else:
            callback = compat._NATIVE_CAPTURE_CPP_BOUNDARY[-1](reducer)
            assert callback.__closure__
            monkeypatch.setattr(callback.__closure__[0], "cell_contents", None)
    with pytest.raises(ValueError, match=expected):
        compat.selected_native_capture(target)
    assert not capture.check_target("changed-bitwise-and-state")
    assert capture.target is target and invoked == []


@pytest.mark.parametrize("kind", ["same-hash", "frompyfunc", "subclass"])
def test_native_bitwise_and_foreign_ufuncs_remain_unselected(kind, tmp_path):
    from tests.extra_tests.converter import source_precision_raw_replay as raw
    from tests.extra_tests.converter.source_precision_support import (
        SourcePreparationError,
    )

    target = _original_native_bitwise_and()
    invoked = []
    mutable = [0]

    def foreign_function(x, y):
        invoked.append(True)
        return x + y + mutable[0]

    if kind == "frompyfunc":
        foreign = jnp.frompyfunc(foreign_function, nin=2, nout=1, identity=-1)
    elif kind == "subclass":

        class ForeignUfunc(type(target)):
            pass

        foreign = _same_identity_ufunc(target, cls=ForeignUfunc)
    else:
        foreign = _same_identity_ufunc(target)
        assert foreign == target and hash(foreign) == hash(target)
    assert compat.selected_native_capture(foreign) is None
    capture = raw.RawReplayCapture(tmp_path / kind)
    with pytest.raises(
        SourcePreparationError,
        match="Actual instantiated target needs an exact state capture adapter",
    ):
        capture.bind_target(foreign)
    assert capture.target is foreign and invoked == []


def test_original_add_metadata_retains_root_and_native_leaf_across_caller_contexts(
    original_scope,
    tmp_path,
):
    from tests.extra_tests.converter import source_precision_raw_replay as raw
    from tests.extra_tests.converter import source_precision_support as source

    _, plan, registrations, adapters = original_scope
    row = next(
        row
        for row in plan.rows
        if (row.origin, row.ordinal) == ("plugin::jax.numpy.add", 0)
    )
    metadata = probe._resolve_case(
        row, {origin: value for origin, _, value in registrations}
    )
    original_root = metadata["callable"]
    original_leaf = _original_native_add()
    assert original_root.__globals__["jnp"].add is original_leaf
    before = source.SourceSettings.observe()
    scenarios = source.raw_source_scenarios(
        metadata,
        source.SourceProvenance(row.origin, row.ordinal),
        criteria=row.criteria,
        preserve_supplied_properties=True,
    )
    for scenario in scenarios:
        capture = raw.RawReplayCapture(tmp_path / scenario.name)
        prepared = raw.prepare_raw_source_case(
            metadata,
            scenario,
            adapter=raw.replay_adapter(
                row, adapters, _inventory_sha256=plan.baseline_sha256
            ),
            capture=capture,
        )
        assert prepared.fn is capture.target is original_root
        assert prepared.metadata["callable"] is original_root
        assert original_root.__globals__["jnp"].add is original_leaf
        assert prepared.scenario.require_reviewed_criteria and prepared.criteria is None
        assert capture.readiness is CheckStatus.VALIDATED
        assert source.SourceSettings.observe() == before
        with source.source_configuration(scenario):
            assert source.SourceSettings.observe() == prepared.settings
            capture.require_stable_target("before-native-add-eager")
            eager = capture.invoke(
                "native-add-eager",
                lambda: prepared.fn(*prepared.arguments, **prepared.named_arguments),
            )
            capture.require_stable_target("after-native-add-eager")
            compiled = jax.jit(prepared.fn)
            fresh = capture.invoke(
                "native-add-fresh",
                lambda: compiled(*prepared.arguments, **prepared.named_arguments),
            )
            capture.require_stable_target("after-native-add-fresh")
            warmed = capture.invoke(
                "native-add-warmed",
                lambda: compiled(*prepared.arguments, **prepared.named_arguments),
            )
            capture.require_stable_target("after-native-add-warmed")
            for result in (eager, fresh, warmed):
                np.testing.assert_array_equal(
                    np.asarray(result), prepared.reference_outputs[0]
                )
            assert source.SourceSettings.observe() == prepared.settings
        assert source.SourceSettings.observe() == before
        assert original_root.__globals__["jnp"].add is original_leaf
    assert (
        probe._resolve_case(row, {origin: value for origin, _, value in registrations})[
            "callable"
        ]
        is original_root
    )


def test_native_add_snapshot_contains_complete_class_and_private_static_properties():
    target = _original_native_add()
    selected = compat.selected_native_capture(target)
    node = selected["state"]["nodes"][str(id(target))]
    assert set(node["members"]) == set(vars(type(target)))
    assert "__init__" in node["members"] and "__call__" in node["members"]
    assert set(vars(target)) == compat._NATIVE_CAPTURE_UFUNC_FIELDS
    props = vars(target)["_ufunc__static_props"]
    assert set(props) == compat._NATIVE_CAPTURE_STATIC_FIELDS
    assert props["func"] is props["call"]
    # These identities are graph nodes, not only the intentionally incomplete
    # upstream ufunc hash/equality projection.
    for value in (
        props,
        props["func"],
        props["reduce"],
        props["accumulate"],
        props["at"],
    ):
        assert str(id(value)) in selected["state"]["nodes"]


@pytest.mark.parametrize(
    "field",
    [
        "func",
        "nin",
        "nout",
        "nargs",
        "identity",
        "call",
        "reduce",
        "accumulate",
        "at",
        "reduceat",
    ],
)
def test_native_add_each_private_semantic_override_invalidates_capture(
    field, monkeypatch, tmp_path
):
    from tests.extra_tests.converter import source_precision_raw_replay as raw

    target = _original_native_add()
    equal_owner = _same_identity_ufunc(target)
    before_hash = hash(target)
    capture = raw.RawReplayCapture(tmp_path / field)
    capture.bind_target(target)

    def replacement(*args, **kwargs):
        pytest.fail("Mutated native implementation must be rejected without invocation")

    values = {"nin": 3, "nout": 2, "nargs": 3, "identity": 1}
    monkeypatch.setitem(
        vars(target)["_ufunc__static_props"], field, values.get(field, replacement)
    )
    if field in {"call", "reduce", "accumulate", "at", "reduceat"}:
        assert hash(target) == before_hash and target == equal_owner
    with pytest.raises(
        ValueError, match="selected native callable behavior/state changed"
    ):
        compat.selected_native_capture(target)
    assert not capture.check_target("mutated-native-add")
    assert capture.target is target


@pytest.mark.parametrize("kind", ["instance-field", "private-field", "class-field"])
def test_native_add_unknown_selected_schema_remains_an_explicit_gap(kind, monkeypatch):
    target = _original_native_add()
    if kind == "instance-field":
        monkeypatch.setattr(target, "unreviewed_native_field", None, raising=False)
        message = "native ufunc instance/static schema changed"
    elif kind == "private-field":
        monkeypatch.setitem(
            vars(target)["_ufunc__static_props"], "unreviewed_static_field", None
        )
        message = "native ufunc instance/static schema changed"
    else:
        monkeypatch.setattr(
            type(target), "unreviewed_native_field", None, raising=False
        )
        message = "native capture class schema changed"
    with pytest.raises(ValueError, match=message):
        compat.selected_native_capture(target)


def test_native_add_equal_private_dictionary_replacement_is_observed(monkeypatch):
    target = _original_native_add()
    props = vars(target)["_ufunc__static_props"]
    before_hash = hash(target)
    monkeypatch.setattr(target, "_ufunc__static_props", dict(props))
    assert hash(target) == before_hash
    with pytest.raises(
        ValueError, match="selected native callable behavior/state changed"
    ):
        compat.selected_native_capture(target)


@pytest.mark.parametrize("member", ["__init__", "__call__"])
def test_native_add_class_constructor_and_call_code_identity_are_pinned(
    member, monkeypatch
):
    target = _original_native_add()
    function = vars(type(target))[member]
    original = function.__code__
    monkeypatch.setattr(function, "__code__", original.replace())
    assert function.__code__ is not original
    with pytest.raises(
        ValueError, match="selected native callable behavior/state changed"
    ):
        compat.selected_native_capture(target)


@pytest.mark.parametrize(
    "kind", ["implementation-code", "cache-callback-code", "jit-config", "jit-function"]
)
def test_native_add_python_jit_and_cache_callback_mutations_are_observed(
    kind, monkeypatch, tmp_path
):
    from tests.extra_tests.converter import source_precision_raw_replay as raw

    target = _original_native_add()
    capture = raw.RawReplayCapture(tmp_path / kind)
    capture.bind_target(target)
    values = (np.asarray([1, 2], np.float32), np.asarray([3, 4], np.float32))
    compiled = jax.jit(target)
    np.testing.assert_array_equal(
        np.asarray(compiled(*values)), np.asarray([4, 6], np.float32)
    )
    np.testing.assert_array_equal(
        np.asarray(compiled(*values)), np.asarray([4, 6], np.float32)
    )
    capture.require_stable_target("warmed-native-add")
    wrapped = vars(target)["_ufunc__static_props"]["func"]
    fields = vars(wrapped)
    if kind == "implementation-code":
        function = fields["_fun"]
        monkeypatch.setattr(function, "__code__", function.__code__.replace())
    elif kind == "cache-callback-code":
        callback = compat._NATIVE_CAPTURE_CPP_BOUNDARY[-1](wrapped)
        monkeypatch.setattr(callback, "__code__", callback.__code__.replace())
    elif kind == "jit-config":
        info = fields["_jit_info"]
        monkeypatch.setitem(
            fields, "_jit_info", info._replace(keep_unused=not info.keep_unused)
        )
    else:

        def replacement(*args, **kwargs):
            pytest.fail("Changed Python JIT function must never be executed")

        monkeypatch.setitem(fields, "_fun", replacement)
    with pytest.raises(
        ValueError, match="selected native callable behavior/state changed"
    ):
        compat.selected_native_capture(target)
    assert not capture.check_target("mutated-native-add-jit")
    # The native C++ compiled cache remains the pre-existing terminal boundary;
    # this control freezes the exposed Python callback/configuration, not its
    # invisible native cache contents.


@pytest.mark.parametrize("owner_name", ["_native_numpy", "_native_ufuncs"])
def test_native_add_public_and_defining_owner_bindings_are_pinned(
    owner_name, monkeypatch
):
    target = _original_native_add()
    foreign = _same_identity_ufunc(target)
    assert hash(foreign) == hash(target) and foreign == target and foreign is not target
    monkeypatch.setattr(getattr(compat, owner_name), "add", foreign)
    with pytest.raises(
        ValueError, match="selected native public/defining binding changed"
    ):
        compat.selected_native_capture(target)
    with pytest.raises(
        ValueError, match="selected native binding points to a foreign owner"
    ):
        compat.selected_native_capture(foreign)


@pytest.mark.parametrize("owner_name", ["_native_numpy", "_native_ufunc_api"])
def test_native_add_class_binding_replacement_is_rejected(owner_name, monkeypatch):
    target = _original_native_add()

    class ForeignUfunc(type(target)):
        pass

    monkeypatch.setattr(getattr(compat, owner_name), "ufunc", ForeignUfunc)
    with pytest.raises(ValueError, match="selected native type binding changed"):
        compat.selected_native_capture(target)


@pytest.mark.parametrize("kind", ["same-hash", "frompyfunc", "subclass"])
def test_native_add_does_not_trust_foreign_ufuncs(kind, tmp_path):
    from tests.extra_tests.converter import source_precision_raw_replay as raw
    from tests.extra_tests.converter.source_precision_support import (
        SourcePreparationError,
    )

    target = _original_native_add()
    if kind == "frompyfunc":

        def original_function(x, y):
            return x + y

        foreign = jnp.frompyfunc(original_function, nin=2, nout=1, identity=0)
    elif kind == "subclass":

        class ForeignUfunc(type(target)):
            pass

        foreign = _same_identity_ufunc(target, cls=ForeignUfunc)
    else:
        foreign = _same_identity_ufunc(target)
        assert foreign == target and hash(foreign) == hash(target)
    assert compat.selected_native_capture(foreign) is None
    capture = raw.RawReplayCapture(tmp_path / kind)
    with pytest.raises(
        SourcePreparationError,
        match="Actual instantiated target needs an exact state capture adapter",
    ):
        capture.bind_target(foreign)
    assert any(
        "No exact instantiated state adapter" in gap["detail"] for gap in capture.gaps
    )
    assert capture.target is foreign


def _retained_add_accumulate():
    target = _original_native_add()
    callback = vars(target)["_ufunc__static_props"]["accumulate"]
    fields = vars(callback)
    function = fields["_fun"]
    assert type(function) is types.FunctionType and fields["__wrapped__"] is function
    namespace = function.__globals__
    owner = sys.modules[dict.get(namespace, "__name__")]
    assert vars(owner) is namespace and vars(owner)["cumsum"] is callback
    return target, callback, function, owner


def test_native_add_records_actual_accumulate_namespace_after_plugin_discovery():
    target, callback, function, owner = _retained_add_accumulate()
    plugin = import_module("jax2onnx.plugins.jax.numpy.cumsum")
    assert callback is plugin._ORIGINAL_JNP_CUMSUM
    assert jnp.cumsum is plugin._runtime_cumsum and jnp.cumsum is not callback
    selected = compat.selected_native_capture(target)
    node = selected["state"]["nodes"][str(id(callback))]
    assert node["defining_namespace"] == {
        "name": dict.get(function.__globals__, "__name__"),
        "module_identity": id(owner),
        "globals_identity": id(function.__globals__),
    }
    bindings = compat._NATIVE_CAPTURE_BINDINGS[id(target)]
    assert any(
        module is owner and name == "cumsum" and original is callback
        for module, name, original in bindings
    )
    assert not any(
        module is jnp and name == "cumsum" and original is callback
        for module, name, original in bindings
    )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("dtype", [np.int32, np.float64])
def test_native_callback_and_existing_cumsum_shim_keep_runtime_behavior(reverse, dtype):
    from tests.extra_tests.converter import source_precision_support as source

    target, callback, _, owner = _retained_add_accumulate()
    plugin = import_module("jax2onnx.plugins.jax.numpy.cumsum")
    shim = plugin._runtime_cumsum
    assert jnp.cumsum is shim and callback is plugin._ORIGINAL_JNP_CUMSUM
    before = source.SourceSettings.observe()
    values = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    scenario = source.SourceScenario("existing-cumsum-runtime-control", True)
    with source.source_configuration(scenario):
        native = target.accumulate(values, axis=1, dtype=dtype)
        np.testing.assert_array_equal(
            np.asarray(native), np.cumsum(values, axis=1, dtype=dtype)
        )
        assert vars(owner)["cumsum"] is callback and jnp.cumsum is shim
        expected = np.asarray(values, dtype=dtype)
        if reverse:
            expected = np.flip(
                np.cumsum(np.flip(expected, axis=1), axis=1, dtype=dtype), axis=1
            )
        else:
            expected = np.cumsum(expected, axis=1, dtype=dtype)
        actual = shim(values, axis=1, reverse=reverse, dtype=dtype)
        mapped = jax.vmap(lambda row: shim(row, axis=0, reverse=reverse, dtype=dtype))(
            values
        )
        for result in (actual, mapped):
            assert np.asarray(result).dtype == np.dtype(dtype)
            np.testing.assert_array_equal(np.asarray(result), expected)
        assert jnp.cumsum is shim and vars(owner)["cumsum"] is callback
        assert compat.selected_native_capture(target) is not None
    assert source.SourceSettings.observe() == before and jnp.cumsum is shim


def test_cumsum_existing_unsupported_exclusive_remains_a_specific_error():
    target, callback, _, owner = _retained_add_accumulate()
    plugin = import_module("jax2onnx.plugins.jax.numpy.cumsum")
    shim = plugin._runtime_cumsum
    # Both exported installed lax.cumsum signatures lack exclusive. Preserve
    # this existing error; it is not a new positive runtime capability.
    with pytest.raises(TypeError, match="unexpected keyword argument 'exclusive'"):
        shim(np.asarray([1, 2, 3], np.int32), axis=0, exclusive=True)
    assert jnp.cumsum is shim and vars(owner)["cumsum"] is callback
    assert compat.selected_native_capture(target) is not None


def test_retained_accumulate_defining_alias_replacement_is_rejected(monkeypatch):
    target, callback, _, owner = _retained_add_accumulate()
    invoked = []

    def foreign(*args, **kwargs):
        invoked.append(True)
        pytest.fail("A foreign defining callback must never be invoked")

    monkeypatch.setattr(owner, "cumsum", foreign)
    assert vars(target)["_ufunc__static_props"]["accumulate"] is callback
    with pytest.raises(
        ValueError, match="selected native implementation binding changed"
    ):
        compat.selected_native_capture(target)
    assert invoked == []


@pytest.mark.parametrize("change", ["remove", "replace", "globals-name"])
def test_accumulate_module_registration_cannot_become_an_unregistered_scope(
    change, monkeypatch
):
    target, _, function, owner = _retained_add_accumulate()
    name = dict.get(function.__globals__, "__name__")
    if change == "remove":
        monkeypatch.delitem(sys.modules, name)
    elif change == "replace":
        foreign = types.ModuleType(name)
        vars(foreign).update(vars(owner))
        monkeypatch.setitem(sys.modules, name, foreign)
        assert vars(foreign) is not function.__globals__
    else:
        monkeypatch.setitem(
            function.__globals__, "__name__", "unregistered.native.namespace"
        )
    with pytest.raises(
        ValueError, match="selected native callable behavior/state changed"
    ):
        compat.selected_native_capture(target)


def test_retained_accumulate_copied_same_name_globals_are_not_authentic(monkeypatch):
    target, callback, function, _ = _retained_add_accumulate()
    clone = types.FunctionType(
        function.__code__,
        dict(function.__globals__),
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    clone.__module__ = function.__module__
    clone.__kwdefaults__ = function.__kwdefaults__
    clone.__annotations__ = function.__annotations__
    assert clone.__globals__ is not function.__globals__
    monkeypatch.setitem(vars(callback), "_fun", clone)
    monkeypatch.setitem(vars(callback), "__wrapped__", clone)
    with pytest.raises(
        ValueError, match="selected native callable behavior/state changed"
    ):
        compat.selected_native_capture(target)


@pytest.mark.parametrize(
    "change", ["module-metadata", "code", "defaults", "kwdefaults", "cache-closure"]
)
def test_retained_accumulate_executable_state_remains_complete(change, monkeypatch):
    target, callback, function, _ = _retained_add_accumulate()
    if change == "module-metadata":
        monkeypatch.setattr(function, "__module__", "jax.numpy")
    elif change == "code":
        monkeypatch.setattr(function, "__code__", function.__code__.replace())
    elif change == "defaults":
        monkeypatch.setattr(
            function, "__defaults__", (*(function.__defaults__ or ()), None)
        )
    elif change == "kwdefaults":
        monkeypatch.setattr(
            function,
            "__kwdefaults__",
            {**(function.__kwdefaults__ or {}), "unreviewed_default": True},
        )
    else:
        cache_callback = compat._NATIVE_CAPTURE_CPP_BOUNDARY[-1](callback)
        cells = cache_callback.__closure__ or ()
        assert cells
        monkeypatch.setattr(cells[0], "cell_contents", None)
    with pytest.raises(
        ValueError, match="selected native callable behavior/state changed"
    ):
        compat.selected_native_capture(target)


@pytest.mark.parametrize("field", ["_fun", "__wrapped__"])
def test_native_jit_unknown_function_provenance_fails_before_foreign_hooks(
    field, monkeypatch
):
    target, callback, _, _ = _retained_add_accumulate()
    touched = []

    class ForeignState:
        def __getattribute__(self, name):
            touched.append(name)
            raise AssertionError("Foreign provenance must not be inspected")

    monkeypatch.setitem(vars(callback), field, ForeignState())
    with pytest.raises(
        ValueError,
        match="native JIT executable provenance needs exact Python functions",
    ):
        compat.selected_native_capture(target)
    assert touched == []


def test_generated_python_scope_keeps_its_existing_exact_graph_state():
    namespace = {"__name__": "unregistered.finite.native.graph.control"}
    exec("def generated(value):\n    return value\n", namespace)
    function = namespace["generated"]
    graph = compat._NativeCaptureGraph()
    graph.take(function)
    node = graph.nodes[str(id(function))]
    assert node["globals_identity"] == id(namespace)
    assert node["defining_namespace"] is None and graph.bindings == []
    assert compat.selected_native_capture(function) is None


def test_actual_original_root_public_add_edge_still_rejects_mutation(
    original_scope, monkeypatch, tmp_path
):
    from tests.extra_tests.converter import source_precision_raw_replay as raw

    _, plan, registrations, _ = original_scope
    row = next(
        row
        for row in plan.rows
        if (row.origin, row.ordinal) == ("plugin::jax.numpy.add", 0)
    )
    metadata = probe._resolve_case(
        row, {origin: value for origin, _, value in registrations}
    )
    original = metadata["callable"]
    capture = raw.RawReplayCapture(tmp_path / "actual-public-edge")
    capture.bind_target(original)
    called = []

    def foreign(*args, **kwargs):
        called.append(True)
        pytest.fail("Changed public source edge must fail before native invocation")

    monkeypatch.setattr(original.__globals__["jnp"], "add", foreign)
    assert not capture.check_target("actual-public-add-edge")
    assert capture.target is original and called == []


def _protocol_numeric_function(tmp_path, *, expected=None):
    scope = _protocol_scope(_numeric_scope())
    function = _function(
        scope,
        _descriptors(scope)[1],
        tmp_path,
        runtime="both",
        expected=expected or {"runtime": "validated", "numerics": "validated"},
    )
    return scope, function


def test_protocol_first_mode_mismatch_is_retained_and_second_really_runs(
    tmp_path, monkeypatch
):
    import jax2onnx.precision as precision

    original_validate, original_compare = (
        precision._validate_source_samples,
        precision._compare,
    )
    active, modes, comparison_counts = [], [], {}

    def validate(*args, **kwargs):
        modes.append(kwargs["optimization"])
        active.append(kwargs["optimization"])
        try:
            return original_validate(*args, **kwargs)
        finally:
            active.pop()

    def compare(actual, reference, criteria):
        if active:
            mode = active[-1]
            # The actual output was copied before the validator calls criteria.
            index = comparison_counts.get(mode, 0)
            comparison_counts[mode] = index + 1
            assert list(tmp_path.rglob(f"metadata-ort-{mode}-{index}.npz"))
            if mode == "disabled":
                return "Injected first-mode criterion failure"
        return original_compare(actual, reference, criteria)

    monkeypatch.setattr(precision, "_validate_source_samples", validate)
    monkeypatch.setattr(precision, "_compare", compare)
    scope, function = _protocol_numeric_function(
        tmp_path, expected={"runtime": "validated", "numerics": "failed"}
    )
    function()
    record = function._source_cell.history[-1]
    runtime = record.replay_capture["metadata_runtime"]
    assert modes == ["disabled", "all"]
    assert comparison_counts == {"disabled": 2, "all": 2}
    assert runtime["modes"]["disabled"]["status"] == "failed"
    assert runtime["modes"]["all"]["status"] == "validated"
    assert record.numerics is CheckStatus.FAILED
    scientific = inventory_owner.source_metadata_numerical_checks(
        scope[1].rows[0], asdict(record)
    )
    assert scientific["ort_disabled"]["status"] == "failed"
    assert scientific["ort_all"]["status"] == "validated"
    assert scientific["criteria_comparison"]["status"] == "failed"


def test_protocol_warm_only_mismatch_retains_all_native_roles_and_trace(
    tmp_path, monkeypatch
):
    from tests.extra_tests.converter import source_precision_raw_replay as raw

    original = raw.RawReplayCapture.invoke

    def invoke(self, role, call):
        value = original(self, role, call)
        if role == "warm-jit" and role in self.native_values:
            value = jax.tree_util.tree_map(
                lambda array: np.asarray(array) + np.asarray(1, dtype=array.dtype),
                value,
            )
            self.native_values[role] = value
        return value

    monkeypatch.setattr(raw.RawReplayCapture, "invoke", invoke)
    _, function = _protocol_numeric_function(
        tmp_path, expected={"source_reference": "failed", "artifact": "unverified"}
    )
    function()
    record = function._source_cell.history[-1]
    capture = record.replay_capture
    assert all(
        capture[role]["status"] == "captured"
        for role in (
            "eager",
            "fresh-jit",
            "warm-jit",
            "concrete-typed-trace",
            "declared-typed-trace",
        )
    )
    assert any(
        check["status"] == "failed"
        for comparison in capture["native_protocol"]["samples"][""]["comparisons"]
        for check in comparison["checks"]
    )
    with (
        np.load(capture["warm-jit"]["path"]) as warm,
        np.load(capture["eager"]["path"]) as eager,
    ):
        assert not np.array_equal(warm["value_0"], eager["value_0"])


@pytest.mark.parametrize("mutation", ["artifact", "feed", "source"])
def test_protocol_between_mode_mutation_stops_later_mode(
    tmp_path, monkeypatch, mutation
):
    import jax2onnx.precision as precision
    from tests.extra_tests.converter import source_precision_raw_replay as raw

    original = precision._validate_source_samples
    modes, captures = [], []
    original_bind = raw.RawReplayCapture.bind_target

    def bind(self, target):
        original_bind(self, target)
        captures.append(self)

    def validate(artifact, **kwargs):
        modes.append(kwargs["optimization"])
        report = original(artifact, **kwargs)
        if mutation == "artifact":
            with Path(artifact).open("ab") as stream:
                stream.write(b"preserved-test-mutation")
        elif mutation == "feed":
            inputs = kwargs["samples"][0].inputs
            name = next(iter(inputs))
            inputs[name] = np.array(inputs[name], copy=True) + np.asarray(
                1, dtype=inputs[name].dtype
            )
        else:

            def changed_source(x):
                return x

            captures[0].target.__code__ = changed_source.__code__
        return report

    monkeypatch.setattr(raw.RawReplayCapture, "bind_target", bind)
    monkeypatch.setattr(precision, "_validate_source_samples", validate)
    _, function = _protocol_numeric_function(
        tmp_path,
        expected={"numerics": "failed" if mutation != "source" else "unverified"},
    )
    function()
    observation = function._source_cell.history[-1].replay_capture["metadata_runtime"]
    assert modes == ["disabled"]
    assert observation["failure_phase"] == "binding"
    assert "all" not in observation["modes"]
    assert observation["modes"]["disabled"]["samples"]


def test_protocol_capture_writer_error_is_not_an_ort_execution_error(
    tmp_path, monkeypatch
):
    from tests.extra_tests.converter import source_precision_raw_replay as raw

    original = raw.RawReplayCapture.arrays

    def arrays(self, role, values):
        if role == "metadata-ort-disabled-0":
            raise OSError("Injected capture writer failure")
        return original(self, role, values)

    monkeypatch.setattr(raw.RawReplayCapture, "arrays", arrays)
    _, function = _protocol_numeric_function(
        tmp_path, expected={"runtime": "validated", "numerics": "unverified"}
    )
    function()
    runtime = function._source_cell.history[-1].replay_capture["metadata_runtime"]
    assert set(runtime["modes"]) == {"disabled", "all"}
    sample = runtime["modes"]["disabled"]["samples"][0]
    statuses = {check["category"]: check["status"] for check in sample["checks"]}
    assert statuses["runtime-execution"] == "validated"
    assert statuses["output-observation"] == "failed"
    assert statuses["numerical-acceptance"] == "unverified"
    assert runtime["modes"]["all"]["status"] == "validated"


def test_unreviewed_source_gets_no_extra_warm_invocation(tmp_path):
    scope = _numeric_scope()
    function = _function(
        scope,
        _descriptors(scope)[1],
        tmp_path,
        runtime="disabled",
        expected={"runtime": "validated"},
    )
    function()
    capture = function._source_cell.history[-1].replay_capture
    assert not any("warm-jit" in key for key in capture)
    assert set(capture["native_protocol"]["samples"][""]["roles"]) == {
        "eager",
        "fresh-jit",
    }
    assert capture["native_protocol"]["samples"][""]["wrapper_calls"] == 1


@pytest.mark.parametrize("mutation", ["input", "source"])
def test_protocol_native_input_or_source_drift_stops_before_next_role(
    tmp_path, monkeypatch, mutation
):
    from tests.extra_tests.converter import source_precision_raw_replay as raw
    from tests.extra_tests.converter.source_precision_support import (
        SourceSettings,
        SourcePreparationError,
    )

    value = np.asarray([0.25, -1], dtype=np.float32)

    def original(x):
        return x

    capture = raw.RawReplayCapture(tmp_path / "capture")
    capture.bind_target(original)
    invoke = capture.invoke

    def changed(role, call):
        result = invoke(role, call)
        if role == "eager":
            if mutation == "input":
                value[0] = np.float32(9)
            else:

                def changed_source(x):
                    return x + 1

                original.__code__ = changed_source.__code__
        return result

    monkeypatch.setattr(capture, "invoke", changed)
    with pytest.raises(SourcePreparationError):
        capture.native_sample(
            original,
            (value,),
            label="",
            settings=SourceSettings.observe(),
            metadata={},
            three_roles=True,
        )
    assert capture.records["eager"]["status"] == "captured"
    assert "fresh-jit" not in capture.records["native_outcomes"]
    assert "warm-jit" not in capture.records["native_outcomes"]


@pytest.mark.parametrize(
    "case,category",
    [
        ("integer", "criteria_comparison"),
        ("boolean", "criteria_comparison"),
        ("dtype", "dtype_shape_output_count"),
        ("shape", "dtype_shape_output_count"),
        ("tree", "dtype_shape_output_count"),
        ("nan-component", "component_nan_masks"),
        ("signed-zero", "zero_signs"),
    ],
)
def test_protocol_independent_scientific_checks_reject_counterexamples(case, category):
    from jax2onnx.precision import SampleCriteria
    from tests.extra_tests.converter.source_precision_raw_replay import (
        _protocol_comparison,
    )

    left, right = np.asarray([1], dtype=np.float32), np.asarray([1], dtype=np.float32)
    actual_tree = reference_tree = "one-leaf"
    if case == "integer":
        left, right = np.asarray([1], dtype=np.int32), np.asarray([2], dtype=np.int32)
    elif case == "boolean":
        left, right = np.asarray([True]), np.asarray([False])
    elif case == "dtype":
        left = left.astype(np.float64)
    elif case == "shape":
        left = left.reshape(1, 1)
    elif case == "tree":
        actual_tree = "different-tree"
    elif case == "nan-component":
        left = np.asarray([complex(np.nan, 2)], dtype=np.complex64)
        right = np.asarray([complex(2, np.nan)], dtype=np.complex64)
    elif case == "signed-zero":
        left, right = np.asarray([-0.0], dtype=np.float32), np.asarray(
            [0.0], dtype=np.float32
        )
    result = _protocol_comparison(
        [left],
        [right],
        SampleCriteria("tolerant", rtol=1e6, atol=1e6),
        actual_tree=actual_tree,
        reference_tree=reference_tree,
    )
    assert {item["category"]: item["status"] for item in result["checks"]}[
        category
    ] == "failed"


def test_protocol_distinct_nan_payloads_are_accepted_but_missing_criteria_is_not_exact():
    from jax2onnx.precision import SampleCriteria
    from tests.extra_tests.converter.source_precision_raw_replay import (
        _protocol_comparison,
    )

    left = np.asarray([0x7FC00001], dtype=np.uint32).view(np.float32)
    right = np.asarray([0xFFC00002], dtype=np.uint32).view(np.float32)
    result = _protocol_comparison([left], [right], SampleCriteria("exact"))
    assert all(item["status"] == "validated" for item in result["checks"])
    missing = _protocol_comparison([np.asarray([1.0])], [np.asarray([2.0])], None)
    assert {item["category"]: item["status"] for item in missing["checks"]}[
        "criteria_comparison"
    ] == "unverified"


def test_both_mode_request_without_reviewed_protocol_stays_unverified(tmp_path):
    scope = _numeric_scope()
    function = _function(
        scope,
        _descriptors(scope)[1],
        tmp_path,
        runtime="both",
        expected={"runtime": "unverified", "numerics": "unverified"},
    )
    function()
    record = function._source_cell.history[-1]
    assert not record.replay_capture["metadata_runtime"]["modes"]
    assert "warm-jit" not in record.replay_capture["native_outcomes"]
    scientific = inventory_owner.source_metadata_numerical_checks(
        scope[1].rows[0], asdict(record)
    )
    assert scientific["native_three_role_agreement"]["status"] == "unverified"
    assert scientific["ort_all"]["status"] == "unverified"


def test_protocol_uses_the_supplied_jitted_callable_without_unwrapping(
    tmp_path, monkeypatch
):
    from tests.extra_tests.converter import source_precision_raw_replay as raw

    supplied = jax.jit(lambda x: x + jnp.asarray(1, dtype=jnp.float32))
    scope = _protocol_scope(
        _synthetic(
            {
                "testcase": "original-jitted",
                "callable": supplied,
                "input_values": [np.asarray([0.25, -1], dtype=np.float32)],
                "input_shapes": [(2,)],
                "rtol": 0.0,
                "atol": 0.0,
            }
        )
    )
    scope[3]["rows"]["plugin::synthetic::0"]["capture_original_object"] = True
    original_bind, targets = raw.RawReplayCapture.bind_target, []

    def bind(self, target):
        targets.append(target)
        return original_bind(self, target)

    monkeypatch.setattr(raw.RawReplayCapture, "bind_target", bind)
    function = _function(
        scope,
        _descriptors(scope)[1],
        tmp_path,
        runtime="both",
        expected={"runtime": "validated", "numerics": "validated"},
    )
    function()
    assert targets == [supplied] and targets[0] is supplied
    sample = function._source_cell.history[-1].replay_capture["native_protocol"][
        "samples"
    ][""]
    assert sample["wrapper_calls"] == 2
    assert set(sample["roles"]) == {"eager", "fresh-jit", "warm-jit"}
    assert _retained_stage_record(function)["replay_capture"][
        "metadata_runtime"
    ] == json.loads(
        json.dumps(
            function._source_cell.history[-1].replay_capture["metadata_runtime"],
            sort_keys=True,
            separators=(",", ":"),
        )
    )


_SUPPLIED_JIT_GLOBAL = [1]


def _supplied_jit_global_original(x):
    return x + _SUPPLIED_JIT_GLOBAL[0]


def test_original_jit_capture_preserves_cache_growth_and_original_identity(tmp_path):
    from tests.extra_tests.converter import source_precision_raw_replay as raw
    from tests.extra_tests.converter.source_precision_support import SourceSettings

    supplied = jax.jit(lambda x: x + 1)
    capture = raw.RawReplayCapture(tmp_path / "jit-cache-growth")
    recipe = raw._capture_recipe(supplied)
    capture.bind_target(supplied)
    assert capture.target_witness is not None
    assert capture.target_witness.native_references
    result = capture.native_sample(
        supplied,
        (np.asarray([0.25, -1], np.float32),),
        label="",
        settings=SourceSettings.observe(),
        metadata={},
        three_roles=True,
    )
    np.testing.assert_array_equal(result, np.asarray([1.25, 0], np.float32))
    assert capture.target is supplied
    assert capture.check_target("after-actual-cache-growth")
    assert raw._capture_recipe(supplied) == recipe
    assert {
        key: value["status"]
        for key, value in capture.records["native_outcomes"].items()
    } == {"eager": "returned", "fresh-jit": "returned", "warm-jit": "returned"}
    assert capture.native_samples[""]["wrapper_calls"] == 2


@pytest.mark.parametrize(
    "mutation",
    [
        "fun",
        "wrapped",
        "coherent-functions",
        "jit-info",
        "code",
        "global",
        "unknown-field",
        "native-type",
        "cache-miss-code",
    ],
)
def test_original_jit_capture_rejects_mutation_before_next_role(
    tmp_path, monkeypatch, mutation
):
    from jax2onnx._compat import jax as compat
    from tests.extra_tests.converter import source_precision_raw_replay as raw
    from tests.extra_tests.converter.source_precision_support import SourceSettings

    if mutation == "global":
        monkeypatch.setitem(globals(), "_SUPPLIED_JIT_GLOBAL", [1])
    original = (
        _supplied_jit_global_original if mutation == "global" else (lambda x: x + 1)
    )
    supplied = jax.jit(original)
    capture = raw.RawReplayCapture(tmp_path / mutation)
    recipe = raw._capture_recipe(supplied)
    capture.bind_target(supplied)
    assert capture.check_target("unchanged")

    def replacement(x):
        return x - 1

    if mutation in {"fun", "coherent-functions"}:
        monkeypatch.setattr(supplied, "_fun", replacement)
    if mutation in {"wrapped", "coherent-functions"}:
        monkeypatch.setattr(supplied, "__wrapped__", replacement)
    elif mutation == "jit-info":
        monkeypatch.setattr(
            supplied,
            "_jit_info",
            supplied._jit_info._replace(keep_unused=not supplied._jit_info.keep_unused),
        )
    elif mutation == "code":
        changed_code = replacement.__code__.replace(
            co_name=original.__code__.co_name,
            co_qualname=original.__code__.co_qualname,
        )
        assert changed_code.co_name == original.__code__.co_name
        assert changed_code.co_qualname == original.__code__.co_qualname
        monkeypatch.setattr(original, "__code__", changed_code)
    elif mutation == "global":
        _SUPPLIED_JIT_GLOBAL[0] = 2
    elif mutation == "unknown-field":
        monkeypatch.setattr(supplied, "unreviewed_field", True, raising=False)
    elif mutation == "native-type":
        monkeypatch.setattr(
            compat._cast_cpu_native, "PjitFunction", type("PjitFunction", (), {})
        )
    elif mutation == "cache-miss-code":
        callback = compat._NATIVE_CAPTURE_CPP_BOUNDARY[-1](supplied)
        cache_miss = callback.__closure__[0].cell_contents

        def counterfeit(fun, jit_info):
            def cache_miss(*args, **kwargs):
                return fun, jit_info

            return cache_miss

        forged = counterfeit(original, supplied._jit_info)
        assert forged.__code__.co_name == cache_miss.__code__.co_name
        assert forged.__code__.co_freevars == cache_miss.__code__.co_freevars
        monkeypatch.setattr(cache_miss, "__code__", forged.__code__)
    attempted = []

    def forbidden(*args, **kwargs):
        attempted.append(True)
        raise AssertionError("A changed original must not be invoked")

    monkeypatch.setattr(capture, "invoke", forbidden)
    with pytest.raises(raw.SourcePreparationError, match="state is not preserved"):
        capture.native_sample(
            supplied,
            (np.asarray([1], np.float32),),
            label="",
            settings=SourceSettings.observe(),
            metadata={},
            three_roles=True,
        )
    assert attempted == []
    assert capture.records["native_outcomes"] == {}
    assert capture.readiness is CheckStatus.UNVERIFIED
    if mutation == "code":
        assert raw._capture_recipe(supplied) != recipe
    elif mutation != "global":
        with pytest.raises(raw.SourcePreparationError):
            raw._capture_recipe(supplied)


def test_original_jit_capture_rejects_native_typename_spoof():
    from tests.extra_tests.converter import source_precision_raw_replay as raw

    spoof = type(
        "PjitFunction", (), {"__module__": "jaxlib._jax", "__call__": lambda self, x: x}
    )()
    with pytest.raises(
        raw.SourcePreparationError, match="No exact original-object capture adapter"
    ):
        raw._capture_recipe(spoof)


def test_protocol_scientific_projection_rejects_stale_and_duplicate_bindings(tmp_path):
    scope, function = _protocol_numeric_function(tmp_path)
    function()
    record = asdict(function._source_cell.history[-1])
    for mutation in (
        "contract",
        "scenario",
        "wrapper",
        "role",
        "feed",
        "outputs",
        "references",
        "output-index",
        "sample-index",
    ):
        broken = copy.deepcopy(record)
        capture = broken["replay_capture"]
        if mutation == "contract":
            capture["protocol_binding"]["row_contract_sha256"] = "0" * 64
        elif mutation == "scenario":
            capture["protocol_binding"]["scenario_sha256"] = "0" * 64
        elif mutation == "wrapper":
            samples = list(capture["native_protocol"]["samples"].values())
            samples[1]["wrapper_identity"] = samples[0]["wrapper_identity"]
        elif mutation == "role":
            comparisons = capture["native_protocol"]["samples"][""]["comparisons"]
            comparisons[1]["role"] = comparisons[0]["role"]
        elif mutation == "feed":
            capture["metadata_runtime"]["modes"]["all"]["sample_identities"][0][
                "inputs"
            ][0]["sha256"] = ("0" * 64)
        elif mutation == "outputs":
            capture["metadata-ort-disabled-0"] = copy.deepcopy(
                capture["metadata-ort-disabled-1"]
            )
            capture["metadata_runtime"]["modes"]["disabled"]["output_captures"][0][
                "outputs"
            ] = copy.deepcopy(capture["metadata-ort-disabled-1"]["array_evidence"])
        elif mutation == "references":
            capture["eager-physical-reference"] = copy.deepcopy(
                capture["metadata-runtime-reference-0-eager-physical-reference"]
            )
        elif mutation == "output-index":
            output = capture["metadata_runtime"]["modes"]["disabled"][
                "output_captures"
            ][0]
            output["index"] = 2
            output["capture_role"] = "metadata-ort-disabled-2"
            capture["metadata-ort-disabled-2"] = copy.deepcopy(
                capture["metadata-ort-disabled-0"]
            )
        else:
            capture["metadata_runtime"]["modes"]["disabled"]["samples"][0]["index"] = 2
        with pytest.raises(ValueError):
            inventory_owner.source_metadata_numerical_checks(scope[1].rows[0], broken)
    missing = copy.deepcopy(record)
    missing["replay_capture"]["native_protocol"]["samples"][""]["roles"].pop("warm-jit")
    projected = inventory_owner.source_metadata_numerical_checks(
        scope[1].rows[0], missing
    )
    assert projected["native_three_role_agreement"]["status"] == "unverified"
    for role in ("metadata-ort-disabled-0", "eager-physical-reference"):
        missing = copy.deepcopy(record)
        missing["replay_capture"].pop(role)
        projected = inventory_owner.source_metadata_numerical_checks(
            scope[1].rows[0], missing
        )
        for name in (
            "ort_disabled",
            "dtype_shape_output_count",
            "criteria_comparison",
            "component_nan_masks",
            "zero_signs",
            "same_artifact_and_feeds_both_modes",
        ):
            assert projected[name]["status"] == "unverified"


@pytest.mark.parametrize("capture_name", ["adapter", "metadata-ort-disabled-0"])
def test_protocol_projection_does_not_accept_changed_retained_bytes(
    tmp_path, capture_name
):
    scope, function = _protocol_numeric_function(tmp_path)
    function()
    record = asdict(function._source_cell.history[-1])
    path = Path(record["replay_capture"][capture_name]["path"])
    with path.open("ab") as stream:
        stream.write(b"retained-evidence-tamper")
    with pytest.raises(ValueError):
        inventory_owner.source_metadata_numerical_checks(scope[1].rows[0], record)


def test_stateful_original_is_not_reset_or_given_an_extra_warm_role(tmp_path):
    from tests.extra_tests.converter import source_precision_raw_replay as raw
    from tests.extra_tests.converter.source_precision_support import (
        SourceSettings,
        SourcePreparationError,
    )

    calls = []

    def original(x):
        calls.append("actual-call")
        return x

    capture = raw.RawReplayCapture(tmp_path / "stateful-capture")
    capture.bind_target(original)
    with pytest.raises(SourcePreparationError):
        capture.native_sample(
            original,
            (np.asarray([1], dtype=np.float32),),
            label="",
            settings=SourceSettings.observe(),
            metadata={},
            three_roles=False,
        )
    assert calls == ["actual-call"]
    assert "warm-jit" not in capture.records["native_outcomes"]
    assert capture.records["eager"]["status"] == "captured"


@pytest.mark.parametrize("failure", ["load", "run"])
def test_protocol_first_mode_load_or_run_failure_does_not_hide_second(
    tmp_path, monkeypatch, failure
):
    import jax2onnx.precision as precision

    original = precision._validate_source_samples
    modes = []

    def fail(*args, **kwargs):
        raise RuntimeError("Injected first-mode " + failure + " failure")

    def validate(*args, **kwargs):
        modes.append(kwargs["optimization"])
        if kwargs["optimization"] == "disabled":
            with monkeypatch.context() as patch:
                patch.setattr(
                    precision,
                    "_artifact_bytes" if failure == "load" else "_run_cpu_samples",
                    fail,
                )
                return original(*args, **kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(precision, "_validate_source_samples", validate)
    scope, function = _protocol_numeric_function(
        tmp_path,
        expected={
            "runtime": "unverified" if failure == "load" else "failed",
            "numerics": "failed",
        },
    )
    function()
    record = function._source_cell.history[-1]
    runtime = record.replay_capture["metadata_runtime"]
    assert modes == ["disabled", "all"]
    assert runtime["modes"]["disabled"]["status"] == "failed"
    assert runtime["modes"]["all"]["status"] == "validated"
    assert runtime["status"] == "failed"
    projected = inventory_owner.source_metadata_numerical_checks(
        scope[1].rows[0], asdict(record)
    )
    assert projected["ort_disabled"]["status"] == "failed"
    assert projected["ort_all"]["status"] == "validated"
    assert projected["same_artifact_and_feeds_both_modes"]["status"] == "unverified"


def test_unreviewed_protocol_is_rejected_before_any_native_role(tmp_path, monkeypatch):
    from tests.extra_tests.converter import source_precision_raw_replay as raw

    scope = _protocol_scope(_numeric_scope())
    scope[3]["rows"]["plugin::synthetic::0"]["native_protocol"]["reviewed"] = False

    def forbidden(*args, **kwargs):
        pytest.fail("Unreviewed protocol must stop before invoking a native role")

    monkeypatch.setattr(raw.RawReplayCapture, "native_sample", forbidden)
    function = _function(
        scope,
        _descriptors(scope)[1],
        tmp_path,
        runtime="both",
        expected={"source_reference": "unverified", "runtime": "unverified"},
    )
    function()
    assert not function._source_cell.history[-1].replay_capture["native_outcomes"]


def test_generated_cell_drift_retains_serializable_partial_native_evidence(
    tmp_path, monkeypatch
):
    from tests.extra_tests.converter import source_precision_raw_replay as raw

    original = raw.RawReplayCapture.invoke

    def invoke(self, role, call):
        value = original(self, role, call)
        if role == "eager":

            def changed_source(x):
                return x

            self.target.__code__ = changed_source.__code__
        return value

    monkeypatch.setattr(raw.RawReplayCapture, "invoke", invoke)
    _, function = _protocol_numeric_function(
        tmp_path, expected={"source_reference": "unverified", "runtime": "unverified"}
    )
    function()
    record = function._source_cell.history[-1]
    payload = asdict(record)
    assert _retained_stage_record(function) == json.loads(json.dumps(payload))
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    assert record.sha256 == hashlib.sha256(encoded).hexdigest()
    capture = payload["replay_capture"]
    assert capture["eager"]["status"] == "captured"
    assert capture["native_outcomes"]["eager"]["status"] == "returned"
    assert "fresh-jit" not in capture["native_outcomes"]
    assert "warm-jit" not in capture["native_outcomes"]
    assert capture["native_protocol"]["samples"][""]["roles"]["eager"]


@pytest.mark.parametrize("failed_role", ["eager", "fresh-jit", "warm-jit"])
def test_alternate_native_exception_is_failed_after_all_roles_are_retained(
    tmp_path, monkeypatch, failed_role
):
    from tests.extra_tests.converter import source_precision_raw_replay as raw

    original = raw.RawReplayCapture.invoke
    target = "metadata-runtime-reference-0-" + failed_role

    def invoke(self, role, call):
        if role == target:

            def raised():
                call()
                raise RuntimeError("Injected alternate role exception: " + failed_role)

            return original(self, role, raised)
        return original(self, role, call)

    monkeypatch.setattr(raw.RawReplayCapture, "invoke", invoke)
    scope, function = _protocol_numeric_function(
        tmp_path,
        expected={
            "source_reference": "failed",
            "runtime": "unverified",
            "numerics": "failed",
        },
    )
    function()
    record = function._source_cell.history[-1]
    json.dumps(asdict(record))
    capture = record.replay_capture
    assert record.source_reference is CheckStatus.FAILED
    assert record.numerics is CheckStatus.FAILED
    assert capture["metadata_runtime"]["reference_status"] == "failed"
    assert not capture["metadata_runtime"]["modes"]
    roles = capture["native_protocol"]["samples"]["metadata-runtime-reference-0"][
        "roles"
    ]
    assert set(roles) == {"eager", "fresh-jit", "warm-jit"}
    for role in roles:
        name = "metadata-runtime-reference-0-" + role
        if role == failed_role:
            assert roles[role]["outcome"]["status"] == "raised"
            assert (
                "Injected alternate role exception" in roles[role]["outcome"]["detail"]
            )
        else:
            assert roles[role]["outcome"]["status"] == "returned"
            assert capture[name]["status"] == "captured"
    assert capture["metadata-runtime-reference-0-inputs"]["status"] == "captured"
    scientific = inventory_owner.source_metadata_numerical_checks(
        scope[1].rows[0], asdict(record)
    )
    assert scientific["native_three_role_agreement"]["status"] == "failed"


def test_stage_record_write_failure_remains_missing_evidence(tmp_path, monkeypatch):
    from tests.extra_tests.converter import source_precision_raw_replay as raw

    original = raw.RawReplayCapture.document

    def document(self, role, value):
        if role == "stage-evidence":
            raise OSError("Injected stage record writer failure")
        return original(self, role, value)

    monkeypatch.setattr(raw.RawReplayCapture, "document", document)
    _, function = _protocol_numeric_function(
        tmp_path,
        expected={
            "source_reference": "unverified",
            "runtime": "validated",
            "numerics": "unverified",
        },
    )
    function()
    record = function._source_cell.history[-1]
    json.dumps(asdict(record))
    assert record.sha256
    capture = record.replay_capture
    assert capture["stage-evidence-persistence"] == {
        "status": "failed",
        "exception": "builtins.OSError",
        "detail": "Injected stage record writer failure",
    }
    assert record.source_reference is CheckStatus.UNVERIFIED
    assert record.numerics is CheckStatus.UNVERIFIED
    assert record.runtime is CheckStatus.VALIDATED
    assert not (
        Path(capture["artifact"]["path"]).parent / "stage-evidence.json"
    ).exists()
    assert set(capture["metadata_runtime"]["modes"]) == {"disabled", "all"}
    assert function._source_outcome["capability_status"] == "unverified"


def _discrete_protocol_scope(case):
    scope = _protocol_scope(_synthetic(case))
    scope[3]["rows"]["plugin::synthetic::0"]["criteria"] = {
        "mode": "exact",
        "applicability": "all-logical-discrete",
    }
    return scope


def _discrete_numeric_function(
    tmp_path,
    *,
    dtype=np.int32,
    caller="raw-x64-on",
    expected=None,
    supplied_jit=False,
    scalar=False,
):
    def fn(x):
        return jnp.bitwise_not(x)

    if supplied_jit:
        fn = jax.jit(fn)
    scope = _discrete_protocol_scope(
        {
            "testcase": "conditional-discrete",
            "callable": fn,
            "input_values": [np.asarray(1 if scalar else [1, 0, 3], dtype=dtype)],
            "runtime_input_values": [
                [np.asarray(0 if scalar else [0, 1, 2], dtype=dtype)]
            ],
        }
    )
    if supplied_jit:
        scope[3]["rows"]["plugin::synthetic::0"]["capture_original_object"] = True
    descriptor = _descriptors(scope, selection=[("plugin::synthetic", 0, caller)])[0]
    function = _function(
        scope,
        descriptor,
        tmp_path,
        runtime="both",
        expected=expected or {"runtime": "validated", "numerics": "validated"},
    )
    return scope, function


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.bool_])
@pytest.mark.parametrize("caller", ["raw-x64-off", "raw-x64-on"])
def test_conditional_discrete_criterion_retains_original_jit_and_all_samples(
    tmp_path, monkeypatch, dtype, caller
):
    from tests.extra_tests.converter import source_precision_raw_replay as raw

    targets = []
    original = raw.RawReplayCapture.bind_target

    def bind(self, target):
        targets.append(target)
        return original(self, target)

    monkeypatch.setattr(raw.RawReplayCapture, "bind_target", bind)
    scope, function = _discrete_numeric_function(
        tmp_path, dtype=dtype, caller=caller, supplied_jit=True
    )
    supplied = scope[2][0][2]["testcases"][0]["callable"]
    row = scope[1].rows[0]
    frozen = asdict(row)
    function()
    record = function._source_cell.history[-1]
    retained = _retained_stage_record(function)
    capture = record.replay_capture
    runtime = capture["metadata_runtime"]
    assert targets == [supplied] and targets[0] is supplied
    assert asdict(row) == frozen and row.criteria is None
    assert runtime["criteria"] == {
        "mode": "exact",
        "rtol": None,
        "atol": None,
        "equal_nan": True,
    }
    assert set(runtime["modes"]) == {"disabled", "all"}
    assert runtime["native_sample_labels"] == ["", "metadata-runtime-reference-0"]
    assert len(runtime["sample_identities"]) == 2
    assert retained["replay_capture"]["metadata_runtime"] == json.loads(
        json.dumps(runtime)
    )
    resolution = capture["criteria_resolution"]
    assert resolution["status"] == "validated"
    assert resolution["choice"] == {
        "mode": "exact",
        "applicability": "all-logical-discrete",
    }
    assert resolution["contract_sha256"] == row.contract_sha256
    assert resolution["protocol_binding"] == capture["protocol_binding"]
    assert resolution["adapter_sha256"] == capture["adapter"]["sha256"]
    assert resolution["trace_sha256"] == capture[resolution["trace_role"]]["sha256"]
    assert resolution["criteria"] == runtime["criteria"]
    assert resolution["output_applicability"] == {
        "component_nan_masks": "not-applicable",
        "zero_signs": "not-applicable",
    }
    saved = capture["criteria-resolution"]
    assert saved["status"] == "captured"
    payload = Path(saved["path"]).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == saved["sha256"]
    assert json.loads(payload) == json.loads(json.dumps(resolution))
    assert retained["replay_capture"]["criteria_resolution"] == json.loads(
        json.dumps(resolution)
    )
    samples = capture["native_protocol"]["samples"]
    assert set(samples) == {"", "metadata-runtime-reference-0"}
    assert len({sample["wrapper_identity"] for sample in samples.values()}) == 2
    assert [sample["label"] for sample in resolution["samples"]] == list(samples)
    for label, sample in samples.items():
        assert sample["wrapper_calls"] == 2
        assert set(sample["roles"]) == {"eager", "fresh-jit", "warm-jit"}
        assert (
            sample["roles"]["fresh-jit"]["jit_wrapper_identity"]
            == sample["roles"]["warm-jit"]["jit_wrapper_identity"]
            == sample["wrapper_identity"]
        )
        for role in sample["roles"]:
            name = (label + "-" if label else "") + role
            assert capture[name]["status"] == "captured"
    for mode, result in runtime["modes"].items():
        assert result["artifact_sha256"] == record.artifact_sha256
        assert result["sample_identities"] == runtime["sample_identities"]
        assert len(result["samples"]) == len(result["output_captures"]) == 2
        for index in range(2):
            assert capture[f"metadata-ort-{mode}-{index}"]["status"] == "captured"
    scientific = inventory_owner.source_metadata_numerical_checks(row, asdict(record))
    assert scientific["declared_criteria_pair"]["status"] == "unverified"
    for name in (
        "reviewed_criteria",
        "native_three_role_agreement",
        "ort_disabled",
        "ort_all",
        "same_artifact_and_feeds_both_modes",
        "dtype_shape_output_count",
        "criteria_comparison",
    ):
        assert scientific[name]["status"] == "validated"
    for name in ("operation_rounding", "required_internal_fp64"):
        assert scientific[name]["status"] == "validated"
        assert scientific[name]["applicability"] == "not-applicable"
    for name in ("component_nan_masks", "zero_signs"):
        assert scientific[name]["status"] == "validated"
        assert scientific[name]["applicability"] == "not-applicable"


@pytest.mark.parametrize("dtype", [np.int32, np.bool_])
@pytest.mark.parametrize("failed_mode", ["disabled", "all"])
def test_conditional_discrete_exact_mismatch_keeps_both_real_ort_modes(
    tmp_path, monkeypatch, dtype, failed_mode
):
    import jax2onnx.precision as precision

    original_validate = precision._validate_source_samples
    original_run = precision._run_cpu_samples
    active, modes, calls = [], [], {}

    def validate(*args, **kwargs):
        mode = kwargs["optimization"]
        active.append(mode)
        modes.append(mode)
        try:
            return original_validate(*args, **kwargs)
        finally:
            active.pop()

    def run(*args, **kwargs):
        actual = original_run(*args, **kwargs)
        mode = active[-1]
        calls[mode] = calls.get(mode, 0) + 1
        if mode == failed_mode:
            changed = np.array(actual[0], copy=True)
            if changed.dtype.kind == "b":
                changed[0] = not changed[0]
            else:
                changed[0] += np.asarray(1, dtype=changed.dtype)
            return (changed, *actual[1:])
        return actual

    monkeypatch.setattr(precision, "_validate_source_samples", validate)
    monkeypatch.setattr(precision, "_run_cpu_samples", run)
    scope, function = _discrete_numeric_function(
        tmp_path,
        dtype=dtype,
        expected={"runtime": "validated", "numerics": "failed"},
    )
    function()
    record = function._source_cell.history[-1]
    capture = record.replay_capture
    runtime = capture["metadata_runtime"]
    assert modes == ["disabled", "all"]
    assert calls == {"disabled": 2, "all": 2}
    assert runtime["modes"][failed_mode]["status"] == "failed"
    other = "all" if failed_mode == "disabled" else "disabled"
    assert runtime["modes"][other]["status"] == "validated"
    assert record.numerics is CheckStatus.FAILED
    for index in range(2):
        with (
            np.load(capture[f"metadata-ort-{failed_mode}-{index}"]["path"]) as actual,
            np.load(capture[f"metadata-ort-{other}-{index}"]["path"]) as reference,
        ):
            assert not np.array_equal(actual["value_0"], reference["value_0"])
    scientific = inventory_owner.source_metadata_numerical_checks(
        scope[1].rows[0], asdict(record)
    )
    assert scientific["reviewed_criteria"]["status"] == "validated"
    assert scientific["ort_" + failed_mode]["status"] == "failed"
    assert scientific["ort_" + other]["status"] == "validated"
    assert scientific["criteria_comparison"]["status"] == "failed"


@pytest.mark.parametrize("caller", ["raw-x64-off", "raw-x64-on"])
def test_real_bitwise_and_conditional_criterion_preserves_raw_metadata(
    original_scope, tmp_path, monkeypatch, caller
):
    from tests.extra_tests.converter import source_precision_raw_replay as raw

    payload = inventory_owner.CURRENT_BASELINE.read_bytes()
    inventory = json.loads(payload)
    plan = probe.build_probe_plan(
        inventory, baseline_sha256=hashlib.sha256(payload).hexdigest()
    )
    row = next(
        row
        for row in plan.rows
        if (row.origin, row.ordinal) == ("plugin::jax.numpy.bitwise_and", 0)
    )
    assert row.contract_sha256 == (
        "7a0ab58cda9df361c9ccb707366f2b0c7864f1a3f4a74d9b31afc77845690d95"
    )
    assert row.criteria is None
    registrations = original_scope[2]
    metadata = probe._resolve_case(
        row, {origin: value for origin, _, value in registrations}
    )
    original = metadata["callable"]
    callback = metadata["post_check_onnx_graph"]
    inputs = [value.tobytes() for value in metadata["input_values"]]
    adapters = {
        "inventory_sha256": plan.baseline_sha256,
        "rows": {
            "plugin::jax.numpy.bitwise_and::0": {
                "contract_sha256": row.contract_sha256,
                "native_protocol": {
                    "mode": "stateless-three-role-v1",
                    "reviewed": True,
                },
                "criteria": {
                    "mode": "exact",
                    "applicability": "all-logical-discrete",
                },
            }
        },
    }
    scope = (inventory, plan, registrations, adapters)
    descriptor = _descriptors(scope, selection=[(row.origin, row.ordinal, caller)])[0]
    exported = []
    original_export = raw._export_prepared

    def export(*args, **kwargs):
        exported.append(args[0].fn)
        return original_export(*args, **kwargs)

    monkeypatch.setattr(raw, "_export_prepared", export)
    function = _function(
        scope,
        descriptor,
        tmp_path,
        runtime="both",
        expected={
            "source_reference": "validated",
            "raw_callback": "validated",
            "runtime": "validated",
            "numerics": "validated",
        },
    )
    function()
    record = function._source_cell.history[-1]
    capture = record.replay_capture
    assert exported == [original] and exported[0] is original
    assert metadata["post_check_onnx_graph"] is callback
    assert [value.tobytes() for value in metadata["input_values"]] == inputs
    assert record.raw_callback_status == "passed"
    assert record.requested_stage == "product"
    assert record.category == "audited-profile"
    assert record.lowering_audit is CheckStatus.VALIDATED
    assert record.export_report is not None
    assert record.export_report.profile == "source-static-v1"
    assert "audit_pending" not in record.detail
    for mode in ("disabled", "all"):
        with np.load(capture[f"metadata-ort-{mode}-0"]["path"]) as outputs:
            np.testing.assert_array_equal(
                outputs["value_0"], np.asarray([1, 0, 2], dtype=np.int32)
            )
    with np.load(capture["eager"]["path"]) as eager:
        np.testing.assert_array_equal(
            eager["value_0"], np.asarray([1, 0, 2], dtype=np.int32)
        )
    scientific = inventory_owner.source_metadata_numerical_checks(row, asdict(record))
    assert scientific["reviewed_criteria"]["status"] == "validated"
    assert scientific["declared_criteria_pair"]["status"] == "unverified"
    for name in ("operation_rounding", "required_internal_fp64"):
        assert scientific[name]["status"] == "validated"
        assert scientific[name]["applicability"] == "not-applicable"
    ledger = _account(scope, [descriptor])
    assert ledger["counts"]["raw_caller_cells"] == 3712
    assert ledger["counts"]["criterion_unresolved_raw_rows"] == 1838
    assert ledger["counts"]["planned_source_selectors"] == 1
    assert ledger["complete_migration_acceptance"] != "validated"


@pytest.mark.parametrize("dtype", [np.bool_, np.int32, np.uint32])
def test_bitwise_applicability_binds_complete_native_source_and_saved_ir(
    tmp_path, dtype
):
    scope, function = _discrete_numeric_function(tmp_path, dtype=dtype)
    function()
    record = asdict(function._source_cell.history[-1])
    capture = record["replay_capture"]
    structure = capture["source_structure"]
    proof = structure["floating_applicability"]
    assert proof["status"] == "validated"
    assert json.loads(json.dumps(proof["protocol_binding"])) == json.loads(
        json.dumps(capture["protocol_binding"])
    )
    assert proof["artifact_sha256"] == record["artifact_sha256"]
    assert proof["source_structure_sha256"] == inventory_owner._contract_hash(
        {
            key: value
            for key, value in structure.items()
            if key != "floating_applicability"
        }
    )
    document = capture["source-bitwise-applicability"]
    payload = Path(document["path"]).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == document["sha256"]
    assert json.loads(payload) == json.loads(json.dumps(proof))
    assert proof["closure"]["constant_count"] == proof["closure"]["function_count"] == 0
    assert len(proof["closure"]["ir_nodes"]) == 1
    assert len(capture["native_protocol"]["samples"]) == 2
    assert set(capture["metadata_runtime"]["modes"]) == {"disabled", "all"}
    checks = inventory_owner.source_metadata_numerical_checks(scope[1].rows[0], record)
    for name in ("operation_rounding", "required_internal_fp64"):
        assert checks[name]["status"] == "validated"
        assert checks[name]["applicability"] == "not-applicable"
    assert checks["native_three_role_agreement"]["status"] == "validated"
    assert checks["criteria_comparison"]["status"] == "validated"
    assert checks["same_artifact_and_feeds_both_modes"]["status"] == "validated"


def test_bitwise_applicability_rejects_missing_forged_and_detached_bindings(tmp_path):
    scope, function = _discrete_numeric_function(tmp_path)
    function()
    original = asdict(function._source_cell.history[-1])
    assert (
        original["replay_capture"]["source_structure"]["floating_applicability"][
            "status"
        ]
        == "validated"
    )
    for mutation in (
        "source",
        "artifact",
        "protocol",
        "closure",
        "coverage",
        "trace",
        "export-manifest",
        "original-trace",
    ):
        record = copy.deepcopy(original)
        capture = record["replay_capture"]
        structure = capture["source_structure"]
        proof = structure["floating_applicability"]
        if mutation == "source":
            proof["source_structure_sha256"] = "0" * 64
        elif mutation == "artifact":
            proof["artifact_sha256"] = "0" * 64
        elif mutation == "protocol":
            proof["protocol_binding"]["scenario_sha256"] = "0" * 64
        elif mutation == "closure":
            proof["closure"]["ir_nodes"][0]["op"] = "Identity"
        elif mutation == "original-trace":
            trace = json.loads(record["original_trace_json"])
            trace["equations"][0]["primitive"] = "and"
            record["original_trace_json"] = json.dumps(trace)
        else:
            if mutation == "coverage":
                structure["ir_result"]["checked_equations"] = 0
            elif mutation == "trace":
                for key in (
                    "prepared_trace_sha256",
                    "actual_trace_sha256",
                    "fresh_prepared_trace_sha256",
                ):
                    structure["source_binding"][key] = "0" * 64
            else:
                exported = structure["source_binding"]["export_trace_manifest"]
                payload = json.loads(Path(exported["path"]).read_bytes())
                payload["manifest"]["equations"][0]["primitive"] = "add"
                path = tmp_path / "forged-export-manifest.json"
                path.write_text(
                    json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
                )
                structure["source_binding"]["export_trace_manifest"] = {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "bytes": path.stat().st_size,
                }
            proof["source_structure_sha256"] = inventory_owner._contract_hash(
                {
                    key: value
                    for key, value in structure.items()
                    if key != "floating_applicability"
                }
            )
        # Rebind the actual forged document too: hash-only validation is insufficient.
        path = tmp_path / ("forged-bitwise-" + mutation + ".json")
        path.write_text(json.dumps(proof, sort_keys=True, separators=(",", ":")) + "\n")
        capture["source-bitwise-applicability"] = {
            "status": "captured",
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        with pytest.raises(ValueError):
            inventory_owner.source_metadata_numerical_checks(scope[1].rows[0], record)
    for missing in (
        "floating_applicability",
        "source-bitwise-applicability",
        "artifact",
    ):
        record = copy.deepcopy(original)
        capture = record["replay_capture"]
        (
            capture["source_structure"]
            if missing == "floating_applicability"
            else capture
        ).pop(missing)
        checks = inventory_owner.source_metadata_numerical_checks(
            scope[1].rows[0], record
        )
        for name in ("operation_rounding", "required_internal_fp64"):
            assert checks[name]["status"] == "unverified"
            assert checks[name].get("applicability") != "not-applicable"
    contradictory = copy.deepcopy(original)
    contradictory["replay_capture"].pop("declared-typed-trace")
    with pytest.raises(ValueError):
        inventory_owner.source_metadata_numerical_checks(
            scope[1].rows[0], contradictory
        )
    failed = copy.deepcopy(original)
    failed["source_reference"] = "failed"
    for item in failed["replay_capture"]["native_protocol"]["samples"][""][
        "comparisons"
    ][0]["checks"]:
        if item["category"] == "criteria_comparison":
            item.update(status="failed", detail="Original exact mismatch")
    checks = inventory_owner.source_metadata_numerical_checks(scope[1].rows[0], failed)
    assert checks["native_three_role_agreement"]["status"] == "failed"


def test_bitwise_applicability_document_failure_cannot_discharge_checks(
    tmp_path, monkeypatch
):
    from tests.extra_tests.converter import source_precision_raw_replay as raw

    original = raw.RawReplayCapture.document

    attempts = []

    def document(self, role, value):
        if role == "source-bitwise-applicability":
            attempts.append(self)
            raise OSError("Injected applicability document failure")
        return original(self, role, value)

    monkeypatch.setattr(raw.RawReplayCapture, "document", document)
    scope, function = _discrete_numeric_function(tmp_path)
    function()
    record = asdict(function._source_cell.history[-1])
    capture = record["replay_capture"]
    assert len(attempts) == 1
    attempts[0].finalize_source_structure(attempts[0].model)
    assert len(attempts) == 1
    assert (
        capture["source_structure"]["floating_applicability"]["status"] == "unverified"
    )
    assert capture["source-bitwise-applicability"] == {
        "status": "failed",
        "exception": "builtins.OSError",
        "detail": "Injected applicability document failure",
    }
    assert not (
        Path(capture["artifact"]["path"]).parent / "source-bitwise-applicability.json"
    ).exists()
    checks = inventory_owner.source_metadata_numerical_checks(scope[1].rows[0], record)
    assert checks["criteria_comparison"]["status"] == "validated"
    for name in ("operation_rounding", "required_internal_fp64"):
        assert checks[name]["status"] == "unverified"


def test_bitwise_applicability_repeated_fences_preserve_single_document(
    tmp_path, monkeypatch
):
    from tests.extra_tests.converter import source_precision_raw_replay as raw

    original = raw.RawReplayCapture.document
    captures = []

    def document(self, role, value):
        if role == "source-bitwise-applicability":
            captures.append(self)
        return original(self, role, value)

    monkeypatch.setattr(raw.RawReplayCapture, "document", document)
    scope, function = _discrete_numeric_function(tmp_path)
    function()
    assert len(captures) == 1
    capture = captures[0]
    record = asdict(function._source_cell.history[-1])
    path = Path(capture.records["source-bitwise-applicability"]["path"])
    before = path.read_bytes()
    capture.finalize_source_structure(capture.model)
    assert capture.source_structure_status() is CheckStatus.VALIDATED
    assert (
        capture.records["source_structure"]["floating_applicability"]["status"]
        == "validated"
    )
    assert path.read_bytes() == before and len(captures) == 1
    changed = json.loads(before)
    changed["closure"]["ir_nodes"][0]["op"] = "Identity"
    path.write_text(json.dumps(changed, sort_keys=True, separators=(",", ":")) + "\n")
    capture.finalize_source_structure(capture.model)
    assert len(captures) == 1 and path.read_bytes() != before
    assert capture.source_structure_status() is CheckStatus.FAILED
    assert capture.records["source_structure"]["artifact_binding"]["status"] == "failed"
    assert (
        capture.records["source_structure"]["floating_applicability"]["status"]
        == "failed"
    )
    with pytest.raises(ValueError):
        inventory_owner.source_metadata_numerical_checks(scope[1].rows[0], record)


def _bitwise_closure_fixture():
    import onnx
    from tests.extra_tests.converter.source_precision_diagnostics import (
        typed_trace_manifest,
    )

    closed = jax.make_jaxpr(lambda x, y: jax.lax.bitwise_and(x, y))(
        np.asarray([1, 2, 3], np.int32), np.asarray([3, 1, 2], np.int32)
    )
    manifest = json.loads(json.dumps(typed_trace_manifest(closed)))
    value = onnx.helper.make_tensor_value_info
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("BitwiseAnd", ["x", "y"], ["out"])],
            "complete-bitwise",
            [
                value("x", onnx.TensorProto.INT32, [3]),
                value("y", onnx.TensorProto.INT32, [3]),
            ],
            [value("out", onnx.TensorProto.INT32, [3])],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 23)],
    )
    assert inventory_owner._source_bitwise_closure(manifest, model) is not None
    return closed, manifest, model


@pytest.mark.parametrize(
    "mutation",
    [
        "float-internal",
        "cast",
        "reduction",
        "nested",
        "effects",
        "opaque",
        "literal",
        "params",
        "weak",
        "symbol",
    ],
)
def test_bitwise_applicability_requires_all_source_values_and_operations(mutation):
    import onnx
    from tests.extra_tests.converter.source_precision_diagnostics import (
        typed_trace_manifest,
    )

    _, manifest, model = _bitwise_closure_fixture()
    if mutation == "float-internal":
        closed = jax.make_jaxpr(
            lambda x, y: (x.astype(np.float32) + y.astype(np.float32)).astype(np.int32)
        )(np.asarray([1, 2, 3], np.int32), np.asarray([3, 1, 2], np.int32))
        manifest = json.loads(json.dumps(typed_trace_manifest(closed)))
        del model.graph.node[:]
        model.graph.node.extend(
            [
                onnx.helper.make_node("Cast", ["x"], ["fx"], to=onnx.TensorProto.FLOAT),
                onnx.helper.make_node("Cast", ["y"], ["fy"], to=onnx.TensorProto.FLOAT),
                onnx.helper.make_node("Add", ["fx", "fy"], ["sum"]),
                onnx.helper.make_node(
                    "Cast", ["sum"], ["out"], to=onnx.TensorProto.INT32
                ),
            ]
        )
        model.graph.value_info.extend(
            onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [3])
            for name in ("fx", "fy", "sum")
        )
        assert len(manifest["inputs"]) == len(model.graph.input) == 2
        assert len(manifest["equations"]) == len(model.graph.node) == 4
        assert manifest["outputs"][0]["aval"][0] == "int32"
    elif mutation == "cast":
        manifest["equations"][0]["primitive"] = "convert_element_type"
    elif mutation == "reduction":
        manifest["equations"][0]["primitive"] = "reduce_sum"
    elif mutation == "nested":
        original = copy.deepcopy(manifest)
        manifest["equations"][0]["primitive"] = "jit"
        manifest["equations"][0]["params"] = {"jaxpr": original}
    elif mutation == "effects":
        manifest["equations"][0]["effects"] = ["effect"]
    elif mutation == "opaque":
        closed = jax.make_jaxpr(lambda key: jax.random.key_data(key))(jax.random.key(0))
        manifest = json.loads(json.dumps(typed_trace_manifest(closed)))
        assert manifest["outputs"][0]["aval"][0] == "uint32"
        assert manifest["inputs"][0][0].startswith("key<")
    elif mutation == "literal":
        manifest["consts"] = [["float32", [], False]]
        manifest["const_payloads"] = [1.0]
    elif mutation == "params":
        manifest["equations"][0]["params"] = {"unreviewed": True}
    elif mutation == "weak":
        manifest["equations"][0]["outputs"][0]["aval"][2] = True
    else:
        manifest["inputs"][0][1] = ["B"]
    assert inventory_owner._source_bitwise_closure(manifest, model) is None


@pytest.mark.parametrize(
    "mutation",
    [
        "hidden-bitwise",
        "matched-dead",
        "hidden-float",
        "float-value-info",
        "float-initializer",
        "wrong-op",
        "wrong-edge",
        "dtype",
        "shape",
        "attribute",
        "subgraph",
        "domain",
        "function",
    ],
)
def test_bitwise_applicability_checks_complete_final_artifact(mutation):
    import onnx

    _, manifest, model = _bitwise_closure_fixture()
    graph = model.graph
    node = graph.node[0]
    if mutation == "hidden-bitwise":
        graph.node.append(onnx.helper.make_node("BitwiseAnd", ["x", "y"], ["dead"]))
    elif mutation == "matched-dead":
        graph.node.append(onnx.helper.make_node("BitwiseNot", ["out"], ["dead"]))
        graph.value_info.append(
            onnx.helper.make_tensor_value_info("dead", onnx.TensorProto.INT32, [3])
        )
        manifest["equations"].append(
            {
                "primitive": "not",
                "inputs": [copy.deepcopy(manifest["outputs"][0])],
                "outputs": [{"var": 3, "aval": ["int32", [3], False]}],
                "params": {},
                "effects": [],
            }
        )
        assert len(manifest["equations"]) == len(graph.node) == 2
    elif mutation == "hidden-float":
        graph.node.append(onnx.helper.make_node("Sin", ["hidden"], ["dead"]))
    elif mutation == "float-value-info":
        graph.value_info.append(
            onnx.helper.make_tensor_value_info("dead", onnx.TensorProto.FLOAT, [3])
        )
    elif mutation == "float-initializer":
        graph.initializer.append(
            onnx.numpy_helper.from_array(np.asarray([1.0], np.float32), "hidden")
        )
    elif mutation == "wrong-op":
        node.op_type = "BitwiseOr"
    elif mutation == "wrong-edge":
        node.input[1] = "x"
    elif mutation == "dtype":
        graph.input[0].type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    elif mutation == "shape":
        graph.output[0].type.tensor_type.shape.dim[0].dim_value = 4
    elif mutation == "attribute":
        node.attribute.append(onnx.helper.make_attribute("unreviewed", 1))
    elif mutation == "subgraph":
        node.attribute.append(onnx.helper.make_attribute("body", copy.deepcopy(graph)))
    elif mutation == "domain":
        node.domain = "unowned.domain"
    else:
        model.functions.append(
            onnx.helper.make_function("hidden", "Unused", [], [], [], [])
        )
    assert inventory_owner._source_bitwise_closure(manifest, model) is None


@pytest.mark.parametrize(
    "mutation", ["primitive", "params", "effects", "foreign-equation", "weak"]
)
def test_bitwise_applicability_keeps_native_rule_identity_before_closure(
    tmp_path, monkeypatch, mutation
):
    import onnx_ir as ir
    from tests.extra_tests.converter import source_precision_raw_replay as raw

    closed, _, model = _bitwise_closure_fixture()
    capture = raw.RawReplayCapture(tmp_path / mutation)
    capture.records["source_structure"]["source_binding"]["actual_trace_sha256"] = (
        raw._structure_trace_manifest(closed)[1]
    )
    body = raw._structure_body(closed)
    native = body.eqns[0]
    original_aval = body.invars[0].aval
    called = []

    class ForeignEquation:
        @property
        def primitive(self):
            called.append("primitive")
            raise AssertionError("Foreign primitive getter executed")

        @property
        def invars(self):
            called.append("invars")
            raise AssertionError("Foreign input getter executed")

    if mutation == "primitive":
        changed = native.replace(primitive=types.SimpleNamespace(name="and"))
    elif mutation == "params":
        changed = native.replace(params={"unknown": object()})
    elif mutation == "effects":
        changed = native.replace(effects=frozenset({"injected"}))
    elif mutation == "foreign-equation":
        changed = ForeignEquation()
    else:
        from jax2onnx.converter.source_structure import TensorContract

        invalid = object.__new__(raw.JaxShapedArray)
        object.__setattr__(invalid, "weak_type", 1)
        assert type(invalid) is raw.JaxShapedArray
        assert type(invalid.weak_type) is int and invalid.weak_type == 1
        body.invars[0].aval = invalid
        changed = native

        def forbidden(*args, **kwargs):
            called.append("from_aval")
            raise AssertionError("Weak type was coerced before admission")

        monkeypatch.setattr(TensorContract, "from_aval", classmethod(forbidden))
    try:
        body.eqns[0] = changed
        capture._collect_bitwise_applicability(closed, ir.from_proto(model))
    finally:
        body.eqns[0] = native
        body.invars[0].aval = original_aval
    assert (
        "closure" not in capture.records["source_structure"]["floating_applicability"]
    )
    assert called == []


def test_bitwise_applicability_binds_the_ir_actually_inspected(tmp_path, monkeypatch):
    import onnx
    import onnx_ir as ir
    from tests.extra_tests.converter import source_precision_raw_replay as raw

    closed, _, model = _bitwise_closure_fixture()
    final_ir = ir.from_proto(model)
    original = ir.to_proto
    expected = raw._structure_digest(
        original(final_ir).SerializeToString(deterministic=True)
    )
    capture = raw.RawReplayCapture(tmp_path / "same-ir")
    record = capture.records["source_structure"]
    record["source_binding"]["actual_trace_sha256"] = raw._structure_trace_manifest(
        closed
    )[1]
    record["artifact_binding"]["checked_ir_proto"] = expected
    capture._collect_bitwise_applicability(closed, final_ir)
    assert record["floating_applicability"]["closure"]["constant_count"] == 0
    assert record["floating_applicability"]["status"] == "unverified"

    def changed(value):
        proto = original(value)
        proto.graph.value_info.append(
            onnx.helper.make_tensor_value_info("hidden", onnx.TensorProto.FLOAT, [1])
        )
        return proto

    monkeypatch.setattr(ir, "to_proto", changed)
    with pytest.raises(
        raw._StructureBindingMismatch, match="observed-ir-and-checked-artifact-differ"
    ):
        capture._collect_bitwise_applicability(closed, final_ir)


@pytest.mark.parametrize(
    "criterion",
    [
        None,
        {"mode": "exact"},
        {"mode": "exact", "applicability": "all-physical-discrete"},
        {"mode": "tolerant", "applicability": "all-logical-discrete"},
        {"mode": "exact", "applicability": "all-logical-discrete", "reviewed": True},
    ],
)
def test_conditional_discrete_requires_the_exact_reviewed_adapter_choice(
    tmp_path, monkeypatch, criterion
):
    scope = _discrete_protocol_scope(
        {
            "testcase": "not-an-approved-discrete-choice",
            "callable": lambda x: x,
            "input_values": [np.asarray([1, 2], dtype=np.int32)],
        }
    )
    adapter = scope[3]["rows"]["plugin::synthetic::0"]
    if criterion is None:
        adapter.pop("criteria")
    else:
        adapter["criteria"] = criterion

    def forbidden(*args, **kwargs):
        pytest.fail("An unreviewed conditional criterion must not create ORT")

    monkeypatch.setattr(onnxruntime, "InferenceSession", forbidden)
    function = _function(
        scope,
        _descriptors(scope)[1],
        tmp_path,
        runtime="both",
        expected={"runtime": "unverified", "numerics": "unverified"},
    )
    function()
    record = function._source_cell.history[-1]
    assert scope[1].rows[0].criteria is None
    scientific = inventory_owner.source_metadata_numerical_checks(
        scope[1].rows[0], asdict(record)
    )
    assert scientific["reviewed_criteria"]["status"] == "unverified"


@pytest.mark.parametrize("changed_output", ["float", "mixed", "complex"])
def test_conditional_discrete_collects_all_alternate_roles_before_rejecting_output(
    tmp_path, monkeypatch, changed_output
):
    from tests.extra_tests.converter import source_precision_raw_replay as raw

    original = raw.RawReplayCapture.invoke

    def invoke(self, role, call):
        value = original(self, role, call)
        if role in {
            "metadata-runtime-reference-0-eager",
            "metadata-runtime-reference-0-fresh-jit",
            "metadata-runtime-reference-0-warm-jit",
        }:
            if changed_output == "mixed":
                value = (value, np.asarray(value, dtype=np.float32))
            else:
                value = np.asarray(
                    value,
                    dtype=np.complex64 if changed_output == "complex" else np.float32,
                )
            self.native_values[role] = value
        return value

    def forbidden(*args, **kwargs):
        pytest.fail("An inapplicable alternate output must not execute ORT")

    monkeypatch.setattr(raw.RawReplayCapture, "invoke", invoke)
    monkeypatch.setattr(onnxruntime, "InferenceSession", forbidden)
    scope, function = _discrete_numeric_function(
        tmp_path,
        expected={
            "source_reference": "failed",
            "runtime": "unverified",
            "numerics": "failed",
        },
    )
    function()
    record = function._source_cell.history[-1]
    capture = record.replay_capture
    for label in ("", "metadata-runtime-reference-0"):
        sample = capture["native_protocol"]["samples"][label]
        assert set(sample["roles"]) == {"eager", "fresh-jit", "warm-jit"}
        assert sample["wrapper_calls"] == 2
        for role in sample["roles"]:
            name = (label + "-" if label else "") + role
            assert sample["roles"][role]["outcome"]["status"] == "returned"
            assert capture[name]["status"] == "captured"
    assert not capture["metadata_runtime"]["modes"]
    assert record.source_reference is CheckStatus.FAILED
    assert record.numerics is CheckStatus.FAILED
    assert scope[1].rows[0].criteria is None
    scientific = inventory_owner.source_metadata_numerical_checks(
        scope[1].rows[0], asdict(record)
    )
    assert scientific["reviewed_criteria"]["status"] == "unverified"
    for name in ("component_nan_masks", "zero_signs"):
        assert scientific[name].get("applicability") != "not-applicable"


@pytest.mark.parametrize(
    "mutation",
    [
        "float",
        "mixed",
        "complex",
        "dtype",
        "shape",
        "weakness",
        "typed-key",
        "opaque",
        "empty-tree",
        "missing-warm",
        "wrong-warm",
    ],
)
def test_conditional_discrete_resolution_requires_logical_native3_proof(
    tmp_path, monkeypatch, mutation
):
    from tests.extra_tests.converter import source_precision_raw_replay as raw

    original = raw._resolve_discrete_criteria
    attempts, key_dtype_calls = [], []
    key = jax.random.key(0) if mutation == "typed-key" else None
    with jax.enable_x64(False):
        weak = jnp.asarray(1) if mutation == "weakness" else None

    def resolve(prepared, capture, *, adapter, references, labels):
        assert prepared.criteria is None
        assert list(labels) == ["", "metadata-runtime-reference-0"]
        assert len(references) == 2
        for label in labels:
            assert set(capture.native_samples[label]["roles"]) == {
                "eager",
                "fresh-jit",
                "warm-jit",
            }
            for entry in capture.native_samples[label]["roles"].values():
                assert capture.records[entry["capture_role"]]["status"] == "captured"
        label = labels[-1]
        sample = capture.native_samples[label]
        logical = capture.native_sample_values[label]["original_leaves"]
        if mutation == "missing-warm":
            sample["roles"].pop("warm-jit")
        elif mutation == "wrong-warm":
            sample["roles"]["warm-jit"]["jit_wrapper_identity"] = (
                sample["wrapper_identity"] + 1
            )
        else:
            for role, leaves in list(logical.items()):
                if mutation == "float":
                    logical[role] = (np.asarray(leaves[0], dtype=np.float32),)
                elif mutation == "mixed":
                    logical[role] = (*leaves, np.asarray(leaves[0], dtype=np.float32))
                elif mutation == "complex":
                    logical[role] = (np.asarray(leaves[0], dtype=np.complex64),)
                elif mutation == "dtype":
                    logical[role] = (np.asarray(leaves[0], dtype=np.int32),)
                elif mutation == "shape":
                    logical[role] = (leaves[0].reshape(3, 1),)
                elif mutation == "weakness":
                    assert weak.weak_type and weak.dtype == np.dtype(np.int32)
                    logical[role] = (weak,)
                elif mutation == "typed-key":
                    logical[role] = (key,)
                elif mutation == "opaque":
                    logical[role] = (object(),)
                else:
                    logical[role] = ()
                    sample["roles"][role]["tree"] = "PyTreeDef(())"
        original_dtype = np.dtype

        def dtype(value, *args, **kwargs):
            if key is not None and value is key.dtype:
                key_dtype_calls.append(value)
            return original_dtype(value, *args, **kwargs)

        with monkeypatch.context() as patch:
            if key is not None:
                patch.setattr(raw.np, "dtype", dtype)
            criterion = original(
                prepared, capture, adapter=adapter, references=references, labels=labels
            )
        attempts.append(
            (criterion, copy.deepcopy(capture.records["criteria_resolution"]))
        )
        return criterion

    def forbidden(*args, **kwargs):
        pytest.fail("Incomplete logical Native3 proof must not execute ORT")

    monkeypatch.setattr(raw, "_resolve_discrete_criteria", resolve)
    monkeypatch.setattr(onnxruntime, "InferenceSession", forbidden)
    scope, function = _discrete_numeric_function(
        tmp_path,
        dtype=np.int32 if mutation == "weakness" else np.uint32,
        scalar=mutation == "weakness",
        expected={"runtime": "unverified", "numerics": "unverified"},
    )
    function()
    assert len(attempts) == 1
    criterion, resolution = attempts[0]
    assert criterion is None and resolution["status"] == "unverified"
    assert resolution["criteria"] is None
    assert not key_dtype_calls
    assert scope[1].rows[0].criteria is None
    capture = function._source_cell.history[-1].replay_capture
    assert capture["metadata-runtime-reference-0-eager"]["properties"][0]["dtype"] == (
        "int32" if mutation == "weakness" else "uint32"
    )
    assert not capture["metadata_runtime"]["modes"]


def test_conditional_discrete_accepts_empty_shaped_leaves(tmp_path):
    scope = _discrete_protocol_scope(
        {
            "testcase": "empty-discrete-leaf",
            "callable": lambda x: x[:0],
            "input_values": [np.asarray([1, 2, 3], dtype=np.int32)],
            "runtime_input_values": [[np.asarray([3, 1, 2], dtype=np.int32)]],
        }
    )
    function = _function(
        scope,
        _descriptors(scope)[1],
        tmp_path,
        runtime="both",
        expected={"runtime": "validated", "numerics": "validated"},
    )
    function()
    record = function._source_cell.history[-1]
    capture = record.replay_capture
    assert capture["criteria_resolution"]["status"] == "validated"
    for label in ("", "metadata-runtime-reference-0"):
        role = (label + "-" if label else "") + "eager"
        assert capture[role]["properties"] == [
            {"dtype": "int32", "shape": [0], "weak_type": False}
        ]
    assert set(capture["metadata_runtime"]["modes"]) == {"disabled", "all"}


def test_conditional_discrete_source_failure_cannot_resolve_a_criterion(
    tmp_path, monkeypatch
):
    from tests.extra_tests.converter import source_precision_raw_replay as raw

    original = raw.RawReplayCapture.invoke

    def invoke(self, role, call):
        if role == "metadata-runtime-reference-0-fresh-jit":

            def raised():
                call()
                raise RuntimeError("Conditional discrete original role failed")

            return original(self, role, raised)
        return original(self, role, call)

    def forbidden(*args, **kwargs):
        pytest.fail("An original native failure must not execute ORT")

    monkeypatch.setattr(raw.RawReplayCapture, "invoke", invoke)
    monkeypatch.setattr(onnxruntime, "InferenceSession", forbidden)
    scope, function = _discrete_numeric_function(
        tmp_path,
        expected={
            "source_reference": "failed",
            "runtime": "unverified",
            "numerics": "failed",
        },
    )
    function()
    record = function._source_cell.history[-1]
    capture = record.replay_capture
    roles = capture["native_protocol"]["samples"]["metadata-runtime-reference-0"][
        "roles"
    ]
    assert set(roles) == {"eager", "fresh-jit", "warm-jit"}
    assert roles["fresh-jit"]["outcome"]["status"] == "raised"
    assert roles["eager"]["outcome"]["status"] == "returned"
    assert roles["warm-jit"]["outcome"]["status"] == "returned"
    assert not capture["metadata_runtime"]["modes"]
    assert capture.get("criteria_resolution", {}).get("status") != "validated"
    scientific = inventory_owner.source_metadata_numerical_checks(
        scope[1].rows[0], asdict(record)
    )
    assert scientific["native_three_role_agreement"]["status"] == "failed"
    assert scientific["reviewed_criteria"]["status"] == "unverified"


def test_conditional_discrete_projection_rejects_forged_resolution_bindings(tmp_path):
    scope, function = _discrete_numeric_function(tmp_path)
    function()
    record = asdict(function._source_cell.history[-1])
    for mutation in (
        "criteria",
        "runtime-criteria",
        "choice",
        "contract",
        "scenario",
        "adapter",
        "trace",
        "applicability",
        "logical-dtype",
        "empty-proof",
        "sample-label",
    ):
        broken = copy.deepcopy(record)
        capture = broken["replay_capture"]
        resolution = capture["criteria_resolution"]
        if mutation == "criteria":
            resolution["criteria"] = {
                "mode": "tolerant",
                "rtol": 1e9,
                "atol": 1e9,
                "equal_nan": True,
            }
        elif mutation == "runtime-criteria":
            criterion = {
                "mode": "tolerant",
                "rtol": 1e9,
                "atol": 1e9,
                "equal_nan": True,
            }
            runtime = capture["metadata_runtime"]
            runtime["criteria"] = copy.deepcopy(criterion)
            for identity in runtime["sample_identities"]:
                identity["criteria"] = copy.deepcopy(criterion)
            for mode in runtime["modes"].values():
                for identity in mode["sample_identities"]:
                    identity["criteria"] = copy.deepcopy(criterion)
                for sample in mode["samples"]:
                    sample["criteria"] = copy.deepcopy(criterion)
        elif mutation == "choice":
            resolution["choice"]["applicability"] = "all-physical-discrete"
        elif mutation == "contract":
            resolution["contract_sha256"] = "0" * 64
        elif mutation == "scenario":
            resolution["protocol_binding"]["scenario_sha256"] = "0" * 64
        elif mutation == "adapter":
            resolution["adapter_sha256"] = "0" * 64
        elif mutation == "trace":
            resolution["trace_sha256"] = "0" * 64
        elif mutation == "applicability":
            resolution["output_applicability"]["zero_signs"] = "applicable"
        elif mutation == "logical-dtype":
            resolution["samples"][-1]["roles"]["warm-jit"]["properties"][0][
                "dtype"
            ] = "float32"
        elif mutation == "empty-proof":
            resolution["samples"] = []
        else:
            resolution["samples"][-1]["label"] = resolution["samples"][0]["label"]
        # Bind the forged document bytes as well: rejection must come from the
        # original source/adapter/trace/role proof, not just a stale file digest.
        document = tmp_path / ("forged-" + mutation + ".json")
        payload = json.dumps(resolution, sort_keys=True, separators=(",", ":")) + "\n"
        document.write_text(payload)
        capture["criteria-resolution"] = {
            "status": "captured",
            "path": str(document),
            "sha256": hashlib.sha256(document.read_bytes()).hexdigest(),
        }
        with pytest.raises(ValueError):
            inventory_owner.source_metadata_numerical_checks(scope[1].rows[0], broken)

    for missing in ("criteria_resolution", "criteria-resolution", "warm-jit"):
        broken = copy.deepcopy(record)
        capture = broken["replay_capture"]
        capture.pop(missing)
        try:
            checks = inventory_owner.source_metadata_numerical_checks(
                scope[1].rows[0], broken
            )
        except ValueError:
            continue
        assert checks["reviewed_criteria"]["status"] == "unverified"
        for name in ("component_nan_masks", "zero_signs"):
            assert checks[name].get("applicability") != "not-applicable"


@pytest.mark.parametrize(
    "choice,expected_runtime",
    [
        ("conditional", "both"),
        ("absent", "all"),
        ("different-applicability", "all"),
        ("no-native-protocol", "all"),
    ],
)
def test_metadata_cli_selects_both_only_for_the_reviewed_discrete_path(
    original_scope, tmp_path, monkeypatch, choice, expected_runtime
):
    payload = inventory_owner.CURRENT_BASELINE.read_bytes()
    inventory = json.loads(payload)
    plan = probe.build_probe_plan(
        inventory, baseline_sha256=hashlib.sha256(payload).hexdigest()
    )
    row = next(
        row
        for row in plan.rows
        if (row.origin, row.ordinal) == ("plugin::jax.numpy.bitwise_and", 0)
    )
    adapter = {
        "contract_sha256": row.contract_sha256,
        "native_protocol": {"mode": "stateless-three-role-v1", "reviewed": True},
        "criteria": {"mode": "exact", "applicability": "all-logical-discrete"},
    }
    if choice == "absent":
        adapter.pop("criteria")
    elif choice == "different-applicability":
        adapter["criteria"]["applicability"] = "all-physical-discrete"
    elif choice == "no-native-protocol":
        adapter.pop("native_protocol")
    adapter_path = tmp_path / "adapters.json"
    adapter_path.write_text(
        json.dumps(
            {
                "inventory_sha256": plan.baseline_sha256,
                "rows": {"plugin::jax.numpy.bitwise_and::0": adapter},
            }
        )
    )
    options = {
        "source_metadata_baseline": None,
        "source_metadata_case": ["plugin::jax.numpy.bitwise_and::0::raw-x64-off"],
        "source_metadata_adapters": str(adapter_path),
    }
    state = {
        "configured": False,
        "report": tmp_path / "acceptance.json",
        "baseline_payload": payload,
        "inventory": inventory,
        "observed_inventory": inventory,
        "plan": plan,
        "epoch": _EPOCH,
        "input_bindings": {},
    }
    calls = []

    def make_function(*args, **kwargs):
        calls.append(kwargs["runtime_optimization"])

        def function():
            pytest.fail("CLI admission must not execute the selected model")

        return function

    monkeypatch.setattr(
        inventory_owner, "raw_registrations", lambda: (original_scope[2], ())
    )
    monkeypatch.setattr(t_generator, "make_source_test_function", make_function)
    _forbid_legacy_and_native(monkeypatch)
    config = types.SimpleNamespace(getoption=lambda name: options.get(name))
    configure_source_metadata_acceptance(config, type("Owner", (), {}), state)
    assert calls == [expected_runtime]
    assert state["configured"] is True
    assert state["identities"] == ((row.origin, row.ordinal, "raw-x64-off"),)


def _family_discrete_protocol(tmp_path, *, supplied_jit=False):
    captured = np.asarray([[7, 3], [1, 0]], dtype=np.int32)

    def original(x, y):
        left = jax.lax.reshape(x, (2, 2))
        promoted = jax.lax.convert_element_type(y, np.int32)
        right = jax.lax.broadcast_in_dim(promoted, (2, 2), (1,))
        bits = jax.lax.bitwise_xor(jax.lax.bitwise_or(left, right), captured)
        result = jax.lax.bitwise_and(jax.lax.bitwise_not(bits), np.int32(3))
        return jax.lax.transpose(result, (1, 0))

    target = jax.jit(original) if supplied_jit else original
    scope = _discrete_protocol_scope(
        {
            "testcase": "composed-discrete-family",
            "callable": target,
            "input_values": [
                np.asarray([1, 2, 3, 4], np.int32),
                np.asarray([True, False]),
            ],
            "runtime_input_values": [
                [np.asarray([5, 6, 7, 8], np.int32), np.asarray([False, True])]
            ],
        }
    )
    if supplied_jit:
        scope[3]["rows"]["plugin::synthetic::0"]["capture_original_object"] = True
    descriptor = _descriptors(
        scope, selection=[("plugin::synthetic", 0, "raw-x64-on")]
    )[0]
    function = _function(
        scope,
        descriptor,
        tmp_path,
        runtime="both",
        expected={"runtime": "validated", "numerics": "validated"},
    )
    return scope, function, captured


@pytest.mark.parametrize("supplied_jit", [False, True])
def test_discrete_family_complete_constants_shapes_and_plain_jit(
    tmp_path, monkeypatch, supplied_jit
):
    from tests.extra_tests.converter import source_precision_raw_replay as raw

    targets, witnesses, invoked_targets = [], [], []
    original_bind = raw.RawReplayCapture.bind_target
    original_sample = raw.RawReplayCapture.native_sample

    def nested_code(parent, name):
        matches = [
            code
            for code in parent.co_consts
            if type(code) is types.CodeType and code.co_name == name
        ]
        assert len(matches) == 1
        return matches[0]

    primary_code = nested_code(raw.prepare_raw_source_case.__code__, "original")
    alternate_code = nested_code(
        nested_code(probe._raw_metadata_runtime_protocol.__code__, "run_reference"),
        "original",
    )

    def bind(capture, target):
        targets.append(target)
        result = original_bind(capture, target)
        witnesses.append(capture.target_witness)
        return result

    def sample(capture, original, arguments, **kwargs):
        assert type(original) is types.FunctionType
        assert original.__code__ is (
            primary_code if kwargs["label"] == "" else alternate_code
        )
        cells = dict(
            zip(original.__code__.co_freevars, original.__closure__, strict=True)
        )
        primary = kwargs["label"] == ""
        assert set(cells) == (
            {"arguments", "names", "target"}
            if primary
            else {"fn", "names", "positional"}
        )
        owner = (
            raw.prepare_raw_source_case
            if primary
            else probe._raw_metadata_runtime_protocol
        )
        assert (
            original.__globals__ is owner.__globals__
            and original.__builtins__ is owner.__builtins__
        )
        target = cells["target" if primary else "fn"].cell_contents
        positional = cells["arguments" if primary else "positional"].cell_contents
        assert target is supplied is capture.target
        assert cells["names"].cell_contents == ()
        assert len(arguments) == len(positional) == 2
        assert all(
            actual is bound for actual, bound in zip(arguments, positional, strict=True)
        )
        testcase = scope[2][0][2]["testcases"][0]
        expected_inputs = (
            testcase["input_values"] if primary else testcase["runtime_input_values"][0]
        )
        for actual, expected in zip(arguments, expected_inputs, strict=True):
            actual, expected = np.asarray(actual), np.asarray(expected)
            assert actual.dtype == expected.dtype and actual.shape == expected.shape
            assert actual.tobytes() == expected.tobytes()
        invoked_targets.append(target)
        return original_sample(capture, original, arguments, **kwargs)

    monkeypatch.setattr(raw.RawReplayCapture, "bind_target", bind)
    monkeypatch.setattr(raw.RawReplayCapture, "native_sample", sample)
    scope, function, captured = _family_discrete_protocol(
        tmp_path, supplied_jit=supplied_jit
    )
    supplied = scope[2][0][2]["testcases"][0]["callable"]
    function()
    record = function._source_cell.history[-1]
    capture = record.replay_capture
    proof = capture["source_structure"]["floating_applicability"]
    assert record.source_reference is CheckStatus.VALIDATED
    assert record.structure is CheckStatus.VALIDATED
    assert record.lowering_audit is CheckStatus.VALIDATED
    assert proof["status"] == "validated" and proof["closure"]["function_count"] == 0
    assert json.loads(json.dumps(proof["closure"]["source_storage"])) == json.loads(
        json.dumps(capture["source_structure"]["source_binding"]["prepared_storage"])
    )
    assert any(
        item["logical"]["shape"] == [2, 2]
        for item in proof["closure"]["source_storage"]
    )
    assert any(
        item["logical"]["shape"] == [] for item in proof["closure"]["source_storage"]
    )
    assert proof["closure"]["constant_count"] >= 2
    assert targets == [supplied] and targets[0] is supplied
    assert len(invoked_targets) == 2 and all(
        target is supplied for target in invoked_targets
    )
    if supplied_jit:
        original = vars(supplied)["_fun"]
        assert vars(supplied)["__wrapped__"] is original
        assert any(cell.cell_contents is captured for cell in original.__closure__)
        witness = witnesses[0]
        assert witness is not None
        for actual in (supplied, original, captured):
            assert any(value is actual for _, value in witness.edges)
        assert any(value is original for value in witness.native_references)
        assert any(value is captured for value in witness.native_references)
    expected = np.bitwise_and(
        np.bitwise_not(
            np.bitwise_xor(
                np.bitwise_or(
                    np.asarray([[1, 2], [3, 4]], np.int32), np.asarray([1, 0], np.int32)
                ),
                captured,
            )
        ),
        np.int32(3),
    ).T
    for role in [
        "eager",
        "fresh-jit",
        "warm-jit",
        "metadata-ort-disabled-0",
        "metadata-ort-all-0",
    ]:
        with np.load(capture[role]["path"], allow_pickle=False) as actual:
            assert len(actual.files) == 1
            np.testing.assert_array_equal(actual[actual.files[0]], expected)
    checks = inventory_owner.source_metadata_numerical_checks(
        scope[1].rows[0], asdict(record)
    )
    for name in ("operation_rounding", "required_internal_fp64"):
        assert (
            checks[name]["status"] == "validated"
            and checks[name]["applicability"] == "not-applicable"
        )
    assert checks["same_artifact_and_feeds_both_modes"]["status"] == "validated"
    retained = _retained_stage_record(function)
    assert retained["replay_capture"]["source_structure"][
        "floating_applicability"
    ] == json.loads(json.dumps(proof))


@pytest.mark.parametrize("mutation", ["word", "equal-array"])
def test_original_jit_owned_array_mutation_stops_before_next_role(
    tmp_path, monkeypatch, mutation
):
    from tests.extra_tests.converter import source_precision_raw_replay as raw
    from tests.extra_tests.converter.source_precision_support import SourceSettings

    captured = np.asarray([[7, 3], [1, 0]], np.int32)

    def original(x):
        return jax.lax.bitwise_xor(x, captured)

    supplied = jax.jit(original)
    capture = raw.RawReplayCapture(tmp_path / mutation)
    recipe = raw._capture_recipe(supplied)
    capture.bind_target(supplied)
    assert capture.target is supplied and capture.check_target("unchanged")
    cell = original.__closure__[0]
    assert cell.cell_contents is captured
    if mutation == "word":
        captured[0, 0] ^= np.int32(1)
        assert cell.cell_contents is captured
    else:
        replacement = np.array(captured, copy=True)
        assert (
            replacement is not captured and replacement.tobytes() == captured.tobytes()
        )
        monkeypatch.setattr(cell, "cell_contents", replacement)
    attempted = []

    def forbidden(*args, **kwargs):
        attempted.append(True)
        pytest.fail("Changed owned array must not reach a native role")

    monkeypatch.setattr(capture, "invoke", forbidden)
    with pytest.raises(raw.SourcePreparationError, match="state is not preserved"):
        capture.native_sample(
            supplied,
            (np.asarray([[1, 2], [3, 4]], np.int32),),
            label="",
            settings=SourceSettings.observe(),
            metadata={},
            three_roles=True,
        )
    assert attempted == [] and capture.records["native_outcomes"] == {}
    assert raw._capture_recipe(supplied) != recipe


def test_original_jit_unsupported_owned_capture_still_rejects_before_native(
    tmp_path, monkeypatch
):
    from tests.extra_tests.converter import source_precision_raw_replay as raw

    opaque = object()

    def original(x):
        return x if opaque is not None else x

    supplied = jax.jit(original)
    attempted = []

    def forbidden(*args, **kwargs):
        attempted.append(True)
        pytest.fail("Unsupported owned state must not reach native execution")

    monkeypatch.setattr(raw.RawReplayCapture, "invoke", forbidden)
    with pytest.raises(raw.SourcePreparationError):
        raw._capture_recipe(supplied)
    capture = raw.RawReplayCapture(tmp_path / "unsupported")
    with pytest.raises(raw.SourcePreparationError):
        capture.bind_target(supplied)
    assert attempted == [] and capture.records["native_outcomes"] == {}


def test_generic_native_capture_graph_still_rejects_user_array():
    captured = np.asarray([[7, 3], [1, 0]], np.int32)
    with pytest.raises(ValueError, match="unsupported native callable state type"):
        compat._NativeCaptureGraph().take(captured)


@pytest.mark.parametrize("dtype", [np.int32, np.uint32])
@pytest.mark.parametrize("caller", ["raw-x64-off", "raw-x64-on"])
def test_discrete_family_high_words_retain_exact_native_and_target_outputs(
    tmp_path, dtype, caller
):
    if dtype is np.int32:
        left = (-2147483648, 2147483647, -1, 0)
        right = (2147483647, -2147483648, 0, -1)
    else:
        left = (0, 2147483648, 4294967295, 1)
        right = (4294967295, 1, 2147483648, 0)
    samples = ((left, right), (right, left))

    def original(x, y):
        return (
            jax.lax.bitwise_and(x, y),
            jax.lax.bitwise_or(x, y),
            jax.lax.bitwise_xor(x, y),
            jax.lax.bitwise_not(x),
        )

    scope = _discrete_protocol_scope(
        {
            "testcase": "discrete-family-high-words",
            "callable": original,
            "input_values": [np.asarray(value, dtype=dtype) for value in samples[0]],
            "runtime_input_values": [
                [np.asarray(value, dtype=dtype) for value in samples[1]]
            ],
        }
    )
    descriptor = _descriptors(scope, selection=[("plugin::synthetic", 0, caller)])[0]
    function = _function(
        scope,
        descriptor,
        tmp_path,
        runtime="both",
        expected={"runtime": "validated", "numerics": "validated"},
    )
    function()
    record = function._source_cell.history[-1]
    capture = record.replay_capture
    assert record.source_reference is CheckStatus.VALIDATED
    assert record.structure is CheckStatus.VALIDATED
    assert record.lowering_audit is CheckStatus.VALIDATED
    for index, (x, y) in enumerate(samples):
        # Python integer words form the independent fixed-width oracle. No
        # NumPy/JAX bitwise operation supplies the expected target values.
        mask = (1 << 32) - 1
        xwords = [word & mask for word in x]
        ywords = [word & mask for word in y]
        expected_words = (
            [a & b for a, b in zip(xwords, ywords)],
            [a | b for a, b in zip(xwords, ywords)],
            [a ^ b for a, b in zip(xwords, ywords)],
            [(~a) & mask for a in xwords],
        )
        prefix = "" if index == 0 else f"metadata-runtime-reference-{index - 1}-"
        roles = [prefix + role for role in ("eager", "fresh-jit", "warm-jit")]
        roles.extend(f"metadata-ort-{mode}-{index}" for mode in ("disabled", "all"))
        for role in roles:
            assert capture[role]["status"] == "captured"
            with np.load(capture[role]["path"], allow_pickle=False) as actual:
                assert actual.files == [f"value_{leaf}" for leaf in range(4)]
                for leaf, words in enumerate(expected_words):
                    array = actual[f"value_{leaf}"]
                    assert array.dtype == np.dtype(dtype) and array.shape == (4,)
                    assert array.view(np.uint32).tolist() == words
    checks = inventory_owner.source_metadata_numerical_checks(
        scope[1].rows[0], asdict(record)
    )
    for name in ("operation_rounding", "required_internal_fp64"):
        assert checks[name]["status"] == "validated"
        assert checks[name]["applicability"] == "not-applicable"
    assert checks["native_three_role_agreement"]["status"] == "validated"
    assert checks["same_artifact_and_feeds_both_modes"]["status"] == "validated"
    _retained_stage_record(function)


def test_discrete_family_full_artifact_rejects_unclaimed_storage_and_wrong_words(
    tmp_path,
):
    import onnx

    scope, function, _ = _family_discrete_protocol(tmp_path)
    function()
    record = asdict(function._source_cell.history[-1])
    capture = record["replay_capture"]
    manifest = json.loads(record["original_trace_json"])
    original = onnx.load(capture["artifact"]["path"], load_external_data=False)
    accepted = inventory_owner._source_bitwise_closure(manifest, original)
    assert accepted is not None and accepted["source_storage"]
    concats = [
        index
        for index, node in enumerate(original.graph.node)
        if node.op_type == "Concat"
    ]
    assert len(concats) == 2
    initializers = {value.name: value for value in original.graph.initializer}
    actual_vectors = []
    for index in concats:
        node = original.graph.node[index]
        pieces = [onnx.numpy_helper.to_array(initializers[name]) for name in node.input]
        assert all(
            value.dtype == np.dtype("int64") and value.shape == (1,) for value in pieces
        )
        actual_vectors.append(tuple(int(value[0]) for value in pieces))
        for mutation in (
            "axis",
            "axis-type",
            "output-type",
            "output-rank",
            "arity",
            "piece-type",
            "piece-rank",
            "word",
            "runtime",
            "unknown",
            "nested",
        ):
            model = copy.deepcopy(original)
            concat = model.graph.node[index]
            piece_name = concat.input[0]
            piece = next(
                value for value in model.graph.initializer if value.name == piece_name
            )
            word = int(onnx.numpy_helper.to_array(piece)[0])
            if mutation == "axis":
                concat.attribute[0].i = 1
            elif mutation == "axis-type":
                # FLOAT zero compares equal to integer zero in Python; the
                # actual ONNX attribute type is part of this finite contract.
                concat.attribute[0].CopyFrom(onnx.helper.make_attribute("axis", 0.0))
            elif mutation in {"output-type", "output-rank"}:
                output = next(
                    value
                    for value in model.graph.value_info
                    if value.name == concat.output[0]
                )
                output.CopyFrom(
                    onnx.helper.make_tensor_value_info(
                        output.name,
                        (
                            onnx.TensorProto.INT32
                            if mutation == "output-type"
                            else onnx.TensorProto.INT64
                        ),
                        [2] if mutation == "output-type" else [1, 2],
                    )
                )
            elif mutation == "arity":
                del concat.input[-1]
            elif mutation in {"piece-type", "piece-rank", "word"}:
                dtype = (
                    onnx.TensorProto.INT32
                    if mutation == "piece-type"
                    else onnx.TensorProto.INT64
                )
                shape = [] if mutation == "piece-rank" else [1]
                piece.CopyFrom(
                    onnx.helper.make_tensor(
                        piece_name, dtype, shape, [word + (mutation == "word")]
                    )
                )
                declared = next(
                    (
                        value
                        for value in model.graph.value_info
                        if value.name == piece_name
                    ),
                    None,
                )
                if declared is not None:
                    declared.CopyFrom(
                        onnx.helper.make_tensor_value_info(piece_name, dtype, shape)
                    )
            else:
                edge = "unowned-shape-piece"
                if mutation == "runtime":
                    # No extra model input or arity mismatch: these runtime
                    # data edges remain disallowed even with exact INT64[1].
                    indices = onnx.helper.make_tensor(
                        "unowned-shape-index", onnx.TensorProto.INT64, [1], [word - 1]
                    )
                    model.graph.initializer.append(indices)
                    producers = [
                        onnx.helper.make_node(
                            "Gather",
                            [model.graph.input[0].name, indices.name],
                            ["runtime-shape-word"],
                            axis=0,
                        ),
                        onnx.helper.make_node(
                            "Cast",
                            ["runtime-shape-word"],
                            [edge],
                            to=onnx.TensorProto.INT64,
                        ),
                    ]
                    model.graph.value_info.append(
                        onnx.helper.make_tensor_value_info(
                            "runtime-shape-word", onnx.TensorProto.INT32, [1]
                        )
                    )
                else:
                    producers = [
                        onnx.helper.make_node(
                            (
                                "Concat"
                                if mutation == "nested"
                                else "UnreviewedShapeProducer"
                            ),
                            [piece_name],
                            [edge],
                            **({"axis": 0} if mutation == "nested" else {}),
                        )
                    ]
                model.graph.value_info.append(
                    onnx.helper.make_tensor_value_info(
                        edge, onnx.TensorProto.INT64, [1]
                    )
                )
                concat.input[0] = edge
                for offset, producer in enumerate(producers):
                    model.graph.node.insert(index + offset, producer)
            assert inventory_owner._source_bitwise_closure(manifest, model) is None, (
                index,
                mutation,
            )
    assert sorted(actual_vectors) == [(1, 2), (2, 2)]
    unclaimed = copy.deepcopy(original)
    unclaimed.graph.node.append(
        onnx.helper.make_node(
            "Identity", [unclaimed.graph.input[0].name], ["unclaimed-discrete-node"]
        )
    )
    unclaimed.graph.value_info.append(
        onnx.helper.make_tensor_value_info(
            "unclaimed-discrete-node", onnx.TensorProto.INT32, [4]
        )
    )
    assert inventory_owner._source_bitwise_closure(manifest, unclaimed) is None
    for dtype in (onnx.TensorProto.INT32, onnx.TensorProto.FLOAT):
        model = copy.deepcopy(original)
        model.graph.initializer.append(
            onnx.helper.make_tensor("unclaimed-initializer", dtype, [1], [0])
        )
        assert inventory_owner._source_bitwise_closure(manifest, model) is None
    storage = copy.deepcopy(manifest)

    def alter_constant(body):
        if body["const_payloads"]:
            body["const_payloads"][0]["sha256"] = "0" * 64
            return True
        for eqn in body["equations"]:
            if eqn["primitive"] == "jit" and alter_constant(eqn["params"]["jaxpr"]):
                return True
        return False

    assert alter_constant(storage)
    assert inventory_owner._source_bitwise_closure(storage, original) is None
    for mutation in ("scope", "word", "weakness"):
        forged = copy.deepcopy(record)
        witness = forged["replay_capture"]["source_structure"]
        for key in ("prepared_storage", "actual_storage", "fresh_prepared_storage"):
            item = witness["source_binding"][key][0]
            if mutation == "scope":
                item["role"] = "unowned/constant:0"
            elif mutation == "word":
                item["logical"]["sha256"] = "0" * 64
                item["physical"]["sha256"] = "0" * 64
            else:
                item["weak_type"] = not item["weak_type"]
        proof = witness["floating_applicability"]
        proof["source_structure_sha256"] = inventory_owner._contract_hash(
            {k: v for k, v in witness.items() if k != "floating_applicability"}
        )
        proof["closure"]["source_storage"] = copy.deepcopy(
            witness["source_binding"]["prepared_storage"]
        )
        path = tmp_path / ("forged-family-" + mutation + ".json")
        path.write_text(json.dumps(proof, sort_keys=True, separators=(",", ":")) + "\n")
        forged["replay_capture"]["source-bitwise-applicability"] = {
            "status": "captured",
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        with pytest.raises(ValueError):
            inventory_owner.source_metadata_numerical_checks(scope[1].rows[0], forged)


@pytest.mark.parametrize("name", ["bitwise_or", "bitwise_xor"])
def test_discrete_family_native_owners_have_complete_reducer_binding(name):
    target = vars(compat._native_numpy)[name]
    assert target is vars(compat._native_ufuncs)[name] is vars(jnp)[name]
    selected = compat.selected_native_capture(target)
    assert selected is not None and type(target) is compat._NATIVE_CAPTURE_UFUNC_TYPE
    nodes = selected["state"]["nodes"]
    props = vars(target)["_ufunc__static_props"]
    assert set(props) == compat._NATIVE_CAPTURE_STATIC_FIELDS
    assert (props["nin"], props["nout"], props["nargs"], props["identity"]) == (
        2,
        1,
        2,
        0,
    )
    assert props["func"] is props["call"]
    assert props["at"] is props["accumulate"] is props["reduceat"] is None
    reducer = props["reduce"]
    function = vars(reducer)["_fun"]
    owner = sys.modules[function.__module__]
    assert vars(owner)["_reduce_" + name] is reducer
    for value in (
        target,
        props,
        props["func"],
        reducer,
        function,
        vars(reducer)["_jit_info"],
    ):
        assert str(id(value)) in nodes
    assert any(
        module is owner and key == "_reduce_" + name and original is reducer
        for module, key, original in compat._NATIVE_CAPTURE_BINDINGS[id(target)]
    )


@pytest.mark.parametrize("name", ["bitwise_or", "bitwise_xor"])
@pytest.mark.parametrize(
    "mutation",
    [
        "func",
        "call",
        "reduce",
        "nin",
        "nout",
        "nargs",
        "identity",
        "accumulate",
        "at",
        "reduceat",
        "public",
        "defining",
        "reducer-alias",
        "reducer-code",
        "reducer-config",
        "unknown",
        "equal-dict",
        "foreign",
    ],
)
def test_discrete_family_native_owner_state_mutations_stop_before_call(
    tmp_path, monkeypatch, name, mutation
):
    from tests.extra_tests.converter import source_precision_raw_replay as raw

    target = vars(compat._native_numpy)[name]
    assert compat.selected_native_capture(target) is not None
    props = vars(target)["_ufunc__static_props"]
    capture = raw.RawReplayCapture(tmp_path / name / mutation)
    capture.bind_target(target)
    invoked = []

    def replacement(*args, **kwargs):
        invoked.append(True)
        pytest.fail("Changed native state must never execute")

    if mutation in compat._NATIVE_CAPTURE_STATIC_FIELDS:
        values = {"nin": 3, "nout": 2, "nargs": 3, "identity": -1}
        changed = values.get(mutation, replacement)
        assert props[mutation] is not changed
        monkeypatch.setitem(props, mutation, changed)
    elif mutation in {"public", "defining", "foreign"}:
        foreign = _same_identity_ufunc(target)
        assert (
            foreign is not target
            and foreign == target
            and hash(foreign) == hash(target)
        )
        if mutation == "foreign":
            assert compat.selected_native_capture(foreign) is None
            assert capture.target is target and invoked == []
            return
        monkeypatch.setattr(
            compat._native_numpy if mutation == "public" else compat._native_ufuncs,
            name,
            foreign,
        )
    elif mutation == "unknown":
        monkeypatch.setattr(target, "unreviewed_field", None, raising=False)
    elif mutation == "equal-dict":
        monkeypatch.setattr(target, "_ufunc__static_props", dict(props))
    else:
        reducer = props["reduce"]
        function = vars(reducer)["_fun"]
        if mutation == "reducer-alias":
            monkeypatch.setattr(
                sys.modules[function.__module__], "_reduce_" + name, replacement
            )
        elif mutation == "reducer-code":
            monkeypatch.setattr(function, "__code__", function.__code__.replace())
        else:
            info = vars(reducer)["_jit_info"]
            monkeypatch.setitem(
                vars(reducer),
                "_jit_info",
                info._replace(keep_unused=not info.keep_unused),
            )
    with pytest.raises(ValueError):
        compat.selected_native_capture(target)
    assert not capture.check_target("changed-family-native-owner")
    assert capture.target is target and invoked == []


def _predicate_protocol_case(tmp_path, case, *, expected=None):
    scope = _discrete_protocol_scope(case)
    scope[3]["rows"]["plugin::synthetic::0"]["capture_original_object"] = True
    descriptor = _descriptors(
        scope, selection=[("plugin::synthetic", 0, "raw-x64-on")]
    )[0]
    function = _function(
        scope,
        descriptor,
        tmp_path,
        runtime="both",
        expected=expected or {"runtime": "validated", "numerics": "validated"},
    )
    return scope, function


def _predicate_retained_outputs(capture, expected_samples):
    labels = capture["metadata_runtime"]["native_sample_labels"]
    assert labels == [""] + [
        f"metadata-runtime-reference-{i}" for i in range(len(expected_samples) - 1)
    ]
    assert set(capture["metadata_runtime"]["modes"]) == {"disabled", "all"}
    wrappers = []
    for index, (label, expected) in enumerate(
        zip(labels, expected_samples, strict=True)
    ):
        expected = np.asarray(expected, dtype=np.bool_)
        sample = capture["native_protocol"]["samples"][label]
        wrappers.append(sample["wrapper_identity"])
        assert sample["wrapper_calls"] == 2
        assert (
            sample["roles"]["fresh-jit"]["jit_wrapper_identity"]
            == sample["roles"]["warm-jit"]["jit_wrapper_identity"]
            == sample["wrapper_identity"]
        )
        names = [
            (label + "-" if label else "") + role
            for role in ("eager", "fresh-jit", "warm-jit")
        ] + [f"metadata-ort-{mode}-{index}" for mode in ("disabled", "all")]
        for name in names:
            with np.load(capture[name]["path"], allow_pickle=False) as saved:
                assert saved.files == ["value_0"]
                actual = saved["value_0"]
                assert (
                    actual.dtype == np.dtype("bool") and actual.shape == expected.shape
                )
                np.testing.assert_array_equal(actual, expected)
    assert len(set(wrappers)) == len(wrappers)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "name,target,oracle",
    [
        ("eq", jnp.equal, np.equal),
        ("ne", jnp.not_equal, np.not_equal),
        ("lt", jnp.less, np.less),
        ("le", jnp.less_equal, np.less_equal),
        ("gt", jnp.greater, np.greater),
        ("ge", jnp.greater_equal, np.greater_equal),
        ("finite", jnp.isfinite, np.isfinite),
    ],
)
def test_predicate_protocol_retains_special_float_input_words_and_boolean_results(
    tmp_path, monkeypatch, dtype, name, target, oracle
):
    if dtype is np.float32:
        words = np.asarray(
            [
                0xFF800000,
                0xBF800000,
                0x80000000,
                0,
                0x7F800000,
                0x7FC00001,
                1,
                0x80000001,
            ],
            dtype=np.uint32,
        )
        other_words = np.asarray(
            [0xFF800000, 0x3F800000, 0, 0x80000000, 0xFF800000, 0x7FC00002, 0, 0],
            dtype=np.uint32,
        )
    else:
        words = np.asarray(
            [
                0xFFF0000000000000,
                0xBFF0000000000000,
                0x8000000000000000,
                0,
                0x7FF0000000000000,
                0x7FF8000000000001,
                1,
                0x8000000000000001,
            ],
            dtype=np.uint64,
        )
        other_words = np.asarray(
            [
                0xFFF0000000000000,
                0x3FF0000000000000,
                0,
                0x8000000000000000,
                0xFFF0000000000000,
                0x7FF8000000000002,
                0,
                0,
            ],
            dtype=np.uint64,
        )
    import io
    import re

    from jax2onnx import precision

    reports = []
    original_validate = precision._validate_source_samples

    def observe_validation(*args, **kwargs):
        report = original_validate(*args, **kwargs)
        reports.append((kwargs["optimization"], report))
        return report

    monkeypatch.setattr(precision, "_validate_source_samples", observe_validation)
    inputs = [words.view(dtype)]
    if name != "finite":
        inputs.append(other_words.view(dtype))
    expected = oracle(*inputs)
    scope, function = _predicate_protocol_case(
        tmp_path,
        {
            "testcase": "predicate-special-" + name,
            "callable": target,
            "input_values": inputs,
        },
    )
    if name == "finite":
        function()
    else:
        # The generated scientific stage still requests positive acceptance and fails.
        with pytest.raises(
            AssertionError,
            match=r"^SOURCE stage contract incomplete: numerics: requested validated, observed CheckStatus\.FAILED \(Product export/schema checked; raw callback recorded separately\. Runtime/numerics unverified\.\)$",
        ):
            function()
    outcome = function._source_outcome
    assert outcome["stage_body_complete"] is True
    assert outcome["stage_contract"] == {
        "runtime": "validated",
        "numerics": "validated",
    }
    assert outcome["stage_contract_passed"] is (name == "finite")
    assert outcome["capability_status"] == (
        "validated" if name == "finite" else "failed"
    )
    record = function._source_cell.history[-1]
    capture = record.replay_capture
    assert (
        record.schema
        is record.artifact
        is record.structure
        is record.lowering_audit
        is CheckStatus.VALIDATED
    )
    assert record.source_reference is record.runtime is CheckStatus.VALIDATED
    assert record.numerics is (
        CheckStatus.VALIDATED if name == "finite" else CheckStatus.FAILED
    )
    decoded = {}

    def arrays(role):
        if role not in decoded:
            entry = capture[role]
            assert entry["status"] == "captured"
            payload = Path(entry["path"]).read_bytes()
            assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
            with np.load(io.BytesIO(payload), allow_pickle=False) as saved:
                assert saved.files == [f"value_{i}" for i in range(len(saved.files))]
                decoded[role] = tuple(saved[key] for key in saved.files)
        return decoded[role]

    def exact_arrays(actual, original):
        assert len(actual) == len(original)
        for value, reference in zip(actual, original, strict=True):
            assert value.dtype == reference.dtype and value.shape == reference.shape
            assert value.tobytes() == reference.tobytes()

    if name == "finite":
        native_expected = target_expected = expected
        _predicate_retained_outputs(capture, [expected])
    else:
        bits = np.dtype(dtype).itemsize * 8
        signatures = [
            np.asarray(
                [
                    _predicate_comparison_word_oracle(
                        int(left), int(right), name, bits, flush_subnormals=flush
                    )
                    for left, right in zip(words, other_words, strict=True)
                ],
                dtype=np.bool_,
            )
            for flush in (True, False)
        ]
        native_expected, target_expected = signatures
        exact_arrays(
            [target_expected], [expected]
        )  # NumPy is only an extra target cross-check.
        mismatch_indices = {
            "eq": [6, 7],
            "ne": [6, 7],
            "lt": [7],
            "ge": [7],
            "gt": [6],
            "le": [6],
        }[name]
        assert (
            np.flatnonzero(native_expected != target_expected).tolist()
            == mismatch_indices
        )
    assert native_expected.dtype == target_expected.dtype == np.dtype("bool")
    assert native_expected.shape == target_expected.shape == (8,)
    sample = capture["native_protocol"]["samples"][""]
    assert set(sample["roles"]) == {"eager", "fresh-jit", "warm-jit"}
    assert sample["wrapper_calls"] == 2
    assert (
        sample["roles"]["fresh-jit"]["jit_wrapper_identity"]
        == sample["roles"]["warm-jit"]["jit_wrapper_identity"]
        == sample["wrapper_identity"]
    )
    for role, entry in sample["roles"].items():
        assert entry["input_capture_role"] == role + "-inputs"
        assert entry["outcome"]["status"] == "returned"
        assert entry["settings_before"] == entry["settings_after"]
        assert entry["settings_before"]["x64_enabled"] is True
        assert entry["settings_before"]["default_matmul_precision"] is None
        assert (
            entry["input_identity_before"]
            == entry["input_identity_after"]
            == sample["input_identity"]
        )
        exact_arrays(arrays(entry["input_capture_role"]), inputs)
        exact_arrays(arrays(entry["capture_role"]), [native_expected])
    runtime = capture["metadata_runtime"]
    assert runtime["native_sample_labels"] == [""]
    assert runtime["reference_status"] == runtime["binding_status"] == "validated"
    assert (
        runtime["reference_role"] == "original-eager-first-warm-same-wrapper-per-sample"
    )
    assert runtime["criteria"] == {
        "mode": "exact",
        "rtol": None,
        "atol": None,
        "equal_nan": True,
    }
    assert (
        runtime["artifact_sha256"]
        == runtime["loaded_artifact_sha256"]
        == record.artifact_sha256
    )
    artifact = Path(capture["artifact"]["path"])
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == record.artifact_sha256
    exact_arrays(arrays("metadata-protocol-0-feeds"), inputs)
    exact_arrays(arrays("metadata-protocol-0-references"), [native_expected])
    assert [mode for mode, _ in reports] == ["disabled", "all"]
    assert set(runtime["modes"]) == {"disabled", "all"}
    for mode, report in reports:
        retained = runtime["modes"][mode]
        assert report._artifact == artifact
        assert (
            report.artifact_sha256
            == retained["artifact_sha256"]
            == record.artifact_sha256
        )
        assert asdict(report.target) == retained["target"]
        assert [asdict(check) for check in report.checks] == retained["checks"]
        assert [asdict(value) for value in report.samples] == retained["samples"]
        assert all(check.status is CheckStatus.VALIDATED for check in report.checks)
        assert len(report.samples) == len(retained["sample_identities"]) == 1
        identity = retained["sample_identities"][0]
        assert retained["sample_identities"] == runtime["sample_identities"]
        evidence = report.samples[0]
        assert asdict(evidence.criteria) == identity["criteria"] == runtime["criteria"]
        assert [asdict(value) for value in evidence.inputs] == identity["inputs"]
        assert [asdict(value) for value in evidence.references] == identity[
            "references"
        ]
        exact_arrays(arrays(f"metadata-ort-{mode}-0"), [target_expected])
        assert len(retained["output_captures"]) == 1
        assert [asdict(value) for value in evidence.outputs] == retained[
            "output_captures"
        ][0]["outputs"]
        for records, values in (
            (evidence.inputs, inputs),
            (evidence.references, [native_expected]),
            (evidence.outputs, [target_expected]),
        ):
            for item, value in zip(records, values, strict=True):
                header = json.dumps(
                    {
                        "dtype": value.dtype.name,
                        "dtype_str": value.dtype.str,
                        "shape": list(value.shape),
                    },
                    sort_keys=True,
                ).encode()
                assert item.dtype == value.dtype.name and item.shape == value.shape
                assert (
                    item.sha256
                    == hashlib.sha256(header + b"\0" + value.tobytes()).hexdigest()
                )
        sample_checks = {check.category: check for check in evidence.checks}
        assert sample_checks["runtime-execution"].status is CheckStatus.VALIDATED
        if name == "finite":
            assert report.status is CheckStatus.VALIDATED
            report.require_sample_acceptance()
        else:
            mismatch = "Output 0: values fail exact integer/Boolean comparison"
            assert report.status is CheckStatus.FAILED
            assert sample_checks["numerical-acceptance"].status is CheckStatus.FAILED
            assert sample_checks["numerical-acceptance"].detail == mismatch
            with pytest.raises(
                precision.SampleAcceptanceError,
                match="^failed: " + re.escape(mismatch) + "$",
            ):
                report.require_sample_acceptance()
    checks = inventory_owner.source_metadata_numerical_checks(
        scope[1].rows[0],
        asdict(record),
        _original_contract=scope[0]["contracts"][scope[0]["cases"][0][5]],
    )
    for category in (
        "native_three_role_agreement",
        "same_artifact_and_feeds_both_modes",
        "reviewed_criteria",
        "dtype_shape_output_count",
    ):
        assert checks[category]["status"] == "validated"
    for category in ("ort_disabled", "ort_all", "criteria_comparison"):
        assert checks[category]["status"] == (
            "validated" if name == "finite" else "failed"
        )
    for category in ("operation_rounding", "required_internal_fp64"):
        assert checks[category]["status"] == "validated"
        assert checks[category]["applicability"] == "not-applicable"
        assert (
            "no floating operations, values or constants"
            not in checks[category]["detail"]
        )
    assert any(
        value[1][0] == np.dtype(dtype).name
        for value in capture["source_structure"]["floating_applicability"]["closure"][
            "source_values"
        ]
    )
    _retained_stage_record(function)


def _predicate_comparison_word_oracle(left, right, name, bits, *, flush_subnormals):
    """Independent integer-only IEEE ordering, optionally collapsing exponent zero."""
    assert type(left) is type(right) is int and bits in (32, 64)
    fraction_bits = 23 if bits == 32 else 52
    sign = 1 << (bits - 1)
    fraction_mask = (1 << fraction_bits) - 1
    exponent_mask = (sign - 1) ^ fraction_mask
    assert 0 <= left < sign * 2 and 0 <= right < sign * 2

    def is_nan(word):
        return word & exponent_mask == exponent_mask and word & fraction_mask != 0

    if is_nan(left) or is_nan(right):
        return name == "ne"

    def canonical(word):
        magnitude = word & (sign - 1)
        if magnitude == 0 or (flush_subnormals and magnitude & exponent_mask == 0):
            return 0
        return word

    left, right = canonical(left), canonical(right)
    equal = left == right
    if bool(left & sign) != bool(right & sign):
        less = bool(left & sign)
    elif left & sign:
        less = (left & (sign - 1)) > (right & (sign - 1))
    else:
        less = left < right
    return {
        "eq": equal,
        "ne": not equal,
        "lt": less,
        "le": less or equal,
        "gt": not less and not equal,
        "ge": not less,
    }[name]


def _predicate_comparison_law_words(bits):
    if bits == 32:
        magnitudes = (
            0,
            1,
            2,
            0x007FFFFF,
            0x00800000,
            0x00800001,
            0x3F800000,
            0x7F800000,
            0x7FC00001,
            0x7FC00002,
            0x7F800001,
        )
    else:
        assert bits == 64
        magnitudes = (
            0,
            1,
            2,
            0x000FFFFFFFFFFFFF,
            0x0010000000000000,
            0x0010000000000001,
            0x3FF0000000000000,
            0x7FF0000000000000,
            0x7FF8000000000001,
            0x7FF8000000000002,
            0x7FF0000000000001,
        )
    words = magnitudes + tuple(word | (1 << (bits - 1)) for word in magnitudes)
    assert len(words) == len(set(words)) == 22
    return words


@pytest.mark.parametrize(
    "dtype,x64",
    [(np.float32, False), (np.float32, True), (np.float64, True)],
    ids=["f32-x64off", "f32-x64on", "f64-x64on"],
)
@pytest.mark.parametrize(
    "case,target",
    [
        ("eq", jnp.equal),
        ("ne", jnp.not_equal),
        ("lt", jnp.less),
        ("le", jnp.less_equal),
        ("gt", jnp.greater),
        ("ge", jnp.greater_equal),
        ("scalar-left", None),
        ("scalar-right", None),
    ],
)
def test_predicate_native_comparison_law_full_words_before_target(
    tmp_path, monkeypatch, record_property, dtype, x64, case, target
):
    """Observe native comparison law; retain actual per-sample target failures."""
    import io
    from tests.extra_tests.converter import source_precision_raw_replay as raw

    bits = 32 if dtype is np.float32 else 64
    uint = np.uint32 if bits == 32 else np.uint64
    words = _predicate_comparison_law_words(bits)
    names = ("eq", "ne", "lt", "le", "gt", "ge")
    scalar_case = case.startswith("scalar-")

    def composites(left, right):
        return (
            jnp.equal(left, right),
            jnp.not_equal(left, right),
            jnp.less(left, right),
            jnp.less_equal(left, right),
            jnp.greater(left, right),
            jnp.greater_equal(left, right),
        )

    if scalar_case:
        target = composites
        sign = 1 << (bits - 1)
        tiny = sign | 1
        minimum_normal = sign | (1 << (23 if bits == 32 else 52))
        word_samples = [
            ((tiny,), words),
            ((minimum_normal,), tuple(reversed(words))),
        ]
        if case == "scalar-right":
            word_samples = [(right, left) for left, right in word_samples]
        shapes = [(), (22,)] if case == "scalar-left" else [(22,), ()]
        output_shape = (22,)
        output_names = names
    else:
        word_samples = [
            (words, words),
            (tuple(reversed(words)), words[-7:] + words[:-7]),
        ]
        shapes = [(22, 1), (1, 22)]
        output_shape = (22, 22)
        output_names = (case,)
    assert target is not None
    samples = [
        [
            np.asarray(value, dtype=uint).view(dtype).reshape(shape)
            for value, shape in zip(sample, shapes, strict=True)
        ]
        for sample in word_samples
    ]

    def expected(sample, *, flush_subnormals):
        left, right = sample
        if scalar_case:
            pairs = (
                [(left[0], word) for word in right]
                if len(left) == 1
                else [(word, right[0]) for word in left]
            )
        else:
            pairs = [(a, b) for a in left for b in right]
        return [
            np.asarray(
                [
                    _predicate_comparison_word_oracle(
                        a, b, name, bits, flush_subnormals=flush_subnormals
                    )
                    for a, b in pairs
                ],
                dtype=np.bool_,
            ).reshape(output_shape)
            for name in output_names
        ]

    source_expected = [
        expected(sample, flush_subnormals=True) for sample in word_samples
    ]
    ieee_expected = [
        expected(sample, flush_subnormals=False) for sample in word_samples
    ]
    scope = _discrete_protocol_scope(
        {
            "testcase": "native-comparison-law-" + case,
            "callable": target,
            "input_values": samples[0],
            "runtime_input_values": [samples[1]],
        }
    )
    scope[3]["rows"]["plugin::synthetic::0"]["capture_original_object"] = True
    caller = "raw-x64-on" if x64 else "raw-x64-off"
    descriptor = _descriptors(scope, selection=[("plugin::synthetic", 0, caller)])[0]
    function = _function(
        scope,
        descriptor,
        tmp_path,
        runtime="both",
        expected={"source_reference": "validated"},
    )
    original_resolve = raw._resolve_discrete_criteria
    gates = []

    def arrays(entry):
        assert entry["status"] == "captured"
        payload = Path(entry["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
        with np.load(io.BytesIO(payload), allow_pickle=False) as saved:
            assert saved.files == [f"value_{i}" for i in range(len(saved.files))]
            return [saved[name].copy() for name in saved.files]

    def assert_arrays(actual, wanted):
        assert len(actual) == len(wanted)
        for observed, reference in zip(actual, wanted, strict=True):
            assert (
                observed.dtype == reference.dtype and observed.shape == reference.shape
            )
            assert observed.tobytes() == reference.tobytes()

    def native_gate(prepared, capture, **kwargs):
        assert capture.target is target
        assert kwargs["labels"] == ["", "metadata-runtime-reference-0"]
        assert len(gates) == 0
        wrappers = []
        for index, label in enumerate(kwargs["labels"]):
            sample = capture.records["native_protocol"]["samples"][label]
            assert sample["wrapper_calls"] == 2
            wrappers.append(sample["wrapper_identity"])
            for role in ("eager", "fresh-jit", "warm-jit"):
                entry = sample["roles"][role]
                assert entry["outcome"]["status"] == "returned"
                assert entry["input_identity_before"] == entry["input_identity_after"]
                assert entry["settings_before"] == entry["settings_after"]
                assert entry["settings_before"]["x64_enabled"] is x64
                prefix = (label + "-" if label else "") + role
                assert entry["capture_role"] == prefix
                assert_arrays(arrays(capture.records[prefix]), source_expected[index])
                assert_arrays(
                    arrays(capture.records[entry["input_capture_role"]]), samples[index]
                )
                if role != "eager":
                    assert entry["jit_wrapper_identity"] == sample["wrapper_identity"]
        assert len(set(wrappers)) == 2
        gates.append(capture)
        return original_resolve(prepared, capture, **kwargs)

    monkeypatch.setattr(raw, "_resolve_discrete_criteria", native_gate)
    function()
    assert len(gates) == 1
    record = function._source_cell.history[-1]
    capture = record.replay_capture
    assert (
        record.source_reference
        is record.structure
        is record.runtime
        is CheckStatus.VALIDATED
    )
    assert capture["metadata_runtime"]["native_sample_labels"] == [
        "",
        "metadata-runtime-reference-0",
    ]
    observations = []
    any_failed = False
    for mode in ("disabled", "all"):
        result = capture["metadata_runtime"]["modes"][mode]
        assert len(result["samples"]) == 2
        for index, sample in enumerate(result["samples"]):
            actual = arrays(capture[f"metadata-ort-{mode}-{index}"])
            assert len(actual) == len(source_expected[index])
            matches_source = True
            matches_ieee = True
            for output, src, ieee in zip(
                actual, source_expected[index], ieee_expected[index], strict=True
            ):
                assert output.dtype == np.dtype("bool") and output.shape == output_shape
                matches_source &= output.tobytes() == src.tobytes()
                matches_ieee &= output.tobytes() == ieee.tobytes()
            expected_status = "validated" if matches_source else "failed"
            checks = {check["category"]: check["status"] for check in sample["checks"]}
            assert checks["runtime-execution"] == "validated"
            assert checks["numerical-acceptance"] == expected_status
            any_failed |= not matches_source
            assert_arrays(
                arrays(capture[f"metadata-protocol-{index}-feeds"]), samples[index]
            )
            assert_arrays(
                arrays(capture[f"metadata-protocol-{index}-references"]),
                source_expected[index],
            )
            observations.append(
                {
                    "mode": mode,
                    "sample": index,
                    "scientific_status": expected_status,
                    "matches_ieee_word_order": matches_ieee,
                    "outputs_hex": [output.tobytes().hex() for output in actual],
                }
            )
    assert record.numerics is (
        CheckStatus.FAILED if any_failed else CheckStatus.VALIDATED
    )
    record_property(
        "native_comparison_law",
        json.dumps(
            {
                "case": case,
                "bits": bits,
                "x64": x64,
                "canonical_words": words,
                "samples": word_samples,
                "source_outputs_hex": [
                    [value.tobytes().hex() for value in sample]
                    for sample in source_expected
                ],
                "actual_native_protocol": capture["native_protocol"],
                "target_observations": observations,
                "scope": "Diagnostic Native3 integer-word law; actual ORT statuses retained, no numerical acceptance inferred",
            },
            sort_keys=True,
        ),
    )
    _retained_stage_record(function)


@pytest.mark.parametrize("mode", ["and", "or", "xor"])
@pytest.mark.parametrize(
    "geometry",
    ["columns", "keepdims", "vmap", "no-axes", "empty-reduced", "empty-retained"],
)
def test_predicate_protocol_boolean_reductions_keep_native_geometry_and_exact_results(
    tmp_path, mode, geometry
):
    def original(x):
        if geometry == "vmap":
            if mode == "and":
                return jax.vmap(lambda row: jax.lax.reduce_and(row, (0,)))(x)
            if mode == "or":
                return jax.vmap(lambda row: jax.lax.reduce_or(row, (0,)))(x)
            return jax.vmap(lambda row: jax.lax.reduce_xor(row, (0,)))(x)
        axes = () if geometry == "no-axes" else (0,)
        if mode == "and":
            result = jax.lax.reduce_and(x, axes)
        elif mode == "or":
            result = jax.lax.reduce_or(x, axes)
        else:
            result = jax.lax.reduce_xor(x, axes)
        if geometry == "keepdims":
            return jax.lax.broadcast_in_dim(result, (1, x.shape[1]), (1,))
        return result

    values = np.asarray([[True, False, True], [True, False, False]], dtype=np.bool_)
    if geometry == "empty-reduced":
        values = np.zeros((0, 3), dtype=np.bool_)
    elif geometry == "empty-retained":
        values = np.zeros((2, 0), dtype=np.bool_)
    alternate = np.logical_not(values)
    axis = () if geometry == "no-axes" else 1 if geometry == "vmap" else 0
    oracle = {"and": np.logical_and, "or": np.logical_or, "xor": np.logical_xor}[mode]
    expected = [
        oracle.reduce(value, axis=axis, keepdims=geometry == "keepdims")
        for value in (values, alternate)
    ]
    if mode == "xor" and geometry == "columns":
        np.testing.assert_array_equal(expected[0], np.asarray([False, False, True]))
    scope, function = _predicate_protocol_case(
        tmp_path,
        {
            "testcase": "boolean-reduction-" + mode + "-" + geometry,
            "callable": original,
            "input_values": [values],
            "runtime_input_values": [[alternate]],
        },
    )
    function()
    record = function._source_cell.history[-1]
    _predicate_retained_outputs(record.replay_capture, expected)
    assert record.structure is record.numerics is CheckStatus.VALIDATED
    checks = inventory_owner.source_metadata_numerical_checks(
        scope[1].rows[0],
        asdict(record),
        _original_contract=scope[0]["contracts"][scope[0]["cases"][0][5]],
    )
    assert all(
        checks[name]["status"] == "validated"
        for name in (
            "native_three_role_agreement",
            "ort_disabled",
            "ort_all",
            "operation_rounding",
            "required_internal_fp64",
        )
    )
    import onnx

    capture = record.replay_capture
    manifest = json.loads(Path(capture["declared-typed-trace"]["path"]).read_bytes())
    model = onnx.load_model(capture["artifact"]["path"])
    assert inventory_owner._source_bitwise_closure(manifest, model) is not None
    checked = set()
    for index, node in enumerate(model.graph.node):
        for attribute_index, attribute in enumerate(node.attribute):
            if (node.op_type, attribute.name) not in {
                ("ReduceMin", "keepdims"),
                ("ReduceSum", "keepdims"),
                ("Mod", "fmod"),
                ("Cast", "to"),
            }:
                continue
            assert attribute.type == onnx.AttributeProto.INT
            forged = copy.deepcopy(model)
            replacement = onnx.helper.make_attribute(attribute.name, float(attribute.i))
            assert replacement.type == onnx.AttributeProto.FLOAT
            assert onnx.helper.get_attribute_value(
                replacement
            ) == onnx.helper.get_attribute_value(attribute)
            forged.graph.node[index].attribute[attribute_index].CopyFrom(replacement)
            assert inventory_owner._source_bitwise_closure(manifest, forged) is None
            checked.add((node.op_type, attribute.name))
    if geometry in {"columns", "keepdims", "vmap", "empty-retained"}:
        assert ("ReduceMin" if mode == "and" else "ReduceSum", "keepdims") in checked
        assert ("Cast", "to") in checked
        if mode == "xor":
            assert ("Mod", "fmod") in checked
    _retained_stage_record(function)


@pytest.mark.parametrize(
    "scalar_type",
    [
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.longlong,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.ulonglong,
    ],
)
def test_source_relation_owns_exact_numpy_integer_parameters(scalar_type):
    from tests.extra_tests.converter import source_precision_raw_replay as raw

    axis = scalar_type(1)
    params = {"axes": (axis,)}
    before = axis.tobytes()
    assert raw._structure_flat_parameters(params) is None
    assert params["axes"][0] is axis
    assert type(params["axes"][0]) is scalar_type
    assert params["axes"][0].tobytes() == before
    # Existing Python Boolean JIT metadata is still owned, not an axis proof.
    assert raw._structure_flat_parameters({"keep_unused": False}) is None


@pytest.mark.parametrize(
    "kind",
    ["numpy-bool", "numpy-float", "python-float", "integer-like", "subclass", "array"],
)
def test_source_relation_rejects_unowned_integer_parameter_objects(kind):
    from tests.extra_tests.converter import source_precision_raw_replay as raw

    called = []

    class IntegerLike:
        def __int__(self):
            called.append("int")
            raise AssertionError("Foreign integer conversion executed")

        def __index__(self):
            called.append("index")
            raise AssertionError("Foreign integer index executed")

        @property
        def dtype(self):
            called.append("dtype")
            raise AssertionError("Foreign dtype getter executed")

        @property
        def shape(self):
            called.append("shape")
            raise AssertionError("Foreign shape getter executed")

    class IntegerSubclass(np.int64):
        def __int__(self):
            called.append("subclass-int")
            raise AssertionError("Scalar subclass conversion executed")

        def __index__(self):
            called.append("subclass-index")
            raise AssertionError("Scalar subclass index executed")

        @property
        def dtype(self):
            called.append("subclass-dtype")
            raise AssertionError("Scalar subclass dtype getter executed")

    value = {
        "numpy-bool": lambda: np.bool_(True),
        "numpy-float": lambda: np.float64(1),
        "python-float": lambda: 1.0,
        "integer-like": IntegerLike,
        "subclass": lambda: np.int64.__new__(IntegerSubclass, 1),
        "array": lambda: np.asarray([1], dtype=np.int64),
    }[kind]()
    assert called == []
    with pytest.raises(
        raw._StructureBindingGap, match="nested-or-unowned-source-parameter"
    ):
        raw._structure_flat_parameters({"axes": (value,)})
    assert called == []


@pytest.mark.parametrize("mutation", ["dtype", "word"])
def test_source_relation_rejects_changed_numpy_axis_storage(
    tmp_path, monkeypatch, mutation
):
    from tests.extra_tests.converter import source_precision_raw_replay as raw

    def original(x):
        return jax.vmap(lambda row: jax.lax.reduce_and(row, (0,)))(x)

    values = np.asarray([[True, True, True], [True, False, False]], dtype=np.bool_)
    _, function = _predicate_protocol_case(
        tmp_path,
        {
            "testcase": "changed-numpy-axis-" + mutation,
            "callable": original,
            "input_values": [values],
        },
        expected={
            "structure": "failed",
            "runtime": "unverified",
            "numerics": "unverified",
        },
    )
    actual = user_interface._to_onnx
    observations = []

    def export(**kwargs):
        collector = kwargs["source_structure_collector"]

        def changed_trace(trace, final_ir, result, returned):
            closed = trace.closed_jaxpr
            body = raw._structure_body(closed)
            assert len(body.eqns) == 1
            equation = body.eqns[0]
            assert equation.primitive is jax.lax.reduce_and_p
            axis = equation.params["axes"][0]
            assert type(axis) is np.int64 and axis == 1
            original_manifest, original_sha, pending = raw._structure_trace_manifest(
                closed
            )
            assert not pending
            changed = np.int32(1) if mutation == "dtype" else np.int64(0)
            changed_params = {**equation.params, "axes": (changed,)}
            assert raw._structure_flat_parameters(changed_params) is None
            changed_body = body.replace(eqns=[equation.replace(params=changed_params)])
            changed_closed = (
                changed_body if closed is body else closed.replace(jaxpr=changed_body)
            )
            changed = replace(trace, closed_jaxpr=changed_closed, jaxpr=changed_body)
            collector(changed, final_ir, result, returned)
            assert raw._structure_trace_manifest(closed) == (
                original_manifest,
                original_sha,
                pending,
            )
            assert equation.params["axes"][0] is axis
            observations.append((original_manifest, original_sha))

        kwargs["source_structure_collector"] = changed_trace
        return actual(**kwargs)

    monkeypatch.setattr(user_interface, "_to_onnx", export)
    function()
    record = function._source_cell.history[-1]
    capture = record.replay_capture
    structure = capture["source_structure"]
    binding = structure["source_binding"]
    assert len(observations) == 1
    original_manifest, original_sha = observations[0]
    assert binding["status"] == "failed"
    assert "complete-source-trace-differs" in binding["reasons"]
    assert (
        binding["prepared_trace_sha256"]
        == binding["fresh_prepared_trace_sha256"]
        == original_sha
    )
    assert binding["actual_trace_sha256"] != original_sha
    assert structure["ir_result"]["status"] == "validated"
    assert structure["artifact_binding"]["status"] == "validated"
    assert record.runtime is record.numerics is CheckStatus.UNVERIFIED
    assert capture["metadata_runtime"]["modes"] == {}
    declared = Path(capture["declared-typed-trace"]["path"]).read_bytes()
    assert (
        hashlib.sha256(declared).hexdigest()
        == capture["declared-typed-trace"]["sha256"]
    )
    assert json.loads(declared) == json.loads(json.dumps(original_manifest))
    assert record.original_trace_json == json.dumps(
        original_manifest, sort_keys=True, separators=(",", ":")
    )
    _retained_stage_record(function)


def test_predicate_boolean_reduction_reader_preserves_canonical_numpy_axes(tmp_path):
    import onnx

    def original(x):
        return jax.vmap(lambda row: jax.lax.reduce_and(row, (0,)))(x)

    values = np.asarray([[True, True, True], [True, False, False]], dtype=np.bool_)
    _, function = _predicate_protocol_case(
        tmp_path,
        {
            "testcase": "boolean-reduction-reader-numpy-axes",
            "callable": original,
            "input_values": [values],
            "runtime_input_values": [[np.logical_not(values)]],
        },
    )
    function()
    record = function._source_cell.history[-1]
    assert (
        record.structure
        is record.lowering_audit
        is record.numerics
        is CheckStatus.VALIDATED
    )
    capture = record.replay_capture
    _predicate_retained_outputs(capture, [[True, False], [False, False]])
    manifest_path = Path(capture["declared-typed-trace"]["path"])
    manifest_payload = manifest_path.read_bytes()
    manifest = json.loads(manifest_payload)
    artifact_path = Path(capture["artifact"]["path"])
    artifact_payload = artifact_path.read_bytes()
    model = onnx.load_model(artifact_path)
    unresolved = []
    scalar = inventory_owner.canonical_contract(np.int64(1), unresolved=unresolved)
    assert unresolved == []
    assert len(manifest["equations"]) == 1
    equation = manifest["equations"][0]
    assert equation["primitive"] == "reduce_and"
    assert equation["params"] == {"axes": [scalar]}
    assert equation["outputs"][0]["aval"] == ["bool", [2], False]
    original_manifest = copy.deepcopy(manifest)
    original_sha256 = inventory_owner._contract_hash(manifest)
    assert json.loads(record.original_trace_json) == original_manifest
    assert (
        original_sha256
        == hashlib.sha256(record.original_trace_json.encode()).hexdigest()
    )
    assert inventory_owner._source_bitwise_closure(manifest, model) is not None
    assert manifest == original_manifest
    assert inventory_owner._contract_hash(manifest) == original_sha256

    for mutation in (
        "dtype-name",
        "dtype-str",
        "byte-length",
        "hex",
        "noncanonical-hex",
        "extra-key",
        "boolean-scalar",
        "floating-scalar",
        "boolean-axis",
        "floating-axis",
        "mixed-duplicate",
        "coherent-wrong-axis",
    ):
        forged = copy.deepcopy(manifest)
        forged_equation = forged["equations"][0]
        axis = forged_equation["params"]["axes"][0]
        if mutation == "dtype-name":
            axis["scalar_dtype"] = "int128"
        elif mutation == "dtype-str":
            axis["dtype_str"] = np.dtype("int32").str
        elif mutation == "byte-length":
            axis["bytes_hex"] = axis["bytes_hex"][:-2]
        elif mutation == "hex":
            axis["bytes_hex"] = "zz" * np.dtype("int64").itemsize
        elif mutation == "noncanonical-hex":
            axis["bytes_hex"] += " "
        elif mutation == "extra-key":
            axis["extra"] = None
        elif mutation in {"boolean-scalar", "floating-scalar"}:
            value = np.bool_(True) if mutation == "boolean-scalar" else np.float64(1)
            forged_equation["params"]["axes"] = [
                inventory_owner.canonical_contract(value, unresolved=unresolved)
            ]
        elif mutation in {"boolean-axis", "floating-axis"}:
            forged_equation["params"]["axes"] = [
                True if mutation == "boolean-axis" else 1.0
            ]
        elif mutation == "mixed-duplicate":
            forged_equation["params"]["axes"] = [1, axis]
        else:
            forged_equation["params"]["axes"] = [
                inventory_owner.canonical_contract(np.int64(0), unresolved=unresolved)
            ]
            # Axis zero gives shape [3] for the same source input [2, 3].
            # Keep this source internally coherent while retaining the saved IR.
            forged_equation["outputs"][0]["aval"][1] = [3]
            forged["outputs"][0]["aval"][1] = [3]
        before_replay = copy.deepcopy(forged)
        before_sha256 = inventory_owner._contract_hash(forged)
        assert inventory_owner._source_bitwise_closure(forged, model) is None, mutation
        assert forged == before_replay, mutation
        assert inventory_owner._contract_hash(forged) == before_sha256, mutation
    assert unresolved == []
    assert manifest == original_manifest
    assert inventory_owner._contract_hash(manifest) == original_sha256
    assert manifest_path.read_bytes() == manifest_payload
    assert artifact_path.read_bytes() == artifact_payload
    _retained_stage_record(function)


def _predicate_mixed_promotion(tmp_path, *, lossy=False):
    def original(x, y):
        return jnp.equal(x, y)

    primary = [
        np.asarray([1, 2, 3], dtype=np.int32),
        np.asarray([1, 2.5, 3], dtype=np.float32),
    ]
    alternate = [
        np.asarray([2**24 + int(lossy), -(2**24), 0], dtype=np.int32),
        np.asarray([2**24, -(2**24), 0.5], dtype=np.float32),
    ]
    scope, function = _predicate_protocol_case(
        tmp_path,
        {
            "testcase": "mixed-predicate-promotion",
            "callable": original,
            "input_values": primary,
            "runtime_input_values": [alternate],
        },
    )
    return scope, function, primary, alternate


@pytest.mark.parametrize("lossy", [False, True])
def test_predicate_protocol_promotion_needs_losslessness_for_every_original_sample(
    tmp_path, lossy
):
    scope, function, primary, alternate = _predicate_mixed_promotion(
        tmp_path, lossy=lossy
    )
    function()
    record = function._source_cell.history[-1]
    expected = [
        np.equal(values[0].astype(np.float32), values[1])
        for values in (primary, alternate)
    ]
    _predicate_retained_outputs(record.replay_capture, expected)
    assert record.runtime is record.numerics is CheckStatus.VALIDATED
    checks = inventory_owner.source_metadata_numerical_checks(
        scope[1].rows[0],
        asdict(record),
        _original_contract=scope[0]["contracts"][scope[0]["cases"][0][5]],
    )
    if lossy:
        assert int(alternate[0][0]) == 2**24 + 1
        assert float(np.float32(alternate[0][0])) != int(alternate[0][0])
        for category in ("operation_rounding", "required_internal_fp64"):
            assert checks[category]["status"] == "unverified"
            assert checks[category].get("applicability") != "not-applicable"
    else:
        proof = record.replay_capture["source_structure"]["floating_applicability"]
        assert proof["status"] == "validated"
        assert [item["label"] for item in proof["lossless_input_samples"]] == [
            "",
            "metadata-runtime-reference-0",
        ]
        for conversion in proof["closure"]["lossless_conversions"]:
            assert [item["sample_index"] for item in conversion["samples"]] == [0, 1]
        assert proof["closure"]["lossless_conversions"]
        for category in ("operation_rounding", "required_internal_fp64"):
            assert checks[category]["status"] == "validated"
            assert checks[category]["applicability"] == "not-applicable"
    _retained_stage_record(function)


def test_predicate_protocol_lossless_proof_cannot_detach_from_actual_input_copies(
    tmp_path, monkeypatch
):
    scope, function, _, _ = _predicate_mixed_promotion(tmp_path)
    function()
    record = asdict(function._source_cell.history[-1])
    original_contract = scope[0]["contracts"][scope[0]["cases"][0][5]]
    capture = record["replay_capture"]
    copied = copy.deepcopy(capture)
    original_path = Path(capture["inputs"]["path"])
    replacement = tmp_path / "predicate-input-replaced-after-read.npz"
    replacement.write_bytes(original_path.read_bytes())
    alternate_payload = Path(
        capture["metadata-runtime-reference-0-inputs"]["path"]
    ).read_bytes()
    assert hashlib.sha256(alternate_payload).hexdigest() != capture["inputs"]["sha256"]
    copied["inputs"]["path"] = str(replacement)
    read_bytes = Path.read_bytes
    replaced = []

    def replace_after_read(path):
        payload = read_bytes(path)
        if path == replacement:
            assert replaced == []
            replaced.append(path)
            path.write_bytes(alternate_payload)
        return payload

    primary = inventory_owner._canonical_container(original_contract["input_values"])
    alternate = inventory_owner._canonical_container(
        original_contract["runtime_input_values"]
    )
    with monkeypatch.context() as guarded:
        guarded.setattr(Path, "read_bytes", replace_after_read)
        retained = inventory_owner._source_predicate_input_samples(
            copied, expected_sample_count=2, original_samples=[primary, *alternate]
        )
    assert replaced == [replacement]
    assert retained is not None
    assert retained[1][0]["original_input_sha256"] == capture["inputs"]["sha256"]
    assert (
        hashlib.sha256(retained[0][0][0].tobytes()).hexdigest() == primary[0]["sha256"]
    )
    assert replacement.read_bytes() == alternate_payload
    assert (
        hashlib.sha256(original_path.read_bytes()).hexdigest()
        == capture["inputs"]["sha256"]
    )
    for mutation in (
        "missing-warm-input",
        "wrong-role",
        "original-sample-copy",
        "input-digest",
        "duplicate-sample",
        "cast-edge",
        "missing-lossless-bindings",
    ):
        forged = copy.deepcopy(record)
        capture = forged["replay_capture"]
        sample = capture["native_protocol"]["samples"]["metadata-runtime-reference-0"]
        proof = capture["source_structure"]["floating_applicability"]
        if mutation == "missing-warm-input":
            capture.pop(sample["roles"]["warm-jit"]["input_capture_role"])
        elif mutation == "wrong-role":
            sample["roles"]["warm-jit"]["input_capture_role"] = sample["roles"][
                "eager"
            ]["input_capture_role"]
        elif mutation == "original-sample-copy":
            capture["metadata-runtime-reference-0-inputs"] = copy.deepcopy(
                capture["inputs"]
            )
        elif mutation == "input-digest":
            proof["closure"]["lossless_conversions"][0]["samples"][1][
                "input_sha256"
            ] = ("0" * 64)
        elif mutation == "duplicate-sample":
            proof["closure"]["lossless_conversions"][0]["samples"][1][
                "sample_index"
            ] = 0
        elif mutation == "cast-edge":
            proof["closure"]["lossless_conversions"][0][
                "cast_input_edge"
            ] = "unattested-input"
        else:
            proof.pop("lossless_input_samples")
        path = tmp_path / ("predicate-forged-" + mutation + ".json")
        path.write_text(json.dumps(proof, sort_keys=True, separators=(",", ":")) + "\n")
        capture["source-bitwise-applicability"] = {
            "status": "captured",
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        try:
            checks = inventory_owner.source_metadata_numerical_checks(
                scope[1].rows[0], forged, _original_contract=original_contract
            )
        except ValueError:
            continue
        for category in ("operation_rounding", "required_internal_fp64"):
            assert checks[category]["status"] != "validated", mutation
    altered = copy.deepcopy(original_contract)
    altered["runtime_input_values"] = []
    with pytest.raises(ValueError):
        inventory_owner.source_metadata_numerical_checks(
            scope[1].rows[0], record, _original_contract=altered
        )


@pytest.mark.parametrize("failed_mode", ["disabled", "all"])
def test_predicate_protocol_target_mismatch_keeps_both_modes_and_failure(
    tmp_path, monkeypatch, failed_mode
):
    import jax2onnx.precision as precision

    active, modes = [], []
    original_validate, original_run = (
        precision._validate_source_samples,
        precision._run_cpu_samples,
    )

    def validate(*args, **kwargs):
        mode = kwargs["optimization"]
        modes.append(mode)
        active.append(mode)
        try:
            return original_validate(*args, **kwargs)
        finally:
            active.pop()

    def run(*args, **kwargs):
        returned = original_run(*args, **kwargs)
        if active[-1] == failed_mode:
            changed = np.array(returned[0], copy=True)
            changed.flat[0] = not changed.flat[0]
            return (changed, *returned[1:])
        return returned

    monkeypatch.setattr(precision, "_validate_source_samples", validate)
    monkeypatch.setattr(precision, "_run_cpu_samples", run)
    scope, function = _predicate_protocol_case(
        tmp_path,
        {
            "testcase": "predicate-target-mismatch",
            "callable": jnp.isfinite,
            "input_values": [np.asarray([0.0, np.inf, np.nan], dtype=np.float32)],
        },
        expected={"runtime": "validated", "numerics": "failed"},
    )
    function()
    record = function._source_cell.history[-1]
    assert modes == ["disabled", "all"] and record.numerics is CheckStatus.FAILED
    runtime = record.replay_capture["metadata_runtime"]
    other = "all" if failed_mode == "disabled" else "disabled"
    assert runtime["modes"][failed_mode]["status"] == "failed"
    assert runtime["modes"][other]["status"] == "validated"
    for mode in modes:
        with np.load(
            record.replay_capture[f"metadata-ort-{mode}-0"]["path"], allow_pickle=False
        ) as saved:
            assert bool(saved["value_0"][0]) is (mode != failed_mode)
    checks = inventory_owner.source_metadata_numerical_checks(
        scope[1].rows[0],
        asdict(record),
        _original_contract=scope[0]["contracts"][scope[0]["cases"][0][5]],
    )
    assert checks["ort_" + failed_mode]["status"] == "failed"
    assert checks["ort_" + other]["status"] == "validated"
    assert checks["criteria_comparison"]["status"] == "failed"
    _retained_stage_record(function)


def test_predicate_plain_jit_hint_keeps_actual_early_member_without_boolean_aliasing():
    hints = compat.native_plain_jit_inline_hints()
    assert hints[:2] == compat.native_jit_boolean_inline_hints()
    kind = compat._NATIVE_PLAIN_JIT_INLINE_CLASS
    if kind is None:
        assert hints == (False, True)
    else:
        assert len(hints) == 3
        assert hints[2] is vars(kind)["JAX_EARLY"]
        assert type(hints[2]) is kind
        assert all(hints[2] is not value for value in hints[:2])


def test_predicate_plain_jit_hint_rejects_foreign_class_before_getters(monkeypatch):
    called = []

    class ForeignMeta(type):
        def __getattribute__(self, name):
            called.append(name)
            raise AssertionError("Foreign inline class must not be inspected")

    class ForeignInline(metaclass=ForeignMeta):
        pass

    monkeypatch.setitem(vars(compat._native_api), "Inline", ForeignInline)
    with pytest.raises(TypeError, match="inline class binding changed"):
        compat.native_plain_jit_inline_hints()
    assert called == []


def test_predicate_plain_jit_hint_rejects_replaced_native_member():
    kind = compat._NATIVE_PLAIN_JIT_INLINE_CLASS
    if kind is None:
        assert compat.native_plain_jit_inline_hints() == (False, True)
        return
    original = vars(kind)["JAX_EARLY"]
    try:
        type.__setattr__(kind, "JAX_EARLY", object())
        with pytest.raises(TypeError, match="inline member binding changed"):
            compat.native_plain_jit_inline_hints()
    finally:
        type.__setattr__(kind, "JAX_EARLY", original)
    assert compat.native_plain_jit_inline_hints()[2] is original


def test_predicate_plain_jit_reader_rejects_same_name_foreign_hint_before_use():
    from enum import Enum
    from tests.extra_tests.converter import source_precision_raw_replay as raw

    supplied = jax.jit(lambda value: jax.lax.eq(value, value))
    closed = jax.make_jaxpr(supplied)(np.asarray([1.0, -0.0], dtype=np.float32))
    eqn = closed.jaxpr.eqns[0]
    assert eqn.primitive is compat.native_jit_primitive()
    foreign = Enum("Inline", {"JAX_EARLY": 2}).JAX_EARLY
    original = eqn.params["inline"]
    try:
        for value in (foreign, "JAX_EARLY"):
            eqn.params["inline"] = value
            with pytest.raises(
                ValueError, match="plain-jit-compilation-metadata-unverified"
            ):
                raw._structure_plain_jit_child(eqn)
    finally:
        eqn.params["inline"] = original
    assert raw._structure_plain_jit_child(eqn) is not None


def test_isfinite_original_target_capture_keeps_ordinary_issubdtype_cache_growth(
    tmp_path,
):
    from tests.extra_tests.converter import source_precision_raw_replay as raw

    target = jnp.isfinite
    function = vars(target)["_fun"]
    owner = function.__globals__["dtypes"]
    issubdtype = vars(owner)["issubdtype"]
    cached = vars(owner)["_issubdtype_cached"]
    capture = raw.RawReplayCapture(tmp_path / "isfinite-dtype-cache-growth")
    capture.bind_target(target)
    original_state = copy.deepcopy(capture.target_state)
    assert capture.target is target
    assert capture.records["instantiated-target-state"]["status"] == "captured"
    assert capture.check_target("unchanged-isfinite")

    class SourceFiniteFloat(np.float32):
        pass

    before = cached.cache_info()
    assert issubdtype(SourceFiniteFloat, np.floating)
    assert not issubdtype(SourceFiniteFloat, np.integer)
    assert issubdtype(np.float64, np.floating)
    after = cached.cache_info()
    assert after.misses >= before.misses + 2
    assert after.hits + after.misses >= before.hits + before.misses + 3
    assert capture.check_target("after-ordinary-issubdtype-cache-growth")
    assert capture.target is target and capture.target_state == original_state
    assert capture.records["native_outcomes"] == {}


@pytest.mark.parametrize(
    "mutation",
    [
        "issubdtype-code",
        "cached-code",
        "issubclass-code",
        "cached-global",
        "foreign-cache",
        "equal-types-tuple",
        "custom-float-membership",
    ],
)
def test_isfinite_dtype_dependency_mutations_reject_without_invocation(
    tmp_path, monkeypatch, mutation
):
    import functools
    from jax._src import util as native_util
    from tests.extra_tests.converter import source_precision_raw_replay as raw

    target = jnp.isfinite
    owner = vars(target)["_fun"].__globals__["dtypes"]
    namespace = vars(owner)
    function = namespace["issubdtype"]
    cached = namespace["_issubdtype_cached"]
    helper = namespace["_issubclass"]
    scalar_types = namespace["_custom_float_scalar_types"]
    original_members = scalar_types[:]
    capture = raw.RawReplayCapture(tmp_path / mutation)
    capture.bind_target(target)
    assert capture.target is target and capture.check_target("unchanged-isfinite")
    attempted = []

    def changed(*args, **kwargs):
        raise AssertionError("Altered dtype code must never execute")

    class ForeignCache:
        def __getattribute__(self, name):
            attempted.append(("get", name))
            raise AssertionError("Foreign dtype cache getters must never execute")

        def __call__(self, *args, **kwargs):
            attempted.append(("call",))
            raise AssertionError("Foreign dtype cache must never execute")

    ForeignCache.__name__ = type(cached).__name__
    ForeignCache.__qualname__ = type(cached).__qualname__
    ForeignCache.__module__ = type(cached).__module__
    try:
        with monkeypatch.context() as patch:
            if mutation in {"issubdtype-code", "cached-code", "issubclass-code"}:
                selected = {
                    "issubdtype-code": function,
                    "cached-code": cached.__wrapped__,
                    "issubclass-code": helper,
                }[mutation]
                assert selected.__closure__ is None
                patch.setattr(
                    selected,
                    "__code__",
                    changed.__code__.replace(
                        co_name=selected.__code__.co_name,
                        co_qualname=selected.__code__.co_qualname,
                    ),
                )
            elif mutation == "cached-global":
                if type(cached) is vars(functools)["_lru_cache_wrapper"]:
                    replacement = functools.lru_cache(maxsize=512)(cached.__wrapped__)
                else:
                    native = vars(native_util)["lib_weakref_lru_cache"]
                    assert type(cached) is vars(native)["StrongLRUCache"]
                    replacement = vars(native)["strong_lru_cache"](
                        cached.__wrapped__, None, 512, num_shards=64
                    )
                assert type(replacement) is type(cached) and replacement is not cached
                patch.setitem(namespace, "_issubdtype_cached", replacement)
            elif mutation == "foreign-cache":
                patch.setitem(namespace, "_issubdtype_cached", ForeignCache())
            elif mutation == "equal-types-tuple":
                original = namespace["_types_for_issubdtype"]
                replacement = tuple(list(original))
                assert replacement == original and replacement is not original
                patch.setitem(namespace, "_types_for_issubdtype", replacement)
            else:
                assert np.float32 not in scalar_types
                scalar_types.append(np.float32)
            assert not capture.check_target("changed-isfinite-dtype-dependency")
            assert attempted == [] and capture.records["native_outcomes"] == {}
    finally:
        scalar_types[:] = original_members
    assert capture.target is target
    assert capture.check_target("restored-isfinite-dtype-dependency")
    assert attempted == []


@pytest.mark.parametrize(
    "scope,name",
    [
        ("numpy", "dtype"),
        ("numpy", "generic"),
        ("numpy", "floating"),
        ("numpy", "inexact"),
        ("numpy", "number"),
        ("numpy", "signedinteger"),
        ("numpy", "unsignedinteger"),
        ("numpy", "integer"),
        ("numpy-code", "issubdtype"),
        ("numpy-code", "issubclass_"),
        ("numpy-global", "issubclass_"),
        ("numpy-global", "dtype"),
        ("numpy-global", "generic"),
        ("numpy-global", "issubclass"),
        ("numpy-global", "TypeError"),
        ("jax-global", "isinstance"),
        ("jax-global", "type"),
        ("jax-global", "bool"),
        ("jax-global", "issubclass"),
        ("jax-global", "TypeError"),
    ],
)
def test_isfinite_numpy_dtype_aliases_and_resolved_builtins_reject_mutations(
    tmp_path, monkeypatch, scope, name
):
    from tests.extra_tests.converter import source_precision_raw_replay as raw

    target = jnp.isfinite
    owner = vars(target)["_fun"].__globals__["dtypes"]
    numpy_function = vars(np)["issubdtype"]
    numpy_namespace = numpy_function.__globals__
    capture = raw.RawReplayCapture(tmp_path / (scope + "-" + name))
    capture.bind_target(target)
    assert capture.target is target and capture.check_target("unchanged-isfinite")

    def changed(*args, **kwargs):
        raise AssertionError("Altered dtype binding must never execute")

    with monkeypatch.context() as patch:
        if scope == "numpy-code":
            function = numpy_function if name == "issubdtype" else numpy_namespace[name]
            assert function.__closure__ is None
            patch.setattr(
                function,
                "__code__",
                changed.__code__.replace(
                    co_name=function.__code__.co_name,
                    co_qualname=function.__code__.co_qualname,
                ),
            )
        else:
            namespace = (
                vars(np)
                if scope == "numpy"
                else numpy_namespace if scope == "numpy-global" else vars(owner)
            )
            # Shadow resolved builtins in the owning library's globals. The
            # interpreter's shared builtins dictionary remains untouched.
            patch.setitem(namespace, name, changed)
        assert not capture.check_target("changed-isfinite-numpy-dependency")
        assert capture.records["native_outcomes"] == {}
    assert capture.target is target
    assert capture.check_target("restored-isfinite-numpy-dependency")


@pytest.mark.parametrize(
    "mutation",
    [
        "numpy-module",
        "strong-module",
        "foreign-object",
        "foreign-class",
        "strict-instancecheck",
        "strict-subclasscheck",
    ],
)
def test_isfinite_selected_module_and_class_edges_reject_without_getters(
    tmp_path, monkeypatch, mutation
):
    import functools
    import types
    from jax._src import util as native_util
    from tests.extra_tests.converter import source_precision_raw_replay as raw

    target = jnp.isfinite
    namespace = vars(vars(target)["_fun"].__globals__["dtypes"])
    util_namespace = vars(native_util)
    capture = raw.RawReplayCapture(tmp_path / mutation)
    capture.bind_target(target)
    assert capture.target is target and capture.check_target("unchanged-isfinite")
    attempted = []

    class ForeignModule(types.ModuleType):
        def __getattribute__(self, name):
            attempted.append(("module", name))
            raise AssertionError("Foreign module getters must never execute")

    class ForeignObject:
        def __getattribute__(self, name):
            attempted.append(("object", name))
            raise AssertionError("Foreign __class__ getters must never execute")

    class ForeignMeta(type):
        def __getattribute__(self, name):
            attempted.append(("metaclass", name))
            raise AssertionError("Foreign metaclass getters must never execute")

        def __eq__(self, other):
            attempted.append(("equal",))
            raise AssertionError("Foreign class equality must never execute")

        def __hash__(self):
            attempted.append(("hash",))
            raise AssertionError("Foreign class hash must never execute")

    class ForeignClass(metaclass=ForeignMeta):
        pass

    def changed_check(*args, **kwargs):
        attempted.append(("type-check",))
        raise AssertionError("Altered StrictABC checks must never execute")

    members = namespace["_custom_float_scalar_types"]
    original_members = members[:]
    try:
        with monkeypatch.context() as patch:
            if mutation == "numpy-module":
                patch.setitem(namespace, "np", ForeignModule("numpy"))
            elif mutation == "strong-module":
                cached = namespace["_issubdtype_cached"]
                if type(cached) is vars(functools)["_lru_cache_wrapper"]:
                    patch.setitem(vars(functools), "lru_cache", changed_check)
                else:
                    patch.setitem(
                        util_namespace,
                        "lib_weakref_lru_cache",
                        ForeignModule("jaxlib.weakref_lru_cache"),
                    )
            elif mutation == "foreign-object":
                members.append(ForeignObject())
            elif mutation == "foreign-class":
                members.append(ForeignClass)
            else:
                method = (
                    "__instancecheck__"
                    if mutation == "strict-instancecheck"
                    else "__subclasscheck__"
                )
                patch.setattr(util_namespace["StrictABCMeta"], method, changed_check)
            assert not capture.check_target("changed-selected-dtype-edge")
            assert attempted == [] and capture.records["native_outcomes"] == {}
    finally:
        members[:] = original_members
    assert capture.target is target and capture.check_target(
        "restored-selected-dtype-edge"
    )
    assert attempted == []


@pytest.mark.parametrize(
    "mutation", ["fn", "cache_context_fn", "maxsize", "explain", "num_shards"]
)
def test_isfinite_owned_native_cache_state_is_reread_without_live_mutation(
    tmp_path, mutation
):
    import functools
    from jax2onnx._compat import jax as compat

    live = compat._SOURCE_DTYPE_CACHE
    graph = compat._NativeCaptureGraph()
    original_fn, original_state = compat._source_dtype_cache_state(live, graph)
    live_class = type(live)
    if live_class is vars(functools)["_lru_cache_wrapper"]:
        # The old cohort has a different exact implementation, not a Strong
        # state protocol. Its separately owned wrapper still has the strict
        # callback/configuration identity fence.
        owned = functools.lru_cache(maxsize=512)(original_fn)
        before_graph = compat._NativeCaptureGraph()
        before_fn, before = compat._source_dtype_cache_state(owned, before_graph)
        assert before_fn is original_fn and before["kind"] == "functools"
        if mutation == "fn":

            def changed(*args, **kwargs):
                raise AssertionError("Owned altered callback must never execute")

            vars(owned)["__wrapped__"] = changed
            after_graph = compat._NativeCaptureGraph()
            after_fn, after = compat._source_dtype_cache_state(owned, after_graph)
            assert after_fn is changed
            assert (after, after_graph.nodes) != (before, before_graph.nodes)
        else:
            invalid = functools.lru_cache(maxsize=511)(original_fn)
            with pytest.raises(ValueError, match="configuration changed"):
                compat._source_dtype_cache_state(invalid, compat._NativeCaptureGraph())
    else:
        native = compat._SOURCE_DTYPE_STRONG_NAMESPACE
        assert native is not None and native["StrongLRUCache"] is live_class
        owned = native["strong_lru_cache"](original_fn, None, 512, num_shards=64)
        assert type(owned) is live_class and owned is not live
        before_graph = compat._NativeCaptureGraph()
        before_fn, before = compat._source_dtype_cache_state(owned, before_graph)
        assert before_fn is original_fn and before["kind"] == "strong"
        before_graph.take(before_fn)
        fields = dict(compat._SOURCE_DTYPE_CACHE_MEMBERS)
        state = fields["__getstate__"](owned)
        assert type(state) is dict and set(state) == {
            "version",
            "fn",
            "cache_context_fn",
            "maxsize",
            "explain",
            "num_shards",
        }
        assert state["version"] == 1 and state["fn"] is original_fn
        assert state["cache_context_fn"] is None and state["maxsize"] == 512
        assert state["explain"] is None and state["num_shards"] == 64
        attempted = []

        def changed(*args, **kwargs):
            attempted.append("callback")
            raise AssertionError("Owned altered callback must never execute")

        factory_arguments = {
            "fn": original_fn,
            "cache_context_fn": None,
            "maxsize": 512,
            "explain": None,
            "num_shards": 64,
        }
        factory_arguments[mutation] = (
            changed
            if mutation in {"fn", "cache_context_fn", "explain"}
            else 511 if mutation == "maxsize" else 32
        )
        # The native setter is pickle construction, not mutation of an already
        # initialized cache. Use a second genuine owned factory instance.
        altered = native["strong_lru_cache"](**factory_arguments)
        assert (
            type(altered) is live_class and altered is not owned and altered is not live
        )
        actual = fields["__getstate__"](altered)
        assert type(actual) is dict and set(actual) == set(state)
        for field_name in state:
            expected = 1 if field_name == "version" else factory_arguments[field_name]
            if field_name in {"fn", "cache_context_fn", "explain"}:
                assert actual[field_name] is expected
            else:
                assert (
                    type(actual[field_name]) is int and actual[field_name] == expected
                )
        if mutation == "fn":
            after_graph = compat._NativeCaptureGraph()
            after_fn, after = compat._source_dtype_cache_state(altered, after_graph)
            assert after_fn is changed
            after_graph.take(after_fn)
            assert after_graph.nodes[str(id(changed))]["code_identity"] == id(
                changed.__code__
            )
            assert (after, after_graph.nodes) != (before, before_graph.nodes)
        else:
            with pytest.raises(ValueError, match="immutable state changed"):
                compat._source_dtype_cache_state(altered, compat._NativeCaptureGraph())
        baseline_graph = compat._NativeCaptureGraph()
        baseline_fn, baseline_state = compat._source_dtype_cache_state(
            owned, baseline_graph
        )
        baseline_graph.take(baseline_fn)
        assert baseline_fn is original_fn
        assert (baseline_state, baseline_graph.nodes) == (before, before_graph.nodes)
        assert attempted == []
    final_graph = compat._NativeCaptureGraph()
    final_fn, final_state = compat._source_dtype_cache_state(live, final_graph)
    assert final_fn is original_fn
    assert (final_state, final_graph.nodes) == (original_state, graph.nodes)


def test_isfinite_functools_module_class_rejects_before_dictionary_getter(tmp_path):
    import functools
    import types
    from jax2onnx._compat import jax as compat
    from tests.extra_tests.converter import source_precision_raw_replay as raw

    capture = raw.RawReplayCapture(tmp_path / "functools-module-class")
    capture.bind_target(jnp.isfinite)
    assert capture.check_target("unchanged-isfinite")
    attempted = []

    class ForeignModule(types.ModuleType):
        def __getattribute__(self, name):
            attempted.append(name)
            raise AssertionError("Reclassified functools getters must not execute")

    setter = next(
        vars(base)["__setattr__"]
        for base in vars(type)["__mro__"].__get__(types.ModuleType)
        if "__setattr__" in vars(base)
    )
    failure = None
    setter(functools, "__class__", ForeignModule)
    try:
        try:
            compat.source_internal_dtype_capture(compat._SOURCE_DTYPE_HELPER)
        except ValueError as error:
            failure = str(error)
    finally:
        setter(functools, "__class__", types.ModuleType)
    assert failure is not None and "functools module changed" in failure
    assert capture.check_target("restored-functools-module-class")
    assert capture.records["native_outcomes"] == {} and attempted == []


@pytest.mark.parametrize("mutation", ["dict-subclass", "non-string-key"])
def test_isfinite_owned_functools_fields_reject_before_getters(mutation):
    import functools
    from jax2onnx._compat import jax as compat

    attempted = []

    def callback(*args, **kwargs):
        attempted.append("callback")
        raise AssertionError("Unowned cache must not execute")

    class ForeignFields(dict):
        def get(self, *args, **kwargs):
            attempted.append("get")
            raise AssertionError("Foreign field get must not execute")

        def __iter__(self):
            attempted.append("iter")
            raise AssertionError("Foreign field iteration must not execute")

    class ForeignKey:
        def __hash__(self):
            return 42

        def __eq__(self, other):
            attempted.append("equal")
            raise AssertionError("Foreign key equality must not execute")

    owned = functools.lru_cache(maxsize=512)(callback)
    fields = dict(vars(owned))
    if mutation == "dict-subclass":
        fields = ForeignFields(fields)
    else:
        fields[ForeignKey()] = None
    owned.__dict__ = fields
    assert vars(owned) is fields
    # 3.11 reaches the exact dictionary fence. On the Strong-cache cohort,
    # this foreign native cache class is rejected even earlier.
    reason = (
        "field dictionary changed"
        if type(owned) is compat._SOURCE_DTYPE_CACHE_CLASS
        else "native cache type changed"
    )
    with pytest.raises(ValueError, match=reason):
        compat._source_dtype_cache_state(owned, compat._NativeCaptureGraph())
    assert attempted == []


def test_isfinite_selected_ml_scalar_snapshot_covers_original_finite_set(tmp_path):
    from jax2onnx._compat import jax as compat
    from tests.extra_tests.converter import source_precision_raw_replay as raw

    target = jnp.isfinite
    capture = raw.RawReplayCapture(tmp_path / "selected-ml-scalar-set")
    capture.bind_target(target)
    assert capture.target is target and capture.check_target("selected-ml-original")
    snapshot = compat.source_internal_dtype_capture(compat._SOURCE_DTYPE_HELPER)
    assert snapshot is not None
    classes = snapshot.document["classes"]
    records = {
        int(key): record for key, record in classes.items() if "selected_ml" in record
    }
    assert set(records) == {id(value) for value in compat._SOURCE_DTYPE_ML_SCALARS}
    public = compat._SOURCE_DTYPE_ML_NAMESPACE
    extension = compat._SOURCE_DTYPE_ML_EXTENSION_NAMESPACE
    assert public is not None and extension is not None
    for identity, record in records.items():
        detail = record["selected_ml"]
        name = detail["name"]
        original = next(
            value for value in compat._SOURCE_DTYPE_ML_SCALARS if id(value) == identity
        )
        assert public[name] is original and extension[name] is original
        assert record["metaclass"] == id(type)
        assert record["mro"][0] == identity
        assert detail["heaptype"] or detail["immutabletype"]
        assert (
            type(detail["heaptype"]) is bool and type(detail["immutabletype"]) is bool
        )
        assert set(detail["member_schema"]) == set(record["members"])
        assert detail["member_schema"]["dtype"] == "simple-native-dtype"
        assert detail["dtype"]["identity"] in {id(item) for item in snapshot.retained}
    assert capture.check_target("all-selected-ml-classes-retained")
    assert capture.records["native_outcomes"] == {}


@pytest.mark.parametrize(
    "mutation",
    [
        "public-alias",
        "extension-alias",
        "jax-module-alias",
        "heap-add",
        "heap-delete",
        "heap-replace",
        "dtype-replace",
        "python-function",
        "python-property",
        "foreign-module",
        "foreign-metaclass",
    ],
)
def test_isfinite_selected_ml_scalar_mutations_reject_before_foreign_hooks(
    tmp_path, monkeypatch, mutation
):
    import sys
    import types
    from jax2onnx._compat import jax as compat
    from tests.extra_tests.converter import source_precision_raw_replay as raw

    target = jnp.isfinite
    capture = raw.RawReplayCapture(tmp_path / mutation)
    capture.bind_target(target)
    assert capture.target is target and capture.check_target("selected-ml-unchanged")
    selected = next(
        value
        for value in compat._SOURCE_DTYPE_ML_SCALARS
        if compat._SOURCE_DTYPE_TYPE_FLAGS.__get__(value) & (1 << 9)
        and not compat._SOURCE_DTYPE_TYPE_FLAGS.__get__(value) & (1 << 8)
    )
    name = next(
        name for name, value in compat._SOURCE_DTYPE_ML_EXPORTS if value is selected
    )
    attempted = []

    class ForeignModule(types.ModuleType):
        def __getattribute__(self, key):
            attempted.append(("module", key))
            raise AssertionError("Foreign ML module getters must not execute")

    class ForeignMeta(type):
        def __getattribute__(self, key):
            attempted.append(("metaclass", key))
            raise AssertionError("Foreign ML class getters must not execute")

        def __eq__(self, other):
            attempted.append(("equal",))
            raise AssertionError("Foreign ML class equality must not execute")

        def __hash__(self):
            attempted.append(("hash",))
            raise AssertionError("Foreign ML class hash must not execute")

    class ForeignClass(metaclass=ForeignMeta):
        pass

    def changed(*args, **kwargs):
        attempted.append(("call",))
        raise AssertionError("Altered ML class callbacks must not execute")

    with monkeypatch.context() as patch:
        if mutation == "public-alias":
            patch.setitem(compat._SOURCE_DTYPE_ML_NAMESPACE, name, ForeignClass)
        elif mutation == "extension-alias":
            patch.setitem(
                compat._SOURCE_DTYPE_ML_EXTENSION_NAMESPACE, name, ForeignClass
            )
        elif mutation == "jax-module-alias":
            patch.setitem(
                compat._SOURCE_DTYPE_NAMESPACE, "ml_dtypes", ForeignModule("ml_dtypes")
            )
        elif mutation == "heap-add":
            patch.setattr(selected, "source_finite_added", 7, raising=False)
        elif mutation == "heap-delete":
            patch.delattr(selected, "__format__")
        elif mutation == "heap-replace":
            patch.setattr(selected, "__eq__", changed)
        elif mutation == "dtype-replace":
            patch.setattr(selected, "dtype", np.dtype("float32"))
        elif mutation == "python-function":
            patch.setattr(selected, "source_finite_added", changed, raising=False)
        elif mutation == "python-property":
            patch.setattr(
                selected, "source_finite_added", property(changed), raising=False
            )
        elif mutation == "foreign-module":
            patch.setitem(
                sys.modules,
                "ml_dtypes._ml_dtypes_ext",
                ForeignModule("ml_dtypes._ml_dtypes_ext"),
            )
        else:
            patch.setattr(selected, "source_finite_added", ForeignClass, raising=False)
        assert not capture.check_target("changed-selected-ml-class")
        assert attempted == [] and capture.records["native_outcomes"] == {}
    assert capture.target is target and capture.check_target(
        "restored-selected-ml-class"
    )
    assert attempted == []


def test_isfinite_direct_numpy_classes_have_exact_native_witness_roles(tmp_path):
    from jax2onnx._compat import jax as compat
    from tests.extra_tests.converter import source_precision_raw_replay as raw

    capture = raw.RawReplayCapture(tmp_path / "isfinite-direct-numpy-classes")
    capture.bind_target(jnp.isfinite)
    assert capture.target is jnp.isfinite and capture.check_target(
        "original-direct-classes"
    )
    assert capture.target_witness is not None
    for name, original in compat._SOURCE_ISFINITE_NUMPY_CLASSES:
        snapshot = compat.source_internal_dtype_capture(original)
        assert snapshot is not None
        assert snapshot.document["role"] == "source-isfinite-numpy-class"
        assert snapshot.document["selected_name"] == "numpy." + name
        assert snapshot.document["class_ref"] == str(id(original))
        record = snapshot.document["classes"][str(id(original))]
        assert record["identity"] == id(original) and record["metaclass"] == id(type)
        assert record["selected_numpy"]["immutabletype"] is True
        assert record["selected_numpy"]["name"] == name
        assert record["mro"][0] == id(original)
        assert any(item is original for item in snapshot.retained)
        assert any(
            path[-2:] == ("global:np", "attribute:" + name)
            and role == "source-isfinite-numpy-class"
            for path, role in capture.target_witness.selected.items()
        )
    assert capture.check_target("all-original-direct-class-roles-retained")
    assert capture.records["native_outcomes"] == {}


@pytest.mark.parametrize(
    "scope,name",
    [
        (scope, name)
        for scope in ("public", "definition")
        for name in ("floating", "complexfloating", "bool_")
    ]
    + [
        ("public-module", "numpy"),
        ("native-module", "numpy._core._multiarray_umath"),
        ("foreign-metaclass", "complexfloating"),
        ("typeinfo-key", "bool_"),
    ],
)
def test_isfinite_direct_numpy_alias_mutations_reject_without_foreign_hooks(
    tmp_path, monkeypatch, scope, name
):
    import sys
    import types
    from jax2onnx._compat import jax as compat
    from tests.extra_tests.converter import source_precision_raw_replay as raw

    target = jnp.isfinite
    capture = raw.RawReplayCapture(tmp_path / (scope + "-" + name))
    capture.bind_target(target)
    assert capture.target is target and capture.check_target(
        "original-direct-numpy-aliases"
    )
    attempted = []

    class ForeignMeta(type):
        def __getattribute__(self, key):
            attempted.append(("class", key))
            raise AssertionError("Foreign NumPy class getters must not execute")

        def __eq__(self, other):
            attempted.append(("equal",))
            raise AssertionError("Foreign NumPy class equality must not execute")

        def __hash__(self):
            attempted.append(("hash",))
            raise AssertionError("Foreign NumPy class hash must not execute")

    class ForeignClass(metaclass=ForeignMeta):
        pass

    class ForeignModule(types.ModuleType):
        def __getattribute__(self, key):
            attempted.append(("module", key))
            raise AssertionError("Foreign NumPy module getters must not execute")

    collision_hash = hash("bool")

    class ForeignKey:
        def __hash__(self):
            return collision_hash

        def __eq__(self, other):
            attempted.append(("typeinfo-key-equality",))
            raise AssertionError("Non-string typeinfo keys must reject before lookup")

    if scope == "typeinfo-key":
        mapping = compat._SOURCE_DTYPE_NUMPY_TYPEINFO
        assert type(mapping) is dict and all(type(key) is str for key in mapping)
        assert sum(hash(key) == collision_hash for key in mapping) == 1
        original = mapping.pop("bool")
        key = ForeignKey()
        mapping[key] = original
        try:
            assert not capture.check_target("non-string-typeinfo-key")
            assert attempted == [] and capture.records["native_outcomes"] == {}
        finally:
            del mapping[key]
            mapping["bool"] = original
        assert capture.check_target("restored-typeinfo-keys")
        assert attempted == []
        return

    with monkeypatch.context() as patch:
        if scope in {"public", "foreign-metaclass"}:
            assert compat.source_internal_dtype_capture(ForeignClass) is None
            patch.setitem(compat._SOURCE_DTYPE_NUMPY_NAMESPACE, name, ForeignClass)
        elif scope == "definition":
            if name == "bool_":
                patch.setitem(compat._SOURCE_DTYPE_NUMPY_TYPEINFO, "bool", ForeignClass)
            else:
                patch.setitem(
                    compat._SOURCE_DTYPE_NUMPY_NATIVE_NAMESPACE, name, ForeignClass
                )
        else:
            patch.setitem(sys.modules, name, ForeignModule(name))
        assert not capture.check_target("changed-direct-numpy-alias")
        assert attempted == [] and capture.records["native_outcomes"] == {}
    assert capture.target is target and capture.check_target(
        "restored-direct-numpy-alias"
    )
    assert attempted == []


@pytest.mark.parametrize("entry", ["recipe", "target"])
@pytest.mark.parametrize(
    "mutation", ["sys-module", "module-class", "generic-alias", "raw-alias"]
)
def test_source_numpy_entry_guards_reject_before_import_or_foreign_hooks(
    tmp_path, monkeypatch, entry, mutation
):
    import sys
    import types
    from tests.extra_tests.converter import source_precision_raw_replay as raw
    from tests.extra_tests.converter.source_precision_support import (
        SourcePreparationError,
    )

    original_module = np
    namespace = vars(types.ModuleType)["__dict__"].__get__(original_module)
    native_setattr = next(
        vars(base)["__setattr__"]
        for base in vars(type)["__mro__"].__get__(types.ModuleType)
        if "__setattr__" in vars(base)
    )
    capture = raw.RawReplayCapture(tmp_path / (entry + "-" + mutation))
    unbound = raw.RawReplayCapture(tmp_path / "unbound-target")
    capture.bind_target(jnp.isfinite)
    assert capture.check_target("original-entry-guard-state")
    assert raw._capture_recipe(None) is None
    attempted = []

    class ForeignModule(types.ModuleType):
        def __getattribute__(self, name):
            attempted.append(("module", name))
            raise AssertionError(
                "Entry guard must reject before foreign module getters"
            )

    class ForeignMeta(type):
        def __getattribute__(self, name):
            attempted.append(("class", name))
            raise AssertionError("Entry guard must reject before foreign class getters")

        def __instancecheck__(self, instance):
            attempted.append(("instancecheck",))
            raise AssertionError("Entry guard must reject before foreign type checks")

        def __eq__(self, other):
            attempted.append(("equal",))
            raise AssertionError("Entry guard must reject before foreign equality")

        def __hash__(self):
            attempted.append(("hash",))
            raise AssertionError("Entry guard must reject before foreign hashing")

    class ForeignGeneric(metaclass=ForeignMeta):
        pass

    reclassified = False
    with monkeypatch.context() as patch:
        try:
            if mutation == "sys-module":
                patch.setitem(sys.modules, "numpy", ForeignModule("numpy"))
            elif mutation == "module-class":
                native_setattr(original_module, "__class__", ForeignModule)
                reclassified = True
            elif mutation == "generic-alias":
                patch.setitem(namespace, "generic", ForeignGeneric)
            else:
                patch.setattr(raw, "np", ForeignModule("numpy"))
            if entry == "recipe":
                failure = None
                try:
                    raw._capture_recipe(None)
                except ValueError as error:
                    failure = str(error)
                assert failure == "Source NumPy capture dependency changed"
            else:
                assert not capture.check_target("changed-entry-module")
                failure = None
                try:
                    unbound.bind_target(jnp.isfinite)
                except SourcePreparationError as error:
                    failure = error.category
                assert failure == "replay-lane-gap"
            assert attempted == []
            assert capture.records["native_outcomes"] == {}
        finally:
            if reclassified:
                native_setattr(original_module, "__class__", types.ModuleType)
    assert raw._capture_recipe(None) is None
    assert capture.target is jnp.isfinite and capture.check_target(
        "restored-entry-guard-state"
    )
    assert attempted == []


def test_source_numpy_entry_guard_keeps_exact_original_recipe_encoding():
    from tests.extra_tests.converter import source_precision_raw_replay as raw

    assert raw._capture_recipe(None) is None
    scalar = np.int64(7)
    assert raw._capture_recipe(scalar) == {
        "scalar_dtype": "int64",
        "dtype_str": "<i8",
        "bytes_hex": "0700000000000000",
    }
    array = np.asarray([1, -2], dtype=np.int32)
    assert raw._capture_recipe(array) == {
        "array_dtype": "int32",
        "shape": [2],
        "sha256": hashlib.sha256(bytes.fromhex("01000000feffffff")).hexdigest(),
    }


@pytest.fixture(scope="module")
def _keepdims_callback_models_v2(tmp_path_factory):
    import onnx
    from jax2onnx.plugins._post_check_onnx_graph import expect_graph

    registrations, _ = inventory_owner.raw_registrations()
    result = {}
    for name, operation in (("all", "ReduceMin"), ("any", "ReduceSum")):
        metadata = next(
            item[2] for item in registrations if item[0] == "plugin::jax.numpy." + name
        )
        case = metadata["testcases"][1]
        original = case["callable"]
        values = case["input_values"]
        legacy = user_interface.to_onnx(
            original,
            inputs=[jax.ShapeDtypeStruct(value.shape, value.dtype) for value in values],
            model_name="original_" + name + "_keepdims",
            opset=23,
            return_mode="proto",
        )
        callers = {}
        for caller in ("raw-x64-off", "raw-x64-on"):
            directory = tmp_path_factory.mktemp("original-" + name + "-" + caller)
            scope = _discrete_protocol_scope(dict(case))
            scope[3]["rows"]["plugin::synthetic::0"]["capture_original_object"] = True
            descriptor = _descriptors(
                scope, selection=[("plugin::synthetic", 0, caller)]
            )[0]
            function = _function(
                scope,
                descriptor,
                directory,
                runtime="both",
                expected={"runtime": "validated", "numerics": "validated"},
            )
            function()
            record = function._source_cell.history[-1]
            source = onnx.load_model(record.replay_capture["artifact"]["path"])
            callers[caller] = (source, scope, function)
        source, scope, function = callers["raw-x64-on"]
        result[name + "-callers"] = callers
        old_callback = expect_graph(
            [f"Cast:2x3 -> Cast:2x3 -> {operation}:2x1 -> Cast:2x1"],
            no_unused_inputs=True,
        )
        result[name] = (case, legacy, source, scope, function, old_callback)
    return result


def test_keepdims_callback_v2_preserves_original_two_contracts():
    frozen = _ORIGINAL[0]
    registrations, _ = inventory_owner.raw_registrations()
    for name in ("all", "any"):
        origin = "plugin::jax.numpy." + name
        metadata = next(item[2] for item in registrations if item[0] == origin)
        rows = [
            row
            for row in frozen["cases"]
            if frozen["registrations"][row[0]][0] == origin
        ]
        for row in rows:
            case = metadata["testcases"][row[1]]
            unresolved = []
            actual = inventory_owner.canonical_contract(
                {
                    key: value
                    for key, value in case.items()
                    if key not in {"testcase", "description"}
                },
                unresolved=unresolved,
            )
            assert unresolved == []
            previous = frozen["contracts"][row[5]]
            if row[1] != 1:
                assert actual == previous
                continue
            old_hash = {
                "all": "e321afc95b4026216a265819047b15d30cf7280c0e7f6a781ccc57d17aa566ca",
                "any": "005db3792fe595103f715e86cc6a81ccf3cb22cae4367123fe15c37685fd3595",
            }[name]
            assert inventory_owner._contract_hash(previous) == old_hash
            assert {
                key: value
                for key, value in actual.items()
                if key != "post_check_onnx_graph"
            } == {
                key: value
                for key, value in previous.items()
                if key != "post_check_onnx_graph"
            }
            callback = actual["post_check_onnx_graph"]
            assert callback == {
                "partial": {
                    "callable": "jax2onnx.plugins._post_check_onnx_graph._check_boolean_axis_keepdims_v2"
                },
                "args": {"tuple": []},
                "kwargs": {
                    "operation": "ReduceMin" if name == "all" else "ReduceSum",
                    "input_shape": {"tuple": [2, 3]},
                    "axis": 1,
                    "output_shape": {"tuple": [2, 1]},
                },
            }
            assert inventory_owner._contract_hash(actual) != old_hash


@pytest.mark.parametrize("name", ["all", "any"])
@pytest.mark.parametrize("caller", ["raw-x64-off", "raw-x64-on"])
def test_keepdims_callback_v2_accepts_actual_legacy_and_source_graphs(
    _keepdims_callback_models_v2, name, caller
):
    case, legacy, _, _, _, old_callback = _keepdims_callback_models_v2[name]
    source, scope, function = _keepdims_callback_models_v2[name + "-callers"][caller]
    callback = case["post_check_onnx_graph"]
    assert old_callback(legacy) is True
    assert old_callback(source) is False
    assert callback(legacy) is True
    assert callback(source) is True
    record = function._source_cell.history[-1]
    assert record.raw_callback_status == "passed"
    assert record.actual_settings.x64_enabled is (caller == "raw-x64-on")
    assert (
        record.structure is record.runtime is record.numerics is CheckStatus.VALIDATED
    )
    assert function._source_outcome["stage_contract_passed"] is True
    assert function._source_outcome["capability_status"] == "validated"
    expected = np.asarray([[False], [True]], dtype=np.bool_)
    _predicate_retained_outputs(record.replay_capture, [expected])
    manifest = json.loads(
        Path(record.replay_capture["declared-typed-trace"]["path"]).read_bytes()
    )
    assert inventory_owner._source_bitwise_closure(manifest, source) is not None
    _retained_stage_record(function)


@pytest.mark.parametrize("name", ["all", "any"])
@pytest.mark.parametrize("variant", ["legacy", "source"])
@pytest.mark.parametrize(
    "mutation",
    [
        "axis-word",
        "axis-attribute-kind",
        "keepdims",
        "reduction-kind",
        "cast-dtype",
        "input-edge",
        "output-edge",
        "restore-shape",
        "broadcast-axis",
        "extra-node",
        "extra-initializer",
        "axis-literal-dtype",
        "node-overload",
    ],
)
def test_keepdims_callback_v2_rejects_wrong_typed_connected_contracts(
    tmp_path, _keepdims_callback_models_v2, name, variant, mutation
):
    import onnx
    from onnx import helper, numpy_helper

    case, legacy, source, scope, function, _ = _keepdims_callback_models_v2[name]
    original = source if variant == "source" else legacy
    original_payload = original.SerializeToString(deterministic=True)
    model = copy.deepcopy(original)
    operation = "ReduceMin" if name == "all" else "ReduceSum"
    reduction = next(node for node in model.graph.node if node.op_type == operation)
    axes = next(
        value for value in model.graph.initializer if value.name == reduction.input[1]
    )
    casts = [node for node in model.graph.node if node.op_type == "Cast"]
    if mutation == "axis-word":
        axes.CopyFrom(
            numpy_helper.from_array(np.asarray([0], dtype=np.int64), name=axes.name)
        )
    elif mutation == "axis-attribute-kind":
        attribute = next(
            item for item in reduction.attribute if item.name == "keepdims"
        )
        attribute.CopyFrom(helper.make_attribute("keepdims", float(attribute.i)))
    elif mutation == "keepdims":
        next(item for item in reduction.attribute if item.name == "keepdims").i = (
            0 if variant == "legacy" else 1
        )
    elif mutation == "reduction-kind":
        reduction.op_type = "ReduceSum" if operation == "ReduceMin" else "ReduceMin"
    elif mutation == "cast-dtype":
        next(item for item in casts[1].attribute if item.name == "to").i = (
            onnx.TensorProto.INT32
        )
    elif mutation == "input-edge":
        reduction.input[0] = model.graph.input[0].name
    elif mutation == "output-edge":
        model.graph.output[0].name = casts[0].output[0]
    elif mutation == "restore-shape":
        if variant == "legacy":
            model.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 2
        else:
            restore = next(
                node for node in model.graph.node if node.op_type == "Reshape"
            )
            producer = next(
                node for node in model.graph.node if restore.input[1] in node.output
            )
            literal = next(
                value
                for value in model.graph.initializer
                if value.name == producer.input[1]
            )
            literal.CopyFrom(
                numpy_helper.from_array(
                    np.asarray([2], dtype=np.int64), name=literal.name
                )
            )
    elif mutation == "broadcast-axis":
        if variant == "source":
            concat = next(node for node in model.graph.node if node.op_type == "Concat")
            next(item for item in concat.attribute if item.name == "axis").i = 1
        else:
            reduction.input[1] = casts[0].output[0]
    elif mutation == "extra-node":
        model.graph.node.append(
            helper.make_node(
                "Identity", [model.graph.input[0].name], ["unclaimed_dead_value"]
            )
        )
    elif mutation == "extra-initializer":
        model.graph.initializer.append(
            numpy_helper.from_array(
                np.asarray([1], dtype=np.int64), name="unclaimed_dead_literal"
            )
        )
    elif mutation == "node-overload":
        reduction.overload = "foreign-overload"
    else:
        axes.CopyFrom(
            numpy_helper.from_array(np.asarray([1], dtype=np.int32), name=axes.name)
        )
    assert case["post_check_onnx_graph"](model) is False
    if variant == "source":
        record = function._source_cell.history[-1]
        manifest = json.loads(
            Path(record.replay_capture["declared-typed-trace"]["path"]).read_bytes()
        )
        assert inventory_owner._source_bitwise_closure(manifest, model) is None
    assert original.SerializeToString(deterministic=True) == original_payload


@pytest.mark.parametrize("name", ["all", "any"])
@pytest.mark.parametrize(
    "mutation", ["sparse-storage", "training-info", "foreign-opset", "quantization"]
)
def test_keepdims_saved_ir_reader_retains_complete_protobuf_boundary(
    _keepdims_callback_models_v2, name, mutation
):
    from onnx import helper, numpy_helper

    case, _, source, _, function, _ = _keepdims_callback_models_v2[name]
    model = copy.deepcopy(source)
    if mutation == "sparse-storage":
        model.graph.sparse_initializer.append(
            helper.make_sparse_tensor(
                numpy_helper.from_array(
                    np.asarray([1], dtype=np.int64), name="unclaimed_sparse"
                ),
                numpy_helper.from_array(np.asarray([0], dtype=np.int64)),
                [1],
            )
        )
    elif mutation == "training-info":
        model.training_info.add()
    elif mutation == "foreign-opset":
        model.opset_import.append(helper.make_opsetid("unclaimed.domain", 1))
    else:
        model.graph.quantization_annotation.add().tensor_name = model.graph.input[
            0
        ].name
    record = function._source_cell.history[-1]
    manifest = json.loads(
        Path(record.replay_capture["declared-typed-trace"]["path"]).read_bytes()
    )
    # IR normalization need not preserve these full-Proto storage declarations.
    # The independent saved-artifact reader remains an additional required gate.
    assert inventory_owner._source_bitwise_closure(manifest, model) is None


@pytest.mark.parametrize("name", ["all", "any"])
def test_keepdims_callback_v2_rejects_foreign_ir_graph_ownership(
    _keepdims_callback_models_v2, name
):
    import onnx_ir as ir

    case, _, source, _, _, _ = _keepdims_callback_models_v2[name]
    for mutation in ("node-owner", "value-owner"):
        model = ir.serde.deserialize_model(source)
        foreign = ir.Graph([], [], nodes=[])
        if mutation == "node-owner":
            next(iter(model.graph)).graph = foreign
        else:
            model.graph.inputs[0]._graph = foreign
        assert case["post_check_onnx_graph"](model) is False
