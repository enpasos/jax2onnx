# tests/extra_tests/converter/test_source_precision_metadata.py

"""Bounded real SOURCE metadata execution and complete, non-acceptance accounting."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
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
        adapters={},
        evidence_directory=tmp_path,
        stage=stage,
        expected_statuses=expected or {"source_reference": "validated"},
        runtime_optimization=runtime,
    )


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


def test_complete_explicit_criterion_uses_same_artifact_and_retained_target(tmp_path):
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
    function = _function(
        scope,
        _descriptors(scope)[1],
        tmp_path,
        runtime="disabled",
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
    runtime = function._source_cell.history[-1].replay_capture["metadata_runtime"]
    assert runtime["artifact_sha256"] == outcome["artifact_sha256"]
    assert runtime["reference_role"] == "original-prepared-eager"
    assert runtime["criteria"] == {
        "mode": "tolerant",
        "rtol": 0.0,
        "atol": 0.0,
        "equal_nan": True,
    }
    assert len(runtime["samples"]) == 2
    assert runtime["target"]["providers"] == ("CPUExecutionProvider",)
    assert outcome["capability_status"] == "validated"


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
