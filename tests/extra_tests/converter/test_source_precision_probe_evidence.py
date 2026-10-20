# tests/extra_tests/converter/test_source_precision_probe_evidence.py

"""Immutable artifact stages bind original source, environment and exact payloads."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import FrozenInstanceError, replace
import hashlib
import importlib.metadata
import json
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax2onnx.precision import CheckStatus
from scripts import source_precision_probe as probe
from scripts.source_precision_inventory import build_inventory, raw_registrations
from tests.extra_tests.converter.source_precision_diagnostics import (
    typed_trace_manifest,
)
from tests.extra_tests.converter.source_precision_support import (
    SourceProvenance,
    SourceSettings,
)


_REFERENCE_BIAS: float = 1.0


def _source_with_global(x):
    return x + np.asarray(_REFERENCE_BIAS, np.float32)


def _registration(function=lambda x: x + 1, **options):
    case = {
        "testcase": "evidence",
        "callable": function,
        "input_shapes": [(2,)],
        **options,
    }
    registration = (
        "plugin::focused",
        "sample",
        {"context": "sample", "component": "evidence", "testcases": [case]},
    )
    inventory = build_inventory([registration], [], manual=[], optional={})
    return probe.build_probe_plan(inventory), registration


@pytest.fixture(autouse=True)
def _context(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    import onnxruntime

    previous = SourceSettings.observe()

    def forbidden(*args: Any, **kwargs: Any):
        pytest.fail("Artifact stage evidence must not create runtime sessions")

    monkeypatch.setattr(onnxruntime, "InferenceSession", forbidden)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous.x64_enabled)
        jax.config.update(
            "jax_default_matmul_precision", previous.default_matmul_precision
        )
        jax.config.update("jax_platforms", previous.jax_platforms)
        jax.config.update("jax_numpy_dtype_promotion", previous.numpy_dtype_promotion)
        jax.config.update("jax_numpy_rank_promotion", previous.numpy_rank_promotion)


def test_exact_first_selection_preflights_all_36_cells_without_executing_models() -> (
    None
):
    plan = probe.build_probe_plan(json.loads(probe.DEFAULT_BASELINE.read_text()))
    registrations, _ = raw_registrations()
    selected = frozenset(probe.FIRST_DIAGNOSTIC_CASES)
    probe.preflight_cases(plan, registrations, selected)
    assert len(selected) == 18
    assert len(plan.rows) == 1852 and len(plan.cells) == 3704
    cells = [
        cell
        for cell in plan.cells
        if SourceProvenance(cell.origin, cell.ordinal) in selected
    ]
    assert len(cells) == 36
    unselected = [
        cell
        for cell in plan.cells
        if SourceProvenance(cell.origin, cell.ordinal) not in selected
    ]
    assert len(unselected) == 3668
    assert len({cell.identity for cell in cells}) == 36
    assert (
        hashlib.sha256(
            json.dumps(
                sorted(cell.identity for cell in cells), separators=(",", ":")
            ).encode()
        ).hexdigest()
        == "9fb4b7c179e83cf609c7cff4021caf30a6689d314f3c21b4af3b3f94262dd585"
    )
    optional = {
        (row.origin, row.ordinal)
        for row in plan.rows
        if row.display[0] in {"examples.maxtext", "examples.maxdiffusion"}
    }
    optional_cells = {
        cell.identity for cell in plan.cells if (cell.origin, cell.ordinal) in optional
    }
    assert len(optional_cells) == 100
    assert optional_cells <= {cell.identity for cell in unselected}
    assert {cell.scenario for cell in cells} == set(probe.SCENARIOS)
    assert all(
        row.criteria is None
        for row in plan.rows
        if SourceProvenance(row.origin, row.ordinal) in selected
    )


def test_prepare_to_diagnostic_appends_evidence_and_observes_actual_inner_settings() -> (
    None
):
    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_default_matmul_precision", "high")
    plan, registration = _registration()
    prepared = probe.execute_probe_plan(
        plan, [registration], stage="prepare", limit=1, scenario="raw-x64-on"
    )
    prior = prepared.cells[1]
    assert prior.stage == "source-prepared" and len(prior.history) == 1
    completed = probe.execute_probe_plan(
        prepared, [registration], stage="diagnostic", limit=1, scenario="raw-x64-on"
    )
    cell = completed.cells[1]
    assert cell.stage == "diagnostic-artifact-checked", cell.detail
    assert len(cell.history) == 2 and cell.history[0] == prior.history[0]
    assert cell.history[1].previous_sha256 == cell.history[0].sha256
    assert prepared.cells[1] == prior and plan.cells[1].stage == "scheduled"
    evidence = cell.history[-1]
    assert evidence.cell_identity == cell.identity
    assert (
        evidence.requested_x64
        is evidence.actual_settings.x64_enabled
        is evidence.export_settings.x64_enabled
        is True
    )
    assert (
        evidence.actual_settings.default_matmul_precision
        is evidence.export_settings.default_matmul_precision
        is None
    )
    assert (
        evidence.actual_settings.numpy_dtype_promotion
        == evidence.export_settings.numpy_dtype_promotion
    )
    assert (
        not jax.config.x64_enabled and jax.config.jax_default_matmul_precision == "high"
    )
    assert dict(evidence.environment.versions)["jax"] == importlib.metadata.version(
        "jax"
    )
    assert evidence.environment.architecture and evidence.environment.python
    assert len(evidence.original_trace_sha256) == len(evidence.artifact_sha256) == 64
    assert evidence.structure is evidence.schema is CheckStatus.VALIDATED
    assert (
        evidence.lowering_audit
        is evidence.runtime
        is evidence.numerics
        is CheckStatus.UNVERIFIED
    )
    with pytest.raises(FrozenInstanceError):
        evidence.stage = "forged"


def test_prepare_to_product_is_an_actual_second_stage_and_exact_case_selector_is_not_origin_only() -> (
    None
):
    plan, registration = _registration()
    metadata = registration[2]
    metadata["testcases"].append({**metadata["testcases"][0], "testcase": "second"})
    plan = probe.build_probe_plan(
        build_inventory([registration], [], manual=[], optional={})
    )
    cases = frozenset({SourceProvenance("plugin::focused", 1)})
    prepared = probe.execute_probe_plan(
        plan, [registration], stage="prepare", limit=2, cases=cases
    )
    completed = probe.execute_probe_plan(
        prepared, [registration], stage="product", limit=2, cases=cases
    )
    assert all(cell.stage == "scheduled" for cell in completed.cells[:2])
    assert all(
        cell.stage == "artifact-checked" and len(cell.history) == 2
        for cell in completed.cells[2:]
    )
    assert all(
        cell.lowering_audit is CheckStatus.VALIDATED for cell in completed.cells[2:]
    )


@pytest.mark.parametrize("stage", ["prepare", "diagnostic"])
def test_requested_settings_override_active_contexts_and_restore_effective_and_global_callers(
    stage: str,
) -> None:
    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_default_matmul_precision", "highest")
    jax.config.update("jax_platforms", "cpu")
    jax.config.update("jax_numpy_dtype_promotion", "standard")
    jax.config.update("jax_numpy_rank_promotion", "allow")
    global_caller = SourceSettings.observe()
    plan, registration = _registration(lambda x: x)
    with (
        jax.enable_x64(True),
        jax.default_matmul_precision("high"),
        jax.numpy_dtype_promotion("strict"),
        jax.numpy_rank_promotion("raise"),
    ):
        effective_caller = SourceSettings.observe()
        completed = probe.execute_probe_plan(
            plan, [registration], stage=stage, limit=1, scenario="raw-x64-off"
        )
        cell = completed.cells[0]
        assert cell.stage == (
            "source-prepared" if stage == "prepare" else "diagnostic-artifact-checked"
        ), cell.detail
        expected = SourceSettings(False, None, "cpu", "strict", "raise")
        assert cell.history[-1].actual_settings == expected
        if stage == "diagnostic":
            assert cell.history[-1].export_settings == expected
        assert SourceSettings.observe() == effective_caller
    assert SourceSettings.observe() == global_caller


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("jax_enable_x64", True),
        ("jax_default_matmul_precision", "high"),
        ("jax_numpy_dtype_promotion", "strict"),
        ("jax_numpy_rank_promotion", "raise"),
        # Both spellings select CPU only; this witness never requests a GPU.
        ("jax_platforms", "cpu,cpu"),
    ],
)
@pytest.mark.parametrize("raises", [False, True])
@pytest.mark.parametrize("active_context", [False, True])
def test_callable_mutation_restores_all_recorded_settings_on_rejection_or_exception(
    name: str, value: Any, raises: bool, active_context: bool
) -> None:
    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_default_matmul_precision", None)
    jax.config.update("jax_numpy_dtype_promotion", "standard")
    jax.config.update("jax_numpy_rank_promotion", "allow")
    jax.config.update("jax_platforms", "cpu")
    previous = SourceSettings.observe()

    def mutating_source(x):
        jax.config.update(name, value)
        if raises:
            raise RuntimeError("mutation exception witness")
        return x

    plan, registration = _registration(mutating_source)
    with ExitStack() as contexts:
        if active_context:
            contexts.enter_context(jax.enable_x64(True))
            contexts.enter_context(jax.default_matmul_precision("highest"))
            contexts.enter_context(jax.numpy_dtype_promotion("strict"))
            contexts.enter_context(jax.numpy_rank_promotion("raise"))
        effective_caller = SourceSettings.observe()
        completed = probe.execute_probe_plan(
            plan, [registration], stage="prepare", limit=1
        )
        cell = completed.cells[0]
        assert cell.source_reference is CheckStatus.FAILED
        assert (
            "mutation exception witness" if raises else "changed caller JAX settings"
        ) in cell.detail
        assert cell.history[-1].actual_settings == replace(
            effective_caller, x64_enabled=False, default_matmul_precision=None
        )
        assert SourceSettings.observe() == effective_caller
    assert SourceSettings.observe() == previous


def test_reference_or_environment_drift_retains_original_stage_and_blocks_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, registration = _registration(_source_with_global)
    prepared = probe.execute_probe_plan(plan, [registration], stage="prepare", limit=1)
    monkeypatch.setitem(globals(), "_REFERENCE_BIAS", 2.0)
    failed = probe.execute_probe_plan(
        prepared, [registration], stage="diagnostic", limit=1
    )
    cell = failed.cells[0]
    assert cell.stage == "source-drift" and cell.source_reference is CheckStatus.FAILED
    assert len(cell.history) == 2 and cell.history[0] == prepared.cells[0].history[0]
    assert cell.history[0].references != cell.history[1].references
    assert cell.artifact_sha256 is None
    # A separate fresh plan keeps environment drift explicit too.
    plan, registration = _registration()
    prepared = probe.execute_probe_plan(plan, [registration], stage="prepare", limit=1)
    observed = probe.ProbeEnvironment.observe()
    monkeypatch.setattr(
        probe.ProbeEnvironment,
        "observe",
        classmethod(lambda cls: replace(observed, python="changed")),
    )
    changed = probe.execute_probe_plan(
        prepared, [registration], stage="product", limit=1
    )
    assert changed.cells[0].stage == "source-drift"


def test_configuration_failure_keeps_supplied_payload_and_is_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = np.asarray([1 + 2**-30], np.float64)
    plan, registration = _registration(input_shapes=None, input_values=[value])
    failed = probe.execute_probe_plan(plan, [registration], stage="prepare", limit=1)
    cell = failed.cells[0]
    assert cell.source_reference is CheckStatus.FAILED
    assert cell.category == "configuration-conflict"
    assert cell.history[0].actual_settings.x64_enabled is False
    assert cell.history[0].supplied_inputs[0].logical.dtype == "float64"
    assert not cell.history[0].inputs and not cell.history[0].references

    def forbidden(*args: Any, **kwargs: Any):
        pytest.fail("Known failure must remain terminal")

    monkeypatch.setattr(probe, "prepare_source_case", forbidden)
    repeated = probe.execute_probe_plan(
        failed, [registration], stage="diagnostic", limit=1, scenario="raw-x64-off"
    )
    assert repeated.cells[0] == cell


def test_trace_lineage_contains_nested_bodies_captures_and_scan_role_partitions() -> (
    None
):
    captured = np.asarray([2, 3], np.float32)
    closed = jax.make_jaxpr(lambda x: jax.jit(lambda y: y + captured)(x))(
        jax.ShapeDtypeStruct((2,), np.float32)
    )
    manifest = typed_trace_manifest(closed)
    encoded = json.dumps(manifest, sort_keys=True)
    assert "sha256" in encoded and "const_payloads" in encoded
    assert "0x" not in encoded
    assert not probe._trace_pending(manifest)
    closed = jax.make_jaxpr(
        lambda x: jax.lax.scan(lambda c, y: (c + y, c), jnp.zeros((2,), x.dtype), x)
    )(jax.ShapeDtypeStruct((3, 2), np.float32))
    scan = next(
        row
        for row in typed_trace_manifest(closed)["equations"]
        if row["primitive"] == "scan"
    )
    assert scan["scan_role_arity"] == (0, 1, 1)
    if "ft_in" in scan["params"]:
        assert scan["params"]["ft_in"]["flat_tree_role_partitions"] == [0, 1, 1]
        assert scan["params"]["ft_out"]["flat_tree_role_partitions"] == [1, 1]


def test_unknown_trace_parameter_retains_pending_lineage_instead_of_address_repr() -> (
    None
):
    class Opaque:
        def __repr__(self):
            pytest.fail("Opaque trace lineage must not use repr")

    closed = jax.make_jaxpr(lambda x: x + 1)(jax.ShapeDtypeStruct((1,), np.float32))
    eqn = closed.jaxpr.eqns[0].replace(params={"unreviewed": Opaque()})
    manifest = typed_trace_manifest(closed.jaxpr.replace(eqns=[eqn]))
    assert probe._trace_pending(manifest)
    assert manifest["unresolved_parameters"]
    assert "0x" not in json.dumps(manifest)


def test_plan_only_never_records_execution_environment_or_prepared_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any):
        pytest.fail("Planning must not observe an execution environment")

    monkeypatch.setattr(probe.ProbeEnvironment, "observe", forbidden)
    plan, _ = _registration()
    assert all(not cell.history for cell in plan.cells)
    assert "original_trace_json" not in json.dumps(plan.to_dict())


def test_known_checker_failure_remains_failed_separately_from_lowering_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import onnx
    from tests.extra_tests.converter import source_precision_support as support

    def invalid_artifact(*args: Any, **kwargs: Any):
        raise onnx.checker.ValidationError("focused schema witness")

    monkeypatch.setattr(support, "_export_prepared", invalid_artifact)
    plan, registration = _registration()
    failed = probe.execute_probe_plan(plan, [registration], stage="diagnostic", limit=1)
    cell = failed.cells[0]
    assert cell.schema is cell.artifact is CheckStatus.FAILED
    assert cell.lowering_audit is CheckStatus.UNVERIFIED
    assert cell.history[0].schema is cell.history[0].artifact is CheckStatus.FAILED
    assert "focused schema witness" in cell.detail
