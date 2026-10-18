# tests/extra_tests/converter/test_source_precision_corpus_numerics.py

"""Finite predicate/provenance tests, not execution of the24-cell packet."""

import copy
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnxruntime as ort
import pytest

from jax2onnx.precision import CheckStatus, get_precision_report
from scripts import source_precision_numerical_probe as probe
from scripts import source_precision_probe as artifact_probe
from scripts.source_precision_inventory import raw_registrations
from tests.extra_tests.converter import source_precision_corpus_numerics as numerical
from tests.extra_tests.converter.source_precision_corpus_oracles import (
    HISTORICAL_BASELINE_SHA256,
    counterpart_for,
)
from tests.extra_tests.converter.source_precision_diagnostics import DiagnosticValidator
from tests.extra_tests.converter.source_precision_support import (
    SourceProvenance,
    SourceScenario,
    SourceSettings,
    _global_source_configuration,
    prepare_source_case,
    raw_source_scenarios,
    run_source_case,
)

_REAL_SESSION = ort.InferenceSession


@pytest.fixture(autouse=True)
def _settings_and_runtime_guard(monkeypatch: pytest.MonkeyPatch):
    previous, effective = _global_source_configuration(), SourceSettings.observe()

    def forbidden(*args: Any, **kwargs: Any):
        pytest.fail("Only the explicit tiny real-session unit control may execute")

    monkeypatch.setattr(ort, "InferenceSession", forbidden)
    try:
        with (
            jax.enable_x64(False),
            jax.default_matmul_precision(None),
            jax.numpy_dtype_promotion("standard"),
            jax.numpy_rank_promotion("allow"),
        ):
            yield
    finally:
        for key, value in previous.items():
            jax.config.update(key, value)
        assert _global_source_configuration() == previous
        assert SourceSettings.observe() == effective


def _plan():
    return artifact_probe.build_probe_plan(
        json.loads(artifact_probe.DEFAULT_BASELINE.read_text()),
        baseline_sha256=HISTORICAL_BASELINE_SHA256,
    )


def _original_artifact(origin="plugin::jax.numpy.add", ordinal=0):
    """Independent existing artifact harness, with no numerical criterion/run."""
    plan = _plan()
    registrations, _ = raw_registrations()
    row = next(r for r in plan.rows if (r.origin, r.ordinal) == (origin, ordinal))
    metadata = artifact_probe._resolve_case(row, {k: v for k, _, v in registrations})
    scenario = raw_source_scenarios(metadata, SourceProvenance(origin, ordinal))[0]
    prepared = prepare_source_case(metadata, scenario)
    prior = artifact_probe.execute_probe_plan(
        plan,
        registrations,
        stage="prepare",
        limit=1,
        scenario=artifact_probe.SCENARIOS[0],
        cases=frozenset({SourceProvenance(origin, ordinal)}),
    )
    old_cell = next(asdict(c) for c in prior.cells if c.history)
    old = old_cell["history"][-1]
    checked = run_source_case(
        metadata,
        scenario,
        optimization=None,
        diagnostic_jaxpr_validator=DiagnosticValidator(),
        _prepared=prepared,
        _corpus_counterpart=counterpart_for(origin, ordinal, row.contract_sha256),
    )
    assert checked.model is not None and checked.audit.status is CheckStatus.UNVERIFIED
    old.update(
        artifact_sha256=numerical.sha256(checked.model.SerializeToString()),
        export_settings=asdict(checked.export_settings),
        export_report=asdict(get_precision_report(checked.model)),
        corpus_structure=(
            asdict(checked.corpus_structure) if checked.corpus_structure else None
        ),
        raw_callback_status=checked.raw_callback_status,
    )
    old_cell["artifact_sha256"] = old["artifact_sha256"]
    return row, metadata, numerical.json_value(old_cell)


def _frozen_unit(tmp_path):
    row, metadata, old = _original_artifact()
    cell = numerical.prepare_numerical_cell(row, metadata, old, tmp_path / "payload")
    preparation = tmp_path / "preparation.json"
    frozen = dict(
        criterion=numerical.criterion_descriptor(),
        approval_sha256=numerical.APPROVAL_SHA256,
        cells=[cell],
    )
    payload = numerical.write_new(
        preparation, (json.dumps(frozen, sort_keys=True) + "\n").encode()
    )
    return cell, preparation, payload["sha256"]


def test_exact_twelve_rows_twenty_four_cells_and_unchanged_original_criteria():
    plan = _plan()
    identities = {(c.origin, c.ordinal) for c in numerical.NUMERICAL_CASES}
    selected = [c for c in plan.cells if (c.origin, c.ordinal) in identities]
    assert len(identities) == 12 and len(selected) == 24
    assert all(c.origin != "plugin::gather" for c in selected)
    assert len(plan.cells) - len(selected) == 3580
    assert (
        sum(
            c.stage == "recipe-pending"
            for c in plan.cells
            if (c.origin, c.ordinal) not in identities
        )
        == 126
    )
    assert all(
        r.criteria is None for r in plan.rows if (r.origin, r.ordinal) in identities
    )
    assert numerical.CRITERIA.mode == "tolerant"
    assert numerical.CRITERIA.rtol == numerical.CRITERIA.atol == 1e-5
    assert numerical.criterion_descriptor()["historical_criteria_unchanged"] is None
    dots = [
        c
        for c in numerical.NUMERICAL_CASES
        if c.origin == "plugin::dot_general" and c.ordinal in (2, 3)
    ]
    assert len(dots) == 2 and dots[0].contract_sha256 == dots[1].contract_sha256
    assert (
        counterpart_for(dots[0].origin, 2, dots[0].contract_sha256).rhs_contract_axis
        != counterpart_for(dots[1].origin, 3, dots[1].contract_sha256).rhs_contract_axis
    )


def test_asymmetric_existing_threshold_fixture_requires_both_directions():
    reference = np.asarray([0], np.float32)
    actual = np.frombuffer(bytes.fromhex("adc52737"), np.float32)
    assert np.allclose(reference, actual, rtol=1e-5, atol=1e-5, equal_nan=True)
    assert not np.allclose(actual, reference, rtol=1e-5, atol=1e-5, equal_nan=True)
    compared = numerical.compare_outputs((actual,), (reference,), (reference,), ((1,),))
    assert compared["status"] == "failed"
    assert compared["source_actual_first"] is False
    assert compared["raw_reference_first"] is True
    assert compared["historical_reference_first"] is True


def test_distinct_historical_context_reference_is_a_required_third_predicate():
    raw = np.asarray([0], np.float32)
    historical = np.asarray([1], np.float32)
    result = numerical.compare_outputs((raw,), (raw,), (historical,), ((1,),))
    assert (
        result["output_contract"]
        and result["source_actual_first"]
        and result["raw_reference_first"]
    )
    assert (
        result["historical_reference_first"] is False and result["status"] == "failed"
    )


@pytest.mark.parametrize(
    "bad", ["count", "dtype", "shape", "raw-dtype", "historical-shape"]
)
def test_output_contract_rejects_before_any_comparison(bad, monkeypatch):
    good = np.asarray([0], np.float32)
    actual, raw, historical = (good,), (good,), (good,)
    if bad == "count":
        actual = ()
    elif bad == "dtype":
        actual = (good.astype(np.float64),)
    elif bad == "shape":
        actual = (good.reshape(1, 1),)
    elif bad == "raw-dtype":
        raw = (good.astype(np.float16),)
    else:
        historical = (good.reshape(1, 1),)

    def forbidden(*args, **kwargs):
        pytest.fail("Contract gate must precede allclose")

    monkeypatch.setattr(numerical, "_compare", forbidden)
    monkeypatch.setattr(np, "allclose", forbidden)
    result = numerical.compare_outputs(actual, raw, historical, ((1,),))
    assert result["status"] == "failed" and result["output_contract"] is False
    assert all(
        result[k] is None
        for k in (
            "source_actual_first",
            "raw_reference_first",
            "historical_reference_first",
        )
    )


def test_inherited_allclose_nan_and_signed_zero_semantics_are_not_bitwise():
    raw = np.asarray([np.nan, -0.0], np.float32)
    actual = np.asarray([np.nan, 0.0], np.float32)
    assert (
        numerical.compare_outputs((actual,), (raw,), (raw,), ((2,),))["status"]
        == "validated"
    )


def test_actual_native_reference_context_and_all_five_settings_restore():
    observed = []

    def native(x):
        observed.append((SourceSettings.observe(), _global_source_configuration()))
        return jax.lax.dot_general(x, x, (((1,), (0,)), ((), ())))

    metadata = dict(
        callable=native,
        testcase="native-context-control",
        input_values=[np.asarray([[1.0, 0.25], [-2, 0.5]], np.float32)],
    )
    hidden = _global_source_configuration()
    with (
        jax.enable_x64(True),
        jax.default_matmul_precision("highest"),
        jax.numpy_dtype_promotion("strict"),
        jax.numpy_rank_promotion("raise"),
    ):
        effective = SourceSettings.observe()
        raw = prepare_source_case(metadata, SourceScenario("raw-x64-on", True))
        raw_bytes = raw.reference_outputs[0].tobytes()
        observed.clear()
        historical, changes = numerical.prepare_harness_reference(raw)
        assert observed and all(
            not settings.x64_enabled and settings.default_matmul_precision == "float32"
            for settings, _ in observed
        )
        assert historical.input_evidence == raw.input_evidence
        assert raw.reference_outputs[0].tobytes() == raw_bytes
        assert historical.settings.x64_enabled is False
        assert changes and all(c["path"].endswith(".params.precision") for c in changes)
        assert (
            SourceSettings.observe() == effective
            and _global_source_configuration() == hidden
        )


@pytest.mark.parametrize("change", ["dtype", "shape", "weak", "source", "axis"])
def test_real_fixed_source_drift_blocks_before_runtime(change, tmp_path):
    row, metadata, accepted = _original_artifact("plugin::dot_general", 2)
    metadata = dict(metadata)
    if change == "dtype":
        metadata["callable"] = lambda a, b: jax.lax.dot_general(
            a, b, (((0,), (1,)), ((), ()))
        ).astype(jnp.float16)
    elif change == "shape":
        metadata["callable"] = lambda a, b: jax.lax.dot_general(
            a, b, (((0,), (1,)), ((), ()))
        )[0]
    elif change == "weak":
        metadata["callable"] = lambda a, b: jnp.asarray(0.0)
    elif change == "source":
        metadata["callable"] = lambda a, b: jax.lax.dot_general(
            a, b, (((0,), (1,)), ((), ()))
        ) + np.float32(1)
    else:
        metadata["callable"] = lambda a, b: jax.lax.dot_general(
            a, b, (((1,), (1,)), ((), ()))
        )
    with pytest.raises(numerical.NumericalPreflightError, match="drift"):
        numerical.prepare_numerical_cell(row, metadata, accepted, tmp_path / "cell")
    assert not (tmp_path / "cell").exists()


@pytest.mark.parametrize(
    "origin,ordinal,contract",
    [
        ("plugin::gather", 0, "bad"),
        ("plugin::dot_general", 4, "bad"),
        ("plugin::dot_general", 2, "bad"),
        ("unknown", 0, "bad"),
    ],
)
def test_unknown_row_ordinal_contract_and_gather_cannot_be_selected(
    origin, ordinal, contract
):
    with pytest.raises(numerical.NumericalPreflightError, match="twelve"):
        numerical.selected_case(origin, ordinal, contract)


@pytest.mark.parametrize(
    "field",
    ["artifact_sha256", "original_trace_sha256", "references", "actual_settings"],
)
def test_stale_artifact_or_source_reference_evidence_is_rejected(field, tmp_path):
    row, metadata, accepted = _original_artifact()
    if field == "artifact_sha256":
        accepted[field] = "0" * 64
    else:
        accepted["history"][-1][field] = "changed"
    with pytest.raises(numerical.NumericalPreflightError):
        numerical.prepare_numerical_cell(row, metadata, accepted, tmp_path / "cell")


@pytest.mark.parametrize("alias", ["same", "normalized", "symlink"])
def test_cli_aliases_fail_before_provenance_or_source_execution(
    alias, tmp_path, monkeypatch, capsys
):
    output = tmp_path / "result.json"
    preparation = output
    if alias == "normalized":
        parent = tmp_path / "child"
        parent.mkdir()
        preparation = parent / ".." / output.name
    elif alias == "symlink":
        parent = tmp_path / "link"
        parent.symlink_to(tmp_path, target_is_directory=True)
        preparation = parent / output.name
    before = set(tmp_path.iterdir())

    def forbidden(*args, **kwargs):
        pytest.fail("Aliased paths reached source/provenance execution")

    monkeypatch.setattr(probe, "load_approved_context", forbidden)
    args = [
        "--approval",
        str(tmp_path / "approval"),
        "--accepted-artifact",
        str(tmp_path / "accepted"),
        "--criterion-version",
        numerical.CRITERION_VERSION,
        "--limit",
        "24",
        "--output",
        str(output),
        "--preparation-output",
        str(preparation),
    ]
    with pytest.raises(SystemExit) as error:
        probe.main(args)
    assert (
        error.value.code == 2 and "distinct resolved paths" in capsys.readouterr().err
    )
    assert set(tmp_path.iterdir()) == before


def test_missing_or_changed_approval_precedes_source_and_creates_no_files(
    tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("Approval gate reached source execution")

    monkeypatch.setattr(probe, "code_snapshot", forbidden)
    approval = tmp_path / "approval.json"
    accepted = tmp_path / "accepted.json"
    accepted.write_text("{}")
    with pytest.raises(FileNotFoundError):
        probe.load_approved_context(approval, accepted)
    approval.write_text("{}")
    with pytest.raises(numerical.NumericalPreflightError, match="approval"):
        probe.load_approved_context(approval, accepted)


def test_one_real_run_per_mode_uses_same_frozen_arrays_for_all_predicates(
    tmp_path, monkeypatch
):
    cell, preparation, digest = _frozen_unit(tmp_path)
    arrays = numerical.load_frozen_payloads(cell)
    sessions = []
    runs = []
    returned = []
    comparisons = []
    original_allclose = np.allclose

    def observe_comparison(a, b, **kwargs):
        comparisons.append((id(a), id(b)))
        return original_allclose(a, b, **kwargs)

    class Session:
        def __init__(self, *args, **kwargs):
            assert numerical.sha256(preparation.read_bytes()) == digest
            assert all(
                Path(v["path"]).is_file()
                for role in ("feed", "raw_reference", "historical_reference")
                for v in cell[role]
            )
            self.real = _REAL_SESSION(*args, **kwargs)
            sessions.append(self)

        def __getattr__(self, name):
            return getattr(self.real, name)

        def run(self, names, feed):
            runs.append((names, tuple(id(v) for v in feed.values())))
            output = self.real.run(names, feed)
            returned.append(tuple(np.array(v, copy=True) for v in output))
            return output

    monkeypatch.setattr(ort, "InferenceSession", Session)
    monkeypatch.setattr(np, "allclose", observe_comparison)
    for index, mode in enumerate(numerical.TARGET_MODES):
        comparisons.clear()
        result = numerical.execute_frozen_cell(
            cell, mode, preparation, digest, payloads=arrays
        )
        assert result["run_attempts"] == result["successful_runs"] == 1
        assert result["runtime"] == result["numerics"] == "validated"
        assert result["lowering_audit"] == "unverified"
        assert len(comparisons) == 3
        assert comparisons[0][0] == comparisons[1][1] == comparisons[2][1]
        assert comparisons[0][1] == comparisons[1][0] == id(arrays.raw[0])
        assert comparisons[2][0] == id(arrays.historical[0])
        actual = numerical.load_arrays(result["outputs"])
        assert [a.tobytes() for a in actual] == [a.tobytes() for a in returned[index]]
        assert (
            numerical.compare_outputs(
                actual,
                arrays.raw,
                arrays.historical,
                tuple(tuple(s) for s in cell["output_shapes"]),
            )
            == result["comparison"]
        )
        assert (
            result["target"]["enable_mem_pattern"] is False
            and result["target"]["fallback_disabled"] is True
        )
        assert result["target"]["providers"] == ["CPUExecutionProvider"]
        assert (
            result["artifact_bytes_unchanged"]
            and result["preparation_bytes_unchanged"]
            and result["feed_bytes_unchanged"]
        )
        with pytest.raises(FileExistsError):
            numerical.execute_frozen_cell(
                cell, mode, preparation, digest, payloads=arrays
            )
    assert len(sessions) == len(runs) == 2 and runs[0][1] == runs[1][1]
    assert numerical.sha256(preparation.read_bytes()) == digest


@pytest.mark.parametrize("phase", ["session-load", "execution"])
def test_target_failures_stay_failed_without_retry(phase, tmp_path, monkeypatch):
    cell, preparation, digest = _frozen_unit(tmp_path)
    arrays = numerical.load_frozen_payloads(cell)
    calls = []

    class Session:
        def __init__(self, *args, **kwargs):
            calls.append("load")
            if phase == "session-load":
                raise RuntimeError("injected load failure")
            self.real = _REAL_SESSION(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.real, name)

        def run(self, *args, **kwargs):
            calls.append("run")
            raise RuntimeError("injected execution failure")

    monkeypatch.setattr(ort, "InferenceSession", Session)
    result = numerical.execute_frozen_cell(
        cell, "disabled", preparation, digest, payloads=arrays
    )
    assert result["runtime"] == "failed" and result["numerics"] == "unverified"
    assert result["stage"] == phase and result["successful_runs"] == 0
    assert calls == (["load"] if phase == "session-load" else ["load", "run"])
    with pytest.raises(FileExistsError):
        numerical.execute_frozen_cell(
            cell, "disabled", preparation, digest, payloads=arrays
        )


@pytest.mark.parametrize(
    "role", ["feed", "raw_reference", "historical_reference", "model", "preparation"]
)
def test_missing_or_changed_payloads_cannot_construct_session(role, tmp_path):
    cell, preparation, digest = _frozen_unit(tmp_path)
    arrays = numerical.load_frozen_payloads(cell)
    path = (
        preparation
        if role == "preparation"
        else Path(cell[role]["path"] if role == "model" else cell[role][0]["path"])
    )
    path.chmod(0o644)
    path.write_bytes(b"changed")
    with pytest.raises(numerical.NumericalPreflightError):
        numerical.execute_frozen_cell(
            cell, "disabled", preparation, digest, payloads=arrays
        )
    assert not (Path(cell["payload_directory"]) / "disabled").exists()


def test_unexpected_reference_trace_changes_are_not_normalized():
    raw = dict(
        equations=[
            dict(
                primitive="dot_general",
                params=dict(precision=None, preferred_element_type="float32"),
            )
        ]
    )
    changed = copy.deepcopy(raw)
    changed["equations"][0]["params"]["preferred_element_type"] = "float64"
    with pytest.raises(numerical.NumericalPreflightError, match="Unexpected"):
        numerical.reference_context_changes(raw, changed)


def test_current_default_add_admission_and_matrix_exclusion_remain():
    for origin in ("plugin::jax.numpy.add", "plugin::dot_general"):
        row, metadata, _ = _original_artifact(origin, 0)
        scenario = raw_source_scenarios(metadata, SourceProvenance(origin, 0))[0]
        result = run_source_case(metadata, scenario, optimization=None)
        if origin == "plugin::jax.numpy.add":
            assert (
                result.model is not None
                and result.audit.status is CheckStatus.VALIDATED
            )
        else:
            assert (
                result.model is None and result.audit.status is CheckStatus.UNVERIFIED
            )
    assert row.criteria is None


def _bound_record():
    cases = [asdict(case) for case in numerical.NUMERICAL_CASES]
    selected = []
    bound = []
    for case in cases:
        for scenario in artifact_probe.SCENARIOS:
            identity = dict(
                origin=case["origin"], ordinal=case["ordinal"], scenario=scenario
            )
            old = dict(
                row_contract_sha256=case["contract_sha256"],
                scenario_sha256="a" * 64,
                original_trace_sha256="b" * 64,
            )
            selected.append(
                dict(
                    **identity,
                    artifact_sha256="c" * 64,
                    history=[old],
                    schema="validated",
                    artifact="validated",
                    source_reference="validated",
                    lowering_audit="unverified",
                    runtime="unverified",
                    numerics="unverified",
                )
            )
            bound.append(
                dict(
                    **identity,
                    artifact_sha256="c" * 64,
                    scenario_sha256="a" * 64,
                    original_trace_sha256="b" * 64,
                )
            )
    return dict(
        cases=cases,
        criterion_version=numerical.CRITERION_VERSION,
        accepted_main_artifact_records=dict(unit=dict(cells=bound)),
    ), dict(selected=selected)


@pytest.mark.parametrize(
    "change",
    [
        "none",
        "gather",
        "ordinal",
        "contract",
        "trace",
        "artifact",
        "duplicate",
        "status",
    ],
)
def test_pinned_complete_cell_mapping_blocks_every_changed_identity(change):
    approval, record = _bound_record()
    if change == "none":
        frozen = json.dumps(record, sort_keys=True)
        assert len(probe.validate_accepted_cells(approval, record, "unit")) == 24
        assert json.dumps(record, sort_keys=True) == frozen
        return
    cell = record["selected"][0]
    if change == "gather":
        cell["origin"] = "plugin::gather"
    elif change == "ordinal":
        cell["ordinal"] = 4
    elif change == "contract":
        cell["history"][-1]["row_contract_sha256"] = "changed"
    elif change == "trace":
        cell["history"][-1]["original_trace_sha256"] = "changed"
    elif change == "artifact":
        cell["artifact_sha256"] = "changed"
    elif change == "duplicate":
        record["selected"][1] = copy.deepcopy(cell)
    else:
        cell["lowering_audit"] = "validated"
    with pytest.raises(numerical.NumericalPreflightError):
        probe.validate_accepted_cells(approval, record, "unit")


def test_native_reference_failure_restores_hidden_and_enclosing_settings():
    def native(x):
        if jax.config.jax_default_matmul_precision == "float32":
            jax.config.update("jax_numpy_rank_promotion", "allow")
            raise RuntimeError("native historical-context failure")
        return x + x

    metadata = dict(
        callable=native,
        testcase="reference-failure-control",
        input_values=[np.asarray([0.25], np.float32)],
    )
    with (
        jax.enable_x64(True),
        jax.default_matmul_precision("highest"),
        jax.numpy_dtype_promotion("strict"),
        jax.numpy_rank_promotion("raise"),
    ):
        hidden, effective = _global_source_configuration(), SourceSettings.observe()
        raw = prepare_source_case(metadata, SourceScenario("raw-x64-on", True))
        with pytest.raises(RuntimeError, match="historical-context failure"):
            numerical.prepare_harness_reference(raw)
        assert (
            _global_source_configuration() == hidden
            and SourceSettings.observe() == effective
        )


@pytest.mark.parametrize("occupied", ["output", "preparation", "payload-alias"])
def test_existing_evidence_and_payload_aliases_are_never_overwritten(
    occupied, tmp_path
):
    output, preparation = tmp_path / "out.json", tmp_path / "prep.json"
    if occupied == "payload-alias":
        output = tmp_path / (preparation.name + ".payloads")
    else:
        (output if occupied == "output" else preparation).write_bytes(
            b"immutable history"
        )
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(numerical.NumericalPreflightError):
        probe.fresh_paths(output, preparation)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_preparation_retains_all_pending_cells_before_any_session(
    tmp_path, monkeypatch
):
    accepted = tmp_path / "accepted.json"
    accepted.write_text("unit artifact provenance")
    digest = numerical.sha256(accepted.read_bytes())
    code = {"unit": "frozen"}
    context = dict(
        code=code,
        approval={"unit": "fixture"},
        lineage={"unit": "fixture"},
        historical_files={str(accepted): digest},
    )
    pending = [
        dict(
            identity=[case.origin, case.ordinal, scenario],
            state="preflight-pending",
            detail="injected reference failure",
        )
        for case in numerical.NUMERICAL_CASES
        for scenario in artifact_probe.SCENARIOS
    ]
    monkeypatch.setattr(probe, "load_approved_context", lambda *args: context)
    monkeypatch.setattr(probe, "code_snapshot", lambda: code)
    monkeypatch.setattr(probe, "prepare_packet", lambda *args: ([], pending))

    def forbidden(*args, **kwargs):
        pytest.fail("Pending preflight attempted target execution")

    monkeypatch.setattr(numerical, "execute_frozen_cell", forbidden)
    result = probe.run_packet(
        approval=tmp_path / "approval",
        accepted=accepted,
        output=tmp_path / "result.json",
        preparation=tmp_path / "preparation.json",
        execute=True,
    )
    assert result["status"] == "preflight-pending" and result["executions"] == []
    frozen = json.loads((tmp_path / "preparation.json").read_bytes())
    assert frozen["pending"] == pending and len(frozen["pending"]) == 24
    assert (
        frozen["original_criteria"] is None and frozen["lowering_audit"] == "unverified"
    )
    assert result["preparation"]["sha256"] == numerical.sha256(
        (tmp_path / "preparation.json").read_bytes()
    )
    assert accepted.read_text() == "unit artifact provenance"


@pytest.mark.parametrize("change", ["scenario", "mode", "version", "shape"])
def test_frozen_identity_and_target_errors_precede_manifest_or_session_access(
    change, tmp_path
):
    case = next(
        c
        for c in numerical.NUMERICAL_CASES
        if c.origin == "plugin::jax.numpy.add" and c.ordinal == 0
    )
    cell = dict(
        case=asdict(case),
        identity=[case.origin, case.ordinal, "raw-x64-off"],
        criterion_version=numerical.CRITERION_VERSION,
        output_shapes=[[3]],
    )
    mode = "disabled"
    if change == "scenario":
        cell["identity"][2] = "unreviewed"
    elif change == "mode":
        mode = "basic"
    elif change == "version":
        cell["criterion_version"] = "unreviewed"
    else:
        cell["output_shapes"] = [[1, 3]]
    with pytest.raises(numerical.NumericalPreflightError, match="identity/target"):
        numerical.execute_frozen_cell(
            cell,
            mode,
            tmp_path / "not-created",
            "0" * 64,
            payloads=numerical.FrozenPayloads((), (), ()),
        )


@pytest.mark.parametrize("role", ["raw", "historical"])
def test_missing_reference_arrays_are_rejected_before_target_load(role, tmp_path):
    cell, preparation, digest = _frozen_unit(tmp_path)
    values = numerical.load_frozen_payloads(cell)
    missing = numerical.FrozenPayloads(
        values.feed,
        () if role == "raw" else values.raw,
        () if role == "historical" else values.historical,
    )
    with pytest.raises(numerical.NumericalPreflightError, match="In-memory payload"):
        numerical.execute_frozen_cell(
            cell, "disabled", preparation, digest, payloads=missing
        )
    assert not (Path(cell["payload_directory"]) / "disabled").exists()
