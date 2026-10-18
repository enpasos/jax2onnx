# tests/extra_tests/converter/test_source_precision_corpus_replay.py

"""Fixed corpus counterparts preserve source, raw criteria and immutable lineage."""

from __future__ import annotations

import ast
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from typing import Any

import jax
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
import pytest

from jax2onnx.precision import CheckStatus
from jax2onnx.user_interface import to_onnx
from scripts import source_precision_probe as probe
from scripts.source_precision_inventory import raw_registrations
from tests.extra_tests.converter import source_precision_corpus_oracles as oracles
from tests.extra_tests.converter.source_precision_diagnostics import DiagnosticValidator
from tests.extra_tests.converter.source_precision_support import (
    SourceProvenance,
    SourceSettings,
    prepare_source_case,
    raw_source_case,
    raw_source_scenarios,
    run_source_case,
)


@pytest.fixture(autouse=True)
def _no_runtime(monkeypatch: pytest.MonkeyPatch):
    import onnxruntime

    previous = SourceSettings.observe()
    calls = []

    def forbidden(*args: Any, **kwargs: Any):
        calls.append((args, kwargs))
        pytest.fail("Corpus artifact evidence must never construct an ORT session")

    monkeypatch.setattr(onnxruntime, "InferenceSession", forbidden)
    yield
    assert not calls
    assert SourceSettings.observe() == previous


def _plan():
    # Current manual accounting may grow; the equal raw contracts are tested here.
    # execute_corpus_revision separately hashes actual historical inventory bytes.
    inventory = json.loads(probe.DEFAULT_BASELINE.read_text())
    return probe.build_probe_plan(
        inventory, baseline_sha256=oracles.HISTORICAL_BASELINE_SHA256
    )


@pytest.mark.parametrize("alias", ["same-path", "normalized-parent", "symlink-parent"])
def test_corpus_cli_rejects_aliased_output_before_source_execution(
    alias: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    output = tmp_path / "supplement.json"
    if alias == "normalized-parent":
        parent = tmp_path / "child"
        parent.mkdir()
        preparation = parent / ".." / output.name
    elif alias == "symlink-parent":
        parent = tmp_path / "alias"
        parent.symlink_to(tmp_path, target_is_directory=True)
        preparation = parent / output.name
    else:
        preparation = output
    paths_before = set(tmp_path.iterdir())
    calls = []

    def forbidden(*args: Any, **kwargs: Any):
        calls.append((args, kwargs))
        pytest.fail("Aliased outputs must be rejected before any source execution")

    monkeypatch.setattr(probe.subprocess, "check_output", forbidden)
    monkeypatch.setattr(probe, "raw_registrations", forbidden)
    monkeypatch.setattr(probe, "execute_corpus_revision", forbidden)
    with pytest.raises(SystemExit) as error:
        probe.main(
            [
                "--corpus-replay",
                "--limit",
                "26",
                "--output",
                str(output),
                "--preparation-output",
                str(preparation),
                "--history",
                str(tmp_path / "history36.json"),
                "--history",
                str(tmp_path / "history2.json"),
            ]
        )
    assert error.value.code == 2
    assert "need distinct resolved paths" in capsys.readouterr().err
    assert not calls
    assert not output.exists() and not preparation.exists()
    assert set(tmp_path.iterdir()) == paths_before


def _dot(ordinal=3, x64=False):
    metadata = raw_source_case("plugin::dot_general", ordinal)
    scenario = raw_source_scenarios(
        metadata, SourceProvenance("plugin::dot_general", ordinal)
    )[int(x64)]
    return metadata, scenario, prepare_source_case(metadata, scenario)


def _square_model():
    return h.make_model(
        h.make_graph(
            [
                h.make_node("Transpose", ["y"], ["rhs"], perm=[1, 0]),
                h.make_node("MatMul", ["x", "rhs"], ["out"]),
            ],
            "square_counterpart",
            [h.make_tensor_value_info(name, T.FLOAT, [3, 3]) for name in ("x", "y")],
            [h.make_tensor_value_info("out", T.FLOAT, [3, 3])],
        ),
        opset_imports=[h.make_opsetid("", 23)],
        ir_version=10,
    )


def test_exact_packet_selector_has13_raw_rows26_cells_and_no_numerical_criteria():
    plan = _plan()
    selected = probe.corpus_replay_cases(plan)
    registrations, _ = raw_registrations()
    probe.preflight_cases(plan, registrations, selected)
    assert len(selected) == 13
    cells = [
        cell
        for cell in plan.cells
        if SourceProvenance(cell.origin, cell.ordinal) in selected
    ]
    assert len(cells) == 26 and {cell.scenario for cell in cells} == set(
        probe.SCENARIOS
    )
    rest = [
        cell
        for cell in plan.cells
        if SourceProvenance(cell.origin, cell.ordinal) not in selected
    ]
    assert (
        len(rest) == 3578
        and sum(cell.stage == "recipe-pending" for cell in rest) == 126
    )
    assert all(
        row.criteria is None
        for row in plan.rows
        if SourceProvenance(row.origin, row.ordinal) in selected
    )
    assert len(oracles.DOT_COUNTERPARTS) == 4
    assert (
        oracles.DOT_COUNTERPARTS[2].case.contract_sha256
        == oracles.DOT_COUNTERPARTS[3].case.contract_sha256
    )
    assert (
        oracles.DOT_COUNTERPARTS[2].rhs_contract_axis
        != oracles.DOT_COUNTERPARTS[3].rhs_contract_axis
    )


@pytest.mark.parametrize("ordinal", range(4))
@pytest.mark.parametrize("x64", [False, True])
def test_real_raw_dots_keep_original_gemm_failure_and_require_source_structure(
    ordinal, x64
):
    metadata, scenario, prepared = _dot(ordinal, x64)
    counterpart = oracles.DOT_COUNTERPARTS[ordinal]
    raw = run_source_case(
        metadata,
        scenario,
        optimization=None,
        diagnostic_jaxpr_validator=DiagnosticValidator(),
        _prepared=prepared,
    )
    assert raw.model is None and raw.schema is CheckStatus.VALIDATED
    assert (
        raw.audit.status is CheckStatus.FAILED
        and "Raw metadata structural callback failed" in raw.audit.detail
    )
    checked = run_source_case(
        metadata,
        scenario,
        optimization=None,
        diagnostic_jaxpr_validator=DiagnosticValidator(),
        _prepared=prepared,
        _corpus_counterpart=counterpart,
    )
    assert checked.model is not None and checked.schema is CheckStatus.VALIDATED
    evidence = checked.corpus_structure
    assert evidence is not None and evidence.raw_callback_status == "failed"
    assert evidence.source_structure is CheckStatus.VALIDATED and evidence.accepted
    assert evidence.counterpart_sha256 == counterpart.sha256
    assert (
        checked.stage == "diagnostic-artifact-checked"
        and checked.audit.status is checked.numerical_status is CheckStatus.UNVERIFIED
    )
    assert prepared.criteria is scenario.criteria is None
    assert all(
        value.dtype == "float32" and not value.weak_type
        for value in (*prepared.traced_inputs, *prepared.traced_outputs)
    )


@pytest.mark.parametrize("ordinal", range(4))
@pytest.mark.parametrize("double_precision", [False, True])
def test_original_callbacks_still_check_real_legacy_gemm_artifacts(
    ordinal, double_precision
):
    metadata, _, prepared = _dot(ordinal, double_precision)
    # LEGACY deliberately updates global x64 in its own compatibility scope.
    # The SOURCE callable-mutation guard is not a LEGACY export context.
    previous_global_x64 = jax.enable_x64.get_global()
    try:
        with jax.enable_x64(double_precision):
            model = to_onnx(
                prepared.fn,
                prepared.input_specs,
                enable_double_precision=double_precision,
            )
    finally:
        jax.config.update("jax_enable_x64", previous_global_x64)
    assert isinstance(model, onnx.ModelProto)
    onnx.checker.check_model(model, full_check=True)
    assert sum(node.op_type == "Gemm" for node in model.graph.node) == 1
    assert all(
        value.type.tensor_type.elem_type == (T.DOUBLE if double_precision else T.FLOAT)
        for value in (*model.graph.input, *model.graph.output)
    )
    assert metadata["post_check_onnx_graph"](model)


@pytest.mark.parametrize(
    "fault",
    [
        "wrong-permutation",
        "swapped",
        "disconnected",
        "bias",
        "cast",
        "dtype",
        "shape",
        "unused-input",
        "custom-domain",
        "unrelated-matmul",
        "unreviewed-alias",
    ],
)
def test_valid_onnx_counterexamples_do_not_acquire_a_source_counterpart(fault):
    _, _, prepared = _dot()
    model = _square_model()
    graph = model.graph
    if fault == "wrong-permutation":
        graph.node[0].attribute[0].ints[:] = [0, 1]
    elif fault == "swapped":
        graph.node[1].input[:] = ["rhs", "x"]
    elif fault == "disconnected":
        graph.node[1].input[:] = ["x", "x"]
    elif fault == "bias":
        graph.node[1].output[0] = "product"
        graph.initializer.append(nh.from_array(np.asarray(1, np.float32), "offset"))
        graph.node.append(h.make_node("Add", ["product", "offset"], ["out"]))
    elif fault == "cast":
        graph.node.insert(0, h.make_node("Cast", ["y"], ["cast_y"], to=T.FLOAT))
        graph.node[1].input[0] = "cast_y"
    elif fault in {"dtype", "shape"}:
        for value in (*graph.input, *graph.output):
            if fault == "dtype":
                value.type.tensor_type.elem_type = T.DOUBLE
            else:
                for dim in value.type.tensor_type.shape.dim:
                    dim.dim_value = 4
    elif fault == "unused-input":
        graph.input.append(h.make_tensor_value_info("unused", T.FLOAT, [3, 3]))
    elif fault == "custom-domain":
        graph.node[1].domain = "custom"
        model.opset_import.append(h.make_opsetid("custom", 1))
        model.functions.append(
            h.make_function(
                "custom",
                "MatMul",
                ["a", "b"],
                ["c"],
                [h.make_node("MatMul", ["a", "b"], ["c"])],
                [h.make_opsetid("", 23)],
            )
        )
    elif fault == "unrelated-matmul":
        del graph.node[:]
        graph.initializer.extend(
            [nh.from_array(np.eye(3, dtype=np.float32), name) for name in ("a", "b")]
        )
        graph.node.append(h.make_node("MatMul", ["a", "b"], ["out"]))
    else:
        graph.node[1].output[0] = "product"
        graph.node.append(h.make_node("Identity", ["product"], ["out"]))
    onnx.checker.check_model(model, full_check=True)
    result = oracles.DOT_COUNTERPARTS[3].check(prepared, model)
    assert result.source_structure is CheckStatus.FAILED and not result.accepted


@pytest.mark.parametrize(
    "fault", ["ordinal", "contract", "trace", "source-bytes", "unreviewed-mapping"]
)
def test_counterpart_binds_exact_provenance_contract_trace_and_source_bytes(
    fault, monkeypatch
):
    _, scenario, prepared = _dot()
    counterpart = oracles.DOT_COUNTERPARTS[3]
    if fault == "ordinal":
        prepared = replace(
            prepared,
            scenario=replace(
                scenario, provenance=SourceProvenance("plugin::dot_general", 2)
            ),
        )
    elif fault == "contract":
        prepared = replace(
            prepared, metadata={**prepared.metadata, "input_shapes": [(4, 4), (4, 4)]}
        )
    elif fault == "trace":
        trace = json.loads(prepared.original_trace_json)
        trace["equations"][0]["params"]["dimension_numbers"][0][1] = [0]
        prepared = replace(prepared, original_trace_json=json.dumps(trace))
    elif fault == "source-bytes":
        monkeypatch.setattr(oracles, "DOT_SOURCE_SHA256", "0" * 64)
    else:
        counterpart = replace(counterpart, rhs_contract_axis=0)
    evidence = counterpart.check(prepared, _square_model())
    assert evidence.source_structure is CheckStatus.FAILED and not evidence.accepted
    assert (
        oracles.counterpart_for(
            "plugin::dot_general", 4, counterpart.case.contract_sha256
        )
        is None
    )
    assert oracles.counterpart_for("plugin::dot_general", 3, "0" * 64) is None


def test_unknown_callback_failure_and_exception_still_fail_without_counterpart():
    from tests.extra_tests.converter.source_precision_support import SourceScenario

    for callback in (
        lambda model: False,
        lambda model: (_ for _ in ()).throw(RuntimeError("raw callback error")),
    ):
        result = run_source_case(
            {
                "callable": lambda x: x + np.float32(1),
                "input_shapes": [(2,)],
                "post_check_onnx_graph": callback,
            },
            SourceScenario("unlisted", False),
            optimization=None,
        )
        assert result.model is None and result.audit.status is CheckStatus.FAILED
        assert result.corpus_structure is None


def test_counterpart_never_swallows_original_callback_exception():
    _, _, prepared = _dot()

    def broken(model):
        raise RuntimeError("original callback failed unexpectedly")

    prepared = replace(
        prepared, metadata={**prepared.metadata, "post_check_onnx_graph": broken}
    )
    evidence = oracles.DOT_COUNTERPARTS[3].check(prepared, _square_model())
    assert evidence.raw_callback_status == "error" and not evidence.accepted
    assert "unexpectedly" in evidence.raw_callback_detail


def test_unlisted_vmap_callback_absence_remains_absent_and_default_admission_unchanged():
    metadata = raw_source_case("plugin::jax.numpy.matmul", 8)
    assert "post_check_onnx_graph" not in metadata
    scenario = raw_source_scenarios(
        metadata, SourceProvenance("plugin::jax.numpy.matmul", 8)
    )[0]
    absent = run_source_case(
        metadata,
        scenario,
        optimization=None,
        diagnostic_jaxpr_validator=DiagnosticValidator(),
    )
    assert absent.model is not None and absent.raw_callback_status == "absent"
    assert absent.corpus_structure is None
    plan = _plan()
    registrations, _ = raw_registrations()
    before = probe._corpus_product_controls(plan, registrations)
    _, scenario, prepared = _dot(0)
    run_source_case(
        prepared.metadata,
        scenario,
        optimization=None,
        diagnostic_jaxpr_validator=DiagnosticValidator(),
        _prepared=prepared,
        _corpus_counterpart=oracles.DOT_COUNTERPARTS[0],
    )
    after = probe._corpus_product_controls(plan, registrations)
    assert before == after
    assert all(
        r["audit"]["status"] is CheckStatus.VALIDATED
        for r in after
        if r["origin"] == "plugin::jax.numpy.add"
    )
    assert probe._corpus_controls_admitted(after)
    assert [(r["origin"], r["ordinal"], r["scenario"]) for r in after] == [
        (origin, 0, scenario)
        for origin in ("plugin::jax.numpy.add", "plugin::dot_general")
        for scenario in ("raw-x64-off", "raw-x64-on")
    ]
    for item in after:
        assert item["export_report"]["profile"] == "source-static-v1"
        assert item["schema"] is CheckStatus.VALIDATED
        assert item["runtime"] is item["numerics"] is CheckStatus.UNVERIFIED
        if item["origin"] == "plugin::dot_general":
            assert item["raw_callback_status"] == "failed"
            assert (
                item["corpus_structure"]["counterpart_sha256"]
                == oracles.DOT_COUNTERPARTS[0].sha256
            )


@pytest.mark.parametrize(
    "fault",
    [
        "empty",
        "missing-dot",
        "duplicate",
        "scenario",
        "counterpart",
        "digest",
        "report",
    ],
)
def test_audited_control_invariant_cannot_pass_by_omission_or_corruption(fault):
    registrations, _ = raw_registrations()
    controls = probe._corpus_product_controls(_plan(), registrations)
    assert probe._corpus_controls_admitted(controls)
    if fault == "empty":
        controls = []
    elif fault == "missing-dot":
        controls = controls[:2]
    elif fault == "duplicate":
        controls[-1] = controls[-2]
    elif fault == "scenario":
        controls[-1]["scenario"] = "not-approved"
    elif fault == "counterpart":
        controls[-1]["corpus_structure"] = None
    elif fault == "digest":
        controls[-1]["artifact_sha256"] = None
    else:
        controls[-1]["export_report"]["profile"] = "source-diagnostic-v1"
    assert not probe._corpus_controls_admitted(controls)


@pytest.mark.parametrize("missing", [True, False])
def test_live_audited_dot_control_requires_its_actual_reviewed_counterpart(
    missing, monkeypatch
):
    registrations, _ = raw_registrations()
    if missing:
        monkeypatch.setattr(probe, "counterpart_for", lambda *args: None)
        with pytest.raises(ValueError, match="reviewed counterpart"):
            probe._corpus_product_controls(_plan(), registrations)
    else:
        original = oracles.DotCounterpart.check

        def broken(self, prepared, model):
            return replace(
                original(self, prepared, model), source_structure=CheckStatus.FAILED
            )

        monkeypatch.setattr(oracles.DotCounterpart, "check", broken)
        controls = probe._corpus_product_controls(_plan(), registrations)
        assert not probe._corpus_controls_admitted(controls)


@pytest.mark.parametrize(
    "fault", ["selection", "limit", "scenario", "stage", "baseline"]
)
def test_revision_scope_is_checked_before_source_execution(fault, monkeypatch):
    plan = _plan()
    registrations, _ = raw_registrations()
    options = dict(
        stage="prepare",
        limit=26,
        cases=probe.corpus_replay_cases(plan),
        _corpus_replay=True,
    )
    if fault == "selection":
        options["cases"] = frozenset(list(options["cases"])[1:])
    elif fault == "limit":
        options["limit"] = 25
    elif fault == "scenario":
        options["scenario"] = "raw-x64-off"
    elif fault == "stage":
        options["stage"] = "product"
    else:
        plan = replace(plan, baseline_sha256="different")
    monkeypatch.setattr(
        probe,
        "prepare_source_case",
        lambda *args, **kwargs: pytest.fail("Invalid packet must not execute source"),
    )
    with pytest.raises(ValueError):
        probe.execute_probe_plan(plan, registrations, **options)


def test_same_stack_lineage_comparison_preserves_old_dictionaries_and_detects_drift():
    plan = _plan()
    registrations, _ = raw_registrations()
    cases = frozenset({SourceProvenance("plugin::dot_general", 0)})
    prepared = probe.execute_probe_plan(
        plan, registrations, stage="prepare", limit=2, cases=cases
    )
    prior = {
        "cells": [
            json.loads(json.dumps(asdict(c))) for c in prepared.cells if c.history
        ]
    }
    for cell in prior["cells"]:
        del cell["history"][-1]["corpus_structure"]
    frozen = json.dumps(prior, sort_keys=True)
    compared = probe._corpus_prior_comparisons(prepared, [prior], exported=False)
    assert len(compared) == 2 and all(not row["drift_fields"] for row in compared)
    changed = copy.deepcopy(prior)
    changed["cells"][0]["history"][-1]["references"][0]["logical_sha256"] = "changed"
    assert probe._corpus_prior_comparisons(prepared, [changed], exported=False)[0][
        "drift_fields"
    ] == ["references"]
    assert json.dumps(prior, sort_keys=True) == frozen
    failed_cell = next(c for c in prepared.cells if c.history)
    failed_record = replace(failed_cell.history[-1], artifact=CheckStatus.FAILED)
    failed = replace(
        prepared,
        cells=tuple(
            (
                replace(c, history=(failed_record,))
                if c.identity == failed_cell.identity
                else c
            )
            for c in prepared.cells
        ),
    )
    repeated = probe.execute_probe_plan(
        failed, registrations, stage="diagnostic", limit=2, cases=cases
    )
    assert next(
        c for c in repeated.cells if c.identity == failed_cell.identity
    ).history == (failed_record,)


def test_proposal_only_ledger_retains_complete_pinned_comparator_provenance():
    ledger = oracles.criterion_proposal_ledger()
    assert len(ledger["rows"]) == 12
    assert all(
        row["historical_criteria"] is row["raw_scenario_criteria"] is None
        and row["review"]["approval"] is None
        for row in ledger["rows"]
    )
    assert all(
        "reference, actual" in row["proposed_predicate"] for row in ledger["rows"]
    )
    assert "float32" in json.dumps(
        ledger["policy"]["original_harness_policy"]["reference_configuration"]
    )
    assert ledger["gather"]["status"].startswith("PENDING")
    root = Path(__file__).resolve().parents[3]
    for path, name in [
        ("tests/t_generator.py", "_get_rtol_atol"),
        ("jax2onnx/user_interface.py", "_run_allclose"),
        ("jax2onnx/precision.py", "_compare"),
    ]:
        text = (root / path).read_text()
        lines = text.splitlines(True)
        node = next(
            n
            for n in ast.walk(ast.parse(text))
            if isinstance(n, ast.FunctionDef) and n.name == name
        )
        source = "".join(lines[node.lineno - 1 : node.end_lineno])
        assert (
            hashlib.sha256(source.encode()).hexdigest()
            == ledger["policy"]["pinned_source"][path]["functions"][name]
        )
    reference = np.asarray([0], np.float32)
    actual = np.frombuffer(bytes.fromhex("adc52737"), np.float32)
    assert np.allclose(reference, actual, rtol=1e-5, atol=1e-5, equal_nan=True)
    assert not np.allclose(actual, reference, rtol=1e-5, atol=1e-5, equal_nan=True)
