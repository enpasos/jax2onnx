# tests/extra_tests/converter/test_source_precision_probe.py

"""Raw migration plans preserve factories, provenance, inputs and unknown evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from jax2onnx.plugins.plugin_system import construct_and_call, with_requested_dtype
from jax2onnx.precision import CheckStatus, SampleCriteria
from scripts import source_precision_probe as probe
from scripts.source_precision_inventory import build_inventory
from tests.extra_tests.converter.source_precision_support import (
    PRESERVE_FACTORY_DTYPE,
    SourcePreparationError,
    SourceProvenance,
    SourceScenario,
    prepare_source_case,
    raw_source_scenarios,
    run_source_case,
)


def _metadata(function: Any, **values: Any) -> dict[str, Any]:
    return {"testcase": "same-display", "callable": function, **values}


def _registration(*cases: dict[str, Any]):
    return (
        "plugin::raw",
        "sample.raw",
        {"context": "sample", "component": "raw", "testcases": list(cases)},
    )


def _plan(*cases: dict[str, Any]):
    registration = _registration(*cases)
    inventory = build_inventory([registration], [], manual=[], optional={})
    return probe.build_probe_plan(inventory), registration


def test_raw_factory_preservation_and_explicit_override_are_distinct() -> None:
    calls = []

    def constructor(*, dtype, fixed_dtype):
        calls.append((np.dtype(dtype), np.dtype(fixed_dtype)))
        return lambda x: x

    recipe = construct_and_call(
        constructor, dtype=with_requested_dtype(), fixed_dtype=np.float16
    ).with_dtype(np.float16)
    metadata = _metadata(recipe, input_shapes=[(2,)])
    scenarios = raw_source_scenarios(metadata, SourceProvenance("plugin::raw", 0))
    for scenario in scenarios:
        assert scenario.model_dtype is PRESERVE_FACTORY_DTYPE
        prepare_source_case(metadata, scenario)
    prepare_source_case(metadata, SourceScenario("explicit-model", True, np.float64))
    assert calls == [
        (np.dtype(np.float16), np.dtype(np.float16)),
        (np.dtype(np.float16), np.dtype(np.float16)),
        (np.dtype(np.float64), np.dtype(np.float16)),
    ]
    assert recipe._dtype is np.float16


def test_raw_factory_default_is_not_inferred_from_x64_or_legacy_flags() -> None:
    calls = []

    def constructor(*, dtype):
        calls.append(np.dtype(dtype))
        return lambda x: x

    metadata = _metadata(
        construct_and_call(constructor, dtype=with_requested_dtype()),
        input_shapes=[(1,)],
        run_only_f64_variant=True,
        expected_output_dtypes=[np.float64],
    )
    for scenario in raw_source_scenarios(metadata, SourceProvenance("plugin::raw", 0)):
        prepared = prepare_source_case(metadata, scenario)
        assert prepared.traced_inputs[0].dtype == "float32"
    assert calls == [np.dtype(np.float32), np.dtype(np.float32)]


def test_raw_pair_snapshots_same_typed_values_and_explicit_specs() -> None:
    original = np.asarray([1 + 2**-30], np.float64)
    scenarios = raw_source_scenarios(
        _metadata(lambda x: x, input_values=[original]),
        SourceProvenance("plugin::raw", 3),
        criteria=SampleCriteria("bitwise"),
    )
    original_bytes = original.tobytes()
    original[0] = 99
    off, on = scenarios
    assert off.input_specs is on.input_specs
    assert off.input_values is on.input_values
    assert off.input_values[0].tobytes() == original_bytes
    assert not off.input_values[0].flags.writeable
    assert off.input_specs[0].dtype == np.dtype(np.float64)
    assert [scenario.cell_identity for scenario in scenarios] == [
        ("plugin::raw", 3, "raw-x64-off"),
        ("plugin::raw", 3, "raw-x64-on"),
    ]
    assert off.default_matmul_precision is on.default_matmul_precision is None
    with pytest.raises(SourcePreparationError, match="canonicalize"):
        prepare_source_case(_metadata(lambda x: x), off)
    assert (
        prepare_source_case(_metadata(lambda x: x), on).arguments[0].dtype == np.float64
    )


def test_generated_raw_inputs_use_provenance_seed_and_match_both_contexts() -> None:
    metadata = _metadata(lambda x: x, input_shapes=[(3, 4)])

    def values(ordinal):
        return [
            prepare_source_case(metadata, scenario).arguments[0]
            for scenario in raw_source_scenarios(
                metadata, SourceProvenance("plugin::raw", ordinal)
            )
        ]

    off, on = values(0)
    assert off.dtype == on.dtype == np.float32
    assert np.asarray(off).tobytes() == np.asarray(on).tobytes()
    assert np.asarray(off).tobytes() != np.asarray(values(1)[0]).tobytes()


def test_untyped_raw_values_need_explicit_recipe_before_context_inference() -> None:
    with pytest.raises(SourcePreparationError, match="reviewed fixed input recipe"):
        raw_source_scenarios(
            _metadata(lambda x: x, input_values=[1.0]),
            SourceProvenance("plugin::raw", 0),
        )


def test_full_grid_has_two_unique_cells_per_anchored_row_and_keeps_pending_scope() -> (
    None
):
    inventory = json.loads(probe.DEFAULT_BASELINE.read_text())
    plan = probe.build_probe_plan(inventory)
    assert len(plan.rows) == 1802
    assert len(plan.cells) == 3604
    assert len({cell.identity for cell in plan.cells}) == 3604
    assert len({row.display for row in plan.rows}) == 1768
    assert sum(bool(row.unresolved_recipes) for row in plan.rows) == 63
    assert sum(cell.stage == "recipe-pending" for cell in plan.cells) == 126
    assert sum(row.criteria is not None for row in plan.rows) == 17
    for cell in plan.cells:
        assert (
            cell.lowering_audit
            is cell.runtime
            is cell.numerics
            is CheckStatus.UNVERIFIED
        )
    optional = plan.to_dict()["optional"]
    assert optional["maxtext"]["source_audit"] == "not-run"
    assert optional["maxdiffusion"]["numerical_acceptance"] == "unverified"


def test_grid_freezes_explicit_criteria_and_keeps_missing_or_legacy_only_unresolved() -> (
    None
):
    explicit = _metadata(
        lambda x: x, input_shapes=[(1,)], rtol=1e-5, atol=np.float64(1e-6)
    )
    legacy = _metadata(lambda x: x, input_shapes=[(1,)], rtol_f32=2e-5, atol_f32=3e-6)
    one_sided = _metadata(lambda x: x, input_shapes=[(1,)], rtol=9e-5)
    skipped = _metadata(
        lambda x: x,
        input_shapes=[(1,)],
        skip_numeric_validation=True,
        skip_numeric_validation_reason="nondeterministic model",
    )
    plan, _ = _plan(explicit, legacy, one_sided, skipped)
    explicit["rtol"] = 99
    assert plan.rows[0].criteria == SampleCriteria("tolerant", rtol=1e-5, atol=1e-6)
    assert all(row.criteria is None and row.criteria_pending for row in plan.rows[1:])
    assert plan.rows[3].skip_reason == "nondeterministic model"


@pytest.mark.parametrize("criteria_fields", [{}, {"rtol": 1e-5}])
def test_unresolved_raw_criteria_do_not_turn_valid_source_preparation_into_failure(
    criteria_fields,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import onnxruntime

    def forbidden(*args: Any, **kwargs: Any):
        pytest.fail("Unresolved raw criteria must never execute runtime samples")

    monkeypatch.setattr(onnxruntime, "InferenceSession", forbidden)
    metadata = _metadata(lambda x: x, input_shapes=[(1,)], **criteria_fields)
    plan, registration = _plan(metadata)
    assert plan.rows[0].criteria is None
    scenarios = raw_source_scenarios(metadata, SourceProvenance("plugin::raw", 0))
    prepared = prepare_source_case(metadata, scenarios[0])
    assert prepared.criteria is None
    completed = probe.execute_probe_plan(plan, [registration], stage="prepare", limit=1)
    assert completed.cells[0].source_reference is CheckStatus.VALIDATED
    assert completed.cells[0].numerics is CheckStatus.UNVERIFIED
    artifact = run_source_case(metadata, scenarios[0])
    assert artifact.audit.status is CheckStatus.VALIDATED
    assert artifact.stage == "artifact-checked"
    assert artifact.validation.target is None and not artifact.validation.samples
    assert artifact.numerical_status is CheckStatus.UNVERIFIED


def test_bounded_off_batch_keeps_all_on_cells_and_colliding_alias_rows() -> None:
    case = _metadata(lambda x: x, input_shapes=[(1,)])
    plan, registration = _plan(case, dict(case))
    completed = probe.execute_probe_plan(
        plan, [registration], stage="prepare", limit=2, scenario="raw-x64-off"
    )
    assert len(completed.cells) == 4
    off = [cell for cell in completed.cells if cell.scenario == "raw-x64-off"]
    on = [cell for cell in completed.cells if cell.scenario == "raw-x64-on"]
    assert all(cell.stage == "source-prepared" for cell in off)
    assert all(cell.source_reference is CheckStatus.VALIDATED for cell in off)
    assert all(cell.stage == "scheduled" for cell in on)
    assert all(
        cell.lowering_audit is CheckStatus.UNVERIFIED for cell in completed.cells
    )
    assert all(cell.stage == "scheduled" for cell in plan.cells)


def test_product_probe_checks_actual_artifact_without_implicit_numerical_criteria(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import onnxruntime

    def forbidden(*args: Any, **kwargs: Any):
        pytest.fail("Product artifact stage must not create a runtime session")

    monkeypatch.setattr(onnxruntime, "InferenceSession", forbidden)
    plan, registration = _plan(_metadata(lambda x: -x, input_shapes=[(2,)]))
    completed = probe.execute_probe_plan(plan, [registration], stage="product", limit=1)
    first, second = completed.cells
    assert first.stage == "artifact-checked"
    assert (
        first.source_reference
        is first.schema
        is first.lowering_audit
        is CheckStatus.VALIDATED
    )
    assert first.runtime is first.numerics is CheckStatus.UNVERIFIED
    assert len(first.artifact_sha256) == 64
    assert plan.rows[0].criteria is None
    assert second.stage == "scheduled"


def test_opaque_recipes_and_raw_contract_drift_never_execute_models() -> None:
    class Opaque:
        def __call__(self, x):
            pytest.fail("An opaque recipe must remain pending")

    plan, registration = _plan(_metadata(Opaque(), input_shapes=[(1,)]))
    assert (
        probe.execute_probe_plan(plan, [registration], stage="prepare", limit=2) == plan
    )
    case = _metadata(lambda x: x, input_shapes=[(1,)])
    plan, registration = _plan(case)
    case["input_shapes"] = [(2,)]
    completed = probe.execute_probe_plan(plan, [registration], stage="prepare", limit=1)
    assert completed.cells[0].stage == "preparation-pending"
    assert "contract changed" in completed.cells[0].detail
    assert completed.cells[0].source_reference is CheckStatus.UNVERIFIED


def test_failed_cell_is_not_overwritten_by_a_later_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, registration = _plan(
        _metadata(lambda x: x, input_values=[np.asarray([1], np.float64)])
    )
    failed = probe.execute_probe_plan(plan, [registration], stage="prepare", limit=1)
    assert failed.cells[0].source_reference is CheckStatus.FAILED

    def forbidden(*args: Any, **kwargs: Any):
        pytest.fail("A completed failed cell must not be overwritten")

    monkeypatch.setattr(probe, "prepare_source_case", forbidden)
    repeated = probe.execute_probe_plan(
        failed, [registration], stage="prepare", limit=1, scenario="raw-x64-off"
    )
    assert repeated.cells[0] == failed.cells[0]


def test_default_cli_only_plans_and_execution_requires_explicit_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any):
        pytest.fail("Default planning must not import/execute raw models")

    monkeypatch.setattr(probe, "raw_registrations", forbidden)
    output = tmp_path / "plan.json"
    assert probe.main(["--output", str(output)]) == 0
    payload = json.loads(output.read_text())
    assert payload["counts"]["cells"] == 3604
    assert (
        payload["scenario_policy"]["factory"] == "instantiate original recipe unchanged"
    )
    with pytest.raises(SystemExit):
        probe.main(["--stage", "product"])
