# tests/extra_tests/converter/test_source_precision_raw_replay.py

"""General replay tests retain original sources, options and separate failures."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import warnings

import jax
import numpy as np
import onnx
import pytest

from jax2onnx import user_interface
from jax2onnx.precision import CheckStatus, get_precision_report
from scripts import source_precision_probe as probe
from scripts.source_precision_inventory import raw_registrations
from tests.extra_tests.converter import source_precision_raw_replay as raw
from tests.extra_tests.converter.source_precision_support import (
    SourceProvenance,
    raw_source_scenarios,
)


@pytest.fixture(scope="module")
def original_scope():
    payload = probe.DEFAULT_BASELINE.read_bytes()
    inventory = json.loads(payload)
    plan = probe.build_probe_plan(
        inventory, baseline_sha256=hashlib.sha256(payload).hexdigest()
    )
    registrations, _ = raw_registrations()
    adapters = json.loads(
        (
            probe.REPOSITORY_ROOT
            / "internal_docs/precision/source_replay_adapters.json"
        ).read_text()
    )
    return inventory, plan, registrations, adapters


@pytest.fixture(autouse=True)
def no_runtime(monkeypatch: pytest.MonkeyPatch):
    import onnxruntime

    def forbidden(*args: Any, **kwargs: Any):
        pytest.fail("The general product lane must not execute ORT")

    monkeypatch.setattr(onnxruntime, "InferenceSession", forbidden)


def _cell(plan, origin, ordinal, scenario="raw-x64-on"):
    return next(
        cell for cell in plan.cells if cell.identity == (origin, ordinal, scenario)
    )


def _execute(original_scope, tmp_path, origin, ordinal, **options):
    _, plan, registrations, adapters = original_scope
    return probe.execute_probe_plan(
        plan,
        registrations,
        raw_replay=True,
        stage="product",
        limit=1,
        cases=frozenset({SourceProvenance(origin, ordinal)}),
        scenario="raw-x64-on",
        replay_adapters=adapters,
        evidence_directory=tmp_path,
        **options,
    )


@pytest.mark.parametrize("expected", [None, "2" * 64])
def test_direct_adapter_requires_independent_inventory_binding(expected):
    row = SimpleNamespace(
        origin="plugin::synthetic",
        ordinal=0,
        contract_sha256="3" * 64,
        unresolved_recipes=(),
    )
    manifest = {
        "inventory_sha256": "1" * 64,
        "rows": {"plugin::synthetic::0": {"contract_sha256": row.contract_sha256}},
    }
    with pytest.raises(ValueError):
        raw.replay_adapter(row, manifest, _inventory_sha256=expected)


def test_direct_adapter_preserves_explicit_empty_and_bound_row_contracts():
    row = SimpleNamespace(
        origin="plugin::synthetic",
        ordinal=0,
        contract_sha256="3" * 64,
        unresolved_recipes=(),
    )
    adapter = {"contract_sha256": row.contract_sha256}
    manifest = {"inventory_sha256": "1" * 64, "rows": {"plugin::synthetic::0": adapter}}
    assert raw.replay_adapter(row, {}) == {}
    assert raw.replay_adapter(row, manifest, _inventory_sha256="1" * 64) is adapter
    adapter["contract_sha256"] = "4" * 64
    with pytest.raises(ValueError, match="exact raw contract"):
        raw.replay_adapter(row, manifest, _inventory_sha256="1" * 64)


def test_adapter_manifest_keeps_frozen_denominator_and_unbound_dimensions(
    original_scope,
):
    inventory, plan, _, adapters = original_scope
    assert len(plan.rows) == 1852 and len(plan.cells) == 3704
    assert adapters["inventory_sha256"] == plan.baseline_sha256
    assert adapters["counts"] == {
        "raw_rows": 1852,
        "symbolic_rows": 310,
        "opaque_rows": 63,
        "B_only_rows": 296,
    }
    rows = {(row.origin, row.ordinal): row for row in plan.rows}
    opaque_indices = {index for index, _ in inventory["unresolved_recipes"]}
    expected_keys = {
        f"{inventory['registrations'][case[0]][0]}::{case[1]}"
        for index, case in enumerate(inventory["cases"])
        if index in opaque_indices
        or any(
            not isinstance(dim, int)
            for shape in inventory["contracts"][case[5]].get("input_shapes", [])
            for dim in shape["tuple"]
        )
    }
    assert set(adapters["rows"]) == expected_keys
    assert len(expected_keys) == 357
    assert sum(bool(row.unresolved_recipes) for row in plan.rows) == 63
    for key, entry in adapters["rows"].items():
        origin, ordinal = key.rsplit("::", 1)
        assert entry["contract_sha256"] == rows[origin, int(ordinal)].contract_sha256
        if entry.get("pending_symbols"):
            assert entry["samples"] == []
        elif "samples" in entry:
            assert entry["samples"] == [{"B": 2}, {"B": 5}]
    # These are separate obligations, never added to the raw-row denominator.
    assert len(inventory["registry_without_metadata"]) == 104
    assert (
        sum(
            row.origin.startswith("example::examples.maxtext::")
            or row.origin.startswith("example::examples.maxdiffusion::")
            for row in plan.rows
        )
        == 50
    )


def test_original_nonleading_symbol_captures_B2_B5_before_product_admission(
    original_scope, tmp_path
):
    origin = "plugin::jax.numpy.squeeze"
    result = _execute(original_scope, tmp_path, origin, 7)
    cell = _cell(result, origin, 7)
    record = cell.history[-1]
    assert cell.source_reference is CheckStatus.VALIDATED
    for role, shape in (("inputs", (1, 2, 1)), ("native-sample-1-inputs", (1, 5, 1))):
        saved = record.replay_capture[role]
        assert saved["status"] == "captured"
        with np.load(saved["path"]) as data:
            assert data["value_0"].shape == shape
    for role in ("eager", "fresh-jit", "concrete-typed-trace", "declared-typed-trace"):
        assert Path(record.replay_capture[role]["path"]).is_file()
    # The real facade, with its ordinary SOURCE validator, owns this outcome.
    if cell.artifact is CheckStatus.UNVERIFIED:
        failure = record.replay_capture["source_error"]
        assert failure["category"] == "profile-exclusion"
        assert failure["phase"] and failure["requirement"] and failure["action"]
    assert record.runtime is record.numerics is CheckStatus.UNVERIFIED
    before = original_scope[1]
    assert [value for value in result.cells if value.identity != cell.identity] == [
        value for value in before.cells if value.identity != cell.identity
    ]


def test_original_opaque_linear_is_captured_without_reconstructing_its_weights(
    original_scope, tmp_path
):
    _, plan, registrations, adapters = original_scope
    origin = "plugin::eqx.nn.linear"
    row = next(row for row in plan.rows if (row.origin, row.ordinal) == (origin, 2))
    metadata = probe._resolve_case(
        row, {name: value for name, _, value in registrations}
    )
    target = metadata["callable"]
    capture = raw.RawReplayCapture(tmp_path / "linear")
    scenario = raw_source_scenarios(metadata, SourceProvenance(origin, 2))[1]
    prepared = raw.prepare_raw_source_case(
        metadata,
        scenario,
        adapter=raw.replay_adapter(
            row, adapters, _inventory_sha256=plan.baseline_sha256
        ),
        capture=capture,
    )
    assert prepared.fn is target
    recipe = json.loads(Path(capture.records["original-recipe"]["path"]).read_text())
    assert recipe["original_type"] == "equinox.nn._linear.Linear"
    assert (
        recipe["fields"]["weight"]["sha256"]
        == hashlib.sha256(np.asarray(target.weight).tobytes()).hexdigest()
    )
    assert np.array_equal(
        prepared.reference_outputs[0], np.asarray(target(*prepared.arguments))
    )
    assert capture.records["fresh-jit"]["status"] == "captured"


def test_unhandled_original_state_stays_a_replay_gap(original_scope, tmp_path):
    result = _execute(original_scope, tmp_path, "plugin::eqx.nn.batch_norm", 0)
    cell = _cell(result, "plugin::eqx.nn.batch_norm", 0)
    assert cell.category == "replay-lane-gap"
    assert cell.source_reference is cell.artifact is CheckStatus.UNVERIFIED
    assert cell.history[-1].raw_callback_status == "not-executed"
    assert "capture adapter" in cell.detail or "adapter" in cell.detail


def test_missing_optional_origin_gets_an_explicit_outcome_without_losing_cells(
    original_scope, tmp_path
):
    _, plan, _, adapters = original_scope
    row = next(
        row for row in plan.rows if row.origin.startswith("example::examples.maxtext::")
    )
    result = probe.execute_probe_plan(
        plan,
        [],
        stage="product",
        raw_replay=True,
        limit=2,
        origins=frozenset({row.origin}),
        replay_adapters=adapters,
        evidence_directory=tmp_path,
    )
    selected = [cell for cell in result.cells if cell.origin == row.origin]
    assert len(selected) == 2
    assert all(cell.category == "environment-unavailable" for cell in selected)
    assert all(
        cell.source_reference is cell.artifact is CheckStatus.UNVERIFIED
        for cell in selected
    )
    assert len(result.cells) == 3704 and len(result.rows) == 1852


@pytest.mark.parametrize("return_mode", ["proto", "ir", "file"])
def test_original_source_forwards_names_recording_web_and_return_facade(
    original_scope, tmp_path, monkeypatch, return_mode
):
    _, plan, registrations, _ = original_scope
    row = next(
        row for row in plan.rows if (row.origin, row.ordinal) == ("plugin::add", 0)
    )
    metadata = probe._resolve_case(
        row, {name: value for name, _, value in registrations}
    )
    scenario = raw_source_scenarios(
        metadata, SourceProvenance(row.origin, row.ordinal)
    )[1]
    capture = raw.RawReplayCapture(tmp_path / return_mode)
    prepared = raw.prepare_raw_source_case(
        metadata, scenario, adapter={}, capture=capture
    )
    options = {
        **metadata,
        "input_names": [f"raw_input_{i}" for i in range(len(prepared.input_specs))],
        "output_names": ["raw_output"],
        "return_mode": return_mode,
        "export_mode": "web",
        "record_primitive_calls_file": "original-recording.json",
    }
    prepared = replace(prepared, metadata=options)
    actual = user_interface._to_onnx
    seen = []

    def observed(**kwargs):
        seen.append(kwargs)
        return actual(**kwargs)

    monkeypatch.setattr(user_interface, "_to_onnx", observed)
    result = raw.run_raw_source_case(prepared, capture=capture)
    assert seen[0]["return_mode"] == return_mode and seen[0]["export_mode"] == "web"
    assert seen[0]["input_names"] == options["input_names"]
    assert seen[0]["record_primitive_calls_file"] == str(
        capture.directory / "primitive-calls.json"
    )
    assert result.model is not None
    assert Path(capture.records["artifact"]["path"]).is_file()
    assert result.schema is CheckStatus.VALIDATED
    assert result.numerical_status is CheckStatus.UNVERIFIED
    if return_mode == "file":
        assert (capture.directory / "facade-return.onnx").is_file()


def test_original_NCHW_recipe_has_correct_physical_native_feeds_and_references(
    original_scope, tmp_path, monkeypatch
):
    actual = user_interface._to_onnx
    seen = []

    def observed(**kwargs):
        seen.append(kwargs)
        return actual(**kwargs)

    monkeypatch.setattr(user_interface, "_to_onnx", observed)
    result = _execute(original_scope, tmp_path, "example::examples.nnx::SimpleModel", 1)
    cell = _cell(result, "example::examples.nnx::SimpleModel", 1)
    assert seen[0]["inputs_as_nchw"] == seen[0]["outputs_as_nchw"] == [0]
    record = cell.history[-1].replay_capture
    with (
        np.load(record["inputs"]["path"]) as native,
        np.load(record["physical-feeds"]["path"]) as physical,
    ):
        assert np.array_equal(
            physical["value_0"], np.transpose(native["value_0"], (0, 3, 1, 2))
        )
    with (
        np.load(record["eager"]["path"]) as native,
        np.load(record["eager-physical-reference"]["path"]) as physical,
    ):
        assert np.array_equal(
            physical["value_0"], np.transpose(native["value_0"], (0, 3, 1, 2))
        )
    assert cell.runtime is cell.numerics is CheckStatus.UNVERIFIED


def test_raw_callback_negative_retains_the_actual_original_export(
    original_scope, tmp_path
):
    _, plan, registrations, _ = original_scope
    row = next(
        row for row in plan.rows if (row.origin, row.ordinal) == ("plugin::add", 0)
    )
    metadata = probe._resolve_case(
        row, {name: value for name, _, value in registrations}
    )
    capture = raw.RawReplayCapture(tmp_path / "callback-negative")
    scenario = raw_source_scenarios(
        metadata, SourceProvenance(row.origin, row.ordinal)
    )[1]
    prepared = raw.prepare_raw_source_case(
        metadata, scenario, adapter={}, capture=capture
    )
    original_callback = metadata.get("post_check_onnx_graph")
    observed = []

    def failed_callback(model):
        # Fault injection verifies failure retention using the real original
        # export and callback; it supplies no positive acceptance evidence.
        saved = capture.records["artifact"]
        assert (
            hashlib.sha256(Path(saved["path"]).read_bytes()).hexdigest()
            == saved["sha256"]
        )
        observed.append(original_callback(model) if original_callback else None)
        return False

    prepared = replace(
        prepared, metadata={**metadata, "post_check_onnx_graph": failed_callback}
    )
    result = raw.run_raw_source_case(prepared, capture=capture)
    assert observed and result.raw_callback_status == "failed"
    assert result.model is not None and result.schema is CheckStatus.VALIDATED
    onnx.checker.check_model(
        onnx.load(capture.records["artifact"]["path"]), full_check=True
    )
    assert capture.records["raw_callback"]["status"] == "failed"
    assert result.numerical_status is CheckStatus.UNVERIFIED


def test_original_FP64_payload_conflict_is_saved_without_a_feed_cast(
    original_scope, tmp_path
):
    _, plan, registrations, adapters = original_scope
    result = probe.execute_probe_plan(
        plan,
        registrations,
        stage="product",
        raw_replay=True,
        limit=1,
        cases=frozenset({SourceProvenance("plugin::gather", 0)}),
        scenario="raw-x64-off",
        replay_adapters=adapters,
        evidence_directory=tmp_path,
    )
    cell = _cell(result, "plugin::gather", 0, "raw-x64-off")
    assert cell.category == "configuration-conflict"
    assert cell.source_reference is CheckStatus.FAILED
    assert cell.artifact is cell.runtime is cell.numerics is CheckStatus.UNVERIFIED
    record = cell.history[-1]
    assert [item.logical.dtype for item in record.supplied_inputs] == [
        "float64",
        "int64",
    ]
    with np.load(record.replay_capture["supplied-inputs"]["path"]) as values:
        assert (
            values["value_0"].dtype == np.float64
            and values["value_1"].dtype == np.int64
        )
    repeated = probe.execute_probe_plan(
        result,
        registrations,
        stage="product",
        raw_replay=True,
        limit=1,
        cases=frozenset({SourceProvenance("plugin::gather", 0)}),
        scenario="raw-x64-off",
        replay_adapters=adapters,
        evidence_directory=tmp_path / "repeat",
    )
    assert repeated == result


def test_original_runtime_parameter_is_not_frozen_away(original_scope, tmp_path):
    result = _execute(original_scope, tmp_path, "plugin::linen.dropout", 1)
    record = _cell(result, "plugin::linen.dropout", 1).history[-1]
    inputs = record.replay_capture["inputs"]
    with np.load(inputs["path"]) as values:
        assert values["value_1"].shape == ()
        assert values["value_1"].dtype == np.bool_ and bool(values["value_1"])
    assert "fresh-jit" in record.replay_capture
    assert record.numerics is record.runtime is CheckStatus.UNVERIFIED


def _original_metadata(original_scope, origin, ordinal):
    _, plan, registrations, adapters = original_scope
    row = next(
        row for row in plan.rows if (row.origin, row.ordinal) == (origin, ordinal)
    )
    metadata = probe._resolve_case(
        row, {name: value for name, _, value in registrations}
    )
    return row, metadata, adapters


def test_physical_adapter_failure_preserves_real_product_observations(
    original_scope, tmp_path, monkeypatch
):
    def unavailable(*args, **kwargs):
        raise TypeError("injected physical encoding failure")

    monkeypatch.setattr(raw, "_physical_arrays", unavailable)
    result = _execute(original_scope, tmp_path, "plugin::add", 0)
    cell = _cell(result, "plugin::add", 0)
    capture = cell.history[-1].replay_capture
    assert cell.category == "replay-lane-gap"
    assert cell.source_reference is CheckStatus.UNVERIFIED
    assert capture["replay_readiness"]["status"] == "unverified"
    assert capture["product_admission"] == "returned-artifact"
    assert cell.schema is cell.lowering_audit is cell.artifact is CheckStatus.VALIDATED
    assert cell.history[-1].raw_callback_status in {"absent", "passed", "failed"}
    assert all(
        outcome["status"] == "returned"
        for outcome in capture["native_outcomes"].values()
    )
    bindings = json.loads(Path(capture["physical-port-bindings"]["path"]).read_text())
    assert bindings["captures"] == {}
    assert cell.runtime is cell.numerics is CheckStatus.UNVERIFIED


@pytest.mark.parametrize(
    "role",
    ["eager", "captured-constants", "concrete-typed-trace", "declared-typed-trace"],
)
def test_serialization_failure_does_not_invent_native_failure_or_suppress_declared_trace(
    original_scope, tmp_path, monkeypatch, role
):
    arrays = raw.RawReplayCapture.arrays
    document = raw.RawReplayCapture.document

    def save_arrays(self, label, values):
        if label == role:
            raise TypeError("injected array evidence adapter failure")
        return arrays(self, label, values)

    def save_document(self, label, value):
        if label == role:
            raise TypeError("injected trace evidence adapter failure")
        return document(self, label, value)

    monkeypatch.setattr(raw.RawReplayCapture, "arrays", save_arrays)
    monkeypatch.setattr(raw.RawReplayCapture, "document", save_document)
    result = _execute(original_scope, tmp_path, "plugin::add", 0)
    cell = _cell(result, "plugin::add", 0)
    capture = cell.history[-1].replay_capture
    assert cell.category == "replay-lane-gap"
    assert cell.source_reference is CheckStatus.UNVERIFIED
    assert capture["replay_readiness"]["status"] == "unverified"
    assert capture[role]["status"] == "failed"
    for attempt in (
        "eager",
        "fresh-jit",
        "concrete-typed-trace",
        "declared-typed-trace",
    ):
        assert capture["native_outcomes"][attempt]["status"] == "returned"
    assert cell.runtime is cell.numerics is CheckStatus.UNVERIFIED


def test_concrete_manifest_failure_still_attempts_declared_native_trace(
    original_scope, tmp_path, monkeypatch
):
    from tests.extra_tests.converter import source_precision_diagnostics as diagnostics

    actual = diagnostics.typed_trace_manifest
    observed = []

    def first_manifest_unavailable(closed):
        observed.append(closed)
        if len(observed) == 1:
            raise TypeError("injected concrete typed-manifest failure")
        return actual(closed)

    monkeypatch.setattr(diagnostics, "typed_trace_manifest", first_manifest_unavailable)
    result = _execute(original_scope, tmp_path, "plugin::add", 0)
    cell = _cell(result, "plugin::add", 0)
    capture = cell.history[-1].replay_capture
    # Two original manifest attempts remain, followed by the actual export and
    # fresh prepared manifests used only by the optional artifact bridge.
    assert len(observed) == 4
    assert capture["native_outcomes"]["concrete-typed-trace"]["status"] == "returned"
    assert capture["native_outcomes"]["declared-typed-trace"]["status"] == "returned"
    assert capture["concrete-typed-trace"]["status"] == "failed"
    assert capture["declared-typed-trace"]["status"] == "captured"
    assert cell.category == "replay-lane-gap"


def test_selected_typed_key_capture_retains_logical_properties_and_original_words(
    tmp_path,
):
    import jax

    key = jax.random.key(17, impl="threefry2x32")
    capture = raw.RawReplayCapture(tmp_path / "typed-key")
    capture.arrays("captured-constants", [key])
    record = capture.records["captured-constants"]
    assert record["properties"][0] == {
        "dtype": key.dtype.name,
        "shape": [],
        "weak_type": False,
        "encoding": "threefry-key-words",
        "physical_dtype": "uint32",
        "physical_shape": [2],
    }
    with np.load(record["path"]) as saved:
        assert np.array_equal(saved["value_0"], np.asarray(jax.random.key_data(key)))


def test_original_factory_binds_and_checks_the_actual_instantiated_Eqx_target(
    original_scope, tmp_path
):
    row, metadata, adapters = _original_metadata(
        original_scope, "example::examples.eqx::MlpExample", 1
    )
    capture = raw.RawReplayCapture(tmp_path / "original-eqx-factory")
    scenario = raw_source_scenarios(
        metadata, SourceProvenance(row.origin, row.ordinal)
    )[1]
    prepared = raw.prepare_raw_source_case(
        metadata,
        scenario,
        adapter=raw.replay_adapter(
            row, adapters, _inventory_sha256=original_scope[1].baseline_sha256
        ),
        capture=capture,
    )
    assert capture.target is prepared.fn and prepared.fn is not metadata["callable"]
    assert capture.records["instantiated-target-state"]["status"] == "captured"
    assert capture.records["native-preparation-complete-state"]["status"] == "unchanged"
    assert capture.readiness is CheckStatus.VALIDATED


def test_mutating_factory_instance_is_a_lane_gap_before_repeated_native_calls(
    original_scope, tmp_path
):
    from flax import nnx
    from jax2onnx.plugins.plugin_system import construct_and_call
    from tests.extra_tests.converter.source_precision_support import (
        SourcePreparationError,
    )

    row, metadata, _ = _original_metadata(
        original_scope, "example::examples.nnx::SimpleModel", 1
    )
    original_recipe = metadata["callable"]
    instances = []

    class MutatingOriginal(nnx.Module):
        def __init__(self):
            self.original = original_recipe.instantiate()
            self.calls = nnx.Variable(np.asarray(0, np.int32))

        def __call__(self, value):
            self.calls.value = self.calls.value + np.asarray(1, np.int32)
            return self.original(value)

    def constructor():
        target = MutatingOriginal()
        instances.append(target)
        return target

    # A deliberate state fault wraps the complete original callable/recipe. It
    # tests capture rejection and supplies no positive model acceptance evidence.
    metadata = {**metadata, "callable": construct_and_call(constructor)}
    scenario = raw_source_scenarios(
        metadata, SourceProvenance(row.origin, row.ordinal)
    )[1]
    capture = raw.RawReplayCapture(tmp_path / "mutable-factory")
    with pytest.raises(SourcePreparationError, match="state is not preserved") as error:
        raw.prepare_raw_source_case(metadata, scenario, adapter={}, capture=capture)
    assert error.value.category == "replay-lane-gap"
    assert len(instances) == 1 and capture.target is instances[0]
    assert int(instances[0].calls.value) == 1
    assert capture.records["native_outcomes"]["eager"]["status"] == "returned"
    assert "fresh-jit" not in capture.records["native_outcomes"]
    assert capture.readiness is CheckStatus.UNVERIFIED
    assert capture.records["product_admission"] == "not-attempted"


@pytest.mark.parametrize(
    "component,prefix",
    [("ContinuousTanhPolicy", "_CONTINUOUS"), ("DiscreteArgmaxPolicy", "_DISCRETE")],
)
def test_original_policy_binds_every_referenced_global_weight(
    original_scope, tmp_path, component, prefix
):
    _, metadata, _ = _original_metadata(
        original_scope, "example::examples.rl::" + component, 0
    )
    target = metadata["callable"]
    capture = raw.RawReplayCapture(tmp_path / component)
    capture.bind_target(target)
    assert capture.target is target
    state = capture.target_state["referenced_globals"]
    expected = {prefix + suffix for suffix in ("_W1", "_B1", "_W2", "_B2")}
    assert expected <= state.keys()
    for name in expected:
        original = target.__globals__[name]
        assert state[name]["sha256"] == hashlib.sha256(original.tobytes()).hexdigest()
        assert state[name]["array_dtype"] == original.dtype.name
    assert capture.check_target("original-global-weight-state")


def test_original_global_weight_rebinding_is_detected_without_rebuilding_source(
    original_scope, tmp_path, monkeypatch
):
    _, metadata, _ = _original_metadata(
        original_scope, "example::examples.rl::ContinuousTanhPolicy", 0
    )
    target = metadata["callable"]
    original = target.__globals__["_CONTINUOUS_W1"]
    original_bytes = original.tobytes()
    capture = raw.RawReplayCapture(tmp_path / "global-state-fault")
    capture.bind_target(target)
    changed = original.copy()
    changed.flat[0] = np.nextafter(changed.flat[0], np.float32(np.inf))
    monkeypatch.setitem(target.__globals__, "_CONTINUOUS_W1", changed)
    assert not capture.check_target("global-weight-fault")
    assert capture.target is target and original.tobytes() == original_bytes
    assert capture.records["global-weight-fault-state"]["status"] == "failed"
    assert capture.readiness is CheckStatus.UNVERIFIED
    assert capture.records["native_outcomes"] == {}


def _bridge_prepared(original_scope, tmp_path, *, return_mode="proto"):
    row, metadata, _ = _original_metadata(original_scope, "plugin::add", 0)
    capture = raw.RawReplayCapture(tmp_path)
    scenario = raw_source_scenarios(
        metadata, SourceProvenance(row.origin, row.ordinal)
    )[1]
    prepared = raw.prepare_raw_source_case(
        metadata, scenario, adapter={}, capture=capture
    )
    return replace(prepared, metadata={**metadata, "return_mode": return_mode}), capture


def _bridge_record(capture):
    # A persisted bridge record must remain ordinary metadata, with no native
    # trace, IR, ModelProto, array, or original payload object retained in it.
    record = capture.records["source_structure"]
    json.dumps(record, allow_nan=False)
    return record


@pytest.mark.parametrize("return_mode", ["proto", "ir", "file"])
def test_shared_structure_binds_real_return_without_changing_its_mode(
    original_scope, tmp_path, return_mode
):
    prepared, capture = _bridge_prepared(
        original_scope, tmp_path / return_mode, return_mode=return_mode
    )
    result = raw.run_raw_source_case(prepared, capture=capture)
    record = _bridge_record(capture)
    assert result.model is not None and result.schema is CheckStatus.VALIDATED
    assert capture.readiness is CheckStatus.VALIDATED
    assert record["collector_calls"] == 1
    assert record["ir_result"]["status"] == "validated"
    assert record["source_binding"]["status"] == "validated"
    expected = (
        CheckStatus.UNVERIFIED if return_mode == "file" else CheckStatus.VALIDATED
    )
    assert capture.source_structure_status() is expected
    assert capture.records["facade_return"]["kind"] == return_mode
    assert record["artifact_binding"]["return_kind"] == return_mode
    if return_mode != "file":
        binding = record["artifact_binding"]
        for role in (
            "actual_return",
            "normalized_return",
            "saved_artifact",
            "after_callback",
            "facade_after_callback",
        ):
            assert binding[role] == binding["checked_ir_proto"]
        assert binding["fenced_after_callback"] is True
        assert (
            capture.records["artifact"]["sha256"]
            == binding["checked_ir_proto"]["sha256"]
        )
    else:
        assert (
            "file-or-external-bundle-binding-unverified"
            in record["artifact_binding"]["reasons"]
        )
    assert result.numerical_status is CheckStatus.UNVERIFIED


def test_raw_probe_uses_bound_shared_structure_separately_from_callback(
    original_scope, tmp_path
):
    result = _execute(original_scope, tmp_path, "plugin::add", 0)
    cell = _cell(result, "plugin::add", 0)
    record = cell.history[-1]
    assert cell.source_reference is CheckStatus.VALIDATED
    assert record.source_reference is CheckStatus.VALIDATED
    assert record.structure is CheckStatus.VALIDATED
    assert record.replay_capture["source_structure"]["status"] == "validated"
    assert cell.runtime is cell.numerics is CheckStatus.UNVERIFIED


def test_missing_shared_rule_is_unverified_without_affecting_product(
    original_scope, tmp_path, monkeypatch
):
    from jax2onnx.plugins import _source_structure as rules

    monkeypatch.setattr(
        rules, "freeze_source_structure_rule", lambda eqn, inputs, outputs: None
    )
    prepared, capture = _bridge_prepared(original_scope, tmp_path / "missing-rule")
    result = raw.run_raw_source_case(prepared, capture=capture)
    record = _bridge_record(capture)
    assert record["ir_result"]["status"] == "unverified"
    assert capture.source_structure_status() is CheckStatus.UNVERIFIED
    assert result.model is not None and result.schema is CheckStatus.VALIDATED
    assert capture.readiness is CheckStatus.VALIDATED


def test_actual_ir_contradiction_retains_the_original_shared_result(
    original_scope, tmp_path, monkeypatch
):
    actual = user_interface.validate_source_structure

    def mutate_then_check(model):
        next(node for node in model.graph if node.op_type == "Add").op_type = "Sub"
        return actual(model)

    monkeypatch.setattr(user_interface, "validate_source_structure", mutate_then_check)
    prepared, capture = _bridge_prepared(
        original_scope, tmp_path / "bad-ir", return_mode="ir"
    )
    result = raw.run_raw_source_case(prepared, capture=capture)
    record = _bridge_record(capture)
    assert record["ir_result"]["status"] == "failed"
    assert record["ir_result"]["reasons"]
    assert record["source_binding"]["status"] == "validated"
    assert record["artifact_binding"]["status"] == "validated"
    assert capture.source_structure_status() is CheckStatus.FAILED
    assert result.model is not None and capture.readiness is CheckStatus.VALIDATED


def test_same_typed_returned_proto_drift_cannot_reuse_ir_acceptance(
    original_scope, tmp_path, monkeypatch
):
    actual = user_interface._to_onnx

    def return_changed(**kwargs):
        value = actual(**kwargs)
        next(node for node in value.graph.node if node.op_type == "Add").op_type = "Sub"
        return value

    monkeypatch.setattr(user_interface, "_to_onnx", return_changed)
    prepared, capture = _bridge_prepared(original_scope, tmp_path / "bad-return")
    result = raw.run_raw_source_case(prepared, capture=capture)
    record = _bridge_record(capture)
    assert record["ir_result"]["status"] == "validated"
    assert record["artifact_binding"]["status"] == "failed"
    assert (
        "normalized-return-and-checked-ir-differ"
        in record["artifact_binding"]["reasons"]
    )
    assert capture.source_structure_status() is CheckStatus.FAILED
    assert result.schema is CheckStatus.VALIDATED
    assert capture.readiness is CheckStatus.VALIDATED


@pytest.mark.parametrize("fault", ["model", "saved", "trace"])
@pytest.mark.parametrize("callback_outcome", ["false", "raises"])
def test_after_callback_fence_detects_mutation_even_when_callback_fails(
    original_scope, tmp_path, fault, callback_outcome
):
    prepared, capture = _bridge_prepared(original_scope, tmp_path / "callback-fault")

    def callback(model):
        if fault == "model":
            next(node for node in model.graph.node if node.op_type == "Add").op_type = (
                "Sub"
            )
        elif fault == "saved":
            path = Path(capture.records["artifact"]["path"])
            changed = onnx.load_model(path)
            next(
                node for node in changed.graph.node if node.op_type == "Add"
            ).op_type = "Sub"
            path.write_bytes(changed.SerializeToString(deterministic=True))
        else:
            path = Path(
                capture.records["source_structure"]["source_binding"][
                    "export_trace_manifest"
                ]["path"]
            )
            path.write_text("{}")
        if callback_outcome == "raises":
            raise RuntimeError("injected callback failure")
        return False

    prepared = replace(
        prepared, metadata={**prepared.metadata, "post_check_onnx_graph": callback}
    )
    result = raw.run_raw_source_case(prepared, capture=capture)
    record = _bridge_record(capture)
    assert result.raw_callback_status == (
        "error" if callback_outcome == "raises" else "failed"
    )
    assert record["ir_result"]["status"] == "validated"
    assert capture.source_structure_status() is CheckStatus.FAILED
    assert capture.readiness is CheckStatus.VALIDATED
    assert result.schema is CheckStatus.VALIDATED


def test_failed_callback_keeps_independent_validated_shared_structure(
    original_scope, tmp_path
):
    prepared, capture = _bridge_prepared(original_scope, tmp_path / "failed-callback")
    prepared = replace(
        prepared,
        metadata={**prepared.metadata, "post_check_onnx_graph": lambda model: False},
    )
    result = raw.run_raw_source_case(prepared, capture=capture)
    assert result.raw_callback_status == "failed"
    assert capture.source_structure_status() is CheckStatus.VALIDATED
    assert capture.readiness is CheckStatus.VALIDATED


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "raises", "sink", "ir-result"]
)
def test_bridge_failure_is_local_to_structure_evidence(
    original_scope, tmp_path, monkeypatch, fault
):
    actual = user_interface._to_onnx

    def export(**kwargs):
        collector = kwargs["source_structure_collector"]
        if fault == "missing":
            kwargs["source_structure_collector"] = None
        elif fault == "duplicate":

            def duplicate(*args):
                collector(*args)
                collector(*args)

            kwargs["source_structure_collector"] = duplicate
        elif fault == "raises":

            def broken(*args):
                raise OSError("injected private evidence failure")

            kwargs["source_structure_collector"] = broken
        elif fault == "ir-result":

            def missing_result(trace, final_ir, result, returned):
                collector(trace, final_ir, None, returned)

            kwargs["source_structure_collector"] = missing_result
        return actual(**kwargs)

    monkeypatch.setattr(user_interface, "_to_onnx", export)
    if fault == "sink":
        actual_open = Path.open

        def open_evidence(path, *args, **kwargs):
            if path.name == "source-export-typed-trace.json":
                raise OSError("injected bridge sink failure")
            return actual_open(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", open_evidence)
    prepared, capture = _bridge_prepared(original_scope, tmp_path / fault)
    result = raw.run_raw_source_case(prepared, capture=capture)
    assert capture.source_structure_status() is CheckStatus.UNVERIFIED
    assert capture.readiness is CheckStatus.VALIDATED and capture.gaps == []
    assert result.model is not None and result.schema is CheckStatus.VALIDATED
    assert result.audit.status is CheckStatus.VALIDATED
    assert all(
        value["status"] == "returned"
        for value in capture.records["native_outcomes"].values()
    )
    record = _bridge_record(capture)
    if fault == "ir-result":
        assert record["artifact_binding"]["checked_ir_proto"]
        assert record["artifact_binding"]["actual_return"]
        assert record["artifact_binding"]["status"] == "unverified"


_BRIDGE_LITERAL = np.float32(0)


def _bridge_root_literal():
    return _BRIDGE_LITERAL


def _bridge_sub_literal(x):
    return jax.lax.sub(x, _BRIDGE_LITERAL)


def _bridge_literal_prepared(tmp_path, *, scalar=True):
    metadata = {
        "context": "bridge-test",
        "component": "literal",
        "testcase": "exact-words",
        "callable": _bridge_root_literal if scalar else _bridge_sub_literal,
        "input_shapes": [] if scalar else [(2,)],
        "opset_version": 23,
    }
    capture = raw.RawReplayCapture(tmp_path)
    scenario = raw_source_scenarios(metadata, SourceProvenance("bridge-test", 0))[1]
    prepared = raw.prepare_raw_source_case(
        metadata, scenario, adapter={}, capture=capture
    )
    return prepared, capture


@pytest.mark.parametrize("return_mode", ["proto", "ir"])
def test_actual_raw_collector_preserves_literal_bytes_and_product_evidence(
    tmp_path, monkeypatch, return_mode
):
    monkeypatch.setitem(
        _bridge_sub_literal.__globals__, "_BRIDGE_LITERAL", np.float32(1.25)
    )
    actual = user_interface._to_onnx
    observations = []
    structures = []

    def without_collector(**kwargs):
        kwargs["source_structure_collector"] = None
        return actual(**kwargs)

    for enabled in (False, True):
        prepared, capture = _bridge_literal_prepared(
            tmp_path / str(enabled), scalar=False
        )
        prepared = replace(
            prepared, metadata={**prepared.metadata, "return_mode": return_mode}
        )
        with monkeypatch.context() as context:
            if not enabled:
                context.setattr(user_interface, "_to_onnx", without_collector)
            with warnings.catch_warnings(record=True) as emitted:
                warnings.simplefilter("always")
                result = raw.run_raw_source_case(prepared, capture=capture)
        assert result.model is not None
        observations.append(
            (
                result.model.SerializeToString(deterministic=True),
                get_precision_report(result.model),
                tuple((entry.category, str(entry.message)) for entry in emitted),
                result.audit,
                result.schema,
                result.raw_callback_status,
                capture.readiness,
                capture.records["facade_return"]["kind"],
                capture.records["native_outcomes"],
                prepared.original_trace_json,
                prepared.input_evidence,
                prepared.reference_evidence,
            )
        )
        structures.append(capture.source_structure_status())
    assert observations[0] == observations[1]
    assert observations[1][4] is CheckStatus.VALIDATED
    assert observations[1][6] is CheckStatus.VALIDATED
    assert observations[1][7] == return_mode
    assert structures == [CheckStatus.UNVERIFIED, CheckStatus.VALIDATED]


@pytest.mark.parametrize("word", [0x80000000, 0x7FC00001])
@pytest.mark.parametrize("phase", ["prepared", "export-trace"])
def test_frozen_source_words_detect_same_typed_literal_drift(
    tmp_path, monkeypatch, word, phase
):
    original = np.asarray(word, np.uint32).view(np.float32)[()]
    changed = np.asarray(0 if word == 0x80000000 else 0x7FC00002, np.uint32).view(
        np.float32
    )[()]
    monkeypatch.setitem(_bridge_root_literal.__globals__, "_BRIDGE_LITERAL", original)
    prepared, capture = _bridge_literal_prepared(tmp_path / "literal")
    baseline = json.dumps(
        capture.records["source_structure"]["source_binding"]["prepared_storage"],
        sort_keys=True,
    )
    actual = user_interface._to_onnx

    def export(**kwargs):
        collector = kwargs["source_structure_collector"]

        def changed_trace(trace, final_ir, result, returned):
            literal = raw._structure_body(trace.closed_jaxpr).outvars[0]
            assert isinstance(literal, raw.JaxLiteral)
            previous = literal.val
            try:
                literal.val = changed
                collector(trace, final_ir, result, returned)
            finally:
                literal.val = previous

        kwargs["source_structure_collector"] = changed_trace
        return actual(**kwargs)

    if phase == "export-trace":
        monkeypatch.setattr(user_interface, "_to_onnx", export)
        result = raw.run_raw_source_case(prepared, capture=capture)
    else:
        # Mutation before collection must not become a newly derived expected
        # value: the exact-word baseline was frozen during preparation.
        literal = raw._structure_body(prepared.original_jaxpr).outvars[0]
        assert isinstance(literal, raw.JaxLiteral)
        previous = literal.val
        try:
            literal.val = changed
            result = raw.run_raw_source_case(prepared, capture=capture)
        finally:
            literal.val = previous
    record = _bridge_record(capture)
    assert (
        json.dumps(record["source_binding"]["prepared_storage"], sort_keys=True)
        == baseline
    )
    assert record["ir_result"]["status"] == "validated"
    assert record["source_binding"]["status"] == "failed"
    assert (
        "source-constant-or-literal-storage-differs"
        in record["source_binding"]["reasons"]
    )
    assert capture.source_structure_status() is CheckStatus.FAILED
    assert result.schema is CheckStatus.VALIDATED
    assert capture.readiness is CheckStatus.VALIDATED
    saved = json.loads(
        Path(record["source_binding"]["export_trace_manifest"]["path"]).read_text()
    )
    literal = saved["manifest"]["outputs"][0]["literal"]
    assert set(literal) == {"canonical_payload"}
    assert set(literal["canonical_payload"]) == {"sha256", "bytes"}
    assert set(saved["manifest"]["const_payloads"]) == {"canonical_payload"}
    assert "bytes_hex" not in json.dumps(saved)


@pytest.mark.parametrize("drift", [False, True])
def test_known_word_mismatch_precedes_independent_unowned_parameter_gap(
    tmp_path, monkeypatch, drift
):
    monkeypatch.setitem(
        _bridge_sub_literal.__globals__, "_BRIDGE_LITERAL", np.float32(1.25)
    )
    prepared, capture = _bridge_literal_prepared(
        tmp_path / "unowned-param", scalar=False
    )
    actual = user_interface._to_onnx

    def export(**kwargs):
        collector = kwargs["source_structure_collector"]

        def changed_trace(trace, final_ir, result, returned):
            eqn = raw._structure_body(trace.closed_jaxpr).eqns[0]
            literal = next(var for var in eqn.invars if isinstance(var, raw.JaxLiteral))
            previous = literal.val
            assert "injected_unowned" not in eqn.params
            try:
                eqn.params["injected_unowned"] = object()
                if drift:
                    literal.val = np.float32(2.5)
                collector(trace, final_ir, result, returned)
            finally:
                eqn.params.pop("injected_unowned")
                literal.val = previous

        kwargs["source_structure_collector"] = changed_trace
        return actual(**kwargs)

    monkeypatch.setattr(user_interface, "_to_onnx", export)
    raw.run_raw_source_case(prepared, capture=capture)
    record = _bridge_record(capture)
    assert record["ir_result"]["status"] == "validated"
    expected = CheckStatus.FAILED if drift else CheckStatus.UNVERIFIED
    assert capture.source_structure_status() is expected
    assert capture.readiness is CheckStatus.VALIDATED
    if drift:
        assert (
            "source-constant-or-literal-storage-differs"
            in record["source_binding"]["reasons"]
        )


@pytest.mark.parametrize("word", [0x80000000, 0x7FC00001])
def test_same_typed_returned_constant_words_cannot_reuse_ir_observation(
    tmp_path, monkeypatch, word
):
    original = np.asarray(word, np.uint32).view(np.float32)[()]
    changed = np.asarray(0 if word == 0x80000000 else 0x7FC00002, np.uint32).view(
        np.float32
    )
    monkeypatch.setitem(_bridge_root_literal.__globals__, "_BRIDGE_LITERAL", original)
    prepared, capture = _bridge_literal_prepared(tmp_path / "returned-constant")
    actual = user_interface._to_onnx

    def export(**kwargs):
        value = actual(**kwargs)
        tensors = list(value.graph.initializer)
        tensors.extend(
            attribute.t
            for node in value.graph.node
            for attribute in node.attribute
            if attribute.type == onnx.AttributeProto.TENSOR
        )
        assert tensors
        tensor = tensors[0]
        tensor.CopyFrom(onnx.numpy_helper.from_array(changed, name=tensor.name))
        return value

    monkeypatch.setattr(user_interface, "_to_onnx", export)
    result = raw.run_raw_source_case(prepared, capture=capture)
    record = _bridge_record(capture)
    assert record["ir_result"]["status"] == "validated"
    assert record["artifact_binding"]["status"] == "failed"
    assert capture.source_structure_status() is CheckStatus.FAILED
    assert result.schema is CheckStatus.VALIDATED
    assert capture.readiness is CheckStatus.VALIDATED


def test_structure_requires_original_source_readiness(
    original_scope, tmp_path, monkeypatch
):
    prepared, capture = _bridge_prepared(original_scope, tmp_path / "source-gap")
    capture.gap(
        "injected-original-source-gap", TypeError("source evidence unavailable")
    )
    raw.run_raw_source_case(prepared, capture=capture)
    record = _bridge_record(capture)
    assert record["ir_result"]["status"] == "validated"
    assert record["source_binding"]["status"] == "validated"
    assert record["artifact_binding"]["status"] == "validated"
    assert capture.source_structure_status() is CheckStatus.UNVERIFIED
    assert capture.readiness is CheckStatus.UNVERIFIED
    assert capture.records["native_outcomes"]
    assert all(
        value["status"] == "returned"
        for value in capture.records["native_outcomes"].values()
    )


def test_original_function_attributes_are_part_of_target_state(
    original_scope, tmp_path, monkeypatch
):
    _, metadata, _ = _original_metadata(
        original_scope, "example::examples.rl::ContinuousTanhPolicy", 0
    )
    target = metadata["callable"]
    attribute = np.asarray([1, 2], np.float32)
    monkeypatch.setattr(
        target, "source_replay_control_weights", attribute, raising=False
    )
    capture = raw.RawReplayCapture(tmp_path / "function-attribute-fault")
    capture.bind_target(target)
    attribute[0] = np.float32(3)
    assert not capture.check_target("function-attribute-fault")
    assert capture.target is target and capture.readiness is CheckStatus.UNVERIFIED


def test_referenced_custom_class_state_has_an_explicit_unknown_boundary(
    original_scope, tmp_path, monkeypatch
):
    from tests.extra_tests.converter.source_precision_support import (
        SourcePreparationError,
    )

    _, metadata, _ = _original_metadata(
        original_scope, "example::examples.rl::ContinuousTanhPolicy", 0
    )
    target = metadata["callable"]

    class UnknownClassState:
        weight = np.asarray([1], np.float32)

    # A negative binding fault; no native model or original array is modified.
    monkeypatch.setitem(target.__globals__, "_CONTINUOUS_W1", UnknownClassState)
    capture = raw.RawReplayCapture(tmp_path / "class-state-gap")
    with pytest.raises(
        SourcePreparationError, match="exact state capture adapter"
    ) as error:
        capture.bind_target(target)
    assert error.value.category == "replay-lane-gap"
    assert capture.records["native_outcomes"] == {}
    failure = capture.records["instantiated-target-state"]
    assert failure["detail"] == (
        "Referenced function class/static state needs an exact adapter"
    )
    diagnostic = failure["diagnostic"]
    assert diagnostic["class"] == {
        "module": UnknownClassState.__module__,
        "qualname": UnknownClassState.__qualname__,
    }
    assert diagnostic["metaclass"] == {"module": "builtins", "qualname": "type"}
    assert diagnostic["path"] == ["root", "global:_CONTINUOUS_W1"]
    assert diagnostic["callable"] == {
        "module": target.__module__,
        "qualname": target.__qualname__,
        "code_sha256": raw.canonical_code_fingerprint(target.__code__),
    }
    assert diagnostic["defining_module_binding"] == "not-top-level"
    assert failure["status"] == "failed"
    assert capture.readiness is CheckStatus.UNVERIFIED
    assert capture.records["product_admission"] == "not-attempted"
    assert capture.records["facade_return"] == {"status": "not-attempted"}
    assert capture.records["raw_callback"] == {"status": "not-executed"}


@pytest.mark.parametrize("return_mode", ["ir", "file"])
def test_actual_facade_return_survives_deferred_normalization_failure(
    original_scope, tmp_path, monkeypatch, return_mode
):
    from tests.extra_tests.converter import source_precision_support as support

    row, metadata, _ = _original_metadata(original_scope, "plugin::add", 0)
    capture = raw.RawReplayCapture(tmp_path / (return_mode + "-readback-fault"))
    scenario = raw_source_scenarios(
        metadata, SourceProvenance(row.origin, row.ordinal)
    )[1]
    prepared = raw.prepare_raw_source_case(
        metadata, scenario, adapter={}, capture=capture
    )
    prepared = replace(
        prepared,
        metadata={**metadata, "return_mode": return_mode, "export_mode": "web"},
    )
    actual = user_interface._to_onnx
    actual_normalizer = (
        support.ir.to_proto if return_mode == "ir" else support.onnx.load_model
    )
    returned = []
    normalizer_faults = []

    def ordinary_return(**kwargs):
        value = actual(**kwargs)
        returned.append(value)  # Arm only after the real facade has returned.
        return value

    def unavailable_after_return(*args, **kwargs):
        if returned:
            normalizer_faults.append(args[0])
            raise TypeError("injected post-return evidence normalization failure")
        return actual_normalizer(*args, **kwargs)

    monkeypatch.setattr(user_interface, "_to_onnx", ordinary_return)
    if return_mode == "ir":
        monkeypatch.setattr(support.ir, "to_proto", unavailable_after_return)
    else:
        monkeypatch.setattr(support.onnx, "load_model", unavailable_after_return)
    result = raw.run_raw_source_case(prepared, capture=capture)
    assert len(returned) == len(normalizer_faults) == 1
    assert capture.facade_value is returned[0]
    assert capture.records["facade_return"]["status"] == "returned"
    assert capture.records["facade_return"]["kind"] == return_mode
    assert capture.records["product_admission"] == "returned-artifact"
    assert capture.records["product_audit_status"] is CheckStatus.UNVERIFIED
    assert result.category == "replay-lane-gap" and result.model is None
    assert result.schema is result.numerical_status is CheckStatus.UNVERIFIED
    assert result.raw_callback_status == "not-executed"
    if return_mode == "file":
        evidence = capture.records["facade_return"]
        assert evidence["path"] == returned[0]
        assert evidence["files"] and all(
            Path(item["path"]).is_file() for item in evidence["files"]
        )


_RAW_REPLAY_GLOBAL_DEPENDENCY: Any = None


def _call_original_global_dependency(value):
    return _RAW_REPLAY_GLOBAL_DEPENDENCY(value)


@pytest.mark.parametrize("binding_kind", ["model", "bound-method"])
def test_original_global_model_and_bound_owner_keep_actual_weights(
    original_scope, tmp_path, monkeypatch, binding_kind
):
    from tests.extra_tests.converter.source_precision_support import (
        SourcePreparationError,
    )

    _, metadata, _ = _original_metadata(original_scope, "plugin::eqx.nn.linear", 4)
    _, alternate_metadata, _ = _original_metadata(
        original_scope, "plugin::eqx.nn.linear", 3
    )
    original = metadata["callable"]
    # Both complete original Linear instances have the same configuration and
    # different original fixed weights. Neither is rebuilt or changed here.
    alternate = alternate_metadata["callable"].__defaults__[0]
    assert type(original) is type(alternate)
    assert original.in_features == alternate.in_features == 128
    assert original.out_features == alternate.out_features == 64
    originals = [np.asarray(model.weight).tobytes() for model in (original, alternate)]
    assert originals[0] != originals[1]
    value = original if binding_kind == "model" else original.__call__
    changed = alternate if binding_kind == "model" else alternate.__call__
    if binding_kind == "bound-method":
        assert value.__self__ is original and changed.__self__ is alternate
    assert raw._library_namespace(type(original).__module__)
    assert raw._imported_executable_binding(value) is None
    monkeypatch.setitem(
        _call_original_global_dependency.__globals__,
        "_RAW_REPLAY_GLOBAL_DEPENDENCY",
        value,
    )
    capture = raw.RawReplayCapture(tmp_path / ("owned-" + binding_kind))
    try:
        capture.bind_target(_call_original_global_dependency)
    except SourcePreparationError as error:
        # A method's code/owner adapter may be unknown on a supported stack.
        # That is a visible lane gap, never a successful static binding.
        assert binding_kind == "bound-method"
        assert error.category == "replay-lane-gap"
        assert capture.gaps and capture.readiness is CheckStatus.UNVERIFIED
    else:
        state = capture.target_state["referenced_globals"][
            "_RAW_REPLAY_GLOBAL_DEPENDENCY"
        ]
        assert hashlib.sha256(originals[0]).hexdigest() in json.dumps(state)
        assert "library_binding" not in state
        assert capture.check_target("stable-owned-global")
    monkeypatch.setitem(
        _call_original_global_dependency.__globals__,
        "_RAW_REPLAY_GLOBAL_DEPENDENCY",
        changed,
    )
    assert not capture.check_target("changed-owned-global")
    assert capture.records["changed-owned-global-state"]["status"] == "failed"
    assert capture.readiness is CheckStatus.UNVERIFIED
    assert capture.records["native_outcomes"] == {}
    assert capture.records["product_admission"] == "not-attempted"
    assert [
        np.asarray(model.weight).tobytes() for model in (original, alternate)
    ] == originals


@pytest.mark.parametrize("api_name", ["tanh", "relu", "add", "sum", "sin"])
def test_imported_executable_APIs_remain_usable_without_model_exemption(
    tmp_path, monkeypatch, api_name
):
    import math
    import jax

    api = {
        "tanh": jax.numpy.tanh,
        "relu": jax.nn.relu,
        "add": np.add,
        "sum": np.sum,
        "sin": math.sin,
    }[api_name]
    binding = raw._imported_executable_binding(api)
    assert binding is not None
    monkeypatch.setitem(
        _call_original_global_dependency.__globals__,
        "_RAW_REPLAY_GLOBAL_DEPENDENCY",
        api,
    )
    capture = raw.RawReplayCapture(tmp_path / ("imported-api-" + api_name))
    capture.bind_target(_call_original_global_dependency)
    assert (
        capture.target_state["referenced_globals"]["_RAW_REPLAY_GLOBAL_DEPENDENCY"]
        == binding
    )
    assert capture.check_target("stable-imported-api")
    assert capture.records["native_outcomes"] == {}
    assert capture.records["product_admission"] == "not-attempted"


def test_imported_model_class_is_an_explicit_static_state_gap(
    original_scope, tmp_path, monkeypatch
):
    from tests.extra_tests.converter.source_precision_support import (
        SourcePreparationError,
    )

    _, metadata, _ = _original_metadata(original_scope, "plugin::eqx.nn.linear", 4)
    model_class = type(metadata["callable"])
    assert raw._library_namespace(model_class.__module__)
    assert raw._imported_executable_binding(model_class) is None
    monkeypatch.setitem(
        _call_original_global_dependency.__globals__,
        "_RAW_REPLAY_GLOBAL_DEPENDENCY",
        model_class,
    )
    capture = raw.RawReplayCapture(tmp_path / "owned-static-class")
    with pytest.raises(SourcePreparationError) as error:
        capture.bind_target(_call_original_global_dependency)
    assert error.value.category == "replay-lane-gap"
    assert capture.readiness is CheckStatus.UNVERIFIED and capture.gaps
    assert capture.records["native_outcomes"] == {}


def test_owned_native_array_method_stays_an_explicit_owner_adapter_gap(
    original_scope, tmp_path, monkeypatch
):
    from tests.extra_tests.converter.source_precision_support import (
        SourcePreparationError,
    )

    _, metadata, _ = _original_metadata(
        original_scope, "example::examples.rl::ContinuousTanhPolicy", 0
    )
    original = metadata["callable"].__globals__["_CONTINUOUS_W1"]
    value = original.sum
    original_bytes = original.tobytes()
    assert value.__self__ is original
    assert raw._imported_executable_binding(value) is None
    monkeypatch.setitem(
        _call_original_global_dependency.__globals__,
        "_RAW_REPLAY_GLOBAL_DEPENDENCY",
        value,
    )
    capture = raw.RawReplayCapture(tmp_path / "owned-native-method")
    with pytest.raises(SourcePreparationError) as error:
        capture.bind_target(_call_original_global_dependency)
    assert error.value.category == "replay-lane-gap"
    assert capture.readiness is CheckStatus.UNVERIFIED and capture.gaps
    assert capture.records["native_outcomes"] == {}
    assert original.tobytes() == original_bytes


@pytest.mark.parametrize("name", ["bool_", "int32", "int64", "float32", "bfloat16"])
def test_loaded_jax_scalar_dtype_alias_retains_alias_and_canonical_dtype(name):
    import jax.numpy as jnp

    alias = getattr(jnp, name)
    binding = raw._jax_scalar_dtype_binding(alias)
    canonical = np.dtype(alias)
    assert binding == {
        "jax_scalar_alias": "jax.numpy." + name,
        "metaclass": f"{type(alias).__module__}.{type(alias).__qualname__}",
        "dtype": canonical.name,
        "dtype_str": canonical.str,
        "numpy_scalar_type": f"{canonical.type.__module__}.{canonical.type.__qualname__}",
    }

    def no_fallback(value):
        pytest.fail("Exact JAX alias must not collapse to the generic dtype adapter")

    assert raw._function_owned_data(alias, no_fallback) == binding


@pytest.mark.parametrize("replacement", ["jax-int64", "numpy-int32"])
def test_original_discrete_policy_scalar_alias_rebinding_is_detected(
    original_scope, tmp_path, monkeypatch, replacement
):
    import jax.numpy as jnp

    _, metadata, _ = _original_metadata(
        original_scope, "example::examples.rl::DiscreteArgmaxPolicy", 0
    )
    target = metadata["callable"]
    originals = {
        name: target.__globals__[name].tobytes()
        for name in ("_DISCRETE_W1", "_DISCRETE_B1", "_DISCRETE_W2", "_DISCRETE_B2")
    }
    capture = raw.RawReplayCapture(tmp_path / replacement)
    capture.bind_target(target)
    binding = capture.target_state["referenced_globals"]["jnp"]["attributes"]["int32"]
    assert binding["jax_scalar_alias"] == "jax.numpy.int32"
    assert binding["dtype"] == "int32"
    monkeypatch.setattr(
        jnp, "int32", jnp.int64 if replacement == "jax-int64" else np.int32
    )
    assert not capture.check_target("dtype-alias-rebound")
    assert capture.target is target and capture.readiness is CheckStatus.UNVERIFIED
    assert capture.records["native_outcomes"] == {}
    assert capture.records["dtype-alias-rebound-state"]["status"] == "failed"
    assert {name: target.__globals__[name].tobytes() for name in originals} == originals


@pytest.mark.parametrize(
    "fault", ["custom-class", "same-metaclass", "public-spoof", "owned-state", "dtype"]
)
def test_jax_scalar_alias_boundary_rejects_spoofed_or_owned_class_state(
    monkeypatch, fault
):
    import jax.numpy as jnp

    from tests.extra_tests.converter.source_precision_support import (
        SourcePreparationError,
    )

    original = jnp.int32
    if fault == "custom-class":
        value = type(
            "int32", (), {"__module__": "jax.numpy", "dtype": np.dtype("int32")}
        )
    elif fault in {"same-metaclass", "public-spoof"}:
        value = type(original)("int32", (object,), {"dtype": np.dtype("int32")})
        value.__module__ = "jax.numpy"
        if fault == "public-spoof":
            monkeypatch.setattr(jnp, "int32", value)
    else:
        value = original
        if fault == "owned-state":
            monkeypatch.setattr(
                value, "unowned_weight", np.asarray([1], np.float32), raising=False
            )
        else:
            monkeypatch.setattr(value, "dtype", np.dtype("int64"))
    assert raw._jax_scalar_dtype_binding(value) is None
    with pytest.raises(SourcePreparationError, match="class/static state") as error:
        raw._function_owned_data(value, lambda item: {"unexpected": str(item)})
    assert error.value.category == "replay-lane-gap"


@pytest.mark.parametrize(
    "owner,expected_path",
    [
        ("defaults", ["root", "defaults", "item[0]"]),
        ("kwdefaults", ["root", "kwdefaults", "mapping[0].value"]),
        ("attributes", ["root", "attributes", "mapping[0].value"]),
        (
            "closure",
            ["root", "capture:payload", "mapping[0].value", "item[0]"],
        ),
        (
            "partial",
            ["root", "partial-args", "item[0]", "defaults", "item[0]"],
        ),
        ("dataclass", ["root", "field:callback", "defaults", "item[0]"]),
    ],
)
def test_class_gap_diagnostic_retains_supported_owner_paths(
    tmp_path, owner, expected_path
):
    from dataclasses import dataclass
    import functools

    from tests.extra_tests.converter.source_precision_support import (
        SourcePreparationError,
    )

    class Unknown:
        def __repr__(self):
            pytest.fail("Class-gap capture must not format an unknown instance")

    def target(value=None):
        pytest.fail("Class-gap capture must not invoke the source")

    if owner == "defaults":
        target.__defaults__ = (Unknown,)
    elif owner == "kwdefaults":

        def target(*, value=None):
            pytest.fail("Class-gap capture must not invoke the source")

        target.__kwdefaults__ = {"value": Unknown}
    elif owner == "attributes":
        target.state = Unknown
    elif owner == "closure":
        payload = {"state": [Unknown]}

        def target(value=None):
            return payload[value]

    elif owner == "partial":
        target.__defaults__ = (Unknown,)
        target = functools.partial(lambda callback: callback(), target)
    else:
        target.__defaults__ = (Unknown,)

        @dataclass
        class DataclassOwner:
            callback: Any

        DataclassOwner.__module__ = "jax2onnx.plugins.examples.eqx.diagnostic_fixture"
        target = DataclassOwner(target)
    capture = raw.RawReplayCapture(tmp_path / owner)
    with pytest.raises(SourcePreparationError, match="exact state capture adapter"):
        capture.bind_target(target)
    failure = capture.records["instantiated-target-state"]
    assert failure["detail"] == (
        "Referenced function class/static state needs an exact adapter"
    )
    assert failure["diagnostic"]["path"] == expected_path
    assert failure["diagnostic"]["class"]["qualname"] == Unknown.__qualname__
    assert capture.records["native_outcomes"] == {}
    assert capture.records["product_admission"] == "not-attempted"
    assert capture.records["raw_callback"] == {"status": "not-executed"}


def test_class_gap_diagnostic_retains_bounded_module_binding(tmp_path, monkeypatch):
    import sys
    import types

    from tests.extra_tests.converter.source_precision_support import (
        SourcePreparationError,
    )

    module = types.ModuleType("jax.raw_replay_diagnostic_fixture")
    branch = types.ModuleType(module.__name__ + ".branch")
    module.branch = branch
    unknown = type("Dependency", (), {"__module__": branch.__name__})
    branch.Dependency = unknown
    monkeypatch.setitem(sys.modules, branch.__name__, branch)
    monkeypatch.setitem(globals(), "_RAW_REPLAY_DIAGNOSTIC_MODULE", module)

    def target(value):
        return _RAW_REPLAY_DIAGNOSTIC_MODULE.branch.Dependency(value)  # noqa: F821

    capture = raw.RawReplayCapture(tmp_path / "module-binding")
    with pytest.raises(SourcePreparationError, match="exact state capture adapter"):
        capture.bind_target(target)
    diagnostic = capture.records["instantiated-target-state"]["diagnostic"]
    assert diagnostic["path"] == [
        "root",
        "global:_RAW_REPLAY_DIAGNOSTIC_MODULE",
        "attribute:branch",
        "attribute:Dependency",
    ]
    assert diagnostic["class"] == {
        "module": branch.__name__,
        "qualname": "Dependency",
    }
    assert diagnostic["defining_module_binding"] == "exact"
    assert capture.records["native_outcomes"] == {}


@pytest.mark.parametrize("binding", ["exact", "different", "missing"])
def test_class_diagnostic_binding_is_observation_only(monkeypatch, binding):
    import sys
    import types

    module = types.ModuleType("raw_replay_diagnostic_binding_fixture")
    unknown = type("Dependency", (), {"__module__": module.__name__})
    if binding != "missing":
        module.Dependency = unknown if binding == "exact" else object()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    diagnostic = raw._class_state_diagnostic(
        unknown, location=("root", "global:dependency"), function=None
    )
    assert diagnostic["defining_module_binding"] == binding
    assert diagnostic["callable"] is None


def test_class_diagnostic_bypasses_hostile_metaclass_and_module_descriptors(
    monkeypatch,
):
    import sys
    import types

    calls = []

    class Descriptor:
        def __get__(self, instance, owner=None):
            calls.append("descriptor")
            raise AssertionError("diagnostics resolved a hostile descriptor")

        def __set__(self, instance, value):
            calls.append("descriptor-set")
            raise AssertionError("diagnostics changed hostile state")

    class Meta(type):
        __dict__ = Descriptor()
        __module__ = Descriptor()

        def __getattribute__(self, name):
            calls.append("metaclass-getattribute")
            raise AssertionError("diagnostics resolved metaclass attributes")

        def __repr__(self):
            calls.append("repr")
            raise AssertionError("diagnostics formatted an unknown class")

    class HostileModule(types.ModuleType):
        __dict__ = Descriptor()

        def __getattribute__(self, name):
            calls.append("module-getattribute")
            raise AssertionError("diagnostics resolved module attributes")

    module_name = "raw_replay_diagnostic_hostile_fixture"
    module = HostileModule(module_name)
    unknown = Meta("Dependency", (), {"__module__": module_name})
    namespace = vars(types.ModuleType)["__dict__"].__get__(module)
    namespace["Dependency"] = unknown
    monkeypatch.setitem(sys.modules, module_name, module)
    diagnostic = raw._class_state_diagnostic(
        unknown, location=("root", "global:dependency"), function=None
    )
    assert diagnostic["class"] == {"module": module_name, "qualname": "Dependency"}
    assert diagnostic["metaclass"] == {
        "module": None,
        "qualname": Meta.__qualname__,
    }
    assert diagnostic["defining_module_binding"] == "exact"
    assert calls == []


def test_class_gap_diagnostic_rejects_hostile_nonmodule_without_property_calls(
    tmp_path, monkeypatch
):
    import sys

    from tests.extra_tests.converter.source_precision_support import (
        SourcePreparationError,
    )

    calls = []

    class NotAModule:
        @property
        def __class__(self):
            calls.append("__class__")
            raise RuntimeError("diagnostic must not read this property")

        def __repr__(self):
            calls.append("repr")
            raise RuntimeError("diagnostic must not format this object")

    name = "raw_replay_bad_module_fixture"
    unknown = type("Dependency", (), {"__module__": name})
    monkeypatch.setitem(sys.modules, name, NotAModule())

    def target(value=unknown):
        raise AssertionError("Class-gap capture must not invoke the source")

    capture = raw.RawReplayCapture(tmp_path / "nonmodule")
    with pytest.raises(
        SourcePreparationError, match="exact state capture adapter"
    ) as error:
        capture.bind_target(target)
    cause = error.value.__cause__
    assert type(cause) is SourcePreparationError
    assert cause.category == error.value.category == "replay-lane-gap"
    failure = capture.records["instantiated-target-state"]
    assert failure["exception"].endswith(".SourcePreparationError")
    assert (
        failure["detail"]
        == "Referenced function class/static state needs an exact adapter"
    )
    assert (
        failure["diagnostic"]["defining_module_binding"]
        == "module-dictionary-unavailable"
    )
    assert calls == []
    assert capture.records["native_outcomes"] == {}
    assert capture.readiness is CheckStatus.UNVERIFIED
    assert capture.records["product_admission"] == "not-attempted"
    assert capture.records["facade_return"] == {"status": "not-attempted"}
    assert capture.records["raw_callback"] == {"status": "not-executed"}
    assert all(
        capture.records[role] == {"status": "not-attempted"}
        for role in (
            "eager",
            "fresh-jit",
            "concrete-typed-trace",
            "declared-typed-trace",
        )
    )


def test_class_gap_diagnostic_failure_preserves_original_rejection(
    tmp_path, monkeypatch
):
    from tests.extra_tests.converter.source_precision_support import (
        SourcePreparationError,
    )

    def unavailable(*args, **kwargs):
        raise RuntimeError("injected unavailable diagnostic fact")

    class Unknown:
        pass

    def target(value=Unknown):
        raise AssertionError("Class-gap capture must not invoke the source")

    monkeypatch.setattr(raw, "_class_state_diagnostic", unavailable)
    capture = raw.RawReplayCapture(tmp_path / "diagnostic-unavailable")
    with pytest.raises(
        SourcePreparationError, match="exact state capture adapter"
    ) as error:
        capture.bind_target(target)
    assert type(error.value.__cause__) is SourcePreparationError
    assert error.value.__cause__.category == "replay-lane-gap"
    failure = capture.records["instantiated-target-state"]
    assert (
        failure["detail"]
        == "Referenced function class/static state needs an exact adapter"
    )
    assert failure["diagnostic"] == {
        "kind": "unsupported-function-class-state",
        "unavailable": "diagnostic-facts-unavailable",
    }
    assert capture.records["native_outcomes"] == {}
    assert capture.records["product_admission"] == "not-attempted"


def test_class_diagnostic_exact_oversize_counterexample_is_bounded():
    unknown = type("D" * 1_000_000, (), {"__module__": "m" * 1_000_000})
    location = ("root", "global:" + "g" * 1_000_000)
    diagnostic = raw._class_state_diagnostic(unknown, location=location, function=None)
    repeated = raw._class_state_diagnostic(unknown, location=location, function=None)
    payload = json.dumps(diagnostic, ensure_ascii=True, separators=(",", ":"))
    assert len(payload.encode("ascii")) <= 32768
    assert diagnostic == repeated
    for label in (
        diagnostic["class"]["module"],
        diagnostic["class"]["qualname"],
        diagnostic["path"][1],
    ):
        assert label["truncated"] is True
        assert label["characters"] >= 1_000_000
        assert len(label["prefix"]) + len(label["suffix"]) <= 96
        assert len(label["sha256_utf8_surrogatepass"]) == 64


@pytest.mark.parametrize("suffix", ["end", "\ud800"])
def test_class_diagnostic_bounds_all_labels_and_preserves_original_loaded_binding(
    monkeypatch, suffix
):
    import sys
    import types

    text = "\U0001f680" * 4096 + suffix
    module_name = "module" + text
    qualname = "Dependency" + text
    module = types.ModuleType(module_name)
    meta = type("DiagnosticMeta", (type,), {"__module__": "meta-module" + text})
    type.__setattr__(meta, "__qualname__", "Meta" + text)
    unknown = meta("Dependency", (), {"__module__": module_name})
    type.__setattr__(unknown, "__qualname__", qualname)
    namespace = vars(types.ModuleType)["__dict__"].__get__(module)
    namespace[qualname] = unknown
    monkeypatch.setitem(sys.modules, module_name, module)

    def target(value):
        raise AssertionError("Diagnostics must not invoke the source")

    target.__module__ = "callable-module" + text
    target.__qualname__ = "Callable" + text
    location = ("root",) + tuple("entry:" + str(index) + text for index in range(100))
    diagnostic = raw._class_state_diagnostic(
        unknown, location=location, function=target
    )
    repeated = raw._class_state_diagnostic(unknown, location=location, function=target)
    payload = json.dumps(diagnostic, ensure_ascii=True, separators=(",", ":"))
    assert payload == json.dumps(repeated, ensure_ascii=True, separators=(",", ":"))
    assert len(payload.encode("ascii")) <= 32768
    assert diagnostic["defining_module_binding"] == "exact"
    assert len(diagnostic["path"]) == 12
    assert diagnostic["path_truncation"] == {
        "total_segments": 101,
        "retained_prefix": 4,
        "retained_suffix": 8,
    }
    for group in ("class", "metaclass", "callable"):
        for field in ("module", "qualname"):
            label = diagnostic[group][field]
            assert label["truncated"] is True
            assert label["characters"] > 4096
            assert len(label["prefix"]) + len(label["suffix"]) == 96
    for label in diagnostic["path"][1:]:
        assert label["truncated"] is True
        assert len(label["prefix"]) + len(label["suffix"]) == 64
    original_module = diagnostic["class"]["module"]
    assert (
        original_module["sha256_utf8_surrogatepass"]
        == hashlib.sha256(module_name.encode("utf-8", "surrogatepass")).hexdigest()
    )
    assert diagnostic["callable"]["code_sha256"] == raw.canonical_code_fingerprint(
        target.__code__
    )
