# tests/extra_tests/converter/test_source_precision_raw_replay.py

"""General replay tests retain original sources, options and separate failures."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest

from jax2onnx import user_interface
from jax2onnx.precision import CheckStatus
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
        metadata, scenario, adapter=raw.replay_adapter(row, adapters), capture=capture
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
    assert len(observed) == 2
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
        metadata, scenario, adapter=raw.replay_adapter(row, adapters), capture=capture
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
