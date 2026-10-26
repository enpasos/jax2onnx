# tests/extra_tests/converter/test_source_native_wrapper_capture.py

"""Finite original JAX transformed-callable state; native qualification by ROOT."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
import builtins
import copy
import functools
import gc
import hashlib
import json
from pathlib import Path
import sys
import types
from typing import Any
import typing
import weakref

import jax
import jax.numpy as jnp
import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest

from jax2onnx._compat import jax as compat
from jax2onnx.converter.precision_policy import active_precision_policy
from jax2onnx.precision import CheckStatus
from scripts import source_precision_probe as probe
from scripts.source_precision_inventory import raw_registrations
from tests.extra_tests.converter import source_precision_raw_replay as raw
from tests.extra_tests.converter import source_precision_support as source
from tests.extra_tests.converter.source_precision_support import SourceProvenance


@pytest.fixture(autouse=True)
def _restore_caller(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    context = compat.native_context_snapshot()
    axis_owner = compat._NATIVE_WRAPPER_CONTEXT_ROLES["axis_config"]
    read_axis_local, set_axis_local = axis_owner.get_local, axis_owner.set_local
    axis_value_descriptor = vars(type(axis_owner))["value"]
    axis_local, axis_effective = read_axis_local(), axis_owner.value
    policy, x64 = active_precision_policy(), jax.config.x64_enabled
    states = tuple(
        (owner, owner.get_local(), owner.value)
        for owner in compat._NATIVE_WRAPPER_CONFIGS.values()
    )
    try:
        yield
    finally:
        try:
            monkeypatch.undo()
            compat.require_native_context_restored(context)
            assert active_precision_policy() is policy
            assert jax.config.x64_enabled is x64
            for owner, local, effective in states:
                assert owner.get_local() is local and owner.value is effective
        finally:
            # Ordinary test setup can materialize axis local outside the selected
            # vmap execution contract. Isolate it even when another check fails.
            set_axis_local(axis_local)
            assert compat._native_core.axis_env_state is axis_owner
            assert read_axis_local() is axis_local
            assert (
                axis_value_descriptor.__get__(axis_owner, type(axis_owner))
                is axis_effective
            )


@contextmanager
def _changed_class_symbol(
    cls: type[Any], field: str, replacement: str
) -> Iterator[None]:
    descriptor = vars(type)[field]
    original = descriptor.__get__(cls, type(cls))
    try:
        descriptor.__set__(cls, replacement)
        yield
    finally:
        descriptor.__set__(cls, original)
        assert descriptor.__get__(cls, type(cls)) is original


def _closure(function: Any, name: str) -> Any:
    return dict(
        zip(function.__code__.co_freevars, function.__closure__ or (), strict=True)
    )[name]


def _source(value: Any, scale: Any = 2) -> Any:
    return value * scale + value


def _mapped() -> Any:
    return jax.vmap(_source)


def _capture(
    target: Any, tmp_path: Path, name: str = "capture"
) -> raw.RawReplayCapture:
    capture = raw.RawReplayCapture(tmp_path / name)
    capture.bind_target(target)
    return capture


def test_original_vmap_distinct_edges_eager_fresh_and_warmed_jit(
    tmp_path: Path,
) -> None:
    mapped = _mapped()
    vmap_f = _closure(mapped, "fun").cell_contents
    assert (
        compat.native_transformed_function_role(mapped)
        == "reraise_with_filtered_traceback"
    )
    assert compat.native_transformed_function_role(vmap_f) == "vmap_f"
    assert mapped.__wrapped__ is _source
    assert vmap_f is not mapped.__wrapped__
    assert _closure(vmap_f, "fun").cell_contents is _source
    assert (
        tuple(vmap_f.__code__.co_freevars) == compat._NATIVE_WRAPPER_FREEVARS["vmap_f"]
    )
    capture = _capture(mapped, tmp_path)
    assert capture.target is mapped and capture.target_witness is not None
    values = np.arange(8, dtype=np.float32)
    expected = values * np.float32(3)
    axis_owner = compat._NATIVE_WRAPPER_CONTEXT_ROLES["axis_config"]
    axis_local, axis_effective = axis_owner.get_local(), axis_owner.value
    eager = capture.invoke("native-eager", lambda: mapped(values))
    assert axis_owner.get_local() is axis_local and axis_owner.value is axis_effective
    assert capture.check_target("eager")
    compiled = jax.jit(mapped)
    fresh = capture.invoke("native-fresh-jit", lambda: compiled(values))
    assert axis_owner.get_local() is axis_local and axis_owner.value is axis_effective
    assert capture.check_target("fresh-jit")
    warmed = capture.invoke("native-warmed-jit", lambda: compiled(values))
    assert axis_owner.get_local() is axis_local and axis_owner.value is axis_effective
    assert capture.check_target("warmed-jit")
    for result in (eager, fresh, warmed):
        np.testing.assert_array_equal(result, expected)
    capture.save_arrays("native-eager-fresh-warmed", (eager, fresh, warmed))


@pytest.mark.parametrize("starting_local", ("unset", "effective"))
@pytest.mark.parametrize("outcome", ("returned", "raised"))
def test_native_vmap_execution_restores_exact_axis_caller_context(
    starting_local: str, outcome: str, tmp_path: Path
) -> None:
    owner = compat._NATIVE_WRAPPER_CONTEXT_ROLES["axis_config"]
    caller_local, caller_effective = owner.get_local(), owner.value
    local = (
        compat._native_xla_metadata.config_ext.unset
        if starting_local == "unset"
        else caller_effective
    )
    try:
        owner.set_local(local)
        mapped = _mapped()
        capture = _capture(mapped, tmp_path)
        values = (
            np.arange(8, dtype=np.float32) if outcome == "returned" else np.float32(1)
        )
        result = capture.invoke("exact-axis-context", lambda: mapped(values))
        assert owner.get_local() is local and owner.value is caller_effective
        assert (
            capture.records["native_outcomes"]["exact-axis-context"]["status"]
            == outcome
        )
        if outcome == "returned":
            np.testing.assert_array_equal(result, values * np.float32(3))
        else:
            assert result is None
            assert type(capture.native_errors["exact-axis-context"]) is ValueError
        assert capture.check_target("after-exact-axis-context")
        assert capture.gaps == []
    finally:
        owner.set_local(caller_local)
    assert owner.get_local() is caller_local and owner.value is caller_effective


@pytest.mark.parametrize("fault", ("unset", "different-effective", "axis-backing"))
@pytest.mark.parametrize("outcome", ("returned", "raised"))
def test_native_axis_context_drift_stays_a_gap_without_erasing_actual_outcome(
    fault: str, outcome: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = compat._NATIVE_WRAPPER_CONTEXT_ROLES
    owner = context["axis_config"]
    local, effective = owner.get_local(), owner.value
    capture = _capture(_mapped(), tmp_path)
    returned = object()
    original_error = ValueError("actual native operation error")
    axis_name = "precision_replay_probe_axis"
    different_effective = None
    if fault == "different-effective":
        assert axis_name not in effective.axis_sizes
        different_effective = effective.extend_pure(((axis_name, 2),))
        assert type(different_effective) is context["axis_type"]
        assert different_effective is not effective
        assert different_effective.axis_sizes is not effective.axis_sizes
        assert different_effective.axis_sizes[axis_name] == 2
        assert axis_name not in effective.axis_sizes

    def operation() -> Any:
        if fault == "unset":
            owner.set_local(compat._native_xla_metadata.config_ext.unset)
        elif fault == "different-effective":
            assert different_effective is not None
            owner.set_local(different_effective)
            assert owner.get_local() is different_effective
            assert owner.value is different_effective and owner.value is not effective
            assert owner.value.axis_sizes[axis_name] == 2
            assert axis_name not in effective.axis_sizes
        else:
            mapping = context["axis_slots"][0][1].__get__(
                effective, context["axis_type"]
            )
            _, descriptor = context["axis_mapping_dict"]
            backing = descriptor.__get__(mapping, type(mapping))["_d"]
            monkeypatch.setitem(backing, "changed", 2)
        if outcome == "raised":
            raise original_error
        return returned

    result = capture.invoke("actual-operation", operation)
    assert owner.get_local() is local and owner.value is effective
    assert capture.records["native_outcomes"]["actual-operation"]["status"] == outcome
    if outcome == "returned":
        assert (
            result is returned and capture.native_values["actual-operation"] is returned
        )
    else:
        assert (
            result is None
            and capture.native_errors["actual-operation"] is original_error
        )
    assert any(gap["role"] == "actual-operation-native-context" for gap in capture.gaps)
    monkeypatch.undo()
    assert capture.check_target("after-restoring-context-drift")
    assert capture.readiness is CheckStatus.UNVERIFIED
    assert capture.gaps


@pytest.mark.parametrize("callback_outcome", ("passed", "failed", "error"))
def test_axis_context_gaps_preserve_export_and_raw_callback_observations(
    callback_outcome: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = compat._NATIVE_WRAPPER_CONTEXT_ROLES["axis_config"]
    local, effective = owner.get_local(), owner.value
    mapped = _mapped()
    capture = _capture(mapped, tmp_path)
    model = raw.onnx.helper.make_model(
        raw.onnx.helper.make_graph([], "boundary-fixture", [], [])
    )
    callback_error = ValueError("actual raw callback error")

    def export(prepared: Any, **kwargs: Any) -> Any:
        kwargs["facade_return_observer"](model)
        owner.set_local(compat._native_xla_metadata.config_ext.unset)
        return model

    def callback(actual: Any) -> bool:
        assert actual is model
        owner.set_local(compat._native_xla_metadata.config_ext.unset)
        if callback_outcome == "error":
            raise callback_error
        return callback_outcome == "passed"

    monkeypatch.setattr(raw, "_export_prepared", export)
    monkeypatch.setattr(raw, "source_configuration", lambda scenario: nullcontext())
    prepared = types.SimpleNamespace(
        fn=mapped, scenario=None, metadata={"post_check_onnx_graph": callback}
    )
    result = raw.run_raw_source_case(prepared, capture=capture)
    assert result.model is model and result.schema is CheckStatus.VALIDATED
    assert capture.records["facade_return"]["status"] == "returned"
    assert result.raw_callback_status == callback_outcome
    assert capture.records["raw_callback"]["status"] == callback_outcome
    assert owner.get_local() is local and owner.value is effective
    roles = {gap["role"] for gap in capture.gaps}
    assert {"product-export-native-context", "raw-callback-native-context"} <= roles
    assert capture.check_target("after-export-and-callback-context-gaps")
    assert capture.readiness is CheckStatus.UNVERIFIED


def test_capture_does_not_invoke_source_factory_or_resolve_auto(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_source(value: Any) -> Any:
        raise AssertionError("capture invoked original source")

    mapped = jax.vmap(forbidden_source)

    def forbidden_factory(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("capture invoked transformation factory")

    monkeypatch.setattr(jax, "vmap", forbidden_factory)
    calls = []
    filtering_code = compat._native_traceback._filtering_mode.__code__

    def observe(frame: Any, event: str, argument: Any) -> None:
        if event == "call" and frame.f_code is filtering_code:
            calls.append(event)

    previous = sys.getprofile()
    with compat._native_config.traceback_filtering("auto"):
        try:
            sys.setprofile(observe)
            capture = _capture(mapped, tmp_path)
            assert capture.check_target("no-execution")
        finally:
            sys.setprofile(previous)
    assert calls == []


def test_module_native_member_views_retain_original_owner_and_method(
    tmp_path: Path,
) -> None:
    def source(value: Any) -> Any:
        return jax.lax.logistic_p.bind(value)

    target = jax.vmap(source)
    capture = _capture(target, tmp_path)
    assert capture.check_target("same-native-owner-first")
    assert capture.check_target("same-native-owner-second")


@pytest.mark.parametrize(
    "field",
    (
        "__name__",
        "__module__",
        "__qualname__",
        "__doc__",
        "__annotations__",
        "__defaults__",
        "__kwdefaults__",
    ),
)
def test_selected_metadata_drift_is_observed(
    field: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mapped = _mapped()
    capture = _capture(mapped, tmp_path)
    replacements = {
        "__annotations__": {"state": "changed"},
        "__defaults__": (1,),
        "__kwdefaults__": {"state": 1},
    }
    monkeypatch.setattr(mapped, field, replacements.get(field, "changed"))
    assert not capture.check_target("changed-metadata")
    assert capture.readiness is CheckStatus.UNVERIFIED


@pytest.mark.parametrize("edge", ("wrapped", "vmap-fun", "model-fun", "closure-tuple"))
def test_equal_state_function_alias_replacement_is_rejected_and_old_owner_retained(
    edge: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mapped = _mapped()
    vmap_f = _closure(mapped, "fun").cell_contents
    capture = _capture(mapped, tmp_path)
    original = vmap_f if edge == "vmap-fun" else _source
    clone = types.FunctionType(
        original.__code__,
        original.__globals__,
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    clone.__dict__.update(original.__dict__)
    clone.__kwdefaults__, clone.__annotations__ = original.__kwdefaults__, dict(
        original.__annotations__
    )
    clone.__qualname__, clone.__module__, clone.__doc__ = (
        original.__qualname__,
        original.__module__,
        original.__doc__,
    )
    if edge == "wrapped":
        monkeypatch.setattr(mapped, "__wrapped__", clone)
    elif edge == "closure-tuple":
        cells = tuple(
            types.CellType(cell.cell_contents) for cell in mapped.__closure__ or ()
        )
        replacement = types.FunctionType(
            mapped.__code__,
            mapped.__globals__,
            mapped.__name__,
            mapped.__defaults__,
            cells,
        )
        replacement.__dict__.update(mapped.__dict__)
        capture.target = replacement
    else:
        owner = mapped if edge == "vmap-fun" else vmap_f
        cell = _closure(owner, "fun")
        monkeypatch.setattr(cell, "cell_contents", clone)
    reference = weakref.ref(original)
    del original
    gc.collect()
    assert reference() is not None
    assert not capture.check_target("equal-value-alias-split")


@pytest.mark.parametrize("field", ("weight", "key", "alias"))
def test_original_parameter_key_and_shared_alias_edges_are_strongly_retained(
    field: str, tmp_path: Path
) -> None:
    def source(value: Any) -> Any:
        return value

    weight = jnp.arange(4, dtype=jnp.float32)
    source.state = {"weight": weight, "alias": weight, "key": jax.random.key(31)}
    mapped = jax.vmap(source)
    capture = _capture(mapped, tmp_path)
    original = source.state[field]
    reference = weakref.ref(original)
    replacement = (
        jax.random.wrap_key_data(jax.random.key_data(original))
        if field == "key"
        else jnp.array(original, copy=True)
    )
    source.state[field] = replacement
    if field == "weight":
        source.state["alias"] = replacement
    del weight, original
    gc.collect()
    assert reference() is not None
    assert not capture.check_target("equal-array-or-key-replacement")


@pytest.mark.parametrize("name", compat._NATIVE_WRAPPER_FREEVARS["vmap_f"])
def test_each_original_axis_or_fun_closure_is_still_captured(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mapped = _mapped()
    capture = _capture(mapped, tmp_path)
    vmap_f = _closure(mapped, "fun").cell_contents
    cell = _closure(vmap_f, name)
    replacement = _source if name == "fun" else 13
    if name == "fun":
        replacement = types.FunctionType(
            _source.__code__, _source.__globals__, "replacement"
        )
    monkeypatch.setattr(cell, "cell_contents", replacement)
    assert not capture.check_target("changed-closure")


@pytest.mark.parametrize(
    "route", ("foreign-code", "foreign-globals", "copied-metadata")
)
def test_copied_code_globals_and_metadata_do_not_select(route: str) -> None:
    original = _mapped()
    if route == "foreign-code":
        candidate = types.FunctionType(
            original.__code__.replace(),
            original.__globals__,
            closure=original.__closure__,
        )
    elif route == "foreign-globals":
        candidate = types.FunctionType(
            original.__code__, dict(original.__globals__), closure=original.__closure__
        )
    else:
        candidate = types.FunctionType(_source.__code__, _source.__globals__)
        candidate.__dict__.update(original.__dict__)
    candidate.__name__, candidate.__qualname__, candidate.__module__ = (
        original.__name__,
        original.__qualname__,
        original.__module__,
    )
    assert compat.native_transformed_function_role(candidate) is None
    assert compat.selected_native_transformed_function(candidate) is None


def test_no_axis_sentinel_is_only_contextual_and_current_defining_binding_is_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mapped = _mapped()
    vmap_f = _closure(mapped, "fun").cell_contents
    sentinel = _closure(vmap_f, "axis_name").cell_contents
    assert sentinel is compat._NATIVE_WRAPPER_SENTINEL
    assert compat.native_transformed_closure_state(vmap_f, "axis_name", sentinel)[
        "identity"
    ] == id(sentinel)
    assert (
        compat.native_transformed_closure_state(mapped, "axis_name", sentinel) is None
    )
    with pytest.raises(raw.SourcePreparationError):
        raw._capture_target_state(sentinel)
    capture = _capture(mapped, tmp_path)
    monkeypatch.setattr(compat._native_core, "no_axis_name", object())
    assert _closure(vmap_f, "axis_name").cell_contents is sentinel
    assert not capture.check_target("rebound-old-sentinel-alias")


@pytest.mark.parametrize("fault", ("code", "binding", "doc", "attribute"))
def test_selected_traceback_helper_executable_mutation_is_detected(
    fault: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = compat._native_traceback.api_boundary(_source)
    capture = _capture(target, tmp_path)
    owner = compat._native_traceback._filtering_mode
    if fault == "code":
        monkeypatch.setattr(
            owner, "__code__", owner.__code__.replace(co_filename="changed.py")
        )
    elif fault == "binding":
        monkeypatch.setattr(compat._native_traceback, "_filtering_mode", _source)
    elif fault == "doc":
        monkeypatch.setattr(owner, "__doc__", "changed")
    else:
        monkeypatch.setitem(owner.__dict__, "changed", True)
    assert not capture.check_target("changed-native-helper")


@pytest.mark.parametrize(
    "fault", ("member", "doc", "binding", "exclusion", "message", "mode")
)
def test_exception_dependency_and_error_configuration_mutation_is_detected(
    fault: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = compat._native_traceback.api_boundary(_source)
    capture = _capture(target, tmp_path)
    cls = compat._native_traceback.UnfilteredStackTrace
    if fault == "member":
        monkeypatch.setattr(cls, "unexpected", True, raising=False)
    elif fault == "doc":
        monkeypatch.setattr(cls, "__doc__", "changed")
    elif fault == "binding":
        monkeypatch.setattr(
            compat._native_traceback,
            "UnfilteredStackTrace",
            type("UnfilteredStackTrace", (Exception,), {}),
        )
    elif fault == "exclusion":
        original = list(compat._native_traceback._exclude_paths)
        try:
            compat._native_traceback._exclude_paths.append("/changed")
            assert not capture.check_target("changed-exclusions")
        finally:
            compat._native_traceback._exclude_paths[:] = original
        return
    elif fault == "message":
        monkeypatch.setattr(compat._native_traceback, "_simplified_tb_msg", "changed")
    else:
        changed_mode = (
            "off"
            if compat._native_config.traceback_filtering.value != "off"
            else "quiet_remove_frames"
        )
        with compat._native_config.traceback_filtering(changed_mode):
            assert not capture.check_target("changed-error-mode")
        return
    assert not capture.check_target("changed-native-error-dependency")


@pytest.mark.parametrize(
    "mode", ("off", "quiet_remove_frames", "remove_frames", "tracebackhide")
)
def test_original_native_error_object_chain_notes_and_traceback_are_retained(
    mode: str, tmp_path: Path
) -> None:
    def failure(value: Any) -> Any:
        cause = RuntimeError("original cause")
        error = ValueError("original failure")
        error.source_marker = error
        error.add_note("original note")
        raise error from cause

    target = compat._native_traceback.api_boundary(failure)
    with compat._native_config.traceback_filtering(mode):
        capture = _capture(target, tmp_path)
        assert capture.invoke("native-error", lambda: target(1)) is None
        error = capture.native_errors["native-error"]
        assert error is error.source_marker
        assert type(error) is ValueError and str(error) == "original failure"
        assert error.__notes__[0] == "original note"
        assert error.__traceback__ is not None
        assert error.__suppress_context__
        if mode == "remove_frames":
            assert (
                type(error.__cause__) is compat._native_traceback.UnfilteredStackTrace
            )
            assert type(error.__cause__.__cause__) is RuntimeError
            assert str(error.__cause__.__cause__) == "original cause"
            assert error.__context__ is None
        else:
            assert type(error.__cause__) is RuntimeError
        if mode == "quiet_remove_frames":
            assert len(error.__notes__) == 2
        assert capture.records["native_outcomes"]["native-error"]["status"] == "raised"
        assert capture.check_target("after-native-error")


@pytest.mark.parametrize("slot", compat._NATIVE_AXISDATA_FIELD_SLOTS)
def test_actual_axisdata_every_field_slot_is_owned(
    slot: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mapped = _mapped()
    capture = _capture(mapped, tmp_path)
    field = compat._NATIVE_AXISDATA["field_objects"][0]
    monkeypatch.setattr(field, slot, object())
    assert not capture.check_target("changed-axisdata-field-" + slot)


@pytest.mark.parametrize(
    "slot", compat._NATIVE_AXISDATA["carriers"]["_DataclassParams"][1]
)
def test_actual_axisdata_every_params_slot_is_owned(
    slot: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mapped = _mapped()
    capture = _capture(mapped, tmp_path)
    params = compat._NATIVE_AXISDATA["params"]
    monkeypatch.setattr(params, slot, not getattr(params, slot))
    assert not capture.check_target("changed-axisdata-params-" + slot)


@pytest.mark.parametrize(
    "fault",
    (
        "foreign-field",
        "descriptor",
        "metadata",
        "marker",
        "pre-slots-closure",
        "generated-code",
        "class-name",
        "old-export-alias",
    ),
)
def test_actual_axisdata_foreign_storage_alias_and_descriptor_controls(
    fault: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mapped = _mapped()
    capture = _capture(mapped, tmp_path)
    roles = compat._NATIVE_AXISDATA
    exported = roles["classes"]["exported"]
    original_field = roles["field_objects"][0]
    calls = []
    if fault == "foreign-field":
        replacement = copy.copy(original_field)
        assert replacement is not original_field
        for descriptor in roles["carriers"]["Field"][2]:
            assert descriptor.__get__(
                replacement, type(replacement)
            ) is descriptor.__get__(original_field, type(original_field))
        monkeypatch.setitem(roles["fields"], "name", replacement)
    elif fault == "descriptor":

        def hostile(owner: Any) -> Any:
            calls.append("descriptor")
            raise AssertionError("foreign descriptor invoked")

        monkeypatch.setattr(roles["carriers"]["Field"][0], "name", property(hostile))
    elif fault == "metadata":
        monkeypatch.setattr(original_field, "metadata", types.MappingProxyType({}))
    elif fault == "marker":
        monkeypatch.setattr(roles["markers"]["_FIELD"], "name", "changed")
    elif fault == "pre-slots-closure":
        monkeypatch.setattr(
            _closure(exported.__setattr__, "cls"), "cell_contents", exported
        )
    elif fault == "generated-code":
        monkeypatch.setattr(
            exported.__init__, "__code__", exported.__init__.__code__.replace()
        )
    elif fault == "class-name":
        with _changed_class_symbol(exported, "__qualname__", "ChangedAxisData"):
            assert not capture.check_target("changed-native-axisdata")
        assert calls == []
        return
    else:
        monkeypatch.setattr(
            compat._native_batching, "AxisData", type("AxisData", (), {})
        )
        assert roles["classes"]["exported"] is exported
    assert not capture.check_target("changed-native-axisdata")
    assert calls == []


def test_actual_axisdata_namespace_variant_and_replace_function_are_owned(
    tmp_path: Path,
) -> None:
    assert compat._NATIVE_AXISDATA_GAP is None
    roles = compat._NATIVE_AXISDATA
    exported, old = (roles["classes"][name] for name in ("exported", "pre-slots"))
    extra = compat._NATIVE_AXISDATA_EXTRA_MEMBERS
    extended = "__replace__" in vars(exported)
    additions = extra if extended else frozenset()
    assert set(vars(exported)) == compat._NATIVE_AXISDATA_EXPORTED_MEMBERS | additions
    assert set(vars(old)) == compat._NATIVE_AXISDATA_PRESLOTS_MEMBERS | additions
    document = compat._AxisDataCaptureGraph().state()
    assert set(document["state_markers"]) == set(roles["markers"])
    for name, marker in roles["markers"].items():
        assert document["state_markers"][name]["native_axisdata_marker"] == name
        assert document["state_markers"][name]["identity"] == id(marker)
    if extended:
        replacement = vars(compat._native_dataclasses)["_replace"]
        assert type(replacement) is types.FunctionType
        assert roles["helpers"]["_replace"] is replacement
        for role, cls in (("exported", exported), ("pre-slots", old)):
            assert type(vars(cls)["__firstlineno__"]) is int
            assert vars(cls)["__static_attributes__"] == ()
            assert vars(cls)["__replace__"] is replacement
            assert (
                document["classes"][role]["members"]["__replace__"]
                == document["state_helpers"]["_replace"]
            )
        function = document["nodes"][str(id(replacement))]
        assert function["code_identity"] == id(replacement.__code__)
        assert function["code_sha256"] == compat.canonical_code_fingerprint(
            replacement.__code__
        )
        assert {"_FIELD_CLASSVAR", "_FIELD_INITVAR"} <= set(roles["markers"])
    else:
        assert "_replace" not in roles["helpers"]
        assert set(roles["markers"]) == {"MISSING", "_FIELD"}
    capture = _capture(_mapped(), tmp_path)
    assert capture.check_target("original-axisdata-complete-namespace-variant")


@pytest.mark.parametrize("role", ("exported", "pre-slots"))
@pytest.mark.parametrize("fault", ("unknown", "missing", "partial", "mixed"))
def test_actual_axisdata_namespace_variants_reject_partial_mixed_and_unknown(
    role: str, fault: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    cls = compat._NATIVE_AXISDATA["classes"][role]
    extended = "__replace__" in vars(cls)
    if fault == "unknown":
        monkeypatch.setattr(cls, "unobserved_member", True, raising=False)
    elif fault == "missing":
        monkeypatch.delattr(cls, "__match_args__")
    elif fault == "partial":
        if extended:
            monkeypatch.delattr(cls, "__firstlineno__")
        else:
            monkeypatch.setattr(cls, "__firstlineno__", 1, raising=False)
    elif extended:
        for name in compat._NATIVE_AXISDATA_EXTRA_MEMBERS:
            monkeypatch.delattr(cls, name)
    else:
        for name, value in (
            ("__firstlineno__", 1),
            ("__static_attributes__", ()),
            ("__replace__", cls.__init__),
        ):
            monkeypatch.setattr(cls, name, value, raising=False)
    with pytest.raises(ValueError, match="native AxisData class namespace"):
        compat._initial_native_axisdata_roles()


@pytest.mark.parametrize("role", ("exported", "pre-slots"))
@pytest.mark.parametrize(
    "member,value",
    (
        ("__firstlineno__", True),
        ("__static_attributes__", []),
        ("__static_attributes__", ("unobserved",)),
        ("__replace__", _source),
    ),
)
def test_actual_axisdata_extended_metadata_requires_observed_types_and_binding(
    role: str, member: str, value: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    cls = compat._NATIVE_AXISDATA["classes"][role]
    monkeypatch.setattr(cls, member, value, raising=False)
    with pytest.raises(ValueError, match="native AxisData"):
        compat._initial_native_axisdata_roles()


@pytest.mark.parametrize("role", ("exported", "pre-slots"))
@pytest.mark.parametrize(
    "member", ("__firstlineno__", "__static_attributes__", "__replace__")
)
def test_actual_axisdata_extended_class_state_is_not_discarded(
    role: str, member: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture = _capture(_mapped(), tmp_path)
    cls = compat._NATIVE_AXISDATA["classes"][role]
    value: Any
    if member == "__firstlineno__":
        value = vars(cls).get(member, 0) + 1
    elif member == "__static_attributes__":
        value = ("changed",)
    else:
        value = _source
    monkeypatch.setattr(cls, member, value, raising=False)
    assert not capture.check_target("changed-axisdata-" + role + "-" + member)


@pytest.mark.parametrize("name", tuple(compat._NATIVE_AXISDATA["helpers"]))
def test_actual_axisdata_all_helper_defining_bindings_are_owned(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture = _capture(_mapped(), tmp_path)
    monkeypatch.setattr(compat._native_dataclasses, name, object())
    assert not capture.check_target("changed-axisdata-helper-binding-" + name)


@pytest.mark.parametrize(
    "name",
    tuple(
        name
        for name, value in compat._NATIVE_AXISDATA["helpers"].items()
        if type(value) is types.FunctionType
    ),
)
def test_actual_axisdata_owned_helper_code_is_not_discarded(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture = _capture(_mapped(), tmp_path)
    function = compat._NATIVE_AXISDATA["helpers"][name]
    monkeypatch.setattr(
        function,
        "__code__",
        function.__code__.replace(co_filename="changed-axisdata.py"),
    )
    assert not capture.check_target("changed-axisdata-helper-code-" + name)


@pytest.mark.parametrize("name", ("_FIELD_CLASSVAR", "_FIELD_INITVAR"))
@pytest.mark.parametrize("fault", ("binding", "name"))
def test_actual_axisdata_replace_markers_have_exact_contextual_ownership(
    name: str, fault: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture = _capture(_mapped(), tmp_path)
    owned = name in compat._NATIVE_AXISDATA["markers"]
    if fault == "binding":
        monkeypatch.setattr(compat._native_dataclasses, name, object())
    else:
        monkeypatch.setattr(vars(compat._native_dataclasses)[name], "name", "changed")
    assert capture.check_target("changed-axisdata-replace-marker-" + name) is not owned


@pytest.mark.parametrize(
    "fault",
    (
        "flatten-binding",
        "flatten-annotations",
        "registry",
        "mutable-array-checks",
        "mesh-axis-config",
        "config-owner",
        "axis-contents",
        "mesh-contents",
    ),
)
def test_selected_vmap_live_dependency_registry_configuration_and_context_are_owned(
    fault: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mapped = _mapped()
    capture = _capture(mapped, tmp_path)
    if fault == "flatten-binding":
        monkeypatch.setattr(compat._native_batching, "flatten_fun_for_vmap", _source)
    elif fault == "flatten-annotations":
        monkeypatch.setitem(
            compat._native_batching.flatten_fun_for_vmap.func.__annotations__,
            "state",
            "changed",
        )
    elif fault == "registry":
        monkeypatch.setitem(compat._native_batching.vmappables, object, (int, int))
    elif fault in {"mutable-array-checks", "mesh-axis-config"}:
        name = (
            "mutable_array_checks"
            if fault == "mutable-array-checks"
            else "remove_size_one_mesh_axis_from_type"
        )
        owner = compat._NATIVE_WRAPPER_CONFIGS[name]
        with owner(not owner.value):
            assert not capture.check_target("changed-vmap-config")
        return
    elif fault == "config-owner":

        class Hostile:
            def get_local(self) -> Any:
                raise AssertionError("foreign configuration invoked")

            @property
            def value(self) -> Any:
                raise AssertionError("foreign configuration property invoked")

        monkeypatch.setattr(compat._native_config, "mutable_array_checks", Hostile())
    elif fault == "axis-contents":
        context = compat._NATIVE_WRAPPER_CONTEXT_ROLES
        axis = context["axis_config"].value
        mapping = context["axis_slots"][0][1].__get__(axis, context["axis_type"])
        base, descriptor = context["axis_mapping_dict"]
        assert vars(base)["__dict__"] is descriptor
        backing = descriptor.__get__(mapping, type(mapping))["_d"]
        monkeypatch.setitem(backing, "changed", 2)
    else:
        context = compat._NATIVE_WRAPPER_CONTEXT_ROLES
        mesh = compat._native_api.get_abstract_mesh()
        _, descriptor = context["mesh_dict"]
        backing = descriptor.__get__(mesh, type(mesh))
        # Keep size unchanged: capture must inspect the structural axis field.
        monkeypatch.setitem(backing, "axis_sizes", (1,))
    assert not capture.check_target("changed-native-vmap-dependency")


def test_actual_flatten_partial_stays_supported_and_curried_export_gap_is_not_expanded() -> (
    None
):
    graph = compat._NativeCaptureGraph()
    graph.take(compat._native_batching.flatten_fun_for_vmap)
    assert graph.nodes
    with pytest.raises(ValueError, match="unsupported native callable state type"):
        compat._NativeCaptureGraph().take(
            compat._native_linear_util.transformation_with_aux2
        )


@pytest.mark.parametrize("scenario", probe.SCENARIOS)
def test_unchanged_original_mlp2_full_case_api_preserves_missing_criteria_and_next_outcome(
    scenario: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("raw product observation must not execute ORT")

    monkeypatch.setattr(ort, "InferenceSession", forbidden)
    payload = probe.DEFAULT_BASELINE.read_bytes()
    plan = probe.build_probe_plan(
        json.loads(payload), baseline_sha256=hashlib.sha256(payload).hexdigest()
    )
    assert len(plan.rows) == 1852 and len(plan.cells) == 3704
    registrations, _ = raw_registrations()
    provenance = SourceProvenance("example::examples.eqx::MlpExample", 2)
    registered = {origin: value for origin, _, value in registrations}
    row = next(
        row
        for row in plan.rows
        if (row.origin, row.ordinal) == (provenance.origin, provenance.ordinal)
    )
    original_case = registered[provenance.origin]["testcases"][provenance.ordinal]
    metadata = probe._resolve_case(row, registered)
    factory, callback, criteria = (
        metadata["callable"],
        metadata.get("post_check_onnx_graph"),
        metadata.get("criteria"),
    )
    adapters = json.loads(
        (
            probe.REPOSITORY_ROOT
            / "internal_docs/precision/source_replay_adapters.json"
        ).read_text()
    )
    result = probe.execute_probe_plan(
        plan,
        registrations,
        raw_replay=True,
        stage="product",
        limit=1,
        cases=frozenset({provenance}),
        scenario=scenario,
        replay_adapters=adapters,
        evidence_directory=tmp_path / "raw-evidence",
    )
    restored = probe._resolve_case(row, registered)
    assert restored["callable"] is factory
    assert restored.get("post_check_onnx_graph") is callback
    assert restored.get("criteria") is criteria is None
    assert (
        registered[provenance.origin]["testcases"][provenance.ordinal] is original_case
    )
    assert row.criteria is None and row.skip_numeric_validation is True
    assert restored["skip_numeric_validation"] is True
    assert restored["normalization_mode"] == "prefer_native"
    assert "rtol" not in restored and "atol" not in restored
    identity = (provenance.origin, provenance.ordinal, scenario)
    cell = next(cell for cell in result.cells if cell.identity == identity)
    (tmp_path / "actual-cell-and-row.json").write_text(
        json.dumps({"cell": asdict(cell), "row": asdict(row)}, default=str, indent=2)
    )
    assert result.rows is plan.rows and len(result.cells) == 3704
    assert (
        sum(
            before is after
            for before, after in zip(plan.cells, result.cells, strict=True)
        )
        == 3703
    )
    assert cell.numerics is cell.runtime is CheckStatus.UNVERIFIED
    record = cell.history[-1]
    captured = record.replay_capture
    assert captured["instantiated-target-state"]["status"] == "captured"
    assert captured["native_outcomes"]["eager"]["status"] == "returned"
    assert captured["native_outcomes"]["fresh-jit"]["status"] == "returned"
    # Eager/JIT/typed/product facts, including any next explicit gap, are saved
    # by the unchanged raw case API. Missing numerical criteria stay missing.


@pytest.mark.parametrize("name", ("debug_info", "format_exception_only"))
def test_native_helper_default_loaded_layout_and_complete_capture(
    name: str, tmp_path: Path
) -> None:
    assert compat._NATIVE_HELPER_DEFAULT_GAP is None
    entry = compat._NATIVE_HELPER_DEFAULT_ROLES[name]
    graph = compat._NativeHelperDefaultCaptureGraph(entry["role"])
    forbidden_codes = [entry["code"], entry["parent_code"]]
    representation = vars(entry["class"]).get("__repr__")
    if type(representation) is types.FunctionType:
        forbidden_codes.append(representation.__code__)
    calls = []

    def observe(frame: Any, event: str, argument: Any) -> None:
        if event == "call" and any(frame.f_code is code for code in forbidden_codes):
            calls.append(frame.f_code.co_name)

    previous = sys.getprofile()
    try:
        sys.setprofile(observe)
        graph.take_selected_helper(entry["function"], name)
    finally:
        sys.setprofile(previous)
    assert calls == []
    expected = {"__module__", "__dict__", "__weakref__", "__doc__"}
    if name == "format_exception_only":
        expected.add("__repr__")
    assert set(vars(entry["class"])) == expected
    assert type(entry["class"]) is type
    assert entry["class"].__bases__ == (object,)
    assert entry["class"].__mro__ == (entry["class"], object)
    descriptor = vars(entry["class"])["__dict__"]
    assert type(descriptor) is types.GetSetDescriptorType
    assert descriptor.__objclass__ is entry["class"]
    assert descriptor.__get__(entry["instance"], entry["class"]) is entry["dictionary"]
    assert entry["dictionary"] == {}
    for value in (
        entry["instance"],
        entry["class"],
        entry["dictionary"],
        entry["container"],
        entry["function"],
        entry["code"],
        entry["globals"],
        descriptor,
    ):
        assert graph.retained[id(value)] is value
    document = {
        "role": entry["role"],
        "name": name,
        "function_identity": id(entry["function"]),
        "code_identity": id(entry["code"]),
        "globals_identity": id(entry["globals"]),
        "default_container_identity": id(entry["container"]),
        "instance_identity": id(entry["instance"]),
        "class_identity": id(entry["class"]),
        "dictionary_identity": id(entry["dictionary"]),
        "own_members": sorted(expected),
        "class_name": entry["class"].__name__,
        "class_qualname": entry["class"].__qualname__,
        "class_module": entry["class"].__module__,
        "edges": [
            {
                "module_identity": id(owner),
                "namespace_identity": id(namespace),
                "slot": slot,
                "original_identity": id(original),
            }
            for owner, namespace, slot, original in entry["edges"]
        ],
        "nodes": graph.nodes,
    }
    (tmp_path / (name + "-loaded-layout.json")).write_text(
        json.dumps(document, indent=2)
    )
    mapped = _mapped()
    capture = _capture(mapped, tmp_path, name)
    assert capture.check_target("both-helper-defaults-completely-captured")


@pytest.mark.parametrize("name", ("debug_info", "format_exception_only"))
@pytest.mark.parametrize(
    "change",
    ("container", "default", "instance-dictionary", "instance-state", "instance-class"),
)
def test_native_helper_default_original_containers_and_instance_are_required(
    name: str, change: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = compat._NATIVE_HELPER_DEFAULT_ROLES[name]
    graph = compat._NativeHelperDefaultCaptureGraph(entry["role"])
    graph.take_selected_helper(entry["function"], name)
    reference = weakref.ref(entry["instance"])
    function, container = entry["function"], entry["container"]
    attribute = "__kwdefaults__" if entry["keyword"] else "__defaults__"
    descriptor = vars(entry["class"])["__dict__"]
    try:
        if change in ("container", "default"):
            replacement = (
                dict(container) if entry["keyword"] else tuple(list(container))
            )
            if change == "default":
                fresh = object.__new__(entry["class"])
                if entry["keyword"]:
                    replacement["result_paths_thunk"] = fresh
                else:
                    replacement = (fresh,)
            monkeypatch.setattr(function, attribute, replacement)
        elif change == "instance-dictionary":
            descriptor.__set__(entry["instance"], {})
        elif change == "instance-class":

            class Foreign:
                pass

            object.__setattr__(entry["instance"], "__class__", Foreign)
        else:
            entry["dictionary"]["extra"] = 1
        gc.collect()
        assert reference() is entry["instance"]
        assert graph.retained[id(entry["instance"])] is entry["instance"]
        with pytest.raises(ValueError, match="native helper"):
            compat._NativeHelperDefaultCaptureGraph(entry["role"]).take_selected_helper(
                function, name
            )
    finally:
        object.__setattr__(entry["instance"], "__class__", entry["class"])
        descriptor.__set__(entry["instance"], entry["dictionary"])
        entry["dictionary"].clear()
        monkeypatch.undo()


@pytest.mark.parametrize(
    "name,edge",
    (
        ("debug_info", "instance"),
        ("debug_info", "class"),
        ("debug_info", "function"),
        ("debug_info", "core-alias"),
        ("debug_info", "core-class"),
        ("debug_info", "api-function"),
        ("format_exception_only", "instance"),
        ("format_exception_only", "class"),
        ("format_exception_only", "function"),
        ("format_exception_only", "jax-helper"),
    ),
)
def test_native_helper_old_default_alias_rebinding_is_rejected(
    name: str, edge: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = compat._NATIVE_HELPER_DEFAULT_ROLES[name]
    selected = None
    for owner, namespace, slot, original in entry["edges"]:
        defining = compat._native_module_dictionary(owner) is entry["globals"]
        class_owner = (
            owner is compat._native_linear_util if entry["keyword"] else defining
        )
        match = (
            (edge == "instance" and original is entry["instance"] and class_owner)
            or (edge == "class" and original is entry["class"] and class_owner)
            or (edge == "function" and original is entry["function"] and defining)
            or (
                edge == "core-alias"
                and original is entry["instance"]
                and owner is compat._native_core
            )
            or (
                edge == "core-class"
                and original is entry["class"]
                and owner is compat._native_core
            )
            or (
                edge == "api-function"
                and original is entry["function"]
                and owner is compat._native_api
            )
            or (
                edge == "jax-helper"
                and original is entry["parent"]
                and owner is compat._native_traceback
            )
        )
        if match:
            selected = owner, slot
            break
    assert selected is not None
    monkeypatch.setattr(selected[0], selected[1], object())
    with pytest.raises(ValueError, match="alias changed"):
        compat._NativeHelperDefaultCaptureGraph(entry["role"]).take_selected_helper(
            entry["function"], name
        )
    assert entry["container"] is (
        entry["function"].__kwdefaults__
        if entry["keyword"]
        else entry["function"].__defaults__
    )


@pytest.mark.parametrize("name", ("debug_info", "format_exception_only"))
@pytest.mark.parametrize(
    "field",
    (
        "__name__",
        "__qualname__",
        "__module__",
        "__doc__",
        "extra",
        "__weakref__",
        "__repr__",
        "missing-member",
    ),
)
def test_native_helper_class_schema_and_descriptor_changes_fail_before_hooks(
    name: str, field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = compat._NATIVE_HELPER_DEFAULT_ROLES[name]

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("helper class hook executed")

    replacement = (
        property(forbidden)
        if field == "__weakref__"
        else classmethod(forbidden) if field == "__repr__" else "changed"
    )
    if field in ("__name__", "__qualname__"):
        with _changed_class_symbol(entry["class"], field, "changed"):
            with pytest.raises(ValueError, match="native helper"):
                compat._NativeHelperDefaultCaptureGraph(
                    entry["role"]
                ).take_selected_helper(entry["function"], name)
        return
    if field == "missing-member":
        monkeypatch.delattr(entry["class"], "__weakref__")
    else:
        monkeypatch.setattr(
            entry["class"],
            field,
            replacement,
            raising=field not in ("extra", "__repr__"),
        )
    with pytest.raises(ValueError, match="native helper"):
        compat._NativeHelperDefaultCaptureGraph(entry["role"]).take_selected_helper(
            entry["function"], name
        )


@pytest.mark.parametrize(
    "field",
    ("__code__", "__defaults__", "__kwdefaults__", "__annotations__", "__dict__"),
)
def test_stdlib_sentinel_repr_complete_envelope_drift_is_observed(
    field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = compat._NATIVE_HELPER_DEFAULT_ROLES["format_exception_only"]

    def snapshot() -> dict[str, Any]:
        graph = compat._NativeHelperDefaultCaptureGraph(entry["role"])
        graph.take_selected_helper(entry["function"], "format_exception_only")
        return graph.nodes

    baseline = snapshot()
    replacement = {
        "__code__": _source.__code__,
        "__defaults__": (1,),
        "__kwdefaults__": {"state": 1},
        "__annotations__": {"state": "changed"},
        "__dict__": {"state": 1},
    }[field]
    monkeypatch.setattr(vars(entry["class"])["__repr__"], field, replacement)
    assert snapshot() != baseline


@pytest.mark.parametrize("name", ("debug_info", "format_exception_only"))
def test_native_helper_permission_is_only_the_selected_default_edge(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = compat._NATIVE_HELPER_DEFAULT_ROLES[name]
    graph_type = compat._NativeHelperDefaultCaptureGraph
    with pytest.raises(ValueError, match="selected role"):
        graph_type("wrong-role").take_selected_helper(entry["function"], name)

    def unrelated(value: Any = entry["instance"]) -> Any:
        raise AssertionError("unrelated helper executed")

    for value in (entry["instance"], entry["class"], unrelated, entry["function"]):
        with pytest.raises(ValueError, match="unsupported native callable"):
            graph_type(entry["role"]).take(value)
    monkeypatch.setitem(vars(entry["function"]), "foreign_role", entry["instance"])
    with pytest.raises(ValueError, match="unsupported native callable"):
        graph_type(entry["role"]).take_selected_helper(entry["function"], name)


@pytest.mark.parametrize("name", ("debug_info", "format_exception_only"))
def test_native_helper_subclass_and_hostile_lookalike_have_no_permission(
    name: str,
) -> None:
    entry = compat._NATIVE_HELPER_DEFAULT_ROLES[name]

    class Subclass(entry["class"]):
        pass

    class HostileMeta(type):
        def __getattribute__(self, attribute: str) -> Any:
            raise AssertionError("hostile helper metaclass executed")

    class Hostile(metaclass=HostileMeta):
        def __repr__(self) -> str:
            raise AssertionError("hostile helper repr executed")

        def __eq__(self, other: Any) -> bool:
            raise AssertionError("hostile helper equality executed")

        def __hash__(self) -> int:
            raise AssertionError("hostile helper hash executed")

    for value in (object.__new__(Subclass), object.__new__(Hostile)):
        with pytest.raises(ValueError, match="unsupported native callable"):
            compat._NativeHelperDefaultCaptureGraph(entry["role"]).take(value)


@pytest.mark.parametrize("name", ("debug_info", "format_exception_only"))
def test_native_helper_original_module_registration_is_required(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = compat._NATIVE_HELPER_DEFAULT_ROLES[name]
    for owner, namespace, original_name in entry["module_names"]:
        try:
            monkeypatch.setitem(sys.modules, original_name, object())
            with pytest.raises(ValueError, match="defining module binding"):
                compat._NativeHelperDefaultCaptureGraph(
                    entry["role"]
                ).take_selected_helper(entry["function"], name)
        finally:
            monkeypatch.undo()
        assert sys.modules[original_name] is owner


@pytest.mark.parametrize("name", ("debug_info", "format_exception_only"))
def test_native_helper_equivalent_replaced_code_is_rejected(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = compat._NATIVE_HELPER_DEFAULT_ROLES[name]
    monkeypatch.setattr(entry["function"], "__code__", entry["code"].replace())
    with pytest.raises(ValueError, match="original code/globals"):
        compat._NativeHelperDefaultCaptureGraph(entry["role"]).take_selected_helper(
            entry["function"], name
        )


@pytest.mark.parametrize(
    "role,name",
    (
        ("reraise_with_filtered_traceback", "Exception"),
        ("reraise_with_filtered_traceback", "ValueError"),
        ("vmap_f", "len"),
    ),
)
def test_native_direct_builtin_fallback_shadow_invalidates_retained_wrapper(
    role: str, name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mapped = _mapped()
    selected = (
        mapped
        if role == "reraise_with_filtered_traceback"
        else _closure(mapped, "fun").cell_contents
    )
    original = compat.selected_native_transformed_function(selected)
    assert original is not None
    evidence = original.document["implementation"]["direct_builtin_lookups"]
    assert evidence["builtins_identity"] == id(selected.__builtins__)
    assert evidence["direct_fallbacks"][name]["global_present"] is False
    capture = _capture(mapped, tmp_path)
    namespace = selected.__globals__
    assert name not in namespace
    try:
        monkeypatch.setitem(
            namespace, name, ValueError if name == "Exception" else object()
        )
        with pytest.raises(ValueError, match="direct builtin global shadow"):
            compat.selected_native_transformed_function(selected)
        assert not capture.check_target("builtin-global-shadow")
    finally:
        monkeypatch.undo()
    restored = compat.selected_native_transformed_function(selected)
    assert restored is not None and restored.document == original.document


@pytest.mark.parametrize("path", ("zip", "util.safe_zip", "util.jaxlib_utils.safe_zip"))
@pytest.mark.parametrize("fault", ("builtin", "hostile", "missing"))
def test_native_vmap_imported_safe_zip_aliases_are_owned_and_restored(
    path: str, fault: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mapped = _mapped()
    vmap_f = _closure(mapped, "fun").cell_contents
    original = compat.selected_native_transformed_function(vmap_f)
    assert original is not None
    paths = ("zip", "util.safe_zip", "util.jaxlib_utils.safe_zip")
    bindings = tuple(compat._native_wrapper_slot(compat._native_api, p) for p in paths)
    native_zip = bindings[0][-1][-1]
    assert type(native_zip) is types.BuiltinFunctionType
    assert all(binding[-1][-1] is native_zip for binding in bindings)
    assert native_zip is not vmap_f.__builtins__["zip"]
    assert any(value is native_zip for value in original.retained)
    implementation = original.document["implementation"]
    assert "zip" not in implementation["direct_builtin_lookups"]["direct_fallbacks"]
    assert all(
        implementation["roots"][p] == implementation["roots"]["zip"] for p in paths
    )
    capture = _capture(mapped, tmp_path)
    _, namespace, slot, _ = bindings[paths.index(path)][-1]
    calls = []

    class Hostile:
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            calls.append("call")
            raise AssertionError("foreign zip called")

        def __eq__(self, other: Any) -> Any:
            calls.append("equality")
            raise AssertionError("foreign zip equality called")

    try:
        if fault == "missing":
            monkeypatch.delitem(namespace, slot)
        else:
            monkeypatch.setitem(
                namespace,
                slot,
                vmap_f.__builtins__["zip"] if fault == "builtin" else Hostile(),
            )
        with pytest.raises(ValueError, match="implementation slot changed"):
            compat.selected_native_transformed_function(vmap_f)
        assert not capture.check_target("changed-native-safe-zip")
        assert calls == []
    finally:
        monkeypatch.undo()
    restored = compat.selected_native_transformed_function(vmap_f)
    assert restored is not None and restored.document == original.document
    assert capture.check_target("restored-native-safe-zip")


@pytest.mark.parametrize("role", ("reraise_with_filtered_traceback", "vmap_f"))
def test_native_exact_code_globals_foreign_actual_builtins_rejected(role: str) -> None:
    mapped = _mapped()
    original = (
        mapped
        if role == "reraise_with_filtered_traceback"
        else _closure(mapped, "fun").cell_contents
    )
    namespace = original.__globals__
    present, prior = "__builtins__" in namespace, namespace.get("__builtins__")
    try:
        namespace["__builtins__"] = original.__builtins__
        ordinary = types.FunctionType(
            original.__code__, namespace, closure=original.__closure__
        )
        namespace["__builtins__"] = dict(original.__builtins__)
        foreign = types.FunctionType(
            original.__code__, namespace, closure=original.__closure__
        )
    finally:
        if present:
            namespace["__builtins__"] = prior
        else:
            del namespace["__builtins__"]
    assert ordinary.__builtins__ is original.__builtins__
    assert compat.native_transformed_function_role(ordinary) == role
    assert foreign.__code__ is original.__code__ and foreign.__globals__ is namespace
    assert foreign.__builtins__ is not original.__builtins__
    with pytest.raises(ValueError, match="actual builtin mapping"):
        compat.native_transformed_function_role(foreign)
    assert compat.selected_native_transformed_function(original) is not None


def test_native_stdlib_repr_equal_annotation_container_replacement_is_observed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = compat._NATIVE_HELPER_DEFAULT_ROLES["format_exception_only"]
    representation = vars(entry["class"])["__repr__"]

    def snapshot() -> Any:
        graph = compat._NativeHelperDefaultCaptureGraph(entry["role"])
        graph.take_selected_helper(entry["function"], "format_exception_only")
        return graph

    original = representation.__annotations__
    baseline = snapshot()
    monkeypatch.setattr(representation, "__annotations__", dict(original))
    observed = snapshot()
    gc.collect()
    assert representation.__annotations__ is not original
    assert baseline.retained[id(original)] is original
    assert (
        observed.retained[id(representation.__annotations__)]
        is representation.__annotations__
    )
    assert observed.nodes != baseline.nodes


def test_native_stdlib_repr_shared_annotation_container_alias_split_is_observed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = compat._NATIVE_HELPER_DEFAULT_ROLES["format_exception_only"]
    representation = vars(entry["class"])["__repr__"]
    shared = ["metadata"]
    annotations = {"a": shared, "b": shared}
    monkeypatch.setattr(representation, "__annotations__", annotations)

    def snapshot() -> Any:
        graph = compat._NativeHelperDefaultCaptureGraph(entry["role"])
        graph.take_selected_helper(entry["function"], "format_exception_only")
        return graph

    baseline = snapshot()
    annotations["b"] = list(shared)
    observed = snapshot()
    assert (
        annotations["a"] == annotations["b"]
        and annotations["a"] is not annotations["b"]
    )
    assert baseline.retained[id(shared)] is shared
    assert observed.nodes != baseline.nodes


@pytest.mark.parametrize("scenario_index", [0, 1])
def test_unchanged_simple_linear_union_annotation_capture_keeps_native_work_unattempted(
    scenario_index: int, tmp_path: Path
) -> None:
    payload = probe.DEFAULT_BASELINE.read_bytes()
    plan = probe.build_probe_plan(
        json.loads(payload), baseline_sha256=hashlib.sha256(payload).hexdigest()
    )
    provenance = SourceProvenance("example::examples.eqx::SimpleLinearExample", 1)
    row = next(
        row
        for row in plan.rows
        if (row.origin, row.ordinal) == (provenance.origin, provenance.ordinal)
    )
    registrations, _ = raw_registrations()
    registered = {origin: value for origin, _, value in registrations}
    original_case = registered[provenance.origin]["testcases"][provenance.ordinal]
    metadata = probe._resolve_case(row, registered)
    factory = metadata["callable"]
    callback = metadata["post_check_onnx_graph"]
    scenario = source.raw_source_scenarios(metadata, provenance, criteria=row.criteria)[
        scenario_index
    ]
    assert scenario.criteria is row.criteria is None
    assert scenario.model_dtype is source.PRESERVE_FACTORY_DTYPE
    assert scenario.default_matmul_precision is None
    assert scenario.caller_x64_enabled is bool(scenario_index)
    before = source.SourceSettings.observe()
    with source.source_configuration(scenario):
        target = factory.instantiate()
        assert type(target) is types.FunctionType
        annotations = target.__annotations__
        capture = _capture(target, tmp_path)
        assert capture.check_target("unchanged")
        assert target.__annotations__ is annotations
        witness = capture.target_witness
        assert witness is not None
        unions = [
            (path, value) for path, value in witness.edges if path[-1:] == ("union",)
        ]
        assert unions
        for path, union in unions:
            assert type(union) is types.UnionType
            arguments = raw._NATIVE_UNION_ARGS.__get__(union, types.UnionType)
            assert len(arguments) == 2
            assert arguments[0] is compat.Array and arguments[1] is type(None)
            edges = dict(witness.edges)
            base = path[:-1]
            assert edges[base + ("union-args",)] is arguments
            assert edges[base + ("union-arg[0]",)] is compat.Array
            assert edges[base + ("union-arg[1]",)] is type(None)
        assert capture.records["instantiated-target-state"]["status"] == "captured"
        assert capture.records["native_outcomes"] == {}
        assert capture.records["product_admission"] == "not-attempted"
        assert capture.records["raw_callback"]["status"] == "not-executed"
        assert capture.readiness is CheckStatus.UNVERIFIED
    assert source.SourceSettings.observe() == before
    assert metadata["callable"] is factory
    assert metadata["post_check_onnx_graph"] is callback
    assert metadata.get("criteria") is None
    assert (
        registered[provenance.origin]["testcases"][provenance.ordinal] is original_case
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_native_union_order_none_class_and_strong_original_edges(
    reverse: bool, tmp_path: Path
) -> None:
    union = (None | jax.Array) if reverse else (jax.Array | None)
    capture = _capture(union, tmp_path)
    assert capture.check_target("unchanged")
    assert capture.target_state["kind"] == "native-union-type"
    kinds = [leaf["kind"] for leaf in capture.target_state["args"]]
    expected = ["native-jax-array-class", "native-none-type-class"]
    assert kinds == (list(reversed(expected)) if reverse else expected)
    assert raw._capture_target_state(None) not in capture.target_state["args"]
    witness = capture.target_witness
    assert witness is not None
    assert witness.selected == {("root",): "native-union-annotation"}
    arguments = raw._NATIVE_UNION_ARGS.__get__(union, types.UnionType)
    gc.collect()
    edges = dict(witness.edges)
    assert edges[("root", "union")] is union
    assert edges[("root", "union-args")] is arguments
    for index, argument in enumerate(arguments):
        assert edges[("root", f"union-arg[{index}]")] is argument
    assert edges[("root", "array-module")] is jax
    assert edges[("root", "array-module-dict")] is vars(jax)
    assert edges[("root", "array-class")] is compat.Array
    assert edges[("root", "array-metaclass")] is type(compat.Array)


def test_equal_native_union_with_reversed_order_does_not_replace_original(
    tmp_path: Path,
) -> None:
    original = jax.Array | None
    replacement = None | jax.Array
    assert original == replacement and original is not replacement
    capture = _capture(original, tmp_path)
    capture.target = replacement
    assert not capture.check_target("replaced-union")
    assert any(value is original for _, value in capture.target_witness.edges)


@pytest.mark.parametrize("mutation", ["container", "shared-alias"])
def test_union_only_selected_state_retains_annotation_container_aliases(
    mutation: str,
    tmp_path: Path,
) -> None:
    shared = [jax.Array | None]
    annotations = {"left": shared, "right": shared}
    capture = _capture(annotations, tmp_path)
    assert capture.check_target("unchanged")
    assert capture.target_witness is not None
    if mutation == "container":
        capture.target = dict(annotations)
    else:
        annotations["right"] = list(shared)
        assert annotations["left"] == annotations["right"]
        assert annotations["left"] is not annotations["right"]
    assert not capture.check_target("annotation-" + mutation)
    assert any(value is shared for _, value in capture.target_witness.edges)


def test_selected_wrapper_annotation_container_replacement_with_same_union_is_observed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _mapped()
    annotations = {"value": jax.Array | None}
    monkeypatch.setattr(target, "__annotations__", annotations)
    capture = _capture(target, tmp_path)
    monkeypatch.setattr(target, "__annotations__", dict(annotations))
    assert not capture.check_target("replacement-annotations")
    assert any(value is annotations for _, value in capture.target_witness.edges)


@pytest.mark.parametrize("binding", ["public-class", "public-module", "metaclass"])
def test_native_union_rejects_rebound_array_or_metaclass_binding(
    binding: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    union = jax.Array | None
    capture = _capture(union, tmp_path)
    original = jax.Array
    with monkeypatch.context() as change:
        if binding == "public-class":
            change.setattr(jax, "Array", type("Array", (), {"__module__": "jax"}))
        elif binding == "public-module":
            change.setitem(sys.modules, "jax", types.ModuleType("jax"))
        else:
            module = raw._NATIVE_UNION_ARRAY_META_MODULE
            name = raw._NATIVE_UNION_ARRAY_META_SYMBOL[1]
            change.setattr(module, name, type(name, (), {}))
        with pytest.raises(source.SourcePreparationError, match="Native union Array"):
            raw._capture_target_state(union)
        assert not capture.check_target("binding")
    assert jax.Array is original and sys.modules["jax"] is jax
    restored = raw._capture_target_state(union)
    assert restored == capture.target_state


@pytest.mark.parametrize("member", ["class", "generic-alias", "hostile"])
def test_native_union_unknown_member_is_an_explicit_gap_without_hooks(
    member: str,
    tmp_path: Path,
) -> None:
    calls = []

    class HostileMeta(type):
        def __repr__(cls) -> str:
            calls.append("repr")
            raise AssertionError("foreign repr must not normalize an annotation")

        def __getattribute__(cls, name: str) -> Any:
            if name in {"__module__", "__qualname__", "__args__"}:
                calls.append(name)
                raise AssertionError("foreign metadata hook must not run")
            return super().__getattribute__(name)

    class Hostile(metaclass=HostileMeta):
        pass

    unknown = {"class": int, "generic-alias": list[int], "hostile": Hostile}[member]
    union = jax.Array | unknown
    capture = raw.RawReplayCapture(tmp_path / "capture")
    with pytest.raises(
        source.SourcePreparationError, match="exact state capture adapter"
    ):
        capture.bind_target(union)
    assert calls == []
    assert capture.gaps and capture.records["native_outcomes"] == {}
    assert capture.readiness is CheckStatus.UNVERIFIED
    with pytest.raises(source.SourcePreparationError, match="member needs"):
        raw._capture_target_state(union)


@pytest.mark.parametrize("mutation", ["kind", "order", "omission"])
def test_native_union_saved_argument_corruption_is_rejected(
    mutation: str,
    tmp_path: Path,
) -> None:
    capture = _capture(jax.Array | None, tmp_path)
    arguments = capture.target_state["args"]
    if mutation == "kind":
        arguments[1]["kind"] = "none-singleton"
    elif mutation == "order":
        arguments.reverse()
    else:
        arguments.pop()
    assert not capture.check_target("saved-" + mutation)


@pytest.mark.parametrize("kind", ["native-array", "foreign-class", "function-state"])
def test_native_union_leaf_permission_does_not_admit_actual_classes_or_class_state(
    kind: str,
    tmp_path: Path,
) -> None:
    class Foreign:
        pass

    if kind == "native-array":
        target = jax.Array
    elif kind == "foreign-class":
        target = Foreign
    else:
        owned = jax.Array

        def target(value: Any) -> Any:
            return owned, value

    capture = raw.RawReplayCapture(tmp_path / "capture")
    with pytest.raises(
        source.SourcePreparationError, match="exact state capture adapter"
    ):
        capture.bind_target(target)
    assert capture.records["native_outcomes"] == {}
    assert capture.readiness is CheckStatus.UNVERIFIED


def _native_typing_alias(reverse: bool = False) -> Any:
    literal = typing.Literal["scalar"]
    return typing.Union[literal, int] if reverse else typing.Union[int, literal]


def _native_typing_storage(value: Any) -> tuple[Any, dict[str, Any]]:
    boundary = raw._NATIVE_TYPING_BOUNDARY
    index = 0 if type(value) is boundary.union_class else 1
    descriptor = boundary.alias_dictionaries[index][2]
    return descriptor, descriptor.__get__(value, type(value))


def _clone_native_typing_alias(value: Any) -> Any:
    descriptor, namespace = _native_typing_storage(value)
    cloned = object.__new__(type(value))
    descriptor.__set__(cloned, dict(namespace))
    return cloned


@pytest.mark.parametrize("kind", ["union", "reverse", "literal"])
def test_native_typing_annotation_order_storage_and_strong_selected_edges(
    kind: str, tmp_path: Path
) -> None:
    alias = (
        typing.Literal["scalar"]
        if kind == "literal"
        else _native_typing_alias(kind == "reverse")
    )
    descriptor, namespace = _native_typing_storage(alias)
    original_items = tuple(namespace.items())
    arguments = namespace["__args__"]
    capture = _capture(alias, tmp_path)
    assert capture.check_target("unchanged-typing")
    assert capture.target_state["kind"] == "native-typing-annotation"
    assert [key for key, _ in capture.target_state["alias"]["fields"]] == list(
        namespace
    )
    fields = dict(capture.target_state["alias"]["fields"])
    kinds = [argument["kind"] for argument in fields["__args__"]]
    expected = (
        ["literal-string"] if kind == "literal" else ["native-int-class", "literal"]
    )
    assert kinds == (expected[::-1] if kind == "reverse" else expected)
    witness = capture.target_witness
    assert witness is not None
    assert witness.selected == {("root",): "native-typing-annotation"}
    edges = dict(witness.edges)
    assert edges[("root", "typing-alias")] is alias
    assert edges[("root", "typing-dictionary")] is namespace
    assert edges[("root", "typing-field:__args__")] is arguments
    assert edges[("root", "typing-field:__origin__")] is namespace["__origin__"]
    for index, argument in enumerate(arguments):
        assert edges[("root", f"typing-arg[{index}]")] is argument
    assert any(value is typing for _, value in witness.edges)
    assert any(type(value) is types.CodeType for _, value in witness.edges)
    assert descriptor.__get__(alias, type(alias)) is namespace
    assert tuple(namespace.items()) == original_items
    assert capture.records["native_outcomes"] == {}
    assert capture.records["product_admission"] == "not-attempted"
    assert capture.records["raw_callback"]["status"] == "not-executed"
    assert capture.readiness is CheckStatus.UNVERIFIED


@pytest.mark.parametrize(
    "mutation",
    [
        "alias",
        "dictionary",
        "args",
        "nested-literal",
        "container",
        "shared-alias",
        "shared-container",
    ],
)
def test_native_typing_equal_replacements_and_shared_alias_splits_are_observed(
    mutation: str, tmp_path: Path
) -> None:
    alias = _native_typing_alias()
    descriptor, namespace = _native_typing_storage(alias)
    items = tuple(namespace.items())
    arguments = namespace["__args__"]
    shared = [alias]
    target = (
        {"left": shared, "right": shared}
        if mutation == "shared-container"
        else {"left": alias, "right": alias}
    )
    capture = _capture(alias if mutation == "alias" else target, tmp_path)
    assert capture.check_target("unchanged-typing-container")
    try:
        if mutation == "alias":
            capture.target = _clone_native_typing_alias(alias)
        elif mutation == "dictionary":
            descriptor.__set__(alias, dict(namespace))
        elif mutation == "args":
            namespace["__args__"] = tuple(list(arguments))
            assert namespace["__args__"] is not arguments
        elif mutation == "nested-literal":
            namespace["__args__"] = (
                arguments[0],
                _clone_native_typing_alias(arguments[1]),
            )
        elif mutation == "container":
            capture.target = dict(target)
        elif mutation == "shared-alias":
            target["right"] = _clone_native_typing_alias(alias)
        else:
            target["right"] = list(shared)
        assert not capture.check_target("replaced-typing-" + mutation)
        assert any(value is alias for _, value in capture.target_witness.edges)
        assert any(value is namespace for _, value in capture.target_witness.edges)
    finally:
        descriptor.__set__(alias, namespace)
        namespace.clear()
        namespace.update(items)
    assert descriptor.__get__(alias, type(alias)) is namespace
    assert namespace["__args__"] is arguments


@pytest.mark.parametrize(
    "mutation",
    [
        "kind",
        "field-order",
        "origin",
        "args-order",
        "literal-string",
        "parameters",
        "omission",
        "module",
    ],
)
def test_native_typing_saved_snapshot_corruption_reaches_actual_check(
    mutation: str, tmp_path: Path
) -> None:
    capture = _capture(_native_typing_alias(), tmp_path)
    alias = capture.target_state["alias"]
    fields = dict(alias["fields"])
    if mutation == "kind":
        alias["kind"] = "foreign-alias"
    elif mutation == "field-order":
        alias["fields"].reverse()
    elif mutation == "origin":
        fields["__origin__"]["native-origin"] = "literal"
    elif mutation == "args-order":
        fields["__args__"].reverse()
    elif mutation == "literal-string":
        nested = dict(fields["__args__"][1]["fields"])
        nested["__args__"][0]["value"] = "other"
    elif mutation == "parameters":
        fields["__parameters__"].append({"typevar": "T"})
    elif mutation == "omission":
        alias["fields"].pop()
    else:
        fields["__module__"]["str"] = "foreign"
    assert not capture.check_target("saved-typing-" + mutation)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "extra",
        "reorder",
        "inst",
        "name",
        "origin",
        "slots",
        "args-type",
        "parameters",
        "module",
        "paramspec",
    ],
)
def test_native_typing_actual_alias_storage_schema_is_exact(
    mutation: str, tmp_path: Path
) -> None:
    alias = _native_typing_alias()
    descriptor, namespace = _native_typing_storage(alias)
    original = tuple(namespace.items())
    capture = _capture(alias, tmp_path)
    try:
        if mutation == "missing":
            namespace.pop("__slots__")
        elif mutation == "extra":
            namespace["unknown"] = None
        elif mutation == "reorder":
            namespace["_inst"] = namespace.pop("_inst")
        elif mutation == "inst":
            namespace["_inst"] = 1
        elif mutation == "name":
            namespace["_name"] = "Union"
        elif mutation == "origin":
            namespace["__origin__"] = typing.Literal
        elif mutation == "slots":
            namespace["__slots__"] = ()
        elif mutation == "args-type":
            namespace["__args__"] = list(namespace["__args__"])
        elif mutation == "parameters":
            namespace["__parameters__"] = (typing.TypeVar("T"),)
        elif mutation == "module":
            namespace["__module__"] = "foreign"
        else:
            namespace["_paramspec_tvars"] = True
        with pytest.raises(
            source.SourcePreparationError, match="Native typing annotation"
        ):
            raw._capture_target_state(alias)
        assert not capture.check_target("actual-typing-" + mutation)
    finally:
        descriptor.__set__(alias, namespace)
        namespace.clear()
        namespace.update(original)
    assert raw._capture_target_state(alias) == capture.target_state


@pytest.mark.parametrize(
    "kind",
    [
        "literal-other",
        "literal-int",
        "bool-as-int",
        "foreign-class",
        "generic-alias",
        "typevar",
        "forward-ref",
        "annotated",
        "subclass",
    ],
)
def test_native_typing_unmeasured_alias_grammar_remains_an_explicit_gap(
    kind: str, tmp_path: Path
) -> None:
    class Foreign:
        pass

    if kind == "literal-other":
        target = typing.Literal["other"]
    elif kind == "literal-int":
        target = typing.Literal[1]
    elif kind == "bool-as-int":
        target = typing.Union[bool, typing.Literal["scalar"]]
    elif kind == "foreign-class":
        target = typing.Union[Foreign, typing.Literal["scalar"]]
    elif kind == "generic-alias":
        target = list[int]
    elif kind == "typevar":
        target = typing.TypeVar("T")
    elif kind == "forward-ref":
        target = typing.ForwardRef("T")
    elif kind == "annotated":
        target = typing.Annotated[int, "scalar"]
    else:
        derived = types.new_class(
            "ForeignAlias", (type(_native_typing_alias()),), {"_root": True}
        )
        target = object.__new__(derived)
        descriptor, original = _native_typing_storage(_native_typing_alias())
        descriptor.__set__(target, dict(original))
    capture = raw.RawReplayCapture(tmp_path / "capture")
    with pytest.raises(
        source.SourcePreparationError, match="exact state capture adapter"
    ):
        capture.bind_target(target)
    assert capture.gaps and capture.target_witness is None
    assert capture.records["native_outcomes"] == {}
    assert capture.readiness is CheckStatus.UNVERIFIED


@pytest.mark.parametrize(
    "mutation",
    [
        "module",
        "class",
        "origin",
        "int",
        "descriptor",
        "getitem",
        "code",
        "same-code",
        "function-dictionary",
        "defaults",
        "literal-doc",
        "literal-slot",
        "literal-wrapped",
        "literal-closure",
        "origin-wrapper-storage",
    ],
)
def test_native_typing_binding_descriptor_and_implementation_changes_reject(
    mutation: str, tmp_path: Path
) -> None:
    alias = _native_typing_alias()
    capture = _capture(alias, tmp_path)
    boundary = raw._NATIVE_TYPING_BOUNDARY
    union = boundary.origins[0][1]
    getitem = boundary.origin_slots["_getitem"].__get__(union, type(union))

    def restore() -> None:
        return None

    def foreign(*args: Any) -> None:
        return None

    try:
        if mutation == "module":
            original = sys.modules["typing"]
            sys.modules["typing"] = types.ModuleType("typing")
            restore = functools.partial(sys.modules.__setitem__, "typing", original)
        elif mutation == "class":
            original = typing._UnionGenericAlias
            typing._UnionGenericAlias = type("Foreign", (), {})
            restore = functools.partial(setattr, typing, "_UnionGenericAlias", original)
        elif mutation == "origin":
            original = typing.Union
            typing.Union = object()
            restore = functools.partial(setattr, typing, "Union", original)
        elif mutation == "int":
            original = builtins.int
            builtins.int = type("ForeignInt", (), {})
            restore = functools.partial(setattr, builtins, "int", original)
        elif mutation == "descriptor":
            original = vars(typing._SpecialForm)["_name"]
            typing._SpecialForm._name = object()
            restore = functools.partial(setattr, typing._SpecialForm, "_name", original)
        elif mutation == "getitem":
            boundary.origin_slots["_getitem"].__set__(union, lambda *args: None)
            restore = functools.partial(
                boundary.origin_slots["_getitem"].__set__, union, getitem
            )
        elif mutation in ("code", "same-code"):
            original = getitem.__code__
            getitem.__code__ = (
                original.replace(co_name="changed")
                if mutation == "code"
                else original.replace()
            )
            restore = functools.partial(setattr, getitem, "__code__", original)
        elif mutation == "function-dictionary":
            original = getitem.__dict__
            getitem.__dict__ = dict(original)
            restore = functools.partial(setattr, getitem, "__dict__", original)
        elif mutation == "defaults":
            original = getitem.__defaults__
            getitem.__defaults__ = (None,)
            restore = functools.partial(setattr, getitem, "__defaults__", original)
        elif mutation == "literal-doc":
            namespace = boundary.literal_dictionary.__get__(
                typing.Literal, type(typing.Literal)
            )
            original = namespace["__doc__"]
            namespace["__doc__"] = original + "changed"
            restore = functools.partial(namespace.__setitem__, "__doc__", original)
        elif mutation == "literal-slot":
            descriptor = boundary.origin_slots["__doc__"]
            descriptor.__set__(typing.Literal, "unexpected inherited slot")
            restore = functools.partial(descriptor.__delete__, typing.Literal)
        else:
            literal_getitem = boundary.origin_slots["_getitem"].__get__(
                typing.Literal, type(typing.Literal)
            )
            if mutation == "literal-wrapped":
                namespace = literal_getitem.__dict__
                original = namespace["__wrapped__"]

                namespace["__wrapped__"] = foreign
                restore = functools.partial(
                    namespace.__setitem__, "__wrapped__", original
                )
            elif mutation == "literal-closure":
                cell = literal_getitem.__closure__[-1]
                original = cell.cell_contents

                cell.cell_contents = foreign
                restore = functools.partial(setattr, cell, "cell_contents", original)
            else:
                if boundary.variant == (3, 11):
                    cached = literal_getitem.__closure__[0].cell_contents
                    namespace = boundary.cache_dictionary.__get__(cached, type(cached))
                else:
                    namespace = literal_getitem.__dict__
                namespace["unknown"] = None
                restore = functools.partial(namespace.__delitem__, "unknown")
        # Restore shared builtins before invoking any pytest assertion helpers.
        rejected = not capture.check_target("typing-binding-" + mutation)
    finally:
        restore()
    assert rejected
    assert raw._capture_target_state(alias) == capture.target_state
    assert capture.check_target("restored-typing-binding")


@contextmanager
def _standin_native_array_symbol(
    symbol: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> Iterator[type[Any]]:
    array_module = types.ModuleType("jax")
    meta_module = types.ModuleType("_source_array_symbol_fixture")
    meta = type("ArrayMeta", (type,), {"__module__": meta_module.__name__})
    selected = meta("Array", (), {"__module__": symbol[0], "__qualname__": symbol[1]})
    meta_module.ArrayMeta = meta
    array_module.Array = selected
    bindings = {
        "jax": array_module,
        "_NATIVE_UNION_ARRAY": selected,
        "_NATIVE_UNION_ARRAY_SYMBOL": raw._native_union_class_symbol(selected),
        "_NATIVE_UNION_ARRAY_MODULE": array_module,
        "_NATIVE_UNION_ARRAY_META": meta,
        "_NATIVE_UNION_ARRAY_META_SYMBOL": raw._native_union_class_symbol(meta),
        "_NATIVE_UNION_ARRAY_META_MODULE": meta_module,
    }
    with monkeypatch.context() as change:
        change.setitem(sys.modules, "jax", array_module)
        change.setitem(sys.modules, meta_module.__name__, meta_module)
        for name, value in bindings.items():
            change.setattr(raw, name, value)
        yield selected


@pytest.mark.parametrize("symbol", [("jax", "Array"), ("jax", "jaxlib._jax.Array")])
def test_native_array_both_measured_symbols_preserve_public_alias_and_original_edges(
    symbol: tuple[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _standin_native_array_symbol(symbol, monkeypatch) as selected:
        union = selected | None
        capture = _capture(union, tmp_path)
        assert capture.check_target("unchanged-array-symbol")
        assert raw._NATIVE_UNION_ARRAY_SYMBOL == symbol
        assert raw._native_union_class_symbol(selected) == symbol
        assert vars(raw.jax)["Array"] is selected
        assert "jaxlib._jax.Array" not in vars(raw.jax)
        assert capture.target_state["args"][0] == {
            "kind": "native-jax-array-class",
            "symbol": ["jax", "Array"],
            "metaclass": list(raw._NATIVE_UNION_ARRAY_META_SYMBOL),
        }
        witness = capture.target_witness
        assert witness is not None
        edges = dict(witness.edges)
        assert edges[("root", "array-class")] is selected
        assert edges[("root", "array-module")] is raw.jax
        assert edges[("root", "array-metaclass")] is type(selected)
        assert (
            edges[("root", "array-metaclass-module")]
            is raw._NATIVE_UNION_ARRAY_META_MODULE
        )
        assert capture.records["native_outcomes"] == {}
        assert capture.records["product_admission"] == "not-attempted"
        assert capture.readiness is CheckStatus.UNVERIFIED


@pytest.mark.parametrize("symbol", [("jax", "Array"), ("jax", "jaxlib._jax.Array")])
@pytest.mark.parametrize("drift", ["module", "qualname", "other-measured-symbol"])
def test_native_array_symbol_changes_reject_even_another_measured_spelling(
    symbol: tuple[str, str], drift: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _standin_native_array_symbol(symbol, monkeypatch) as selected:
        union = selected | None
        capture = _capture(union, tmp_path)
        field = "__module__" if drift == "module" else "__qualname__"
        replacement = "foreign" if drift == "module" else "OtherArray"
        if drift == "other-measured-symbol":
            replacement = "jaxlib._jax.Array" if symbol[1] == "Array" else "Array"
        with _changed_class_symbol(selected, field, replacement):
            assert raw._NATIVE_UNION_ARRAY_SYMBOL == symbol
            with pytest.raises(
                source.SourcePreparationError, match="Native union Array"
            ):
                raw._capture_target_state(union)
            assert not capture.check_target("changed-array-symbol")
        assert raw._native_union_class_symbol(selected) == symbol
        assert capture.check_target("restored-array-symbol")


@pytest.mark.parametrize(
    "symbol", [("foreign", "Array"), ("jax", "OtherArray"), ("jaxlib._jax", "Array")]
)
def test_native_array_unmeasured_initial_symbol_is_a_local_explicit_gap(
    symbol: tuple[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _standin_native_array_symbol(symbol, monkeypatch) as selected:
        capture = raw.RawReplayCapture(tmp_path / "capture")
        with pytest.raises(
            source.SourcePreparationError, match="exact state capture adapter"
        ):
            capture.bind_target(selected | None)
        assert capture.gaps
        assert capture.target_witness is None
        assert capture.records["native_outcomes"] == {}
        assert capture.readiness is CheckStatus.UNVERIFIED
