# tests/extra_tests/converter/test_source_native_capture.py

"""Bounded original native callable capture; no SOURCE capability admission."""

from __future__ import annotations

from collections.abc import Iterator
import functools
import json
from pathlib import Path
import sys
import struct
from types import FunctionType, SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax2onnx._compat.jax import (
    Primitive,
    canonical_code_fingerprint,
    native_context_snapshot,
    require_native_context_restored,
    _NativeCaptureGraph,
    selected_native_capture,
    selected_native_capture_member,
)
from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.converter.precision_policy import (
    PrecisionPolicy,
    active_precision_policy,
    precision_policy_scope,
)
from tests.extra_tests.converter import source_precision_raw_replay as replay

selected_operation: Any = None


_OWNERS = {
    "reduce_window_sum_p": jax.lax.reduce_window_sum_p,
    "reduce_window_max_p": jax.lax.reduce_window_max_p,
    "logistic_p": jax.lax.logistic_p,
    "maximum": jnp.maximum,
    "minimum": jnp.minimum,
    "logical_xor": jnp.logical_xor,
}
_PRIMITIVES = ("reduce_window_sum_p", "reduce_window_max_p", "logistic_p")
_UFUNCS = ("maximum", "minimum", "logical_xor")


@pytest.fixture(autouse=True)
def _restore_caller() -> Iterator[None]:
    context = native_context_snapshot()
    policy, x64 = active_precision_policy(), jax.config.x64_enabled
    before = {name: selected_native_capture(owner) for name, owner in _OWNERS.items()}
    yield
    require_native_context_restored(context)
    assert active_precision_policy() is policy
    assert jax.config.x64_enabled is x64
    assert {
        name: selected_native_capture(owner) for name, owner in _OWNERS.items()
    } == before


def _member(name: str) -> str:
    return "bind" if name in _PRIMITIVES else "reduce"


def _options(name: str) -> dict[str, Any]:
    if name == "logistic_p":
        # Derive this supplemental fixture's version-specific native kwargs;
        # no inventory callable/options are edited or replayed by these tests.
        native = jax.make_jaxpr(jax.lax.logistic)(
            jax.ShapeDtypeStruct((4,), np.float32)
        ).jaxpr
        assert len(native.eqns) == 1
        assert native.eqns[0].primitive is _OWNERS[name]
        return dict(native.eqns[0].params)
    if name in _PRIMITIVES:
        return {
            "window_dimensions": (2,),
            "window_strides": (1,),
            "padding": ((0, 0),),
            "base_dilation": (1,),
            "window_dilation": (1,),
        }
    return {}


def _target(name: str, route: str) -> Any:
    owner, options = _OWNERS[name], _options(name)
    alias = getattr(owner, _member(name))  # Only these exact selected owners.
    if route == "partial":
        return functools.partial(alias, **options)
    if route == "default":

        def target(value: Any, operation: Any = alias) -> Any:
            return operation(value, **options)

        return target
    if route == "closure":

        def target(value: Any) -> Any:
            return alias(value, **options)

        return target
    if route == "owner-default":
        if name in _PRIMITIVES:

            def target(value: Any, operation: Any = owner) -> Any:
                return operation.bind(value, **options)

        else:

            def target(value: Any, operation: Any = owner) -> Any:
                return operation.reduce(value, **options)

        return target
    if route == "owner-closure":
        if name in _PRIMITIVES:

            def target(value: Any) -> Any:
                return owner.bind(value, **options)

        else:

            def target(value: Any) -> Any:
                return owner.reduce(value, **options)

        return target
    assert route == "module-path"
    if name == "reduce_window_sum_p":

        def target(value: Any) -> Any:
            return jax.lax.reduce_window_sum_p.bind(value, **options)

    elif name == "reduce_window_max_p":

        def target(value: Any) -> Any:
            return jax.lax.reduce_window_max_p.bind(value, **options)

    elif name == "logistic_p":

        def target(value: Any) -> Any:
            return jax.lax.logistic_p.bind(value, **options)

    elif name == "maximum":

        def target(value: Any) -> Any:
            return jnp.maximum.reduce(value)

    elif name == "minimum":

        def target(value: Any) -> Any:
            return jnp.minimum.reduce(value)

    else:

        def target(value: Any) -> Any:
            return jnp.logical_xor.reduce(value)

    return target


def _reject(value: Any) -> None:
    with pytest.raises(replay.SourcePreparationError) as caught:
        replay._capture_target_state(value)
    assert caught.value.category == "replay-lane-gap"


@pytest.mark.parametrize("name", tuple(_OWNERS))
def test_canonical_owners_and_saved_aliases_have_exact_json_capture(name: str) -> None:
    owner = _OWNERS[name]
    alias = getattr(owner, _member(name))
    owner_state, alias_state = selected_native_capture(owner), selected_native_capture(
        alias
    )
    assert owner_state is not None and alias_state is not None
    assert json.loads(json.dumps(owner_state, sort_keys=True)) == owner_state
    assert replay._capture_target_state(owner) == replay._capture_target_state(owner)
    assert replay._capture_target_state(alias) == replay._capture_target_state(
        getattr(owner, _member(name))
    )
    resolved = selected_native_capture_member(owner, _member(name))
    assert resolved is not None
    assert selected_native_capture(resolved) == alias_state


@pytest.mark.parametrize("caller_x64", [False, True])
@pytest.mark.parametrize("name", tuple(_OWNERS))
@pytest.mark.parametrize(
    "route",
    ["default", "closure", "partial", "owner-default", "owner-closure", "module-path"],
)
def test_native_state_survives_callable_routes_and_actual_jit_warmup(
    name: str, route: str, caller_x64: bool, tmp_path: Path
) -> None:
    with jax.enable_x64(caller_x64):
        target = _target(name, route)
        value = (
            np.asarray([True, False, True, True], np.bool_)
            if name == "logical_xor"
            else np.asarray([1, 4, -2, 3], np.float64 if caller_x64 else np.float32)
        )
        original = value.tobytes()
        capture = replay.RawReplayCapture(tmp_path / "native")
        capture.bind_target(target)
        before = replay._capture_target_state(target)

        def fresh(argument: Any) -> Any:
            return target(argument)

        # Keep both real native observations and arrays before state guards.
        for role, call in (
            ("eager", lambda: target(value)),
            ("fresh-jit", lambda: jax.jit(fresh)(value)),
        ):
            result = capture.invoke(role, call)
            if role in capture.native_values:
                capture.save_arrays(role, jax.tree.leaves(result))
        capture.save_document("observations", capture.records["native_outcomes"])
        assert set(capture.native_values) == {"eager", "fresh-jit"}
        assert not capture.native_errors and not capture.gaps
        assert replay._capture_target_state(target) == before
        capture.require_stable_target("after-native")
        bindings = tuple(vars(jnp)[key] for key in _UFUNCS)
        with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
            pass
        assert all(
            vars(jnp)[key] is original
            for key, original in zip(_UFUNCS, bindings, strict=True)
        )
        capture.require_stable_target("after-exporter-world")
        assert value.tobytes() == original


@pytest.mark.parametrize("name", tuple(_OWNERS))
@pytest.mark.parametrize("binding", ["public", "defining"])
def test_changed_public_or_defining_bindings_invalidate_saved_aliases(
    name: str, binding: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = _OWNERS[name]
    alias = getattr(owner, _member(name))
    if binding == "public":
        module = jax.lax if name in _PRIMITIVES else jnp
    else:
        module = sys.modules[
            (
                "jax._src.lax.lax"
                if name == "logistic_p"
                else (
                    "jax._src.lax.windowed_reductions"
                    if name in _PRIMITIVES
                    else "jax._src.numpy.ufuncs"
                )
            )
        ]
    monkeypatch.setattr(module, name, object())
    _reject(owner)
    _reject(alias)


@pytest.mark.parametrize("name", _PRIMITIVES)
def test_same_named_foreign_primitive_is_not_canonical(name: str) -> None:
    owner = _OWNERS[name]
    foreign = Primitive(owner.name)
    assert foreign is not owner and foreign.name == owner.name
    _reject(foreign)
    _reject(foreign.bind)


@pytest.mark.parametrize("name", _UFUNCS)
def test_fresh_ufunc_with_identical_equality_fields_is_not_canonical(name: str) -> None:
    owner = _OWNERS[name]
    props = vars(owner)["_ufunc__static_props"]
    foreign = type(owner)(
        props["func"],
        nin=props["nin"],
        nout=props["nout"],
        name=owner.__name__,
        nargs=props["nargs"],
        identity=props["identity"],
        **{
            field: props[field]
            for field in ("call", "reduce", "accumulate", "at", "reduceat")
        },
    )
    # Native equality deliberately omits callback fields. Never use that
    # equality or hash as an acceptance guard; canonical owner identity wins.
    assert foreign is not owner and type(foreign) is type(owner)
    assert foreign == owner  # Diagnostic witness of the native equality omission.
    vars(foreign)["_ufunc__static_props"]["reduce"] = lambda *args: None
    assert foreign == owner
    _reject(foreign)
    _reject(foreign.reduce)


@pytest.mark.parametrize(
    "field",
    [
        "name",
        "impl",
        "abstract_eval",
        "multiple_results",
        "call_primitive",
        "ref_primitive",
        "skip_canonicalization",
        "ref_allocating",
        "is_effectful",
        "bind",
        "bind_with_trace",
        "is_high",
        "to_lojax",
        "get_bind_params",
        "extra_state",
    ],
)
def test_primitive_behavior_and_extra_instance_state_are_not_hidden(
    field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = _OWNERS["reduce_window_sum_p"]
    alias = owner.bind
    old = getattr(owner, field, None)
    replacement = (
        not old
        if type(old) is bool
        else (
            "changed"
            if field == "name"
            else (
                {"mutable": [1]}
                if field == "extra_state"
                else lambda *args, **kwargs: None
            )
        )
    )
    # Preserve whether the inherited class member was absent on the instance.
    monkeypatch.setitem(vars(owner), field, replacement)
    _reject(owner)
    _reject(alias)


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
def test_every_ufunc_static_behavior_field_is_captured(
    field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = _OWNERS["maximum"]
    props = vars(owner)["_ufunc__static_props"]
    replacement = (
        props[field] + 1
        if type(props[field]) is int
        else False if field == "identity" else lambda *args, **kwargs: None
    )
    monkeypatch.setitem(props, field, replacement)
    _reject(owner)
    _reject(owner.reduce)


@pytest.mark.parametrize(
    "owner_name,field",
    [
        ("reduce_window_sum_p", "bind"),
        ("reduce_window_sum_p", "bind_with_trace"),
        ("reduce_window_sum_p", "multiple_results"),
        ("maximum", "__call__"),
        ("maximum", "reduce"),
        ("maximum", "_func"),
        ("maximum", "unknown_class_state"),
    ],
)
def test_effective_class_members_are_checked_for_saved_aliases(
    owner_name: str, field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = _OWNERS[owner_name]
    alias = getattr(owner, _member(owner_name))
    replacement = True if field == "multiple_results" else lambda *args, **kwargs: None
    monkeypatch.setattr(
        type(owner), field, replacement, raising=field != "unknown_class_state"
    )
    _reject(owner)
    _reject(alias)


@pytest.mark.parametrize(
    "mutation",
    ["code", "defaults", "kwdefaults", "attributes", "closure", "partial-keywords"],
)
def test_same_callable_identity_does_not_hide_python_executable_changes(
    mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = _OWNERS["reduce_window_sum_p"]
    evaluator = vars(owner)["abstract_eval"]
    if mutation == "code":
        monkeypatch.setattr(
            evaluator,
            "__code__",
            evaluator.__code__.replace(
                co_consts=(*evaluator.__code__.co_consts, "changed-native-code")
            ),
        )
    elif mutation == "defaults":
        monkeypatch.setattr(evaluator, "__defaults__", ("changed-default",))
    elif mutation == "kwdefaults":
        monkeypatch.setattr(evaluator, "__kwdefaults__", {"changed": True})
    elif mutation == "attributes":
        monkeypatch.setitem(vars(evaluator), "extra_state", {"mutable": [1]})
    else:
        cells = dict(
            zip(
                evaluator.__code__.co_freevars, evaluator.__closure__ or (), strict=True
            )
        )
        cell = cells["abstract_eval"]
        native_partial = cell.cell_contents
        assert type(native_partial) is functools.partial
        if mutation == "closure":
            monkeypatch.setattr(cell, "cell_contents", lambda *args, **kwargs: None)
        else:
            monkeypatch.setitem(native_partial.keywords, "changed-native-keyword", True)
    assert vars(owner)["abstract_eval"] is evaluator
    _reject(owner)


@pytest.mark.parametrize(
    "mutation",
    [
        "function-code",
        "function-attrs",
        "wrapper-field",
        "jit-info",
        "extra-field",
        "cache-miss-code",
        "cache-miss-attrs",
    ],
)
def test_native_jit_boundary_retains_python_function_and_configuration_state(
    mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = _OWNERS["maximum"]
    wrapper = vars(owner)["_ufunc__static_props"]["func"]
    wrapped = vars(wrapper)["_fun"]
    if mutation == "function-code":
        monkeypatch.setattr(
            wrapped,
            "__code__",
            wrapped.__code__.replace(
                co_consts=(*wrapped.__code__.co_consts, "changed-native-jit-code")
            ),
        )
    elif mutation == "function-attrs":
        monkeypatch.setitem(vars(wrapped), "extra_state", [1])
    elif mutation == "cache-miss-code":
        callback = wrapper._cache_miss
        monkeypatch.setattr(
            callback,
            "__code__",
            callback.__code__.replace(
                co_consts=(*callback.__code__.co_consts, "changed-cache-miss-code")
            ),
        )
    elif mutation == "cache-miss-attrs":
        monkeypatch.setitem(vars(wrapper._cache_miss), "extra_state", [1])
    elif mutation == "wrapper-field":
        monkeypatch.setitem(vars(wrapper), "__wrapped__", lambda *args, **kwargs: None)
    elif mutation == "jit-info":
        info = vars(wrapper)["_jit_info"]
        monkeypatch.setitem(
            vars(wrapper), "_jit_info", info._replace(keep_unused=not info.keep_unused)
        )
    else:
        monkeypatch.setitem(vars(wrapper), "extra_state", {"mutable": [1]})
    _reject(owner)


@pytest.mark.parametrize("raises", [False, True])
def test_native_outcome_is_captured_before_a_changed_state_gate(
    raises: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    owner = _OWNERS["reduce_window_sum_p"]
    target = _target("reduce_window_sum_p", "owner-closure")
    capture = replay.RawReplayCapture(tmp_path / "capture")
    capture.bind_target(target)
    value = np.asarray([1, 2, 3, 4], np.float32)

    def invoke() -> Any:
        monkeypatch.setattr(owner, "extra_state", ["retained"], raising=False)
        if raises:
            raise ValueError("record this native failure before checking state")
        return value

    result = capture.invoke("eager", invoke)
    if not raises:
        capture.save_arrays("eager", (result,))
    capture.save_document("native-outcome", capture.records["native_outcomes"])
    with pytest.raises(replay.SourcePreparationError):
        capture.require_stable_target("after-native")
    assert owner.extra_state == ["retained"]  # The adapter never resets it.
    assert ("eager" in capture.native_errors) is raises
    assert ("eager" in capture.native_values) is not raises
    assert capture.records["native_outcomes"]["eager"]["status"] == (
        "raised" if raises else "returned"
    )
    assert capture.records["after-native-state"]["status"] == "failed"
    assert capture.gaps and capture.records["native-outcome"]["status"] == "captured"
    assert not capture.preparation_complete
    assert capture.records["product_admission"] == "not-attempted"
    assert capture.records["raw_callback"]["status"] == "not-executed"


def test_unknown_owners_and_attribute_paths_remain_explicit_gaps() -> None:
    accessed = []

    class Unknown:
        @property
        def bind(self) -> Any:
            accessed.append("descriptor")
            raise AssertionError("unknown owner descriptor must not run")

    owner = Unknown()
    assert selected_native_capture(owner) is None
    assert selected_native_capture_member(owner, "bind") is None
    assert not accessed
    _reject(owner)
    _reject(SimpleNamespace(state=[1]))
    _reject(jax.lax.add_p)
    assert selected_native_capture(jnp.multiply) is None
    _reject(jnp.multiply)
    with pytest.raises(ValueError):
        selected_native_capture_member(_OWNERS["maximum"], "accumulate")


def test_unknown_class_static_dependency_does_not_become_a_module_path() -> None:
    class Foreign:
        state = [1]

        @staticmethod
        def operation(value: Any) -> Any:
            return value

    def target(value: Any) -> Any:
        return Foreign.operation(value)

    _reject(target)


@pytest.mark.parametrize("raises", [False, True])
def test_exporter_bindings_and_native_state_restore_on_both_exits(raises: bool) -> None:
    before = selected_native_capture(_OWNERS["maximum"])
    bindings = tuple(vars(jnp)[name] for name in _UFUNCS)
    try:
        with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
            if raises:
                raise ValueError("deliberate exporter-world exception")
    except ValueError as error:
        assert raises and str(error) == "deliberate exporter-world exception"
    assert all(
        vars(jnp)[name] is original
        for name, original in zip(_UFUNCS, bindings, strict=True)
    )
    assert selected_native_capture(_OWNERS["maximum"]) == before


def test_foreign_ufunc_public_spoof_cannot_enter_imported_function_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _OWNERS["maximum"]
    props = vars(owner)["_ufunc__static_props"]
    foreign = type(owner)(props["func"], 2, 1, name="maximum", identity=None)
    monkeypatch.setattr(foreign, "__module__", "jax.numpy", raising=False)
    monkeypatch.setattr(jnp, "maximum", foreign)

    def template(value: Any) -> Any:
        return selected_operation(value)

    target = FunctionType(template.__code__, {"selected_operation": foreign})
    _reject(target)


@pytest.mark.parametrize("binding", ["reducer", "implementation", "ufunc-type"])
def test_native_implementation_and_type_bindings_cannot_change_under_saved_owners(
    binding: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if binding == "reducer":
        monkeypatch.setattr(
            sys.modules["jax._src.numpy.reductions"], "_reduce_max", object()
        )
    elif binding == "implementation":
        monkeypatch.setattr(
            sys.modules["jax._src.dispatch"], "apply_primitive", object()
        )
    else:
        monkeypatch.setattr(jnp, "ufunc", object())
    _reject(
        _OWNERS["reduce_window_sum_p"].bind
        if binding == "implementation"
        else _OWNERS["maximum"]
    )


@pytest.mark.parametrize("kind", ["instance", "static-properties"])
def test_unknown_ufunc_fields_remain_explicit_gaps(
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _OWNERS["maximum"]
    if kind == "instance":
        monkeypatch.setattr(owner, "extra_state", [1], raising=False)
    else:
        monkeypatch.setitem(vars(owner)["_ufunc__static_props"], "extra_state", [1])
    _reject(owner)


@pytest.mark.parametrize(
    "fault", ["ufunc-identity", "ufunc-nout", "primitive-flag", "partial-keyword"]
)
def test_equal_python_bool_and_int_values_do_not_hide_native_state_changes(
    fault: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if fault == "ufunc-identity":
        owner = _OWNERS["logical_xor"]
        props = vars(owner)["_ufunc__static_props"]
        assert props["identity"] is False
        monkeypatch.setitem(props, "identity", 0)
    elif fault == "ufunc-nout":
        owner = _OWNERS["maximum"]
        props = vars(owner)["_ufunc__static_props"]
        assert type(props["nout"]) is int and props["nout"] == 1
        monkeypatch.setitem(props, "nout", True)
    elif fault == "primitive-flag":
        owner = _OWNERS["reduce_window_sum_p"]
        assert type(owner).multiple_results is False
        monkeypatch.setattr(type(owner), "multiple_results", 0)
    else:
        owner = _OWNERS["logistic_p"]
        evaluator = vars(owner)["abstract_eval"]
        cells = dict(
            zip(
                evaluator.__code__.co_freevars, evaluator.__closure__ or (), strict=True
            )
        )
        dtype_rule = cells["abstract_eval"].cell_contents.args[2]
        assert type(dtype_rule) is functools.partial
        assert dtype_rule.keywords["supports_narrow_ints"] is True
        monkeypatch.setitem(dtype_rule.keywords, "supports_narrow_ints", 1)
    _reject(owner)


def test_native_state_float_words_retain_signed_zero_and_nan_payloads() -> None:
    words = (
        0x0000000000000000,
        0x8000000000000000,
        0x7FF8000000000001,
        0x7FF8000000000002,
        0xFFF8000000000001,
    )
    snapshots = []
    for word in words:
        value = struct.unpack(">d", word.to_bytes(8, "big"))[0]
        snapshot = _NativeCaptureGraph().take(value)
        assert bytes.fromhex(snapshot["float64_be"]) == word.to_bytes(8, "big")
        snapshots.append(json.dumps(snapshot, sort_keys=True))
    assert len(set(snapshots)) == len(words)


@pytest.mark.parametrize(
    "fault",
    [
        "key-str",
        "annotation-key-hash",
        "metaclass-name",
        "primitive-key-hash",
        "static-key-hash",
        "jit-key-hash",
        "module-hash",
        "owner-getattribute",
    ],
)
def test_unowned_state_callbacks_do_not_run_before_the_replay_gap(
    fault: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = []

    class Key:
        def __str__(self) -> str:
            calls.append("str")
            raise AssertionError("unowned key formatting ran")

        def __hash__(self) -> int:
            if fault.endswith("key-hash") or fault == "module-hash":
                calls.append("hash")
            return object.__hash__(self)

    class ForeignMeta(type):
        def __getattribute__(cls, name: str) -> Any:
            if name in ("__module__", "__qualname__"):
                calls.append("metaclass")
                raise AssertionError("unowned class naming ran")
            return super().__getattribute__(name)

    class Foreign(metaclass=ForeignMeta):
        pass

    owner = (
        _OWNERS["maximum"]
        if fault in ("static-key-hash", "jit-key-hash")
        else _OWNERS["reduce_window_sum_p"]
    )
    target = (
        owner.reduce if fault in ("static-key-hash", "jit-key-hash") else owner.bind
    )
    capture = replay.RawReplayCapture(tmp_path / "capture")
    capture.bind_target(target)
    evaluator = (
        None
        if fault in ("static-key-hash", "jit-key-hash")
        else vars(owner)["abstract_eval"]
    )
    key = Key()
    retained = Foreign() if fault == "metaclass-name" else [1]
    if fault == "annotation-key-hash":
        changed = {key: "unowned annotation"}
        monkeypatch.setattr(evaluator, "__annotations__", changed)
        retained_items = list(changed.items())
    elif fault == "primitive-key-hash":
        changed = vars(owner)
        monkeypatch.setitem(changed, key, retained)
        retained_items = list(changed.items())
    elif fault == "static-key-hash":
        changed = vars(owner)["_ufunc__static_props"]
        monkeypatch.setitem(changed, key, retained)
        retained_items = list(changed.items())
    elif fault == "jit-key-hash":
        changed = vars(vars(owner)["_ufunc__static_props"]["func"])
        monkeypatch.setitem(changed, key, retained)
        retained_items = list(changed.items())
    elif fault == "module-hash":
        monkeypatch.setattr(evaluator, "__module__", key)
        changed = {"module": key}
        retained_items = list(changed.items())
    elif fault == "owner-getattribute":

        def unowned_attribute(self: Any, name: str) -> Any:
            calls.append("owner-descriptor")
            raise AssertionError("unowned owner descriptor ran")

        monkeypatch.setattr(
            type(owner), "__getattribute__", unowned_attribute, raising=False
        )
        changed = vars(type(owner))
        retained_items = list(changed.items())
    else:
        changed = vars(evaluator)
        monkeypatch.setitem(
            changed, "foreign_state" if fault == "metaclass-name" else key, retained
        )
        retained_items = list(changed.items())
    calls.clear()  # Disregard the fault injection's own ordinary dict insertion.
    assert not capture.check_target("unowned-state")
    assert not calls
    assert capture.target is target
    if fault == "module-hash":
        assert evaluator.__module__ is key
    actual_items = list(changed.items())
    assert len(actual_items) == len(retained_items)
    assert all(
        key_after is key_before and value_after is value_before
        for (key_after, value_after), (key_before, value_before) in zip(
            actual_items, retained_items, strict=True
        )
    )
    assert capture.records["unowned-state-state"]["status"] == "failed"
    assert capture.records["product_admission"] == "not-attempted"


def test_actual_class_callable_is_a_gap_without_rejecting_recursive_type_metadata(
    tmp_path: Path,
) -> None:
    invoked = []

    class Unknown:
        def __new__(cls) -> Any:
            invoked.append("constructor")
            raise AssertionError("capture must not instantiate a class target")

    metadata = replay._capture_target_state({"node_type": Unknown})
    assert metadata["mapping"][0][1] == {
        "type": f"{Unknown.__module__}.{Unknown.__qualname__}"
    }
    capture = replay.RawReplayCapture(tmp_path / "capture")
    with pytest.raises(replay.SourcePreparationError) as caught:
        capture.bind_target(Unknown)
    assert caught.value.category == "replay-lane-gap"
    assert capture.target is Unknown
    assert not invoked and capture.target_state is None
    assert capture.gaps and not capture.preparation_complete
    assert capture.records["product_admission"] == "not-attempted"
    assert capture.records["raw_callback"]["status"] == "not-executed"


def test_code_fingerprint_is_stable_when_native_execution_retains_nested_code() -> None:
    # ROOT's two-stack stdlib witness: retaining inner changes marshal reference
    # flags even though the original outer code and every constant stay intact.
    def outer(value: Any) -> Any:
        def inner() -> Any:
            return value

        return inner

    code = outer.__code__
    constants = tuple(code.co_consts)
    direct = canonical_code_fingerprint(code)
    raw = replay._capture_target_state(outer)
    graph = _NativeCaptureGraph()
    graph_root = graph.take(outer)
    graph_nodes = json.dumps(graph.nodes, sort_keys=True)
    retained = outer(1)
    assert outer.__code__ is code
    assert all(a is b for a, b in zip(constants, code.co_consts, strict=True))
    assert any(retained.__code__ is item for item in code.co_consts)
    assert canonical_code_fingerprint(code) == direct
    assert replay._capture_target_state(outer) == raw
    observed_graph = _NativeCaptureGraph()
    assert observed_graph.take(outer) == graph_root
    assert json.dumps(observed_graph.nodes, sort_keys=True) == graph_nodes


@pytest.mark.parametrize(
    "field",
    [
        "co_argcount",
        "co_posonlyargcount",
        "co_kwonlyargcount",
        "co_nlocals",
        "co_stacksize",
        "co_flags",
        "co_code",
        "co_consts",
        "co_names",
        "co_varnames",
        "co_filename",
        "co_name",
        "co_qualname",
        "co_firstlineno",
        "co_linetable",
        "co_exceptiontable",
        "co_freevars",
        "co_cellvars",
    ],
)
def test_code_fingerprint_retains_every_structural_constructor_field(
    field: str,
) -> None:
    def template(first: Any, /, second: Any, *, keyword: Any) -> Any:
        return first + second + keyword

    code = template.__code__
    replacements: dict[str, Any] = {
        "co_argcount": code.co_argcount - 1,
        "co_posonlyargcount": 0,
        "co_kwonlyargcount": 0,
        "co_nlocals": code.co_nlocals + 1,
        "co_stacksize": code.co_stacksize + 1,
        "co_flags": code.co_flags ^ 0x40,
        "co_code": code.co_code + b"\x09\x00",
        "co_consts": (*code.co_consts, "changed-constant"),
        "co_names": (*code.co_names, "changed_global"),
        "co_varnames": tuple("changed_" + name for name in code.co_varnames),
        "co_filename": code.co_filename + ".changed",
        "co_name": code.co_name + "_changed",
        "co_qualname": code.co_qualname + "_changed",
        "co_firstlineno": code.co_firstlineno + 1,
        "co_linetable": code.co_linetable + b"\x00",
        "co_exceptiontable": code.co_exceptiontable + b"\x00",
        "co_freevars": (*code.co_freevars, "changed_freevar"),
        "co_cellvars": (*code.co_cellvars, "changed_cellvar"),
    }
    changes = {field: replacements[field]}
    if field == "co_nlocals":
        # CodeType requires nlocals to equal the length of varnames.
        changes["co_varnames"] = (*code.co_varnames, "extra_local")
    changed = code.replace(**changes)
    assert canonical_code_fingerprint(changed) != canonical_code_fingerprint(code)


@pytest.mark.parametrize(
    "pair",
    [
        "bool-int",
        "none-ellipsis",
        "str-bytes",
        "tuple-frozenset",
        "signed-zero",
        "nan-payload",
        "complex-real-word",
        "complex-imag-word",
        "nested-code",
    ],
)
def test_code_constants_preserve_recursive_types_and_all_floating_words(
    pair: str,
) -> None:
    def template() -> None:
        return None

    nan1 = struct.unpack(">d", bytes.fromhex("7ff8000000000001"))[0]
    nan2 = struct.unpack(">d", bytes.fromhex("7ff8000000000002"))[0]
    pairs: dict[str, tuple[Any, Any]] = {
        "bool-int": (True, 1),
        "none-ellipsis": (None, Ellipsis),
        "str-bytes": ("a", b"a"),
        "tuple-frozenset": ((1,), frozenset((1,))),
        "signed-zero": (0.0, -0.0),
        "nan-payload": (nan1, nan2),
        "complex-real-word": (complex(0.0, nan1), complex(-0.0, nan1)),
        "complex-imag-word": (complex(nan1, nan1), complex(nan1, nan2)),
        "nested-code": (
            template.__code__,
            template.__code__.replace(co_consts=(None, "changed")),
        ),
    }
    left, right = pairs[pair]
    code = template.__code__
    assert canonical_code_fingerprint(code.replace(co_consts=((left,),))) != (
        canonical_code_fingerprint(code.replace(co_consts=((right,),)))
    )


def test_code_frozenset_constants_are_order_independent() -> None:
    def template() -> None:
        return None

    left = frozenset(("a", "b", 1, 2.0))
    right = frozenset(reversed(("a", "b", 1, 2.0)))
    code = template.__code__
    assert canonical_code_fingerprint(code.replace(co_consts=(left,))) == (
        canonical_code_fingerprint(code.replace(co_consts=(right,)))
    )


@pytest.mark.parametrize("constant", ["list", "dict", "foreign"])
def test_unowned_code_constants_are_rejected_without_invoking_callbacks(
    constant: str,
) -> None:
    invoked = []

    class Foreign:
        def __repr__(self) -> str:
            invoked.append("repr")
            raise AssertionError("unowned code constant formatting ran")

        def __str__(self) -> str:
            invoked.append("str")
            raise AssertionError("unowned code constant formatting ran")

        def __hash__(self) -> int:
            invoked.append("hash")
            raise AssertionError("unowned code constant hashing ran")

    def template() -> None:
        return None

    value = {"list": [], "dict": {}, "foreign": Foreign()}[constant]
    with pytest.raises(TypeError, match="constant type"):
        canonical_code_fingerprint(template.__code__.replace(co_consts=(value,)))
    assert not invoked


def test_code_constant_recursion_limit_remains_an_explicit_gap() -> None:
    def template() -> None:
        return None

    value: Any = 1
    for _ in range(50):
        value = (value,)
    with pytest.raises(ValueError, match="nesting"):
        canonical_code_fingerprint(template.__code__.replace(co_consts=(value,)))


@pytest.mark.parametrize("name", ["bool_", "int32", "int64", "float32", "bfloat16"])
def test_exact_scalar_alias_constructor_retains_its_existing_root_adapter(
    name: str,
    tmp_path: Path,
) -> None:
    target = vars(jnp)[name]
    binding = replay._jax_scalar_dtype_binding(target)
    assert binding is not None
    capture = replay.RawReplayCapture(tmp_path / "capture")
    capture.bind_target(target)
    assert capture.target is target and capture.target_state == binding
    capture.require_stable_target("unchanged-scalar-constructor")


def test_scalar_alias_constructor_rebinding_remains_a_root_state_gap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = jnp.float32
    capture = replay.RawReplayCapture(tmp_path / "capture")
    capture.bind_target(target)
    monkeypatch.setattr(jnp, "float32", jnp.float64)
    assert not capture.check_target("scalar-constructor-rebound")
    assert capture.target is target and capture.gaps


def test_code_integer_constants_have_no_decimal_string_conversion_limit() -> None:
    def template() -> None:
        return None

    large = 1 << 16000
    assert large > 10**4300
    code = template.__code__
    fingerprints = {
        canonical_code_fingerprint(code.replace(co_consts=(value,)))
        for value in (large, large + 1, -large)
    }
    assert len(fingerprints) == 3


def test_native_jit_configuration_snapshot_retains_all_actual_21_or_22_fields() -> None:
    owner = _OWNERS["maximum"]
    wrapper = vars(owner)["_ufunc__static_props"]["func"]
    info = vars(wrapper)["_jit_info"]
    snapshot = selected_native_capture(owner)
    assert snapshot is not None
    captured = snapshot["state"]["nodes"][str(id(info))]["fields"]
    assert set(captured) == set(info._fields)
    assert len(captured) in (21, 22)
    if "abstracted_axes" in info._fields:
        assert len(captured) == 22 and info.abstracted_axes is None
        assert captured["abstracted_axes"] is None
    else:
        assert len(captured) == 21


@pytest.mark.parametrize(
    "fault", ["extra", "missing", "order", "abstracted-axes", "field-lookalike"]
)
def test_native_jit_configuration_rejects_unowned_schema_or_abstracted_axes(
    fault: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _OWNERS["maximum"]
    wrapper = vars(owner)["_ufunc__static_props"]["func"]
    info = vars(wrapper)["_jit_info"]
    fields = info._fields
    invoked = []

    class ForeignField:
        def __eq__(self, other: Any) -> bool:
            invoked.append("eq")
            raise AssertionError("unowned JIT field equality ran")

    if fault == "abstracted-axes" and "abstracted_axes" in fields:
        monkeypatch.setitem(
            vars(wrapper), "_jit_info", info._replace(abstracted_axes=("B",))
        )
    else:
        altered = {
            "extra": (*fields, "unowned_field"),
            "missing": fields[:-1],
            "order": tuple(reversed(fields)),
            "abstracted-axes": (*fields, "abstracted_axes"),
            "field-lookalike": (ForeignField(), *fields[1:]),
        }[fault]
        monkeypatch.setattr(type(info), "_fields", altered)
    _reject(owner)
    assert not invoked


def test_original_primitive_true_bind_behavior_is_captured_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _OWNERS["reduce_window_sum_p"]
    # This is the older native execution seam; on newer classes it is an
    # unowned added field and must remain rejected rather than gaining a role.
    monkeypatch.setitem(vars(owner), "_true_bind", lambda *args, **kwargs: None)
    _reject(owner)
