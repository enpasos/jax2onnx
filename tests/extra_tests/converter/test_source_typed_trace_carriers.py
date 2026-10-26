# tests/extra_tests/converter/test_source_typed_trace_carriers.py

"""Typed host literals keep exact diagnostic lineage without changing inventory."""

from __future__ import annotations

import hashlib
from types import MemberDescriptorType, SimpleNamespace
from typing import Any

import jax
import numpy as np
import pytest

from jax2onnx._compat import jax as compat
from scripts.source_precision_inventory import canonical_contract
from tests.extra_tests.converter.source_precision_diagnostics import (
    typed_trace_manifest,
)


def _carrier(array: np.ndarray, weak: bool = False, *, cached: bool = True) -> Any:
    owner = compat._NATIVE_TYPED_ARRAY_TYPE
    if owner.__bases__ == (object,):
        return owner(array, weak_type=weak)
    if cached:
        return owner(
            array, aval=compat.ShapedArray(array.shape, array.dtype, weak_type=weak)
        )
    value = owner(array)
    value._weak_type = weak
    return value


def _state(value: Any) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    members = dict(compat._NATIVE_TYPED_ARRAY_MEMBERS)
    slots = []
    for name in members["__slots__"]:
        try:
            slots.append(members[name].__get__(value, type(value)))
        except AttributeError:
            slots.append(_state)
    aval = next((item for item in slots if type(item) is compat.ShapedArray), None)
    return tuple(slots), (
        () if aval is None else (aval.shape, aval.dtype, aval.weak_type)
    )


def _unchanged(value: Any, before: tuple[tuple[Any, ...], tuple[Any, ...]]) -> None:
    after = _state(value)
    for actual_group, expected_group in zip(after, before, strict=True):
        assert len(actual_group) == len(expected_group)
        assert all(
            actual is expected
            for actual, expected in zip(actual_group, expected_group, strict=True)
        )


def _native_aval_slot_state(aval: Any) -> tuple[Any, ...]:
    assert type(aval) is compat.ShapedArray
    assert all(
        not vars(parent).get("__slots__") for parent in compat.ShapedArray.__mro__[1:-1]
    )
    result = []
    for name in compat.ShapedArray.__slots__:
        if name == "__weakref__":
            continue
        slot = vars(compat.ShapedArray)[name]
        assert type(slot) is MemberDescriptorType
        assert slot.__objclass__ is compat.ShapedArray and slot.__name__ == name
        try:
            item = slot.__get__(aval, compat.ShapedArray)
        except AttributeError as error:
            assert type(error) is AttributeError
            result.append((name, slot, False, None))
        else:
            result.append((name, slot, True, item))
    return tuple(result)


def _assert_original_native_aval_untouched(audit: tuple[Any, ...]) -> None:
    original, state, array, dtype, shape, words = audit
    current = _native_aval_slot_state(original)
    assert len(current) == len(state)
    assert all(
        actual[0] == expected[0]
        and all(a is b for a, b in zip(actual[1:], expected[1:], strict=True))
        for actual, expected in zip(current, state, strict=True)
    )
    assert array.dtype == dtype and array.shape == shape
    assert array.tobytes() == words


def _corrupt_isolated_native_cached_aval(
    value: Any, field: str, malformed: Any
) -> tuple[Any, ...]:
    owner = compat._NATIVE_TYPED_ARRAY_TYPE
    assert type(value) is owner and owner.__bases__ == (np.ndarray,)
    cache_slot = dict(compat._NATIVE_TYPED_ARRAY_MEMBERS)["_aval"]
    assert type(cache_slot) is MemberDescriptorType
    assert cache_slot.__objclass__ is owner and cache_slot.__name__ == "_aval"
    original = cache_slot.__get__(value, owner)
    state = _native_aval_slot_state(original)
    valid = compat.native_typed_array_payload(value)
    assert valid is not None
    array, weak = valid
    audit = (original, state, array, array.dtype, array.shape, array.tobytes())
    # Bypass the constructor/interner only for this explicitly malformed fixture.
    # Every initialized canonical slot is copied; unset slots remain unset.
    isolated = object.__new__(compat.ShapedArray)
    assert type(isolated) is compat.ShapedArray and isolated is not original
    for _, slot, initialized, item in state:
        if initialized:
            slot.__set__(isolated, item)
    cache_slot.__set__(value, isolated)
    calibrated = compat.native_typed_array_payload(value)
    assert calibrated is not None
    assert type(calibrated[0]) is np.ndarray
    assert calibrated[0].dtype == array.dtype and calibrated[0].shape == array.shape
    assert calibrated[0].tobytes() == audit[-1] and calibrated[1] is weak
    _assert_original_native_aval_untouched(audit)
    selected = next(slot for name, slot, _, _ in state if name == field)
    selected.__set__(isolated, malformed)
    _assert_original_native_aval_untouched(audit)
    return audit


def _trace(value: Any, aval: Any) -> Any:
    # An actual native Literal plus a minimal diagnostic body isolates serialization
    # from tracing/evaluation and exercises both literal and const-payload paths.
    body = SimpleNamespace(
        constvars=(compat.Var(aval),),
        invars=(),
        outvars=(compat.Literal(value, aval),),
        eqns=(),
        effects=(),
    )
    return SimpleNamespace(jaxpr=body, consts=(value,))


def _expected(array: np.ndarray) -> dict[str, Any]:
    return {
        "array_dtype": array.dtype.name,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _backing(case: str) -> np.ndarray:
    if case == "scalar":
        return np.asarray(-0.0, np.float64)
    if case == "empty":
        return np.empty((0, 2), np.float32)
    if case == "rank-strided":
        return np.arange(12, dtype=np.float16).reshape(2, 2, 3).transpose(2, 0, 1)
    if case == "high-uint64":
        return np.asarray([2**63 + 5, 2**64 - 1], np.uint64)
    if case == "fp64-words":
        return np.asarray(
            [0x8000000000000000, 0x3FF0000000000001, 0x7FF8000000000042], np.uint64
        ).view(np.float64)
    if case == "bf16-words":
        return np.asarray([0x8000, 0x3F81, 0x7FC1], np.uint16).view(
            np.dtype("bfloat16")
        )
    assert case == "complex128-words"
    return np.asarray(
        [
            0x8000000000000000,
            0x3FF0000000000001,
            0x7FF8000000000042,
            0xFFF0000000000000,
        ],
        np.uint64,
    ).view(np.complex128)


@pytest.mark.parametrize(
    "case",
    [
        "scalar",
        "empty",
        "rank-strided",
        "high-uint64",
        "fp64-words",
        "bf16-words",
        "complex128-words",
    ],
)
@pytest.mark.parametrize("weak", [False, True])
@pytest.mark.parametrize("caller_x64", [False, True])
def test_typed_host_literal_has_exact_array_JSON_and_separate_aval_weakness(
    case: str, weak: bool, caller_x64: bool
) -> None:
    backing = _backing(case)
    value = _carrier(backing, weak)
    before = _state(value)
    words = backing.tobytes(order="C")
    aval = compat.ShapedArray(backing.shape, backing.dtype, weak_type=weak)
    with jax.enable_x64(caller_x64):
        manifest = typed_trace_manifest(_trace(value, aval))
    expected = _expected(backing)
    assert manifest["outputs"] == [
        {"literal": expected, "aval": (backing.dtype.name, backing.shape, weak)}
    ]
    assert manifest["const_payloads"] == [expected]
    assert manifest["consts"] == [(backing.dtype.name, backing.shape, weak)]
    assert not manifest["unresolved_parameters"]
    assert backing.tobytes(order="C") == words
    _unchanged(value, before)
    # The raw inventory serializer remains unchanged. Only diagnostic ownership
    # recognizes the legacy direct-slot carrier, not a global new contract form.
    unresolved: list[str] = []
    raw = canonical_contract(value, unresolved=unresolved)
    if type(value).__bases__ == (object,):
        assert raw == {"unresolved": "jax._src.literals.TypedNdArray"}
        assert unresolved
    else:
        assert raw == expected and not unresolved
    _unchanged(value, before)


@pytest.mark.parametrize("cached", [False, True])
def test_typed_host_literal_does_not_populate_or_replace_native_cache(
    cached: bool,
) -> None:
    backing = _backing("fp64-words")
    value = _carrier(backing, True, cached=cached)
    before = _state(value)
    if type(value).__bases__ == (np.ndarray,):
        assert (before[0][0] is None) is (not cached)
        if cached:
            # Native cached aval is authoritative while raw initializer flag is
            # False. The serializer must neither rewrite it nor fill empty cache.
            assert before[0][1] is False
    # The old direct-slot native layout has no cache; its actual val/weak slots
    # are preserved here rather than inventing an unavailable cached protocol.
    manifest = typed_trace_manifest(
        _trace(value, compat.ShapedArray(backing.shape, backing.dtype, weak_type=True))
    )
    assert manifest["outputs"][0]["literal"] == _expected(backing)
    assert manifest["outputs"][0]["aval"][2] is True
    _unchanged(value, before)


@pytest.mark.parametrize(
    "fault", ["weakness", "object-storage", "missing-slot", "foreign-storage-or-cache"]
)
def test_typed_host_literal_rejects_malformed_owned_payload_without_foreign_hooks(
    fault: str,
) -> None:
    backing = _backing("fp64-words")
    value = _carrier(backing)
    owner = type(value)
    legacy = owner.__bases__ == (object,)
    invoked = []

    class Foreign:
        def __array__(self, *args: Any, **kwargs: Any) -> Any:
            invoked.append("array")
            raise AssertionError("foreign payload conversion ran")

        def __getattribute__(self, name: str) -> Any:
            invoked.append(name)
            raise AssertionError("foreign payload reflection ran")

    original_audit = None
    if fault == "weakness":
        if legacy:
            value.weak_type = 1
        else:
            original_audit = _corrupt_isolated_native_cached_aval(value, "weak_type", 1)
    elif fault == "object-storage":
        value = (
            owner(np.asarray([object()], dtype=object), False)
            if legacy
            else owner(np.asarray([object()], dtype=object))
        )
    elif fault == "missing-slot":
        delattr(value, "val" if legacy else "_aval")
    elif legacy:
        value.val = Foreign()
    else:
        value._aval = Foreign()
    before = _state(value)
    words = backing.tobytes(order="C")
    with pytest.raises(TypeError, match="native typed-array"):
        typed_trace_manifest(
            _trace(value, compat.ShapedArray(backing.shape, backing.dtype))
        )
    assert not invoked and backing.tobytes(order="C") == words
    _unchanged(value, before)
    if original_audit is not None:
        _assert_original_native_aval_untouched(original_audit)


@pytest.mark.parametrize("mutation", ["native-attribute-hook", "numpy-conversion-hook"])
def test_typed_host_literal_checks_owner_before_reflection_or_conversion(
    mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    backing = _backing("fp64-words")
    value = _carrier(backing)
    before = _state(value)
    words = backing.tobytes(order="C")
    trace = _trace(value, compat.ShapedArray(backing.shape, backing.dtype))
    invoked = []

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        invoked.append("hook")
        raise AssertionError("unowned diagnostic hook ran")

    with monkeypatch.context() as changes:
        if mutation == "native-attribute-hook":
            changes.setattr(type(value), "__getattribute__", forbidden, raising=False)
        else:
            changes.setattr(np, "asarray", forbidden)
        with pytest.raises(TypeError, match="native typed-array"):
            typed_trace_manifest(trace)
        assert not invoked and backing.tobytes(order="C") == words
    _unchanged(value, before)
    assert typed_trace_manifest(trace)["outputs"][0]["literal"] == _expected(backing)
    _unchanged(value, before)


def test_actual_stripped_cache_remains_internal_to_typed_trace_json() -> None:
    backing = _backing("fp64-words")
    value = _carrier(backing, True)
    aval = compat.ShapedArray(backing.shape, backing.dtype, weak_type=True)
    trace = _trace(value, aval)
    baseline = typed_trace_manifest(trace)
    before = _state(value)
    assert typed_trace_manifest(trace) == baseline
    _unchanged(value, before)
    if compat.native_aval_metadata_schema() != "manual-stripped":
        with pytest.raises(TypeError, match="exact schema"):
            compat._require_manual_stripped_aval_cache(aval)
        return
    audit = _corrupt_isolated_native_cached_aval(value, "_stripped_weak_type", None)
    base = value._aval
    lazy = typed_trace_manifest(trace)
    assert lazy == baseline and base._stripped_weak_type is None
    strong = base.strip_weak_type()
    assert base._stripped_weak_type is strong and strong.weak_type is False
    assert strong._stripped_weak_type is None
    base_before, strong_before = _native_aval_slot_state(base), _native_aval_slot_state(
        strong
    )
    carrier_before = _state(value)
    for _ in range(3):
        assert typed_trace_manifest(trace) == baseline
        result = compat.native_typed_array_payload(value)
        assert result is not None and result[1] is True
        assert result[0].tobytes() == backing.tobytes()
        _unchanged(value, carrier_before)
        _assert_original_native_aval_untouched((base, base_before, *audit[2:]))
        _assert_original_native_aval_untouched((strong, strong_before, *audit[2:]))
        _assert_original_native_aval_untouched(audit)
