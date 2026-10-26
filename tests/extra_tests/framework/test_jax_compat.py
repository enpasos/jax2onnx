# tests/extra_tests/framework/test_jax_compat.py

from __future__ import annotations

from typing import Any
from types import MemberDescriptorType, SimpleNamespace

import numpy as np

import jax
import jax.numpy as jnp
import pytest
from pytest import MonkeyPatch

from jax2onnx._compat import jax as compat


def test_jax_compat_exports_core_types() -> None:
    prim = compat.Primitive("jax2onnx_test_compat")
    aval = compat.ShapedArray((2, 3), jnp.float32)

    assert prim.name == "jax2onnx_test_compat"
    assert aval.shape == (2, 3)
    assert aval.dtype == jnp.float32
    assert isinstance(compat.Literal, type)


def test_jax_compat_literal_falls_back_to_jax_src_core(
    monkeypatch: MonkeyPatch,
) -> None:
    from jax._src import core as jax_core_src

    monkeypatch.delattr(compat.jax_core_ext, "Literal", raising=False)
    monkeypatch.delattr(compat.jax_core, "Literal", raising=False)

    assert compat._resolve_literal_type() is jax_core_src.Literal


def test_jax_compat_exposes_not_mapped_alias() -> None:
    assert compat.ensure_batching_not_mapped_attr() is compat.NOT_MAPPED
    assert compat.batching.not_mapped is compat.NOT_MAPPED


def test_jax_compat_exposes_shape_equality_helper() -> None:
    assert compat.definitely_equal_shape((2, 3), (2, 3))
    assert not compat.definitely_equal_shape((2, 3), (2, 4))


def test_jax_compat_concrete_or_error_returns_static_value() -> None:
    assert compat.concrete_or_error(int, 7, "must be static") == 7


def test_fresh_var_like_uses_current_jax_constructor() -> None:
    aval = compat.ShapedArray((2, 3), jnp.float32)
    original = compat.Var(aval)

    fresh = compat.fresh_var_like(original)

    assert isinstance(fresh, compat.Var)
    assert fresh is not original
    assert fresh.aval is aval


def test_fresh_var_like_preserves_legacy_quantization_metadata(
    monkeypatch: MonkeyPatch,
) -> None:
    class LegacyVar:
        def __init__(
            self,
            aval: Any,
            initial_qdd: Any = None,
            final_qdd: Any = None,
        ) -> None:
            self.aval = aval
            self.initial_qdd = initial_qdd
            self.final_qdd = final_qdd

    original = LegacyVar("aval", initial_qdd="initial", final_qdd="final")
    monkeypatch.setattr(compat, "Var", LegacyVar)

    fresh = compat.fresh_var_like(original)

    assert fresh is not original
    assert fresh.aval == "aval"
    assert fresh.initial_qdd == "initial"
    assert fresh.final_qdd == "final"


def _scan_arity_of(fn: Any, *args: Any) -> tuple[int, int, int]:
    closed = jax.make_jaxpr(fn)(*args)
    for eqn in closed.jaxpr.eqns:
        if eqn.primitive.name == "scan":
            inner = eqn.params["jaxpr"]
            return compat.scan_arity(eqn.params, len(inner.jaxpr.invars))
    raise AssertionError("no scan equation found")


def test_scan_arity_without_xs() -> None:
    fn = lambda x: jax.lax.scan(  # noqa: E731
        lambda carry, _: (carry + 1, carry), x, xs=None, length=5
    )[1]
    assert _scan_arity_of(fn, jnp.float32(0.0)) == (0, 1, 0)


def test_scan_arity_with_consts_carry_and_xs() -> None:
    const = jnp.ones((2,))

    def fn(init: Any, xs: Any) -> Any:
        def body(carry: Any, x: Any) -> Any:
            a, b = carry
            return (a + const + x, b + 1.0), a.sum()

        return jax.lax.scan(body, init, xs)

    arity = _scan_arity_of(fn, (jnp.ones((2,)), jnp.float32(0.0)), jnp.ones((4, 2)))
    assert arity == (1, 2, 1)


def test_scan_arity_falls_back_to_legacy_params() -> None:
    # JAX <0.11 exposes integer counts instead of ``ft_in``.
    params = {"num_consts": 2, "num_carry": 1}
    assert compat.scan_arity(params, 5) == (2, 1, 2)


def test_scan_arity_rejects_inconsistent_arity() -> None:
    with pytest.raises(ValueError, match="Inconsistent Scan arity"):
        compat.scan_arity({"num_consts": 2, "num_carry": 1}, 2)


def test_selected_threefry_key_uses_the_native_layout_identity_and_words() -> None:
    key = jax.random.key(17, impl="threefry2x32")
    assert compat.is_selected_prng_key_dtype(key.dtype)
    assert compat.is_selected_threefry_impl(key.dtype._impl)
    words = np.asarray(compat.selected_prng_key_words(key))
    assert words.dtype == np.uint32 and words.shape == (2,)
    restored = compat.selected_prng_key_from_words(words)
    assert type(restored.dtype) is compat.PRNGKeyDType
    assert restored.dtype._impl is key.dtype._impl
    assert (
        np.asarray(compat.selected_prng_key_words(restored)).tobytes()
        == words.tobytes()
    )
    lookalike = key.dtype._impl._replace()
    assert lookalike == key.dtype._impl and lookalike is not key.dtype._impl
    assert not compat.is_selected_threefry_impl(lookalike)
    assert not compat.is_selected_prng_key_dtype(compat.PRNGKeyDType(lookalike))


@pytest.mark.parametrize("impl", ["rbg", "unsafe_rbg"])
def test_unselected_native_key_implementations_remain_rejected(impl: str) -> None:
    key = jax.random.key(17, impl=impl)
    assert not compat.is_selected_prng_key_dtype(key.dtype)
    with pytest.raises(ValueError, match="threefry"):
        compat.selected_prng_key_words(key)


def test_selected_key_adapter_rejects_batched_logical_keys_and_untyped_words() -> None:
    batched = jax.random.split(jax.random.key(17, impl="threefry2x32"), 2)
    with pytest.raises(ValueError, match="scalar"):
        compat.selected_prng_key_words(batched)
    with pytest.raises(ValueError, match="threefry"):
        compat.selected_prng_key_words(np.asarray([0, 17], np.uint32))


def test_native_context_snapshot_distinguishes_unset_from_equal_effective_none() -> (
    None
):
    configuration = compat._native_config.xla_metadata_context_manager
    previous = configuration.get_local()
    unset = compat._native_xla_metadata.config_ext.unset
    try:
        configuration.set_local(unset)
        assert configuration.value is None
        before = compat.native_context_snapshot()
        configuration.set_local(None)
        with pytest.raises(RuntimeError, match="restored"):
            compat.require_native_context_restored(before)
        configuration.set_local(unset)
        compat.require_native_context_restored(before)
    finally:
        configuration.set_local(previous)


def test_native_context_snapshot_checks_metadata_contents_without_foreign_equality() -> (
    None
):
    from jax.experimental.xla_metadata import set_xla_metadata

    with set_xla_metadata(checkpoint="original"):
        before = compat.native_context_snapshot()
        wrapper = compat._native_config.xla_metadata_context_manager.value
        wrapper.val["checkpoint"] = "changed"
        with pytest.raises(RuntimeError, match="restored"):
            compat.require_native_context_restored(before)
        wrapper.val["checkpoint"] = "original"
        compat.require_native_context_restored(before)


def test_native_recorded_context_preserves_real_manager_entry_policy_and_caller() -> (
    None
):
    configuration = compat._native_config.compute_on_context_manager
    previous = configuration.get_local()
    try:
        configuration.set_local("device")
        recorded = compat.current_native_eqn_context()
        configuration.set_local("device_host")
        before = compat.native_context_snapshot()
        if compat._native_context_protocol() == "mutable":
            with pytest.raises(NotImplementedError, match="Nesting"):
                with compat.recorded_native_context(recorded):
                    raise AssertionError("conflicting old native manager entered")
        else:
            with compat.recorded_native_context(recorded):
                assert compat.current_native_eqn_context().compute_type == "device"
        compat.require_native_context_restored(before)
    finally:
        configuration.set_local(previous)


@pytest.mark.parametrize(
    "fault", ["compute", "threefry", "metadata", "foreign-key", "foreign-value"]
)
def test_malformed_native_recorded_context_rejects_before_native_manager(
    fault: str,
) -> None:
    invoked = []

    class Key:
        def __hash__(self) -> int:
            return object.__hash__(self)

        def __str__(self) -> str:
            invoked.append("str")
            raise AssertionError("unowned context key formatting ran")

    class ForeignValue:
        def __eq__(self, other: Any) -> bool:
            invoked.append("eq")
            raise AssertionError("unowned context metadata equality ran")

        def __hash__(self) -> int:
            invoked.append("hash")
            raise AssertionError("unowned context metadata hashing ran")

    original = compat.current_native_eqn_context()
    context = object.__new__(type(original))
    for field in type(original).__slots__:
        if field != "__weakref__" and hasattr(original, field):
            object.__setattr__(context, field, getattr(original, field))
    field, value = {
        "compute": ("compute_type", object()),
        "threefry": ("threefry_partitionable", 1),
        "metadata": ("xla_metadata", SimpleNamespace()),
        "foreign-key": ("xla_metadata", {Key(): "value"}),
        "foreign-value": ("xla_metadata", {"key": ForeignValue()}),
    }[fault]
    object.__setattr__(context, field, value)
    before = compat.native_context_snapshot()
    with pytest.raises(TypeError):
        with compat.recorded_native_context(context):
            raise AssertionError("malformed context entered native manager")
    compat.require_native_context_restored(before)
    assert not invoked


def test_native_scan_schema_is_fixed_and_cannot_follow_extra_installed_fields(
    monkeypatch: MonkeyPatch,
) -> None:
    expected = compat.native_scan_parameter_names()
    assert {"jaxpr", "reverse", "length", "unroll"} < expected
    monkeypatch.setattr(
        compat,
        "_NATIVE_SCAN_ABSTRACT_PARAMETERS",
        (*compat._NATIVE_SCAN_ABSTRACT_PARAMETERS, "unowned"),
    )
    with pytest.raises(TypeError, match="schema"):
        compat.native_scan_parameter_names()


def test_native_multiply_parameter_schema_matches_actual_native_equation() -> None:
    equation = jax.make_jaxpr(jax.lax.mul)(
        jax.ShapeDtypeStruct((2,), np.float32),
        jax.ShapeDtypeStruct((2,), np.float32),
    ).jaxpr.eqns[0]
    fields = compat.native_mul_parameter_names()
    assert set(equation.params) == fields
    assert all(value is None for value in equation.params.values())


@pytest.mark.parametrize("current", [False, True])
def test_native_multiply_parameter_schemas_keep_exact_optional_result_role(
    current: bool, monkeypatch: MonkeyPatch
) -> None:
    import inspect

    parameter = inspect.Parameter
    signature = (
        ("x", parameter.POSITIONAL_OR_KEYWORD, parameter.empty),
        ("y", parameter.POSITIONAL_OR_KEYWORD, parameter.empty),
    )
    if current:
        signature += (("out_dtype", parameter.KEYWORD_ONLY, None),)
    monkeypatch.setattr(compat, "_NATIVE_MUL_SIGNATURE", signature)
    assert compat.native_mul_parameter_names() == (
        frozenset({"out_dtype"}) if current else frozenset()
    )


@pytest.mark.parametrize("fault", ["missing", "unknown", "positional", "default"])
def test_native_multiply_unknown_signature_fields_kinds_and_defaults_reject(
    fault: str, monkeypatch: MonkeyPatch
) -> None:
    import inspect

    parameter = inspect.Parameter
    operands = (
        ("x", parameter.POSITIONAL_OR_KEYWORD, parameter.empty),
        ("y", parameter.POSITIONAL_OR_KEYWORD, parameter.empty),
    )
    signature = {
        "missing": operands[:1],
        "unknown": (*operands, ("unowned", parameter.KEYWORD_ONLY, None)),
        "positional": (*operands, ("out_dtype", parameter.POSITIONAL_OR_KEYWORD, None)),
        "default": (*operands, ("out_dtype", parameter.KEYWORD_ONLY, False)),
    }[fault]
    monkeypatch.setattr(compat, "_NATIVE_MUL_SIGNATURE", signature)
    with pytest.raises(TypeError, match="multiply parameter schema"):
        compat.native_mul_parameter_names()


@pytest.mark.parametrize("enabled", [False, True])
def test_native_logistic_metadata_matches_frozen_recipe_without_native_mutation(
    enabled: bool,
) -> None:
    from scripts.source_precision_inventory import canonical_contract

    function = compat._NATIVE_LOGISTIC_FUNCTION
    before, _ = compat._native_logistic_metadata_state()
    caller = bool(jax.config.jax_enable_x64)
    with jax.enable_x64(enabled):
        unresolved: list[str] = []
        assert canonical_contract(function, unresolved=unresolved) == {
            "callable": "jax.lax.logistic",
            "kwdefaults": {"accuracy": None},
        }
        assert not unresolved
        assert bool(jax.config.jax_enable_x64) is enabled
        after, _ = compat._native_logistic_metadata_state()
        assert after == before
    assert bool(jax.config.jax_enable_x64) is caller


@pytest.mark.parametrize(
    "fault",
    [
        "code",
        "name",
        "qualname",
        "module",
        "defaults",
        "kwdefaults",
        "attributes",
        "annotations",
        "signature",
        "wrapped",
    ],
)
def test_native_logistic_metadata_rejects_actual_function_state_mutations(
    fault: str,
    monkeypatch: MonkeyPatch,
) -> None:
    import inspect
    from scripts.source_precision_inventory import canonical_contract

    function = compat._NATIVE_LOGISTIC_FUNCTION
    assert callable(function)
    expected = canonical_contract(function, unresolved=[])

    def replacement(x: Any, accuracy: Any = None) -> Any:
        return x

    attribute, value = {
        "code": ("__code__", replacement.__code__),
        "name": ("__name__", "other"),
        "qualname": ("__qualname__", "other"),
        "module": ("__module__", "other"),
        "defaults": ("__defaults__", (False,)),
        "kwdefaults": ("__kwdefaults__", {"accuracy": False}),
        "annotations": ("__annotations__", {"x": "changed"}),
        "attributes": ("unowned", object()),
        "signature": ("__signature__", inspect.signature(function)),
        "wrapped": ("__wrapped__", function),
    }[fault]
    with monkeypatch.context() as patch:
        if fault in {"attributes", "signature", "wrapped"}:
            patch.setitem(vars(function), attribute, value)
        else:
            patch.setattr(function, attribute, value)
        with pytest.raises((TypeError, ValueError), match="native logistic"):
            canonical_contract(function, unresolved=[])
        assert getattr(function, attribute) is value
    assert canonical_contract(function, unresolved=[]) == expected


@pytest.mark.parametrize("fault", ["public", "defining", "both", "missing"])
def test_native_logistic_metadata_rejects_binding_drift_without_repair(
    fault: str,
    monkeypatch: MonkeyPatch,
) -> None:
    from scripts.source_precision_inventory import canonical_contract

    function = compat._NATIVE_LOGISTIC_FUNCTION
    expected = canonical_contract(function, unresolved=[])

    def foreign(x: Any, accuracy: Any = None) -> Any:
        return x

    modules = {
        "public": (compat._native_lax,),
        "defining": (compat._native_lax_impl,),
        "both": (compat._native_lax, compat._native_lax_impl),
        "missing": (compat._native_lax,),
    }[fault]
    with monkeypatch.context() as patch:
        for module in modules:
            if fault == "missing":
                patch.delitem(vars(module), "logistic")
            else:
                patch.setitem(vars(module), "logistic", foreign)
        with pytest.raises(TypeError, match="native logistic"):
            canonical_contract(function, unresolved=[])
        if fault != "missing":
            with pytest.raises(TypeError, match="native logistic"):
                canonical_contract(foreign, unresolved=[])
        for module in modules:
            if fault == "missing":
                assert "logistic" not in vars(module)
            else:
                assert vars(module)["logistic"] is foreign
    assert canonical_contract(function, unresolved=[]) == expected


def test_native_logistic_metadata_rejects_same_name_signature_impostor() -> None:
    import inspect
    from scripts.source_precision_inventory import canonical_contract

    def impostor(x: Any, accuracy: Any = None) -> Any:
        return x

    impostor.__module__ = "jax.lax"
    impostor.__qualname__ = "logistic"
    impostor.__name__ = "logistic"
    vars(impostor)["__signature__"] = inspect.signature(compat._native_lax.logistic)
    with pytest.raises(TypeError, match="foreign function"):
        canonical_contract(impostor, unresolved=[])
    assert vars(impostor)["__signature__"] is not None
    assert (
        compat.native_logistic_metadata_contract(compat._NATIVE_LOGISTIC_FUNCTION)
        is not None
    )


@pytest.mark.parametrize("keyword", [False, True])
def test_native_logistic_default_layout_is_exact_and_does_not_call_signature_hooks(
    keyword: bool,
) -> None:
    calls: list[str] = []

    class Unowned:
        def __getattribute__(self, name: str) -> Any:
            calls.append(name)
            raise AssertionError("signature hook executed")

    def positional(x: Any, accuracy: Any = None) -> Any:
        return x

    def keywords(x: Any, *, accuracy: Any = None) -> Any:
        return x

    function = keywords if keyword else positional
    vars(function)["__signature__"] = Unowned()
    assert compat._native_logistic_default_layout(function) == (
        "keyword" if keyword else "positional"
    )
    assert not calls
    # Matching the native layout never grants this foreign function ownership.
    assert compat.native_logistic_metadata_contract(function) is None


@pytest.mark.parametrize(
    "fault",
    ["name", "extra", "positional-only", "varargs", "required", "default"],
)
def test_native_logistic_unknown_argument_and_default_layouts_reject(
    fault: str,
) -> None:
    def wrong_name(x: Any, tolerance: Any = None) -> Any:
        return x

    def extra(x: Any, accuracy: Any = None, other: Any = None) -> Any:
        return x

    def positional_only(x: Any, /, accuracy: Any = None) -> Any:
        return x

    def varargs(x: Any, accuracy: Any = None, *other: Any) -> Any:
        return x

    def required(x: Any, accuracy: Any) -> Any:
        return x

    def default(x: Any, *, accuracy: Any = False) -> Any:
        return x

    function = {
        "name": wrong_name,
        "extra": extra,
        "positional-only": positional_only,
        "varargs": varargs,
        "required": required,
        "default": default,
    }[fault]
    with pytest.raises(TypeError, match="native logistic.*schema"):
        compat._native_logistic_default_layout(function)


def test_native_logistic_metadata_keeps_unrelated_defaults_and_explicit_partial_args() -> (
    None
):
    import functools
    from scripts.source_precision_inventory import canonical_contract

    def unrelated(x: Any, accuracy: Any = None, *, other: bool = False) -> Any:
        return x

    assert canonical_contract(unrelated, unresolved=[]) == {
        "callable": f"{unrelated.__module__}.{unrelated.__qualname__}",
        "defaults": {"tuple": [None]},
        "kwdefaults": {"other": False},
    }
    function = compat._native_lax.logistic
    assert function is compat._NATIVE_LOGISTIC_FUNCTION
    # These are serializer observations; no native call or accuracy substitution.
    for explicit in (None, "unchanged-explicit-value"):
        recipe = functools.partial(function, 7, accuracy=explicit)
        assert canonical_contract(recipe, unresolved=[]) == {
            "partial": {
                "callable": "jax.lax.logistic",
                "kwdefaults": {"accuracy": None},
            },
            "args": {"tuple": [7]},
            "kwargs": {"accuracy": explicit},
        }
        assert recipe.func is function
        assert recipe.args == (7,)
        assert recipe.keywords == {"accuracy": explicit}


def test_native_logistic_metadata_uses_original_row_hash_guard_and_pending_criterion(
    monkeypatch: MonkeyPatch,
) -> None:
    import json
    from scripts import source_precision_probe as probe
    from scripts.source_precision_inventory import DEFAULT_BASELINE, raw_registrations

    inventory = json.loads(DEFAULT_BASELINE.read_text())
    plan = probe.build_probe_plan(inventory)
    row = next(
        row
        for row in plan.rows
        if row.origin == "plugin::logistic" and row.ordinal == 0
    )
    assert (
        row.contract_sha256
        == "c673276c11a5fe9a1f5374c1eaacdcb19fda915c7e2012c2a4ef8f7799be3277"
    )
    assert row.opset == 23 and row.criteria is None
    registrations = {origin: metadata for origin, _, metadata in raw_registrations()[0]}
    raw = registrations[row.origin]["testcases"][row.ordinal]
    resolved = probe._resolve_case(row, registrations)
    assert resolved["callable"] is raw["callable"] is compat._NATIVE_LOGISTIC_FUNCTION
    assert resolved["input_shapes"] is raw["input_shapes"]
    assert resolved["post_check_onnx_graph"] is raw["post_check_onnx_graph"]
    before = dict(raw)
    with monkeypatch.context() as patch:
        patch.setitem(raw, "input_shapes", [(4, 3)])
        with pytest.raises(ValueError, match="contract changed after"):
            probe._resolve_case(row, registrations)
        assert raw["input_shapes"] == [(4, 3)]
    assert raw == before
    assert probe._resolve_case(row, registrations)["callable"] is raw["callable"]


def _native_typed_constant(array: np.ndarray, weak: bool = False) -> Any:
    owner = compat._NATIVE_TYPED_ARRAY_TYPE
    if owner.__bases__ == (object,):
        return owner(array, weak_type=weak)
    return owner(
        array, aval=compat.ShapedArray(array.shape, array.dtype, weak_type=weak)
    )


def _native_typed_constant_state(value: Any) -> tuple[Any, ...]:
    members = dict(compat._NATIVE_TYPED_ARRAY_MEMBERS)
    result = []
    for name in members["__slots__"]:
        try:
            result.append(members[name].__get__(value, type(value)))
        except AttributeError:
            result.append(_native_typed_constant_state)
    return tuple(result)


def _assert_native_typed_constant_state(value: Any, before: tuple[Any, ...]) -> None:
    assert all(
        actual is expected
        for actual, expected in zip(
            _native_typed_constant_state(value), before, strict=True
        )
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


@pytest.mark.parametrize(
    "dtype", ["bool", "uint64", "float32", "float64", "bfloat16", "complex128"]
)
@pytest.mark.parametrize("weak", [False, True])
@pytest.mark.parametrize("caller_x64", [False, True])
def test_native_typed_constant_adapter_preserves_storage_and_weakness(
    dtype: str, weak: bool, caller_x64: bool
) -> None:
    backing = np.full((2, 3), True if dtype == "bool" else -0.0, dtype)
    value = _native_typed_constant(backing, weak)
    before = _native_typed_constant_state(value)
    words = backing.tobytes()
    with jax.enable_x64(caller_x64):
        result = compat.native_typed_array_payload(value)
    assert result is not None
    array, actual_weak = result
    assert type(array) is np.ndarray
    assert array.dtype == backing.dtype and array.shape == backing.shape
    assert array.tobytes() == words and actual_weak is weak
    if type(value).__bases__ == (object,):
        assert array is backing
    _assert_native_typed_constant_state(value, before)
    assert backing.tobytes() == words


def test_native_typed_constant_cache_empty_and_cached_weakness_precedence() -> None:
    backing = np.asarray([-0.0, np.nan], np.float64)
    owner = compat._NATIVE_TYPED_ARRAY_TYPE
    if owner.__bases__ == (object,):
        values = [_native_typed_constant(backing, weak) for weak in (False, True)]
    else:
        empty = owner(backing)
        assert (
            dict(compat._NATIVE_TYPED_ARRAY_MEMBERS)["_aval"].__get__(empty, owner)
            is None
        )
        cached = _native_typed_constant(backing, True)
        # __new__(aval=...) leaves the raw initializer flag False. The cached
        # native aval remains the authoritative effective weakness.
        assert (
            dict(compat._NATIVE_TYPED_ARRAY_MEMBERS)["_weak_type"].__get__(
                cached, owner
            )
            is False
        )
        values = [empty, cached]
    for value, weak in zip(values, (False, True), strict=True):
        before = _native_typed_constant_state(value)
        result = compat.native_typed_array_payload(value)
        assert result is not None and result[1] is weak
        assert result[0].tobytes() == backing.tobytes()
        _assert_native_typed_constant_state(value, before)


@pytest.mark.parametrize(
    "fault", ["weakness", "object-storage", "missing-slot", "foreign-backing-or-aval"]
)
def test_native_typed_constant_rejects_malformed_owned_storage_without_hooks(
    fault: str,
) -> None:
    backing = np.asarray([-0.0, np.nan], np.float32)
    value = _native_typed_constant(backing)
    assert compat.native_typed_array_payload(value) is not None
    owner = type(value)
    legacy = owner.__bases__ == (object,)
    invoked = []
    isolation = None

    class Foreign:
        def __array__(self, *args: Any, **kwargs: Any) -> Any:
            invoked.append("array")
            raise AssertionError("foreign storage conversion ran")

        def __getattribute__(self, name: str) -> Any:
            invoked.append(name)
            raise AssertionError("foreign storage reflection ran")

    if fault == "weakness":
        if legacy:
            value.weak_type = 1
        else:
            isolation = _corrupt_isolated_native_cached_aval(value, "weak_type", 1)
    elif fault == "object-storage":
        object_array = np.asarray([object(), object()], dtype=object)
        value = owner(object_array, False) if legacy else owner(object_array)
    elif fault == "missing-slot":
        delattr(value, "val" if legacy else "_aval")
    elif legacy:
        value.val = Foreign()
    else:
        value._aval = Foreign()
    before = _native_typed_constant_state(value)
    words = backing.tobytes()
    with pytest.raises(TypeError, match="native typed-array"):
        compat.native_typed_array_payload(value)
    assert not invoked and backing.tobytes() == words
    _assert_native_typed_constant_state(value, before)
    if isolation is not None:
        _assert_original_native_aval_untouched(isolation)


@pytest.mark.parametrize("kind", ["foreign", "subclass"])
def test_native_typed_constant_adapter_never_calls_unowned_array_or_attribute_hooks(
    kind: str,
) -> None:
    backing = np.asarray([-0.0, np.nan], np.float32)
    valid = _native_typed_constant(backing)
    assert compat.native_typed_array_payload(valid) is not None
    invoked = []

    class Foreign:
        def __array__(self, *args: Any, **kwargs: Any) -> Any:
            invoked.append("array")
            raise AssertionError("unowned array hook ran")

        def __getattribute__(self, name: str) -> Any:
            invoked.append(name)
            raise AssertionError("unowned attribute hook ran")

    class Subclass(compat._NATIVE_TYPED_ARRAY_TYPE):
        pass

    if kind == "foreign":
        value = Foreign()
        assert compat.native_typed_array_payload(value) is None
    else:
        if type(valid).__bases__ == (object,):
            value = Subclass(backing, False)
        else:
            value = Subclass(backing)
        Subclass.__array__ = Foreign.__array__
        Subclass.__getattribute__ = Foreign.__getattribute__
        with pytest.raises(TypeError, match="subclasses"):
            compat.native_typed_array_payload(value)
    assert not invoked
    assert backing.tobytes() == np.asarray([-0.0, np.nan], np.float32).tobytes()


@pytest.mark.parametrize(
    "fault",
    ["binding", "member", "conversion-code", "conversion-defaults", "numpy-binding"],
)
def test_native_typed_constant_owner_and_invoked_conversion_remain_exact(
    fault: str, monkeypatch: MonkeyPatch
) -> None:
    backing = np.asarray([-0.0, np.nan], np.float32)
    value = _native_typed_constant(backing)
    assert compat.native_typed_array_payload(value) is not None
    before = _native_typed_constant_state(value)
    words = backing.tobytes()
    owner = type(value)
    invoked = []

    def foreign_array(self: Any, dtype: Any = None, copy: Any = None) -> Any:
        invoked.append("array")
        raise AssertionError("changed conversion implementation ran")

    with monkeypatch.context() as changes:
        if fault == "binding":
            changes.setattr(compat._native_literals, "TypedNdArray", object)
        elif fault == "member":
            slot = "val" if owner.__bases__ == (object,) else "_aval"
            changes.setattr(owner, slot, property(foreign_array))
        elif fault == "numpy-binding":
            changes.setattr(np, "asarray", foreign_array)
        elif owner.__bases__ == (object,):
            conversion = compat._NATIVE_TYPED_ARRAY_ARRAY_FUNCTION
            if fault == "conversion-code":
                # Same function identity, no closure: identity-only checks
                # would miss this exact downstream invoked-method mutation.
                def replacement(self: Any, dtype: Any = None, copy: Any = None) -> Any:
                    raise AssertionError("changed native conversion code ran")

                changes.setattr(conversion, "__code__", replacement.__code__)
            else:
                changes.setattr(conversion, "__defaults__", (np.float64, False))
        else:
            # Current ndarray carriers have no Python __array__ method to pin;
            # introducing one changes their exact native conversion ownership.
            changes.setattr(owner, "__array__", foreign_array, raising=False)
        with pytest.raises(TypeError, match="native typed-array"):
            compat.native_typed_array_payload(value)
        assert not invoked and backing.tobytes() == words
        _assert_native_typed_constant_state(value, before)
    assert compat.native_typed_array_payload(value) is not None
    _assert_native_typed_constant_state(value, before)


@pytest.mark.parametrize("mutation", ["dtype", "shape", "placement"])
def test_native_typed_constant_cached_aval_geometry_or_legacy_direct_storage(
    mutation: str,
) -> None:
    backing = np.asarray([-0.0, np.nan], np.float32)
    value = _native_typed_constant(backing)
    valid = compat.native_typed_array_payload(value)
    assert valid is not None
    if type(value).__bases__ == (object,):
        # This native layout has direct val/weak slots and no cached aval role.
        assert valid[0] is backing and valid[1] is False
        return
    aval = value._aval
    isolation = None
    if mutation == "dtype":
        value._aval = aval.update(dtype=np.float64)
    elif mutation == "shape":
        value._aval = aval.update(shape=(1,))
    else:
        field, varying = compat.native_manual_axis_metadata(frozenset({"test_axis"}))
        isolation = _corrupt_isolated_native_cached_aval(value, field, varying)
    before = _native_typed_constant_state(value)
    words = backing.tobytes()
    with pytest.raises(TypeError, match="cached aval|native signature"):
        compat.native_typed_array_payload(value)
    _assert_native_typed_constant_state(value, before)
    assert backing.tobytes() == words

    if isolation is not None:
        _assert_original_native_aval_untouched(isolation)


def test_native_typed_constant_nested_foreign_dtype_or_legacy_backing_has_no_hooks() -> (
    None
):
    backing = np.asarray([-0.0, np.nan], np.float32)
    value = _native_typed_constant(backing)
    assert compat.native_typed_array_payload(value) is not None
    invoked = []

    class Foreign:
        def __array__(self, *args: Any, **kwargs: Any) -> Any:
            invoked.append("array")
            raise AssertionError("nested foreign dtype or storage conversion ran")

        def __getattribute__(self, name: str) -> Any:
            invoked.append(name)
            raise AssertionError("nested foreign dtype or storage reflection ran")

    foreign = Foreign()
    isolation = None
    if type(value).__bases__ == (object,):
        # The native legacy wrapper has no cached aval. Exercise its actual
        # direct-storage boundary instead of inventing a synthetic cache.
        value.val = foreign
        cached_aval = None
        cached_state = None
    else:
        isolation = _corrupt_isolated_native_cached_aval(value, "dtype", foreign)
        cached_aval = value._aval
        cached_state = (
            cached_aval.shape,
            cached_aval.dtype,
            cached_aval.weak_type,
        )
    before = _native_typed_constant_state(value)
    words = backing.tobytes()
    with pytest.raises(TypeError, match="native typed-array"):
        compat.native_typed_array_payload(value)
    assert not invoked and backing.tobytes() == words
    _assert_native_typed_constant_state(value, before)
    if cached_aval is not None:
        assert cached_state is not None
        assert all(
            actual is expected
            for actual, expected in zip(
                (cached_aval.shape, cached_aval.dtype, cached_aval.weak_type),
                cached_state,
                strict=True,
            )
        )

    if isolation is not None:
        _assert_original_native_aval_untouched(isolation)


def test_type_promotion_error_is_exact_native_exception_and_restores_context() -> None:
    from jax._src.dtypes import TypePromotionError as NativeTypePromotionError

    assert compat.TypePromotionError is NativeTypePromotionError
    previous = jax.config.jax_numpy_dtype_promotion
    with jax.numpy_dtype_promotion("strict"):
        with pytest.raises(compat.TypePromotionError) as caught:
            jnp.result_type(np.dtype(np.int32), np.dtype(np.float32))
        assert type(caught.value) is compat.TypePromotionError
    assert jax.config.jax_numpy_dtype_promotion == previous


def test_native_aval_declaration_is_owned_and_unknown_sequences_never_fall_back(
    monkeypatch: MonkeyPatch,
) -> None:
    owner = compat.ShapedArray
    declaration = vars(owner)["__slots__"]
    sequence = tuple(declaration)
    common = (
        "shape",
        "dtype",
        "weak_type",
        "sharding",
        "manual_axis_type",
        "memory_space",
    )
    expected = {
        ("shape", "sharding", "vma", "memory_space"): "vma",
        (*common, "__weakref__"): "manual",
        (*common, "layout", "_stripped_weak_type", "__weakref__"): "manual-layout",
        (*common, "_stripped_weak_type", "__weakref__"): "manual-stripped",
    }
    assert compat.native_aval_metadata_schema() == expected[sequence]
    # Even an equal replacement declaration must not replace the native owner.
    with monkeypatch.context() as change:
        change.setattr(owner, "__slots__", list(sequence))
        with pytest.raises(TypeError, match="slot declaration changed"):
            compat.native_aval_metadata_schema()
    invalid = [sequence[::-1], (*sequence, "unknown"), sequence[:-1]]
    if "_stripped_weak_type" in sequence:
        # In 0.11 this would otherwise masquerade as the older manual tuple.
        invalid.append(
            tuple(name for name in sequence if name != "_stripped_weak_type")
        )
    invoked = []

    class Foreign:
        def __eq__(self, other: Any) -> bool:
            invoked.append("equality")
            raise AssertionError("foreign declaration equality ran")

        def __iter__(self) -> Any:
            invoked.append("iteration")
            raise AssertionError("foreign declaration iteration ran")

    invalid.append((*sequence[:-1], Foreign()))
    for malformed in invalid:
        if type(declaration) is list:
            try:
                declaration[:] = malformed
                with pytest.raises(TypeError, match="slot declaration changed"):
                    compat.native_aval_metadata_schema()
            finally:
                declaration[:] = sequence
        else:
            with monkeypatch.context() as change:
                change.setattr(owner, "__slots__", malformed)
                with pytest.raises(TypeError, match="slot declaration changed"):
                    compat.native_aval_metadata_schema()
        assert vars(owner)["__slots__"] is declaration
        assert compat.native_aval_metadata_schema() == expected[sequence]
    with monkeypatch.context() as change:
        change.setattr(owner, "__slots__", Foreign())
        with pytest.raises(TypeError, match="slot declaration changed"):
            compat.native_aval_metadata_schema()
    with monkeypatch.context() as change:
        change.setattr(compat, "ShapedArray", Foreign())
        with pytest.raises(TypeError, match="class binding changed"):
            compat.native_aval_metadata_schema()
    assert invoked == []


def test_native_stripped_cache_lazy_and_actual_populated_reads_are_unchanged(
    monkeypatch: MonkeyPatch,
) -> None:
    backing = np.asarray([-0.0, np.nan], np.float32)
    value = _native_typed_constant(backing, True)
    carrier_state = _native_typed_constant_state(value)
    assert compat.native_typed_array_payload(value)[1] is True
    _assert_native_typed_constant_state(value, carrier_state)
    if compat.native_aval_metadata_schema() != "manual-stripped":
        # The new cache relation confers no permission on older layouts.
        with pytest.raises(TypeError, match="exact schema"):
            compat._require_manual_stripped_aval_cache(
                compat.ShapedArray((2,), np.float32)
            )
        return
    audit = _corrupt_isolated_native_cached_aval(value, "_stripped_weak_type", None)
    base = value._aval
    lazy_state = _native_aval_slot_state(base)
    assert compat.native_typed_array_payload(value)[1] is True
    _assert_original_native_aval_untouched((base, lazy_state, *audit[2:]))
    # The real native method, before observation, populates only this isolated
    # aval. The original interned aval and its every slot remain unchanged.
    strong = base.strip_weak_type()
    assert base._stripped_weak_type is strong and strong.weak_type is False
    assert strong._stripped_weak_type is None
    _assert_original_native_aval_untouched(audit)
    base_state, strong_state = _native_aval_slot_state(base), _native_aval_slot_state(
        strong
    )
    carrier_state = _native_typed_constant_state(value)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("native cache method ran during observation")

    with monkeypatch.context() as change:
        change.setattr(compat.ShapedArray, "strip_weak_type", forbidden)
        change.setattr(compat.ShapedArray, "update_weak_type", forbidden)
        for _ in range(3):
            result = compat.native_typed_array_payload(value)
            assert result is not None and result[1] is True
            assert result[0].dtype == backing.dtype and result[0].shape == backing.shape
            assert result[0].tobytes() == audit[-1]
            _assert_native_typed_constant_state(value, carrier_state)
            _assert_original_native_aval_untouched((base, base_state, *audit[2:]))
            _assert_original_native_aval_untouched((strong, strong_state, *audit[2:]))
            _assert_original_native_aval_untouched(audit)


def test_native_stripped_cache_rejects_stale_foreign_and_cyclic_state_without_hooks() -> (
    None
):
    backing = np.asarray([-0.0, np.nan], np.float32)
    value = _native_typed_constant(backing, True)
    assert compat.native_typed_array_payload(value)[1] is True
    if compat.native_aval_metadata_schema() != "manual-stripped":
        with pytest.raises(TypeError, match="exact schema"):
            compat._require_manual_stripped_aval_cache(
                compat.ShapedArray((2,), np.float32)
            )
        return
    audit = _corrupt_isolated_native_cached_aval(value, "_stripped_weak_type", None)
    base = value._aval
    strong = base.strip_weak_type()
    strong_state = _native_aval_slot_state(strong)
    slots = {name: slot for name, slot, _, _ in _native_aval_slot_state(base)}
    carrier_slot = dict(compat._NATIVE_TYPED_ARRAY_MEMBERS)["_aval"]
    invoked = []

    class Foreign:
        def __getattribute__(self, name: str) -> Any:
            invoked.append(name)
            raise AssertionError("foreign cache reflection ran")

        def __eq__(self, other: Any) -> bool:
            invoked.append("equality")
            raise AssertionError("foreign cache equality ran")

        def __repr__(self) -> str:
            invoked.append("repr")
            raise AssertionError("foreign cache representation ran")

    class Derived(compat.ShapedArray):
        def __getattribute__(self, name: str) -> Any:
            invoked.append(name)
            raise AssertionError("derived cache reflection ran")

    def clone(original: Any) -> Any:
        copied = object.__new__(compat.ShapedArray)
        for _, slot, initialized, item in _native_aval_slot_state(original):
            if initialized:
                slot.__set__(copied, item)
        return copied

    faults = (
        "weak",
        "weak-type",
        "dtype",
        "foreign-dtype",
        "shape",
        "foreign-shape",
        "sharding",
        "manual",
        "memory",
        "self",
        "nested",
        "foreign",
        "subclass",
        "uninitialized",
        "strong-base",
    )
    for fault in faults:
        isolated, cached = clone(base), clone(strong)
        slots["_stripped_weak_type"].__set__(isolated, cached)
        carrier_slot.__set__(value, isolated)
        if fault == "weak":
            slots["weak_type"].__set__(cached, True)
        elif fault == "weak-type":
            slots["weak_type"].__set__(cached, 0)
        elif fault == "dtype":
            slots["dtype"].__set__(cached, np.dtype(np.float64))
        elif fault == "foreign-dtype":
            slots["dtype"].__set__(cached, Foreign())
        elif fault == "shape":
            slots["shape"].__set__(cached, (3,))
        elif fault == "foreign-shape":
            slots["shape"].__set__(cached, (Foreign(),))
        elif fault in ("sharding", "manual", "memory"):
            field = {
                "sharding": "sharding",
                "manual": "manual_axis_type",
                "memory": "memory_space",
            }[fault]
            slots[field].__set__(cached, Foreign())
        elif fault == "self":
            slots["_stripped_weak_type"].__set__(isolated, isolated)
        elif fault == "nested":
            slots["_stripped_weak_type"].__set__(cached, isolated)
        elif fault == "foreign":
            slots["_stripped_weak_type"].__set__(isolated, Foreign())
        elif fault == "subclass":
            slots["_stripped_weak_type"].__set__(isolated, object.__new__(Derived))
        elif fault == "uninitialized":
            slots["_stripped_weak_type"].__delete__(cached)
        else:
            slots["weak_type"].__set__(isolated, False)
        before = _native_typed_constant_state(value)
        base_before, cached_before = _native_aval_slot_state(
            isolated
        ), _native_aval_slot_state(cached)
        with pytest.raises(TypeError, match="native signature.*stripped") as caught:
            compat.native_typed_array_payload(value)
        labels = {
            "weak": "cached_weakness",
            "weak-type": "cached_weakness",
            "manual": "manual_axis_identity",
            "memory": "memory_space_identity",
            "self": "self_cycle, cached_weakness, cached_terminal",
            "nested": "cached_terminal",
            "strong-base": "base_weakness",
        }
        if fault in labels:
            assert str(caught.value) == (
                "native signature has invalid stripped weak-type cache: "
                + labels[fault]
            )
        if fault == "sharding":
            assert str(caught.value) == (
                "native signature stripped cache requires native plain sharding"
            )
        assert invoked == [], fault
        _assert_native_typed_constant_state(value, before)
        _assert_original_native_aval_untouched((isolated, base_before, *audit[2:]))
        _assert_original_native_aval_untouched((cached, cached_before, *audit[2:]))
        _assert_original_native_aval_untouched((strong, strong_state, *audit[2:]))
        _assert_original_native_aval_untouched(audit)
    assert backing.tobytes() == audit[-1]


def test_native_stripped_cache_symbolic_identity_retains_scope_without_foreign_hooks() -> (
    None
):
    if compat.native_aval_metadata_schema() != "manual-stripped":
        aval = compat.ShapedArray((2,), np.float32, weak_type=True)
        before = _native_aval_slot_state(aval)
        compat._require_plain_native_aval(aval)
        assert all(
            a[0] == b[0] and all(x is y for x, y in zip(a[1:], b[1:], strict=True))
            for a, b in zip(_native_aval_slot_state(aval), before, strict=True)
        )
        with pytest.raises(TypeError, match="exact schema"):
            compat._require_manual_stripped_aval_cache(aval)
        return
    from jax import export

    scope = export.SymbolicScope()
    (dim,) = export.symbolic_shape("B", scope=scope)
    original = compat.ShapedArray((dim,), np.float32, weak_type=True)
    original_state = _native_aval_slot_state(original)
    base = object.__new__(compat.ShapedArray)
    for _, slot, initialized, item in original_state:
        if initialized:
            slot.__set__(base, item)
    cache_slot = vars(compat.ShapedArray)["_stripped_weak_type"]
    cache_slot.__set__(base, None)
    strong = base.strip_weak_type()
    before = (_native_aval_slot_state(base), _native_aval_slot_state(strong))
    for _ in range(3):
        compat._require_plain_native_aval(base)
        for actual, expected in zip(
            (_native_aval_slot_state(base), _native_aval_slot_state(strong)),
            before,
            strict=True,
        ):
            assert all(
                a[0] == b[0] and all(x is y for x, y in zip(a[1:], b[1:], strict=True))
                for a, b in zip(actual, expected, strict=True)
            )
    other_scope = export.SymbolicScope()
    (other_dim,) = export.symbolic_shape("B", scope=other_scope)
    stale = object.__new__(compat.ShapedArray)
    for _, slot, initialized, item in before[1]:
        if initialized:
            slot.__set__(stale, item)
    vars(compat.ShapedArray)["shape"].__set__(stale, (other_dim,))
    cache_slot.__set__(base, stale)
    with pytest.raises(TypeError, match="symbolic identity"):
        compat._require_plain_native_aval(base)
    assert (
        dim.scope is scope
        and other_dim.scope is other_scope
        and scope is not other_scope
    )
    assert all(
        a[0] == b[0] and all(x is y for x, y in zip(a[1:], b[1:], strict=True))
        for a, b in zip(_native_aval_slot_state(original), original_state, strict=True)
    )


@pytest.mark.parametrize("method", ("__setattr__", "__delattr__"))
@pytest.mark.parametrize("kind", ("function", "non-function", "hostile-freevar"))
def test_native_axisdata_frozen_closure_gap_reports_only_intrinsic_schema(
    monkeypatch: MonkeyPatch, method: str, kind: str
) -> None:
    invoked = []

    class Foreign:
        def __getattribute__(self, name: str) -> Any:
            invoked.append(name)
            raise AssertionError("foreign frozen member reflection ran")

        def __repr__(self) -> str:
            invoked.append("repr")
            raise AssertionError("foreign frozen member representation ran")

    class HostileName(str):
        def __eq__(self, other: Any) -> bool:
            invoked.append("equality")
            raise AssertionError("foreign freevar equality ran")

        def __repr__(self) -> str:
            invoked.append("name-repr")
            raise AssertionError("foreign freevar representation ran")

    def accepted_member() -> Any:
        FrozenInstanceError, cls = object(), object()

        def member(*args: Any) -> Any:
            return FrozenInstanceError, cls

        return member

    secret = Foreign()

    def mismatched_member(*args: Any) -> Any:
        return secret

    diagnostic = "co_freevars=('secret',), closure_length=1"
    member: Any = mismatched_member
    if kind == "non-function":
        member = Foreign()
        diagnostic = "non-function"
    elif kind == "hostile-freevar":
        # Some native CodeType constructors reject or canonicalize subclasses.
        # The catch covers construction alone, never the compatibility guard.
        try:
            changed = mismatched_member.__code__.replace(
                co_freevars=(HostileName("secret"),)
            )
        except (TypeError, SystemError):
            assert invoked == []
            return
        mismatched_member.__code__ = changed
        actual = mismatched_member.__code__.co_freevars
        if type(actual) is tuple and all(type(symbol) is str for symbol in actual):
            assert actual == ("secret",)
        else:
            assert type(actual) is tuple
            assert len(actual) == 1 and type(actual[0]) is HostileName
            diagnostic = "non-literal-freevars"
    namespace = {name: accepted_member() for name in ("__setattr__", "__delattr__")}
    namespace[method] = member
    selected = type("DiagnosticAxisData", (), namespace)
    monkeypatch.setitem(
        compat._NATIVE_WRAPPER_CLASSES, ("vmap_f", "AxisData"), selected
    )
    with pytest.raises(ValueError, match="generated frozen closure") as caught:
        compat._initial_native_axisdata_roles()
    assert str(caught.value) == (
        "native AxisData generated frozen closure needs an exact adapter: "
        + method
        + ": "
        + diagnostic
    )
    assert invoked == []


@pytest.mark.parametrize("hostile_leaf", (False, True))
def test_code_fingerprint_keeps_unmeasured_slice_constants_outside_admission(
    hostile_leaf: bool,
) -> None:
    invoked = []

    class Foreign:
        def __getattribute__(self, name: str) -> Any:
            invoked.append(name)
            raise AssertionError("foreign slice leaf reflection ran")

        def __repr__(self) -> str:
            invoked.append("repr")
            raise AssertionError("foreign slice leaf representation ran")

        def __eq__(self, other: Any) -> bool:
            invoked.append("equality")
            raise AssertionError("foreign slice leaf equality ran")

        def __hash__(self) -> int:
            invoked.append("hash")
            raise AssertionError("foreign slice leaf hashing ran")

    def selected() -> None:
        pass

    value = slice(Foreign() if hostile_leaf else 1, None, 2)
    changed = selected.__code__.replace(co_consts=(*selected.__code__.co_consts, value))
    with pytest.raises(TypeError, match="code constant type needs an exact adapter"):
        compat.canonical_code_fingerprint(changed)
    assert invoked == []


def test_native_interned_cache_accepts_distinct_plain_sharding_without_mutation() -> (
    None
):
    if compat.native_aval_metadata_schema() != "manual-stripped":
        with pytest.raises(TypeError, match="exact schema"):
            compat._require_manual_stripped_aval_cache(
                compat.ShapedArray((2,), np.float32)
            )
        return
    from jax import export
    from jax.sharding import NamedSharding, PartitionSpec
    from jax._src.mesh import empty_abstract_mesh

    scope = export.SymbolicScope()
    (dim,) = export.symbolic_shape("B", scope=scope)
    explicit = NamedSharding(empty_abstract_mesh, PartitionSpec(None))
    strong = compat.ShapedArray((dim,), np.float32, weak_type=False, sharding=explicit)
    # Explicit placement uses get_sharding; the default has its separate native cache.
    base = compat.ShapedArray((dim,), np.float32, weak_type=True)
    first, second = strong.sharding, base.sharding
    assert type(first) is NamedSharding and type(second) is NamedSharding
    assert first is not second
    assert base.strip_weak_type() is strong
    assert base._stripped_weak_type is strong and strong._stripped_weak_type is None
    assert strong.weak_type is False and base.weak_type is True
    before = (_native_aval_slot_state(base), _native_aval_slot_state(strong))
    for _ in range(3):
        compat._require_plain_native_aval(base)
        compat._require_plain_native_aval(strong)
        for aval, expected in zip((base, strong), before, strict=True):
            actual = _native_aval_slot_state(aval)
            assert len(actual) == len(expected)
            assert all(
                a[0] == z[0] and all(x is y for x, y in zip(a[1:], z[1:], strict=True))
                for a, z in zip(actual, expected, strict=True)
            )
    assert type(base.shape) is tuple and type(strong.shape) is tuple
    assert len(base.shape) == len(strong.shape) == 1
    assert base.shape[0] is dim and strong.shape[0] is dim and dim.scope is scope
    assert first is not second and base.sharding is second and strong.sharding is first


def test_native_interned_cache_rejects_nonplain_native_and_foreign_spec_state() -> None:
    if compat.native_aval_metadata_schema() != "manual-stripped":
        with pytest.raises(TypeError, match="exact schema"):
            compat._require_manual_stripped_aval_cache(
                compat.ShapedArray((2,), np.float32)
            )
        return
    from jax.sharding import NamedSharding, PartitionSpec
    from jax._src.mesh import AbstractMesh, empty_abstract_mesh

    base = compat.ShapedArray((2,), np.float32, weak_type=True)
    strong = base.strip_weak_type()
    originals = (_native_aval_slot_state(base), _native_aval_slot_state(strong))
    slots = {name: slot for name, slot, _, _ in originals[0]}
    canonical_spec = PartitionSpec(None)
    spec_slots = {}
    for name in ("_partitions", "unreduced", "reduced", "unreduced_kind"):
        descriptor = vars(PartitionSpec)[name]
        assert type(descriptor) is MemberDescriptorType
        assert descriptor.__objclass__ is PartitionSpec and descriptor.__name__ == name
        spec_slots[name] = descriptor
    spec_before = {
        name: slot.__get__(canonical_spec, PartitionSpec)
        for name, slot in spec_slots.items()
    }
    invoked = []

    class Foreign:
        def __getattribute__(self, name: str) -> Any:
            invoked.append(name)
            raise AssertionError("foreign partition reflection ran")

        def __eq__(self, other: Any) -> bool:
            invoked.append("equality")
            raise AssertionError("foreign partition equality ran")

        def __repr__(self) -> str:
            invoked.append("repr")
            raise AssertionError("foreign partition representation ran")

        def __len__(self) -> int:
            invoked.append("len")
            raise AssertionError("foreign partition length ran")

        def __iter__(self) -> Any:
            invoked.append("iter")
            raise AssertionError("foreign partition iteration ran")

        def __bool__(self) -> bool:
            invoked.append("bool")
            raise AssertionError("foreign partition truth ran")

    def clone(original: Any) -> Any:
        copied = object.__new__(compat.ShapedArray)
        for _, slot, ready, item in _native_aval_slot_state(original):
            if ready:
                slot.__set__(copied, item)
        return copied

    nonempty_mesh = AbstractMesh((1,), ("cache_axis",))
    faults = (
        "native-mesh",
        "native-axis",
        "native-rank",
        "native-memory-kind",
        "native-manual",
        "native-memory",
        "foreign-mesh-size",
        "foreign-leaf",
        "foreign-partitions",
        "foreign-unreduced",
        "foreign-reduced",
        "foreign-kind",
        "uninitialized",
    )
    for fault in faults:
        isolated, cached = clone(base), clone(strong)
        slots["_stripped_weak_type"].__set__(isolated, cached)
        spec_copy = object.__new__(PartitionSpec)
        for name, descriptor in spec_slots.items():
            descriptor.__set__(spec_copy, spec_before[name])
        placement = NamedSharding(empty_abstract_mesh, spec_copy)
        assert placement.spec is spec_copy
        if fault == "native-mesh":
            placement = NamedSharding(nonempty_mesh, canonical_spec)
        elif fault == "native-axis":
            placement = NamedSharding(nonempty_mesh, PartitionSpec("cache_axis"))
        elif fault == "native-rank":
            placement = NamedSharding(empty_abstract_mesh, PartitionSpec())
        elif fault == "native-memory-kind":
            placement = NamedSharding(
                empty_abstract_mesh, canonical_spec, memory_kind="device"
            )
        elif fault == "native-manual":
            slots["manual_axis_type"].__set__(
                cached,
                compat._native_core.ManualAxisType(varying=frozenset({"cache_axis"})),
            )
        elif fault == "native-memory":
            slots["memory_space"].__set__(cached, compat._native_core.MemorySpace.Host)
        elif fault == "foreign-mesh-size":
            mesh_copy = object.__new__(AbstractMesh)
            for name in (
                "axis_sizes",
                "axis_names",
                "axis_types",
                "abstract_device",
                "size",
            ):
                object.__setattr__(mesh_copy, name, vars(empty_abstract_mesh)[name])
            placement = NamedSharding(mesh_copy, canonical_spec)
            assert placement.mesh is mesh_copy
            object.__setattr__(mesh_copy, "size", Foreign())
        elif fault == "foreign-leaf":
            spec_slots["_partitions"].__set__(spec_copy, (Foreign(),))
        elif fault == "uninitialized":
            spec_slots["_partitions"].__delete__(spec_copy)
        else:
            field = {
                "foreign-partitions": "_partitions",
                "foreign-unreduced": "unreduced",
                "foreign-reduced": "reduced",
                "foreign-kind": "unreduced_kind",
            }[fault]
            spec_slots[field].__set__(spec_copy, Foreign())
        slots["sharding"].__set__(cached, placement)
        before = (_native_aval_slot_state(isolated), _native_aval_slot_state(cached))
        with pytest.raises(TypeError, match="native signature"):
            compat._require_plain_native_aval(isolated)
        assert invoked == [], fault
        for aval, expected in zip(
            (base, strong, isolated, cached), (*originals, *before), strict=True
        ):
            actual = _native_aval_slot_state(aval)
            assert len(actual) == len(expected)
            assert all(
                a[0] == z[0] and all(x is y for x, y in zip(a[1:], z[1:], strict=True))
                for a, z in zip(actual, expected, strict=True)
            )
        assert all(
            slot.__get__(canonical_spec, PartitionSpec) is spec_before[name]
            for name, slot in spec_slots.items()
        )
