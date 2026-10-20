# tests/extra_tests/framework/test_jax_compat.py

from __future__ import annotations

from typing import Any
from types import SimpleNamespace

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
