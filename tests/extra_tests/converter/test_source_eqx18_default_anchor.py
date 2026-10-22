# tests/extra_tests/converter/test_source_eqx18_default_anchor.py

"""Exact supplemental provenance for the one historical opaque Eqx default."""

from contextlib import contextmanager
from functools import partial
from types import FunctionType
from typing import Any, Iterator

import equinox as eqx
import jax
import pytest

from tests.extra_tests.converter.source_eqx18_default_anchor import (
    Eqx18DefaultAnchor,
    _RAW_SHA256,
    _UNRESOLVED,
    raw_eqx18_contract,
)
from tests.extra_tests.converter.source_precision_support import raw_source_case
from tests.extra_tests.converter.test_source_fixed_parameter_models import _CONTRACTS
from tests.extra_tests.converter.test_source_original22_state_preparation import (
    _bound_original,
    _contract,
)


def _original() -> tuple[Any, Eqx18DefaultAnchor]:
    origin, ordinal, digest = _CONTRACTS[18]
    assert digest == _RAW_SHA256
    case = raw_source_case(origin, ordinal)
    return case, Eqx18DefaultAnchor.capture(case, origin, ordinal, digest)


@contextmanager
def _model_field(model: Any, name: str, value: Any) -> Iterator[None]:
    missing = object()
    original = vars(model).get(name, missing)
    object.__setattr__(model, name, value)
    try:
        yield
    finally:
        if original is missing:
            object.__delattr__(model, name)
        else:
            object.__setattr__(model, name, original)


def test_exact_original_eqx_default_keeps_raw_unresolved_recipe_and_caller_key() -> (
    None
):
    case, anchor = _original()
    assert raw_eqx18_contract(case) == (_RAW_SHA256, _UNRESOLVED)
    with pytest.raises(AssertionError):
        _contract(case)
    assert _contract(case, default_anchor=anchor) == _RAW_SHA256
    key = jax.random.key(20261021, impl="threefry2x32")
    first = _bound_original(case, 18, key, default_anchor=anchor)
    second = _bound_original(case, 18, key, default_anchor=anchor)
    assert first is not second and first.func is second.func is case["callable"]
    assert first.func.__defaults__[1] is anchor.model
    assert first.keywords["key"] is second.keywords["key"] is key
    anchor.validate(case)
    evidence = anchor.evidence()
    assert evidence["raw_contract_sha256"] == _RAW_SHA256
    assert evidence["raw_unresolved"] == list(_UNRESOLVED)
    assert evidence["retained_exact_partial_bindings"] == 2
    assert evidence["caller_key_identity"] == id(key)
    assert raw_eqx18_contract(case) == (_RAW_SHA256, _UNRESOLVED)


@pytest.mark.parametrize("name,value", (("p", 0.75), ("inference", True)))
def test_scalar_model_mutation_keeps_old_opaque_hash_but_fails_new_anchor(
    name: str, value: Any
) -> None:
    case, anchor = _original()
    with _model_field(anchor.model, name, value):
        assert raw_eqx18_contract(case) == (_RAW_SHA256, _UNRESOLVED)
        with pytest.raises(AssertionError):
            anchor.validate(case)
        with pytest.raises(AssertionError):
            _contract(case, default_anchor=anchor)
    anchor.validate(case)


@pytest.mark.parametrize(
    "kind",
    (
        "same-looking-model",
        "same-default-tuple",
        "function",
        "code",
        "foreign-class",
        "subclass",
        "hidden-field",
    ),
)
def test_changed_default_owner_tuple_function_code_or_model_structure_is_rejected(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    case, anchor = _original()
    function, defaults = case["callable"], anchor.defaults
    if kind == "hidden-field":
        with _model_field(anchor.model, "unreviewed_state", False):
            with pytest.raises(AssertionError):
                anchor.validate(case)
        return
    if kind == "function":
        clone = FunctionType(
            function.__code__,
            function.__globals__,
            function.__name__,
            defaults,
            function.__closure__,
        )
        clone.__qualname__ = function.__qualname__
        clone.__kwdefaults__ = function.__kwdefaults__
        case = {**case, "callable": clone}
        assert raw_eqx18_contract(case) == (_RAW_SHA256, _UNRESOLVED)
    elif kind == "code":

        def changed(x: Any, inference: Any, key: Any = None, model: Any = None) -> Any:
            return x

        monkeypatch.setattr(function, "__code__", changed.__code__)
    else:
        if kind == "same-looking-model":
            model = eqx.nn.Dropout(p=0.5, inference=False)
        elif kind == "foreign-class":
            model = object()
        elif kind == "subclass":

            class ForeignDropout(eqx.nn.Dropout):
                pass

            model = ForeignDropout(p=0.5, inference=False)
        else:
            model = anchor.model
        replaced = (defaults[0], model)
        assert replaced is not defaults
        monkeypatch.setattr(function, "__defaults__", replaced)
        if kind in {"same-looking-model", "same-default-tuple"}:
            assert raw_eqx18_contract(case) == (_RAW_SHA256, _UNRESOLVED)
    with pytest.raises(AssertionError):
        anchor.validate(case)


@pytest.mark.parametrize(
    "kind",
    ("origin", "ordinal", "raw-digest", "extra-unresolved", "missing-unresolved"),
)
def test_opaque_default_exception_is_limited_to_exact_original_recipe(
    kind: str,
) -> None:
    case, anchor = _original()
    origin, ordinal, digest = _CONTRACTS[18]
    if kind in {"extra-unresolved", "missing-unresolved"}:
        unresolved = (
            (*_UNRESOLVED, "unknown.opaque: unsupported object")
            if kind == "extra-unresolved"
            else ()
        )
        with pytest.raises(AssertionError):
            anchor.validate_raw(case, digest, unresolved)
    else:
        selected = (
            (origin + ".other", ordinal, digest)
            if kind == "origin"
            else (
                (origin, ordinal + 1, digest)
                if kind == "ordinal"
                else (origin, ordinal, "0" * 64)
            )
        )
        with pytest.raises(AssertionError):
            Eqx18DefaultAnchor.capture(case, *selected)


@pytest.mark.parametrize(
    "kind",
    (
        "different-key-object",
        "changed-held-binding",
        "positional",
        "extra-keyword",
        "foreign-function",
    ),
)
def test_exact_caller_key_partial_and_alias_bindings_cannot_drift(kind: str) -> None:
    case, anchor = _original()
    key = jax.random.key(20261021, impl="threefry2x32")
    target = _bound_original(case, 18, key, default_anchor=anchor)
    second_key = jax.random.key(20261021, impl="threefry2x32")
    assert second_key is not key
    if kind == "changed-held-binding":
        target.keywords["key"] = second_key
        with pytest.raises(AssertionError):
            anchor.validate(case)
        return
    if kind == "different-key-object":
        candidate, caller_key = partial(case["callable"], key=second_key), second_key
    elif kind == "positional":
        candidate, caller_key = partial(case["callable"], None, key=key), key
    elif kind == "extra-keyword":
        candidate, caller_key = partial(case["callable"], key=key, inference=True), key
    else:
        candidate, caller_key = partial(lambda *args, **kwargs: None, key=key), key
    with pytest.raises(AssertionError):
        anchor.bind(case, candidate, caller_key)
