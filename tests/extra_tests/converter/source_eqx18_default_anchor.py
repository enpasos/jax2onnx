# tests/extra_tests/converter/source_eqx18_default_anchor.py

"""Supplement one exact historical opaque recipe without changing its raw hash."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from functools import partial
import hashlib
import inspect
import json
from types import FunctionType
from typing import Any

import equinox as eqx
import jax
import numpy as np

from jax2onnx._compat.jax import is_selected_prng_key_dtype
from scripts.source_precision_inventory import canonical_contract


_ORIGIN = "plugin::eqx.nn.dropout"
_ORDINAL = 2
_RAW_SHA256 = "7f9d6b14ae54a436c7ff39fefedcd1f9b7d3a5bc4be6b539737ed27735c2d2c5"
_UNRESOLVED = ("contract.callable.defaults.1: unsupported equinox.nn._dropout.Dropout",)


def raw_eqx18_contract(case: Any) -> tuple[str, tuple[str, ...]]:
    unresolved: list[str] = []
    value = canonical_contract(
        {
            key: value
            for key, value in case.items()
            if key not in {"testcase", "description", "context", "component"}
        },
        unresolved=unresolved,
    )
    blob = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest(), tuple(unresolved)


def _key_digest(key: Any) -> str:
    assert isinstance(key, jax.Array) and tuple(key.shape) == ()
    assert is_selected_prng_key_dtype(key.dtype)
    words = np.asarray(jax.random.key_data(key))
    assert words.dtype == np.dtype(np.uint32) and words.shape == (2,)
    return hashlib.sha256(words.tobytes()).hexdigest()


@dataclass(eq=False)
class Eqx18DefaultAnchor:
    original: Any
    code: Any
    globals: Any
    closure: Any
    defaults: Any
    kwdefaults: Any
    model: Any
    dataclass_fields: tuple[Any, ...]
    tree: Any
    key: Any = None
    key_type: Any = None
    key_dtype: Any = None
    key_impl: Any = None
    key_digest: str | None = None
    bindings: list[Any] = field(default_factory=list)

    @classmethod
    def capture(
        cls, case: Any, origin: str, ordinal: int, raw_digest: str
    ) -> Eqx18DefaultAnchor:
        assert (origin, ordinal, raw_digest) == (_ORIGIN, _ORDINAL, _RAW_SHA256)
        assert raw_eqx18_contract(case) == (_RAW_SHA256, _UNRESOLVED)
        function = case["callable"]
        assert type(function) is FunctionType and function.__name__ == "<lambda>"
        assert function.__defaults__ is not None and len(function.__defaults__) == 2
        assert function.__defaults__[0] is None
        model = function.__defaults__[1]
        assert type(model) is eqx.nn.Dropout
        _, tree = jax.tree.flatten(model)
        result = cls(
            function,
            function.__code__,
            function.__globals__,
            function.__closure__,
            function.__defaults__,
            function.__kwdefaults__,
            model,
            tuple(fields(model)),
            tree,
        )
        result.validate(case)
        return result

    def validate_raw(self, case: Any, digest: str, unresolved: Any) -> None:
        assert digest == _RAW_SHA256 and tuple(unresolved) == _UNRESOLVED
        self.validate(case)

    def validate(self, case: Any) -> None:
        assert raw_eqx18_contract(case) == (_RAW_SHA256, _UNRESOLVED)
        function = case["callable"]
        assert type(function) is FunctionType and function is self.original
        assert function.__code__ is self.code and function.__globals__ is self.globals
        assert function.__closure__ is self.closure is None
        assert function.__defaults__ is self.defaults
        assert function.__kwdefaults__ is self.kwdefaults
        assert self.kwdefaults is None or (
            type(self.kwdefaults) is dict and not self.kwdefaults
        )
        assert self.defaults[0] is None and self.defaults[1] is self.model
        assert inspect.getmodule(function) is not None
        assert function.__globals__ is vars(inspect.getmodule(function))
        signature = inspect.signature(function)
        parameters = tuple(signature.parameters.values())
        assert tuple(p.name for p in parameters) == ("x", "inference", "key", "model")
        assert all(
            p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for p in parameters
        )
        assert parameters[0].default is parameters[1].default is inspect.Parameter.empty
        assert parameters[2].default is None and parameters[3].default is self.model
        assert type(self.model) is eqx.nn.Dropout
        actual_fields = tuple(fields(self.model))
        assert len(actual_fields) == len(self.dataclass_fields) == 2
        assert all(
            actual is held
            for actual, held in zip(actual_fields, self.dataclass_fields, strict=True)
        )
        assert tuple(f.name for f in actual_fields) == ("p", "inference")
        assert all(not f.metadata for f in actual_fields)
        assert set(vars(self.model)) == {"p", "inference"}
        assert type(self.model.p) is float and self.model.p == 0.5
        assert type(self.model.inference) is bool and self.model.inference is False
        leaves, tree = jax.tree.flatten(self.model)
        assert tree == self.tree and len(leaves) == 2
        assert type(leaves[0]) is float and leaves[0] == 0.5
        assert type(leaves[1]) is bool and leaves[1] is False
        if self.key is not None:
            assert type(self.key) is self.key_type and self.key.dtype == self.key_dtype
            assert str(jax.random.key_impl(self.key)) == self.key_impl
            assert _key_digest(self.key) == self.key_digest
        for bound in self.bindings:
            self._validate_binding(bound)

    def _validate_binding(self, bound: Any) -> None:
        assert (
            type(bound) is partial and bound.func is self.original and bound.args == ()
        )
        assert set(bound.keywords) == {"key"} and bound.keywords["key"] is self.key
        assert bound.func.__defaults__ is self.defaults
        assert bound.func.__defaults__[1] is self.model

    def bind(self, case: Any, bound: Any, key: Any) -> None:
        self.validate(case)
        digest = _key_digest(key)
        if self.key is None:
            self.key, self.key_type, self.key_dtype = key, type(key), key.dtype
            self.key_impl, self.key_digest = str(jax.random.key_impl(key)), digest
        else:
            assert key is self.key and digest == self.key_digest
        self._validate_binding(bound)
        if not any(value is bound for value in self.bindings):
            self.bindings.append(bound)
        self.validate(case)

    def evidence(self) -> dict[str, Any]:
        return dict(
            kind="supplemental exact native default anchor; historical raw recipe unchanged",
            raw_contract_sha256=_RAW_SHA256,
            raw_unresolved=list(_UNRESOLVED),
            function_identity=id(self.original),
            code_identity=id(self.code),
            globals_identity=id(self.globals),
            defaults_identity=id(self.defaults),
            native_default_identity=id(self.model),
            native_default_class=f"{type(self.model).__module__}.{type(self.model).__qualname__}",
            native_default_fields={"p": 0.5, "inference": False},
            native_tree=str(self.tree),
            caller_key_identity=None if self.key is None else id(self.key),
            caller_key_impl=self.key_impl,
            caller_key_words_sha256=self.key_digest,
            retained_exact_partial_bindings=len(self.bindings),
        )
