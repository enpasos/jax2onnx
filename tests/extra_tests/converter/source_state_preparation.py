# tests/extra_tests/converter/source_state_preparation.py

"""Internal, explicitly requested state lifting of unchanged native calls.

This is preparation, not automatic conversion of a captured mutable program.
The adapters below cover only this test's exact native Dropout owners.
Control-flow callers must thread every returned word through every invocation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import FunctionType
from typing import Any, Protocol

import equinox as eqx
from flax import linen, nnx
import jax
import numpy as np

from jax2onnx._compat.jax import (
    is_selected_prng_key_dtype,
    selected_prng_key_words,
)


def closure_contents(target: FunctionType) -> dict[str, Any]:
    return {
        name: cell.cell_contents
        for name, cell in zip(
            target.__code__.co_freevars, target.__closure__ or (), strict=True
        )
    }


def _cell(value: Any) -> Any:
    return (lambda: value).__closure__[0]


def _attributes(target: FunctionType) -> tuple[Any, ...]:
    return (
        target.__code__,
        target.__globals__,
        target.__name__,
        target.__qualname__,
        target.__module__,
        target.__doc__,
        target.__defaults__,
        target.__kwdefaults__,
        target.__annotations__,
        target.__dict__,
    )


@dataclass(frozen=True)
class NativeClosure:
    """Retain the actual callable; only a declared owner cell may be rebound."""

    original: FunctionType
    owner_name: str | None
    attributes: tuple[Any, ...]
    captures: tuple[Any, ...]
    mapping_entries: tuple[tuple[str, tuple[tuple[Any, Any], ...]], ...]

    @classmethod
    def capture(cls, original: Any, owner_name: str | None) -> NativeClosure:
        if type(original) is not FunctionType:
            raise TypeError("Preparation requires the original factory function")
        captures = tuple(cell.cell_contents for cell in original.__closure__ or ())
        if owner_name is not None and owner_name not in original.__code__.co_freevars:
            raise ValueError("The declared mutable owner is not an actual capture")
        mappings = tuple(
            (name, tuple(value.items()))
            for name in ("__dict__", "__annotations__", "__kwdefaults__")
            if (value := getattr(original, name)) is not None
        )
        return cls(original, owner_name, _attributes(original), captures, mappings)

    def assert_original(self) -> None:
        assert all(
            actual is expected
            for actual, expected in zip(
                _attributes(self.original), self.attributes, strict=True
            )
        )
        for name, entries in self.mapping_entries:
            mapping = getattr(self.original, name)
            assert list(mapping) == [key for key, _ in entries]
            assert all(mapping[key] is value for key, value in entries)
        assert all(
            cell.cell_contents is expected
            for cell, expected in zip(
                self.original.__closure__ or (), self.captures, strict=True
            )
        )

    def rebind_owner(self, owner: Any) -> FunctionType:
        self.assert_original()
        assert self.owner_name is not None
        cells = tuple(
            _cell(owner) if name == self.owner_name else cell
            for name, cell in zip(
                self.original.__code__.co_freevars,
                self.original.__closure__ or (),
                strict=True,
            )
        )
        result = FunctionType(
            self.original.__code__,
            self.original.__globals__,
            self.original.__name__,
            self.original.__defaults__,
            cells,
        )
        result.__qualname__ = self.original.__qualname__
        result.__module__ = self.original.__module__
        result.__doc__ = self.original.__doc__
        result.__kwdefaults__ = self.original.__kwdefaults__
        result.__annotations__ = self.original.__annotations__
        result.__dict__.update(self.original.__dict__)
        for actual, expected in zip(_attributes(result), self.attributes, strict=True):
            assert actual is expected or (
                actual is result.__dict__ and actual == self.original.__dict__
            )
        for name, actual, expected in zip(
            result.__code__.co_freevars,
            cells,
            self.original.__closure__ or (),
            strict=True,
        ):
            assert actual.cell_contents is (
                owner if name == self.owner_name else expected.cell_contents
            )
        return result


class StateAdapter(Protocol):
    original: NativeClosure
    initial_words: tuple[Any, ...]
    state_names: tuple[str, ...]

    @property
    def flag_is_unused(self) -> bool: ...

    def target_from_words(self, words: tuple[Any, ...]) -> FunctionType: ...

    def words_from_target(self, target: FunctionType) -> tuple[Any, ...]: ...


@dataclass(frozen=True)
class NNXDropoutState:
    original: NativeClosure
    graphdef: Any
    fixed: Any
    tree: Any
    implementations: tuple[str | None, ...]
    initial_words: tuple[Any, ...]
    state_names: tuple[str, ...]

    @property
    def flag_is_unused(self) -> bool:
        return closure_contents(self.original.original)["block"].rate == 0.0

    @classmethod
    def from_target(cls, target: Any) -> NNXDropoutState:
        original = NativeClosure.capture(target, "block")
        captures = closure_contents(target)
        if set(captures) != {"block"} or type(captures["block"]) is not nnx.Dropout:
            raise TypeError("Only the declared native NNX Dropout capture is prepared")
        graphdef, rng, fixed = nnx.split(captures["block"], nnx.RngState, ...)
        leaves, tree = jax.tree_util.tree_flatten(rng)
        implementations = []
        words = []
        names = []
        for index, value in enumerate(leaves):
            if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
                if not is_selected_prng_key_dtype(value.dtype) or value.shape != ():
                    raise ValueError("Only scalar selected native keys are prepared")
                implementations.append(str(jax.random.key_impl(value)))
                words.append(selected_prng_key_words(value))
                names.append(f"rng_key_{index}")
            else:
                if value.dtype != np.dtype(np.uint32) or value.shape != ():
                    raise ValueError(
                        "Only the original UINT32 scalar RNG count is prepared"
                    )
                implementations.append(None)
                words.append(value)
                names.append(f"rng_count_{index}")
        if tuple(implementations) != (None, "threefry2x32"):
            raise ValueError("This adapter requires the original count/key leaf order")
        return cls(
            original,
            graphdef,
            fixed,
            tree,
            tuple(implementations),
            tuple(words),
            tuple(names),
        )

    def target_from_words(self, words: tuple[Any, ...]) -> FunctionType:
        if len(words) != len(self.initial_words):
            raise ValueError("The complete declared RNG state must be supplied")
        leaves = [
            (
                jax.random.wrap_key_data(value, impl=implementation)
                if implementation is not None
                else value
            )
            for value, implementation in zip(words, self.implementations, strict=True)
        ]
        state = jax.tree_util.tree_unflatten(self.tree, leaves)
        owner = nnx.merge(self.graphdef, state, self.fixed, copy=True)
        return self.original.rebind_owner(owner)

    def words_from_target(self, target: FunctionType) -> tuple[Any, ...]:
        owner = closure_contents(target)["block"]
        leaves = jax.tree_util.tree_leaves(nnx.state(owner, nnx.RngState))
        return tuple(
            jax.random.key_data(value) if implementation is not None else value
            for value, implementation in zip(leaves, self.implementations, strict=True)
        )


@dataclass(frozen=True)
class LinenDropoutState:
    """Linen's original fixed external key stays captured, never a new input."""

    original: NativeClosure
    initial_words: tuple[Any, ...] = ()
    state_names: tuple[str, ...] = ()

    @property
    def flag_is_unused(self) -> bool:
        return closure_contents(self.original.original)["block"].rate == 0.0

    @classmethod
    def from_target(cls, target: Any) -> LinenDropoutState:
        original = NativeClosure.capture(target, None)
        captures = closure_contents(target)
        if (
            set(captures) != {"block", "key"}
            or type(captures["block"]) is not linen.Dropout
        ):
            raise TypeError(
                "Only the declared native Linen Dropout captures are prepared"
            )
        key = captures["key"]
        if not is_selected_prng_key_dtype(key.dtype) or key.shape != ():
            raise ValueError(
                "The original fixed key must be a scalar selected native key"
            )
        return cls(original)

    def target_from_words(self, words: tuple[Any, ...]) -> FunctionType:
        if words:
            raise ValueError("A fixed Linen external key is not mutable input state")
        self.original.assert_original()
        return self.original.original

    def words_from_target(self, target: FunctionType) -> tuple[Any, ...]:
        assert target is self.original.original
        self.original.assert_original()
        return ()


@dataclass(frozen=True)
class EqxDropoutState:
    """Keep the actual Equinox owner and fixed external key in native branches."""

    original: NativeClosure
    initial_words: tuple[Any, ...] = ()
    state_names: tuple[str, ...] = ()

    @property
    def flag_is_unused(self) -> bool:
        probability = closure_contents(self.original.original)["block"].p
        return isinstance(probability, (int, float)) and probability == 0

    @classmethod
    def from_target(cls, target: Any) -> EqxDropoutState:
        original = NativeClosure.capture(target, None)
        captures = closure_contents(target)
        if (
            set(captures) != {"block", "key"}
            or type(captures["block"]) is not eqx.nn.Dropout
        ):
            raise TypeError(
                "Only the declared native Equinox Dropout captures are prepared"
            )
        key = captures["key"]
        if not is_selected_prng_key_dtype(key.dtype) or key.shape != ():
            raise ValueError(
                "The original fixed key must be a scalar selected native key"
            )
        return cls(original)

    def target_from_words(self, words: tuple[Any, ...]) -> FunctionType:
        if words:
            raise ValueError("A fixed Equinox external key is not mutable input state")
        self.original.assert_original()
        return self.original.original

    def words_from_target(self, target: FunctionType) -> tuple[Any, ...]:
        assert target is self.original.original
        self.original.assert_original()
        return ()


@dataclass(frozen=True)
class ExplicitStatePreparation:
    """Complete concrete-BOOL calls; enclosing scopes own explicit state carry."""

    adapter: StateAdapter
    original_body: Callable[..., Any]

    def concrete(self, flag: bool, x: Any, *words: Any) -> tuple[Any, ...]:
        assert type(flag) is bool
        target = self.adapter.target_from_words(tuple(words))
        y = self.original_body(x, target=target, deterministic=flag)
        return (y, *self.adapter.words_from_target(target))

    def __call__(self, x: Any, flag: Any, *words: Any) -> tuple[Any, ...]:
        if self.adapter.flag_is_unused:
            # Let the unchanged zero-rate body take its own native short circuit.
            # It accepts the real operand without reading it or consuming RNG.
            target = self.adapter.target_from_words(tuple(words))
            y = self.original_body(x, target=target, deterministic=flag)
            return (y, *self.adapter.words_from_target(target))
        return jax.lax.cond(
            flag,
            lambda operands: self.concrete(True, *operands),
            lambda operands: self.concrete(False, *operands),
            (x, *words),
        )
