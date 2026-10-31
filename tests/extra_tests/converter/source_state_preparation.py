# tests/extra_tests/converter/source_state_preparation.py

"""Internal, explicitly requested state lifting of unchanged native calls.

This is preparation, not automatic conversion of a captured mutable program.
The adapters below cover only this test's exact native Dropout owners.
Control-flow callers must thread every returned word through every invocation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from types import FunctionType
from typing import Any, Protocol

import equinox as eqx
from flax import linen, nnx

from jax2onnx._source_state_preparation import (
    NativeClosure,
    PreparedStatefulCall,
    _prepare_nnx_closure_call,
    closure_contents,
    prepare_pure_control_call,
)

from jax2onnx._compat.jax import (
    is_selected_prng_key_dtype,
)


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
    """Thin witness compatibility around the shared production state boundary."""

    original: NativeClosure
    prepared: PreparedStatefulCall

    @property
    def initial_words(self) -> tuple[Any, ...]:
        return self.prepared.initial_words

    @property
    def state_names(self) -> tuple[str, ...]:
        # Preserve the established witness's count/key role names without
        # assuming native port order. Production names stay path-based.
        return tuple(
            f"rng_{'key' if spec.key_implementation is not None else 'count'}_{index}"
            for index, spec in enumerate(self.prepared.state_specs)
        )

    @property
    def flag_is_unused(self) -> bool:
        return closure_contents(self.original.original)["block"].rate == 0.0

    @classmethod
    def from_target(cls, target: Any) -> NNXDropoutState:
        original = NativeClosure.capture(target, "block")
        captures = closure_contents(target)
        if set(captures) != {"block"} or type(captures["block"]) is not nnx.Dropout:
            raise TypeError("Only the declared native NNX Dropout capture is prepared")
        return cls(original, _prepare_nnx_closure_call(target, owner_name="block"))

    def target_from_words(self, words: tuple[Any, ...]) -> FunctionType:
        return self.prepared.target_from_words(words)

    def words_from_target(self, target: FunctionType) -> tuple[Any, ...]:
        return self.prepared.words_from_target(target)


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
        if target is not self.original.original:
            raise ValueError(
                "The supplied callable is not the original fixed-key target"
            )
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
        if target is not self.original.original:
            raise ValueError(
                "The supplied callable is not the original fixed-key target"
            )
        self.original.assert_original()
        return ()


@dataclass(frozen=True)
class ExplicitStatePreparation:
    """Keep established witness call sites on the production whole-call path."""

    adapter: StateAdapter
    original_body: Callable[..., Any]

    def _prepared(self) -> PreparedStatefulCall:
        if isinstance(self.adapter, NNXDropoutState):
            return _prepare_nnx_closure_call(
                self.adapter.original.original,
                owner_name="block",
                native_control=self.adapter.flag_is_unused,
                call_body=self.original_body,
            )
        result = prepare_pure_control_call(
            partial(self.original_body, target=self.adapter.original.original),
            control_parameter="deterministic",
        )
        return replace(result, _native_control=self.adapter.flag_is_unused)

    def concrete(self, flag: bool, x: Any, *words: Any) -> tuple[Any, ...]:
        self.adapter.original.assert_original()
        return self._prepared().concrete(flag, x, *words)

    def __call__(self, x: Any, flag: Any, *words: Any) -> tuple[Any, ...]:
        self.adapter.original.assert_original()
        return self._prepared()(x, flag, *words)
