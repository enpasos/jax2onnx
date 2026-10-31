# jax2onnx/plugins/_source_function_state.py

"""Framework-independent state ports for an internal SOURCE Function call."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class FunctionOperandLayout:
    """User positional/keyword operands precede explicitly ordered owner state."""

    positional_count: int
    argument_count: int
    state_count: int = 0

    def check(self, operands: Sequence[object]) -> None:
        if (
            not 0 <= self.positional_count <= self.argument_count
            or self.state_count < 0
            or len(operands) != self.argument_count + self.state_count
        ):
            raise ValueError("SOURCE Function operand layout does not match its ports")


class FunctionStateAdapter(Protocol):
    """An immutable pure-call descriptor; numeric state never lives in it."""

    @property
    def state_count(self) -> int: ...

    @property
    def signature(self) -> tuple[object, ...]: ...

    def invoke(
        self,
        arguments: tuple[Any, ...],
        keywords: Mapping[str, object],
        state: tuple[Any, ...],
    ) -> tuple[Any, tuple[Any, ...], tuple[int, ...]]: ...

    def with_effects(self, effects: tuple[int, ...]) -> FunctionStateAdapter: ...

    def validate_owner(self, owner: object) -> None: ...

    def update_owner(self, owner: object, state: tuple[Any, ...]) -> None: ...


@dataclass(frozen=True)
class FunctionStateCall:
    """One callable shared by abstract evaluation, implementation and retracing."""

    adapter: FunctionStateAdapter
    positional_count: int
    key_reuse: object

    @property
    def signature(self) -> tuple[object, ...]:
        return (self.positional_count, self.key_reuse, *self.adapter.signature)

    def __call__(self, *arguments: Any, **keywords: object) -> Any:
        if len(arguments) != self.positional_count + self.adapter.state_count:
            raise ValueError("SOURCE Function state call has the wrong operand arity")
        result, state, _ = self.adapter.invoke(
            tuple(arguments[: self.positional_count]),
            keywords,
            tuple(arguments[self.positional_count :]),
        )
        return result, state
