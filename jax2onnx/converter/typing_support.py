# jax2onnx/converter/typing_support.py

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Collection,
    Mapping,
    Protocol,
    runtime_checkable,
)

import numpy as np
import onnx_ir as ir

ResolverFn = Callable[[object], object | None]


@dataclass(frozen=True)
class SymbolicDimOrigin:
    """Typed representation of where a symbolic dimension came from."""

    value: ir.Value
    axis: int

    def as_tuple(self) -> tuple[ir.Value, int]:
        return (self.value, self.axis)

    @classmethod
    def from_unknown(cls, origin: object | None) -> SymbolicDimOrigin | None:
        """Normalise `(value, axis)` tuples into ``SymbolicDimOrigin``."""
        if origin is None:
            return None
        if isinstance(origin, cls):
            return origin
        if (
            isinstance(origin, tuple)
            and len(origin) == 2
            and isinstance(origin[1], int)
            and isinstance(origin[0], ir.Value)
        ):
            return cls(value=origin[0], axis=int(origin[1]))
        return None

    @classmethod
    def resolve(
        cls,
        resolver: ResolverFn | None,
        dim: object,
    ) -> SymbolicDimOrigin | None:
        """Lookup ``dim`` via ``resolver`` while tolerating string fallbacks."""

        if resolver is None:
            return None
        origin = resolver(dim)
        if origin is None and not isinstance(dim, str):
            origin = resolver(str(dim))
        return cls.from_unknown(origin)


@runtime_checkable
class SymbolicDimTracker(Protocol):
    def get_symbolic_dim_origin(self, dim: object) -> SymbolicDimOrigin | None: ...


@runtime_checkable
class LoweringContextProtocol(SymbolicDimTracker, Protocol):
    builder: Any

    def fresh_name(self, base: str) -> str: ...

    def get_value_for_var(
        self,
        var: Any,
        *,
        name_hint: str | None = None,
        prefer_np_dtype: np.dtype[Any] | None = None,
    ) -> ir.Value: ...

    def bind_value_for_var(self, var: object, value: ir.Value) -> None: ...


@dataclass(frozen=True)
class AxisOverrideInfo:
    """Structured axis-0 override metadata captured during lowering."""

    extent: int
    op_type: str | None = None

    def allows_restamp(self, allowed_ops: Collection[str] | None = None) -> bool:
        """Return True when the override may safely restamp ONNX metadata."""

        if self.extent <= 1:
            return False
        if not allowed_ops:
            return True
        return self.op_type in allowed_ops

    def as_tuple(self) -> tuple[int, str | None]:
        return (self.extent, self.op_type)


AxisOverrideMap = dict[str, AxisOverrideInfo]


@dataclass(frozen=True)
class RngTrace:
    """Tracks deterministic RNG helpers requested via construct_and_call()."""

    kind: str
    seed: int | None

    def describe(self) -> str:
        return f"{self.kind}(seed={self.seed})"


@runtime_checkable
class PrimitiveLowering(Protocol):
    def lower(
        self,
        ctx: LoweringContextProtocol,
        eqn: Any,
        *extra: Any,
        **kwargs: Any,
    ) -> Any: ...


@runtime_checkable
class FunctionLowering(Protocol):
    def get_handler(
        self, converter: Any
    ) -> Callable[[Any, Any, Mapping[str, Any]], Any]: ...
