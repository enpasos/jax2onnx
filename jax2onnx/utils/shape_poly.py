# jax2onnx/utils/shape_poly.py

from __future__ import annotations

import numbers
from typing import Any, Protocol, Sequence, TypeAlias, TypeGuard, runtime_checkable


@runtime_checkable
class DimFactorLike(Protocol):
    var: object
    operation: str | None
    operands: Sequence[Any]


DimFactorWithPower: TypeAlias = tuple[DimFactorLike, int]


@runtime_checkable
class DimTermLike(Protocol):
    _factors: tuple[DimFactorWithPower, ...]
    is_constant: bool


DimTermWithCoeff: TypeAlias = tuple[DimTermLike, int]


@runtime_checkable
class DimExprLike(Protocol):
    _sorted_terms: tuple[DimTermWithCoeff, ...]


def is_dim_expr(value: Any) -> TypeGuard[DimExprLike]:
    """Return True if ``value`` looks like a JAX symbolic dimension expression."""
    return isinstance(value, DimExprLike)


def dim_expr_constant_value(value: Any) -> int | None:
    """Best-effort conversion of ints or DimExpr objects to plain Python ints."""
    if isinstance(value, numbers.Integral):
        return int(value)
    if is_dim_expr(value):
        try:
            text = str(value).strip()
        except Exception:
            return None
        if text.lstrip("-").isdigit():
            return int(text)
    return None


def symbolic_dim_eq(lhs: Any, rhs: Any) -> bool:
    """Heuristic equality test between symbolic dims (DimExpr, ints, strings, etc.)."""
    if lhs is rhs:
        return True
    if lhs == rhs:  # covers matching strings or ints
        return True
    lhs_const = dim_expr_constant_value(lhs)
    rhs_const = dim_expr_constant_value(rhs)
    if lhs_const is not None and rhs_const is not None:
        return lhs_const == rhs_const
    try:
        return str(lhs) == str(rhs)
    except Exception:
        return False


def is_symbolic_dim(value: Any) -> bool:
    """Whether ``value`` behaves like a symbolic dimension (DimExpr or string label)."""
    return is_dim_expr(value) or isinstance(value, str)


__all__ = [
    "DimExprLike",
    "DimTermLike",
    "DimFactorLike",
    "DimFactorWithPower",
    "DimTermWithCoeff",
    "dim_expr_constant_value",
    "is_dim_expr",
    "is_symbolic_dim",
    "symbolic_dim_eq",
]
