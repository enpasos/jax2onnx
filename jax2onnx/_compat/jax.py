# jax2onnx/_compat/jax.py

"""Compatibility helpers for JAX APIs that are moving across releases."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, cast

from jax import core as jax_core

try:  # JAX 0.10+ exposes this through jax.errors.
    from jax.errors import InconclusiveDimensionOperation
except ImportError:  # pragma: no cover - compatibility with older JAX versions
    from jax.core import InconclusiveDimensionOperation
from jax.extend import core as jax_core_ext
from jax.extend.core import ClosedJaxpr, Jaxpr, JaxprEqn, Primitive
from jax.interpreters import ad, batching


AbstractValue = jax_core.AbstractValue
ShapedArray = jax_core.ShapedArray
Tracer = jax_core.Tracer
Var = jax_core_ext.Var
try:
    DropVar = jax_core_ext.DropVar
except AttributeError:  # pragma: no cover - compatibility with older JAX versions
    DropVar = jax_core.DropVar


def fresh_var_like(var: Any) -> Any:
    """Return a fresh ``Var`` while preserving supported quantization metadata.

    JAX 0.11 reduced ``Var`` construction to the abstract value alone. Older
    supported releases expose optional quantization metadata on both the value
    and constructor, so forward only fields that exist on the active object.
    """

    metadata = {
        name: getattr(var, name)
        for name in ("initial_qdd", "final_qdd")
        if hasattr(var, name)
    }
    return Var(var.aval, **metadata)


def _resolve_literal_type() -> type[Any]:
    literal_type = getattr(jax_core_ext, "Literal", None)
    if isinstance(literal_type, type):
        return cast(type[Any], literal_type)
    literal_type = getattr(jax_core, "Literal", None)
    if isinstance(literal_type, type):
        return cast(type[Any], literal_type)
    from jax._src import core as jax_core_src

    return cast(type[Any], jax_core_src.Literal)


if TYPE_CHECKING:
    from jax.extend.core import Literal as Literal
else:
    Literal = _resolve_literal_type()


def _resolve_concrete_or_error() -> Callable[..., Any]:
    # JAX 0.11 removed ``jax.core.concrete_or_error`` in favour of the
    # ``jax.extend.core`` re-export.
    fn = getattr(jax_core_ext, "concrete_or_error", None)
    if callable(fn):
        return cast(Callable[..., Any], fn)
    return cast(Callable[..., Any], getattr(jax_core, "concrete_or_error"))


concrete_or_error: Callable[..., Any] = _resolve_concrete_or_error()


def _resolve_new_jaxpr_eqn() -> Callable[..., Any]:
    # JAX 0.11 removed ``jax.core.new_jaxpr_eqn``; ``jax.extend.core`` carries it.
    fn = getattr(jax_core_ext, "new_jaxpr_eqn", None)
    if callable(fn):
        return cast(Callable[..., Any], fn)
    return cast(Callable[..., Any], getattr(jax_core, "new_jaxpr_eqn"))


new_jaxpr_eqn: Callable[..., Any] = _resolve_new_jaxpr_eqn()


def _resolve_abstract_token() -> type[Any]:
    # ``AbstractToken`` sits on ``jax.core`` up to JAX 0.10 and moved to
    # ``jax.extend.core``; JAX 0.11 dropped the ``jax.core`` alias.
    token = getattr(jax_core_ext, "AbstractToken", None)
    if isinstance(token, type):
        return cast(type[Any], token)
    return cast(type[Any], getattr(jax_core, "AbstractToken"))


AbstractToken: type[Any] = _resolve_abstract_token()


def scan_arity(params: Any, total_invars: int) -> tuple[int, int, int]:
    """Return ``(num_consts, num_carry, num_xs)`` for a ``scan`` equation.

    JAX 0.11 dropped the ``num_consts``/``num_carry`` scan params in favour of
    ``ft_in``, a ``(consts, carry, xs)`` triple whose entries carry one item per
    operand. Older releases only expose the integer counts.
    """

    ft_in = params.get("ft_in")
    if ft_in is not None:
        # ``ft_in`` is a flat tree: iterating it yields leaves, so the
        # ``(consts, carry, xs)`` grouping lives on ``.elts``.
        groups = getattr(ft_in, "elts", ft_in)
        if len(groups) != 3:
            raise ValueError(f"Unexpected scan ft_in arity: {len(groups)}")
        num_consts, num_carry, num_xs = (len(group) for group in groups)
    else:
        num_carry = int(params.get("num_carry", 0))
        num_consts = int(params.get("num_consts", 0) or 0)
        num_xs = int(params.get("num_xs", total_invars - num_carry - num_consts))
    if num_xs < 0 or (num_consts + num_carry + num_xs) != total_invars:
        raise ValueError(
            "Inconsistent Scan arity: expected consts/carry/scan to match jaxpr invars"
        )
    return num_consts, num_carry, num_xs


def dim_constant(value: int) -> Any:
    """Return a symbolic dimension constant when the active JAX exposes one."""

    dim_constant_fn = getattr(jax_core, "dim_constant", None)
    if callable(dim_constant_fn):
        return dim_constant_fn(value)
    return value


def ensure_batching_not_mapped_attr() -> Any:
    """Expose the legacy ``batching.not_mapped`` name when JAX uses ``None``."""

    try:
        return batching.not_mapped
    except AttributeError:
        setattr(batching, "not_mapped", None)
        return None


NOT_MAPPED: Any = ensure_batching_not_mapped_attr()


def _resolve_definitely_equal_shape() -> Callable[[Any, Any], bool]:
    try:  # Prefer the internal helper when available (moved in newer JAX versions).
        from jax._src import core as jax_core_src

        return cast(Callable[[Any, Any], bool], jax_core_src.definitely_equal_shape)
    except Exception:  # pragma: no cover - fallback for older/older-stub JAX
        try:
            return cast(
                Callable[[Any, Any], bool],
                getattr(jax_core, "definitely_equal_shape"),
            )
        except Exception:  # pragma: no cover - minimal fallback

            def fallback(s1: Any, s2: Any) -> bool:
                if len(s1) != len(s2):
                    return False
                for d1, d2 in zip(s1, s2):
                    if d1 is d2:
                        continue
                    try:
                        if d1 != d2:
                            return False
                    except Exception:
                        return False
                return True

            return fallback


definitely_equal_shape: Callable[[Any, Any], bool] = _resolve_definitely_equal_shape()


__all__ = [
    "AbstractToken",
    "AbstractValue",
    "ClosedJaxpr",
    "DropVar",
    "InconclusiveDimensionOperation",
    "Jaxpr",
    "JaxprEqn",
    "Literal",
    "NOT_MAPPED",
    "Primitive",
    "ShapedArray",
    "Tracer",
    "Var",
    "ad",
    "batching",
    "concrete_or_error",
    "definitely_equal_shape",
    "dim_constant",
    "ensure_batching_not_mapped_attr",
    "fresh_var_like",
    "jax_core",
    "jax_core_ext",
    "new_jaxpr_eqn",
    "scan_arity",
]
