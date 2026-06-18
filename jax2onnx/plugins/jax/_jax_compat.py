# jax2onnx/plugins/jax/_jax_compat.py

"""Compatibility helpers for JAX APIs that are moving across releases."""

from __future__ import annotations

from typing import Any

from jax import core as jax_core

try:  # JAX 0.10+ exposes this through jax.errors.
    from jax.errors import InconclusiveDimensionOperation
except ImportError:  # pragma: no cover - compatibility with older JAX versions
    from jax.core import InconclusiveDimensionOperation
from jax.extend import core as jax_core_ext
from jax.extend.core import ClosedJaxpr, Jaxpr, JaxprEqn, Literal, Primitive
from jax.interpreters import ad, batching


AbstractValue = jax_core.AbstractValue
ShapedArray = jax_core.ShapedArray
Tracer = jax_core.Tracer
Var = jax_core_ext.Var


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


__all__ = [
    "AbstractValue",
    "ClosedJaxpr",
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
    "dim_constant",
    "ensure_batching_not_mapped_attr",
    "jax_core",
    "jax_core_ext",
]
