# jax2onnx/_compat/jax.py

"""Compatibility helpers for JAX APIs that are moving across releases."""

from __future__ import annotations

from contextlib import contextmanager
import re
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Callable, cast

from jax import core as jax_core

try:  # JAX 0.10+ exposes this through jax.errors.
    from jax.errors import InconclusiveDimensionOperation
except ImportError:  # pragma: no cover - compatibility with older JAX versions
    from jax.core import InconclusiveDimensionOperation
from jax.extend import core as jax_core_ext
from jax.extend.core import ClosedJaxpr, Jaxpr, JaxprEqn, Primitive
from jax.interpreters import ad, batching
from jax._src.random.prng import KeyTy as PRNGKeyDType

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


def is_unspecified_jit_sharding(value: object) -> bool:
    """Recognize JAX's actual native plain-JIT sharding sentinel."""
    from jax._src.sharding_impls import UNSPECIFIED

    return value is UNSPECIFIED


def is_empty_jit_mesh(value: object) -> bool:
    """Inspect the concrete Mesh used by native plain JIT on both stacks."""
    from jax._src.mesh import Mesh

    return isinstance(value, Mesh) and bool(value.empty)


def native_jit_boolean_inline_hints() -> tuple[object, object]:
    """Return the native representations of the ordinary False/True hints.

    JAX 0.11 canonicalizes them to Inline.AUTO/JAX_LATE. Do not coerce or
    rewrite the original equation, and do not infer support for XLA options.
    """
    from jax._src import api

    inline = getattr(api, "Inline", None)
    if inline is None:
        return False, True
    return inline.AUTO, inline.JAX_LATE


def is_selected_threefry_impl(value: object) -> bool:
    """Require the installed implementation object, never an equal descriptor."""
    from jax._src.random.threefry2x32 import threefry_prng_impl

    return value is threefry_prng_impl


def is_selected_prng_key_dtype(dtype: object) -> bool:
    """Recognize only the native scalar-key element type selected by this packet."""
    return type(dtype) is PRNGKeyDType and is_selected_threefry_impl(
        cast(PRNGKeyDType, dtype)._impl
    )


def selected_prng_key_words(value: Any) -> Any:
    """Unwrap an actual selected typed key before NumPy attempts conversion."""
    from jax import random

    if not is_selected_prng_key_dtype(getattr(value, "dtype", None)):
        raise ValueError("requires the actual native threefry2x32 typed key")
    if tuple(value.shape) != ():
        raise ValueError("requires a scalar logical key with two uint32 words")
    return random.key_data(value)


def selected_prng_key_from_words(words: Any) -> Any:
    """Restore the selected logical key from its explicitly declared storage."""
    from jax import random
    from jax._src.random.threefry2x32 import threefry_prng_impl

    return random.wrap_key_data(words, impl=threefry_prng_impl)


def is_unpartitioned_prng_placement(value: object, rank: int) -> bool:
    """Recognize the actual absent or empty AbstractMesh native placement."""
    from jax.sharding import NamedSharding, PartitionSpec
    from jax._src.mesh import AbstractMesh

    return value is None or (
        isinstance(value, NamedSharding)
        and type(value) is NamedSharding
        and type(value.mesh) is AbstractMesh
        and value.mesh.empty
        and value.spec == PartitionSpec(*(None for _ in range(rank)))
    )


def has_selected_prng_context(eqn: object) -> bool:
    """Require actual native equation provenance for the partitionable algorithm."""
    from jax._src.core import JaxprEqnContext

    context = getattr(eqn, "ctx", None)
    return (
        type(context) is JaxprEqnContext
        and cast(JaxprEqnContext, context).threefry_partitionable is True
    )


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


def atomic_symbolic_dimension(value: object) -> tuple[str, object]:
    """Inspect one unconstrained JAX dimension without discarding its scope.

    These private inspection seams agree on both supported JAX stacks. Unknown
    forms fail closed; printing an expression is never proof of a variable.
    """
    from jax import export

    if not export.is_symbolic_dim(value):
        raise ValueError("dimension is not a native JAX symbolic variable")
    to_var = getattr(value, "_to_var", None)
    name = to_var() if callable(to_var) else None
    scope = getattr(value, "scope", None)
    constraints = getattr(scope, "_explicit_constraints", None)
    if (
        not isinstance(name, str)
        or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) is None
        or not isinstance(scope, export.SymbolicScope)
        or not isinstance(constraints, (list, tuple))
        or constraints
    ):
        raise ValueError("requires an atomic JAX variable in an unconstrained scope")
    return name, scope


def ensure_batching_not_mapped_attr() -> Any:
    """Expose the legacy ``batching.not_mapped`` name when JAX uses ``None``."""

    try:
        return batching.not_mapped
    except AttributeError:
        setattr(batching, "not_mapped", None)
        return None


NOT_MAPPED: Any = ensure_batching_not_mapped_attr()


def _export_trace_state() -> Any:
    # JAX's control-flow tracing caches consult trace_context, rather than
    # Python ContextVars. Keep this internal API at the compatibility boundary.
    # State's constructor and these options agree on both supported stacks.
    from jax._src import config

    return config.State(
        name="jax2onnx_export_trace",
        default=None,
        help="Private jax2onnx tracing-cache namespace.",
        include_in_trace_context=True,
        include_in_jit_key=True,
    )


_EXPORT_TRACE_STATE = _export_trace_state()


@contextmanager
def export_trace_scope(policy: str) -> Iterator[None]:
    """Separate ordinary JAX, legacy export, and source export trace caches."""
    with _EXPORT_TRACE_STATE(policy):
        yield


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
    "atomic_symbolic_dimension",
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
