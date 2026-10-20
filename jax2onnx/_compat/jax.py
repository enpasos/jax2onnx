# jax2onnx/_compat/jax.py

"""Compatibility helpers for JAX APIs that are moving across releases."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import re
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Callable, cast

import numpy as np

from jax import Array
from jax import core as jax_core

try:  # JAX 0.10+ exposes this through jax.errors.
    from jax.errors import InconclusiveDimensionOperation
except ImportError:  # pragma: no cover - compatibility with older JAX versions
    from jax.core import InconclusiveDimensionOperation
from jax.extend import core as jax_core_ext
from jax.extend.core import ClosedJaxpr, Jaxpr, JaxprEqn, Primitive
from jax.extend.linear_util import WrappedFun as WrappedFun
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


def native_remat_primitive() -> Primitive:
    """Return native checkpoint identity without creating a primitive by name."""
    from jax._src.ad_checkpoint import remat_p

    return cast(Primitive, remat_p)


def is_native_precision_pair(value: object) -> bool:
    """Recognize JAX's two native Precision enums, excluding algorithms/presets."""
    from jax import lax

    return (
        isinstance(value, tuple)
        and len(value) == 2
        and all(type(item) is lax.Precision for item in value)
    )


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


_KeyReuseMask = bool | tuple[tuple[int, ...], bytes]


@dataclass(frozen=True)
class FunctionKeyReuse:
    """Immutable key effects of one actual SOURCE Function body.

    Numeric masks are stored as bytes, with no Jaxpr, captured constant, owner,
    tracer or writable NumPy array retained by primitive metadata.
    """

    sinks: tuple[tuple[int, _KeyReuseMask], ...]
    sources: tuple[tuple[int, _KeyReuseMask], ...]
    forwards: tuple[tuple[int, int], ...]
    failed: bool = False
    error: str | None = field(default=None, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.failed:
            if self.error is None or self.sinks or self.sources or self.forwards:
                raise ValueError(
                    "Failed Function key analysis cannot advertise usable effects"
                )
        elif self.error is not None:
            raise ValueError(
                "Function key diagnostic requires a failed analysis marker"
            )

    def reorder_inputs(self, order: tuple[int, ...]) -> FunctionKeyReuse:
        if tuple(sorted(order)) != tuple(range(len(order))):
            raise ValueError("Function key signature input order must be a permutation")
        positions = {old: new for new, old in enumerate(order)}
        return FunctionKeyReuse(
            tuple(sorted((positions[index], mask) for index, mask in self.sinks)),
            self.sources,
            tuple(
                sorted((positions[source], target) for source, target in self.forwards)
            ),
            failed=self.failed,
            error=self.error,
        )


class FunctionKeyCaptureError(ValueError):
    """An NNX state boundary left a logical key outside its explicit operands."""


def _freeze_key_mask(mask: Any) -> _KeyReuseMask:
    if isinstance(mask, (bool, np.bool_)):
        return bool(mask)
    array = np.asarray(mask)
    if array.dtype != np.dtype(np.bool_):
        raise TypeError("Function key reuse masks must be Boolean")
    return tuple(array.shape), array.tobytes()


def function_key_reuse_signature(closed: Any) -> FunctionKeyReuse:
    """Analyze the temporary native body using JAX's experimental checker.

    The experimental module has no public primitive-registration API in either
    supported stack. Keep this deliberately private dependency here, rather than
    importing its implementation in generic binding or framework plugins.
    """
    from jax.experimental.key_reuse import _core as key_reuse
    from jax.errors import KeyReuseError

    body = closed.jaxpr
    if any(isinstance(var.aval.dtype, PRNGKeyDType) for var in body.constvars):
        raise FunctionKeyCaptureError(
            "NNX Function key-bearing captured constants require explicit state operands"
        )
    try:
        native = key_reuse.jaxpr_type_signature(body)
    except KeyReuseError as exc:
        # Analysis must not turn a caller's debug-off program into an always-on
        # profile rejection. Keep the actual failure, without its traceback or
        # body; JAX's registered checker raises it when checking is requested.
        return FunctionKeyReuse((), (), (), failed=True, error=str(exc))
    return FunctionKeyReuse(
        tuple(sorted((sink.idx, _freeze_key_mask(sink.mask)) for sink in native.sinks)),
        tuple(
            sorted(
                (source.idx, _freeze_key_mask(source.mask)) for source in native.sources
            )
        ),
        tuple(sorted((forward.in_idx, forward.out_idx) for forward in native.forwards)),
    )


_FUNCTION_KEY_RULE_PARAMETERS: dict[Primitive, str] = {}


def detach_function_key_output_aliases(
    inputs: tuple[Any, ...], outputs: list[Any], signature: FunctionKeyReuse
) -> list[Any]:
    """Keep eager Forward accounting from consuming the input wrapper itself.

    The experimental checker marks output wrappers consumed before applying
    Forward. An exact input/output object alias would therefore overwrite the
    input's original consumption state. Public key.copy() preserves that state;
    it does not declare a fresh key. Duplicate outputs retain one shared copy,
    so a second consumption of the same returned key still fails. The native
    copy method does not retain the original source-info traceback metadata.
    """
    copied: dict[int, Any] = {}
    result = list(outputs)
    for _, target in signature.forwards:
        value = outputs[target]
        if (
            isinstance(value, Array)
            and isinstance(value.dtype, PRNGKeyDType)
            and any(value is argument for argument in inputs)
        ):
            identity = id(value)
            if identity not in copied:
                copied[identity] = value.copy()
            result[target] = copied[identity]
    return result


def register_function_key_reuse_rule(primitive: Primitive, parameter: str) -> None:
    """Read per-equation effects; metadata-absent calls keep native unknown rules."""
    from jax.experimental.key_reuse import _core as key_reuse
    from jax.errors import KeyReuseError

    previous = _FUNCTION_KEY_RULE_PARAMETERS.get(primitive)
    if previous == parameter:
        return
    if previous is not None:
        raise ValueError(
            "Function primitive already has a different key-signature parameter"
        )

    def signature(eqn: Any) -> Any:
        metadata = eqn.params.get(parameter)
        if metadata is None:
            return key_reuse.unknown_signature(eqn)
        if not isinstance(metadata, FunctionKeyReuse):
            raise TypeError("Function primitive key signature has invalid metadata")
        if metadata.failed:
            assert metadata.error is not None
            raise KeyReuseError(metadata.error)

        def check_key(variables: Any, index: int) -> Any:
            if not 0 <= index < len(variables):
                raise ValueError("Function key signature port is outside the equation")
            aval = variables[index].aval
            if not isinstance(aval.dtype, PRNGKeyDType):
                raise TypeError("Function key signature refers to a non-key port")
            return aval

        def decode(mask: _KeyReuseMask, aval: Any) -> Any:
            if isinstance(mask, bool):
                return mask
            shape, payload = mask
            array: np.ndarray[Any, np.dtype[np.bool_]] = np.frombuffer(
                payload, dtype=np.bool_
            ).reshape(shape)
            if tuple(array.shape) != tuple(aval.shape):
                raise ValueError(
                    "Function key reuse mask disagrees with its logical key shape"
                )
            return array

        sinks = []
        sources = []
        forwards = []
        for index, mask in metadata.sinks:
            aval = check_key(eqn.invars, index)
            sinks.append(key_reuse.Sink(index, decode(mask, aval)))
        for index, mask in metadata.sources:
            aval = check_key(eqn.outvars, index)
            sources.append(key_reuse.Source(index, decode(mask, aval)))
        for source, target in metadata.forwards:
            before = check_key(eqn.invars, source)
            after = check_key(eqn.outvars, target)
            if before.dtype != after.dtype or before.shape != after.shape:
                raise ValueError(
                    "Function key forward ports have different logical types"
                )
            forwards.append(key_reuse.Forward(source, target))
        return key_reuse.KeyReuseSignature(*sinks, *sources, *forwards)

    key_reuse.key_reuse_signatures[primitive] = key_reuse.DynamicKeyReuseSignature(
        signature
    )
    _FUNCTION_KEY_RULE_PARAMETERS[primitive] = parameter


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


def native_scan_flat_tree_partitions(
    params: Any,
) -> tuple[tuple[int, ...], tuple[int, ...], bool] | None:
    """Inspect JAX 0.11 roles and whether the current lowerer can partition them.

    Older supported scans use integer counts and need no flat-tree imports.
    Counts alone do not prove the output partition or descriptor ownership.
    """
    if "ft_in" not in params:
        return None
    from jax._src.flattree import FlatTree

    descriptors = (params["ft_in"], params["ft_out"])
    result = []
    supported = True
    for index, (descriptor, count) in enumerate(zip(descriptors, (3, 2), strict=True)):
        if not isinstance(descriptor, FlatTree):
            raise ValueError("requires actual native scan input/output flat trees")
        groups = getattr(descriptor, "elts", None)
        if groups is None:
            # Native filtered/pytree descriptors are valid JAX forms. Only
            # ft_in requires .elts in the existing lowerer's scan_arity path.
            groups = descriptor.unpack()
            supported = supported and index != 0
        if (
            not isinstance(groups, tuple)
            or len(groups) != count
            or not all(isinstance(group, FlatTree) for group in groups)
        ):
            raise ValueError("requires actual native scan input/output role groups")
        counts = tuple(len(group) for group in groups)
        if sum(counts) != len(descriptor):
            raise ValueError("scan flat-tree groups disagree with their leaf count")
        result.append(counts)
    return result[0], result[1], supported


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
