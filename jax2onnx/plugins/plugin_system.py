# jax2onnx/plugins/plugin_system.py

from __future__ import annotations

import functools
import re
import importlib
import inspect
import logging
import os
import pkgutil
import weakref
from abc import ABC, abstractmethod
from contextlib import contextmanager, ExitStack
from contextvars import ContextVar
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Final,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Set,
    TypeVar,
    Union,
    cast,
)

import jax
import jax.tree_util as jtu
import numpy as np
import onnx_ir as ir

from jax2onnx.plugins._source_function_state import (
    FunctionOperandLayout,
    FunctionStateAdapter,
    FunctionStateCall,
)
from jax2onnx.plugins._patching import (
    AssignSpec,
    MonkeyPatchSpec,
    _snapshot_attribute,
    _forget_owned_patch,
    _has_owned_leaf_patches,
    _installed_attribute,
    _record_owned_patch,
    _suspend_owned_patches,
    apply_patches,
)
from jax2onnx.plugins._function_keys import (
    snapshot_static_value,
    source_static_value_key,
    static_value_key,
)
from jax2onnx.plugins.jax._autodiff_utils import backfill_missing_transpose_rules
from jax2onnx._compat.jax import (
    NOT_MAPPED,
    ClosedJaxpr,
    FunctionKeyCaptureError,
    FunctionKeyReuse,
    JaxprEqn,
    Literal,
    Primitive,
    PRNGKeyDType,
    ShapedArray,
    Tracer,
    Var,
    batching,
    detach_function_key_output_aliases,
    ensure_batching_not_mapped_attr,
    function_key_reuse_signature,
    is_selected_prng_key_dtype,
    register_function_key_reuse_rule,
)
from jax2onnx.converter.function_scope import FunctionScope, FunctionKey
from jax2onnx.converter.precision_policy import (
    PrecisionPolicy,
    active_precision_policy,
    precision_policy_scope,
)
from jax2onnx.converter.source_profile import SourcePrecisionError
from jax2onnx.converter.optimizer_graph_utils import consumers_in_graph
from jax2onnx.converter.lowering_dispatch import (
    lower_jaxpr_with_plugins,
)
from jax2onnx.converter.lowering_registry import PluginRegistry
from jax2onnx.converter.typing_support import (
    LoweringContextProtocol,
    ConverterContext,
    NativeLoweringContext,
    PrimitiveLowering,
    RngTrace,
    SymbolicDimOrigin,
)

logger: logging.Logger = logging.getLogger("jax2onnx.plugins.plugin_system")
_SOURCE_TRACE: str = "__jax2onnx_source_trace"
_SOURCE_STATE: str = "__jax2onnx_source_state"
_SOURCE_KEY_REUSE: str = "__jax2onnx_source_key_reuse"
_FUNCTION_LAYOUT: str = "__jax2onnx_function_layout"


@dataclass(frozen=True)
class _SourceFunctionCall:
    """Exact owned callable, without hashing a bound receiver's array state."""

    original: Callable[..., Any] = field(compare=False, hash=False, repr=False)
    identity: tuple[int, int]
    instance: bool
    receiver: object | None = field(default=None, compare=False, hash=False, repr=False)

    @classmethod
    def capture(
        cls,
        original: Callable[..., Any],
        *,
        instance: bool,
        receiver: object | None,
    ) -> _SourceFunctionCall:
        if inspect.ismethod(original):
            identity = (id(original.__func__), id(original.__self__))
        else:
            identity = (id(original), id(receiver) if receiver is not None else 0)
        if instance and receiver is None:
            raise ValueError("SOURCE Function call has no exact receiver")
        return cls(original, identity, instance, receiver)


@dataclass(frozen=True)
class _SourceFunctionTrace:
    """Call-site facts required when a source function is retraced later."""

    output_avals: tuple[ShapedArray, ...]
    x64_enabled: bool
    default_matmul_precision: str | None
    numpy_dtype_promotion: str
    numpy_rank_promotion: str
    body: object | None = None
    debug_key_reuse: bool = field(
        default_factory=lambda: bool(jax.config.jax_debug_key_reuse)
    )
    call: _SourceFunctionCall | None = None

    @contextmanager
    def activate(self) -> Iterator[None]:
        with (
            jax.enable_x64(self.x64_enabled),
            jax.default_matmul_precision(self.default_matmul_precision),
            jax.numpy_dtype_promotion(self.numpy_dtype_promotion),
            jax.numpy_rank_promotion(self.numpy_rank_promotion),
            jax.debug_key_reuse(self.debug_key_reuse),
        ):
            yield


ensure_batching_not_mapped_attr()

# ------------------------------------------------------------------------------
# Registries and state
# ------------------------------------------------------------------------------

# Deprecated compatibility alias for TYPE_CHECKING-only legacy plugin imports.
_IRBuildContext = LoweringContextProtocol

# Use a small private domain for ONNX functions. Netron shows the "f"
# marker only when it can resolve a FunctionProto in a domain; ORT also
# requires an opset import for non-empty domains.
_FUNCTION_DOMAIN: Final[str] = "custom"


def _sanitize_op_type(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", s)


# Primitive name -> plugin (FunctionPlugin or PrimitiveLeafPlugin instance)
PluginEntry = Union["FunctionPlugin", PrimitiveLowering]
PLUGIN_REGISTRY: PluginRegistry = PluginRegistry()
PatchCallable = Callable[[Callable[..., Any]], Callable[..., Any]]
PatchSpec = tuple[PatchCallable, list[Any], str]

# Qualified target name -> FunctionPlugin (for reference)
ONNX_FUNCTION_PLUGIN_REGISTRY: Dict[str, "FunctionPlugin"] = {}

# Store instance objects for class-based call targets
INSTANCE_MAP2: weakref.WeakValueDictionary[int, Any] = weakref.WeakValueDictionary()

# Abstract-evaluation caches can outlive a weakly stored model instance. Keep
# primitive keys unique across object lifetimes, even when Python reuses an id.
_FUNCTION_INSTANCE_KEY_COUNTER: Iterator[int] = count(1)
_FUNCTION_INSTANCE_IDENTITIES: dict[int, tuple[weakref.ReferenceType[Any], int]] = {}


def _register_function_instance(instance: Any) -> int:
    identity = id(instance)
    entry = _FUNCTION_INSTANCE_IDENTITIES.get(identity)
    if entry is not None and entry[0]() is instance:
        instance_key = entry[1]
    else:
        instance_key = next(_FUNCTION_INSTANCE_KEY_COUNTER)

        def forget(reference: weakref.ReferenceType[Any]) -> None:
            current = _FUNCTION_INSTANCE_IDENTITIES.get(identity)
            if current is not None and current[0] is reference:
                _FUNCTION_INSTANCE_IDENTITIES.pop(identity)

        reference = weakref.ref(instance, forget)
        _FUNCTION_INSTANCE_IDENTITIES[identity] = (reference, instance_key)
    INSTANCE_MAP2[instance_key] = instance
    return instance_key


# Track @onnx_function hits (optional)
_ONNX_FN_HITS: ContextVar[frozenset[str]] = ContextVar(
    "_ONNX_FN_HITS", default=frozenset()
)

# During function body build, prevent that function from rebinding itself
_IN_FUNCTION_BUILD: ContextVar[frozenset[str]] = ContextVar(
    "_IN_FUNCTION_BUILD", default=frozenset()
)

# Optional examples registry (used by docs/test generation tooling)
EXAMPLE_REGISTRY: Dict[str, dict[str, Any]] = {}

# Patching state
_PATCH_STATE: dict[tuple[Any, str], dict[str, Any]] = {}

_RNG_TRACE_REGISTRY: Set[str] = set()


def _register_rng_trace(trace: RngTrace) -> None:
    """Record RNG helpers for CI reporting."""

    _RNG_TRACE_REGISTRY.add(trace.describe())


def list_registered_rng_traces() -> list[str]:
    """Return sorted human-readable RNG helper descriptions."""

    return sorted(_RNG_TRACE_REGISTRY)


def _sanitize_op_type_name(name: str) -> str:
    """Make a string safe for ONNX op_type (letters, digits, underscore)."""
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def _normalize_namespace(namespace: str | None) -> str:
    raw = namespace if namespace is not None else _FUNCTION_DOMAIN
    parts = [part for part in raw.split(".") if part]
    sanitized_parts: list[str] = []
    for part in parts:
        cleaned = _sanitize_op_type_name(part)
        if cleaned:
            sanitized_parts.append(cleaned)
    if not sanitized_parts:
        return _FUNCTION_DOMAIN
    return ".".join(sanitized_parts)


# Discovery guard (missing before → NameError during test generation)
_already_imported_plugins: bool = False


# ------------------------------------------------------------------------------
# Callable factory helpers
# ------------------------------------------------------------------------------


class _FactoryValue:
    """Sentinel wrapping a callable that produces a value given the requested dtype."""

    __slots__ = ("_fn", "_metadata")

    def __init__(
        self,
        fn: Callable[[Any], Any],
        *,
        metadata: Optional[dict[str, Any]] = None,
    ):
        self._fn = fn
        self._metadata = metadata or {}
        trace = self._metadata.get("rng_trace")
        if isinstance(trace, RngTrace):
            _register_rng_trace(trace)

    def resolve(self, dtype: Any) -> Any:  # noqa: D401 - tiny helper
        return self._fn(dtype)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


def _materialize(value: Any, dtype: Any) -> Any:
    """Resolve dtype-dependent values inside nested containers."""

    if isinstance(value, _FactoryValue):
        return value.resolve(dtype)
    if isinstance(value, dict):
        return {k: _materialize(v, dtype) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        seq_type = type(value)
        return seq_type(_materialize(v, dtype) for v in value)
    return value


def make_callable_factory(
    ctor: Callable[..., Any],
    /,
    *ctor_args: Any,
    **ctor_kwargs: Any,
) -> Callable[[Any], Any]:
    """Build a `callable_factory` compatible with tests.t_generator.

    Each invocation materializes fresh constructor arguments, resolving any
    `_FactoryValue` placeholders with the requested dtype.
    """

    def _factory(dtype: Any) -> Any:
        resolved_args = [_materialize(arg, dtype) for arg in ctor_args]
        resolved_kwargs = {k: _materialize(v, dtype) for k, v in ctor_kwargs.items()}
        return ctor(*resolved_args, **resolved_kwargs)

    return _factory


def with_requested_dtype() -> _FactoryValue:
    """Placeholder that resolves to the dtype requested by the test harness."""

    return _FactoryValue(lambda dtype: dtype)


def with_rng_seed(seed: int | Callable[[Any], int]) -> _FactoryValue:
    """Placeholder that builds an `nnx.Rngs` seeded per request."""

    def _build(dtype: Any) -> Any:
        from flax import nnx  # Local import to avoid mandatory dependency during import

        seed_value = seed(dtype) if callable(seed) else seed
        return nnx.Rngs(seed_value)

    trace = RngTrace(kind="nnx_rng", seed=seed if isinstance(seed, int) else None)
    return _FactoryValue(_build, metadata={"rng_trace": trace})


def with_prng_key(seed: int | Callable[[Any], int]) -> _FactoryValue:
    """Placeholder that returns a JAX `PRNGKey` seeded per request."""

    def _build(dtype: Any) -> Any:
        import jax  # Local import to avoid unconditional dependency at module import time

        seed_value = seed(dtype) if callable(seed) else seed
        return jax.random.PRNGKey(seed_value)

    trace = RngTrace(kind="jax_prng", seed=seed if isinstance(seed, int) else None)
    return _FactoryValue(_build, metadata={"rng_trace": trace})


class _DynamicParamWrapper:
    """Legacy hashable value wrapper retained for primitive-specific plugins."""

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value

    def __hash__(self) -> int:  # pragma: no cover - trivial
        return hash(id(self.value))

    def __eq__(self, other: object) -> bool:  # pragma: no cover - trivial
        return isinstance(other, _DynamicParamWrapper) and other.value is self.value


@dataclass(frozen=True)
class _DynamicParamRef:
    """Hashable reference to a keyword value carried as a primitive operand."""

    index: int


@dataclass(frozen=True)
class _StaticParam:
    """Hashable static configuration for JAX's primitive-parameter caches."""

    value: object = field(compare=False, hash=False, repr=False)
    signature: tuple[object, ...]


_Operand = TypeVar("_Operand")


def _pack_function_arguments(
    args: tuple[object, ...], kwargs: Mapping[str, object]
) -> tuple[tuple[object, ...], dict[str, object]]:
    """Encode dynamic dependencies and hashable static configuration for JAX."""
    operands = list(args)
    params: dict[str, object] = {}
    value_key = (
        source_static_value_key
        if active_precision_policy() is PrecisionPolicy.SOURCE
        else static_value_key
    )
    for name, value in kwargs.items():
        if isinstance(value, (Tracer, jax.Array)):
            params[name] = _DynamicParamRef(len(operands))
            operands.append(value)
        else:
            frozen_value = snapshot_static_value(value)
            params[name] = _StaticParam(frozen_value, value_key(frozen_value))
    return tuple(operands), params


def _unpack_function_arguments(
    operands: Sequence[_Operand],
    params: Mapping[str, object],
    *,
    layout: FunctionOperandLayout | None = None,
) -> tuple[tuple[_Operand, ...], dict[str, object]]:
    """Restore positional arguments and kwargs from the shared primitive format."""
    keyword_count = sum(
        isinstance(value, _DynamicParamRef) for value in params.values()
    )
    if layout is None:
        positional_count = len(operands) - keyword_count
    else:
        layout.check(operands)
        positional_count = layout.positional_count
        if keyword_count != layout.argument_count - positional_count:
            raise ValueError(
                "SOURCE Function keyword operand count does not match its layout"
            )
    kwargs: dict[str, object] = {}
    for name, value in params.items():
        if isinstance(value, _DynamicParamRef):
            if layout is not None and not (
                positional_count <= value.index < layout.argument_count
            ):
                raise ValueError(
                    "SOURCE Function keyword refers outside its argument ports"
                )
            kwargs[name] = operands[value.index]
        elif isinstance(value, _StaticParam):
            kwargs[name] = value.value
        else:
            kwargs[name] = value
    return tuple(operands[:positional_count]), kwargs


class _ConstructAndCall:
    __slots__ = ("_ctor", "_args", "_kwargs", "_dtype", "_instance")
    __jax2onnx_factory__ = True

    def __init__(
        self,
        ctor: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        dtype: Any | None,
        instance: Any | None = None,
    ) -> None:
        self._ctor = ctor
        self._args = args
        self._kwargs = kwargs
        self._dtype = dtype
        self._instance = instance

    def with_dtype(self, dtype: Any) -> "_ConstructAndCall":
        return _ConstructAndCall(
            self._ctor,
            self._args,
            self._kwargs,
            dtype,
            instance=None,  # Defer instantiation until explicit `.instantiate()` call
        )

    def instantiate(self) -> Any:
        dtype = self._dtype
        if dtype is None:
            import jax.numpy as jnp  # Local import to avoid mandatory dependency on module import

            dtype = jnp.float32

        resolved_args = [_materialize(arg, dtype) for arg in self._args]
        resolved_kwargs = {k: _materialize(v, dtype) for k, v in self._kwargs.items()}
        instance = self._ctor(*resolved_args, **resolved_kwargs)
        if not callable(instance):
            raise TypeError(
                "construct_and_call expected constructor to return a callable object"
            )
        return instance

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        instance = self._instance
        if instance is None:
            raise RuntimeError(
                "Deferred _ConstructAndCall was called without being instantiated first. "
                "Call `.instantiate()` to resolve the underlying test callable before tracing/executing."
            )
        return instance(*args, **kwargs)


def construct_and_call(
    ctor: Callable[..., Any],
    /,
    *ctor_args: Any,
    **ctor_kwargs: Any,
) -> _ConstructAndCall:
    """Construct a fresh callable on each invocation, then immediately call it."""

    return _ConstructAndCall(ctor, tuple(ctor_args), dict(ctor_kwargs), dtype=None)


# ------------------------------------------------------------------------------
# Primitive plugin base
# ------------------------------------------------------------------------------


class PrimitivePlugin(ABC):
    @abstractmethod
    def get_patch_params(self) -> list[tuple[Any, str, Callable[..., Any]]]:
        raise NotImplementedError


class PrimitiveLeafPlugin(PrimitivePlugin, PrimitiveLowering):
    """Base class for concrete primitive lowerings.

    Implementations are responsible for wiring the underlying JAX primitive:
    define its impl/abstract-eval in the JAX registry *and* register a
    `batching_rule` so tracing under `vmap` succeeds. The converter only calls
    into `lower`; missing batching support shows up as a JAX error long before
    we reach ONNX.
    """

    primitive: str
    metadata: Mapping[str, Any]
    patch_info: Callable[[], dict[str, Any]] | None = None

    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @classmethod
    def ensure_abstract_eval_bound(cls) -> None:
        if getattr(cls, "_ABSTRACT_EVAL_BOUND", False):
            return
        prim = getattr(cls, "_PRIM", None)
        abstract_eval = getattr(cls, "abstract_eval", None)
        if prim is not None and callable(abstract_eval):
            prim.def_abstract_eval(abstract_eval)  # noqa: SLF001
        cls._ABSTRACT_EVAL_BOUND = True

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        return []

    @classmethod
    @contextmanager
    def plugin_binding(cls) -> Iterator[None]:
        cls.ensure_abstract_eval_bound()
        if active_precision_policy() is PrecisionPolicy.SOURCE:
            # SOURCE observes the native program. Do not evaluate leaf factories,
            # including AssignSpec declarations, while retaining registry/AD setup.
            yield
            return
        with apply_patches(cls.binding_specs(), owner=cls):
            yield

    def get_patch_params(self) -> list[tuple[Any, str, Callable[..., Any]]]:
        if not self.patch_info:
            raise ValueError("patch_info is not defined for this plugin.")
        info = self.patch_info()
        targets = info["patch_targets"]
        patch_func = info["patch_function"]
        attr = info.get("target_attribute", "__call__")
        return [(t, attr, patch_func) for t in targets]


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------


def _extract_ir_ctx(converter: Any) -> Any | None:
    """
    Be tolerant about how the IRContext is exposed by the converter facade.
    Different call-sites may attach it under slightly different names.
    """
    # 1) direct attach on the facade
    for attr in ("_ctx", "ctx", "context"):
        if hasattr(converter, attr):
            ctx = object.__getattribute__(converter, attr)
            if ctx is not None:
                return ctx

    # 2) via builder
    b = (
        object.__getattribute__(converter, "builder")
        if hasattr(converter, "builder")
        else None
    )
    if b is None:
        return None
    for attr in ("_ctx", "ctx", "context", "_context", "ir_context", "parent_ctx"):
        if hasattr(b, attr):
            ctx = object.__getattribute__(b, attr)
            if ctx is not None:
                return ctx

    # 3) optional accessor
    getctx = (
        object.__getattribute__(b, "get_context") if hasattr(b, "get_context") else None
    )
    if callable(getctx):
        try:
            return getctx()
        except Exception:
            logger.debug("builder.get_context() failed", exc_info=True)

    return None


@contextmanager
def _activate_full_plugin_worlds_for_body() -> Iterator[None]:
    """
    For nested function-body tracing: activate BOTH
      1) function patches (via apply_monkey_patches), and
      2) leaf plugin bindings (via PrimitiveLeafPlugin.plugin_binding()).
    Mirrors the outer stack used by converter._activate_plugin_worlds().
    """
    import_all_plugins()
    with ExitStack() as stack:
        # Function plugins' monkey patches
        stack.enter_context(apply_monkey_patches())
        # Leaf plugins' binding_specs (e.g., nnx/jnp/lax rewrites)
        leaf_prims: list[Primitive] = []
        for plugin in PLUGIN_REGISTRY.values():
            try:
                if not isinstance(plugin, PrimitiveLeafPlugin):
                    continue
                cls = plugin.__class__
                stack.enter_context(cls.plugin_binding())
                prim = getattr(cls, "_PRIM", None)
                if isinstance(prim, Primitive):
                    leaf_prims.append(prim)
            except Exception:
                # Best-effort; a non-leaf or misconfigured plugin should not crash nested tracing
                logger.debug(
                    "Skipping leaf binding for %r", plugin.__class__, exc_info=True
                )
        backfill_missing_transpose_rules(leaf_prims)
        yield


def _qualname_of_target(target: Any) -> str:
    if inspect.isclass(target):
        return f"{target.__module__}.{target.__name__}"
    elif callable(target):
        mod = inspect.getmodule(target)
        return f"{(mod.__name__ if mod else '<unknown>')}.{target.__name__}"
    else:
        return repr(target)


# ------------------------------------------------------------------------------
# Function plugin (new-world)
# ------------------------------------------------------------------------------


@dataclass(frozen=True)
class DynamicCapture:
    """One ordered function parameter and its already-bound parent value."""

    name: str
    spec: jax.ShapeDtypeStruct
    value: ir.Value


@dataclass(frozen=True)
class FunctionCaptures:
    """Call arguments, kept in signature order for tracing and call emission."""

    dynamic: tuple[DynamicCapture, ...]
    static_kwargs: Mapping[str, object]
    signature: tuple[tuple[str, tuple[object, ...]], ...]


class FunctionPlugin(PrimitivePlugin):
    """
    Wrap a decorated target (class or function) in a JAX Primitive
    ('onnx_fn::<qualified>') and lower each call to an ONNX Function def + call-site.
    """

    def __init__(
        self,
        primitive_name: str,
        target: Any,
        *,
        unique: bool = False,
        namespace: str | None = None,
    ) -> None:
        self.name = primitive_name
        self.target = target
        self.unique = bool(unique)
        self.namespace = _normalize_namespace(namespace)
        # Optional human-readable override for the function base name/op_type.
        # When unset, we fall back to the underlying callable name.
        self.display_name: str | None = None
        self._qualified_target = _qualname_of_target(target)
        self.primitive = Primitive(primitive_name)
        self.primitive.def_abstract_eval(self._abstract_eval_with_kwargs)
        self.primitive.def_impl(self._primitive_impl)
        batching.primitive_batchers[self.primitive] = self._batching_rule
        # A distinct primitive identity keeps source abstract-evaluation caches
        # separate and supports structured results without altering legacy calls.
        self._source_primitive = Primitive(primitive_name)
        self._source_primitive.multiple_results = True
        self._source_primitive.def_abstract_eval(self._source_abstract_eval)
        self._source_primitive.def_impl(self._source_primitive_impl)
        batching.primitive_batchers[self._source_primitive] = self._source_batching_rule
        self._orig_fn: Callable[..., Any] | None = None  # set by patch wrapper
        self._unbound_orig_fn: Callable[..., Any] | None = None

    def _resolve_orig_fn(
        self, instance_key: object | None = None
    ) -> Callable[..., Any] | None:
        if isinstance(instance_key, int):
            instance = INSTANCE_MAP2.get(instance_key)
            unbound = self._unbound_orig_fn
            if instance is not None and unbound is not None:
                return cast(
                    Callable[..., Any],
                    unbound.__get__(instance, type(instance)),
                )
        return self._orig_fn

    # Implement abstract method (used by monkey-patch activator)
    def get_patch_params(self) -> list[tuple[Any, str, Callable[..., Any]]]:
        info = self.patch_info()
        targets = info["patch_targets"]
        patch_func = info["patch_function"]
        attr = info.get("target_attribute", "__call__")
        return [(t, attr, patch_func) for t in targets]

    def patch_info(self) -> dict[str, Any]:
        if inspect.isclass(self.target):
            return {
                "patch_targets": [self.target],
                "patch_function": self._make_patch_fn(self.primitive, is_class=True),
                "target_attribute": "__call__",
            }
        elif callable(self.target):
            mod = inspect.getmodule(self.target)
            return {
                "patch_targets": [mod],
                "patch_function": self._make_patch_fn(self.primitive, is_class=False),
                "target_attribute": self.target.__name__,
            }
        else:
            raise TypeError(
                f"Unsupported target type for patching: {type(self.target)}"
            )

    @staticmethod
    def _aval_to_shaped_array(
        aval: Any, *, preserve_weak_type: bool = False
    ) -> ShapedArray:
        if isinstance(aval, ShapedArray):
            return aval
        if hasattr(aval, "shape") and hasattr(aval, "dtype"):
            return ShapedArray(
                aval.shape,
                aval.dtype,
                weak_type=(
                    bool(getattr(aval, "weak_type", False))
                    if preserve_weak_type
                    else False
                ),
            )
        raise TypeError(
            f"Cannot convert abstract value of type {type(aval)} to ShapedArray."
        )

    def _abstract_eval_with_kwargs(self, *args: Any, **kwargs: Any) -> Any:
        instance_key = kwargs.get("instance_key")
        orig_fn = self._resolve_orig_fn(instance_key)
        if orig_fn is None:
            raise ValueError(f"Original function not set for '{self.name}'")

        specs = [
            (
                jax.ShapeDtypeStruct(arg.shape, arg.dtype)
                if isinstance(arg, ShapedArray)
                else arg
            )
            for arg in args
        ]

        def _call_with_static_kwargs(*operand_args: Any) -> Any:
            positional_args, call_kwargs = _unpack_function_arguments(
                operand_args, kwargs
            )
            call_kwargs.pop("instance_key", None)
            return orig_fn(*positional_args, **call_kwargs)

        out_aval = jax.eval_shape(_call_with_static_kwargs, *specs)
        if isinstance(out_aval, jax.ShapeDtypeStruct):
            out_aval = self._aval_to_shaped_array(out_aval)
        elif isinstance(out_aval, tuple):
            out_aval = tuple(self._aval_to_shaped_array(a) for a in out_aval)
        elif isinstance(out_aval, list):
            out_aval = [self._aval_to_shaped_array(a) for a in out_aval]
        return out_aval

    @staticmethod
    def _source_abstract_eval(*args: Any, **kwargs: Any) -> list[ShapedArray]:
        trace: _SourceFunctionTrace = kwargs[_SOURCE_TRACE]
        return list(trace.output_avals)

    def _source_primitive_impl(self, *args: Any, **kwargs: Any) -> list[Any]:
        trace: _SourceFunctionTrace = kwargs.pop(_SOURCE_TRACE)
        with (
            precision_policy_scope(PrecisionPolicy.SOURCE),
            _activate_full_plugin_worlds_for_body(),
            trace.activate(),
        ):
            outputs = list(
                jtu.tree_leaves(self._source_call(args, kwargs, trace=trace))
            )
        key_reuse = kwargs.get(_SOURCE_KEY_REUSE)
        if isinstance(key_reuse, FunctionKeyReuse):
            outputs = detach_function_key_output_aliases(args, outputs, key_reuse)
        return outputs

    def _source_call(
        self,
        args: Sequence[Any],
        parameters: Mapping[str, Any],
        *,
        trace: _SourceFunctionTrace | None = None,
    ) -> Any:
        parameters = dict(parameters)
        parameters.pop(_SOURCE_KEY_REUSE, None)
        adapter: FunctionStateAdapter | None = parameters.pop(_SOURCE_STATE, None)
        layout: FunctionOperandLayout | None = parameters.pop(_FUNCTION_LAYOUT, None)
        if adapter is None:
            if trace is None or trace.call is None:
                return self._primitive_impl(*args, **parameters)
            parameters.pop("instance_key", None)
            positional, keywords = _unpack_function_arguments(args, parameters)
            return trace.call.original(*positional, **keywords)
        if layout is None or layout.state_count != adapter.state_count:
            raise ValueError("SOURCE Function state is missing its operand layout")
        positional, keywords = _unpack_function_arguments(
            args, parameters, layout=layout
        )
        result, state, _ = adapter.invoke(
            positional, keywords, tuple(args[layout.argument_count :])
        )
        return result, state

    def _source_batching_rule(
        self, args: tuple[Any, ...], dims: tuple[Any, ...], **params: Any
    ) -> tuple[list[Any], list[Any]]:
        trace: _SourceFunctionTrace = params.pop(_SOURCE_TRACE)
        with (
            precision_policy_scope(PrecisionPolicy.SOURCE),
            _activate_full_plugin_worlds_for_body(),
            trace.activate(),
        ):
            values, output_dims = self._batch_call(
                lambda *single: self._source_call(single, params, trace=trace),
                args,
                dims,
            )
        return jtu.tree_leaves(values), jtu.tree_leaves(output_dims)

    def _bind_function_call(
        self,
        primitive: Primitive,
        callee: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        instance_key: int | None = None,
        owner: object | None = None,
        unbound_method: Callable[..., Any] | None = None,
    ) -> Any:
        operands, parameters = _pack_function_arguments(args, kwargs)
        if instance_key is not None:
            parameters["instance_key"] = instance_key
        if active_precision_policy() is PrecisionPolicy.LEGACY:
            return primitive.bind(*operands, **parameters)
        for reserved in (
            _SOURCE_TRACE,
            _SOURCE_STATE,
            _FUNCTION_LAYOUT,
            _SOURCE_KEY_REUSE,
        ):
            if reserved in kwargs:
                raise ValueError(f"{reserved} is reserved for internal tracing")

        adapter: FunctionStateAdapter | None = None
        layout: FunctionOperandLayout | None = None
        if (
            owner is not None
            and unbound_method is not None
            and instance_key is not None
        ):
            from jax2onnx.plugins.flax.nnx._source_function_state import (
                prepare_nnx_function_state,
            )

            prepared = prepare_nnx_function_state(owner, unbound_method, instance_key)
            if prepared is not None:
                adapter, state = prepared
                parameters.pop("instance_key", None)
                layout = FunctionOperandLayout(len(args), len(operands), len(state))
                operands = (*operands, *state)
        observed_effects: list[tuple[int, ...]] = []

        def source_call(*operand_args: Any) -> Any:
            positional, call_kwargs = _unpack_function_arguments(
                operand_args, parameters, layout=layout
            )
            call_kwargs.pop("instance_key", None)
            if adapter is None:
                return callee(*positional, **call_kwargs)
            assert layout is not None
            result, state, effects = adapter.invoke(
                positional, call_kwargs, tuple(operand_args[layout.argument_count :])
            )
            observed_effects.append(effects)
            return result, state

        call_x64 = bool(jax.config.x64_enabled)
        call_matmul_precision = jax.config.jax_default_matmul_precision
        call_dtype_promotion = str(jax.config.jax_numpy_dtype_promotion)
        call_rank_promotion = str(jax.config.jax_numpy_rank_promotion)
        call_debug_key_reuse = bool(jax.config.jax_debug_key_reuse)
        from jax2onnx.converter._source_regions import freeze_closed_body

        closed_body, outputs = jax.make_jaxpr(source_call, return_shape=True)(*operands)
        original_body = freeze_closed_body(closed_body)
        if adapter is not None:
            parameters[_SOURCE_KEY_REUSE] = self._source_key_reuse(closed_body)
            register_function_key_reuse_rule(self._source_primitive, _SOURCE_KEY_REUSE)
        del closed_body
        if adapter is not None:
            if len(observed_effects) != 1:
                raise ValueError(
                    "SOURCE Function abstract trace did not record its state effects once"
                )
            adapter = adapter.with_effects(observed_effects[0])
            adapter.validate_owner(owner)
            parameters[_SOURCE_STATE] = adapter
            parameters[_FUNCTION_LAYOUT] = layout
        leaves, result_tree = jtu.tree_flatten(outputs)
        parameters[_SOURCE_TRACE] = _SourceFunctionTrace(
            tuple(
                self._aval_to_shaped_array(leaf, preserve_weak_type=True)
                for leaf in leaves
            ),
            x64_enabled=call_x64,
            default_matmul_precision=call_matmul_precision,
            numpy_dtype_promotion=call_dtype_promotion,
            numpy_rank_promotion=call_rank_promotion,
            body=original_body,
            debug_key_reuse=call_debug_key_reuse,
            call=(
                _SourceFunctionCall.capture(
                    callee,
                    instance=instance_key is not None,
                    receiver=owner,
                )
                if adapter is None
                else None
            ),
        )
        results = self._source_primitive.bind(*operands, **parameters)
        restored = jtu.tree_unflatten(result_tree, results)
        if adapter is not None:
            result, state = restored
            adapter.update_owner(owner, tuple(state))
            return result
        return restored

    def _source_key_reuse(self, closed: ClosedJaxpr) -> FunctionKeyReuse:
        try:
            return function_key_reuse_signature(closed)
        except FunctionKeyCaptureError as exc:
            raise SourcePrecisionError(
                str(exc),
                phase="profile-validation",
                category="profile-exclusion",
                location=self.name,
                primitive=self.name,
                requirement="NNX Function keys must be carried explicitly through user or owner-state operands",
                action="prepare the captured key as explicit NNX state or a call operand",
            ) from exc

    def _primitive_impl(self, *args: Any, **kwargs: Any) -> Any:
        instance_key = kwargs.pop("instance_key", None)
        orig_fn = self._resolve_orig_fn(instance_key)
        if orig_fn is None:
            raise ValueError("Original function not set for primitive!")
        positional_args, call_kwargs = _unpack_function_arguments(args, kwargs)
        return orig_fn(*positional_args, **call_kwargs)

    def _batching_rule(
        self, args: tuple[Any, ...], dims: tuple[Any, ...], **params: Any
    ) -> tuple[Any, Any]:
        instance_key = params.pop("instance_key", None)
        original_fn = self._resolve_orig_fn(instance_key)

        if original_fn is None:
            raise NotImplementedError(
                f"Batching rule for '{self.name}' missing original callable."
            )

        def _call_single(*single_args: Any) -> Any:
            positional_args, call_kwargs = _unpack_function_arguments(
                single_args, params
            )
            return original_fn(*positional_args, **call_kwargs)

        return self._batch_call(_call_single, args, dims)

    @staticmethod
    def _batch_call(
        call: Callable[..., Any], args: Sequence[Any], dims: Sequence[Any]
    ) -> tuple[Any, Any]:
        axis_size: Any | None = None
        for arg, bdim in zip(args, dims):
            if bdim is NOT_MAPPED:
                continue
            shape = getattr(arg, "shape", None)
            if shape is None or bdim >= len(shape):
                continue
            axis_size = shape[bdim]
            break

        if axis_size is None:
            result = call(*args)
            out_dims = jtu.tree_map(lambda _: NOT_MAPPED, result)
            return result, out_dims

        prepared_args = [
            batching.bdim_at_front(arg, bdim, axis_size)
            for arg, bdim in zip(args, dims)
        ]

        batched_result = jax.vmap(call)(*prepared_args)
        out_dims = jtu.tree_map(lambda _: 0, batched_result)
        return batched_result, out_dims

    def _make_patch_fn(self, primitive: Primitive, is_class: bool) -> PatchCallable:
        def patch(original_call: Callable[..., Any]) -> Callable[..., Any]:
            # Registered class targets already supply a receiver. Signature
            # inspection would unnecessarily materialize lazy annotations.
            params = (
                [] if is_class else list(inspect.signature(original_call).parameters)
            )
            self._unbound_orig_fn = original_call

            @functools.wraps(original_call)
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                expects_self = is_class or (params and params[0] == "self")
                if expects_self:
                    instance = args[0]
                    instance_key = _register_function_instance(instance)
                    bound_orig = original_call.__get__(instance, type(instance))
                    prev_orig_fn = self._orig_fn
                    self._orig_fn = bound_orig
                    # If we are currently constructing this function's body, do NOT emit
                    # the function primitive again—call through so inner patches take effect.
                    if self.name in _IN_FUNCTION_BUILD.get():
                        try:
                            return bound_orig(*args[1:], **kwargs)
                        finally:
                            self._orig_fn = prev_orig_fn
                    # Record a hit (for optional test bookkeeping) and bind the primitive.
                    _ONNX_FN_HITS.set(
                        _ONNX_FN_HITS.get() | {self.name.split("::", 1)[1]}
                    )
                    call_kwargs = {
                        key: value
                        for key, value in kwargs.items()
                        if key != "instance_key"
                    }
                    restore_owner = False
                    if active_precision_policy() is PrecisionPolicy.SOURCE:
                        from jax2onnx.plugins.flax.nnx._source_function_state import (
                            is_nnx_function_owner,
                        )

                        restore_owner = is_nnx_function_owner(instance)
                    try:
                        return self._bind_function_call(
                            primitive,
                            bound_orig,
                            args[1:],
                            call_kwargs,
                            instance_key=instance_key,
                            owner=instance,
                            unbound_method=original_call,
                        )
                    finally:
                        if restore_owner:
                            self._orig_fn = prev_orig_fn
                else:
                    self._orig_fn = original_call
                    if self.name in _IN_FUNCTION_BUILD.get():
                        return original_call(*args, **kwargs)
                    _ONNX_FN_HITS.set(
                        _ONNX_FN_HITS.get() | {self.name.split("::", 1)[1]}
                    )
                    return self._bind_function_call(
                        primitive, original_call, args, kwargs
                    )

            return wrapped

        return patch

    def _friendly_name_base(self) -> str:
        """Human-readable base name for this function (class or function name)."""
        override = getattr(self, "display_name", None)
        if isinstance(override, str) and override:
            return override
        tgt = self.target
        try:
            if inspect.isclass(tgt):
                return tgt.__name__ or "Function"
            if callable(tgt):
                return getattr(tgt, "__name__", "Function")
        except Exception:
            pass
        return "Function"

    def _fingerprint_instance_state(
        self,
        instance: Any,
        *,
        value_key: Callable[[object], tuple[object, ...]] = static_value_key,
    ) -> tuple[Any, ...]:
        if instance is None:
            return (("instance", "none"),)
        components: list[tuple[Any, ...]] = []
        try:
            leaves, treedef = jtu.tree_flatten(instance)
        except Exception:
            leaves = None
            treedef = None
        else:
            components.append(("treedef", treedef))
        if leaves:
            for idx, leaf in enumerate(leaves):
                components.append(("leaf", idx, value_key(leaf)))
        else:
            state = getattr(instance, "__dict__", None)
            if isinstance(state, dict):
                for key in sorted(state):
                    components.append(("attr", key, value_key(state[key])))
            else:
                components.append(("opaque", value_key(instance)))
        return tuple(components)

    def _build_unique_signature(
        self,
        callee: object,
        capture_items: tuple[tuple[str, tuple[object, ...]], ...],
        *,
        value_key: Callable[[object], tuple[object, ...]] = static_value_key,
    ) -> tuple[object, ...]:
        signature_parts: list[tuple[object, ...]] = [
            ("target", self._qualified_target),
            ("captures", tuple(capture_items)),
        ]
        if inspect.isclass(self.target):
            inst_type = type(callee)
            signature_parts.append(
                (
                    "instance_type",
                    f"{inst_type.__module__}.{getattr(inst_type, '__qualname__', inst_type.__name__)}",
                )
            )
            signature_parts.append(
                (
                    "instance_state",
                    self._fingerprint_instance_state(callee, value_key=value_key),
                )
            )
        else:
            if callee is None:
                signature_parts.append(("callable_module", "<unknown>"))
                signature_parts.append(("callable_name", "<none>"))
            else:
                call_ident = getattr(
                    callee, "__qualname__", getattr(callee, "__name__", repr(callee))
                )
                module_name = getattr(callee, "__module__", "<unknown>")
                signature_parts.append(("callable_module", module_name))
                signature_parts.append(("callable_name", call_ident))
        return tuple(signature_parts)

    def lower(self, ctx: ConverterContext, eqn: JaxprEqn) -> None:
        """Lower directly in the context and bind every function-call output."""
        self._lower_in_context(ctx, eqn, eqn.params)

    def get_handler(self, converter: Any) -> Callable[[Any, Any, dict[str, Any]], Any]:
        """Retain the converter-facade entry point for legacy callers."""
        return self._lower_and_call

    def _allocate_friendly_name(self, ctx: ConverterContext) -> tuple[str, str]:
        """
        Produce the human-readable FunctionProto identifiers.

        Returns
        -------
        tuple[str, str]
            (op_type, domain) where `op_type` preserves the original callable
            name and `domain` carries the per-instance suffix to keep the
            (domain, name) pair unique inside the model.
        """
        base = _sanitize_op_type_name(self._friendly_name_base())
        namespace = self.namespace or _FUNCTION_DOMAIN
        kind = "unique" if self.unique else "shared"
        counter_key = (namespace, base, kind)
        idx = ctx._func_name_counters.get(counter_key, 0) + 1
        ctx._func_name_counters[counter_key] = idx
        if self.unique:
            if idx == 1:
                domain = f"{namespace}.{base}.unique"
            else:
                domain = f"{namespace}.{base}.unique.{idx}"
            return base, domain
        domain = f"{namespace}.{base}.{idx}"
        return base, domain

    def _lower_and_call(self, converter: Any, eqn: Any, params: dict[str, Any]) -> None:
        ctx = _extract_ir_ctx(converter)
        if ctx is None:
            raise RuntimeError("[onnx_function] Cannot locate IRContext")
        self._lower_in_context(ctx, eqn, params)

    def _lower_in_context(
        self,
        ctx: ConverterContext,
        eqn: JaxprEqn,
        params: dict[str, object],
    ) -> None:
        params = dict(params)
        source_trace: _SourceFunctionTrace | None = None
        if ctx.precision_policy is PrecisionPolicy.SOURCE:
            trace_parameter = params.pop(_SOURCE_TRACE)
            if not isinstance(trace_parameter, _SourceFunctionTrace):
                raise TypeError("Source function call is missing its tracing context")
            source_trace = trace_parameter
        key_reuse: FunctionKeyReuse | None = cast(
            FunctionKeyReuse | None, params.pop(_SOURCE_KEY_REUSE, None)
        )
        adapter: FunctionStateAdapter | None = cast(
            FunctionStateAdapter | None, params.pop(_SOURCE_STATE, None)
        )
        layout: FunctionOperandLayout | None = cast(
            FunctionOperandLayout | None, params.pop(_FUNCTION_LAYOUT, None)
        )
        keyword_ports = tuple(
            value.index
            for value in params.values()
            if isinstance(value, _DynamicParamRef)
        )
        original_arity = len(eqn.invars)
        proof_order = (*range(original_arity - len(keyword_ports)), *keyword_ports)
        invars, params = _unpack_function_arguments(eqn.invars, params, layout=layout)
        if adapter is not None:
            if (
                source_trace is None
                or layout is None
                or layout.state_count != adapter.state_count
            ):
                raise ValueError("SOURCE Function state has no valid call-site context")
            if not isinstance(key_reuse, FunctionKeyReuse):
                raise TypeError(
                    "SOURCE NNX Function state is missing its native key signature"
                )
            body_order = (
                *range(layout.positional_count),
                *range(
                    layout.argument_count, layout.argument_count + layout.state_count
                ),
                *range(layout.positional_count, layout.argument_count),
            )
            proof_order = (
                *range(layout.positional_count),
                *range(
                    layout.argument_count, layout.argument_count + layout.state_count
                ),
                *keyword_ports,
            )
            callee: Callable[..., object] = FunctionStateCall(
                adapter, layout.positional_count, key_reuse.reorder_inputs(body_order)
            )
            invars = (*invars, *eqn.invars[layout.argument_count :])
        else:
            if source_trace is not None and source_trace.call is not None:
                params.pop("instance_key", None)
                callee = cast(
                    Callable[..., object],
                    (
                        source_trace.call.receiver
                        if source_trace.call.instance
                        else source_trace.call.original
                    ),
                )
            else:
                callee = self._resolve_lowering_callee(params)
        eqn = eqn.replace(invars=list(invars), params=params)
        captures = self._capture_arguments(ctx, callee, params)
        registry = ctx.get_function_registry()
        if registry is None:
            raise RuntimeError("[onnx_function] Function registry missing")
        key = self._function_key(
            callee, eqn, captures, ctx=ctx, source_trace=source_trace
        )
        definition = registry.get(key)
        if definition is None:
            definition = self._build_function(
                ctx,
                callee,
                eqn,
                captures,
                source_trace=source_trace,
                source_body_order=tuple(proof_order),
            )
            registry.put(key, definition)
        self._emit_function_call(ctx, eqn, definition, captures)
        if ctx._source_structure is not None and source_trace is not None:
            from jax2onnx.plugins._source_function_structure import record_function_call

            ctx._source_structure._observe(
                "function-call",
                lambda: record_function_call(
                    ctx, eqn, definition, source_trace.body, tuple(proof_order)
                ),
            )

    def _resolve_lowering_callee(
        self, params: dict[str, object]
    ) -> Callable[..., object]:
        callee = self._orig_fn
        if "instance_key" in params:
            instance_key = params.pop("instance_key")
            if not isinstance(instance_key, int):
                raise TypeError("[onnx_function] instance_key must be an integer")
            callee = INSTANCE_MAP2.get(instance_key)
        if callee is None:
            raise RuntimeError("[onnx_function] Cannot resolve callee")
        return callee

    def _capture_spec(
        self, ctx: ConverterContext, var: Var | Literal
    ) -> jax.ShapeDtypeStruct:
        aval = self._aval_to_shaped_array(var.aval)
        shape: list[object] = []
        for dim in aval.shape:
            actual_dim = dim
            if not isinstance(dim, (int, np.integer)):
                origin = SymbolicDimOrigin.resolve(ctx.get_symbolic_dim_origin, dim)
                if origin is not None and origin.value.shape is not None:
                    source_dims = origin.value.shape.dims
                    if len(source_dims) > origin.axis:
                        candidate = source_dims[origin.axis]
                        if isinstance(candidate, (int, np.integer)):
                            actual_dim = int(candidate)
            shape.append(actual_dim)
        return jax.ShapeDtypeStruct(
            tuple(shape),
            aval.dtype,
            weak_type=(
                aval.weak_type
                if ctx.precision_policy is PrecisionPolicy.SOURCE
                else False
            ),
        )

    @staticmethod
    def _parameter_spec(
        value: object, *, preserve_weak_type: bool = False
    ) -> jax.ShapeDtypeStruct:
        """Inspect a Python/JAX parameter at the capture boundary."""
        if isinstance(value, (Tracer, jax.Array)):
            aval = value.aval
            if isinstance(aval, ShapedArray):
                try:
                    dtype = np.dtype(aval.dtype)
                except TypeError:
                    dtype = np.dtype(np.float32)
                return jax.ShapeDtypeStruct(
                    aval.shape,
                    dtype,
                    weak_type=aval.weak_type if preserve_weak_type else False,
                )
        array = np.asarray(value)
        return jax.ShapeDtypeStruct(array.shape, array.dtype)

    def _capture_arguments(
        self,
        ctx: ConverterContext,
        callee: Callable[..., object],
        params: Mapping[str, object],
    ) -> FunctionCaptures:
        dynamic: list[DynamicCapture] = []
        static: dict[str, object] = {}
        signature: list[tuple[str, tuple[object, ...]]] = []
        value_key = (
            source_static_value_key
            if ctx.precision_policy is PrecisionPolicy.SOURCE
            else static_value_key
        )
        for name, value in params.items():
            if isinstance(value, (Var, Literal)):
                spec = self._capture_spec(ctx, value)
                parent_value = ctx.require_value_for_var(value)
                dynamic.append(DynamicCapture(name, spec, parent_value))
                aval = self._aval_to_shaped_array(value.aval)
                signature.append(
                    (name, ("dynamic", tuple(aval.shape), str(aval.dtype)))
                )
            elif (
                ctx.precision_policy is not PrecisionPolicy.SOURCE
                and ctx._promote_call_input_params
                and name in ctx._call_input_param_names
            ):
                spec = self._parameter_spec(
                    value,
                    preserve_weak_type=ctx.precision_policy is PrecisionPolicy.SOURCE,
                )
                dynamic.append(
                    DynamicCapture(name, spec, ctx.ensure_external_flag(name, None))
                )
                signature.append(
                    (name, ("call_input", tuple(spec.shape), str(spec.dtype)))
                )
            else:
                signature.append((name, value_key(value)))
                static[name] = value

        handled = set(static) | {capture.name for capture in dynamic}
        implicit_names = (
            ctx._call_input_param_names - handled
            if ctx.precision_policy is not PrecisionPolicy.SOURCE
            and ctx._promote_call_input_params
            else set()
        )
        if implicit_names:
            # Signature inspection is a Python-callable boundary, not an IR capability check.
            accepted_names: Mapping[str, inspect.Parameter]
            try:
                accepted_names = inspect.signature(callee).parameters
            except (TypeError, ValueError):
                accepted_names = {}
            for name in ctx._call_input_param_names:
                if name not in implicit_names or name not in accepted_names:
                    continue
                if name not in ctx._call_input_param_literals:
                    continue
                spec = self._parameter_spec(
                    ctx._call_input_param_literals[name],
                    preserve_weak_type=ctx.precision_policy is PrecisionPolicy.SOURCE,
                )
                dynamic.append(
                    DynamicCapture(name, spec, ctx.ensure_external_flag(name, None))
                )
                signature.append(
                    (name, ("call_input", tuple(spec.shape), str(spec.dtype)))
                )
        return FunctionCaptures(tuple(dynamic), static, tuple(signature))

    def _function_key(
        self,
        callee: Callable[..., object],
        eqn: JaxprEqn,
        captures: FunctionCaptures,
        *,
        ctx: ConverterContext,
        source_trace: _SourceFunctionTrace | None = None,
    ) -> FunctionKey:
        input_signature = tuple(
            (tuple(aval.shape), str(aval.dtype)) for aval in self._input_avals(eqn)
        )
        capture_signature = (
            ("state", callee.signature, captures.signature)
            if isinstance(callee, FunctionStateCall)
            else (
                self._build_unique_signature(
                    callee,
                    captures.signature,
                    value_key=(
                        source_static_value_key
                        if source_trace is not None
                        else static_value_key
                    ),
                )
                if self.unique
                else (id(callee), captures.signature)
            )
        )
        tracing_signature: tuple[object, ...] = ()
        if source_trace is not None:
            tracing_signature = (
                ctx.precision_policy.value,
                ctx.deployment_target,
                source_trace.x64_enabled,
                source_trace.default_matmul_precision,
                source_trace.numpy_dtype_promotion,
                source_trace.numpy_rank_promotion,
                source_trace.debug_key_reuse,
                tuple(aval.weak_type for aval in self._input_avals(eqn)),
                tuple(capture.spec.weak_type for capture in captures.dynamic),
            )
            if source_trace.call is not None:
                call_identity: tuple[int, ...] = source_trace.call.identity
                if (
                    self.unique
                    and inspect.isclass(self.target)
                    and source_trace.call.instance
                    and not isinstance(callee, FunctionStateCall)
                ):
                    # The existing unique signature owns complete instance state;
                    # retain the native method body while reusing equal receivers.
                    call_identity = call_identity[:1]
                tracing_signature += (call_identity,)
        return FunctionKey(
            self.name, input_signature, capture_signature, tracing_signature
        )

    def _input_avals(self, eqn: JaxprEqn) -> tuple[ShapedArray, ...]:
        return tuple(self._aval_to_shaped_array(var.aval) for var in eqn.invars)

    def _trace_function_body(
        self,
        callee: Callable[..., object],
        eqn: JaxprEqn,
        captures: FunctionCaptures,
        *,
        precision_policy: PrecisionPolicy = PrecisionPolicy.LEGACY,
        source_trace: _SourceFunctionTrace | None = None,
    ) -> ClosedJaxpr:
        base_specs = tuple(
            jax.ShapeDtypeStruct(
                aval.shape,
                aval.dtype,
                weak_type=(
                    aval.weak_type
                    if precision_policy is PrecisionPolicy.SOURCE
                    else False
                ),
            )
            for aval in self._input_avals(eqn)
        )
        base_count = len(base_specs)

        def wrapped(*arguments: object) -> object:
            kwargs = dict(captures.static_kwargs)
            kwargs.update(
                (capture.name, argument)
                for capture, argument in zip(captures.dynamic, arguments[base_count:])
            )
            with ExitStack() as call_stack:
                if source_trace is not None:
                    call_stack.enter_context(source_trace.activate())
                original = (
                    source_trace.call.original
                    if source_trace is not None and source_trace.call is not None
                    else callee
                )
                return original(*arguments[:base_count], **kwargs)

        active = _IN_FUNCTION_BUILD.get()
        token = _IN_FUNCTION_BUILD.set(active | {self.name})
        try:
            with ExitStack() as stack:
                stack.enter_context(precision_policy_scope(precision_policy))
                if source_trace is not None:
                    # Recreate already-typed formals without truncating FP64
                    # specs. The callee itself runs under its captured settings;
                    # JAX permits an existing FP64 value in a local x64-off scope.
                    stack.enter_context(jax.enable_x64(True))
                    # JAX also checks key reuse after wrapped returns, so retain
                    # the call-site flag until make_jaxpr has completed.
                    stack.enter_context(
                        jax.debug_key_reuse(source_trace.debug_key_reuse)
                    )
                stack.enter_context(_activate_full_plugin_worlds_for_body())
                closed = jax.make_jaxpr(wrapped)(
                    *base_specs, *(capture.spec for capture in captures.dynamic)
                )
                if isinstance(callee, FunctionStateCall):
                    actual_key_reuse = self._source_key_reuse(closed)
                    if actual_key_reuse != callee.key_reuse:
                        raise SourcePrecisionError(
                            "NNX Function key consumption changed while retracing",
                            phase="profile-validation",
                            category="lowering-defect",
                            location=self.name,
                            primitive=self.name,
                            requirement="actual body key effects must equal the native call-site effects",
                            action="preserve original key consumption and explicit state ports",
                        )
                return closed
        finally:
            _IN_FUNCTION_BUILD.reset(token)

    @staticmethod
    def _keep_unused_captures(
        ctx: NativeLoweringContext,
        captures: tuple[DynamicCapture, ...],
        child_values: list[ir.Value],
    ) -> None:
        """Keep declared, unused dynamic parameters in the function signature."""
        builder = ctx.builder
        for capture, value in zip(captures, child_values):
            if consumers_in_graph(builder.graph, value) or any(
                output is value for output in builder.outputs
            ):
                continue
            sink = ir.Value(
                name=ctx.fresh_name(f"{capture.name}_sink"),
                type=value.type,
                shape=value.shape,
            )
            builder.op("Identity", [value], output=sink)

    def _build_function(
        self,
        ctx: ConverterContext,
        callee: Callable[..., object],
        eqn: JaxprEqn,
        captures: FunctionCaptures,
        *,
        source_trace: _SourceFunctionTrace | None = None,
        source_body_order: tuple[int, ...] = (),
    ) -> ir.Function:
        name, domain = self._allocate_friendly_name(ctx)
        scope = FunctionScope(ctx, name=name, domain=domain)
        parent_values = [ctx.require_value_for_var(var) for var in eqn.invars]
        child_values = scope.begin(
            parent_values + [capture.value for capture in captures.dynamic]
        )
        child_captures = child_values[len(parent_values) :]
        scope.ctx._call_param_value_by_name.update(
            (capture.name, value)
            for capture, value in zip(captures.dynamic, child_captures)
        )
        closed = self._trace_function_body(
            callee,
            eqn,
            captures,
            precision_policy=ctx.precision_policy,
            source_trace=source_trace,
        )
        body = closed.jaxpr
        structure_ledger = ctx._source_structure
        if structure_ledger is not None and source_trace is not None:
            from jax2onnx.converter._source_regions import freeze_closed_body

            actual_body = freeze_closed_body(closed)
            structure_ledger._observe(
                "function-body",
                lambda: structure_ledger.function_body(
                    scope.ctx.builder.graph,
                    source_trace.body,
                    source_body_order,
                    actual_body,
                ),
            )
        if (
            source_trace is not None
            and tuple(var.aval for var in body.outvars) != source_trace.output_avals
        ):
            raise ValueError("Source function retracing changed its result signature")
        try:
            for index, (var, value) in enumerate(zip(body.constvars, closed.consts)):
                if ctx.precision_policy is PrecisionPolicy.SOURCE:
                    aval = var.aval
                    if (
                        isinstance(aval, ShapedArray)
                        and isinstance(aval.dtype, PRNGKeyDType)
                        and not is_selected_prng_key_dtype(aval.dtype)
                    ):
                        raise SourcePrecisionError(
                            "typed key implementation is outside the selected threefry2x32 packet",
                            phase="profile-validation",
                            category="profile-exclusion",
                            location=f"constant {index}",
                            requested_dtypes=(str(aval.dtype),),
                            opset=ctx.opset,
                            requirement="an actual scalar threefry2x32 typed key with two physical uint32 words",
                            action="use the audited key implementation or add audited representation support",
                        )
                scope.ctx.bind_const_for_var(
                    var,
                    (
                        value
                        if ctx.precision_policy is PrecisionPolicy.SOURCE
                        else np.asarray(value)
                    ),
                )
            for var, value in zip(body.invars, child_values):
                scope.ctx.bind_value_for_var(var, value)
            lower_jaxpr_with_plugins(
                ctx=scope.ctx,
                jaxpr=body,
                registry=PLUGIN_REGISTRY,
                source="onnx_function",
                missing_plugin_detail="in function body",
            )
        except SourcePrecisionError as exc:
            if ctx.precision_policy is not PrecisionPolicy.SOURCE:
                raise
            raise exc.with_context(
                f"function {self._friendly_name_base()} ({self._qualified_target})"
            ) from exc
        if ctx.precision_policy is not PrecisionPolicy.SOURCE:
            self._keep_unused_captures(scope.ctx, captures.dynamic, child_captures)
        outputs = [scope.ctx.require_value_for_var(var) for var in body.outvars]
        if ctx.precision_policy is PrecisionPolicy.SOURCE:
            # Function result names must be unique and distinct from formals,
            # including repeated values and input passthrough in result trees.
            seen: set[str | None] = set()
            for index, value in enumerate(outputs):
                if value.is_graph_input() or value.name in seen:
                    outputs[index] = scope.ctx.builder.op(
                        "Identity",
                        [value],
                        output=ir.Value(
                            name=scope.ctx.fresh_name("result_alias"),
                            type=value.type,
                            shape=value.shape,
                        ),
                    )
                seen.add(outputs[index].name)
        definition = scope.end(outputs=outputs)
        ctx.ir_functions.append(definition)
        ctx.ir_functions.extend(scope.ctx.ir_functions)
        return definition

    @staticmethod
    def _emit_function_call(
        ctx: NativeLoweringContext,
        eqn: JaxprEqn,
        definition: ir.Function,
        captures: FunctionCaptures,
    ) -> None:
        inputs = [ctx.require_value_for_var(var) for var in eqn.invars]
        inputs.extend(capture.value for capture in captures.dynamic)
        outputs = [ctx.allocate_value_for_var(var) for var in eqn.outvars]
        builder = ctx.builder
        # Builder names have a numeric suffix starting at zero; calls historically start at one.
        base, _, suffix = builder.fresh_name(definition.name).rpartition("_")
        call_name = f"{base}_{int(suffix) + 1}"
        builder.op_multi_out(
            definition.name,
            inputs,
            outputs=outputs,
            domain=definition.domain,
            name=call_name,
        )


# ------------------------------------------------------------------------------
# Decorators & helpers
# ------------------------------------------------------------------------------


def _onnx_function_override_name(name: str | None, type_name: str | None) -> str | None:
    # Prefer `type` override; fall back to `name` for backwards compatibility.
    return type_name if isinstance(type_name, str) and type_name else name


def _register_onnx_function_plugin(
    actual_target: Any,
    *,
    unique: bool,
    normalized_namespace: str,
    override_name: str | None,
) -> tuple[str, FunctionPlugin]:
    qual = _qualname_of_target(actual_target)
    prim_name = f"onnx_fn::{qual}"
    plugin = PLUGIN_REGISTRY.get(prim_name)

    if plugin is None:
        fp = FunctionPlugin(
            prim_name,
            actual_target,
            unique=unique,
            namespace=normalized_namespace,
        )
        if isinstance(override_name, str) and override_name:
            fp.display_name = override_name
        ONNX_FUNCTION_PLUGIN_REGISTRY[qual] = fp
        PLUGIN_REGISTRY.register_canonical(prim_name, fp)
        return qual, fp

    if isinstance(plugin, FunctionPlugin):
        if normalized_namespace and plugin.namespace != normalized_namespace:
            raise ValueError(
                f"@onnx_function namespace mismatch for {qual}: "
                f"{plugin.namespace} vs {normalized_namespace}"
            )
        if isinstance(override_name, str) and override_name:
            current = getattr(plugin, "display_name", None)
            if current is None or current == "":
                plugin.display_name = override_name
            elif current != override_name:
                raise ValueError(
                    f"@onnx_function name/type mismatch for {qual}: "
                    f"{current} vs {override_name}"
                )
        if unique:
            plugin.unique = True
        ONNX_FUNCTION_PLUGIN_REGISTRY[qual] = plugin
        PLUGIN_REGISTRY.register_canonical(prim_name, plugin)
        return qual, plugin

    raise ValueError(
        f"@onnx_function registry collision for {qual}: "
        f"{plugin.__class__.__name__} already owns {prim_name}"
    )


def _mark_onnx_function_target(
    actual_target: Any,
    *,
    unique: bool,
    normalized_namespace: str,
    override_name: str | None,
) -> None:
    try:
        setattr(actual_target, "__j2o_onnx_function__", True)
        if unique:
            setattr(actual_target, "__j2o_onnx_function_unique__", True)
        setattr(actual_target, "__j2o_onnx_function_namespace__", normalized_namespace)
        if isinstance(override_name, str) and override_name:
            setattr(actual_target, "__j2o_onnx_function_name__", override_name)
            setattr(actual_target, "__j2o_onnx_function_type__", override_name)
    except Exception:
        pass


def onnx_function(
    target: Any | None = None,
    *,
    unique: bool = False,
    namespace: str | None = None,
    name: str | None = None,
    type: str | None = None,  # noqa: A002 - allow user-facing keyword 'type'
) -> Any:
    """
    Mark a class or free function as an ONNX function boundary.
    We do **not** wrap/capture the original callable here to avoid freezing out
    later monkey patches. Instead, we only register a FunctionPlugin so that
    when plugin activation runs, the patch wrapper (above) intercepts calls,
    records a hit, and binds the function primitive.
    """

    def _decorate(actual_target: Any) -> Any:
        normalized_ns = _normalize_namespace(namespace)
        override_name = _onnx_function_override_name(name, type)
        _register_onnx_function_plugin(
            actual_target,
            unique=unique,
            normalized_namespace=normalized_ns,
            override_name=override_name,
        )
        _mark_onnx_function_target(
            actual_target,
            unique=unique,
            normalized_namespace=normalized_ns,
            override_name=override_name,
        )
        return actual_target

    if target is None:
        return _decorate
    if inspect.isclass(target) or callable(target):
        return _decorate(target)
    raise TypeError("@onnx_function expects a class or callable target")


def _call_prepared_nnx_method(
    role: Any,
    receiver: object,
    arguments: tuple[Any, ...],
    keywords: dict[str, Any],
) -> Any:
    """Keep an explicitly prepared body inside its original named NNX boundary.

    Roles contain static source/method/schema records, never a live receiver.
    The actual receiver may have just been reconstructed by NNX split/merge.
    This entry never changes the plugin's shared original-callable fields.
    """
    from jax2onnx._source_nnx_method_preparation import _PreparedMethod
    from jax2onnx._source_preparation import SourcePreparationError

    if type(role) is not _PreparedMethod:
        raise SourcePreparationError("unknown prepared NNX method role")
    role.assert_receiver(receiver)
    target = role.owner_type
    plugin = ONNX_FUNCTION_PLUGIN_REGISTRY.get(_qualname_of_target(target))
    if not isinstance(plugin, FunctionPlugin) or plugin.target is not target:
        raise SourcePreparationError("original named NNX Function owner changed")
    patch = _PATCH_STATE.get((target, "__call__"))
    if patch is None:
        if vars(target).get("__call__") is not role.original:
            raise SourcePreparationError("original NNX method binding changed")
        return role.method(receiver, *arguments, **keywords)
    if (
        patch["orig"] is not role.original
        or vars(target).get("__call__") is not patch["patched"]
    ):
        raise SourcePreparationError("prepared NNX method patch ownership changed")
    if active_precision_policy() is not PrecisionPolicy.SOURCE:
        raise SourcePreparationError("prepared NNX export requires source policy")
    if plugin.name in _IN_FUNCTION_BUILD.get():
        return role.method(receiver, *arguments, **keywords)
    return plugin._bind_function_call(
        plugin.primitive,
        role.method.__get__(receiver, target),
        arguments,
        keywords,
        instance_key=_register_function_instance(receiver),
        owner=receiver,
        unbound_method=role.method,
    )


def _consume_onnx_function_hits() -> set[str]:
    hits = set(_ONNX_FN_HITS.get())
    _ONNX_FN_HITS.set(frozenset())
    return hits


def register_example(**metadata: Any) -> dict[str, Any]:
    """Register example metadata used by plugins/examples/*."""
    comp = metadata.get("component")
    if not isinstance(comp, str) or not comp:
        raise ValueError("register_example requires a non-empty 'component' string.")
    ctx = metadata.get("context")
    key = f"{ctx}::{comp}" if isinstance(ctx, str) and ctx else comp
    if key in EXAMPLE_REGISTRY:
        logger.warning(
            "register_example overriding existing entry for key %s (component=%s, context=%s)",
            key,
            comp,
            ctx,
        )
    EXAMPLE_REGISTRY[key] = metadata

    return metadata


def register_primitive(
    **metadata: Any,
) -> Callable[[type[PrimitiveLeafPlugin]], type[PrimitiveLeafPlugin]]:
    """Register an extension through the legacy calling-convention adapter."""
    return _primitive_decorator(metadata, PLUGIN_REGISTRY.__setitem__)


def register_native_primitive(
    **metadata: Any,
) -> Callable[[type[PrimitiveLeafPlugin]], type[PrimitiveLeafPlugin]]:
    """Register the statically typed ``lower(ctx, eqn) -> None`` contract."""
    return _primitive_decorator(metadata, PLUGIN_REGISTRY.register_canonical)


def _primitive_decorator(
    metadata: dict[str, Any],
    register: Callable[[str, PrimitiveLeafPlugin], None],
) -> Callable[[type[PrimitiveLeafPlugin]], type[PrimitiveLeafPlugin]]:
    primitive = metadata.get("jaxpr_primitive", "")

    def _decorator(cls: type[PrimitiveLeafPlugin]) -> type[PrimitiveLeafPlugin]:
        if not issubclass(cls, PrimitiveLeafPlugin):
            raise TypeError("Plugin must subclass PrimitiveLeafPlugin")
        instance = cls()
        instance.primitive = primitive
        instance.metadata = metadata or {}
        if hasattr(cls, "patch_info"):
            instance.patch_info = cls.patch_info
        if isinstance(primitive, str) and primitive:
            register(primitive, instance)
        return cls

    return _decorator


# ------------------------------------------------------------------------------
# Monkey patching activation
# ------------------------------------------------------------------------------


@dataclass(frozen=True)
class _PluginPatchSpec:
    owner: object
    spec: PatchSpec


def _iter_patch_specs() -> Iterator[_PluginPatchSpec | PatchSpec]:
    """
    SOURCE selects only genuine FunctionPlugin owners before patch_info runs.
    LEGACY retains its existing patch_info compatibility path, including numeric
    leaves; their binding_specs are activated separately by plugin_binding.
    """
    for plugin in PLUGIN_REGISTRY.values():
        if active_precision_policy() is PrecisionPolicy.SOURCE and not isinstance(
            plugin, FunctionPlugin
        ):
            continue
        pinfo = getattr(plugin, "patch_info", None)
        if callable(pinfo):
            try:
                info = pinfo()
            except Exception:
                if active_precision_policy() is PrecisionPolicy.SOURCE:
                    raise
                continue
            if not info:
                continue
            patch_fn = info.get("patch_function")
            targets = info.get("patch_targets", [])
            attr = info.get("target_attribute", "__call__")
            if callable(patch_fn) and targets:
                yield _PluginPatchSpec(plugin, (patch_fn, targets, attr))


@contextmanager
def _source_native_patch_scope() -> Iterator[frozenset[tuple[int, str]]]:
    if active_precision_policy() is not PrecisionPolicy.SOURCE:
        yield frozenset()
        return
    functions = {
        plugin
        for plugin in PLUGIN_REGISTRY.values()
        if isinstance(plugin, FunctionPlugin)
    }
    saved_fields = {
        plugin: (plugin._orig_fn, plugin._unbound_orig_fn) for plugin in functions
    }
    needs_suspension = _has_owned_leaf_patches() or any(
        state.get("policy") is PrecisionPolicy.LEGACY for state in _PATCH_STATE.values()
    )
    try:
        if not needs_suspension:
            yield frozenset()
            return
        saved_states = dict(_PATCH_STATE)
        with _suspend_owned_patches() as blocked:
            _PATCH_STATE.clear()
            try:
                yield blocked
            finally:
                _PATCH_STATE.clear()
                _PATCH_STATE.update(saved_states)
    finally:
        for plugin, (original, unbound) in saved_fields.items():
            plugin._orig_fn = original
            plugin._unbound_orig_fn = unbound


@contextmanager
def apply_monkey_patches() -> Iterator[None]:
    """Temporarily swap in tracing-time shims.

    Conversion may patch framework call-sites so tracing emits custom
    primitives, but those shims must *never* leak beyond the converter’s
    scope—tests/numeric validation run against upstream JAX behaviour. This
    context manager mirrors the policy: apply each patch on entry, reference
    count nested uses, then restore the original attribute on exit so user code
    stays pristine once conversion finishes.
    """
    with _source_native_patch_scope() as blocked:
        touched: list[tuple[Any, str]] = []
        try:
            for specification in _iter_patch_specs():
                if isinstance(specification, _PluginPatchSpec):
                    owner: object | None = specification.owner
                    patch_fn, targets, attr = specification.spec
                else:
                    # Retain the private legacy/testing tuple boundary.
                    owner = None
                    patch_fn, targets, attr = specification
                for tgt in targets:
                    key = (tgt, attr)
                    if (id(tgt), attr) in blocked:
                        if isinstance(owner, FunctionPlugin):
                            raise RuntimeError(
                                "SOURCE Function patch ownership conflicts with a caller replacement"
                            )
                        continue
                    st = _PATCH_STATE.get(key)
                    if st is None:
                        orig = getattr(tgt, attr)
                        snapshot = _snapshot_attribute(tgt, attr, orig)
                        new = patch_fn(orig)
                        setattr(tgt, attr, new)
                        record = (
                            _record_owned_patch(
                                tgt,
                                attr,
                                snapshot,
                                new,
                                owner=owner,
                                leaf=isinstance(owner, PrimitiveLeafPlugin),
                            )
                            if owner is not None
                            else None
                        )
                        _PATCH_STATE[key] = {
                            "orig": orig,
                            "snapshot": snapshot,
                            "patched": new,
                            "count": 1,
                            "policy": active_precision_policy(),
                            "owner": owner,
                            "record": record,
                        }
                    else:
                        if (
                            active_precision_policy() is PrecisionPolicy.SOURCE
                            and isinstance(owner, FunctionPlugin)
                            and _installed_attribute(tgt, attr) is not st["patched"]
                        ):
                            raise RuntimeError(
                                "SOURCE Function patch ownership conflicts with a caller replacement"
                            )
                        st["count"] += 1
                    touched.append(key)
            yield
        finally:
            for key in reversed(touched):
                st = _PATCH_STATE.get(key)
                if not st:
                    continue
                st["count"] -= 1
                if st["count"] == 0:
                    tgt, attr = key
                    try:
                        st["snapshot"].restore(tgt, attr)
                    finally:
                        record = st["record"]
                        if record is not None:
                            _forget_owned_patch(record)
                        _PATCH_STATE.pop(key, None)


# ------------------------------------------------------------------------------
# Discovery
# ------------------------------------------------------------------------------


def _import_tree(root_dir: Path, pkg_prefix: str) -> None:
    """Import every .py under a given directory and then walk via pkgutil."""
    if not root_dir.exists():
        return

    seen_modules: set[str] = set()

    # 1) File-system scan (works even without intermediate __init__.py files)
    for py in sorted(root_dir.rglob("*.py")):
        if py.name in {"plugin_system.py", "__init__.py"}:
            continue
        rel = py.relative_to(root_dir).with_suffix("")
        parts = [pkg_prefix] + list(rel.parts)
        modname = ".".join(parts)
        try:
            importlib.import_module(modname)
            seen_modules.add(modname)
        except Exception as e:
            # Surface import problems loudly so you can spot why a tree didn't load
            logger.warning(
                "Skipping import of %s due to error: %s", modname, e, exc_info=True
            )

    # 2) pkgutil walk (dup-safe; skips modules already imported in step 1)
    for _, module_name, _ in pkgutil.walk_packages(
        [str(root_dir)], prefix=f"{pkg_prefix}."
    ):
        if module_name in seen_modules or module_name == f"{pkg_prefix}.plugin_system":
            continue
        try:
            importlib.import_module(module_name)
            seen_modules.add(module_name)
        except Exception as e:
            logger.warning(
                "Skipping pkgutil import of %s due to error: %s",
                module_name,
                e,
                exc_info=True,
            )


def import_all_plugins() -> None:
    """
    Recursively import every Python module under BOTH
      - jax2onnx/plugins   (preferred)
    so all plugins and examples self-register (no hard-coded lists).
    """
    global _already_imported_plugins
    if _already_imported_plugins:
        return

    # Preferred tree
    plugins_dir = Path(os.path.dirname(__file__))  # .../jax2onnx/plugins
    _import_tree(plugins_dir, "jax2onnx.plugins")

    # mark as done (avoid duplicate imports/runs)
    _already_imported_plugins = True
