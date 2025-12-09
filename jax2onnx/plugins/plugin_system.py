# jax2onnx/plugins/plugin_system.py

from __future__ import annotations

import functools
import hashlib
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
    Set,
    TYPE_CHECKING,
    Union,
    cast,
)

import jax
import jax.tree_util as jtu
from jax.extend import core as jcore_ext
import numpy as np
from jax.core import ShapedArray
from jax.extend.core import Primitive
from jax.interpreters import batching

from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec, apply_patches
from jax2onnx.converter.function_scope import FunctionScope, FunctionKey
from jax2onnx.converter.typing_support import (
    PrimitiveLowering,
    RngTrace,
    SymbolicDimOrigin,
)

logger: logging.Logger = logging.getLogger("jax2onnx.plugins.plugin_system")

# ------------------------------------------------------------------------------
# Registries and state
# ------------------------------------------------------------------------------

# mypy/ruff-only import (avoid runtime cycles)
if TYPE_CHECKING:
    pass

# Use a small private domain for ONNX functions. Netron shows the "f"
# marker only when it can resolve a FunctionProto in a domain; ORT also
# requires an opset import for non-empty domains.
_FUNCTION_DOMAIN: Final[str] = "custom"


def _sanitize_op_type(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", s)


# Primitive name -> plugin (FunctionPlugin or PrimitiveLeafPlugin instance)
PluginEntry = Union["FunctionPlugin", PrimitiveLowering]
PLUGIN_REGISTRY: Dict[str, PluginEntry] = {}

# Qualified target name -> FunctionPlugin (for reference)
ONNX_FUNCTION_PLUGIN_REGISTRY: Dict[str, "FunctionPlugin"] = {}

# Store instance objects for class-based call targets
INSTANCE_MAP2: weakref.WeakValueDictionary[int, Any] = weakref.WeakValueDictionary()

# Track @onnx_function hits (optional)
_ONNX_FN_HITS: ContextVar[set[str]] = ContextVar("_ONNX_FN_HITS", default=set())

# During function body build, prevent that function from rebinding itself
_IN_FUNCTION_BUILD: ContextVar[set[str]] = ContextVar(
    "_IN_FUNCTION_BUILD", default=set()
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
    """Hashable wrapper for traced kwargs passed to ONNX function primitives."""

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value

    def __hash__(self) -> int:  # pragma: no cover - trivial
        return hash(id(self.value))

    def __eq__(self, other: object) -> bool:  # pragma: no cover - trivial
        return isinstance(other, _DynamicParamWrapper) and other.value is self.value


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
        resolved_args = [_materialize(arg, dtype) for arg in self._args]
        resolved_kwargs = {k: _materialize(v, dtype) for k, v in self._kwargs.items()}
        instance = self._ctor(*resolved_args, **resolved_kwargs)
        if not callable(instance):
            raise TypeError(
                "construct_and_call expected constructor to return a callable object"
            )
        return _ConstructAndCall(
            self._ctor,
            self._args,
            self._kwargs,
            dtype,
            instance=instance,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        instance = self._instance
        if instance is None:
            dtype = self._dtype
            if dtype is None:
                import jax.numpy as jnp  # Local import to avoid mandatory dependency on module import

                dtype = jnp.float32

            resolved_args = [_materialize(arg, dtype) for arg in self._args]
            resolved_kwargs = {
                k: _materialize(v, dtype) for k, v in self._kwargs.items()
            }
            instance = self._ctor(*resolved_args, **resolved_kwargs)
            if not callable(instance):
                raise TypeError(
                    "construct_and_call expected constructor to return a callable object"
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
        with apply_patches(cls.binding_specs()):
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


def _extract_ir_ctx(converter: Any):
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
def _activate_full_plugin_worlds_for_body():
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
        for plugin in PLUGIN_REGISTRY.values():
            cls = plugin.__class__
            try:
                if issubclass(cls, PrimitiveLeafPlugin):
                    stack.enter_context(cls.plugin_binding())
            except Exception:
                # Best-effort; a non-leaf or misconfigured plugin should not crash nested tracing
                logger.debug("Skipping leaf binding for %r", cls, exc_info=True)
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
    ):
        self.name = primitive_name
        self.target = target
        self.unique = bool(unique)
        self.namespace = _normalize_namespace(namespace)
        # Optional human-readable override for the function base name/op_type.
        # When unset, we fall back to the underlying callable name.
        self.display_name = None  # type: ignore[assignment]
        self._qualified_target = _qualname_of_target(target)
        self.primitive = Primitive(primitive_name)
        self.primitive.def_abstract_eval(self._abstract_eval_with_kwargs)
        self.primitive.def_impl(self._primitive_impl)
        batching.primitive_batchers[self.primitive] = self._batching_rule
        self._orig_fn = None  # set by patch wrapper

    # Implement abstract method (used by monkey-patch activator)
    def get_patch_params(self):
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
    def _aval_to_shaped_array(aval):
        if isinstance(aval, ShapedArray):
            return aval
        if hasattr(aval, "shape") and hasattr(aval, "dtype"):
            return ShapedArray(aval.shape, aval.dtype)
        raise TypeError(
            f"Cannot convert abstract value of type {type(aval)} to ShapedArray."
        )

    def _abstract_eval_with_kwargs(self, *args, **kwargs):
        if self._orig_fn is None:
            raise ValueError(f"Original function not set for '{self.name}'")

        def _coerce_kwarg(value: Any) -> Any:
            if isinstance(value, _DynamicParamWrapper):
                tracer = value.value
                aval = getattr(tracer, "aval", None)
                if aval is not None:
                    return jax.ShapeDtypeStruct(
                        tuple(getattr(aval, "shape", ())), getattr(aval, "dtype", None)
                    )
                return tracer
            return value

        kwargs = {k: _coerce_kwarg(v) for k, v in kwargs.items() if k != "instance_key"}
        specs = [
            (
                jax.ShapeDtypeStruct(arg.shape, arg.dtype)
                if isinstance(arg, ShapedArray)
                else arg
            )
            for arg in args
        ]
        out_aval = jax.eval_shape(self._orig_fn, *specs, **kwargs)
        if isinstance(out_aval, jax.ShapeDtypeStruct):
            out_aval = self._aval_to_shaped_array(out_aval)
        elif isinstance(out_aval, tuple):
            out_aval = tuple(self._aval_to_shaped_array(a) for a in out_aval)
        elif isinstance(out_aval, list):
            out_aval = [self._aval_to_shaped_array(a) for a in out_aval]
        return out_aval

    def _primitive_impl(self, *args, **kwargs):
        if self._orig_fn is None:
            raise ValueError("Original function not set for primitive!")
        return self._orig_fn(*args, **kwargs)

    def _batching_rule(self, args, dims, **params):
        params.pop("instance_key", None)
        call_kwargs = {}
        for key, value in params.items():
            if isinstance(value, _DynamicParamWrapper):
                call_kwargs[key] = value.value
            else:
                call_kwargs[key] = value

        original_fn = self._orig_fn

        if original_fn is None:
            raise NotImplementedError(
                f"Batching rule for '{self.name}' missing original callable."
            )

        axis_size = None
        for arg, bdim in zip(args, dims):
            if bdim is batching.not_mapped:
                continue
            shape = getattr(arg, "shape", None)
            if shape is None or bdim >= len(shape):
                continue
            axis_size = shape[bdim]
            break

        if axis_size is None:
            result = original_fn(*args, **call_kwargs)
            out_dims = jtu.tree_map(lambda _: batching.not_mapped, result)
            return result, out_dims

        prepared_args = [
            batching.bdim_at_front(arg, bdim, axis_size)
            for arg, bdim in zip(args, dims)
        ]

        def _call_single(*single_args):
            return original_fn(*single_args, **call_kwargs)

        batched_result = jax.vmap(_call_single)(*prepared_args)
        out_dims = jtu.tree_map(lambda _: 0, batched_result)
        return batched_result, out_dims

    def _make_patch_fn(self, primitive: Primitive, is_class: bool) -> Callable:
        def patch(original_call):
            sig = inspect.signature(original_call)
            params = list(sig.parameters.keys())

            @functools.wraps(original_call)
            def wrapped(*args, **kwargs):
                expects_self = is_class or (params and params[0] == "self")
                if expects_self:
                    instance = args[0]
                    instance_key = id(instance)
                    INSTANCE_MAP2[instance_key] = instance
                    bound_orig = original_call.__get__(instance, type(instance))
                    self._orig_fn = bound_orig
                    # If we are currently constructing this function's body, do NOT emit
                    # the function primitive again—call through so inner patches take effect.
                    if self.name in _IN_FUNCTION_BUILD.get():
                        return bound_orig(*args[1:], **kwargs)
                    # Record a hit (for optional test bookkeeping) and bind the primitive.
                    hits = set(_ONNX_FN_HITS.get())
                    hits.add(self.name.split("::", 1)[1])
                    _ONNX_FN_HITS.set(hits)
                    bound_kwargs = {}
                    for key, value in kwargs.items():
                        if key == "instance_key":
                            continue
                        if hasattr(value, "aval"):
                            bound_kwargs[key] = _DynamicParamWrapper(value)
                        else:
                            bound_kwargs[key] = value
                    return primitive.bind(
                        *args[1:],
                        **{**bound_kwargs, "instance_key": instance_key},
                    )
                else:
                    self._orig_fn = original_call
                    if self.name in _IN_FUNCTION_BUILD.get():
                        return original_call(*args, **kwargs)
                    hits = set(_ONNX_FN_HITS.get())
                    hits.add(self.name.split("::", 1)[1])
                    _ONNX_FN_HITS.set(hits)
                    bound_kwargs = {}
                    for key, value in kwargs.items():
                        if hasattr(value, "aval"):
                            bound_kwargs[key] = _DynamicParamWrapper(value)
                        else:
                            bound_kwargs[key] = value
                    return primitive.bind(*args, **bound_kwargs)

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

    @staticmethod
    def _value_fingerprint(value: Any) -> tuple[Any, ...]:
        if value is None:
            return ("none",)
        if isinstance(value, (bool, int, float, complex, str, bytes)):
            literal = value if isinstance(value, (str, bytes)) else repr(value)
            return ("literal", type(value).__name__, literal)
        try:
            arr = np.array(value)
        except Exception:
            arr = None
        if arr is not None and getattr(arr, "dtype", None) is not None:
            dtype = str(arr.dtype)
            if arr.dtype != object:
                shape = tuple(getattr(arr, "shape", ()))
                try:
                    data_bytes = arr.tobytes()
                except Exception:
                    data_bytes = repr(arr).encode("utf-8", "ignore")
                digest = hashlib.sha1(data_bytes).hexdigest()
                return ("array", shape, dtype, digest)
        return ("object", type(value).__name__, repr(value))

    def _fingerprint_instance_state(self, instance: Any) -> tuple[Any, ...]:
        if instance is None:
            return (("instance", "none"),)
        components: list[tuple[Any, ...]] = []
        try:
            leaves, treedef = jtu.tree_flatten(instance)
        except Exception:
            leaves = None
            treedef = None
        else:
            components.append(("treedef", repr(treedef)))
        if leaves:
            for idx, leaf in enumerate(leaves):
                components.append(("leaf", idx, self._value_fingerprint(leaf)))
        else:
            state = getattr(instance, "__dict__", None)
            if isinstance(state, dict):
                for key in sorted(state):
                    components.append(
                        ("attr", key, self._value_fingerprint(state[key]))
                    )
            else:
                components.append(("repr", repr(instance)))
        return tuple(components)

    def _build_unique_signature(
        self,
        callee: Any,
        capture_items: list[tuple[str, tuple[Any, ...]]],
    ) -> tuple[Any, ...]:
        signature_parts: list[tuple[Any, ...]] = [
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
                ("instance_state", self._fingerprint_instance_state(callee))
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

    def get_handler(self, converter: Any) -> Callable:
        return lambda conv, eqn, params: self._lower_and_call(conv, eqn, params)

    def _allocate_friendly_name(self, ctx) -> tuple[str, str]:
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
        if self.unique:
            counters = getattr(ctx, "_func_name_counters", None)
            if counters is None:
                counters = {}
            counter_key = (namespace, base, "unique")
            idx = counters.get(counter_key, 0) + 1
            counters[counter_key] = idx
            setattr(ctx, "_func_name_counters", counters)
            if idx == 1:
                domain = f"{namespace}.{base}.unique"
            else:
                domain = f"{namespace}.{base}.unique.{idx}"
            return base, domain
        counters = getattr(ctx, "_func_name_counters", None)
        if counters is None:
            counters = {}
        counter_key = (namespace, base, "shared")
        idx = counters.get(counter_key, 0) + 1
        counters[counter_key] = idx
        setattr(ctx, "_func_name_counters", counters)
        domain = f"{namespace}.{base}.{idx}"
        return base, domain

    def _lower_and_call(self, converter: Any, eqn: Any, params: dict[str, Any]):
        # Resolve callee
        callee = self._orig_fn
        if "instance_key" in params:
            key = params["instance_key"]
            del params["instance_key"]
            callee = INSTANCE_MAP2.get(key)

        # Parent ctx
        ctx = _extract_ir_ctx(converter)
        if ctx is None:
            raise RuntimeError("[onnx_function] Cannot locate IRContext")

        call_param_names = set(getattr(ctx, "_call_input_param_names", set()))
        # Ensure a function registry exists on the parent (converter sets this)
        freg = ctx.get_function_registry()
        if freg is None:
            raise RuntimeError("[onnx_function] Function registry missing")

        # Dedup key: (qualified, in_sigs, capture)
        in_sigs: list[tuple[tuple[Any, ...], str]] = []
        for v in eqn.invars:
            aval = getattr(v, "aval", None)
            shape = tuple(getattr(aval, "shape", ()))
            dtype = getattr(aval, "dtype", None)
            in_sigs.append((shape, str(dtype)))
        in_sigs_t = tuple(in_sigs)
        qualname = self.name

        def _capture_dynamic_from_var(var: Any) -> tuple[Any, ...]:
            aval = getattr(var, "aval", None)
            if aval is None:
                return ("dynamic", "<unknown>")
            return (
                "dynamic",
                tuple(getattr(aval, "shape", ())),
                str(getattr(aval, "dtype", "")),
            )

        def _capture_const(value: Any) -> tuple[Any, ...]:
            arr = np.asarray(value)
            return (
                "const",
                tuple(arr.shape),
                str(arr.dtype),
                hash(arr.tobytes()),
            )

        def _resolve_tracer_var(tracer: Any):
            frame = getattr(getattr(tracer, "_trace", None), "frame", None)
            if frame is None:
                return None
            tracer_map = getattr(frame, "tracer_to_var", None)
            getter = getattr(tracer_map, "get", None)
            var = getter(id(tracer)) if callable(getter) else None
            if var is None:
                return None
            const_map = getattr(frame, "constvar_to_val", None)
            const_getter = getattr(const_map, "get", None)
            const_val = const_getter(var) if callable(const_getter) else None
            if const_val is not None:
                return const_val
            return var

        def _shape_dtype_struct_from_var(var: Any) -> jax.ShapeDtypeStruct:
            aval = getattr(var, "aval", None)
            raw_shape = tuple(getattr(aval, "shape", ()))
            resolved_shape: list[Any] = []
            origin_lookup = getattr(ctx, "get_symbolic_dim_origin", None)
            for axis, dim in enumerate(raw_shape):
                actual_dim = dim
                if origin_lookup is not None and not isinstance(dim, (int, np.integer)):
                    origin = SymbolicDimOrigin.resolve(origin_lookup, dim)
                    if origin is not None:
                        src_shape = getattr(origin.value, "shape", None)
                        src_dims = getattr(src_shape, "dims", src_shape)
                        if src_dims is not None and len(src_dims) > origin.axis:
                            candidate = src_dims[origin.axis]
                            if isinstance(candidate, (int, np.integer)):
                                actual_dim = int(candidate)
                resolved_shape.append(actual_dim)
            dtype = getattr(aval, "dtype", None) or np.float32
            return jax.ShapeDtypeStruct(tuple(resolved_shape), dtype)

        dynamic_entries: list[dict[str, Any]] = []
        static_params: dict[str, Any] = {}
        capture_items: list[tuple[str, tuple[Any, ...]]] = []

        for pname, pval in params.items():
            original_val = (
                pval.value if isinstance(pval, _DynamicParamWrapper) else pval
            )
            resolved = (
                _resolve_tracer_var(original_val)
                if hasattr(original_val, "aval")
                else None
            )

            if isinstance(resolved, jcore_ext.Var):
                dynamic_entries.append(
                    {
                        "name": pname,
                        "var": resolved,
                        "sds": _shape_dtype_struct_from_var(resolved),
                    }
                )
                capture_items.append((pname, _capture_dynamic_from_var(resolved)))
            elif pname in call_param_names:
                if hasattr(original_val, "aval"):
                    aval = getattr(original_val, "aval", None)
                    shape = tuple(getattr(aval, "shape", ()))
                    aval_dtype = getattr(aval, "dtype", np.float32)
                    try:
                        dtype_np = np.dtype(aval_dtype)
                    except TypeError:
                        dtype_np = np.dtype(np.float32)
                    dtype_for_capture = dtype_np
                else:
                    arr = np.asarray(original_val)
                    shape = tuple(arr.shape)
                    dtype_np = arr.dtype
                    dtype_for_capture = dtype_np
                entry_sds = jax.ShapeDtypeStruct(shape, dtype_np)
                dynamic_entries.append(
                    {
                        "name": pname,
                        "var": None,
                        "sds": entry_sds,
                        "force_external": True,
                    }
                )
                capture_items.append(
                    (pname, ("call_input", shape, str(dtype_for_capture)))
                )
                continue
            else:
                value_for_capture = resolved if resolved is not None else original_val
                try:
                    capture_items.append((pname, _capture_const(value_for_capture)))
                except Exception:
                    capture_items.append(
                        (pname, ("static", type(value_for_capture).__name__))
                    )
                static_params[pname] = original_val

        handled_names: set[str] = {entry["name"] for entry in dynamic_entries}
        handled_names.update(static_params.keys())
        literal_map = getattr(ctx, "_call_input_param_literals", None)
        if isinstance(literal_map, dict):
            for pname in call_param_names:
                if pname in handled_names:
                    continue
                if pname not in literal_map:
                    continue
                accepts_param = False
                target_fn = callee
                if target_fn is not None:
                    try:
                        sig = inspect.signature(target_fn)
                        accepts_param = pname in sig.parameters
                    except Exception:
                        accepts_param = False
                if not accepts_param:
                    continue
                literal = literal_map[pname]
                arr = np.asarray(literal)
                shape = tuple(arr.shape)
                dtype_np = arr.dtype
                dynamic_entries.append(
                    {
                        "name": pname,
                        "var": None,
                        "sds": jax.ShapeDtypeStruct(shape, dtype_np),
                        "force_external": True,
                    }
                )
                capture_items.append((pname, ("call_input", shape, str(dtype_np))))
                handled_names.add(pname)

        for entry in dynamic_entries:
            if entry.get("force_external"):
                entry["ir_value"] = ctx.ensure_external_flag(
                    entry["name"], entry.get("var")
                )
            else:
                entry["ir_value"] = ctx.get_value_for_var(
                    entry["var"], name_hint=entry["name"]
                )

        param_values = [entry["ir_value"] for entry in dynamic_entries]
        if self.unique:
            capture_sig = self._build_unique_signature(callee, capture_items)
        else:
            capture_sig = (id(callee), tuple(capture_items))
        fkey = FunctionKey(
            qualified_name=qualname, input_sig=in_sigs_t, capture_sig=capture_sig
        )

        fdef = freg.get(fkey)
        if fdef is None:
            # new child scope
            fname, fdomain = self._allocate_friendly_name(ctx)
            fscope = FunctionScope(ctx, name=fname, domain=fdomain)
            # Make the CHILD context see the same function registry as the parent.
            parent_registry = ctx.get_function_registry()
            if parent_registry is not None:
                fscope.ctx.set_function_registry(parent_registry)
            counters = getattr(ctx, "_func_name_counters", None)
            if counters is not None:
                setattr(fscope.ctx, "_func_name_counters", counters)
            call_param_names_obj = getattr(ctx, "_call_input_param_names", None)
            if isinstance(call_param_names_obj, set):
                setattr(
                    fscope.ctx,
                    "_call_input_param_names",
                    set(call_param_names_obj),
                )
            call_param_literals_obj = getattr(ctx, "_call_input_param_literals", None)
            if isinstance(call_param_literals_obj, dict):
                setattr(
                    fscope.ctx,
                    "_call_input_param_literals",
                    dict(call_param_literals_obj),
                )

            # parent → child inputs for this call-site
            base_inputs = [ctx.get_value_for_var(v) for v in eqn.invars]
            base_input_count = len(base_inputs)
            in_vals_parent = base_inputs + param_values
            in_vals_child = fscope.begin(in_vals_parent)

            if dynamic_entries:
                child_dynamic_vals = in_vals_child[base_input_count:]
                if child_dynamic_vals:
                    existing_lookup = getattr(
                        fscope.ctx, "_call_param_value_by_name", None
                    )
                    if isinstance(existing_lookup, dict):
                        call_value_map = dict(existing_lookup)
                    else:
                        call_value_map = {}
                    for entry, child_input in zip(dynamic_entries, child_dynamic_vals):
                        entry["child_input"] = cast(Any, child_input)
                        call_value_map[entry["name"]] = child_input
                    if call_value_map:
                        setattr(fscope.ctx, "_call_param_value_by_name", call_value_map)

            # ---- Trace the callee with child input specs and lower into CHILD ctx ----

            # Build ShapeDtypeStructs from the *outer* eqn's invars (safer than IR dtypes)
            sds: list[jax.ShapeDtypeStruct] = []
            for v in eqn.invars:
                aval = getattr(v, "aval", None)
                sds.append(
                    jax.ShapeDtypeStruct(
                        tuple(getattr(aval, "shape", ())), getattr(aval, "dtype", None)
                    )
                )

            dynamic_sds: list[jax.ShapeDtypeStruct] = [
                entry["sds"] for entry in dynamic_entries
            ]
            base_arg_count = len(sds)

            def _wrapped(*all_args):
                core_args = all_args[:base_arg_count]
                dyn_args = all_args[base_arg_count:]
                kw = dict(static_params)
                for dyn_val, entry in zip(dyn_args, dynamic_entries):
                    kw[entry["name"]] = dyn_val
                return callee(*core_args, **kw)

            from types import SimpleNamespace

            active = set(_IN_FUNCTION_BUILD.get())
            _IN_FUNCTION_BUILD.set(active | {self.name})
            try:
                with _activate_full_plugin_worlds_for_body():
                    closed = jax.make_jaxpr(_wrapped)(
                        *(tuple(sds) + tuple(dynamic_sds))
                    )
                jpr_f = closed.jaxpr
                # Allow plugins inside the function body to fold constants by
                # giving the child context visibility into producer equations.
                try:
                    fscope.ctx._const_folder.install_producers(jpr_f)
                except AttributeError:
                    pass
            finally:
                _IN_FUNCTION_BUILD.set(active)

            # Bind consts into CHILD ctx
            for cv, cval in zip(jpr_f.constvars, closed.consts):
                fscope.ctx.bind_const_for_var(cv, np.asarray(cval))
            # Bind function inputs (inner invars) to CHILD f_in_* values
            for v_var, v_val in zip(jpr_f.invars, in_vals_child):
                fscope.ctx.bind_value_for_var(v_var, v_val)

            # Create a child converter facade
            child_conv = SimpleNamespace(builder=fscope.ctx.builder, ctx=fscope.ctx)
            # Walk inner equations and dispatch plugins in CHILD ctx
            for inner_eqn in jpr_f.eqns:
                prim = inner_eqn.primitive.name
                plugin = PLUGIN_REGISTRY.get(prim)
                if plugin is None:
                    raise NotImplementedError(
                        f"[onnx_function] No plugin for '{prim}' in function body"
                    )
                if hasattr(plugin, "lower"):
                    # new/old leaf plugin shape
                    try:
                        import inspect as _ins

                        has_params = "params" in _ins.signature(plugin.lower).parameters
                    except Exception:
                        has_params = False
                    if has_params:
                        plugin.lower(
                            fscope.ctx, inner_eqn, getattr(inner_eqn, "params", None)
                        )
                    else:
                        plugin.lower(fscope.ctx, inner_eqn)
                elif hasattr(plugin, "get_handler"):
                    handler = plugin.get_handler(child_conv)
                    handler(child_conv, inner_eqn, inner_eqn.params)
                else:
                    raise NotImplementedError(
                        f"[onnx_function] Unsupported plugin type for '{prim}'"
                    )

            if dynamic_entries:
                nodes = list(getattr(fscope.ctx.builder, "nodes", []) or [])
                fn_outputs = list(getattr(fscope.ctx.builder, "outputs", []) or [])

                def _value_used(val: Any) -> bool:
                    for node in nodes:
                        for inp in getattr(node, "inputs", []) or []:
                            if inp is val:
                                return True
                    for out in fn_outputs:
                        if out is val:
                            return True
                    return False

                for entry in dynamic_entries:
                    child_val = entry.get("child_input")
                    if child_val is None:
                        continue
                    if _value_used(child_val):
                        continue
                    sink_name = fscope.ctx.fresh_name(f"{entry['name']}_sink")
                    sink_val = fscope.ctx.builder.Identity(
                        child_val,
                        _outputs=[sink_name],
                    )
                    child_type = getattr(child_val, "type", None)
                    child_shape = getattr(child_val, "shape", None)
                    if child_type is not None:
                        sink_val.type = child_type
                    if child_shape is not None:
                        sink_val.shape = child_shape

            # Explicit outputs from inner jaxpr
            child_out_vals = [fscope.ctx.get_value_for_var(v) for v in jpr_f.outvars]
            fdef = fscope.end(outputs=child_out_vals)
            # Create a native onnx_ir.Function and attach to the PARENT context
            ir_fn = fscope.to_ir_function()
            ctx.ir_functions.append(ir_fn)

            nested_ir_fns = list(fscope.ctx.ir_functions)
            if nested_ir_fns:
                ctx.ir_functions.extend(nested_ir_fns)

            # Keep the friendly name and domain around (for call-site)
            freg.put(fkey, fdef)

        # Emit call-site
        base_inputs = [ctx.get_value_for_var(v) for v in eqn.invars]
        in_vals = base_inputs + param_values
        out_vals = [ctx.get_value_for_var(v) for v in eqn.outvars]
        raw_call_name = ctx.builder.fresh_name(fdef.name)

        def _bump_suffix(name: str) -> str:
            pivot = name.rfind("_")
            if pivot < 0:
                return name
            suffix = name[pivot + 1 :]
            if not suffix.isdecimal():
                return name
            try:
                bumped = int(suffix) + 1
            except Exception:
                return name
            return f"{name[:pivot]}_{bumped}"

        call_name = _bump_suffix(raw_call_name)

        ctx.builder.op_multi_out(
            fdef.name,
            in_vals,
            None,
            outputs=out_vals,
            domain=fdef.domain or "",
            name=call_name,
        )


# ------------------------------------------------------------------------------
# Decorators & helpers
# ------------------------------------------------------------------------------


def onnx_function(
    target: Any | None = None,
    *,
    unique: bool = False,
    namespace: str | None = None,
    name: str | None = None,
    type: str | None = None,  # noqa: A002 - allow user-facing keyword 'type'
):
    """
    Mark a class or free function as an ONNX function boundary.
    We do **not** wrap/capture the original callable here to avoid freezing out
    later monkey patches. Instead, we only register a FunctionPlugin so that
    when plugin activation runs, the patch wrapper (above) intercepts calls,
    records a hit, and binds the function primitive.
    """

    def _decorate(actual_target: Any):
        qual = _qualname_of_target(actual_target)
        prim_name = f"onnx_fn::{qual}"
        plugin = PLUGIN_REGISTRY.get(prim_name)
        normalized_ns = _normalize_namespace(namespace)
        # Prefer `type` override; fall back to `name` for backwards compatibility.
        override_name = type if isinstance(type, str) and type else name
        if plugin is None:
            fp = FunctionPlugin(
                prim_name,
                actual_target,
                unique=unique,
                namespace=normalized_ns,
            )
            if isinstance(override_name, str) and override_name:
                fp.display_name = override_name
            ONNX_FUNCTION_PLUGIN_REGISTRY[qual] = fp
            PLUGIN_REGISTRY[prim_name] = fp
            plugin = fp
        elif isinstance(plugin, FunctionPlugin):
            if normalized_ns and plugin.namespace != normalized_ns:
                raise ValueError(
                    f"@onnx_function namespace mismatch for {qual}: "
                    f"{plugin.namespace} vs {normalized_ns}"
                )
            # Allow a late name/type override if none was set; otherwise reject conflicts.
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
        try:
            setattr(actual_target, "__j2o_onnx_function__", True)
            if unique:
                setattr(actual_target, "__j2o_onnx_function_unique__", True)
            setattr(actual_target, "__j2o_onnx_function_namespace__", normalized_ns)
            if isinstance(override_name, str) and override_name:
                setattr(actual_target, "__j2o_onnx_function_name__", override_name)
                setattr(actual_target, "__j2o_onnx_function_type__", override_name)
        except Exception:
            pass
        return actual_target

    if target is None:
        return _decorate
    if inspect.isclass(target) or callable(target):
        return _decorate(target)
    raise TypeError("@onnx_function expects a class or callable target")


def _consume_onnx_function_hits() -> set[str]:
    hits = set(_ONNX_FN_HITS.get())
    _ONNX_FN_HITS.set(set())
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
            PLUGIN_REGISTRY[primitive] = instance
        return cls

    return _decorator


# ------------------------------------------------------------------------------
# Monkey patching activation
# ------------------------------------------------------------------------------


def _iter_patch_specs():
    """
    Only yield patch specs for function plugins via their patch_info().
    Leaf plugin AssignSpec/MonkeyPatchSpec are handled by apply_patches()
    when a plugin opts into its own @plugin_binding context.
    """
    for plugin in PLUGIN_REGISTRY.values():
        pinfo = getattr(plugin, "patch_info", None)
        # Only function plugins implement patch_info() in this system.
        if callable(pinfo):
            try:
                info = pinfo()
            except Exception:
                continue
            if not info:
                continue
            patch_fn = info.get("patch_function")
            targets = info.get("patch_targets", [])
            attr = info.get("target_attribute", "__call__")
            if callable(patch_fn) and targets:
                yield patch_fn, targets, attr


@contextmanager
def apply_monkey_patches():
    """Temporarily swap in tracing-time shims.

    Conversion may patch framework call-sites so tracing emits custom
    primitives, but those shims must *never* leak beyond the converter’s
    scope—tests/numeric validation run against upstream JAX behaviour. This
    context manager mirrors the policy: apply each patch on entry, reference
    count nested uses, then restore the original attribute on exit so user code
    stays pristine once conversion finishes.
    """
    touched: list[tuple[Any, str]] = []
    for patch_fn, targets, attr in _iter_patch_specs():
        for tgt in targets:
            key = (tgt, attr)
            st = _PATCH_STATE.get(key)
            if st is None:
                orig = getattr(tgt, attr)
                new = patch_fn(orig)
                setattr(tgt, attr, new)
                _PATCH_STATE[key] = {"orig": orig, "count": 1}
            else:
                st["count"] += 1
            touched.append(key)
    try:
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
                    setattr(tgt, attr, st["orig"])
                finally:
                    _PATCH_STATE.pop(key, None)


# ------------------------------------------------------------------------------
# Discovery
# ------------------------------------------------------------------------------


def _import_tree(root_dir: Path, pkg_prefix: str) -> None:
    """Import every .py under a given directory and then walk via pkgutil."""
    if not root_dir.exists():
        return

    # 1) File-system scan (works even without intermediate __init__.py files)
    for py in root_dir.rglob("*.py"):
        if py.name in {"plugin_system.py", "__init__.py"}:
            continue
        rel = py.relative_to(root_dir).with_suffix("")
        parts = [pkg_prefix] + list(rel.parts)
        modname = ".".join(parts)
        try:
            importlib.import_module(modname)
        except Exception as e:
            # Surface import problems loudly so you can spot why a tree didn't load
            logger.warning(
                "Skipping import of %s due to error: %s", modname, e, exc_info=True
            )

    # 2) pkgutil walk (dup-safe)
    for _, module_name, _ in pkgutil.walk_packages(
        [str(root_dir)], prefix=f"{pkg_prefix}."
    ):
        try:
            importlib.import_module(module_name)
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
