# file: jax2onnx/plugins2/plugin_system.py

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
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, Optional, TYPE_CHECKING

import jax
import numpy as np
from jax.core import ShapedArray
from jax.extend.core import Primitive

import onnx_ir as ir
from jax2onnx.plugins2._patching import AssignSpec, MonkeyPatchSpec, apply_patches
from jax2onnx.converter2.function_scope import FunctionScope, FunctionKey
from jax2onnx.converter2.ir_builder import IRBuilder

logger = logging.getLogger("jax2onnx.plugins2.plugin_system")

# ------------------------------------------------------------------------------
# Registries and state
# ------------------------------------------------------------------------------

# mypy/ruff-only import (avoid runtime cycles)
if TYPE_CHECKING:
    from jax2onnx.converter2.ir_context import IRContext

# Use a small private domain for ONNX functions. Netron shows the "f"
# marker only when it can resolve a FunctionProto in a domain; ORT also
# requires an opset import for non-empty domains.
_FUNCTION_DOMAIN = "custom"


def _sanitize_op_type(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", s)


# Primitive name -> plugin (FunctionPlugin or PrimitiveLeafPlugin instance)
PLUGIN_REGISTRY2: Dict[str, Any] = {}

# Qualified target name -> FunctionPlugin (for reference)
ONNX_FUNCTION_PLUGIN_REGISTRY2: Dict[str, "FunctionPlugin"] = {}

# Store instance objects for class-based call targets
INSTANCE_MAP2: weakref.WeakValueDictionary[int, Any] = weakref.WeakValueDictionary()

# Track @onnx_function hits (optional)
_ONNX_FN_HITS: ContextVar[set[str]] = ContextVar("_ONNX_FN_HITS", default=set())

# During function body build, prevent that function from rebinding itself
_IN_FUNCTION_BUILD: ContextVar[set[str]] = ContextVar(
    "_IN_FUNCTION_BUILD", default=set()
)

# Optional examples registry (mirrored into legacy on register_example)
EXAMPLE_REGISTRY2: Dict[str, dict[str, Any]] = {}

# Patching state
_PATCH_STATE: dict[tuple[Any, str], dict[str, Any]] = {}


def _sanitize_op_type_name(name: str) -> str:
    """Make a string safe for ONNX op_type (letters, digits, underscore)."""
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


# Discovery guard (missing before → NameError during test generation)
_already_imported_plugins2: bool = False

# ------------------------------------------------------------------------------
# Primitive plugin base
# ------------------------------------------------------------------------------


class PrimitivePlugin(ABC):
    @abstractmethod
    def get_patch_params(self):
        raise NotImplementedError


class PrimitiveLeafPlugin(PrimitivePlugin):
    primitive: str
    metadata: dict[str, Any]
    patch_info: Callable[[], dict[str, Any]] | None = None

    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not getattr(cls, "_ABSTRACT_EVAL_BOUND", False):
            if hasattr(cls, "_PRIM") and hasattr(cls, "abstract_eval"):
                # type: ignore[attr-defined]
                cls._PRIM.def_abstract_eval(cls.abstract_eval)  # noqa: SLF001
            cls._ABSTRACT_EVAL_BOUND = True

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        return []

    @classmethod
    @contextmanager
    def plugin_binding(cls):
        cls.ensure_abstract_eval_bound()
        with apply_patches(cls.binding_specs()):
            yield

    def get_patch_params(self):
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
        ctx = getattr(converter, attr, None)
        if ctx is not None:
            return ctx

    # 2) via builder
    b = getattr(converter, "builder", None)
    if b is None:
        return None
    for attr in ("_ctx", "ctx", "context", "_context", "ir_context", "parent_ctx"):
        ctx = getattr(b, attr, None)
        if ctx is not None:
            return ctx

    # 3) optional accessor
    getctx = getattr(b, "get_context", None)
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
    Mirrors the outer stack used by converter2._activate_plugin_worlds().
    """
    import_all_plugins()
    with ExitStack() as stack:
        # Function plugins' monkey patches
        stack.enter_context(apply_monkey_patches())
        # Leaf plugins' binding_specs (e.g., nnx/jnp/lax rewrites)
        for plugin in PLUGIN_REGISTRY2.values():
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

    def __init__(self, primitive_name: str, target: Any):
        self.name = primitive_name
        self.target = target
        self.primitive = Primitive(primitive_name)
        self.primitive.def_abstract_eval(self._abstract_eval_with_kwargs)
        self.primitive.def_impl(self._primitive_impl)
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
        kwargs = {k: v for k, v in kwargs.items() if k != "instance_key"}
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
                    return primitive.bind(
                        *args[1:], **{**kwargs, "instance_key": instance_key}
                    )
                else:
                    self._orig_fn = original_call
                    if self.name in _IN_FUNCTION_BUILD.get():
                        return original_call(*args, **kwargs)
                    hits = set(_ONNX_FN_HITS.get())
                    hits.add(self.name.split("::", 1)[1])
                    _ONNX_FN_HITS.set(hits)
                    return primitive.bind(*args, **kwargs)

            return wrapped

        return patch

    def _friendly_name_base(self) -> str:
        """Human-readable base name for this function (class or function name)."""
        tgt = self.target
        try:
            if inspect.isclass(tgt):
                return tgt.__name__ or "Function"
            if callable(tgt):
                return getattr(tgt, "__name__", "Function")
        except Exception:
            pass
        return "Function"

    def get_handler(self, converter: Any) -> Callable:
        return lambda conv, eqn, params: self._lower_and_call(conv, eqn, params)

    def _allocate_friendly_name(self, ctx) -> str:
        """
        Produce a stable, human-readable FunctionProto name like 'SuperBlock_1'.
        Keeps a per-context counter per base name.
        """
        base = _sanitize_op_type_name(self._friendly_name_base())
        counters = getattr(ctx, "_func_name_counters", None) or {}
        idx = counters.get(base, 0) + 1
        counters[base] = idx
        setattr(ctx, "_func_name_counters", counters)
        return f"{base}_{idx}"

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

        # Ensure a function registry exists on the parent (converter2 sets this)
        freg = getattr(ctx, "_function_registry", None)
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
        capture_sig = (id(callee),)
        fkey = FunctionKey(
            qualified_name=qualname, input_sig=in_sigs_t, capture_sig=capture_sig
        )

        fdef = freg.get(fkey)
        if fdef is None:
            # new child scope
            # Use a friendly, short name like 'SuperBlock_1' (ORT-friendly; matches old-world style)
            fname = self._allocate_friendly_name(ctx)

            fscope = FunctionScope(ctx, name=fname, domain=_FUNCTION_DOMAIN)

            # parent → child inputs for this call-site
            in_vals_parent = [ctx.get_value_for_var(v) for v in eqn.invars]
            # Start the function body in the child scope (positional mapping).
            # (If a newer FunctionScope ever gains `input_names`, this call
            #  remains valid; here we keep it simple for static typing.)
            in_vals_child = fscope.begin(in_vals_parent)

            # re-trace callee on the same abstract shapes/dtypes (with patches on)
            # (this produces the inner jaxpr that we'll lower into fscope.ctx)
            default_float = (
                np.float64
                if getattr(ctx, "enable_double_precision", False)
                else np.float32
            )

            def _wrapped(*xs):
                return callee(*xs, **params)

            sds = []
            for v in eqn.invars:
                aval = getattr(v, "aval", None)
                sds.append(
                    jax.ShapeDtypeStruct(
                        getattr(aval, "shape", ()), getattr(aval, "dtype", np.float32)
                    )
                )

            prev_build = set(_IN_FUNCTION_BUILD.get())
            with _activate_full_plugin_worlds_for_body():
                token = _IN_FUNCTION_BUILD.set(prev_build | {self.name})
                try:
                    closed = jax.make_jaxpr(_wrapped)(*sds)
                finally:
                    _IN_FUNCTION_BUILD.reset(token)
            jpr_f = closed.jaxpr

            # Tie the traced inner-jaxpr inputs to the child-scope function inputs
            if len(jpr_f.invars) != len(in_vals_child):
                raise RuntimeError(
                    "[onnx_function] arity mismatch between traced invars and function inputs"
                )
            for v, child_val in zip(jpr_f.invars, in_vals_child):
                fscope.ctx.bind_value_for_var(v, child_val)

            # bind consts (captured weights, scalars) into child ctx as initializers
            for cv, cval in zip(jpr_f.constvars, closed.consts):
                np_c = np.asarray(cval)
                if np.issubdtype(np_c.dtype, np.floating):
                    np_c = np_c.astype(default_float, copy=False)
                fscope.ctx.bind_const_for_var(cv, np_c)

            # lower inner body using same plugins
            class _ChildFacade:
                builder: IRBuilder
                # Make mypy aware this attribute exists; it is set immediately below.
                _ctx: "IRContext"

            child = _ChildFacade()
            child.builder = fscope.ctx.builder
            # Expose the IRContext directly so nested FunctionPlugins can find it
            child._ctx = fscope.ctx

            for inner in jpr_f.eqns:
                prim_name = inner.primitive.name
                plugin = PLUGIN_REGISTRY2.get(prim_name)
                if plugin is None:
                    raise NotImplementedError(
                        f"[converter2:onnx_function] No plugins2 for primitive '{prim_name}' inside function body"
                    )
                # FunctionPlugin → dispatch via handler (needs facade carrying builder/_ctx)
                handler = getattr(plugin, "get_handler", None)
                if callable(handler):
                    handler(child)(child, inner, inner.params)
                    continue

                lower_fn = getattr(plugin, "lower", None)
                if not callable(lower_fn):
                    raise NotImplementedError(
                        f"[converter2:onnx_function] Plugin for '{prim_name}' has no 'get_handler' or 'lower'"
                    )
                # Leaf plugins expect the IRContext, not the facade.
                # Most declare: lower(self, ctx, eqn); some accept 'params' as 3rd arg.
                try:
                    lower_fn(fscope.ctx, inner)
                except TypeError:
                    lower_fn(fscope.ctx, inner, inner.params)

            # finalize and register: explicit outputs from the traced inner jaxpr
            child_out_vals = [fscope.ctx.get_value_for_var(v) for v in jpr_f.outvars]
            # Keep API surface compatible with current FunctionScope (no output_names kw).
            # Call-site tensors already carry the right names; the FunctionProto body
            # will serialize with its own local names.
            fdef = fscope.end(outputs=child_out_vals)

            freg.put(fkey, fdef)

        # Emit call-site
        in_vals = [ctx.get_value_for_var(v) for v in eqn.invars]
        out_vals = [ctx.get_value_for_var(v) for v in eqn.outvars]
        call = ir.Node(
            op_type=fdef.name,  # friendly name like 'SuperBlock_1'
            domain=fdef.domain,  # default domain ""
            inputs=in_vals,
            outputs=out_vals,
            name=ctx.builder.fresh_name(fdef.name),
        )
        ctx.add_node(call)


# ------------------------------------------------------------------------------
# Decorators & helpers
# ------------------------------------------------------------------------------


def onnx_function(target: Any):
    """
    Mark a class or free function as an ONNX function boundary.
    We do **not** wrap/capture the original callable here to avoid freezing out
    later monkey patches. Instead, we only register a FunctionPlugin so that
    when plugin activation runs, the patch wrapper (above) intercepts calls,
    records a hit, and binds the function primitive.
    """
    qual = _qualname_of_target(target)
    prim_name = f"onnx_fn::{qual}"
    if prim_name not in PLUGIN_REGISTRY2:
        fp = FunctionPlugin(prim_name, target)
        ONNX_FUNCTION_PLUGIN_REGISTRY2[qual] = fp
        PLUGIN_REGISTRY2[prim_name] = fp
    try:
        setattr(target, "__j2o_onnx_function__", True)
    except Exception:
        pass
    return target


def _consume_onnx_function_hits() -> set[str]:
    hits = set(_ONNX_FN_HITS.get())
    _ONNX_FN_HITS.set(set())
    return hits


def register_example(**metadata: Any) -> dict[str, Any]:
    """
    New-world example registration used by plugins2/examples2/*
    IMPORTANT: Immediately mirror into the legacy registry if available so that
    scripts/generate_tests.py (unchanged) sees the examples.
    """
    comp = metadata.get("component")
    if not isinstance(comp, str) or not comp:
        raise ValueError("register_example requires a non-empty 'component' string.")
    EXAMPLE_REGISTRY2[comp] = metadata

    # Mirror into legacy registry immediately
    # -- Set up a single, typed alias once, then assign from import to avoid mypy no-redef --
    _legacy_register_example_func: Optional[Callable[..., Any]] = None
    try:  # import under a different name, then assign
        from jax2onnx.plugins.plugin_system import (  # type: ignore[attr-defined]
            register_example as _legacy_register_example_ref,
        )

        _legacy_register_example_func = _legacy_register_example_ref
    except Exception:
        pass

    if _legacy_register_example_func is not None:
        try:
            _legacy_register_example_func(**metadata)
        except Exception:
            logger.debug(
                "Mirroring examples2 entry %r into legacy registry failed",
                metadata,
                exc_info=True,
            )

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
            PLUGIN_REGISTRY2[primitive] = instance
        return cls

    return _decorator


def register_primitive2(jax_primitive_name: str):
    def _wrap(cls):
        PLUGIN_REGISTRY2[jax_primitive_name] = cls()
        return cls

    return _wrap


# ------------------------------------------------------------------------------
# Monkey patching activation
# ------------------------------------------------------------------------------


def _iter_patch_specs():
    """
    Only yield patch specs for function plugins via their patch_info().
    Leaf plugin AssignSpec/MonkeyPatchSpec are handled by apply_patches()
    when a plugin opts into its own @plugin_binding context.
    """
    for plugin in PLUGIN_REGISTRY2.values():
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
      - jax2onnx/plugins2   (preferred)
      - jax2onnx/plugin2    (singular fallback; some repos use this path)
    so all plugins and examples self-register (no hard-coded lists).
    """
    global _already_imported_plugins2
    if _already_imported_plugins2:
        return

    # Preferred tree
    plugins2_dir = Path(os.path.dirname(__file__))  # .../jax2onnx/plugins2
    _import_tree(plugins2_dir, "jax2onnx.plugins2")

    # Fallback tree: jax2onnx/plugin2 (singular)
    base_dir = plugins2_dir.parent  # .../jax2onnx
    plugin2_dir = base_dir / "plugin2"
    _import_tree(plugin2_dir, "jax2onnx.plugin2")

    # mark as done (avoid duplicate imports/runs)
    _already_imported_plugins2 = True
    _import_tree(plugin2_dir, "jax2onnx.plugin2")

    # mark as done (avoid duplicate imports/runs)
    _already_imported_plugins2 = True
