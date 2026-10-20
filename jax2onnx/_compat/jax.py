# jax2onnx/_compat/jax.py

"""Compatibility helpers for JAX APIs that are moving across releases."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import re
import functools
import hashlib
import inspect
import json
import sys
import struct
import types
from enum import Enum
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Callable, cast

import numpy as np

from jax import Array
from jax import lax as _native_lax
from jax import numpy as _native_numpy
from jax._src.numpy import ufunc_api as _native_ufunc_api
from jax._src.numpy import ufuncs as _native_ufuncs
from jax._src.lax import lax as _native_lax_impl
from jax._src.lax import windowed_reductions as _native_windowed
from jax._src import pjit as _native_pjit
from jax._src import named_sharding as _native_named_sharding
from jax import tree_util as _native_tree_util
from jax._src import core as _native_core
from jax._src.ad_checkpoint import remat_p as _native_remat_p
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


# Raw replay only: these references are captured before exporter registration.
# This list confers no SOURCE admission and does not alter native objects.
_NATIVE_CAPTURE_OWNERS = (
    (
        _native_lax.logistic_p,
        ((_native_lax, "logistic_p"), (_native_lax_impl, "logistic_p")),
    ),
    (
        _native_lax.reduce_window_sum_p,
        (
            (_native_lax, "reduce_window_sum_p"),
            (_native_windowed, "reduce_window_sum_p"),
        ),
    ),
    (
        _native_lax.reduce_window_max_p,
        (
            (_native_lax, "reduce_window_max_p"),
            (_native_windowed, "reduce_window_max_p"),
        ),
    ),
    *(
        (vars(_native_numpy)[name], ((_native_numpy, name), (_native_ufuncs, name)))
        for name in ("maximum", "minimum", "logical_xor")
    ),
)
_NATIVE_CAPTURE_UFUNC_TYPE = _native_ufunc_api.ufunc
_NATIVE_CAPTURE_JIT_TYPE = type(
    cast(_native_ufunc_api.ufunc, _native_numpy.maximum)._func
)
# These exact native C++ methods/getters have no Python callback dictionaries.
# Their internal compiled-executable cache is the terminal native boundary.
_NATIVE_CAPTURE_CPP_BOUNDARY = tuple(
    vars(_NATIVE_CAPTURE_JIT_TYPE).get(name)
    for name in ("__getstate__", "__setstate__", "_cache_size", "_clear_cache")
) + tuple(
    member.fget if type(member) is property else None
    for name in ("__signature__", "_cache_miss")
    for member in (vars(_NATIVE_CAPTURE_JIT_TYPE).get(name),)
)
_NATIVE_CAPTURE_CLASS_FIELDS = {
    cls: frozenset(vars(cls))
    for cls in (
        Primitive,
        _NATIVE_CAPTURE_UFUNC_TYPE,
        _NATIVE_CAPTURE_JIT_TYPE,
        _native_named_sharding.UnspecifiedValue,
    )
}
_NATIVE_CAPTURE_UFUNC_FIELDS = frozenset(
    ("__doc__", "__name__", "_ufunc__static_props")
)
_NATIVE_CAPTURE_STATIC_FIELDS = frozenset(
    (
        "func",
        "nin",
        "nout",
        "nargs",
        "identity",
        "call",
        "reduce",
        "accumulate",
        "at",
        "reduceat",
    )
)
_NATIVE_CAPTURE_PRIMITIVE_FIELDS = frozenset(("name", "impl", "abstract_eval"))
_NATIVE_CAPTURE_SYMBOLS = (
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    tuple,
    list,
    dict,
    set,
    frozenset,
    object,
    type,
    np.generic,
    np.number,
    np.integer,
    np.signedinteger,
    np.unsignedinteger,
    np.inexact,
    np.floating,
    np.complexfloating,
    np.bool_,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
    inspect.Parameter.empty,
)
_NATIVE_CAPTURE_INITIAL_JIT_INFO = vars(
    cast(_native_ufunc_api.ufunc, _native_numpy.maximum)._func
).get("_jit_info")
_NATIVE_CAPTURE_ENUM_TYPES = (
    type(inspect.Parameter.POSITIONAL_ONLY),
    type(getattr(_NATIVE_CAPTURE_INITIAL_JIT_INFO, "inline", None)),
)
_NATIVE_CAPTURE_STATE_TYPES = (
    tuple,
    list,
    set,
    frozenset,
    dict,
    types.FunctionType,
    functools.partial,
    Primitive,
    _NATIVE_CAPTURE_UFUNC_TYPE,
    _NATIVE_CAPTURE_JIT_TYPE,
    _native_pjit.PjitInfo,
    property,
    inspect.Signature,
    inspect.Parameter,
    _native_tree_util.PyTreeDef,
    types.BuiltinFunctionType,
    types.MethodDescriptorType,
    types.WrapperDescriptorType,
    types.GetSetDescriptorType,
    types.MemberDescriptorType,
    staticmethod,
    classmethod,
)
_NATIVE_CAPTURE_JIT_FIELDS = frozenset(
    (
        "__annotations__",
        "__name__",
        "__module__",
        "__doc__",
        "__qualname__",
        "__wrapped__",
        "_fun",
        "_jit_info",
    )
)
_NATIVE_CAPTURE_PJIT_FIELDS = (
    "fun_sourceinfo",
    "fun_signature",
    "user_specified_in_shardings",
    "in_shardings_treedef",
    "in_shardings_leaves",
    "out_shardings_treedef",
    "out_shardings_leaves",
    "in_layouts_treedef",
    "in_layouts_leaves",
    "out_layouts_treedef",
    "out_layouts_leaves",
    "static_argnums",
    "static_argnames",
    "donate_argnums",
    "donate_argnames",
    "device",
    "backend",
    "keep_unused",
    "inline",
    "use_resource_env",
    "compiler_options_kvs",
)
_NATIVE_CAPTURE_TYPE_BINDINGS = (
    (_native_core, "Primitive", Primitive),
    (jax_core_ext, "Primitive", Primitive),
    (_native_numpy, "ufunc", _NATIVE_CAPTURE_UFUNC_TYPE),
    (_native_ufunc_api, "ufunc", _NATIVE_CAPTURE_UFUNC_TYPE),
    (
        _native_named_sharding,
        "UnspecifiedValue",
        _native_named_sharding.UnspecifiedValue,
    ),
)


_CODE_CONSTRUCTOR_PARAMETERS = (
    "argcount",
    "posonlyargcount",
    "kwonlyargcount",
    "nlocals",
    "stacksize",
    "flags",
    "codestring",
    "constants",
    "names",
    "varnames",
    "filename",
    "name",
    "qualname",
    "firstlineno",
    "linetable",
    "exceptiontable",
    "freevars",
    "cellvars",
)
_CODE_NATIVE_CONSTRUCTOR_PARAMETERS = tuple(
    inspect.signature(types.CodeType).parameters
)
_CODE_STRUCTURAL_FIELDS = (
    "co_argcount",
    "co_posonlyargcount",
    "co_kwonlyargcount",
    "co_nlocals",
    "co_stacksize",
    "co_flags",
    "co_code",
    "co_consts",
    "co_names",
    "co_varnames",
    "co_filename",
    "co_name",
    "co_qualname",
    "co_firstlineno",
    "co_linetable",
    "co_exceptiontable",
    "co_freevars",
    "co_cellvars",
)


def canonical_code_fingerprint(code: types.CodeType) -> str:
    """Fingerprint exact structural code fields, without marshal reference flags.

    Ordinary CodeType constructor fields and recursively owned immutable
    constants are retained; interpreter adaptive caches are not source state.
    This is not a universal semantic equivalence test.
    """
    if type(code) is not types.CodeType:
        raise TypeError("code fingerprint requires an actual CodeType")
    if _CODE_NATIVE_CONSTRUCTOR_PARAMETERS != _CODE_CONSTRUCTOR_PARAMETERS:
        raise TypeError("native code constructor schema needs an exact adapter")

    def encode(value: object, depth: int) -> Any:
        if depth > 48:
            raise ValueError("code constant nesting needs an exact adapter")
        if value is None:
            return {"none": True}
        if value is Ellipsis:
            return {"ellipsis": True}
        if type(value) is bool:
            return {"bool": value}
        if type(value) is int:
            return {"int_hex": hex(value)}
        if type(value) is str:
            return {"str": value}
        if type(value) is bytes:
            return {"bytes": value.hex()}
        if type(value) is float:
            return {"float64_be": struct.pack(">d", value).hex()}
        if type(value) is complex:
            return {"complex128_be": struct.pack(">dd", value.real, value.imag).hex()}
        if type(value) is tuple:
            return {"tuple": [encode(item, depth + 1) for item in value]}
        if type(value) is frozenset:
            items = [encode(item, depth + 1) for item in value]
            return {
                "frozenset": sorted(
                    items, key=lambda item: json.dumps(item, sort_keys=True)
                )
            }
        if type(value) is types.CodeType:
            return {
                "code": {
                    name: encode(getattr(value, name), depth + 1)
                    for name in _CODE_STRUCTURAL_FIELDS
                }
            }
        # No formatting, equality or hashing of unsupported user constants.
        raise TypeError("code constant type needs an exact adapter")

    record = {
        "schema": "CodeType-structural-v1",
        "constructor_parameters": _CODE_NATIVE_CONSTRUCTOR_PARAMETERS,
        "interpreter": sys.implementation.name,
        "python_version": (sys.version_info.major, sys.version_info.minor),
        "value": encode(code, 0),
    }
    return hashlib.sha256(
        json.dumps(record, sort_keys=True).encode("utf-8")
    ).hexdigest()


class _NativeCaptureGraph:
    """Snapshot the bounded native owners and their Python executable state.

    Native C++ descriptors and JIT execution/cache internals end at exact type
    and executable identities. Every Python wrapper field remains in the graph;
    no private dictionary or JIT configuration field is discarded. Native Python
    implementations end at their pinned executable boundary, retaining code,
    defaults, closures, attributes and actual top-level binding identities.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, Any] = {}
        self.bindings: list[tuple[types.ModuleType, str, object]] = []

    def _record_bindings(self, value: Any) -> None:
        module_name = value.__module__
        if type(module_name) is not str:
            raise ValueError("native implementation module requires an actual string")
        module = sys.modules.get(module_name)
        if isinstance(module, types.ModuleType):
            for name, item in vars(module).copy().items():
                if item is value:
                    self.bindings.append((module, name, value))

    @staticmethod
    def _schema(mapping: Any, expected: frozenset[str]) -> bool:
        return (
            all(type(key) is str for key in mapping) and frozenset(mapping) == expected
        )

    def _class(self, cls: type) -> Any:
        expected = _NATIVE_CAPTURE_CLASS_FIELDS[cls]
        if not self._schema(vars(cls), expected):
            raise ValueError("native capture class schema changed")
        return {name: self.take(item) for name, item in sorted(vars(cls).items())}

    def _annotation(self, value: Any) -> Any:
        # Python annotations are metadata, not executable callbacks. Keep exact
        # leaf identities and the entire annotation container without traversing
        # arbitrary typing descriptors or class/static state.
        kind = type(value)
        if kind is dict:
            if not all(type(key) is str for key in value):
                raise ValueError("native annotation keys require actual strings")
            return {key: self._annotation(item) for key, item in value.items()}
        if kind is tuple or kind is list:
            return [self._annotation(item) for item in value]
        if (
            value is None
            or kind is str
            or kind is bool
            or kind is int
            or any(value is symbol for symbol in _NATIVE_CAPTURE_SYMBOLS)
        ):
            return {"identity": id(value), "type": self.typename(value)}
        raise ValueError("unsupported native annotation state")

    @staticmethod
    def typename(value: Any) -> str:
        module, name = type(value).__module__, type(value).__qualname__
        if type(module) is not str or type(name) is not str:
            raise ValueError("native callable type metadata changed")
        return module + "." + name

    def take(self, value: Any) -> Any:
        if value is None:
            return None
        if type(value) is bool:
            return {"bool": value}
        if type(value) is int:
            return {"int": value}
        if type(value) is str:
            return {"str": value}
        if type(value) is float:
            return {"float64_be": struct.pack(">d", value).hex()}
        if type(value) is bytes:
            return {"bytes": value.hex()}
        kind = type(value)
        if not (
            any(kind is allowed for allowed in _NATIVE_CAPTURE_STATE_TYPES)
            or any(kind is allowed for allowed in _NATIVE_CAPTURE_ENUM_TYPES)
            or any(value is symbol for symbol in _NATIVE_CAPTURE_SYMBOLS)
            or any(value is member for member in _NATIVE_CAPTURE_CPP_BOUNDARY)
            or value is _native_named_sharding.UNSPECIFIED
            or value is Ellipsis
            or value is NotImplemented
        ):
            # Reject before naming/formatting an unowned class or its metaclass.
            raise ValueError("unsupported native callable state type")
        key = str(id(value))
        if key in self.nodes:
            return {"ref": key}
        node: dict[str, Any] = {"type": self.typename(value)}
        self.nodes[key] = node
        if type(value) in (tuple, list, set, frozenset):
            items = [self.take(item) for item in value]
            node["values"] = (
                sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
                if type(value) in (set, frozenset)
                else items
            )
        elif type(value) is dict:
            items = [[self.take(k), self.take(v)] for k, v in value.items()]
            node["items"] = sorted(
                items, key=lambda pair: json.dumps(pair[0], sort_keys=True)
            )
        elif type(value) is types.FunctionType:
            node.update(
                code_sha256=canonical_code_fingerprint(value.__code__),
                name=self.take(value.__name__),
                qualname=self.take(value.__qualname__),
                module=self.take(value.__module__),
                doc=self.take(value.__doc__),
                defaults=self.take(value.__defaults__),
                kwdefaults=self.take(value.__kwdefaults__),
                attributes=self.take(vars(value)),
                annotations=self._annotation(value.__annotations__),
                closure=[
                    self.take(cell.cell_contents) for cell in value.__closure__ or ()
                ],
            )
            self._record_bindings(value)
        elif type(value) is functools.partial:
            node.update(
                function=self.take(value.func),
                args=self.take(value.args),
                keywords=self.take(value.keywords),
                attributes=self.take(vars(value)),
            )
        elif type(value) is Primitive:
            if not any(value is owner for owner, _ in _NATIVE_CAPTURE_OWNERS):
                raise ValueError("unselected primitive in native capture")
            members = self._class(Primitive)
            if not self._schema(vars(value), _NATIVE_CAPTURE_PRIMITIVE_FIELDS):
                raise ValueError("native Primitive instance schema changed")
            node.update(fields=self.take(vars(value)), members=members)
        elif type(value) is _NATIVE_CAPTURE_UFUNC_TYPE:
            if not any(value is owner for owner, _ in _NATIVE_CAPTURE_OWNERS):
                raise ValueError("unselected ufunc in native capture")
            members = self._class(_NATIVE_CAPTURE_UFUNC_TYPE)
            if (
                not self._schema(vars(value), _NATIVE_CAPTURE_UFUNC_FIELDS)
                or type(vars(value)["_ufunc__static_props"]) is not dict
                or not self._schema(
                    vars(value)["_ufunc__static_props"], _NATIVE_CAPTURE_STATIC_FIELDS
                )
            ):
                raise ValueError("native ufunc instance/static schema changed")
            node.update(
                fields=self.take(vars(value)),
                members=members,
            )
        elif type(value) is _NATIVE_CAPTURE_JIT_TYPE:
            members = self._class(_NATIVE_CAPTURE_JIT_TYPE)
            if not self._schema(vars(value), _NATIVE_CAPTURE_JIT_FIELDS):
                raise ValueError("native JIT wrapper schema changed")
            self._record_bindings(value)
            node.update(
                fields=self.take(vars(value)),
                members=members,
                # This pinned native getter exposes a mutable Python callback;
                # it must not disappear behind the C++ cache boundary.
                cache_miss=self.take(_NATIVE_CAPTURE_CPP_BOUNDARY[-1](value)),
            )
        elif type(value) is _native_pjit.PjitInfo:
            if tuple(type(value)._fields) != _NATIVE_CAPTURE_PJIT_FIELDS:
                raise ValueError("native JIT configuration schema changed")
            node["fields"] = {
                name: self.take(item)
                for name, item in zip(_NATIVE_CAPTURE_PJIT_FIELDS, value, strict=True)
            }
        elif type(value) is property:
            node.update(
                getter=self.take(value.fget),
                setter=self.take(value.fset),
                deleter=self.take(value.fdel),
                doc=self.take(value.__doc__),
            )
        elif type(value) is inspect.Signature:
            node.update(
                parameters=[self.take(item) for item in value.parameters.values()],
                return_annotation=self._annotation(value.return_annotation),
            )
        elif type(value) is inspect.Parameter:
            node.update(
                name=self.take(value.name),
                kind=self.take(value.kind),
                default=self.take(value.default),
                annotation=self._annotation(value.annotation),
            )
        elif type(value) is _native_tree_util.PyTreeDef:

            def tree_state(tree: Any) -> Any:
                data = tree.node_data()
                if data is not None:
                    raise ValueError(
                        "native JIT treedef needs a bounded structural adapter"
                    )
                if tree.children():
                    raise ValueError("native JIT treedef is not a leaf")
                return {"leaf": True}

            node["tree"] = tree_state(value)
        elif value is _native_named_sharding.UNSPECIFIED:
            members = self._class(type(value))
            node.update(fields=self.take(vars(value)), members=members)
        elif isinstance(value, Enum) and type(value).__module__ in {
            "jax._src.api",
            "inspect",
        }:
            node.update(name=self.take(value.name), value=self.take(value.value))
        elif any(value is symbol for symbol in _NATIVE_CAPTURE_SYMBOLS):
            module, name = value.__module__, value.__qualname__
            if type(module) is not str or type(name) is not str:
                raise ValueError("native symbol metadata changed")
            node["symbol"] = module + "." + name
        elif any(value is member for member in _NATIVE_CAPTURE_CPP_BOUNDARY):
            node["native_jit_executable"] = True
        elif type(value) in (
            types.BuiltinFunctionType,
            types.MethodDescriptorType,
            types.WrapperDescriptorType,
            types.GetSetDescriptorType,
            types.MemberDescriptorType,
            staticmethod,
            classmethod,
        ):
            # Exact native descriptor identity is recorded by the node reference.
            # Python functions inside static/class methods still need their state.
            if type(value) in (staticmethod, classmethod):
                node["function"] = self.take(value.__func__)
        elif value is Ellipsis or value is NotImplemented:
            node["singleton"] = str(value)
        else:
            raise ValueError(
                f"unsupported native callable state: {self.typename(value)}"
            )
        return {"ref": key}


def _initial_native_capture() -> tuple[dict[int, Any], dict[int, Any], dict[int, str]]:
    states, bindings, gaps = {}, {}, {}
    for owner, _ in _NATIVE_CAPTURE_OWNERS:
        graph = _NativeCaptureGraph()
        try:
            if type(owner) is _NATIVE_CAPTURE_UFUNC_TYPE and any(
                member is None for member in _NATIVE_CAPTURE_CPP_BOUNDARY
            ):
                raise ValueError("native JIT executable boundary has an unknown schema")
            roots = [graph.take(owner)]
        except (TypeError, ValueError) as exc:
            # An unsupported installed owner layout remains its own replay gap.
            gaps[id(owner)] = str(exc)
            continue
        states[id(owner)] = {"roots": roots, "nodes": graph.nodes}
        bindings[id(owner)] = tuple(graph.bindings)
    return states, bindings, gaps


_NATIVE_CAPTURE_BASELINE, _NATIVE_CAPTURE_BINDINGS, _NATIVE_CAPTURE_GAPS = (
    _initial_native_capture()
)


def selected_native_capture(value: object) -> dict[str, Any] | None:
    """Capture only exact selected native owners and canonical bound aliases."""
    member = None
    owner = value
    if type(value) is types.MethodType:
        owner = value.__self__
        member = "bind" if type(owner) is Primitive else "reduce"
        if not any(owner is original for original, _ in _NATIVE_CAPTURE_OWNERS):
            if any(
                vars(module).get(name) is owner
                for _, bindings in _NATIVE_CAPTURE_OWNERS
                for module, name in bindings
            ):
                raise ValueError("selected native binding points to a foreign owner")
            return None
        if value.__func__ is not vars(type(owner)).get(member):
            raise ValueError("selected native bound member identity changed")
    elif not any(value is original for original, _ in _NATIVE_CAPTURE_OWNERS):
        if any(
            vars(module).get(name) is value
            for _, bindings in _NATIVE_CAPTURE_OWNERS
            for module, name in bindings
        ):
            raise ValueError("selected native binding points to a foreign owner")
        return None
    if id(owner) in _NATIVE_CAPTURE_GAPS:
        raise ValueError(_NATIVE_CAPTURE_GAPS[id(owner)])
    type_bindings = (
        _NATIVE_CAPTURE_TYPE_BINDINGS[:2]
        if type(owner) is Primitive
        else _NATIVE_CAPTURE_TYPE_BINDINGS[2:]
    )
    for module, name, original in type_bindings:
        if vars(module).get(name) is not original:
            raise ValueError("selected native type binding changed")
    for original, bindings in _NATIVE_CAPTURE_OWNERS:
        if original is owner:
            for module, name in bindings:
                if vars(module).get(name) is not original:
                    raise ValueError("selected native public/defining binding changed")
    for module, name, original in _NATIVE_CAPTURE_BINDINGS[id(owner)]:
        if vars(module).get(name) is not original:
            raise ValueError("selected native implementation binding changed")
    graph = _NativeCaptureGraph()
    roots = [graph.take(owner)]
    observed = {"roots": roots, "nodes": graph.nodes}
    if observed != _NATIVE_CAPTURE_BASELINE[id(owner)]:
        raise ValueError("selected native callable behavior/state changed")
    return {"native_owner": id(owner), "member": member, "state": observed}


def selected_native_capture_member(owner: object, name: str) -> object | None:
    """Resolve only the two selected native descriptors on verified owners."""
    if not any(owner is original for original, _ in _NATIVE_CAPTURE_OWNERS):
        return None
    expected = "bind" if type(owner) is Primitive else "reduce"
    if name != expected:
        raise ValueError("unselected native attribute path")
    selected_native_capture(owner)
    resolved: object = vars(type(owner))[name].__get__(owner, type(owner))
    return resolved


@dataclass(frozen=True)
class _NativeAbstractRule:
    primitive: Primitive
    evaluator: Callable[..., Any]
    multiple_results: bool


# _compat is imported before plugin_system can bind exporter abstract rules.
# Retain native callables and result conventions, never a mutable method lookup
# at validation time. This explicit set does not confer SOURCE eligibility.
_NATIVE_ABSTRACT_RULES = tuple(
    _NativeAbstractRule(primitive, primitive.abstract_eval, primitive.multiple_results)
    for primitive in (
        _native_lax.scan_p,
        _native_lax.while_p,
        _native_lax.scatter_p,
        _native_lax.scatter_add_p,
        _native_lax.scatter_mul_p,
        _native_remat_p,
    )
)


def _require_plain_native_aval(aval: Any) -> None:
    from jax.sharding import NamedSharding, PartitionSpec
    from jax._src.mesh import AbstractMesh

    if type(aval) is not ShapedArray:
        raise TypeError("native signature requires an ordinary exact ShapedArray")
    sharding = aval.sharding
    if not (
        type(sharding) is NamedSharding
        and type(sharding.mesh) is AbstractMesh
        and sharding.mesh.empty
        and sharding.memory_kind is None
        and sharding.spec == PartitionSpec(*(None for _ in aval.shape))
        and aval.manual_axis_type is _native_core.empty_mat
        and aval.memory_space is _native_core.MemorySpace.Device
    ):
        raise TypeError(
            "native signature has unsupported sharding, manual axes or memory space"
        )
    # JAX 0.11 adds layout to ShapedArray; the older native class has no slot.
    if (
        hasattr(_native_core, "AutoLayout")
        and aval.layout is not _native_core.AutoLayout
    ):
        raise TypeError("native signature has unsupported explicit layout")


def infer_native_abstract_results(
    eqn: Any, expected_primitive: Primitive
) -> tuple[tuple[Any, ...], Any]:
    """Infer selected original native avals under their recorded context.

    Callers must first own the parameter and carrier schemas and child bodies.
    This does not validate a lowerer, captures, dataflow or numerical behavior.
    """
    from jax._src.mesh import AbstractMesh

    rule = next(
        (
            rule
            for rule in _NATIVE_ABSTRACT_RULES
            if rule.primitive is expected_primitive
        ),
        None,
    )
    if rule is None or eqn.primitive is not rule.primitive:
        raise TypeError(
            "native signature requires an explicitly selected canonical primitive"
        )
    context = getattr(eqn, "ctx", None)
    if context is None or type(context) is not _native_core.JaxprEqnContext:
        raise TypeError(
            "native signature requires the actual recorded equation context"
        )
    if not (
        type(context.cur_abstract_mesh) is AbstractMesh
        and context.cur_abstract_mesh.empty
    ):
        raise TypeError("native signature requires an empty recorded abstract mesh")
    if (
        hasattr(_native_core, "LayoutMode")
        and context.cur_layout_mode is not _native_core.LayoutMode.AUTO
    ):
        raise TypeError("native signature requires the native automatic layout context")
    for var in (*eqn.invars, *eqn.outvars):
        _require_plain_native_aval(var.aval)
        if any(
            getattr(var, name, None) is not None
            for name in ("initial_qdd", "final_qdd")
        ):
            raise TypeError(
                "native signature does not own quantization variable metadata"
            )
    before = _native_core.JaxprEqnContext()
    try:
        with context.manager:
            inferred, effects = rule.evaluator(
                *(var.aval for var in eqn.invars), **eqn.params
            )
    finally:
        if _native_core.JaxprEqnContext() is not before:
            raise RuntimeError("native equation context was not restored")
    if rule.multiple_results:
        if type(inferred) not in (tuple, list):
            raise TypeError(
                "native multiple-result evaluator did not return an aval sequence"
            )
        inferred = tuple(inferred)
    else:
        inferred = (inferred,)
    for aval in inferred:
        _require_plain_native_aval(aval)
    return inferred, effects


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
