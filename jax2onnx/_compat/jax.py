# jax2onnx/_compat/jax.py

"""Compatibility helpers for JAX APIs that are moving across releases."""

from __future__ import annotations

from contextlib import contextmanager
from importlib import import_module
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
from jax._src import literals as _native_literals
from jax._src.dtypes import TypePromotionError as TypePromotionError
from jax._src import config as _native_config
from jax._src import mesh as _native_mesh
from jax._src import xla_metadata_lib as _native_xla_metadata
from jax._src.lax.control_flow import loops as _native_loops
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

# Native HiJAX nonzero is available only on the newer supported stacks. Pin
# its actual implementation instead of invoking a carrier-provided callback.
try:
    _native_numpy_hijax = import_module("jax._src.numpy.hijax")
except ModuleNotFoundError as error:
    if error.name != "jax._src.numpy.hijax":
        raise
    _NATIVE_NONZERO_TYPE: type[Any] | None = None
    _NATIVE_NONZERO_IMPL: Callable[..., Any] | None = None
    _NATIVE_HI_CALL: Primitive | None = None
else:
    _NATIVE_NONZERO_TYPE = _native_numpy_hijax.Nonzero
    _NATIVE_NONZERO_IMPL = _native_numpy_hijax._nonzero_impl
    _NATIVE_HI_CALL = import_module("jax._src.hijax").call_hi_primitive_p

if TYPE_CHECKING:
    from jax._src.random.prng import KeyTy as PRNGKeyDType
    from jax._src.random.threefry2x32 import threefry_prng_impl as _NATIVE_THREEFRY_IMPL
else:
    try:
        from jax._src.random.prng import KeyTy as PRNGKeyDType
        from jax._src.random.threefry2x32 import (
            threefry_prng_impl as _NATIVE_THREEFRY_IMPL,
        )
    except ModuleNotFoundError as error:
        # Only absence of this exact split module selects the older native layout.
        # Dependency/import failures inside an available module must propagate.
        if error.name != "jax._src.random.prng":
            raise
        _legacy_prng = import_module("jax._src.prng")
        PRNGKeyDType = _legacy_prng.KeyTy
        _NATIVE_THREEFRY_IMPL = _legacy_prng.threefry_prng_impl


AbstractValue = jax_core.AbstractValue
ShapedArray = jax_core.ShapedArray
Tracer = jax_core.Tracer
Var = jax_core_ext.Var
try:
    DropVar = jax_core_ext.DropVar
except AttributeError:  # pragma: no cover - compatibility with older JAX versions
    DropVar = jax_core.DropVar


# Closed native constants: pin the native host carrier before plugin binding.
_NATIVE_TYPED_ARRAY_TYPE: type[Any] = _native_literals.TypedNdArray
_NATIVE_TYPED_ARRAY_MEMBERS: tuple[tuple[str, object], ...] = tuple(
    vars(_NATIVE_TYPED_ARRAY_TYPE).items()
)
_NATIVE_TYPED_ARRAY_BASES: tuple[type[Any], ...] = _NATIVE_TYPED_ARRAY_TYPE.__bases__
_NATIVE_TYPED_ARRAY_ARRAY_FUNCTION: object = vars(_NATIVE_TYPED_ARRAY_TYPE).get(
    "__array__"
)
_NATIVE_TYPED_ARRAY_ARRAY_STATE: tuple[object, ...] | None = (
    (
        _NATIVE_TYPED_ARRAY_ARRAY_FUNCTION.__code__,
        _NATIVE_TYPED_ARRAY_ARRAY_FUNCTION.__defaults__,
        _NATIVE_TYPED_ARRAY_ARRAY_FUNCTION.__kwdefaults__,
        _NATIVE_TYPED_ARRAY_ARRAY_FUNCTION.__closure__,
        _NATIVE_TYPED_ARRAY_ARRAY_FUNCTION.__globals__,
    )
    if type(_NATIVE_TYPED_ARRAY_ARRAY_FUNCTION) is types.FunctionType
    else None
)
_NATIVE_TYPED_ARRAY_ASARRAY: object = np.asarray


# Recipe metadata only: pin the actual native function before plugin binding.
_NATIVE_LOGISTIC_FUNCTION: object = vars(_native_lax_impl).get("logistic")
_NATIVE_LOGISTIC_CODE: object = (
    _NATIVE_LOGISTIC_FUNCTION.__code__
    if type(_NATIVE_LOGISTIC_FUNCTION) is types.FunctionType
    else None
)


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
_NATIVE_CAPTURE_PJIT_FIELDS_WITH_ABSTRACT_AXES = (
    *_NATIVE_CAPTURE_PJIT_FIELDS[:19],
    "abstracted_axes",
    *_NATIVE_CAPTURE_PJIT_FIELDS[19:],
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
            fields = vars(type(value)).get("_fields")
            if type(fields) is not tuple or any(
                type(name) is not str for name in fields
            ):
                raise ValueError(
                    "native JIT configuration fields require an actual string tuple"
                )
            if fields not in (
                _NATIVE_CAPTURE_PJIT_FIELDS,
                _NATIVE_CAPTURE_PJIT_FIELDS_WITH_ABSTRACT_AXES,
            ):
                raise ValueError("native JIT configuration schema changed")
            if (
                fields == _NATIVE_CAPTURE_PJIT_FIELDS_WITH_ABSTRACT_AXES
                and value.abstracted_axes is not None
            ):
                raise ValueError("native JIT abstracted axes needs an exact adapter")
            node["fields"] = {
                name: self.take(item) for name, item in zip(fields, value, strict=True)
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


def _native_logistic_default_layout(function: types.FunctionType) -> str:
    """Recognize only the two native argument/default layouts, without reflection."""
    if type(function) is not types.FunctionType:
        raise TypeError("native logistic metadata requires the actual Python function")
    code = function.__code__
    if (
        code.co_posonlyargcount != 0
        or code.co_flags & (inspect.CO_VARARGS | inspect.CO_VARKEYWORDS)
        or code.co_varnames[:2] != ("x", "accuracy")
    ):
        raise TypeError("native logistic argument schema needs an exact adapter")
    defaults, keywords = function.__defaults__, function.__kwdefaults__
    if code.co_argcount == 2 and code.co_kwonlyargcount == 0:
        if (
            type(defaults) is tuple
            and len(defaults) == 1
            and defaults[0] is None
            and keywords is None
        ):
            return "positional"
    elif code.co_argcount == 1 and code.co_kwonlyargcount == 1:
        if (
            defaults is None
            and type(keywords) is dict
            and len(keywords) == 1
            and all(type(key) is str for key in keywords)
            and "accuracy" in keywords
            and keywords["accuracy"] is None
        ):
            return "keyword"
    raise TypeError("native logistic default schema needs an exact adapter")


def _native_logistic_metadata_state() -> (
    tuple[dict[str, Any], tuple[tuple[types.ModuleType, str, object], ...]]
):
    function = _NATIVE_LOGISTIC_FUNCTION
    if type(function) is not types.FunctionType:
        raise TypeError(
            "native logistic metadata requires the captured Python function"
        )
    if any(
        vars(module).get("logistic") is not function
        for module in (_native_lax, _native_lax_impl)
    ):
        raise TypeError("native logistic public/defining identity changed")
    if function.__code__ is not _NATIVE_LOGISTIC_CODE:
        raise TypeError("native logistic code identity changed")
    if vars(function) or function.__closure__ is not None:
        # Includes __signature__ and __wrapped__; neither may override this guard.
        raise TypeError("native logistic function attribute/closure schema changed")
    for actual, expected in (
        (function.__module__, "jax.lax"),
        (function.__name__, "logistic"),
        (function.__qualname__, "logistic"),
    ):
        if type(actual) is not str or actual != expected:
            raise TypeError("native logistic function naming changed")
    _native_logistic_default_layout(function)
    graph = _NativeCaptureGraph()
    root = graph.take(function)
    return {"root": root, "nodes": graph.nodes}, tuple(graph.bindings)


_NATIVE_LOGISTIC_METADATA_STATE: dict[str, Any] | None = None
_NATIVE_LOGISTIC_METADATA_BINDINGS: tuple[
    tuple[types.ModuleType, str, object], ...
] = ()
_NATIVE_LOGISTIC_METADATA_GAP: str | None = None
try:
    _NATIVE_LOGISTIC_METADATA_STATE, _NATIVE_LOGISTIC_METADATA_BINDINGS = (
        _native_logistic_metadata_state()
    )
except (TypeError, ValueError) as _logistic_error:
    # Unknown installed layouts stay an explicit selected-recipe gap, import safe.
    _NATIVE_LOGISTIC_METADATA_GAP = str(_logistic_error)


def native_logistic_metadata_contract(value: object) -> dict[str, Any] | None:
    """Canonicalize only this native function's default-location recipe metadata.

    The full actual function state is verified independently of the emitted token.
    No callable, native capture, explicit argument, or numerical contract changes.
    """
    if value is not _NATIVE_LOGISTIC_FUNCTION:
        claimed = False
        if type(value) is types.FunctionType:
            module, name = value.__module__, value.__qualname__
            claimed = (
                type(module) is str
                and type(name) is str
                and module == "jax.lax"
                and name == "logistic"
            )
        if claimed or any(
            vars(module).get("logistic") is value
            for module in (_native_lax, _native_lax_impl)
        ):
            raise TypeError("native logistic metadata received a foreign function")
        return None
    if _NATIVE_LOGISTIC_METADATA_GAP is not None:
        raise TypeError(_NATIVE_LOGISTIC_METADATA_GAP)
    state, _ = _native_logistic_metadata_state()
    for binding_module, name, original in _NATIVE_LOGISTIC_METADATA_BINDINGS:
        if vars(binding_module).get(name) is not original:
            raise TypeError("native logistic implementation binding changed")
    if state != _NATIVE_LOGISTIC_METADATA_STATE:
        raise TypeError("native logistic implementation state changed")
    return {"callable": "jax.lax.logistic", "kwdefaults": {"accuracy": None}}


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
        _native_lax.dot_general_p,
    )
)


def native_aval_metadata_schema() -> str:
    """Own exact native ShapedArray layouts, independently of module symbols."""
    slots = tuple(ShapedArray.__slots__)
    if slots == ("shape", "sharding", "vma", "memory_space"):
        return "vma"
    common = (
        "shape",
        "dtype",
        "weak_type",
        "sharding",
        "manual_axis_type",
        "memory_space",
    )
    if slots == (*common, "__weakref__"):
        return "manual"
    if slots == (*common, "layout", "_stripped_weak_type", "__weakref__"):
        return "manual-layout"
    raise TypeError("native ShapedArray metadata schema needs an exact adapter")


def native_manual_axis_metadata(varying: frozenset[str]) -> tuple[str, Any]:
    """Construct the installed manual-axis role for strict metadata fixtures."""
    if native_aval_metadata_schema() == "vma":
        return "vma", varying
    return "manual_axis_type", _native_core.ManualAxisType(varying=varying)


def _require_plain_native_aval(aval: Any) -> None:
    from jax.sharding import NamedSharding, PartitionSpec
    from jax._src.mesh import AbstractMesh

    if type(aval) is not ShapedArray:
        raise TypeError("native signature requires an ordinary exact ShapedArray")
    schema = native_aval_metadata_schema()
    sharding = aval.sharding
    if not (
        type(sharding) is NamedSharding
        and type(sharding.mesh) is AbstractMesh
        and sharding.mesh.empty
        and sharding.memory_kind is None
        and type(sharding.spec) is PartitionSpec
        and sharding.spec == PartitionSpec(*(None for _ in aval.shape))
        and aval.memory_space is _native_core.MemorySpace.Device
    ):
        raise TypeError("native signature has unsupported sharding or memory space")
    if schema == "vma":
        if type(aval.vma) is not frozenset or aval.vma:
            raise TypeError("native signature has unsupported varying manual axes")
    elif aval.manual_axis_type is not _native_core.empty_mat:
        raise TypeError("native signature has unsupported manual axes")
    if schema == "manual-layout" and aval.layout is not _native_core.AutoLayout:
        raise TypeError("native signature has unsupported explicit layout")


def native_typed_array_payload(value: Any) -> tuple[np.ndarray, bool] | None:
    """Read an exact native host carrier without coercion or metadata mutation.

    The original constant remains bound for lowering. None means this is not a
    native typed-array carrier; ordinary payload and typed-key policy is separate.
    """
    carrier_type = _NATIVE_TYPED_ARRAY_TYPE
    if type(value) is not carrier_type:
        if issubclass(type(value), carrier_type):
            raise TypeError("native typed-array subclasses need a separate adapter")
        return None
    members = vars(carrier_type)
    if not (
        vars(_native_literals).get("TypedNdArray") is carrier_type
        and len(carrier_type.__bases__) == len(_NATIVE_TYPED_ARRAY_BASES)
        and all(
            actual is expected
            for actual, expected in zip(
                carrier_type.__bases__, _NATIVE_TYPED_ARRAY_BASES, strict=True
            )
        )
        and tuple(members) == tuple(name for name, _ in _NATIVE_TYPED_ARRAY_MEMBERS)
        and all(members[name] is member for name, member in _NATIVE_TYPED_ARRAY_MEMBERS)
        and np.asarray is _NATIVE_TYPED_ARRAY_ASARRAY
    ):
        raise TypeError("native typed-array class or conversion binding changed")

    def read_slot(name: str) -> Any:
        descriptor = members[name]
        if type(descriptor) is not types.MemberDescriptorType:
            raise TypeError("native typed-array member is not an actual slot")
        try:
            return descriptor.__get__(value, carrier_type)
        except AttributeError as exc:
            raise TypeError("native typed-array slot is uninitialized") from exc

    slots = members["__slots__"]
    if (
        len(_NATIVE_TYPED_ARRAY_BASES) == 1
        and _NATIVE_TYPED_ARRAY_BASES[0] is object
        and slots == ("val", "weak_type")
    ):
        conversion = _NATIVE_TYPED_ARRAY_ARRAY_FUNCTION
        if not (
            type(conversion) is types.FunctionType
            and _NATIVE_TYPED_ARRAY_ARRAY_STATE is not None
            and conversion.__code__ is _NATIVE_TYPED_ARRAY_ARRAY_STATE[0]
            and conversion.__defaults__ is _NATIVE_TYPED_ARRAY_ARRAY_STATE[1]
            and conversion.__kwdefaults__ is _NATIVE_TYPED_ARRAY_ARRAY_STATE[2]
            and conversion.__closure__ is _NATIVE_TYPED_ARRAY_ARRAY_STATE[3]
            and conversion.__globals__ is _NATIVE_TYPED_ARRAY_ARRAY_STATE[4]
            and conversion.__kwdefaults__ is None
            and conversion.__closure__ is None
            and type(conversion.__defaults__) is tuple
            and len(conversion.__defaults__) == 2
            and all(item is None for item in conversion.__defaults__)
            and conversion.__globals__.get("np") is np
        ):
            raise TypeError("native typed-array conversion implementation changed")
        array = read_slot("val")
        weak = read_slot("weak_type")
        if type(array) is not np.ndarray:
            raise TypeError("native typed-array backing is not an exact ndarray")
    elif (
        len(_NATIVE_TYPED_ARRAY_BASES) == 1
        and _NATIVE_TYPED_ARRAY_BASES[0] is np.ndarray
        and slots
        == (
            "_aval",
            "_weak_type",
        )
    ):
        # Calling .aval/.weak_type here would populate a currently empty cache.
        aval = read_slot("_aval")
        array = np.ndarray.view(value, np.ndarray)
        if aval is None:
            weak = read_slot("_weak_type")
        else:
            if not (
                type(aval) is ShapedArray
                and issubclass(type(aval.dtype), np.dtype)
                and type(aval.shape) is tuple
                and all(type(dim) is int and dim >= 0 for dim in aval.shape)
                and aval.dtype == array.dtype
                and aval.shape == array.shape
            ):
                raise TypeError("native typed-array cached aval disagrees with storage")
            _require_plain_native_aval(aval)
            weak = aval.weak_type
    else:
        raise TypeError("native typed-array schema needs an exact adapter")
    if type(weak) is not bool:
        raise TypeError("native typed-array weakness is not a native Boolean")
    if array.dtype.hasobject:
        raise TypeError("native typed-array object storage is unsupported")
    return array, weak


def _native_context_protocol() -> str:
    slots = tuple(_native_core.JaxprEqnContext.__slots__)
    if slots == (
        "compute_type",
        "threefry_partitionable",
        "xla_metadata",
        "cur_abstract_mesh",
    ):
        return "mutable"
    common = ("compute_type", "threefry_partitionable", "cur_abstract_mesh")
    end = ("remove_size_one_mesh_axis", "xla_metadata", "configs", "__weakref__")
    if slots == (*common, *end):
        return "interned"
    if slots == (*common, "cur_layout_mode", *end):
        return "interned-layout"
    if slots == (
        *common,
        "cur_layout_mode",
        "remove_size_one_mesh_axis",
        "no_collectives",
        "xla_metadata",
        "configs",
        "__weakref__",
    ):
        return "interned-layout-no-collectives"
    raise TypeError("native equation context schema needs an exact adapter")


def _native_metadata_value(value: Any, depth: int = 0) -> Any:
    if depth > 48:
        raise TypeError("native context metadata nesting needs an exact adapter")
    if value is None:
        return ("none",)
    if type(value) is bool:
        return ("bool", value)
    if type(value) is int:
        return ("int", hex(value))
    if type(value) is str:
        return ("str", value)
    if type(value) is bytes:
        return ("bytes", value.hex())
    if type(value) is float:
        return ("float64", struct.pack(">d", value).hex())
    if type(value) is complex:
        return ("complex128", struct.pack(">dd", value.real, value.imag).hex())
    if type(value) is tuple:
        return (
            "tuple",
            tuple(_native_metadata_value(item, depth + 1) for item in value),
        )
    if type(value) is frozenset:
        items = [_native_metadata_value(item, depth + 1) for item in value]
        return ("frozenset", tuple(sorted(items, key=lambda item: json.dumps(item))))
    raise TypeError("native context metadata value needs an exact adapter")


def _native_metadata_dict(value: Any) -> Any:
    if value is None:
        return None
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise TypeError("native context metadata requires an exact string-keyed dict")
    return tuple((key, _native_metadata_value(value[key])) for key in sorted(value))


def current_native_eqn_context() -> Any:
    """Read the current native context through its actual constructor protocol."""
    if _native_context_protocol() == "mutable":
        return _native_core.JaxprEqnContext(
            _native_config.compute_on_context_manager.value,
            _native_config.threefry_partitionable.value,
            _native_xla_metadata.current_xla_metadata(),
        )
    return _native_core.JaxprEqnContext()


def _native_context_fields(context: Any) -> Any:
    if context is None or type(context) is not _native_core.JaxprEqnContext:
        raise TypeError(
            "native signature requires the actual recorded equation context"
        )
    protocol = _native_context_protocol()
    if not (
        (context.compute_type is None or type(context.compute_type) is str)
        and type(context.threefry_partitionable) is bool
        and type(context.cur_abstract_mesh) is _native_mesh.AbstractMesh
        and context.cur_abstract_mesh.empty
    ):
        raise TypeError("native equation context fields are malformed or unsupported")
    fields: tuple[Any, ...] = (
        context.compute_type,
        context.threefry_partitionable,
        id(context.cur_abstract_mesh),
        _native_metadata_dict(context.xla_metadata),
    )
    if protocol != "mutable":
        if type(context.remove_size_one_mesh_axis) is not bool or hasattr(
            context, "configs"
        ):
            raise TypeError("native equation context configuration is unsupported")
        fields += (context.remove_size_one_mesh_axis,)
    if protocol in ("interned-layout", "interned-layout-no-collectives"):
        if context.cur_layout_mode is not _native_core.LayoutMode.AUTO:
            raise TypeError(
                "native signature requires the native automatic layout context"
            )
        fields += (id(context.cur_layout_mode),)
    if protocol == "interned-layout-no-collectives":
        if type(context.no_collectives) is not frozenset or context.no_collectives:
            raise TypeError(
                "native signature requires the native empty no-collectives context"
            )
        fields += (_native_metadata_value(context.no_collectives),)
    return fields


def _native_context_configurations() -> tuple[tuple[str, Any], ...]:
    protocol = _native_context_protocol()
    mesh = (
        _native_mesh.abstract_mesh_context_manager
        if protocol in ("interned-layout", "interned-layout-no-collectives")
        else _native_config.abstract_mesh_context_manager
    )
    result: tuple[tuple[str, Any], ...] = (
        ("compute", _native_config.compute_on_context_manager),
        ("threefry", _native_config.threefry_partitionable),
        ("metadata", _native_config.xla_metadata_context_manager),
        ("mesh", mesh),
    )
    if protocol != "mutable":
        result += (
            ("remove-size-one", _native_config.remove_size_one_mesh_axis_from_type),
        )
    if protocol in ("interned-layout", "interned-layout-no-collectives"):
        result += (("layout", _native_core.layout_tracing_mode),)
    if protocol == "interned-layout-no-collectives":
        result += (("no-collectives", _native_config.no_collectives),)
    return result


def _native_context_state(role: str, value: Any) -> Any:
    if value is _native_xla_metadata.config_ext.unset:
        return ("unset",)
    if role == "metadata":
        if value is None:
            return None
        if (
            type(value) is not _native_xla_metadata.XlaMetadata
            or type(value.hash) is not int
        ):
            raise TypeError("native metadata configuration requires its actual wrapper")
        return (value.hash, _native_metadata_dict(value.val))
    if role == "compute":
        if value is not None and type(value) is not str:
            raise TypeError(
                "native compute configuration requires an actual string or None"
            )
        return value
    if role in ("threefry", "remove-size-one"):
        if type(value) is not bool:
            raise TypeError("native Boolean context configuration is malformed")
        return value
    if role == "mesh":
        if value is not None and type(value) is not _native_mesh.AbstractMesh:
            raise TypeError("native mesh configuration requires an actual AbstractMesh")
        return id(value)
    if role == "layout" and type(value) is _native_core.LayoutMode:
        return id(value)
    if role == "no-collectives" and type(value) is frozenset:
        return _native_metadata_value(value)
    raise TypeError("native context configuration role is unsupported")


@dataclass(frozen=True)
class NativeContextSnapshot:
    context: Any
    states: tuple[tuple[str, Any, Any, Any, Any, Any], ...]


def native_context_snapshot() -> NativeContextSnapshot:
    """Retain exact local/effective roles, including unset and metadata contents."""
    states = []
    for role, configuration in _native_context_configurations():
        local, effective = configuration.get_local(), configuration.value
        states.append(
            (
                role,
                configuration,
                local,
                effective,
                _native_context_state(role, local),
                _native_context_state(role, effective),
            )
        )
    context = (
        None
        if _native_context_protocol() == "mutable"
        else current_native_eqn_context()
    )
    return NativeContextSnapshot(context, tuple(states))


def require_native_context_restored(before: NativeContextSnapshot) -> None:
    after = native_context_snapshot()
    if before.context is not after.context or len(before.states) != len(after.states):
        raise RuntimeError("native equation context was not restored")
    for old, new in zip(before.states, after.states, strict=True):
        if not (
            old[0] == new[0]
            and old[1] is new[1]
            and old[2] is new[2]
            and old[3] is new[3]
            and old[4:] == new[4:]
        ):
            raise RuntimeError(
                "native equation context local/effective state was not restored"
            )


@contextmanager
def recorded_native_context(context: Any) -> Iterator[None]:
    """Use the original native manager and verify caller plus recorded state."""
    recorded = _native_context_fields(context)
    before = native_context_snapshot()
    try:
        with context.manager:
            yield
    finally:
        require_native_context_restored(before)
        if _native_context_fields(context) != recorded:
            raise RuntimeError(
                "recorded native equation context changed during inference"
            )


def infer_native_abstract_results(
    eqn: Any, expected_primitive: Primitive
) -> tuple[tuple[Any, ...], Any]:
    """Infer selected original native avals under their recorded context.

    Callers must first own the parameter and carrier schemas and child bodies.
    This does not validate a lowerer, captures, dataflow or numerical behavior.
    """
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
    _native_context_fields(context)
    for var in (*eqn.invars, *eqn.outvars):
        _require_plain_native_aval(var.aval)
        if any(
            getattr(var, name, None) is not None
            for name in ("initial_qdd", "final_qdd")
        ):
            raise TypeError(
                "native signature does not own quantization variable metadata"
            )
    with recorded_native_context(context):
        inferred, effects = rule.evaluator(
            *(var.aval for var in eqn.invars), **eqn.params
        )
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


def native_nonzero_expansion(eqn: JaxprEqn) -> ClosedJaxpr:
    """Trace only the actual native Nonzero implementation, never a callback.

    This adapter exposes the same composition used before public ``nonzero``
    started emitting HiJAX calls. It does not admit other HiJAX carriers or
    extend the SOURCE profile.
    """
    from jax import ShapeDtypeStruct, make_jaxpr

    if (
        type(eqn) is not JaxprEqn
        or _NATIVE_HI_CALL is None
        or eqn.primitive is not _NATIVE_HI_CALL
        or type(eqn.params) is not dict
        or tuple(eqn.params) != ("_prim",)
    ):
        raise NotImplementedError("HiJAX lowering requires the native Nonzero carrier")
    carrier = eqn.params["_prim"]
    if _NATIVE_NONZERO_TYPE is None or type(carrier) is not _NATIVE_NONZERO_TYPE:
        raise NotImplementedError(
            "only the actual native Nonzero HiJAX carrier is supported"
        )
    fields = vars(carrier)
    if frozenset(fields) != frozenset(
        (
            "in_avals",
            "out_aval",
            "params",
            "in_avals_flat",
            "in_tree",
            "out_avals_flat",
            "out_tree",
            "size",
            "axes",
            "out_dtype",
        )
    ):
        raise TypeError("native Nonzero carrier schema needs an exact adapter")
    parameters = fields["params"]
    if (
        type(parameters) is not dict
        or any(type(key) is not str for key in parameters)
        or set(parameters) != {"size", "axes", "out_dtype"}
    ):
        raise TypeError("native Nonzero parameter schema needs an exact adapter")
    size, axes, dtype = (parameters[name] for name in ("size", "axes", "out_dtype"))
    if not (
        type(size) is int
        and size >= 0
        and type(axes) is tuple
        and all(type(axis) is int for axis in axes)
        and len(axes) == len(set(axes))
        and type(dtype) in (type(np.dtype(np.int32)), type(np.dtype(np.int64)))
        and dtype in (np.dtype(np.int32), np.dtype(np.int64))
        and fields["size"] is size
        and fields["axes"] is axes
        and fields["out_dtype"] is dtype
        and len(eqn.invars) in (1, 1 + len(axes))
        and len(eqn.outvars) == len(axes)
        and not eqn.effects
    ):
        raise TypeError("native Nonzero parameters or arity are unsupported")
    for var in (*eqn.invars, *eqn.outvars):
        _require_plain_native_aval(var.aval)
    rank = len(eqn.invars[0].aval.shape)
    if any(axis < 0 or axis >= rank for axis in axes):
        raise TypeError("native Nonzero axes are out of range")
    implementation = _NATIVE_NONZERO_IMPL
    if implementation is None:
        raise TypeError("native Nonzero implementation is unavailable")
    specs = [
        ShapeDtypeStruct(
            var.aval.shape,
            var.aval.dtype,
            sharding=var.aval.sharding,
            weak_type=var.aval.weak_type,
        )
        for var in eqn.invars
    ]
    with recorded_native_context(eqn.ctx):
        return make_jaxpr(
            functools.partial(implementation, size=size, axes=axes, out_dtype=dtype)
        )(*specs)


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


_NATIVE_MUL_SIGNATURE = tuple(
    (parameter.name, parameter.kind, parameter.default)
    for parameter in inspect.signature(_native_lax_impl.mul).parameters.values()
)


def native_mul_parameter_names() -> frozenset[str]:
    """Own the two native multiply APIs without tolerating optional fields."""
    parameter = inspect.Parameter
    if not (
        type(_NATIVE_MUL_SIGNATURE) is tuple
        and all(
            type(item) is tuple
            and len(item) == 3
            and type(item[0]) is str
            and (
                item[1] is parameter.POSITIONAL_OR_KEYWORD
                or item[1] is parameter.KEYWORD_ONLY
            )
            and (item[2] is parameter.empty or item[2] is None)
            for item in _NATIVE_MUL_SIGNATURE
        )
    ):
        raise TypeError("native multiply parameter schema needs an exact adapter")
    operands = (
        ("x", parameter.POSITIONAL_OR_KEYWORD, parameter.empty),
        ("y", parameter.POSITIONAL_OR_KEYWORD, parameter.empty),
    )
    if _NATIVE_MUL_SIGNATURE == operands:
        return frozenset()
    if _NATIVE_MUL_SIGNATURE == (
        *operands,
        ("out_dtype", parameter.KEYWORD_ONLY, None),
    ):
        return frozenset({"out_dtype"})
    raise TypeError("native multiply parameter schema needs an exact adapter")


_NATIVE_TILE_MISSING: object = object()
_NATIVE_TILE_PRIMITIVE: object = vars(_native_lax_impl).get(
    "tile_p", _NATIVE_TILE_MISSING
)


def native_tile_primitive() -> Primitive | None:
    """Return the captured canonical Tile, or the exact older absence schema."""
    public: object = vars(_native_lax).get("tile_p", _NATIVE_TILE_MISSING)
    defining: object = vars(_native_lax_impl).get("tile_p", _NATIVE_TILE_MISSING)
    if _NATIVE_TILE_PRIMITIVE is _NATIVE_TILE_MISSING:
        if public is _NATIVE_TILE_MISSING and defining is _NATIVE_TILE_MISSING:
            return None
    elif type(_NATIVE_TILE_PRIMITIVE) is Primitive:
        if public is defining is _NATIVE_TILE_PRIMITIVE:
            return _NATIVE_TILE_PRIMITIVE
    raise TypeError("native tile primitive bindings need an exact adapter")


_NATIVE_POLYNOMIAL_MISSING: object = object()
_NATIVE_POLYNOMIAL_LOWER: object = vars(_native_lax_impl).get(
    "_polynomial_lower", _NATIVE_POLYNOMIAL_MISSING
)
_NATIVE_POLYNOMIAL_PRIMITIVE: object = vars(_native_lax_impl).get(
    "polynomial_p", _NATIVE_POLYNOMIAL_MISSING
)


def native_polynomial_lowering(
    primitive: Primitive | None = None,
) -> Callable[..., Array] | None:
    """Return the captured native lowering, with exact absence before JAX 0.12."""
    if primitive is not None and primitive is not _NATIVE_POLYNOMIAL_PRIMITIVE:
        raise TypeError("polynomial lowering requires its captured native primitive")
    public: object = vars(_native_lax).get("polynomial_p", _NATIVE_POLYNOMIAL_MISSING)
    defining: object = vars(_native_lax_impl).get(
        "polynomial_p", _NATIVE_POLYNOMIAL_MISSING
    )
    lowering: object = vars(_native_lax_impl).get(
        "_polynomial_lower", _NATIVE_POLYNOMIAL_MISSING
    )
    if _NATIVE_POLYNOMIAL_PRIMITIVE is _NATIVE_POLYNOMIAL_MISSING:
        if (
            public is defining is _NATIVE_POLYNOMIAL_MISSING
            and lowering is _NATIVE_POLYNOMIAL_LOWER is _NATIVE_POLYNOMIAL_MISSING
        ):
            return None
    elif (
        type(_NATIVE_POLYNOMIAL_PRIMITIVE) is Primitive
        and public is defining is _NATIVE_POLYNOMIAL_PRIMITIVE
        and type(_NATIVE_POLYNOMIAL_LOWER) is types.FunctionType
        and lowering is _NATIVE_POLYNOMIAL_LOWER
    ):
        return cast(Callable[..., Array], _NATIVE_POLYNOMIAL_LOWER)
    raise TypeError("native polynomial lowering bindings need an exact adapter")


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
    return value is _NATIVE_THREEFRY_IMPL


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

    return random.wrap_key_data(words, impl=_NATIVE_THREEFRY_IMPL)


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


_NATIVE_SCAN_ABSTRACT_PARAMETERS = tuple(
    inspect.signature(_native_loops._scan_abstract_eval).parameters
)


def native_scan_parameter_names() -> set[str]:
    """Select only one of the three explicitly owned native scan schemas."""
    common = {"jaxpr", "reverse", "length", "unroll"}
    if _NATIVE_SCAN_ABSTRACT_PARAMETERS == (
        "args",
        "reverse",
        "length",
        "num_consts",
        "num_carry",
        "jaxpr",
        "linear",
        "unroll",
        "_split_transpose",
    ):
        return common | {"num_consts", "num_carry", "linear", "_split_transpose"}
    if _NATIVE_SCAN_ABSTRACT_PARAMETERS == (
        "args",
        "reverse",
        "length",
        "num_consts",
        "num_carry",
        "jaxpr",
        "unroll",
    ):
        return common | {"num_consts", "num_carry"}
    if _NATIVE_SCAN_ABSTRACT_PARAMETERS == (
        "args",
        "reverse",
        "length",
        "ft_in",
        "ft_out",
        "jaxpr",
        "unroll",
    ):
        return common | {"ft_in", "ft_out"}
    raise TypeError("native scan parameter schema needs an exact adapter")


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
    "TypePromotionError",
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
    "native_nonzero_expansion",
    "native_polynomial_lowering",
    "new_jaxpr_eqn",
    "scan_arity",
]
