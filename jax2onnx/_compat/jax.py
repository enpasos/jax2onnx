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
import dis
import json
import sys
import struct
import types
import builtins
import dataclasses as _native_dataclasses
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
from jax._src import api as _native_api
from jax._src import traceback_util as _native_traceback
from jax._src import linear_util as _native_linear_util
from jax._src.interpreters import batching as _native_batching
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
# Preserve the actual native declaration before plugins or mutable metadata.
_NATIVE_AVAL_TYPE = ShapedArray
_NATIVE_AVAL_SLOT_CONTAINER = vars(ShapedArray).get("__slots__")
_NATIVE_AVAL_SLOT_SEQUENCE = (
    tuple(_NATIVE_AVAL_SLOT_CONTAINER)
    if type(_NATIVE_AVAL_SLOT_CONTAINER) in (list, tuple)
    and all(type(name) is str for name in _NATIVE_AVAL_SLOT_CONTAINER)
    else None
)
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
        for name in ("add", "maximum", "minimum", "logical_xor")
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
        if type(value) is slice:
            # CPython may retain constant subscription slices in co_consts.
            # Only builtin descriptors are read; component values keep the same
            # immutable grammar and recursion bound as every other constant.
            return {
                "slice": [
                    encode(vars(slice)[name].__get__(value, slice), depth + 1)
                    for name in ("start", "stop", "step")
                ]
            }
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


def _native_module_dictionary(value: object) -> dict[str, Any] | None:
    """Read only a true module's builtin namespace, never instance __class__."""
    mro = vars(type)["__mro__"].__get__(type(value))
    if not any(base is types.ModuleType for base in mro):
        return None
    return cast(dict[str, Any], vars(types.ModuleType)["__dict__"].__get__(value))


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
        # Keep every identity in native evidence alive, including annotation
        # leaves and executable/code/globals edges outside the node graph.
        self.retained: dict[int, object] = {}

    def retain(self, value: object) -> None:
        self.retained[id(value)] = value

    def _record_bindings(self, value: Any) -> dict[str, Any] | None:
        # __module__ is still captured metadata, but a public re-export is not
        # the executable namespace of a retained native callback.
        kind = type(value)
        if kind is not types.FunctionType and kind is not _NATIVE_CAPTURE_JIT_TYPE:
            raise ValueError("unsupported native executable binding provenance")
        if type(value.__module__) is not str:
            raise ValueError("native implementation module requires an actual string")
        if kind is types.FunctionType:
            function = value
        elif type(value) is _NATIVE_CAPTURE_JIT_TYPE:
            fields = vars(value)
            if not self._schema(fields, _NATIVE_CAPTURE_JIT_FIELDS):
                raise ValueError("native JIT wrapper schema changed")
            function, wrapped = fields["_fun"], fields["__wrapped__"]
            if (
                type(function) is not types.FunctionType
                or type(wrapped) is not types.FunctionType
            ):
                raise ValueError(
                    "native JIT executable provenance needs exact Python functions"
                )
            if function is not wrapped:
                raise ValueError("selected native callable behavior/state changed")
        else:
            raise ValueError("unsupported native executable binding provenance")
        globals_dict = function.__globals__
        self.retain(globals_dict)
        module_name = dict.get(globals_dict, "__name__")
        if type(module_name) is not str:
            return None
        module = sys.modules.get(module_name)
        namespace = _native_module_dictionary(module)
        if namespace is not globals_dict:
            # Already-owned generated functions can have an exec namespace.
            # Keep their exact globals/code state without inventing aliases.
            return None
        self.retain(module)
        for name, item in namespace.copy().items():
            if item is value:
                self.bindings.append((cast(types.ModuleType, module), name, value))
        # This observed proof is part of the existing graph snapshot. Losing a
        # real module registration cannot silently become an exec namespace.
        return {
            "name": module_name,
            "module_identity": id(module),
            "globals_identity": id(globals_dict),
        }

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
        self.retain(value)
        kind = type(value)
        if kind is dict:
            if not all(type(key) is str for key in value):
                raise ValueError("native annotation keys require actual strings")
            return {
                "container_identity": id(value),
                "container_type": "dict",
                "items": {key: self._annotation(item) for key, item in value.items()},
            }
        if kind is tuple or kind is list:
            return {
                "container_identity": id(value),
                "container_type": "tuple" if kind is tuple else "list",
                "items": [self._annotation(item) for item in value],
            }
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

    def _function_defaults(self, function: types.FunctionType, keyword: bool) -> Any:
        return self.take(function.__kwdefaults__ if keyword else function.__defaults__)

    def take(self, value: Any) -> Any:
        self.retain(value)
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
            for edge in (
                value.__code__,
                value.__globals__,
                value.__builtins__,
                value.__closure__,
            ):
                self.retain(edge)
            for cell in value.__closure__ or ():
                self.retain(cell)
            node.update(
                code_identity=id(value.__code__),
                globals_identity=id(value.__globals__),
                builtins_identity=id(value.__builtins__),
                closure_identity=id(value.__closure__),
                cell_identities=[id(cell) for cell in value.__closure__ or ()],
                code_sha256=canonical_code_fingerprint(value.__code__),
                name=self.take(value.__name__),
                qualname=self.take(value.__qualname__),
                module=self.take(value.__module__),
                doc=self.take(value.__doc__),
                defaults=self._function_defaults(value, False),
                kwdefaults=self._function_defaults(value, True),
                attributes=self.take(vars(value)),
                annotations=self._annotation(value.__annotations__),
                closure=[
                    self.take(cell.cell_contents) for cell in value.__closure__ or ()
                ],
            )
            node["defining_namespace"] = self._record_bindings(value)
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
            defining_namespace = self._record_bindings(value)
            node.update(
                defining_namespace=defining_namespace,
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


def _initial_native_capture() -> (
    tuple[dict[int, Any], dict[int, Any], dict[int, str], dict[int, tuple[object, ...]]]
):
    states, bindings, gaps, retained = {}, {}, {}, {}
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
        retained[id(owner)] = tuple(graph.retained.values())
    return states, bindings, gaps, retained


(
    _NATIVE_CAPTURE_BASELINE,
    _NATIVE_CAPTURE_BINDINGS,
    _NATIVE_CAPTURE_GAPS,
    _NATIVE_CAPTURE_RETAINED,
) = _initial_native_capture()


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
        _native_lax.transpose_p,
        _native_lax.add_p,
    )
)


def native_aval_metadata_schema() -> str:
    """Own exact native ShapedArray layouts, independently of module symbols."""
    if ShapedArray is not _NATIVE_AVAL_TYPE:
        raise TypeError("native ShapedArray class binding changed")
    declaration = vars(ShapedArray).get("__slots__")
    if (
        declaration is not _NATIVE_AVAL_SLOT_CONTAINER
        or type(declaration) not in (list, tuple)
        or any(type(name) is not str for name in declaration)
        or tuple(declaration) != _NATIVE_AVAL_SLOT_SEQUENCE
    ):
        raise TypeError("native ShapedArray slot declaration changed")
    slots = tuple(declaration)
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
    if slots == (*common, "_stripped_weak_type", "__weakref__"):
        return "manual-stripped"
    if slots == (*common, "layout", "_stripped_weak_type", "__weakref__"):
        return "manual-layout"
    raise TypeError("native ShapedArray metadata schema needs an exact adapter")


def native_manual_axis_metadata(varying: frozenset[str]) -> tuple[str, Any]:
    """Construct the installed manual-axis role for strict metadata fixtures."""
    if native_aval_metadata_schema() == "vma":
        return "vma", varying
    return "manual_axis_type", _native_core.ManualAxisType(varying=varying)


def _require_manual_stripped_aval_cache(aval: Any) -> None:
    """Read only the measured no-layout native weak-to-strong cache relation."""
    if native_aval_metadata_schema() != "manual-stripped":
        raise TypeError("native signature stripped cache requires its exact schema")
    members = vars(ShapedArray)
    names = (
        "shape",
        "dtype",
        "weak_type",
        "sharding",
        "manual_axis_type",
        "memory_space",
        "_stripped_weak_type",
    )
    slots = {}
    for name in names:
        slot = members.get(name)
        if (
            type(slot) is not types.MemberDescriptorType
            or slot.__objclass__ is not ShapedArray
            or slot.__name__ != name
        ):
            raise TypeError("native signature stripped cache descriptor changed")
        slots[name] = slot

    def read(value: Any) -> dict[str, Any]:
        if type(value) is not ShapedArray:
            raise TypeError(
                "native signature stripped cache requires exact ShapedArray"
            )
        try:
            return {
                name: slot.__get__(value, ShapedArray) for name, slot in slots.items()
            }
        except AttributeError as error:
            raise TypeError(
                "native signature stripped cache is uninitialized"
            ) from error

    original = read(aval)
    if type(original["weak_type"]) is not bool:
        raise TypeError("native signature weakness is not a native Boolean")
    cached = original["_stripped_weak_type"]
    if cached is None:
        return
    strong = read(cached)
    if (
        cached is aval
        or original["weak_type"] is not True
        or strong["weak_type"] is not False
        or strong["_stripped_weak_type"] is not None
        or any(
            strong[name] is not original[name]
            for name in (
                "manual_axis_type",
                "memory_space",
            )
        )
    ):
        # Report only fixed predicate labels from already authenticated reads.
        failures = (
            ("self_cycle", cached is aval),
            ("base_weakness", original["weak_type"] is not True),
            ("cached_weakness", strong["weak_type"] is not False),
            ("cached_terminal", strong["_stripped_weak_type"] is not None),
            (
                "manual_axis_identity",
                strong["manual_axis_type"] is not original["manual_axis_type"],
            ),
            (
                "memory_space_identity",
                strong["memory_space"] is not original["memory_space"],
            ),
        )
        raise TypeError(
            "native signature has invalid stripped weak-type cache: "
            + ", ".join(label for label, failed in failures if failed)
        )
    if not (
        issubclass(type(original["dtype"]), np.dtype)
        and issubclass(type(strong["dtype"]), np.dtype)
        and np.dtype.__eq__(original["dtype"], strong["dtype"]) is True
    ):
        raise TypeError("native signature stripped cache dtype changed")
    left, right = original["shape"], strong["shape"]
    if type(left) is not tuple or type(right) is not tuple or len(left) != len(right):
        raise TypeError("native signature stripped cache shape changed")

    def symbol(value: Any) -> tuple[str, object]:
        from jax._src.export import shape_poly

        # Exact native types precede the established atomic/scope contract;
        # core.is_symbolic_dim alone uses getattr on arbitrary objects.
        if type(value) is not shape_poly._DimExpr:
            raise TypeError("native signature stripped cache has an unowned dimension")
        scope_slot = vars(shape_poly._DimExpr).get("_scope")
        if (
            type(scope_slot) is not types.MemberDescriptorType
            or scope_slot.__objclass__ is not shape_poly._DimExpr
            or scope_slot.__name__ != "_scope"
        ):
            raise TypeError("native signature symbolic scope descriptor changed")
        scope = scope_slot.__get__(value, shape_poly._DimExpr)
        if type(scope) is not shape_poly.SymbolicScope:
            raise TypeError("native signature stripped cache has an unowned scope")
        try:
            name, observed_scope = atomic_symbolic_dimension(value)
        except ValueError as error:
            raise TypeError(
                "native signature stripped cache dimension is not atomic"
            ) from error
        if type(name) is not str or observed_scope is not scope:
            raise TypeError("native signature stripped cache symbolic identity changed")
        return name, scope

    for base_dim, cached_dim in zip(left, right, strict=True):
        if type(base_dim) is int and type(cached_dim) is int:
            if base_dim != cached_dim:
                raise TypeError("native signature stripped cache shape changed")
        else:
            base_name, base_scope = symbol(base_dim)
            cached_name, cached_scope = symbol(cached_dim)
            if base_scope is not cached_scope or base_name != cached_name:
                raise TypeError(
                    "native signature stripped cache symbolic identity changed"
                )

    # Equal native placement keys can reuse an older interned strong aval.
    # Prove its plain placement independently; no arbitrary sharding equality.
    from jax.sharding import NamedSharding, PartitionSpec

    sharding = strong["sharding"]
    if (
        type(sharding) is not NamedSharding
        or type(original["sharding"]) is not NamedSharding
    ):
        raise TypeError(
            "native signature stripped cache requires native plain sharding"
        )
    # Native interning can change the NamedSharding object, while its equal
    # AbstractMesh key retains identity. Reuse the outer guard's owned mesh.
    if sharding.mesh is not original["sharding"].mesh:
        raise TypeError("native signature stripped cache mesh identity changed")
    spec = sharding.spec
    if type(spec) is not PartitionSpec:
        raise TypeError(
            "native signature stripped cache requires native partition spec"
        )
    storage = {}
    for name in ("_partitions", "unreduced", "reduced", "unreduced_kind"):
        descriptor = vars(PartitionSpec).get(name)
        if (
            type(descriptor) is not types.MemberDescriptorType
            or descriptor.__objclass__ is not PartitionSpec
            or descriptor.__name__ != name
        ):
            raise TypeError("native signature stripped cache partition storage changed")
        try:
            storage[name] = descriptor.__get__(spec, PartitionSpec)
        except AttributeError as error:
            raise TypeError(
                "native signature stripped cache partition storage is uninitialized"
            ) from error
    partitions = storage["_partitions"]
    if not (
        type(partitions) is tuple
        and len(partitions) == len(right)
        and all(axis is None for axis in partitions)
        and type(storage["unreduced"]) is frozenset
        and not storage["unreduced"]
        and type(storage["reduced"]) is frozenset
        and not storage["reduced"]
        and storage["unreduced_kind"] is None
    ):
        raise TypeError(
            "native signature stripped cache has nonplain partition storage"
        )
    # Its exact False weakness and terminal None cache were established above,
    # so this existing shared validator cannot recurse into another cache edge.
    _require_plain_native_aval(cached)


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
    if schema == "manual-stripped":
        _require_manual_stripped_aval_cache(aval)


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
            # Keep the carrier's established weakness error at this boundary;
            # the new shared signature guard also owns the cache's bool fields.
            if (
                native_aval_metadata_schema() == "manual-stripped"
                and type(aval.weak_type) is not bool
            ):
                raise TypeError("native typed-array weakness is not a native Boolean")
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


_NATIVE_TOP_K_PRIMITIVE: object = vars(_native_lax_impl).get("top_k_p")
_NATIVE_TOP_K_ABSTRACT: object = vars(_native_lax_impl).get("_top_k_abstract_eval")
_NATIVE_TOP_K_ABSTRACT_CODE: object = (
    _NATIVE_TOP_K_ABSTRACT.__code__
    if type(_NATIVE_TOP_K_ABSTRACT) is types.FunctionType
    else None
)


def native_top_k_primitive() -> Primitive:
    """Return the frozen native TopK after checking both defining aliases."""
    public: object = vars(_native_lax).get("top_k_p")
    defining: object = vars(_native_lax_impl).get("top_k_p")
    if (
        type(_NATIVE_TOP_K_PRIMITIVE) is Primitive
        and public is defining is _NATIVE_TOP_K_PRIMITIVE
    ):
        return _NATIVE_TOP_K_PRIMITIVE
    raise TypeError("native TopK primitive bindings need an exact adapter")


def native_top_k_parameter_names() -> frozenset[str]:
    """Own the exact floor and modern native TopK parameter layouts."""
    function: object = vars(_native_lax_impl).get("_top_k_abstract_eval")
    if (
        type(function) is types.FunctionType
        and function is _NATIVE_TOP_K_ABSTRACT
        and function.__code__ is _NATIVE_TOP_K_ABSTRACT_CODE
        and function.__globals__ is vars(_native_lax_impl)
        and function.__defaults__ is None
        and function.__kwdefaults__ is None
        and function.__closure__ is None
    ):
        code = function.__code__
        if (
            code.co_argcount == 1
            and code.co_posonlyargcount == 0
            and not code.co_flags & 0x0C
            and code.co_varnames[:1] == ("operand",)
        ):
            names = code.co_varnames[1 : 1 + code.co_kwonlyargcount]
            if names in (("k", "axis"), ("k", "axis", "is_stable")):
                return frozenset(names)
    raise TypeError("native TopK parameter schema needs an exact adapter")


_NATIVE_SPLIT_PRIMITIVE: object = vars(_native_lax_impl).get("split_p")


def native_split_primitive() -> Primitive:
    """Return the frozen native Split after checking both defining aliases."""
    public: object = vars(_native_lax).get("split_p")
    defining: object = vars(_native_lax_impl).get("split_p")
    if (
        type(_NATIVE_SPLIT_PRIMITIVE) is Primitive
        and public is defining is _NATIVE_SPLIT_PRIMITIVE
    ):
        return _NATIVE_SPLIT_PRIMITIVE
    raise TypeError("native split primitive bindings need an exact adapter")


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
    FlatTree = cast(type[Any], import_module("jax._src.flattree").FlatTree)

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


# Raw replay's two transformed-function roles are pinned from the loaded native
# factories. This boundary does not construct transformations or execute source.
_NATIVE_WRAPPER_FREEVARS = {
    "reraise_with_filtered_traceback": ("fun",),
    "vmap_f": (
        "axis_name",
        "axis_size",
        "fun",
        "in_axes",
        "out_axes",
        "spmd_axis_name",
        "sum_match",
    ),
}
_NATIVE_WRAPPER_TRACE_HELPERS = (
    "_filtering_mode",
    "_is_under_reraiser",
    "_is_reraiser_frame",
    "_add_tracebackhide_to_hidden_frames",
    "filter_traceback",
    "_add_call_stack_frames",
    "format_exception_only",
    "include_frame",
    "include_filename",
    "_path_starts_with",
    "_ignore_known_hidden_frame",
    "_running_under_ipython",
    "_ipython_supports_tracebackhide",
)
_NATIVE_WRAPPER_VMAP_PATHS = (
    "zip",
    "util.safe_zip",
    "util.jaxlib_utils.safe_zip",
    "_mapped_axis_size",
    "_mapped_axis_spec",
    "_check_ema_unmapped_args",
    "debug_info",
    "flatten_axes",
    "keystr",
    "tree_flatten",
    "tree_flatten_with_path",
    "tree_leaves",
    "tree_unflatten",
    "generate_key_paths",
    "lu.wrap_init",
    "api_util.check_no_transformed_refs_args",
    "api_util.check_no_aliased_ref_args",
    "core.typeof",
    "core.get_axis_env",
    "core.no_axis_name",
    "core.dedup_referents",
    "get_abstract_mesh",
    "sharding_impls.flatten_spec",
    "collections.Counter",
    "inspect.signature",
    "batching.flatten_fun_for_vmap",
    "batching.batch",
    "batching.is_vmappable",
)
_NATIVE_WRAPPER_TRACE_STDLIB_PATHS = (
    "os.path.abspath",
    "os.path.commonpath",
    "os.path.samefile",
    "traceback.walk_tb",
    "traceback.walk_stack",
    "traceback.extract_stack",
    "traceback.format_exception_only",
    "traceback.FrameSummary",
    "types.TracebackType",
)


def _native_wrapper_slot(module: types.ModuleType, path: str) -> tuple[Any, ...]:
    """Retain each explicit module edge; custom descriptors are never followed."""
    value: Any = module
    edges = []
    for name in path.split("."):
        namespace = _native_module_dictionary(value)
        if namespace is None or type(namespace.get("__name__")) is not str:
            raise ValueError("native transformed module path needs an exact adapter")
        if sys.modules.get(namespace["__name__"]) is not value:
            raise ValueError("native transformed defining module binding changed")
        if name not in namespace:
            raise ValueError("native transformed module slot is missing: " + path)
        edges.append((value, namespace, name, namespace[name]))
        value = namespace[name]
    return tuple(edges)


def _native_nested_wrapper_code(owner: Any, name: str) -> types.CodeType:
    if type(owner) is not types.FunctionType:
        raise ValueError("native transformed factory requires its original function")
    codes = [
        item
        for item in owner.__code__.co_consts
        if type(item) is types.CodeType and item.co_name == name
    ]
    if len(codes) != 1 or codes[0].co_freevars != _NATIVE_WRAPPER_FREEVARS[name]:
        raise ValueError("native transformed nested-code schema needs an exact adapter")
    return codes[0]


def _native_wrapper_access_schema(code: types.CodeType) -> frozenset[str]:
    paths = set()
    instructions = list(dis.get_instructions(code))
    for index, instruction in enumerate(instructions):
        if instruction.opname not in {"LOAD_GLOBAL", "LOAD_NAME"}:
            continue
        path = [instruction.argval]
        for following in instructions[index + 1 :]:
            if following.opname not in {"LOAD_ATTR", "LOAD_METHOD"}:
                break
            path.append(following.argval)
        paths.add(".".join(path))
    for constant in code.co_consts:
        if type(constant) is types.CodeType:
            paths.update(_native_wrapper_access_schema(constant))
    return frozenset(paths)


_NATIVE_WRAPPER_ACCESS_SCHEMAS = {
    "reraise_with_filtered_traceback": frozenset(
        (
            "Exception",
            "TypeError",
            "UnfilteredStackTrace",
            "ValueError",
            "_add_call_stack_frames",
            "_add_tracebackhide_to_hidden_frames",
            "_filtering_mode",
            "_is_under_reraiser",
            "_jax_message_append",
            "_simplified_tb_msg",
            "filter_traceback",
            "format_exception_only",
        )
    ),
    "vmap_f": frozenset(
        (
            "ValueError",
            "_check_ema_unmapped_args",
            "_mapped_axis_size",
            "_mapped_axis_spec",
            "any",
            "api_util.check_no_aliased_ref_args",
            "api_util.check_no_transformed_refs_args",
            "batching.AxisData",
            "batching.SpecMatchError",
            "batching.batch",
            "batching.flatten_fun_for_vmap",
            "batching.is_vmappable",
            "batching.infer",
            "config.mutable_array_checks.value",
            "config.remove_size_one_mesh_axis_from_type.value",
            "core.typeof",
            "debug_info",
            "flatten_axes",
            "get_abstract_mesh",
            "isinstance",
            "keystr",
            "len",
            "list",
            "lu.wrap_init",
            "tree_flatten",
            "tree_flatten_with_path",
            "tree_leaves",
            "tree_unflatten",
            "tuple",
            "zip",
        )
    ),
}
_NATIVE_WRAPPER_VMAP_LEGACY_ACCESS_SCHEMA = _NATIVE_WRAPPER_ACCESS_SCHEMAS["vmap_f"] - {
    "list"
}


def _initial_native_wrapper_roles() -> (
    tuple[dict[str, tuple[types.CodeType, dict[str, Any]]], tuple[Any, ...]]
):
    boundary = _native_wrapper_slot(_native_traceback, "api_boundary")[-1][-1]
    reraise_code = _native_nested_wrapper_code(
        boundary, "reraise_with_filtered_traceback"
    )
    public = _native_wrapper_slot(_native_api, "vmap")[-1][-1]
    if not (
        type(public) is types.FunctionType
        and public.__code__ is reraise_code
        and public.__globals__ is _native_module_dictionary(_native_traceback)
        and public.__closure__ is not None
        and len(public.__closure__) == 1
    ):
        raise ValueError("native vmap factory wrapper chain needs an exact adapter")
    owner = public.__closure__[0].cell_contents
    if type(
        owner
    ) is not types.FunctionType or owner.__globals__ is not _native_module_dictionary(
        _native_api
    ):
        raise ValueError("native vmap factory globals need an exact adapter")
    vmap_code = _native_nested_wrapper_code(owner, "vmap_f")
    for name, code in (
        ("reraise_with_filtered_traceback", reraise_code),
        ("vmap_f", vmap_code),
    ):
        schemas: tuple[frozenset[str], ...] = (_NATIVE_WRAPPER_ACCESS_SCHEMAS[name],)
        if name == "vmap_f":
            # The newer native argument-error message also reads list(kwargs).
            schemas += (_NATIVE_WRAPPER_VMAP_LEGACY_ACCESS_SCHEMA,)
        if _native_wrapper_access_schema(code) not in schemas:
            raise ValueError(
                "native transformed direct-access schema needs an exact adapter: "
                + name
            )
    owners = tuple(
        (
            function,
            function.__code__,
            function.__globals__,
            function.__closure__,
            tuple(cell.cell_contents for cell in function.__closure__ or ()),
        )
        for function in (boundary, public, owner)
    )
    return (
        {
            "reraise_with_filtered_traceback": (reraise_code, boundary.__globals__),
            "vmap_f": (vmap_code, owner.__globals__),
        },
        owners,
    )


try:
    _NATIVE_WRAPPER_ROLES, _NATIVE_WRAPPER_FACTORIES = _initial_native_wrapper_roles()
except (TypeError, ValueError) as _wrapper_error:
    _NATIVE_WRAPPER_ROLES, _NATIVE_WRAPPER_FACTORIES = {}, ()
    _NATIVE_WRAPPER_LAYOUT_GAP: str | None = str(_wrapper_error)
else:
    _NATIVE_WRAPPER_LAYOUT_GAP = None

# Only direct fallback loads in the two audited native code bodies are owned.
_NATIVE_WRAPPER_BUILTIN_NAMES = {
    "reraise_with_filtered_traceback": ("Exception", "TypeError", "ValueError"),
    "vmap_f": ("ValueError", "any", "isinstance", "len", "list", "tuple"),
}


def _initial_native_wrapper_lookups() -> tuple[dict[str, Any], tuple[object, ...]]:
    if not _NATIVE_WRAPPER_ROLES:
        raise ValueError("native transformed builtin lookup owners are unavailable")
    factories = tuple(owner[0] for owner in _NATIVE_WRAPPER_FACTORIES)
    factory_builtins = tuple(function.__builtins__ for function in factories)
    if any(type(mapping) is not dict for mapping in factory_builtins):
        raise ValueError(
            "native transformed actual builtin mapping needs an exact adapter"
        )
    roles = {}
    for role, owner_index in (("reraise_with_filtered_traceback", 0), ("vmap_f", 2)):
        code, namespace = _NATIVE_WRAPPER_ROLES[role]
        mapping = factory_builtins[owner_index]
        schema = _native_wrapper_access_schema(code)
        if role == "vmap_f":
            # Both supported stacks resolve this direct load through the native
            # imported safe_zip, not the function's actual builtin mapping.
            native_zip = _native_wrapper_slot(
                _native_api, "util.jaxlib_utils.safe_zip"
            )[-1][-1]
            if (
                type(native_zip) is not types.BuiltinFunctionType
                or namespace.get("zip") is not native_zip
                or _native_wrapper_slot(_native_api, "util.safe_zip")[-1][-1]
                is not native_zip
            ):
                raise ValueError("native transformed imported safe_zip layout changed")
        names = tuple(
            name for name in _NATIVE_WRAPPER_BUILTIN_NAMES[role] if name in schema
        )
        entries = []
        for name in names:
            if name in namespace or name not in mapping:
                raise ValueError(
                    "native transformed direct builtin fallback layout changed"
                )
            entries.append((name, mapping[name]))
        roles[role] = {
            "builtins": mapping,
            "globals": namespace,
            "entries": tuple(entries),
        }
    if factory_builtins[1] is not roles["reraise_with_filtered_traceback"]["builtins"]:
        raise ValueError("native transformed original wrapper builtin mapping changed")
    return roles, factory_builtins


try:
    _NATIVE_WRAPPER_LOOKUP_ROLES, _NATIVE_WRAPPER_FACTORY_BUILTINS = (
        _initial_native_wrapper_lookups()
    )
except (KeyError, TypeError, ValueError) as _wrapper_lookup_error:
    _NATIVE_WRAPPER_LOOKUP_ROLES, _NATIVE_WRAPPER_FACTORY_BUILTINS = {}, ()
    _NATIVE_WRAPPER_LOOKUP_GAP: str | None = str(_wrapper_lookup_error)
else:
    _NATIVE_WRAPPER_LOOKUP_GAP = None


def _native_wrapper_builtin_state(role: str, graph: _NativeCaptureGraph) -> Any:
    if _NATIVE_WRAPPER_LOOKUP_GAP is not None:
        raise ValueError(_NATIVE_WRAPPER_LOOKUP_GAP)
    entry = _NATIVE_WRAPPER_LOOKUP_ROLES[role]
    namespace, mapping = entry["globals"], entry["builtins"]
    graph.retain(namespace)
    graph.retain(mapping)
    states = {}
    for name, original in entry["entries"]:
        if name in namespace:
            raise ValueError("native transformed direct builtin global shadow: " + name)
        if name not in mapping or mapping[name] is not original:
            raise ValueError(
                "native transformed direct builtin binding changed: " + name
            )
        graph.retain(original)
        states[name] = {
            "global_present": False,
            "builtin_present": True,
            "resolution": "actual_function_builtins",
            "resolved_identity": id(original),
        }
    return {
        "globals_identity": id(namespace),
        "builtins_identity": id(mapping),
        "direct_fallbacks": states,
    }


_NATIVE_WRAPPER_SENTINEL = vars(_native_core).get("no_axis_name")
_NATIVE_WRAPPER_EXCLUSIONS = vars(_native_traceback).get("_exclude_paths")
_NATIVE_WRAPPER_REGISTRY = vars(_native_batching).get("vmappables")
_NATIVE_WRAPPER_CLASS_ROLES = {
    "reraise_with_filtered_traceback": ((_native_traceback, "UnfilteredStackTrace"),),
    "vmap_f": (
        (_native_batching, "AxisData"),
        (_native_batching, "SpecMatchError"),
        (_native_batching, "Infer"),
        (_native_linear_util, "Store"),
        (_native_linear_util, "EqualStore"),
    ),
}
_NATIVE_WRAPPER_CLASSES = {
    (role, name): vars(module).get(name)
    for role, entries in _NATIVE_WRAPPER_CLASS_ROLES.items()
    for module, name in entries
}
_NATIVE_WRAPPER_CONFIGS = {
    name: vars(_native_config).get(name)
    for name in (
        "traceback_filtering",
        "mutable_array_checks",
        "remove_size_one_mesh_axis_from_type",
    )
}
_NATIVE_WRAPPER_INFER_TYPE = vars(_native_batching).get("Infer")
_NATIVE_WRAPPER_INFER_DICT = (
    vars(type)["__dict__"].__get__(_NATIVE_WRAPPER_INFER_TYPE).get("__dict__")
    if type(_NATIVE_WRAPPER_INFER_TYPE) is type
    else None
)


def _native_wrapper_dictionary_descriptor(cls: type) -> tuple[type, Any]:
    for base in vars(type)["__mro__"].__get__(cls):
        descriptor = vars(type)["__dict__"].__get__(base).get("__dict__")
        if descriptor is not None:
            if type(descriptor) is not types.GetSetDescriptorType:
                raise ValueError(
                    "native transformed context dictionary needs a builtin descriptor"
                )
            return base, descriptor
    raise ValueError(
        "native transformed context dictionary layout needs an exact adapter"
    )


def _initial_native_wrapper_context_roles() -> dict[str, Any]:
    axis_type = vars(_native_core).get("AxisEnv")
    mesh_type = vars(_native_mesh).get("AbstractMesh")
    eqn_type = vars(_native_core).get("JaxprEqnContext")
    if any(type(cls) is not type for cls in (axis_type, mesh_type, eqn_type)):
        raise ValueError(
            "native transformed context class layout needs an exact adapter"
        )
    axis_slots = tuple(
        (name, vars(axis_type).get(name))
        for name in ("axis_sizes", "spmd_axis_names", "explicit_mesh_axis_names")
    )
    if any(
        type(descriptor) is not types.MemberDescriptorType
        for _, descriptor in axis_slots
    ):
        raise ValueError(
            "native transformed axis slots need original member descriptors"
        )
    top_axis = vars(_native_core).get("top_axis_env")
    if type(top_axis) is not axis_type:
        raise ValueError("native transformed original axis environment layout changed")
    mapping_type = type(axis_slots[0][1].__get__(top_axis, axis_type))
    context_configs = _native_context_configurations()
    axis_config = vars(_native_core).get("axis_env_state")
    owners = (
        *_NATIVE_WRAPPER_CONFIGS.values(),
        axis_config,
        *(entry[1] for entry in context_configs),
    )
    if any(owner is None for owner in owners):
        raise ValueError("native transformed configuration owner is missing")
    return {
        "axis_type": axis_type,
        "mesh_type": mesh_type,
        "eqn_type": eqn_type,
        "axis_slots": axis_slots,
        "axis_mapping_type": mapping_type,
        "axis_mapping_dict": _native_wrapper_dictionary_descriptor(mapping_type),
        "mesh_dict": _native_wrapper_dictionary_descriptor(mesh_type),
        "context_configs": context_configs,
        "axis_config": axis_config,
        "empty_mesh": vars(_native_mesh).get("empty_abstract_mesh"),
        "configuration_types": tuple(
            (owner, type(owner), vars(type)["__mro__"].__get__(type(owner)))
            for owner in owners
        ),
        "configuration_class_members": tuple(
            (cls, tuple(vars(type)["__dict__"].__get__(cls).items()))
            for owner in owners
            for cls in vars(type)["__mro__"].__get__(type(owner))
        ),
    }


try:
    _NATIVE_WRAPPER_CONTEXT_ROLES = _initial_native_wrapper_context_roles()
except (AttributeError, KeyError, TypeError, ValueError) as _wrapper_context_error:
    _NATIVE_WRAPPER_CONTEXT_ROLES = {}
    _NATIVE_WRAPPER_CONTEXT_GAP: str | None = str(_wrapper_context_error)
else:
    _NATIVE_WRAPPER_CONTEXT_GAP = None


def _native_wrapper_class_state(
    graph: _NativeCaptureGraph, role: str, name: str
) -> Any:
    if role == "vmap_f" and name == "AxisData":
        if _NATIVE_AXISDATA_GAP is not None:
            raise ValueError(_NATIVE_AXISDATA_GAP)
        codec = _AxisDataCaptureGraph()
        document = codec.state()
        graph.retained.update(codec.retained)
        graph.bindings.extend(codec.bindings)
        return document
    cls = _NATIVE_WRAPPER_CLASSES[role, name]
    if type(cls) is not type:
        raise ValueError(
            "native transformed class metaclass needs an exact adapter: " + name
        )
    namespace = vars(type)["__dict__"].__get__(cls)
    if not all(type(key) is str for key in namespace):
        raise ValueError("native transformed class member schema changed: " + name)
    for item in (
        cls,
        vars(type)["__bases__"].__get__(cls),
        vars(type)["__mro__"].__get__(cls),
    ):
        graph.retain(item)
    symbols = {}
    for field_name in ("__name__", "__qualname__"):
        symbol = vars(type)[field_name].__get__(cls)
        if type(symbol) is not str:
            raise ValueError(
                "native transformed class symbol requires an actual string"
            )
        graph.retain(symbol)
        symbols[field_name] = symbol
    try:
        members = {key: graph.take(value) for key, value in sorted(namespace.items())}
    except (TypeError, ValueError) as exc:
        # In particular, AxisData's generated dataclass metadata is not granted
        # a new general class/annotation policy by this implementation boundary.
        raise ValueError(
            "native transformed class own-member layout needs an exact adapter: " + name
        ) from exc
    return {
        "identity": id(cls),
        "metaclass": id(type(cls)),
        "bases": [id(base) for base in vars(type)["__bases__"].__get__(cls)],
        "mro": [id(base) for base in vars(type)["__mro__"].__get__(cls)],
        "symbols": symbols,
        "members": members,
    }


_NATIVE_AXISDATA_FIELD_SLOTS = (
    "name",
    "type",
    "default",
    "default_factory",
    "repr",
    "hash",
    "init",
    "compare",
    "metadata",
    "kw_only",
    "_field_type",
)
_NATIVE_AXISDATA_PARAMS_SLOTS = ("init", "repr", "eq", "order", "unsafe_hash", "frozen")
_NATIVE_AXISDATA_EXPORTED_MEMBERS = frozenset(
    (
        "__module__",
        "__annotations__",
        "explicit_mesh_axis",
        "__repr__",
        "__str__",
        "__doc__",
        "__dataclass_params__",
        "__dataclass_fields__",
        "__init__",
        "__eq__",
        "__setattr__",
        "__delattr__",
        "__hash__",
        "__match_args__",
        "__slots__",
        "_ema",
        "name",
        "size",
        "spmd_name",
        "__getstate__",
        "__setstate__",
    )
)
_NATIVE_AXISDATA_PRESLOTS_MEMBERS = frozenset(
    (
        "__module__",
        "__annotations__",
        "explicit_mesh_axis",
        "__repr__",
        "__str__",
        "__dict__",
        "__weakref__",
        "__doc__",
        "__dataclass_params__",
        "__dataclass_fields__",
        "__init__",
        "__eq__",
        "__setattr__",
        "__delattr__",
        "__hash__",
        "__match_args__",
    )
)
_NATIVE_AXISDATA_EXTRA_MEMBERS = frozenset(
    ("__firstlineno__", "__replace__", "__static_attributes__")
)


def _initial_native_axisdata_roles() -> dict[str, Any]:
    exported = _NATIVE_WRAPPER_CLASSES["vmap_f", "AxisData"]
    if type(exported) is not type:
        raise ValueError("native AxisData requires builtin type metaclass")
    namespace = vars(type)["__dict__"].__get__(exported)
    closures = []
    closure_schemas = []
    for name in ("__setattr__", "__delattr__"):
        function = namespace[name]
        is_function = type(function) is types.FunctionType
        freevars = function.__code__.co_freevars if is_function else None
        literal_freevars = type(freevars) is tuple and all(
            type(symbol) is str for symbol in freevars
        )
        if (
            not is_function
            or not literal_freevars
            or freevars
            not in (
                ("FrozenInstanceError", "cls"),
                ("FrozenInstanceError", "__class__"),
            )
        ):
            if not is_function:
                detail = name + ": non-function"
            elif not literal_freevars:
                detail = name + ": non-literal-freevars"
            else:
                detail = (
                    name
                    + f": co_freevars={freevars!r}, "
                    + f"closure_length={len(function.__closure__ or ())}"
                )
            raise ValueError(
                "native AxisData generated frozen closure needs an exact adapter: "
                + detail
            )
        closure = function.__closure__
        if (
            type(closure) is not tuple
            or len(closure) != 2
            or any(type(cell) is not types.CellType for cell in closure)
        ):
            raise ValueError("native AxisData frozen closure cell layout changed")
        # Read only exact builtin cells, and reject an uninitialized cell before
        # selecting either of the two measured dataclasses layouts.
        try:
            tuple(cell.cell_contents for cell in closure)
        except ValueError as exc:
            raise ValueError("native AxisData frozen closure cell is empty") from exc
        closures.append(closure)
        closure_schemas.append(freevars)
    modern_closure = closure_schemas[0] == ("FrozenInstanceError", "__class__")
    frozen, closed_class = (cell.cell_contents for cell in closures[0])
    if not (
        closure_schemas[1] == closure_schemas[0]
        and closures[1][0].cell_contents is frozen
        and closures[1][1].cell_contents is closed_class
        and type(closed_class) is type
        and frozen is vars(_native_dataclasses)["FrozenInstanceError"]
        and (
            closed_class is exported
            and all(left is right for left, right in zip(*closures, strict=True))
            if modern_closure
            else closed_class is not exported
        )
    ):
        raise ValueError("native AxisData exported/pre-slots closure aliases changed")
    classes = {"exported": exported}
    namespaces: tuple[Any, ...]
    if modern_closure:
        # Modern dataclasses repair __class__ cells to the exported slots class;
        # no distinct pre-slots class remains reachable through these closures.
        namespaces = (namespace,)
        legacy_namespace = False
        extended_namespace = _NativeCaptureGraph._schema(
            namespace,
            _NATIVE_AXISDATA_EXPORTED_MEMBERS | _NATIVE_AXISDATA_EXTRA_MEMBERS,
        )
    else:
        old_namespace = vars(type)["__dict__"].__get__(closed_class)
        classes["pre-slots"] = closed_class
        namespaces = (namespace, old_namespace)
        legacy_namespace = _NativeCaptureGraph._schema(
            namespace, _NATIVE_AXISDATA_EXPORTED_MEMBERS
        ) and _NativeCaptureGraph._schema(
            old_namespace, _NATIVE_AXISDATA_PRESLOTS_MEMBERS
        )
        extended_namespace = _NativeCaptureGraph._schema(
            namespace,
            _NATIVE_AXISDATA_EXPORTED_MEMBERS | _NATIVE_AXISDATA_EXTRA_MEMBERS,
        ) and _NativeCaptureGraph._schema(
            old_namespace,
            _NATIVE_AXISDATA_PRESLOTS_MEMBERS | _NATIVE_AXISDATA_EXTRA_MEMBERS,
        )
    classes["FrozenInstanceError"] = frozen
    if not (legacy_namespace or extended_namespace):
        raise ValueError("native AxisData class namespace needs an exact adapter")
    replacement = None
    if extended_namespace:
        replacement = vars(_native_dataclasses).get("_replace")
        if type(replacement) is not types.FunctionType or any(
            type(members["__firstlineno__"]) is not int
            or type(members["__static_attributes__"]) is not tuple
            or len(members["__static_attributes__"]) != 0
            or members["__replace__"] is not replacement
            for members in namespaces
        ):
            raise ValueError("native AxisData extended class metadata changed")
    fields = namespace["__dataclass_fields__"]
    params = namespace["__dataclass_params__"]
    names = ("name", "size", "spmd_name", "_ema")
    if not (
        type(fields) is dict
        and all(type(key) is str for key in fields)
        and tuple(fields) == names
        and all(
            members["__dataclass_fields__"] is fields
            and members["__dataclass_params__"] is params
            for members in namespaces
        )
    ):
        raise ValueError("native AxisData shared metadata layout changed")
    field_type = vars(_native_dataclasses)["Field"]
    params_type = vars(_native_dataclasses)["_DataclassParams"]
    extended_params_slots = (
        *_NATIVE_AXISDATA_PARAMS_SLOTS,
        "match_args",
        "kw_only",
        "slots",
        "weakref_slot",
    )
    field_slots: tuple[tuple[str, ...], ...] = (_NATIVE_AXISDATA_FIELD_SLOTS,)
    if modern_closure:
        field_slots += ((*_NATIVE_AXISDATA_FIELD_SLOTS[:-1], "doc", "_field_type"),)
    carriers = {}
    for name, cls, allowed in (
        ("Field", field_type, field_slots),
        (
            "_DataclassParams",
            params_type,
            (
                (extended_params_slots,)
                if modern_closure
                else (_NATIVE_AXISDATA_PARAMS_SLOTS, extended_params_slots)
            ),
        ),
    ):
        if type(cls) is not type:
            raise ValueError("native AxisData storage carrier metaclass changed")
        members = vars(type)["__dict__"].__get__(cls)
        slots = members.get("__slots__")
        if (
            type(slots) is not tuple
            or any(type(slot) is not str for slot in slots)
            or slots not in allowed
        ):
            raise ValueError(
                "native AxisData storage slot schema needs an exact adapter"
            )
        descriptors = tuple(members.get(slot) for slot in slots)
        if any(
            type(descriptor) is not types.MemberDescriptorType
            for descriptor in descriptors
        ):
            raise ValueError(
                "native AxisData storage requires original native slot descriptors"
            )
        carriers[name] = (cls, slots, descriptors, vars(type)["__bases__"].__get__(cls))
    if type(params) is not params_type or any(
        type(field) is not field_type for field in fields.values()
    ):
        raise ValueError("native AxisData storage object identity/type layout changed")
    field_descriptors = dict(
        zip(carriers["Field"][1], carriers["Field"][2], strict=True)
    )
    if "doc" in field_descriptors and any(
        field_descriptors["doc"].__get__(field, field_type) is not None
        for field in fields.values()
    ):
        raise ValueError("native AxisData Field doc state changed")
    metadata = tuple(
        field_descriptors["metadata"].__get__(field, field_type)
        for field in fields.values()
    )
    if any(
        type(value) is not types.MappingProxyType or len(value) != 0
        for value in metadata
    ):
        raise ValueError(
            "native AxisData field metadata needs its original empty mappingproxy"
        )
    marker_names: tuple[str, ...] = ("MISSING", "_FIELD")
    if extended_namespace:
        marker_names += ("_FIELD_CLASSVAR", "_FIELD_INITVAR")
    markers = {name: vars(_native_dataclasses)[name] for name in marker_names}
    marker_descriptors = {}
    for name, value in markers.items():
        cls = type(value)
        descriptor = vars(cls).get("__dict__")
        if type(cls) is not type or type(descriptor) is not types.GetSetDescriptorType:
            raise ValueError(
                "native AxisData marker instance dictionary needs an exact adapter"
            )
        marker_descriptors[name] = (cls, descriptor)
        if name in {"_FIELD_CLASSVAR", "_FIELD_INITVAR"}:
            dictionary = descriptor.__get__(value, cls)
            if (
                cls is not type(markers["_FIELD"])
                or type(dictionary) is not dict
                or any(type(key) is not str for key in dictionary)
                or set(dictionary) != {"name"}
                or type(dictionary["name"]) is not str
                or dictionary["name"] != name
            ):
                raise ValueError("native AxisData extended marker metadata changed")
    helpers = {
        name: vars(_native_dataclasses)[name]
        for name in ("_dataclass_getstate", "_dataclass_setstate", "fields", "_FIELDS")
    }
    if extended_namespace:
        helpers["_replace"] = replacement
    if (
        namespace["__getstate__"] is not helpers["_dataclass_getstate"]
        or namespace["__setstate__"] is not helpers["_dataclass_setstate"]
    ):
        raise ValueError("native AxisData state helper bindings changed")
    return {
        "classes": classes,
        "fields": fields,
        "field_names": names,
        "field_objects": tuple(fields.values()),
        "params": params,
        "carriers": carriers,
        "metadata": metadata,
        "markers": markers,
        "marker_descriptors": marker_descriptors,
        "helpers": helpers,
    }


try:
    _NATIVE_AXISDATA = _initial_native_axisdata_roles()
except (KeyError, TypeError, ValueError) as _axisdata_error:
    _NATIVE_AXISDATA = {}
    _NATIVE_AXISDATA_GAP: str | None = str(_axisdata_error)
else:
    _NATIVE_AXISDATA_GAP = None


class _AxisDataCaptureGraph(_NativeCaptureGraph):
    """Only the audited native AxisData metadata roles extend this envelope."""

    def take(self, value: Any) -> Any:
        self.retain(value)
        for role, cls in _NATIVE_AXISDATA["classes"].items():
            if value is cls:
                return {"native_axisdata_class_role": role, "identity": id(value)}
        if value is _NATIVE_AXISDATA["fields"]:
            names = _NATIVE_AXISDATA["field_names"]
            if (
                type(value) is not dict
                or any(type(key) is not str for key in value)
                or tuple(value) != names
            ):
                raise ValueError("native AxisData field mapping schema changed")
            if any(
                value[name] is not original
                for name, original in zip(
                    names, _NATIVE_AXISDATA["field_objects"], strict=True
                )
            ):
                raise ValueError(
                    "native AxisData field mapping replaced an original Field"
                )
            return {
                "identity": id(value),
                "fields": [[name, self.take(value[name])] for name in names],
            }
        if any(value is field for field in _NATIVE_AXISDATA["field_objects"]):
            return self._storage("Field", value)
        if value is _NATIVE_AXISDATA["params"]:
            return self._storage("_DataclassParams", value)
        for role, marker in _NATIVE_AXISDATA["markers"].items():
            if value is marker:
                cls, descriptor = _NATIVE_AXISDATA["marker_descriptors"][role]
                if (
                    type(value) is not cls
                    or vars(cls).get("__dict__") is not descriptor
                ):
                    raise ValueError("native AxisData marker storage changed")
                namespace = descriptor.__get__(value, cls)
                if type(namespace) is not dict or any(
                    type(key) is not str for key in namespace
                ):
                    raise ValueError(
                        "native AxisData marker fields require an exact dictionary"
                    )
                if set(namespace) != (set() if role == "MISSING" else {"name"}):
                    raise ValueError("native AxisData marker field schema changed")
                if role != "MISSING" and (
                    type(namespace["name"]) is not str or namespace["name"] != role
                ):
                    raise ValueError("native AxisData field marker value changed")
                self.retain(cls)
                self.retain(descriptor)
                return {
                    "native_axisdata_marker": role,
                    "identity": id(value),
                    "class": self._class_document(cls),
                    "fields": super().take(namespace),
                }
        if any(value is metadata for metadata in _NATIVE_AXISDATA["metadata"]):
            if type(value) is not types.MappingProxyType or len(value):
                raise ValueError("native AxisData metadata must stay empty")
            return {"native_axisdata_empty_metadata": id(value), "items": []}
        return super().take(value)

    def _storage(self, name: str, value: Any) -> Any:
        cls, slots, descriptors, bases = _NATIVE_AXISDATA["carriers"][name]
        if (
            type(value) is not cls
            or vars(_native_dataclasses).get(name) is not cls
            or type(cls) is not type
        ):
            raise ValueError("native AxisData storage carrier binding changed")
        namespace = vars(type)["__dict__"].__get__(cls)
        if (
            namespace.get("__slots__") is not slots
            or vars(type)["__bases__"].__get__(cls) is not bases
        ):
            raise ValueError("native AxisData storage layout changed")
        fields = {}
        for slot, descriptor in zip(slots, descriptors, strict=True):
            if namespace.get(slot) is not descriptor:
                raise ValueError("native AxisData original slot descriptor changed")
            self.retain(descriptor)
            item = descriptor.__get__(value, cls)
            if name == "_DataclassParams" and type(item) is not bool:
                raise ValueError("native AxisData params require actual Boolean flags")
            if name == "_DataclassParams":
                expected = slot in {
                    "init",
                    "repr",
                    "eq",
                    "frozen",
                    "match_args",
                    "slots",
                }
                if item is not expected:
                    raise ValueError("native AxisData original params flag changed")
            else:
                index = next(
                    index
                    for index, original in enumerate(_NATIVE_AXISDATA["field_objects"])
                    if value is original
                )
                expected_field_name = _NATIVE_AXISDATA["field_names"][index]
                if slot in {"name", "type"}:
                    expected_text = (
                        expected_field_name
                        if slot == "name"
                        else ("tuple[Any, ...] | None" if index == 3 else "Any")
                    )
                    if type(item) is not str or item != expected_text:
                        raise ValueError("native AxisData Field name/type changed")
                elif (
                    slot in {"default", "default_factory"}
                    and item is not _NATIVE_AXISDATA["markers"]["MISSING"]
                ):
                    raise ValueError(
                        "native AxisData Field default needs an exact adapter"
                    )
                elif slot in {"repr", "init", "compare"} and item is not True:
                    raise ValueError("native AxisData Field Boolean flag changed")
                elif slot == "hash" and item is not None:
                    raise ValueError("native AxisData Field hash state changed")
                elif slot == "doc" and item is not None:
                    raise ValueError("native AxisData Field doc state changed")
                elif slot == "kw_only" and item is not False:
                    raise ValueError("native AxisData Field keyword flag changed")
                elif (
                    slot == "metadata"
                    and item is not _NATIVE_AXISDATA["metadata"][index]
                ):
                    raise ValueError(
                        "native AxisData original metadata identity changed"
                    )
                elif (
                    slot == "_field_type"
                    and item is not _NATIVE_AXISDATA["markers"]["_FIELD"]
                ):
                    raise ValueError("native AxisData Field marker changed")
            fields[slot] = self.take(item)
        for item in (cls, slots, bases):
            self.retain(item)
        return {
            "identity": id(value),
            "carrier": id(cls),
            "slots": fields,
            "slot_descriptors": [id(descriptor) for descriptor in descriptors],
            "metaclass": id(type(cls)),
            "bases": [id(base) for base in bases],
        }

    def _class_document(self, cls: type) -> Any:
        if type(cls) is not type:
            raise ValueError("native AxisData role class metaclass changed")
        namespace = vars(type)["__dict__"].__get__(cls)
        if not all(type(key) is str for key in namespace):
            raise ValueError("native AxisData class member keys changed")
        bases, mro = vars(type)["__bases__"].__get__(cls), vars(type)[
            "__mro__"
        ].__get__(cls)
        for value in (cls, bases, mro):
            self.retain(value)
        symbols = {}
        for name in ("__name__", "__qualname__"):
            value = vars(type)[name].__get__(cls)
            if type(value) is not str:
                raise ValueError(
                    "native AxisData class metadata requires actual strings"
                )
            self.retain(value)
            symbols[name] = value
        return {
            "identity": id(cls),
            "metaclass": id(type(cls)),
            "symbols": symbols,
            "bases": [id(base) for base in bases],
            "mro": [id(base) for base in mro],
            "members": {
                name: self.take(value) for name, value in sorted(namespace.items())
            },
        }

    def state(self) -> Any:
        for name in (
            "Field",
            "_DataclassParams",
            "FrozenInstanceError",
            "MISSING",
            "_FIELD",
            "fields",
            "_FIELDS",
        ):
            _native_wrapper_slot(_native_dataclasses, name)
        if "_replace" in _NATIVE_AXISDATA["helpers"]:
            for name in ("_replace", "_FIELD_CLASSVAR", "_FIELD_INITVAR"):
                _native_wrapper_slot(_native_dataclasses, name)
        for name, original in _NATIVE_AXISDATA["helpers"].items():
            if vars(_native_dataclasses).get(name) is not original:
                raise ValueError(
                    "native AxisData state helper defining binding changed"
                )
        for name, original in _NATIVE_AXISDATA["markers"].items():
            if vars(_native_dataclasses).get(name) is not original:
                raise ValueError("native AxisData marker defining binding changed")
        classes = _NATIVE_AXISDATA["classes"]
        if (
            vars(_native_batching).get("AxisData") is not classes["exported"]
            or vars(_native_dataclasses).get("FrozenInstanceError")
            is not classes["FrozenInstanceError"]
        ):
            raise ValueError("native AxisData class defining binding changed")
        documents = {role: self._class_document(cls) for role, cls in classes.items()}
        helpers = {
            name: self.take(value)
            for name, value in _NATIVE_AXISDATA["helpers"].items()
        }
        markers = {
            name: self.take(value)
            for name, value in _NATIVE_AXISDATA["markers"].items()
        }
        return {
            "classes": documents,
            "state_helpers": helpers,
            "state_markers": markers,
            "nodes": self.nodes,
        }


# These two original default roles are an edge-only implementation dependency.
# They never participate in the ordinary native/source graph admission policy.
def _initial_native_helper_default_roles() -> dict[str, Any]:
    roles: dict[str, Any] = {}
    for name, role, module, function_path, keyword, class_name, instance_name in (
        (
            "debug_info",
            "vmap_f",
            _native_api,
            "api_util.debug_info",
            True,
            "InitialResultPaths",
            "initial_result_paths",
        ),
        (
            "format_exception_only",
            "reraise_with_filtered_traceback",
            _native_traceback,
            "traceback.format_exception_only",
            False,
            "_Sentinel",
            "_sentinel",
        ),
    ):
        function_edges = _native_wrapper_slot(module, function_path)
        function = function_edges[-1][-1]
        defining_module = function_edges[-1][0]
        if type(
            function
        ) is not types.FunctionType or function.__globals__ is not _native_module_dictionary(
            defining_module
        ):
            raise ValueError("native helper default needs original function/globals")
        container = function.__kwdefaults__ if keyword else function.__defaults__
        if keyword:
            if type(container) is not dict or not _NativeCaptureGraph._schema(
                container,
                frozenset(
                    (
                        "result_paths_thunk",
                        "static_argnums",
                        "static_argnames",
                        "sourceinfo",
                        "signature",
                    )
                ),
            ):
                raise ValueError("native debug_info keyword-default schema changed")
            instance = container["result_paths_thunk"]
            paths: tuple[str, ...] = (
                "debug_info",
                "api_util.debug_info",
                "core.InitialResultPaths",
                "core.initial_result_paths",
                "lu.InitialResultPaths",
                "lu.initial_result_paths",
            )
            if (
                cast(dict[str, Any], _native_module_dictionary(module)).get("core")
                is not _native_core
                or cast(dict[str, Any], _native_module_dictionary(module)).get("lu")
                is not _native_linear_util
            ):
                raise ValueError(
                    "native helper original core/linear-util module aliases changed"
                )
            owner_module = _native_linear_util
        else:
            if type(container) is not tuple or len(container) != 1:
                raise ValueError(
                    "native stdlib helper positional-default schema changed"
                )
            instance = container[0]
            paths = (
                "format_exception_only",
                "traceback.format_exception_only",
                "traceback._Sentinel",
                "traceback._sentinel",
            )
            owner_module = defining_module
        edges = tuple(
            edge for path in paths for edge in _native_wrapper_slot(module, path)
        )
        cls = type(instance)
        if (
            type(cls) is not type
            or cast(dict[str, Any], _native_module_dictionary(owner_module)).get(
                class_name
            )
            is not cls
            or cast(dict[str, Any], _native_module_dictionary(owner_module)).get(
                instance_name
            )
            is not instance
        ):
            raise ValueError("native helper original class/instance aliases changed")
        for _, _, slot, item in edges:
            if slot == class_name and item is not cls:
                raise ValueError("native helper class alias split")
            if slot == instance_name and item is not instance:
                raise ValueError("native helper singleton alias split")
            if slot in ("debug_info", "format_exception_only") and item is not function:
                # The JAX formatter is distinct from the stdlib formatter.
                if not (
                    not keyword
                    and slot == "format_exception_only"
                    and item
                    is cast(
                        dict[str, Any], _native_module_dictionary(_native_traceback)
                    ).get(slot)
                ):
                    raise ValueError("native helper function alias split")
        namespace = vars(type)["__dict__"].__get__(cls)
        expected = frozenset(("__module__", "__dict__", "__weakref__", "__doc__"))
        if not keyword:
            expected |= {"__repr__"}
        if sys.version_info[:2] == (3, 13):
            expected |= {"__firstlineno__", "__static_attributes__"}
            firstline = namespace.get("__firstlineno__")
            static = namespace.get("__static_attributes__")
            if type(firstline) is not int or type(static) is not tuple or static:
                raise ValueError(
                    "native helper compiler metadata needs an exact adapter"
                )
        bases, mro = (
            vars(type)[field].__get__(cls) for field in ("__bases__", "__mro__")
        )
        symbols = tuple(
            (field, vars(type)[field].__get__(cls))
            for field in ("__name__", "__qualname__")
        )
        if (
            any(type(value) is not str for _, value in symbols)
            or not _NativeCaptureGraph._schema(namespace, expected)
            or len(bases) != 1
            or bases[0] is not object
            or len(mro) != 2
            or mro[0] is not cls
            or mro[1] is not object
            or any(value != class_name for _, value in symbols)
            or type(namespace["__module__"]) is not str
            or namespace["__module__"]
            != cast(dict[str, Any], _native_module_dictionary(owner_module))["__name__"]
            or namespace["__doc__"] is not None
        ):
            raise ValueError("native helper class schema needs an exact adapter")
        for descriptor_name in ("__dict__", "__weakref__"):
            descriptor = namespace[descriptor_name]
            if (
                type(descriptor) is not types.GetSetDescriptorType
                or descriptor.__objclass__ is not cls
                or descriptor.__name__ != descriptor_name
            ):
                raise ValueError("native helper needs original own builtin descriptors")
        dictionary = namespace["__dict__"].__get__(instance, cls)
        if type(dictionary) is not dict or dictionary:
            raise ValueError("native helper singleton requires an empty dictionary")
        if not keyword:
            representation = namespace["__repr__"]
            if (
                type(representation) is not types.FunctionType
                or representation.__globals__ is not function.__globals__
                or _native_wrapper_access_schema(representation.__code__)
            ):
                raise ValueError(
                    "native helper repr needs its exact global-free envelope"
                )
        parent = cast(dict[str, Any], _native_module_dictionary(module)).get(
            "debug_info" if keyword else "format_exception_only"
        )
        if type(parent) is not types.FunctionType or (
            not keyword and parent.__globals__ is not _native_module_dictionary(module)
        ):
            raise ValueError(
                "native helper selected parent requires original function/globals"
            )
        roles[name] = {
            "role": role,
            "function": function,
            "code": function.__code__,
            "globals": function.__globals__,
            "container": container,
            "keyword": keyword,
            "instance": instance,
            "class": cls,
            "dictionary": dictionary,
            "members": tuple(namespace.items()),
            "bases": bases,
            "mro": mro,
            "symbols": symbols,
            "edges": edges,
            "module_names": tuple(
                (owner, namespace, namespace["__name__"])
                for owner, namespace, _, _ in edges
            ),
            "parent": parent,
            "parent_code": parent.__code__,
            "parent_globals": parent.__globals__,
        }
    return roles


try:
    _NATIVE_HELPER_DEFAULT_ROLES = _initial_native_helper_default_roles()
except (AttributeError, KeyError, TypeError, ValueError) as _helper_default_error:
    _NATIVE_HELPER_DEFAULT_ROLES = {}
    _NATIVE_HELPER_DEFAULT_GAP: str | None = str(_helper_default_error)
else:
    _NATIVE_HELPER_DEFAULT_GAP = None


class _NativeHelperDefaultCaptureGraph(_NativeCaptureGraph):
    """Route two exact selected helper-default edges through the existing graph."""

    def __init__(self, role: str) -> None:
        super().__init__()
        self.role = role
        self._active_helper: str | None = None

    def _helper_entry(self, name: str) -> dict[str, Any]:
        if _NATIVE_HELPER_DEFAULT_GAP is not None:
            raise ValueError(_NATIVE_HELPER_DEFAULT_GAP)
        entry: dict[str, Any] = _NATIVE_HELPER_DEFAULT_ROLES[name]
        namespace: dict[str, Any]
        original_name: str
        if self.role != entry["role"]:
            raise ValueError("native helper default used outside its selected role")
        function, parent = entry["function"], entry["parent"]
        for value, code, namespace in (
            (function, entry["code"], entry["globals"]),
            (parent, entry["parent_code"], entry["parent_globals"]),
        ):
            if value.__code__ is not code or value.__globals__ is not namespace:
                raise ValueError("native helper original code/globals changed")
            for item in (value, code, namespace):
                self.retain(item)
        container = (
            function.__kwdefaults__ if entry["keyword"] else function.__defaults__
        )
        if container is not entry["container"]:
            raise ValueError("native helper original default container changed")
        for owner, namespace, original_name in entry["module_names"]:
            if (
                _native_module_dictionary(owner) is not namespace
                or namespace.get("__name__") is not original_name
                or sys.modules.get(original_name) is not owner
            ):
                raise ValueError(
                    "native helper original defining module binding changed"
                )
            self.retain(original_name)
        for owner, namespace, slot, original in entry["edges"]:
            if (
                _native_module_dictionary(owner) is not namespace
                or namespace.get(slot) is not original
            ):
                raise ValueError("native helper original module/default alias changed")
            for item in (owner, namespace, original):
                self.retain(item)
            self.bindings.append((owner, slot, original))
        return entry

    def take_selected_helper(self, function: Any, name: str) -> Any:
        entry = self._helper_entry(name)
        if function is not entry["function"]:
            raise ValueError("native helper selected function identity changed")
        previous = self._active_helper
        self._active_helper = name
        try:
            return self.take(function)
        finally:
            self._active_helper = previous

    def _function_defaults(self, function: types.FunctionType, keyword: bool) -> Any:
        name = self._active_helper
        if name is None:
            return super()._function_defaults(function, keyword)
        entry = self._helper_entry(name)
        if function is not entry["function"] or keyword is not entry["keyword"]:
            return super()._function_defaults(function, keyword)
        container, instance = entry["container"], entry["instance"]
        if keyword:
            if (
                not self._schema(
                    container,
                    frozenset(
                        (
                            "result_paths_thunk",
                            "static_argnums",
                            "static_argnames",
                            "sourceinfo",
                            "signature",
                        )
                    ),
                )
                or container["result_paths_thunk"] is not instance
            ):
                raise ValueError("native helper exact keyword-default edge changed")
        elif len(container) != 1 or container[0] is not instance:
            raise ValueError("native helper exact positional-default edge changed")
        # Only these original containers get a single contextual leaf; every
        # other key/value delegates to the existing native envelope walker.
        self.retain(container)
        key = str(id(container))
        if key in self.nodes:
            raise ValueError(
                "native helper default container occurred outside its exact edge"
            )
        node: dict[str, Any] = {"type": self.typename(container)}
        self.nodes[key] = node
        if keyword:
            node["items"] = sorted(
                [
                    [
                        self.take(k),
                        (
                            self._helper_instance(entry)
                            if k == "result_paths_thunk"
                            else self.take(v)
                        ),
                    ]
                    for k, v in container.items()
                ],
                key=lambda pair: json.dumps(pair[0], sort_keys=True),
            )
        else:
            node["values"] = [self._helper_instance(entry)]
        return {"ref": key}

    def _helper_instance(self, entry: dict[str, Any]) -> Any:
        cls, instance = entry["class"], entry["instance"]
        if type(instance) is not cls or type(cls) is not type:
            raise ValueError("native helper original singleton class changed")
        namespace = vars(type)["__dict__"].__get__(cls)
        if not self._schema(namespace, frozenset(name for name, _ in entry["members"])):
            raise ValueError("native helper complete class member schema changed")
        for name, original in entry["members"]:
            if namespace[name] is not original:
                raise ValueError("native helper original class member changed")
        bases, mro = (
            vars(type)[field].__get__(cls) for field in ("__bases__", "__mro__")
        )
        if (
            len(bases) != 1
            or bases[0] is not object
            or len(mro) != 2
            or mro[0] is not cls
            or mro[1] is not object
        ):
            raise ValueError("native helper class bases/MRO changed")
        symbols = {}
        for name, original in entry["symbols"]:
            current = vars(type)[name].__get__(cls)
            if type(current) is not str or current is not original:
                raise ValueError("native helper class name/qualname changed")
            self.retain(current)
            symbols[name] = {"identity": id(current), "value": current}
        descriptor = namespace["__dict__"]
        if (
            type(descriptor) is not types.GetSetDescriptorType
            or descriptor.__objclass__ is not cls
        ):
            raise ValueError("native helper original dictionary descriptor changed")
        dictionary = descriptor.__get__(instance, cls)
        if (
            dictionary is not entry["dictionary"]
            or type(dictionary) is not dict
            or dictionary
        ):
            raise ValueError("native helper original empty instance dictionary changed")
        for item in (instance, cls, type(cls), bases, mro, descriptor, dictionary):
            self.retain(item)
        members = {name: self.take(item) for name, item in sorted(namespace.items())}
        return {
            "helper_default_instance_identity": id(instance),
            "class_identity": id(cls),
            "metaclass_identity": id(type(cls)),
            "bases": [id(value) for value in bases],
            "mro": [id(value) for value in mro],
            "symbols": symbols,
            "members": members,
            "dictionary": self.take(dictionary),
        }


def _native_wrapper_implementation(
    role: str,
) -> tuple[dict[str, Any], tuple[object, ...], tuple[Any, ...]]:
    graph = _NativeHelperDefaultCaptureGraph(role)
    builtin_state = _native_wrapper_builtin_state(role, graph)
    edges: list[Any] = []
    classes = {}
    for module, name in _NATIVE_WRAPPER_CLASS_ROLES[role]:
        selected = _native_wrapper_slot(module, name)
        if selected[-1][-1] is not _NATIVE_WRAPPER_CLASSES[role, name]:
            raise ValueError("native transformed class binding changed: " + name)
        edges.extend(selected)
        classes[name] = _native_wrapper_class_state(graph, role, name)
    roots: dict[str, Any] = {}
    module = (
        _native_traceback if role == "reraise_with_filtered_traceback" else _native_api
    )
    paths: tuple[str, ...] = (
        _NATIVE_WRAPPER_TRACE_HELPERS
        if role == "reraise_with_filtered_traceback"
        else _NATIVE_WRAPPER_VMAP_PATHS
    )
    for path in paths:
        selected = _native_wrapper_slot(module, path)
        edges.extend(selected)
        function = selected[-1][-1]
        if path == "core.no_axis_name":
            if function is not _NATIVE_WRAPPER_SENTINEL:
                raise ValueError("native vmap no-axis defining binding changed")
            roots[path] = {"contextual_no_axis_name_identity": id(function)}
        else:
            roots[path] = (
                {"imported_stdlib_class_identity": id(function)}
                if path == "collections.Counter"
                else (
                    graph.take_selected_helper(function, "debug_info")
                    if role == "vmap_f" and path == "debug_info"
                    else graph.take(function)
                )
            )
        graph.retain(function)
        if type(function) is types.FunctionType:
            graph.retain(function.__globals__)
            roots[path + ":globals"] = id(function.__globals__)
    if role == "reraise_with_filtered_traceback":
        for path in _NATIVE_WRAPPER_TRACE_STDLIB_PATHS:
            selected = _native_wrapper_slot(module, path)
            edges.extend(selected)
            # The native stdlib executable boundary owns these explicit slots.
            # Runtime class permissions are not inferred from them.
            imported = selected[-1][-1]
            roots[path] = (
                (
                    graph.take_selected_helper(imported, "format_exception_only")
                    if path == "traceback.format_exception_only"
                    else graph.take(imported)
                )
                if type(imported) is types.FunctionType
                else {"imported_executable_identity": id(imported)}
            )
            graph.retain(selected[-1][-1])
        for path in ("_jax_message_append", "_simplified_tb_msg", "__file__"):
            selected = _native_wrapper_slot(module, path)
            edges.extend(selected)
            if type(selected[-1][-1]) is not str:
                raise ValueError(
                    "native traceback message/filename requires actual strings"
                )
            roots[path] = graph.take(selected[-1][-1])
    else:
        partial = vars(_native_batching)["flatten_fun_for_vmap"]
        if not (
            type(partial) is functools.partial
            and type(partial.func) is types.FunctionType
            and len(partial.args) == 1
            and type(partial.args[0]) is types.FunctionType
        ):
            raise ValueError(
                "native vmap flatten partial needs its exact constructed graph"
            )
        for owner, paths in (
            (partial.args[0], ("tree_flatten", "tree_unflatten", "is_vmappable")),
            (partial.func, ("Store", "EqualStore")),
        ):
            namespace = owner.__globals__
            graph.retain(namespace)
            for path in paths:
                expected_module = (
                    _native_batching
                    if owner is partial.args[0]
                    else _native_linear_util
                )
                selected = _native_wrapper_slot(expected_module, path)
                if (
                    namespace is not _native_module_dictionary(expected_module)
                    or namespace.get(path) is not selected[-1][-1]
                ):
                    raise ValueError("native flatten partial defining globals changed")
                edges.extend(selected)
        selected = _native_wrapper_slot(_native_batching, "infer")
        edges.extend(selected)
        infer = selected[-1][-1]
        if (
            type(infer) is not _NATIVE_WRAPPER_INFER_TYPE
            or type(_NATIVE_WRAPPER_INFER_DICT) is not types.GetSetDescriptorType
        ):
            raise ValueError("native vmap infer sentinel type changed")
        if vars(_NATIVE_WRAPPER_INFER_TYPE).get(
            "__dict__"
        ) is not _NATIVE_WRAPPER_INFER_DICT or _NATIVE_WRAPPER_INFER_DICT.__get__(
            infer, _NATIVE_WRAPPER_INFER_TYPE
        ):
            raise ValueError("native vmap infer sentinel layout changed")
        graph.retain(infer)
        roots["infer"] = {"identity": id(infer), "type_identity": id(type(infer))}
    for owner, namespace, _, value in edges:
        for item in (owner, namespace, value):
            graph.retain(item)
    return (
        {
            "roots": roots,
            "classes": classes,
            "nodes": graph.nodes,
            "direct_builtin_lookups": builtin_state,
        },
        tuple(graph.retained.values()),
        tuple(edges) + tuple(graph.bindings),
    )


_NATIVE_WRAPPER_IMPLEMENTATIONS: dict[str, Any] = {}
_NATIVE_WRAPPER_IMPLEMENTATION_GAPS: dict[str, str] = {}
for _wrapper_role in _NATIVE_WRAPPER_ROLES:
    try:
        _NATIVE_WRAPPER_IMPLEMENTATIONS[_wrapper_role] = _native_wrapper_implementation(
            _wrapper_role
        )
    except (TypeError, ValueError) as _wrapper_error:
        _NATIVE_WRAPPER_IMPLEMENTATION_GAPS[_wrapper_role] = str(_wrapper_error)


@dataclass(frozen=True)
class NativeTransformedSnapshot:
    document: dict[str, Any]
    retained: tuple[object, ...]


def native_transformed_function_role(value: object) -> str | None:
    """Select only live native nested-code and defining-globals identities."""
    if type(value) is not types.FunctionType:
        return None
    for role, (code, namespace) in _NATIVE_WRAPPER_ROLES.items():
        if value.__code__ is code and value.__globals__ is namespace:
            if _NATIVE_WRAPPER_LOOKUP_GAP is not None:
                raise ValueError(_NATIVE_WRAPPER_LOOKUP_GAP)
            if value.__builtins__ is not _NATIVE_WRAPPER_LOOKUP_ROLES[role]["builtins"]:
                raise ValueError("native transformed actual builtin mapping changed")
            if value.__code__.co_freevars != _NATIVE_WRAPPER_FREEVARS[role]:
                raise ValueError("native transformed closure schema changed")
            return role
    return None


def native_transformed_closure_state(
    function: object, name: str, value: object
) -> dict[str, Any] | None:
    """The native no-axis sentinel is permitted only at this selected edge."""
    if native_transformed_function_role(function) != "vmap_f" or name != "axis_name":
        return None
    if vars(_native_core).get("no_axis_name") is not _NATIVE_WRAPPER_SENTINEL:
        raise ValueError("native vmap no-axis defining binding changed")
    if value is _NATIVE_WRAPPER_SENTINEL:
        return {"native_role": "vmap_f.axis_name:no_axis_name", "identity": id(value)}
    return None


def _native_wrapper_dynamic_state(role: str, graph: _NativeCaptureGraph) -> Any:
    if _NATIVE_WRAPPER_CONTEXT_GAP is not None:
        raise ValueError(_NATIVE_WRAPPER_CONTEXT_GAP)
    context = _NATIVE_WRAPPER_CONTEXT_ROLES
    for owner, cls, mro in context["configuration_types"]:
        if type(owner) is not cls or vars(type)["__mro__"].__get__(cls) is not mro:
            raise ValueError("native transformed configuration owner type changed")
    for cls, original in context["configuration_class_members"]:
        namespace = vars(type)["__dict__"].__get__(cls)
        if len(namespace) != len(original) or any(
            namespace.get(name) is not value for name, value in original
        ):
            raise ValueError("native transformed configuration class accessors changed")
    config_names = (
        ("traceback_filtering",)
        if role == "reraise_with_filtered_traceback"
        else ("mutable_array_checks", "remove_size_one_mesh_axis_from_type")
    )
    states: dict[str, Any] = {}
    for name in config_names:
        owner = vars(_native_config).get(name)
        if owner is not _NATIVE_WRAPPER_CONFIGS[name]:
            raise ValueError("native transformed configuration owner changed: " + name)
        local, effective = owner.get_local(), owner.value
        kind = str if name == "traceback_filtering" else bool
        if type(effective) is not kind or (
            local is not _native_xla_metadata.config_ext.unset
            and type(local) is not kind
        ):
            raise ValueError(
                "native transformed configuration value layout changed: " + name
            )
        graph.retain(owner)
        graph.retain(local)
        graph.retain(effective)
        states[name] = {
            "owner": id(owner),
            "local": (
                {"unset": True}
                if local is _native_xla_metadata.config_ext.unset
                else graph.take(local)
            ),
            "effective": graph.take(effective),
        }
    if role == "reraise_with_filtered_traceback":
        paths = vars(_native_traceback).get("_exclude_paths")
        if (
            paths is not _NATIVE_WRAPPER_EXCLUSIONS
            or type(paths) is not list
            or any(type(path) is not str for path in paths)
        ):
            raise ValueError("native traceback exclusion list binding/layout changed")
        states["exclude_paths"] = {"identity": id(paths), "contents": graph.take(paths)}
        for owner_name, namespace in (
            ("globals", vars(_native_traceback)),
            ("builtins", vars(builtins)),
        ):
            present = "get_ipython" in namespace
            value = namespace.get("get_ipython")
            graph.retain(value)
            states["get_ipython:" + owner_name] = {
                "present": present,
                "identity": id(value) if present else None,
            }
        return states
    registry = vars(_native_batching).get("vmappables")
    if registry is not _NATIVE_WRAPPER_REGISTRY or type(registry) is not dict:
        raise ValueError("native vmappables registry binding changed")
    graph.retain(registry)
    entries = []
    for key, pair in registry.items():
        if type(pair) is not tuple or len(pair) != 2:
            raise ValueError("native vmappables entry layout needs an exact adapter")
        for value in (key, pair, *pair):
            graph.retain(value)
        entries.append([id(key), id(pair), [id(value) for value in pair]])
    states["vmappables"] = {"identity": id(registry), "entries": sorted(entries)}
    if (
        vars(_native_core).get("JaxprEqnContext") is not context["eqn_type"]
        or vars(_native_mesh).get("AbstractMesh") is not context["mesh_type"]
    ):
        raise ValueError("native transformed context type binding changed")
    current_configurations = _native_context_configurations()
    if len(current_configurations) != len(context["context_configs"]) or any(
        name != original_name or owner is not original_owner
        for (name, owner), (original_name, original_owner) in zip(
            current_configurations, context["context_configs"], strict=True
        )
    ):
        raise ValueError("native transformed context configuration owner changed")
    before = native_context_snapshot()
    graph.retain(before.context)
    states["equation_context"] = {
        "identity": id(before.context),
        "fields": (
            None if before.context is None else _native_context_fields(before.context)
        ),
    }
    states["context_configurations"] = []
    for name, owner, local, effective, local_fields, effective_fields in before.states:
        for value in (owner, local, effective):
            graph.retain(value)
        states["context_configurations"].append(
            [name, id(owner), id(local), id(effective), local_fields, effective_fields]
        )
    if vars(_native_core).get("axis_env_state") is not context["axis_config"]:
        raise ValueError(
            "native transformed axis environment configuration binding changed"
        )
    axis_local, axis = context["axis_config"].get_local(), context["axis_config"].value
    for value in (context["axis_config"], axis_local, axis):
        graph.retain(value)
    states["axis_configuration"] = {
        "owner": id(context["axis_config"]),
        "local": id(axis_local),
        "effective": id(axis),
    }
    if (
        vars(_native_core).get("AxisEnv") is not context["axis_type"]
        or type(axis) is not context["axis_type"]
    ):
        raise ValueError("native axis environment needs its exact native type")
    graph.retain(axis)
    axis_fields = {}
    for name, descriptor in context["axis_slots"]:
        if (
            type(descriptor) is not types.MemberDescriptorType
            or vars(context["axis_type"]).get(name) is not descriptor
        ):
            raise ValueError("native axis environment slot schema changed")
        value = descriptor.__get__(axis, type(axis))
        graph.retain(value)
        if name == "axis_sizes":
            mapping_base, mapping_descriptor = context["axis_mapping_dict"]
            if (
                type(value) is not context["axis_mapping_type"]
                or vars(mapping_base).get("__dict__") is not mapping_descriptor
            ):
                raise ValueError("native axis size mapping type/descriptor changed")
            namespace = mapping_descriptor.__get__(value, context["axis_mapping_type"])
            if (
                type(namespace) is not dict
                or any(type(key) is not str for key in namespace)
                or set(namespace) != {"_d"}
                or type(namespace["_d"]) is not dict
            ):
                raise ValueError("native axis size mapping needs an exact adapter")
            graph.retain(namespace)
            axis_fields[name] = graph.take(namespace["_d"])
        else:
            axis_fields[name] = graph.take(value)
    states["axis_environment"] = {"identity": id(axis), "fields": axis_fields}
    mesh = next(
        effective for name, _, _, effective, _, _ in before.states if name == "mesh"
    )
    if mesh is None:
        if vars(_native_mesh).get("empty_abstract_mesh") is not context["empty_mesh"]:
            raise ValueError("native transformed empty mesh defining binding changed")
        mesh = context["empty_mesh"]
    mesh_base, mesh_descriptor = context["mesh_dict"]
    if (
        type(mesh) is not context["mesh_type"]
        or vars(mesh_base).get("__dict__") is not mesh_descriptor
    ):
        raise ValueError("native transformed abstract mesh type/descriptor changed")
    mesh_fields = mesh_descriptor.__get__(mesh, context["mesh_type"])
    required = ("axis_sizes", "axis_names", "axis_types", "size", "abstract_device")
    if (
        type(mesh_fields) is not dict
        or any(type(key) is not str for key in mesh_fields)
        or any(name not in mesh_fields for name in required)
    ):
        raise ValueError(
            "native transformed abstract mesh field layout needs an exact adapter"
        )
    if (
        any(
            type(mesh_fields[name]) is not tuple or mesh_fields[name]
            for name in required[:3]
        )
        or type(mesh_fields["size"]) is not int
        or mesh_fields["size"] != 0
        or mesh_fields["abstract_device"] is not None
    ):
        raise ValueError(
            "native transformed nonempty/device mesh needs an exact adapter"
        )
    for value in (mesh, mesh_fields, *(mesh_fields[name] for name in required)):
        graph.retain(value)
    states["abstract_mesh"] = {
        "identity": id(mesh),
        "fields": {name: graph.take(mesh_fields[name]) for name in required},
    }
    return states


class NativeTransformedExecutionError(RuntimeError):
    """Replay execution could not preserve its exact native caller context."""


@contextmanager
def native_transformed_execution_context(
    *, on_error: Callable[[NativeTransformedExecutionError], None] | None = None
) -> Iterator[None]:
    """Manage native axis-local materialization without losing caller state."""

    def snapshot() -> tuple[dict[str, Any], _NativeCaptureGraph]:
        graph = _NativeCaptureGraph()
        state = _native_wrapper_dynamic_state("vmap_f", graph)
        return {"runtime_state": state, "nodes": graph.nodes}, graph

    def report(detail: str) -> NativeTransformedExecutionError:
        error = NativeTransformedExecutionError(detail)
        if on_error is not None:
            try:
                on_error(error)
            except Exception as exc:
                raise error from exc
        return error

    try:
        caller, _caller_graph = snapshot()
        owner = _NATIVE_WRAPPER_CONTEXT_ROLES["axis_config"]
        # Retain the already validated native methods across the source call.
        # Cleanup never follows a method replacement introduced by the source.
        read_local, set_local = owner.get_local, owner.set_local
        local, effective = read_local(), owner.value
    except Exception as exc:
        raise report(
            "native transformed execution context entry failed: " + str(exc)
        ) from exc
    try:
        set_local(effective)
        managed, _managed_graph = snapshot()
    except Exception as exc:
        detail = "native transformed execution context installation failed: " + str(exc)
        try:
            set_local(local)
            restored, _restored_graph = snapshot()
            if read_local() is not local or restored != caller:
                detail += "; exact caller state was not restored"
        except Exception as restore_error:
            detail += "; caller restoration failed: " + str(restore_error)
        raise report(detail) from exc
    try:
        yield
    finally:
        problems = []
        try:
            observed, _observed_graph = snapshot()
            if observed != managed:
                problems.append("managed native state changed during execution")
        except Exception as exc:
            problems.append("managed native state could not be checked: " + str(exc))
        try:
            set_local(local)
            restored, _restored_graph = snapshot()
            if read_local() is not local or restored != caller:
                problems.append(
                    "exact native caller local/effective state was not restored"
                )
        except Exception as exc:
            problems.append("native caller context restoration failed: " + str(exc))
        if problems:
            error = report("; ".join(problems))
            if on_error is None:
                raise error


def selected_native_transformed_function(
    value: object,
) -> NativeTransformedSnapshot | None:
    """Capture finite implementation globals while raw traversal owns user edges."""
    role = native_transformed_function_role(value)
    if role is None:
        return None
    if _NATIVE_WRAPPER_LAYOUT_GAP is not None:
        raise ValueError(_NATIVE_WRAPPER_LAYOUT_GAP)
    for (owner, code, namespace, closure, contents), actual_builtins in zip(
        _NATIVE_WRAPPER_FACTORIES, _NATIVE_WRAPPER_FACTORY_BUILTINS, strict=True
    ):
        if (
            owner.__code__ is not code
            or owner.__globals__ is not namespace
            or owner.__builtins__ is not actual_builtins
            or owner.__closure__ is not closure
        ):
            raise ValueError("native transformed original factory envelope changed")
        if any(
            cell.cell_contents is not original
            for cell, original in zip(closure or (), contents, strict=True)
        ):
            raise ValueError("native transformed original factory closure changed")
    if (
        vars(_native_traceback).get("api_boundary")
        is not _NATIVE_WRAPPER_FACTORIES[0][0]
        or vars(_native_api).get("vmap") is not _NATIVE_WRAPPER_FACTORIES[1][0]
    ):
        raise ValueError("native transformed factory binding changed")
    if role in _NATIVE_WRAPPER_IMPLEMENTATION_GAPS:
        raise ValueError(_NATIVE_WRAPPER_IMPLEMENTATION_GAPS[role])
    baseline, retained, bindings = _NATIVE_WRAPPER_IMPLEMENTATIONS[role]
    for binding in bindings:
        if len(binding) == 4:
            owner, namespace, name, original = binding
            if (
                _native_module_dictionary(owner) is not namespace
                or cast(dict[str, Any], namespace).get(name) is not original
            ):
                raise ValueError("native transformed implementation slot changed")
        else:
            owner, name, original = binding
            namespace = _native_module_dictionary(owner)
            if namespace is None or namespace.get(name) is not original:
                raise ValueError("native transformed executable binding changed")
    observed, current_references, _ = _native_wrapper_implementation(role)
    if observed != baseline:
        raise ValueError(
            "native transformed implementation executable/class state changed"
        )
    graph = _NativeCaptureGraph()
    dynamic = _native_wrapper_dynamic_state(role, graph)
    return NativeTransformedSnapshot(
        {
            "role": role,
            "code_identity": id(cast(types.FunctionType, value).__code__),
            "globals_identity": id(cast(types.FunctionType, value).__globals__),
            "implementation": observed,
            "runtime_state": dynamic,
            "runtime_nodes": graph.nodes,
        },
        (*retained, *current_references, *graph.retained.values()),
    )


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


@contextmanager
def preserve_caller_configuration() -> Iterator[None]:
    """Restore four caller settings without hiding updates during the call.

    Retain native global and local/unset separately. In particular, installing
    a local snapshot at entry would mask a model's legitimate global update.
    """
    states = (
        _native_config.enable_x64,
        _native_config.default_matmul_precision,
        _native_config.numpy_dtype_promotion,
        _native_config.numpy_rank_promotion,
    )
    saved = tuple((state, state.get_global(), state.get_local()) for state in states)
    unset = _native_config.config_ext.unset
    body_error: BaseException | None = None
    try:
        yield
    except BaseException as error:
        body_error = error
        raise
    finally:
        errors: list[BaseException] = []
        for state, global_value, local_value in saved:
            # State._set performs native set_global and its native global hook.
            try:
                state._set(global_value)
            except BaseException as error:
                errors.append(error)
            # Match native StateContextManager exit, including exact unset.
            try:
                state.set_local(local_value)
            except BaseException as error:
                errors.append(error)
            try:
                hook = state._update_thread_local_hook
                if hook is not None:
                    hook(None if local_value is unset else local_value)
            except BaseException as error:
                errors.append(error)
        if errors:
            if body_error is not None:
                errors.insert(0, body_error)
            raise builtins.BaseExceptionGroup(
                "Caller configuration restoration failed", errors
            ) from None
