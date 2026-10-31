# jax2onnx/plugins/flax/nnx/_source_callback.py

"""Bounded snapshots of ordinary Python callbacks in private SOURCE NNX state.

Only function-owned memory is isolated. Canonical module globals and builtins
remain the source runtime's globals; this is not a sandbox for global effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import marshal
import sys
from types import CellType, FunctionType, ModuleType
from typing import Any, cast

from jax2onnx._compat.jax import Array
from jax2onnx._compat.python_annotations import (
    AnnotationStateError,
    install_isolated_annotations,
    isolated_annotation_parts,
)
from jax2onnx.converter.source_profile import SourcePrecisionError


def _failure(path: str, message: str) -> SourcePrecisionError:
    return SourcePrecisionError(
        f"NNX callback isolation support remains unfinished: {message}",
        phase="profile-validation",
        category="lowering-defect",
        location=path,
        requirement="ordinary Python callbacks with explicitly isolated immutable payloads",
        action="use a callback with supported immutable function-owned memory",
    )


@dataclass(frozen=True)
class CallbackToken:
    index: int


@dataclass(frozen=True)
class _Anchor:
    identity: int
    value: Any = field(compare=False, hash=False, repr=False)

    @classmethod
    def of(cls, value: Any) -> _Anchor:
        return cls(id(value), value)


@dataclass(frozen=True)
class _Value:
    kind: str
    payload: Any


@dataclass(frozen=True)
class _ManagedRole:
    plugin: _Anchor
    primitive: _Anchor
    source_code: _Anchor


@dataclass(frozen=True)
class _Function:
    code: _Anchor
    globals: _Anchor
    builtins: _Anchor
    name: str
    qualname: str
    module: str | None
    doc: str | None
    defaults: _Value
    kwdefaults: int | None
    attributes: int
    closure: tuple[int, ...] | None
    annotate: int | None
    annotations: int | None
    managed: _ManagedRole | None


@dataclass(frozen=True)
class _Watch:
    function: FunctionType = field(compare=False, hash=False, repr=False)
    identities: tuple[Any, ...] = field(compare=False, hash=False, repr=False)
    path: str = field(compare=False, hash=False)


_EMPTY_CELL: object = object()


def _cell_contents(cell: CellType) -> object:
    try:
        return cell.cell_contents
    except ValueError:
        return _EMPTY_CELL


def _identity_fields(function: FunctionType) -> tuple[Any, ...]:
    annotate, cached = isolated_annotation_parts(function)
    return (
        function.__code__,
        function.__globals__,
        function.__builtins__,
        function.__defaults__,
        function.__kwdefaults__,
        function.__dict__,
        function.__closure__,
        getattr(function, "__annotate__", None),
        annotate,
        cached,
        *(_cell_contents(cell) for cell in function.__closure__ or ()),
        *(
            item
            for mapping in (function.__dict__, function.__kwdefaults__, cached)
            if mapping is not None
            for pair in dict.items(mapping)
            for item in pair
        ),
    )


def _canonical_globals(function: FunctionType, path: str) -> None:
    namespace = function.__globals__
    name = namespace.get("__name__")
    module = sys.modules.get(name) if type(name) is str else None
    if type(module) is not ModuleType or vars(module) is not namespace:
        raise _failure(path, "callback globals need the canonical loaded module")
    builtins = namespace.get("__builtins__")
    if type(builtins) is ModuleType:
        builtins = vars(builtins)
    if type(builtins) is not dict or function.__builtins__ is not builtins:
        raise _failure(path, "callback builtins changed from its module bindings")


def _canonical_annotation(value: Any) -> bool:
    # Official pybind Array.__name__ can be fully qualified while its public
    # module is jax. Retain the exact native API object across supported cohorts.
    if value is Array:
        return True
    # Native type declarations are trusted module anchors, like their modules.
    # The type object is not cloned and no metaclass getter is evaluated.
    if not any(base is type for base in type.__getattribute__(type(value), "__mro__")):
        return False
    module_name = type.__getattribute__(value, "__module__")
    name = type.__getattribute__(value, "__name__")
    module = sys.modules.get(module_name) if type(module_name) is str else None
    return type(module) is ModuleType and vars(module).get(name) is value


class CallbackCapture:
    """One local memo for all callback roots in a single NNX descriptor."""

    def __init__(self, *, expected: CallbackGraph | None = None) -> None:
        self._retained: tuple[tuple[object, ...], ...] = (
            ()
            if expected is None
            else tuple(
                (
                    spec.code.value,
                    spec.globals.value,
                    spec.managed.source_code.value,
                    spec.managed.primitive.value,
                    spec.managed.plugin.value,
                )
                for spec in expected.functions
                if spec.managed is not None
            )
        )
        self._functions: list[_Function | None] = []
        self._function_ids: dict[int, int] = {}
        self._cells: list[_Value | None] = []
        self._cell_ids: dict[int, int] = {}
        self._mappings: list[tuple[tuple[str, _Value], ...] | None] = []
        self._mapping_ids: dict[int, int] = {}
        self._watches: list[_Watch] = []
        self._roots: list[CallbackToken] = []

    def _value(
        self,
        value: Any,
        path: str,
        *,
        annotation: bool = False,
        allow_functions: bool = True,
    ) -> _Value:
        if type(value) in {type(None), bool, int, float, complex, str, bytes}:
            return _Value("literal", marshal.dumps(value))
        if type(value) in {tuple, frozenset}:
            values = tuple(
                self._value(
                    item,
                    path,
                    annotation=annotation,
                    allow_functions=allow_functions and type(value) is not frozenset,
                )
                for item in value
            )
            return _Value(
                "tuple" if type(value) is tuple else "frozenset",
                values if type(value) is tuple else frozenset(values),
            )
        if type(value) is FunctionType:
            if not allow_functions:
                raise _failure(
                    path, "function-valued frozensets need an unordered graph adapter"
                )
            return _Value("function", self._function(value, path))
        if annotation and _canonical_annotation(value):
            return _Value("anchor", _Anchor.of(value))
        raise _failure(path, "mutable or opaque closure/default/metadata payload")

    def _mapping(
        self, value: dict[str, Any], path: str, *, annotation: bool = False
    ) -> int:
        if type(value) is not dict or any(type(key) is not str for key in value):
            raise _failure(
                path, "function metadata needs exact dictionaries with string keys"
            )
        identity = id(value)
        if identity in self._mapping_ids:
            return self._mapping_ids[identity]
        index = len(self._mappings)
        self._mapping_ids[identity] = index
        self._mappings.append(None)
        self._mappings[index] = tuple(
            (key, self._value(item, f"{path}.{key}", annotation=annotation))
            for key, item in value.items()
        )
        return index

    def _cell(self, cell: CellType, path: str, special: _Value | None = None) -> int:
        identity = id(cell)
        if identity in self._cell_ids:
            index = self._cell_ids[identity]
            previous = self._cells[index]
            if previous is not None:
                if special is not None:
                    if previous != special:
                        raise _failure(
                            path, "shared closure cell has incompatible roles"
                        )
                else:
                    try:
                        value = cell.cell_contents
                    except ValueError:
                        pass
                    else:
                        self._value(value, path)
            return index
        index = len(self._cells)
        self._cell_ids[identity] = index
        self._cells.append(None)
        if special is not None:
            payload = special
        else:
            try:
                value = cell.cell_contents
            except ValueError:
                payload = _Value("empty-cell", None)
            else:
                payload = self._value(value, path)
        self._cells[index] = payload
        return index

    def _function(self, function: FunctionType, path: str) -> int:
        if type(function) is not FunctionType:
            raise _failure(
                path, "bound methods, partials and custom callables need an adapter"
            )
        identity = id(function)
        if identity in self._function_ids:
            return self._function_ids[identity]
        raw_attributes = function.__dict__
        if type(raw_attributes) is not dict or (
            "__wrapped__" in raw_attributes
            and type(raw_attributes["__wrapped__"]) is not FunctionType
        ):
            raise _failure(path, "wrapped metadata needs an ordinary function edge")
        _canonical_globals(function, path)
        type_params = getattr(function, "__type_params__", ())
        if type(type_params) is not tuple or tuple.__len__(type_params):
            raise _failure(path, "generic function metadata needs an adapter")
        if any(
            type(value) is not str
            for value in (function.__name__, function.__qualname__)
        ) or any(
            value is not None and type(value) is not str
            for value in (function.__module__, function.__doc__)
        ):
            raise _failure(path, "function header metadata needs literal strings")
        index = len(self._functions)
        self._function_ids[identity] = index
        self._functions.append(None)
        try:
            annotate, annotations = isolated_annotation_parts(function)
            watched = _Watch(function, _identity_fields(function), path)
        except AnnotationStateError as exc:
            raise _failure(path, str(exc)) from exc
        self._watches.append(watched)
        from jax2onnx.plugins.plugin_system import _source_callback_wrapper_bindings

        managed = _source_callback_wrapper_bindings(function)
        if managed is None:
            for retained in self._retained:
                managed = _source_callback_wrapper_bindings(function, retained=retained)
                if managed is not None:
                    break
        roles = dict(managed) if managed is not None else {}
        managed_role = (
            None
            if managed is None
            else _ManagedRole(
                _Anchor.of(roles["self"]),
                _Anchor.of(roles["primitive"]),
                _Anchor.of(roles["original_call"].__code__),
            )
        )
        closure = function.__closure__
        closure_ids = None
        if closure is not None:
            ids = []
            for name, cell in zip(function.__code__.co_freevars, closure, strict=True):
                special = None
                if name in roles:
                    if name in {"self", "primitive"}:
                        special = _Value("anchor", _Anchor.of(roles[name]))
                    elif name == "params":
                        special = _Value("managed-params", tuple(roles[name]))
                ids.append(self._cell(cell, f"{path}.closure.{name}", special))
            closure_ids = tuple(ids)
        attribute_mapping = self._mapping(function.__dict__, f"{path}.attributes")
        kwdefaults = (
            None
            if function.__kwdefaults__ is None
            else self._mapping(function.__kwdefaults__, f"{path}.kwdefaults")
        )
        annotation_mapping = (
            None
            if annotations is None
            else self._mapping(annotations, f"{path}.annotations", annotation=True)
        )
        self._functions[index] = _Function(
            _Anchor.of(function.__code__),
            _Anchor.of(function.__globals__),
            _Anchor.of(function.__builtins__),
            function.__name__,
            function.__qualname__,
            function.__module__,
            function.__doc__,
            self._value(function.__defaults__, f"{path}.defaults"),
            kwdefaults,
            attribute_mapping,
            closure_ids,
            None if annotate is None else self._function(annotate, f"{path}.annotate"),
            annotation_mapping,
            managed_role,
        )
        return index

    def capture(
        self, function: FunctionType, path: str = "NNX callback"
    ) -> CallbackToken:
        token = CallbackToken(self._function(function, path))
        self._roots.append(token)
        return token

    def finish(self) -> CallbackGraph:
        return CallbackGraph(
            cast(tuple[_Function, ...], tuple(self._functions)),
            cast(tuple[_Value, ...], tuple(self._cells)),
            cast(tuple[tuple[tuple[str, _Value], ...], ...], tuple(self._mappings)),
            tuple(self._roots),
            tuple(self._watches),
        )


@dataclass(frozen=True)
class CallbackGraph:
    functions: tuple[_Function, ...]
    cells: tuple[_Value, ...]
    mappings: tuple[tuple[tuple[str, _Value], ...], ...]
    roots: tuple[CallbackToken, ...]
    watches: tuple[_Watch, ...] = field(compare=False, hash=False, repr=False)

    @property
    def is_empty(self) -> bool:
        return not self.roots

    def assert_current(self) -> None:
        for watch in self.watches:
            try:
                current = _identity_fields(watch.function)
            except AnnotationStateError as exc:
                raise _failure(watch.path, str(exc)) from exc
            if len(current) != len(watch.identities) or any(
                actual is not expected
                for actual, expected in zip(current, watch.identities, strict=True)
            ):
                raise _failure(
                    watch.path, "prepared callback metadata bindings changed"
                )
        capture = CallbackCapture(expected=self)
        for root in self.roots:
            watch = self.watches[root.index]
            capture.capture(watch.function, watch.path)
        if capture.finish() != self:
            raise _failure("NNX callback", "prepared callback memory changed")

    def rebuild(self) -> CallbackRebuild:
        self.assert_current()
        return CallbackRebuild(self)


class CallbackRebuild:
    def __init__(self, graph: CallbackGraph) -> None:
        self._graph = graph
        self._cells = tuple(CellType() for _ in graph.cells)
        self._mappings: tuple[dict[str, Any], ...] = tuple({} for _ in graph.mappings)
        self._functions = tuple(
            FunctionType(
                spec.code.value,
                spec.globals.value,
                spec.name,
                None,
                (
                    None
                    if spec.closure is None
                    else tuple(self._cells[i] for i in spec.closure)
                ),
            )
            for spec in graph.functions
        )
        for cell, cell_spec in zip(self._cells, graph.cells, strict=True):
            if cell_spec.kind != "empty-cell":
                cell.cell_contents = self._value(cell_spec)
        for mapping, entries in zip(self._mappings, graph.mappings, strict=True):
            mapping.update((key, self._value(value)) for key, value in entries)
        for function, function_spec in zip(
            self._functions, graph.functions, strict=True
        ):
            function.__qualname__ = function_spec.qualname
            # Native functions accept None here despite typeshed's str field.
            object.__setattr__(function, "__module__", function_spec.module)
            function.__doc__ = function_spec.doc
            function.__defaults__ = self._value(function_spec.defaults)
            function.__kwdefaults__ = (
                None
                if function_spec.kwdefaults is None
                else self._mappings[function_spec.kwdefaults]
            )
            function.__dict__ = self._mappings[function_spec.attributes]
            install_isolated_annotations(
                function,
                annotate=(
                    None
                    if function_spec.annotate is None
                    else self._functions[function_spec.annotate]
                ),
                cached=(
                    None
                    if function_spec.annotations is None
                    else self._mappings[function_spec.annotations]
                ),
                validate=_validate_materialized_annotations,
            )
            if function.__builtins__ is not function_spec.builtins.value:
                raise _failure(
                    "NNX callback", "module builtins changed during reconstruction"
                )

    def _value(self, value: _Value) -> Any:
        if value.kind == "literal":
            return marshal.loads(value.payload)
        if value.kind == "function":
            return self._functions[value.payload]
        if value.kind == "tuple":
            return tuple(self._value(item) for item in value.payload)
        if value.kind == "frozenset":
            return frozenset(self._value(item) for item in value.payload)
        if value.kind == "managed-params":
            return list(value.payload)
        if value.kind == "anchor":
            return value.payload.value
        raise _failure("NNX callback", "unknown immutable callback token")

    def resolve(self, token: CallbackToken) -> FunctionType:
        if type(token) is not CallbackToken or not 0 <= token.index < len(
            self._functions
        ):
            raise _failure("NNX callback", "malformed callback root token")
        return self._functions[token.index]


def _validate_materialized_annotations(mapping: dict[str, Any]) -> None:
    # Called only by an explicit annotation read on the isolated callback.
    def validate(value: Any) -> None:
        if type(value) in {type(None), bool, int, float, complex, str, bytes}:
            return
        if type(value) in {tuple, frozenset}:
            for item in value:
                validate(item)
            return
        if _canonical_annotation(value):
            return
        raise _failure(
            "NNX isolated annotations",
            "materialized annotations need immutable declarations",
        )

    if type(mapping) is not dict or any(type(key) is not str for key in mapping):
        raise _failure("NNX isolated annotations", "annotation cache needs string keys")
    for value in mapping.values():
        validate(value)
