# jax2onnx/_source_callable_preparation.py

"""Explicit selected dtype preparation of existing native Python callables.

This private mechanism runs before JAX tracing. It freezes named global bindings
in a private namespace and retains other object references and closure cells.
It neither configures JAX nor imports conversion, plugins, ONNX or test support.
Unsupported Python/state forms remain preparation work, not exporter limits.
"""

from __future__ import annotations

import __future__
import ast
from collections.abc import Iterator
from contextlib import contextmanager
import copy
from dataclasses import dataclass, field
import dis
import hashlib
import inspect
import json
import marshal
import textwrap
from types import CodeType, FunctionType, ModuleType, CellType, BuiltinFunctionType
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np

from jax2onnx._source_preparation import SourcePreparationError
from jax2onnx._compat.python_annotations import AnnotationState, AnnotationStateError


Scope = Literal["global", "closure"]
SharedScope = Literal["global", "closure", "default", "kwdefault"]
AstPath = tuple[str | int, ...]
_DTYPES: tuple[tuple[object, str], ...] = (
    (jnp.float16, "float16"),
    (jnp.bfloat16, "bfloat16"),
    (jnp.float32, "float32"),
    (jnp.float64, "float64"),
    (np.float16, "float16"),
    (np.float32, "float32"),
    (np.float64, "float64"),
)
_FUTURE_FLAGS: int = sum(
    getattr(__future__, name).compiler_flag for name in __future__.all_feature_names
)
_MISSING = object()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _code_sha(code: CodeType) -> str:
    return _sha(marshal.dumps(code))


@dataclass(frozen=True)
class BindingRef:
    scope: Scope
    name: str
    expected: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class DtypeSite:
    path: AstPath
    ast_sha256: str
    root: BindingRef
    expected_dtype: object = field(repr=False, compare=False)
    replacement_dtype: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class SharedBinding:
    scope: SharedScope
    name: str | int
    expected: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class HelperBinding:
    reference: BindingRef
    prepared: FunctionType = field(repr=False, compare=False)
    record: PreparationRecord = field(repr=False, compare=False)


@dataclass(frozen=True)
class CallableRecipe:
    original: FunctionType = field(repr=False, compare=False)
    source_sha256: str
    code_sha256: str
    sites: tuple[DtypeSite, ...]
    required_site_count: int
    intent: str
    helpers: tuple[HelperBinding, ...] = ()
    shared: tuple[SharedBinding, ...] = ()


@dataclass(frozen=True)
class CallablePreimage:
    source: str
    source_sha256: str
    code_sha256: str


def _function(value: object) -> FunctionType:
    if type(value) is not FunctionType:
        raise SourcePreparationError(
            "callable preparation requires an existing plain Python function; methods and callable/decorated owners need separate support"
        )
    fn = value
    if "__wrapped__" in fn.__dict__:
        raise SourcePreparationError(
            "wrapped function source correspondence needs separate support"
        )
    type_params = getattr(fn, "__type_params__", ())
    if not issubclass(type(type_params), tuple) or tuple.__len__(type_params):
        raise SourcePreparationError("generic function headers need separate support")
    if fn.__code__.co_flags & (
        inspect.CO_GENERATOR | inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR
    ):
        raise SourcePreparationError(
            "generator and asynchronous callables need separate support"
        )
    return fn


def callable_preimage(function: FunctionType) -> CallablePreimage:
    """Read explicit recipe preimages without executing the function or header."""
    fn = _function(function)
    try:
        source = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError) as exc:
        raise SourcePreparationError(
            "callable source is unavailable; an exact source adapter remains unfinished"
        ) from exc
    return CallablePreimage(source, _sha(source.encode()), _code_sha(fn.__code__))


def _global_reads(code: CodeType) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            str(instruction.argval)
            for instruction in dis.get_instructions(code)
            if instruction.opname in {"LOAD_GLOBAL", "LOAD_NAME"}
        )
    )


def _cells(fn: FunctionType) -> dict[str, CellType]:
    cells = dict(zip(fn.__code__.co_freevars, fn.__closure__ or (), strict=True))
    for cell in cells.values():
        try:
            cell.cell_contents
        except ValueError as exc:
            raise SourcePreparationError(
                "empty closure cells need separate support"
            ) from exc
    return cells


def _read_ref(fn: FunctionType, reference: BindingRef) -> object:
    if not isinstance(reference.name, str) or not reference.name:
        raise SourcePreparationError("binding names must be nonempty strings")
    if reference.scope == "closure":
        cell = _cells(fn).get(reference.name)
        if cell is None:
            raise SourcePreparationError(
                f"closure binding {reference.name!r} is not an actual capture"
            )
        try:
            value = cell.cell_contents
        except ValueError as exc:
            raise SourcePreparationError(
                "empty closure cells need separate support"
            ) from exc
    elif reference.scope == "global":
        if (
            reference.name not in _global_reads(fn.__code__)
            or reference.name not in fn.__globals__
        ):
            raise SourcePreparationError(
                f"global binding {reference.name!r} is not an actual named read"
            )
        value = fn.__globals__[reference.name]
    else:
        raise SourcePreparationError(
            "only explicit global or closure bindings are supported"
        )
    if value is not reference.expected:
        raise SourcePreparationError(
            f"binding {reference.scope}:{reference.name} changed identity"
        )
    return value


def _dtype(value: object) -> str:
    for original, name in _DTYPES:
        if value is original:
            return name
    raise SourcePreparationError(
        "selected dtype must be an actual supported native floating scalar type"
    )


def _at_path(root: ast.AST, path: AstPath) -> ast.AST:
    current: Any = root
    if not path or path[0] != "body":
        raise SourcePreparationError(
            "dtype sites must be explicit structural paths in the function body"
        )
    for component in path:
        if (
            isinstance(component, str)
            and isinstance(current, ast.AST)
            and component in current._fields
        ):
            current = getattr(current, component)
        elif (
            type(component) is int
            and isinstance(current, list)
            and 0 <= component < len(current)
        ):
            current = current[component]
        else:
            raise SourcePreparationError(
                f"selected dtype site path {path!r} is missing or changed"
            )
    if not isinstance(current, ast.AST):
        raise SourcePreparationError(
            "selected dtype path does not name an AST expression"
        )
    return current


def _expression_binding(
    fn: FunctionType, node: ast.AST, reference: BindingRef
) -> object:
    attributes: list[str] = []
    current = node
    while isinstance(current, ast.Attribute) and isinstance(current.ctx, ast.Load):
        attributes.insert(0, current.attr)
        current = current.value
    if not (
        isinstance(current, ast.Name)
        and isinstance(current.ctx, ast.Load)
        and current.id == reference.name
    ):
        raise SourcePreparationError(
            "selected dtype expression must be the declared Name or module Attribute path"
        )
    if current.id in fn.__code__.co_varnames or current.id in fn.__code__.co_cellvars:
        raise SourcePreparationError(
            "selected dtype name is shadowed by a local binding"
        )
    value = _read_ref(fn, reference)
    for attribute in attributes:
        if type(value) is not ModuleType or attribute not in vars(value):
            raise SourcePreparationError(
                "selected dtype module binding is unsupported or changed"
            )
        value = vars(value)[attribute]
    return value


def _validate_body(fn: FunctionType, definition: ast.FunctionDef) -> None:
    forbidden = (
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Lambda,
        ast.Global,
        ast.Nonlocal,
        ast.Yield,
        ast.YieldFrom,
        ast.Await,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
    )
    for statement in definition.body:
        for node in ast.walk(statement):
            if isinstance(node, (ast.FunctionDef, *forbidden)):
                raise SourcePreparationError(
                    "nested definitions, generators, comprehensions and global/nonlocal writes need separate preparation support"
                )
            if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(
                node.ctx, (ast.Store, ast.Del)
            ):
                raise SourcePreparationError(
                    "attribute/subscript state writes need explicit separate state preparation"
                )
    reads = _global_reads(fn.__code__)
    cells = _cells(fn)
    if any(
        name
        in {
            "globals",
            "locals",
            "eval",
            "exec",
            "compile",
            "__import__",
            "__builtins__",
        }
        for name in reads
    ):
        raise SourcePreparationError(
            "dynamic namespace dispatch needs separate preparation support"
        )

    def actual_read(name: str) -> object:
        if name in cells:
            return cast(object, cells[name].cell_contents)
        if name not in reads:
            return _MISSING
        if name in fn.__globals__:
            return cast(object, fn.__globals__[name])
        # Python uses the function's cached builtin mapping, not a later
        # globals['__builtins__'] rebind. Do not invoke an unknown mapping hook.
        if type(fn.__builtins__) is not dict:
            raise SourcePreparationError(
                "non-dictionary builtin dispatch needs separate preparation support"
            )
        return cast(object, fn.__builtins__.get(name, _MISSING))

    for name in (*reads, *fn.__code__.co_freevars):
        value = actual_read(name)
        if value is fn:
            raise SourcePreparationError(
                "recursive/self-dispatched preparation needs an explicit recursive owner"
            )
    # Only the executed body has dispatch semantics. Source header expressions
    # are stripped before compilation; their actual original objects are kept.
    local_names = set(fn.__code__.co_varnames) | set(fn.__code__.co_cellvars)
    for node in (
        child for statement in definition.body for child in ast.walk(statement)
    ):
        if not isinstance(node, ast.Attribute):
            continue
        parts: list[str] = []
        current: ast.AST = node
        while isinstance(current, ast.Attribute):
            parts.insert(0, current.attr)
            current = current.value
        if not isinstance(current, ast.Name) or current.id in local_names:
            continue
        value = actual_read(current.id)
        for part in parts:
            if type(value) is not ModuleType or part not in vars(value):
                break
            value = vars(value)[part]
        else:
            if value is fn:
                raise SourcePreparationError(
                    "module self-dispatch needs an explicit recursive owner"
                )


def _definition(fn: FunctionType, source: str) -> ast.FunctionDef:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise SourcePreparationError(
            "callable source is not a supported Python function"
        ) from exc
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise SourcePreparationError(
            "callable source must identify exactly one ordinary function"
        )
    definition = tree.body[0]
    if definition.name != fn.__code__.co_name:
        raise SourcePreparationError(
            "source function name differs from the live code object"
        )
    _validate_body(fn, definition)
    return definition


def _compile_body(
    fn: FunctionType,
    definition: ast.FunctionDef,
    filename: str,
    *,
    imported_globals: tuple[str, ...] = (),
    class_name: str | None = None,
) -> CodeType:
    """Compile code constants only; no def/default/decorator expression executes."""
    target = copy.deepcopy(definition)
    target.decorator_list = []
    target.returns = None
    target.type_comment = None
    arguments = (*target.args.posonlyargs, *target.args.args, *target.args.kwonlyargs)
    for argument in (*arguments, target.args.vararg, target.args.kwarg):
        if argument is not None:
            argument.annotation = None
            argument.type_comment = None
    target.args.defaults = []
    target.args.kw_defaults = [None] * len(target.args.kwonlyargs)
    if getattr(target, "type_params", ()):
        raise SourcePreparationError(
            "generic function compilation needs separate support"
        )
    assignments = [
        ast.Assign(
            targets=[ast.Name(id=name, ctx=ast.Store())], value=ast.Constant(None)
        )
        for name in fn.__code__.co_freevars
    ]
    # An already-qualified NNX role can supply its original class compiler
    # scope. The carrier supplies only lexical context, including private-name
    # mangling and native method flags; its body and headers never execute.
    container: ast.FunctionDef | ast.ClassDef = target
    if class_name is not None:
        container = ast.ClassDef(
            name=class_name,
            bases=[],
            keywords=[],
            body=[target],
            decorator_list=[],
            type_params=[],
        )
    factory = ast.FunctionDef(
        name="__prepared_closure_factory__",
        args=ast.arguments(
            posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
        ),
        body=[*assignments, container],
        decorator_list=[],
        type_params=[],
    )
    # CPython can choose different call opcodes for known imported module names.
    # These declarations supply only compiler symbol-table context. This module
    # and its enclosing factory are never executed, including these imports.
    imports = [
        ast.Import(names=[ast.alias(name="builtins", asname=name)])
        for name in imported_globals
    ]
    module = ast.fix_missing_locations(
        ast.Module(body=[*imports, factory], type_ignores=[])
    )
    try:
        compiled = compile(
            module,
            filename,
            "exec",
            flags=fn.__code__.co_flags & _FUTURE_FLAGS,
            dont_inherit=True,
        )
    except (SyntaxError, TypeError, ValueError) as exc:
        raise SourcePreparationError(
            "source body cannot be compiled without changing its supported lexical scope"
        ) from exc
    outer = next(
        value
        for value in compiled.co_consts
        if isinstance(value, CodeType) and value.co_name == factory.name
    )
    if class_name is not None:
        outer = next(
            value
            for value in outer.co_consts
            if isinstance(value, CodeType) and value.co_name == class_name
        )
    return next(
        value
        for value in outer.co_consts
        if isinstance(value, CodeType) and value.co_name == target.name
    )


def _code_contract(code: CodeType) -> tuple[Any, ...]:
    def constant(value: Any) -> Any:
        if isinstance(value, CodeType):
            return _code_contract(value)
        return (type(value), marshal.dumps(value))

    return (
        code.co_code,
        tuple(constant(value) for value in code.co_consts),
        code.co_names,
        code.co_varnames,
        code.co_freevars,
        code.co_cellvars,
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_stacksize,
        code.co_flags & ~inspect.CO_NESTED,
        code.co_exceptiontable,
    )


def _object(value: object) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": id(value),
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
    }
    if type(value) is FunctionType:
        fn = value
        result.update(
            name=fn.__name__,
            qualname=fn.__qualname__,
            module=fn.__module__,
            code_sha256=_code_sha(fn.__code__),
            filename=fn.__code__.co_filename,
            firstlineno=fn.__code__.co_firstlineno,
        )
    elif type(value) is ModuleType:
        result["module"] = value.__name__
    return result


def _immutable(value: object) -> bool:
    if any(value is dtype for dtype, _ in _DTYPES):
        return True
    if value is None or type(value) in {
        bool,
        int,
        float,
        complex,
        str,
        bytes,
        type,
        ModuleType,
        FunctionType,
        BuiltinFunctionType,
    }:
        return True
    if isinstance(value, (jax.Array, np.generic, np.dtype)):
        return True
    if type(value) in {tuple, frozenset}:
        return all(_immutable(item) for item in cast(Any, value))
    return False


def _runtime_bindings(fn: FunctionType) -> dict[tuple[SharedScope, str | int], object]:
    values: dict[tuple[SharedScope, str | int], object] = {
        ("global", name): fn.__globals__[name]
        for name in _global_reads(fn.__code__)
        if name in fn.__globals__
    }
    values.update(
        {("closure", name): cell.cell_contents for name, cell in _cells(fn).items()}
    )
    values.update(
        {("default", index): value for index, value in enumerate(fn.__defaults__ or ())}
    )
    values.update(
        {
            ("kwdefault", name): value
            for name, value in (fn.__kwdefaults__ or {}).items()
        }
    )
    return values


@contextmanager
def _annotation_errors() -> Iterator[None]:
    """Translate the stdlib adapter error at this preparation boundary."""
    try:
        yield
    except AnnotationStateError as exc:
        raise SourcePreparationError(str(exc)) from exc


@dataclass(frozen=True)
class _FunctionState:
    function: FunctionType = field(repr=False, compare=False)
    attributes: tuple[object, ...] = field(repr=False, compare=False)
    mappings: tuple[tuple[dict[str, Any], tuple[tuple[str, object], ...]], ...] = field(
        repr=False, compare=False
    )
    cells: tuple[tuple[CellType, object], ...] = field(repr=False, compare=False)
    globals: tuple[tuple[str, object], ...] = field(repr=False, compare=False)
    annotations: AnnotationState = field(repr=False, compare=False)

    @classmethod
    def capture(
        cls,
        fn: FunctionType,
        *,
        watch_globals: bool,
        annotations: AnnotationState | None = None,
    ) -> _FunctionState:
        attributes = (
            fn.__code__,
            fn.__globals__,
            fn.__builtins__,
            fn.__defaults__,
            fn.__kwdefaults__,
            fn.__name__,
            fn.__qualname__,
            fn.__module__,
            fn.__doc__,
        )
        mappings = tuple(
            (mapping, tuple(dict.items(mapping)))
            for mapping in (fn.__dict__, fn.__kwdefaults__)
            if mapping is not None
        )
        cells = tuple((cell, cell.cell_contents) for cell in fn.__closure__ or ())
        globals_ = (
            tuple(
                (name, fn.__globals__.get(name, _MISSING))
                for name in _global_reads(fn.__code__)
            )
            if watch_globals
            else ()
        )
        if annotations is None:
            with _annotation_errors():
                annotations = AnnotationState.capture(fn)
        return cls(fn, attributes, mappings, cells, globals_, annotations)

    def assert_current(self) -> None:
        self._assert_metadata_current()
        with _annotation_errors():
            self.annotations.assert_current()

    def _assert_metadata_current(self) -> None:
        fn = self.function
        actual = (
            fn.__code__,
            fn.__globals__,
            fn.__builtins__,
            fn.__defaults__,
            fn.__kwdefaults__,
            fn.__name__,
            fn.__qualname__,
            fn.__module__,
            fn.__doc__,
        )
        if not all(a is b for a, b in zip(actual, self.attributes, strict=True)):
            raise SourcePreparationError(
                "prepared dependency function metadata or code changed"
            )
        for mapping, entries in self.mappings:
            actual_entries = tuple(dict.items(mapping))
            if len(actual_entries) != len(entries) or any(
                key is not expected_key or value is not expected_value
                for (key, value), (expected_key, expected_value) in zip(
                    actual_entries, entries, strict=True
                )
            ):
                raise SourcePreparationError(
                    "prepared dependency metadata bindings changed"
                )
        try:
            changed_cell = any(
                cell.cell_contents is not value for cell, value in self.cells
            )
        except ValueError as exc:
            raise SourcePreparationError(
                "prepared dependency closure cell became empty"
            ) from exc
        if changed_cell:
            raise SourcePreparationError("prepared dependency closure binding changed")
        if any(
            fn.__globals__.get(name, _MISSING) is not value
            for name, value in self.globals
        ):
            raise SourcePreparationError("prepared dependency global binding changed")


@dataclass(frozen=True)
class PreparationRecord:
    _json: str = field(repr=False)
    _original: _FunctionState = field(repr=False, compare=False)
    _prepared: _FunctionState = field(repr=False, compare=False)
    _dependencies: tuple[PreparationRecord, ...] = field(
        default=(), repr=False, compare=False
    )

    @property
    def sha256(self) -> str:
        return _sha(self._json.encode())

    def to_json(self) -> str:
        return self._json

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self._json))

    def assert_current(self) -> None:
        """Check function bindings and explicit dependencies, not referenced state.

        Original global rebinding does not update the prepared snapshot. Shared
        objects are not frozen or recursively validated by this record.
        """
        for dependency in self._dependencies:
            dependency.assert_current()
        self._original.assert_current()
        self._prepared.assert_current()


def prepare_callable(
    function: FunctionType, recipe: CallableRecipe
) -> tuple[FunctionType, PreparationRecord]:
    """Prepare selected choices without executing original source or headers.

    Native execution and numerical acceptance remain the caller's separate work.
    Global names are snapshotted; module objects and retained closure cells are
    shared references. Mutable/unknown runtime captures require explicit ownership.
    """
    fn = _function(function)
    if recipe.original is not fn:
        raise SourcePreparationError("recipe does not identify this original callable")
    if not isinstance(recipe.intent, str) or not recipe.intent.strip():
        raise SourcePreparationError(
            "an explicit intentional numerical change is required"
        )
    if type(recipe.required_site_count) is not int or recipe.required_site_count != len(
        recipe.sites
    ):
        raise SourcePreparationError(
            "a required selected dtype site is missing or duplicated"
        )
    if not recipe.sites and not recipe.helpers:
        raise SourcePreparationError(
            "no dtype site or prepared helper binding was explicitly selected"
        )
    preimage = callable_preimage(fn)
    if (preimage.source_sha256, preimage.code_sha256) != (
        recipe.source_sha256,
        recipe.code_sha256,
    ):
        raise SourcePreparationError("callable source/code preimage changed")
    definition = _definition(fn, preimage.source)
    imported_globals: tuple[str, ...] = ()
    untouched = _compile_body(fn, definition, fn.__code__.co_filename)
    if _code_contract(untouched) != _code_contract(fn.__code__):
        imported_globals = tuple(
            name
            for name in _global_reads(fn.__code__)
            if type(fn.__globals__.get(name)) is ModuleType
        )
        untouched = _compile_body(
            fn,
            definition,
            fn.__code__.co_filename,
            imported_globals=imported_globals,
        )
    if _code_contract(untouched) != _code_contract(fn.__code__):
        raise SourcePreparationError(
            "inspected source does not correspond to the supplied live code in a supported compiler binding context"
        )
    original_state = _FunctionState.capture(fn, watch_globals=False)
    runtime = _runtime_bindings(fn)
    shared: list[dict[str, Any]] = []
    permissions: dict[tuple[SharedScope, str | int], object] = {}
    for entry in recipe.shared:
        key = (entry.scope, entry.name)
        if (
            key in permissions
            or key not in runtime
            or runtime[key] is not entry.expected
        ):
            raise SourcePreparationError(
                "declared shared capture is duplicated, missing or changed"
            )
        permissions[key] = entry.expected
        shared.append(
            {
                "scope": entry.scope,
                "name": entry.name,
                "object": _object(entry.expected),
                "ownership": "shared-reference",
            }
        )
    for key, value in runtime.items():
        if key not in permissions and not _immutable(value):
            raise SourcePreparationError(
                f"mutable or unknown capture {key!r} requires declared shared ownership"
            )
    namespace = dict(fn.__globals__)
    # Function objects cache their actual builtin mapping when constructed.
    # A later globals['__builtins__'] rebind must not change this clone's reads.
    namespace["__builtins__"] = fn.__builtins__
    cells = _cells(fn)
    helper_rows: list[dict[str, Any]] = []
    helper_keys: set[tuple[str, str]] = set()
    for helper in recipe.helpers:
        original = _read_ref(fn, helper.reference)
        key = (helper.reference.scope, helper.reference.name)
        if key in helper_keys:
            raise SourcePreparationError("prepared helper binding is duplicated")
        helper_keys.add(key)
        helper.record.assert_current()
        if (
            helper.record._original.function is not original
            or helper.record._prepared.function is not helper.prepared
        ):
            raise SourcePreparationError(
                "prepared helper identity/dependency record does not match this selected binding"
            )
        if helper.reference.scope == "global":
            namespace[helper.reference.name] = helper.prepared
        else:
            cells[helper.reference.name] = CellType(helper.prepared)
        helper_rows.append(
            {
                "scope": helper.reference.scope,
                "name": helper.reference.name,
                "original": _object(original),
                "prepared": _object(helper.prepared),
                "preparation_sha256": helper.record.sha256,
            }
        )
    changed = copy.deepcopy(definition)
    paths: set[AstPath] = set()
    site_rows: list[dict[str, Any]] = []
    reserved = (
        set(namespace)
        | {n.id for n in ast.walk(definition) if isinstance(n, ast.Name)}
        | set(fn.__code__.co_freevars)
        | set(fn.__code__.co_varnames)
        | set(fn.__code__.co_cellvars)
    )
    replacements: dict[int, ast.AST] = {}
    for index, site in enumerate(recipe.sites):
        if site.path in paths:
            raise SourcePreparationError("selected dtype site is duplicated")
        paths.add(site.path)
        node = _at_path(changed, site.path)
        if _sha(ast.dump(node, include_attributes=False).encode()) != site.ast_sha256:
            raise SourcePreparationError(
                "selected dtype site structural fingerprint changed"
            )
        actual_dtype = _expression_binding(fn, node, site.root)
        if actual_dtype is not site.expected_dtype:
            raise SourcePreparationError(
                "selected dtype expression no longer has its expected original binding"
            )
        old_dtype, new_dtype = _dtype(actual_dtype), _dtype(site.replacement_dtype)
        if new_dtype == "float64" and not jax.config.x64_enabled:
            raise SourcePreparationError(
                "FP64 callable preparation requires already-effective JAX x64"
            )
        private = f"__jax2onnx_prepared_dtype_{index}"
        while private in reserved:
            private += "_"
        reserved.add(private)
        namespace[private] = site.replacement_dtype
        replacements[id(node)] = ast.copy_location(
            ast.Name(id=private, ctx=ast.Load()), node
        )
        site_rows.append(
            {
                "path": list(site.path),
                "ast_sha256": site.ast_sha256,
                "expression": ast.unparse(node),
                "old_dtype": old_dtype,
                "new_dtype": new_dtype,
                "root": {
                    "scope": site.root.scope,
                    "name": site.root.name,
                    "object": _object(site.root.expected),
                },
                "private_name": private,
            }
        )

    class Replace(ast.NodeTransformer):
        def visit(self, node: ast.AST) -> ast.AST:
            return replacements.get(id(node)) or super().visit(node)

    changed = cast(ast.FunctionDef, Replace().visit(changed))
    code = _compile_body(
        fn,
        changed,
        f"<jax2onnx-prepared:{preimage.source_sha256}>",
        imported_globals=imported_globals,
    )
    if any(name not in cells for name in code.co_freevars):
        raise SourcePreparationError(
            "prepared code introduced an undeclared closure dependency"
        )
    prepared = FunctionType(
        code,
        namespace,
        fn.__name__,
        fn.__defaults__,
        tuple(cells[name] for name in code.co_freevars) or None,
    )
    prepared.__kwdefaults__ = fn.__kwdefaults__
    prepared.__qualname__ = fn.__qualname__
    prepared.__module__ = fn.__module__
    prepared.__doc__ = fn.__doc__
    prepared.__dict__.update(fn.__dict__)
    with _annotation_errors():
        prepared_annotations = original_state.annotations.attach(
            prepared, check_metadata=original_state._assert_metadata_current
        )
    original_state.assert_current()
    payload = {
        "schema_version": 1,
        "kind": "selected-callable-dtype-preparation-v1",
        "intent": recipe.intent,
        "original": _object(fn),
        "prepared": _object(prepared),
        "source_sha256": preimage.source_sha256,
        "source_compile_context": {"imported_global_names": list(imported_globals)},
        "builtin_binding": _object(fn.__builtins__),
        "prepared_ast_sha256": _sha(
            ast.dump(changed, include_attributes=False).encode()
        ),
        "sites": site_rows,
        "helpers": helper_rows,
        "shared": shared,
        "retained_bindings": [
            {"scope": scope, "name": name, "object": _object(value)}
            for (scope, name), value in runtime.items()
        ],
        "namespace_semantics": "Named global bindings are frozen in a private preparation-time snapshot; later original global rebinding does not update it. Module members, the actual builtin mapping and retained closure cells remain shared references. Mutation of shared objects needs fresh evidence; this record does not freeze state or implicitly prepare it.",
        "acceptance": "UNVERIFIED_PREPARATION_ONLY",
    }
    record = PreparationRecord(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        original_state,
        _FunctionState.capture(
            prepared, watch_globals=True, annotations=prepared_annotations
        ),
        tuple(helper.record for helper in recipe.helpers),
    )
    record.assert_current()
    return prepared, record
