# jax2onnx/_source_nnx_method_preparation.py

"""Private preparation of the original finite NNX Transformer method graph.

Only softmax accumulation is intentionally selected for FP64. Original method
bodies, parameter storage and final casts remain native source. Prepared methods
receive actual NNX receivers, including receivers reconstructed by split/merge.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import asdict, dataclass, field, replace
import hashlib
import inspect
import json
from pathlib import Path
from types import CellType, CodeType, FunctionType, ModuleType
from typing import Any, cast

from flax import nnx
import jax
import jax.numpy as jnp

from jax2onnx._source_callable_preparation import (
    BindingRef,
    CallableRecipe,
    DtypeSite,
    HelperBinding,
    PreparationRecord,
    _FunctionState,
    _cells,
    _code_contract,
    _compile_body,
    _definition,
    _function,
    _global_reads,
    _immutable,
    _runtime_bindings,
    callable_preimage,
    prepare_callable,
)
from jax2onnx._source_preparation import SourcePreparationError
from jax2onnx.plugins.flax.nnx._source_function_state import (
    NNXFunctionState,
    prepare_nnx_function_state,
)


_SOURCE_SHA = "850a02ae4a8ad64d06bfb30a0d18c2e0557d5fd0b2a17fb3abfc7b72f0611922"
_INTENT = "FP64 softmax accumulation; retain original casts and parameter storage"


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _paths(node: ast.AST, path: tuple[str | int, ...] = ()) -> Any:
    yield path, node
    for name, value in ast.iter_fields(node):
        if isinstance(value, ast.AST):
            yield from _paths(value, (*path, name))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                if isinstance(child, ast.AST):
                    yield from _paths(child, (*path, name, index))


def _none_capture_ports(*values: object) -> None:
    # These are the original factory's actual None ports, never coerced flags.
    if any(value is not None for value in values):
        raise SourcePreparationError(
            "prepared original Transformer requires its original None capture ports"
        )


def _owner_metadata(owner_type: type[Any]) -> tuple[object, ...]:
    return tuple(
        type.__getattribute__(owner_type, name)
        for name in ("__name__", "__qualname__", "__module__", "__bases__", "__mro__")
    )


@dataclass(frozen=True)
class _PinnedRole:
    owner: ModuleType | type[Any] = field(repr=False, compare=False)
    attribute: str
    state: _FunctionState = field(repr=False, compare=False)
    source_ast_sha256: str
    compiler_class_name: str | None

    def assert_current(self) -> None:
        self.state.assert_current()
        current = vars(self.owner).get(self.attribute)
        if current is self.state.function:
            return
        from jax2onnx.plugins.plugin_system import _PATCH_STATE

        patch = _PATCH_STATE.get((self.owner, self.attribute))
        if (
            not isinstance(self.owner, type)
            or patch is None
            or patch["orig"] is not self.state.function
            or patch["patched"] is not current
        ):
            raise SourcePreparationError("pinned original GPT role binding changed")


def _pinned_original_roles(source: ModuleType, text: str) -> tuple[_PinnedRole, ...]:
    """Bind finite reachable live roles to the already pinned disk definitions."""
    tree = ast.parse(text)
    expected_module = compile(
        text, cast(str, vars(source)["__file__"]), "exec", dont_inherit=True
    )
    names = (
        "Transformer.__call__",
        "TransformerBlock.__call__",
        "AttentionCore.__call__",
        "MLPBlockCore.__call__",
        "TokenEmbedding.__call__",
        "RMSNorm.__call__",
        "RotaryEmbedding.__call__",
        "_softmax_torch_approx",
        "_sdpa_impl",
        "sdpa",
        "_linear",
        "_matmul_lastdim",
        "_swiglu",
        "_concat",
    )
    filename = Path(cast(str, vars(source)["__file__"])).resolve()
    roles = []
    for qualified in names:
        parts = qualified.split(".")
        owner: ModuleType | type[Any] = source
        statements = tree.body
        expected_container = expected_module
        if len(parts) == 2:
            owner = cast(type[Any], vars(source).get(parts[0]))
            if (
                not isinstance(owner, type)
                or owner.__module__ != source.__name__
                or owner.__qualname__ != parts[0]
            ):
                raise SourcePreparationError("pinned original GPT role owner changed")
            class_definition = next(
                node
                for node in statements
                if isinstance(node, ast.ClassDef) and node.name == parts[0]
            )
            statements = class_definition.body
            expected_container = next(
                value
                for value in expected_module.co_consts
                if isinstance(value, CodeType) and value.co_name == parts[0]
            )
        expected_code = next(
            value
            for value in expected_container.co_consts
            if isinstance(value, CodeType) and value.co_name == parts[-1]
        )
        definition = next(
            node
            for node in statements
            if isinstance(node, ast.FunctionDef) and node.name == parts[-1]
        )
        function = _function(vars(owner).get(parts[-1]))
        if (
            function.__globals__ is not vars(source)
            or function.__module__ != source.__name__
            or function.__name__ != parts[-1]
            or function.__qualname__ != qualified
            or function.__code__.co_name != parts[-1]
            or function.__code__.co_qualname != qualified
            or Path(function.__code__.co_filename).resolve() != filename
            or function.__code__.co_firstlineno != definition.lineno
        ):
            raise SourcePreparationError("pinned original GPT live role changed")
        expected = ast.dump(definition, include_attributes=False)
        actual = ast.parse(callable_preimage(function).source)
        if (
            len(actual.body) != 1
            or ast.dump(actual.body[0], include_attributes=False) != expected
        ):
            raise SourcePreparationError("pinned original GPT role source changed")
        # Compare exact code constants from the pinned whole module, retaining
        # native nested-function/import compiler context without executing it.
        if _code_contract(function.__code__) != _code_contract(expected_code):
            raise SourcePreparationError("pinned original GPT role code changed")
        defaults = tuple(ast.literal_eval(value) for value in definition.args.defaults)
        actual_defaults = function.__defaults__
        if (
            (actual_defaults is None) != (not defaults)
            or len(actual_defaults or ()) != len(defaults)
            or any(
                type(a) is not type(b) or a != b
                for a, b in zip(actual_defaults or (), defaults, strict=True)
            )
        ):
            raise SourcePreparationError("pinned original GPT role defaults changed")
        kwdefaults = {
            argument.arg: ast.literal_eval(value)
            for argument, value in zip(
                definition.args.kwonlyargs, definition.args.kw_defaults, strict=True
            )
            if value is not None
        }
        actual_kwdefaults = function.__kwdefaults__
        if (
            (actual_kwdefaults is None) != (not kwdefaults)
            or tuple(actual_kwdefaults or {}) != tuple(kwdefaults)
            or any(
                type((actual_kwdefaults or {})[key]) is not type(value)
                or (actual_kwdefaults or {})[key] != value
                for key, value in kwdefaults.items()
            )
        ):
            raise SourcePreparationError("pinned original GPT role kwdefaults changed")
        role = _PinnedRole(
            owner,
            parts[-1],
            _FunctionState.capture(function, watch_globals=True),
            _sha(expected.encode()),
            parts[0] if len(parts) == 2 else None,
        )
        role.assert_current()
        roles.append(role)
    return tuple(roles)


@dataclass(frozen=True)
class _PreparedMethod:
    owner_type: type[Any]
    original: FunctionType
    method: FunctionType
    record: PreparationRecord
    schema: NNXFunctionState = field(repr=False, compare=False)
    owner_metadata: tuple[object, ...] = field(repr=False, compare=False)
    config_json: str | None = field(repr=False, compare=False)
    original_roles: tuple[_PinnedRole, ...] = field(
        default=(), repr=False, compare=False
    )

    def assert_receiver(self, receiver: object) -> None:
        for role in self.original_roles:
            role.assert_current()
        if type(receiver) is not self.owner_type:
            raise SourcePreparationError("prepared method receiver type changed")
        if any(
            a is not b
            for a, b in zip(
                _owner_metadata(self.owner_type), self.owner_metadata, strict=True
            )
        ):
            raise SourcePreparationError("prepared method owner metadata changed")
        if (
            self.record._original.function is not self.original
            or self.record._prepared.function is not self.method
            or self.schema.method is not self.method
        ):
            raise SourcePreparationError("prepared method role binding changed")
        self.record.assert_current()
        if (
            self.config_json is not None
            and json.dumps(asdict(cast(Any, receiver).config), sort_keys=True)
            != self.config_json
        ):
            raise SourcePreparationError("prepared method configuration changed")
        # The descriptor has topology/specs/method, without a live receiver.
        self.schema._owner_state(receiver)

    def invoke(self, receiver: object, *args: Any, **kwargs: Any) -> Any:
        from jax2onnx.plugins.plugin_system import _call_prepared_nnx_method

        return _call_prepared_nnx_method(self, receiver, args, kwargs)


def _untouched(
    function: FunctionType,
    definition: ast.FunctionDef,
    *,
    class_name: str | None = None,
) -> tuple[str, ...]:
    imported: tuple[str, ...] = ()
    original = _compile_body(
        function, definition, function.__code__.co_filename, class_name=class_name
    )
    if _code_contract(original) != _code_contract(function.__code__):
        imported = tuple(
            name
            for name in _global_reads(function.__code__)
            if type(function.__globals__.get(name)) is ModuleType
        )
        original = _compile_body(
            function,
            definition,
            function.__code__.co_filename,
            imported_globals=imported,
            class_name=class_name,
        )
    if _code_contract(original) != _code_contract(function.__code__):
        raise SourcePreparationError("original method source/live code changed")
    return imported


def _copy_function(
    original: FunctionType,
    definition: ast.FunctionDef,
    namespace: dict[str, Any],
    cells: dict[str, CellType],
    imported: tuple[str, ...],
    *,
    class_name: str | None = None,
) -> FunctionType:
    code = _compile_body(
        original,
        definition,
        f"<jax2onnx-prepared-nnx:{_sha(ast.dump(definition).encode())}>",
        imported_globals=imported,
        class_name=class_name,
    )
    if any(name not in cells for name in code.co_freevars):
        raise SourcePreparationError("prepared method introduced an unknown closure")
    prepared = FunctionType(
        code,
        namespace,
        original.__name__,
        original.__defaults__,
        tuple(cells[name] for name in code.co_freevars) or None,
    )
    prepared.__kwdefaults__ = original.__kwdefaults__
    prepared.__annotations__ = original.__annotations__
    prepared.__qualname__ = original.__qualname__
    prepared.__module__ = original.__module__
    prepared.__doc__ = original.__doc__
    prepared.__dict__.update(original.__dict__)
    return prepared


def _prepare_method(
    receiver: object,
    *,
    original_roles: tuple[_PinnedRole, ...],
    helpers: tuple[HelperBinding, ...] = (),
    children: tuple[tuple[str, _PreparedMethod], ...] = (),
) -> _PreparedMethod:
    owner_type = type(receiver)
    function = vars(owner_type)["__call__"]
    if type(function) is not FunctionType:
        raise SourcePreparationError("original NNX method is not an unpatched function")
    pinned_roles = tuple(
        role
        for role in original_roles
        if role.owner is owner_type
        and role.attribute == "__call__"
        and role.state.function is function
    )
    if len(pinned_roles) != 1 or pinned_roles[0].compiler_class_name is None:
        raise SourcePreparationError("original NNX method has no pinned class scope")
    pinned_role = pinned_roles[0]
    pinned_role.assert_current()
    class_name = pinned_role.compiler_class_name
    preimage = callable_preimage(function)
    tree = ast.parse(preimage.source)
    if (
        len(tree.body) != 1
        or not isinstance(tree.body[0], ast.FunctionDef)
        or tree.body[0].name != function.__name__
    ):
        raise SourcePreparationError("original method source definition changed")
    definition = tree.body[0]
    imported = _untouched(function, definition, class_name=class_name)
    original_state = _FunctionState.capture(function, watch_globals=False)
    for key, value in _runtime_bindings(function).items():
        if not _immutable(value):
            raise SourcePreparationError(f"unknown method capture {key!r}")
    namespace = dict(function.__globals__)
    namespace["__builtins__"] = function.__builtins__
    changed = copy.deepcopy(definition)
    ports = tuple(
        name
        for name in inspect.signature(function).parameters
        if name.startswith("capture_")
    )
    expected_ports = {
        "Transformer": (
            "capture_routing",
            "capture_hidden_states",
            "capture_block_debug",
        ),
        "TransformerBlock": ("capture_routing", "capture_debug"),
        "AttentionCore": ("capture_debug",),
        "MLPBlockCore": ("capture_routing", "capture_debug"),
    }
    if ports != expected_ports.get(owner_type.__name__):
        raise SourcePreparationError("original method capture-port schema changed")
    for name in ports:
        if inspect.signature(function).parameters[name].default is not None:
            raise SourcePreparationError("original capture default changed")
    forbidden = (
        ast.FunctionDef,
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
    writes = []
    for statement in definition.body:
        for node in ast.walk(statement):
            if isinstance(node, forbidden):
                raise SourcePreparationError("unsupported original method syntax")
            if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(
                node.ctx, (ast.Store, ast.Del)
            ):
                if (
                    not isinstance(node, ast.Subscript)
                    or not isinstance(node.value, ast.Name)
                    or node.value.id not in {"capture_debug", "debug_entry"}
                    or not isinstance(node.ctx, ast.Store)
                ):
                    raise SourcePreparationError("unowned method state write")
                writes.append(ast.dump(node, include_attributes=False))
    dependencies = []
    helper_rows = []
    for helper in helpers:
        name = helper.reference.name
        helper.record.assert_current()
        if (
            helper.reference.scope != "global"
            or namespace.get(name) is not helper.reference.expected
            or helper.record._original.function is not helper.reference.expected
            or helper.record._prepared.function is not helper.prepared
        ):
            raise SourcePreparationError("prepared method helper binding changed")
        namespace[name] = helper.prepared
        dependencies.append(helper.record)
        helper_rows.append(dict(name=name, record_sha256=helper.record.sha256))
    dispatch_rows = []
    reserved = set(namespace) | set(function.__code__.co_varnames)
    for index, (expression, child) in enumerate(children):
        matches = [
            (path, node)
            for path, node in _paths(changed)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == expression
        ]
        if len(matches) != 1:
            raise SourcePreparationError("original child-method call site changed")
        path, call = matches[0]
        private = f"__jax2onnx_nnx_child_{index}__"
        while private in reserved:
            private += "_"
        reserved.add(private)
        namespace[private] = child.invoke
        child.record.assert_current()
        dispatch_rows.append(
            dict(
                path=list(path),
                expression=expression,
                role=child.owner_type.__qualname__,
            )
        )
        call.args.insert(0, call.func)
        call.func = ast.copy_location(ast.Name(id=private, ctx=ast.Load()), call.func)
        dependencies.append(child.record)
    guard = "__jax2onnx_nnx_none_ports__"
    while guard in reserved:
        guard += "_"
    namespace[guard] = _none_capture_ports
    # Retain every original branch/write; enforce the fixed factory call contract
    # even when NNXFunctionState invokes the unbound prepared body directly.
    changed.body.insert(
        0,
        ast.Expr(
            value=ast.Call(
                func=ast.Name(id=guard, ctx=ast.Load()),
                args=[ast.Name(id=name, ctx=ast.Load()) for name in ports],
                keywords=[],
            )
        ),
    )
    ast.fix_missing_locations(changed)
    prepared = _copy_function(
        function, changed, namespace, _cells(function), imported, class_name=class_name
    )
    payload = dict(
        schema_version=1,
        kind="original-nnx-method-preparation-v1",
        intent=_INTENT,
        owner_role=owner_type.__qualname__,
        source_sha256=preimage.source_sha256,
        code_sha256=preimage.code_sha256,
        capture_ports=list(ports),
        original_guarded_writes=writes,
        retained_branches=True,
        helpers=helper_rows,
        child_dispatch=dispatch_rows,
        prepared_ast_sha256=_sha(ast.dump(changed, include_attributes=False).encode()),
        acceptance="UNVERIFIED_PREPARATION_ONLY",
    )
    record = PreparationRecord(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        original_state,
        _FunctionState.capture(prepared, watch_globals=True),
        tuple(dependencies),
    )
    result = prepare_nnx_function_state(receiver, prepared, 0)
    if result is None:
        raise SourcePreparationError("original receiver has no NNX state boundary")
    schema, _ = result
    config_json = (
        json.dumps(asdict(cast(Any, receiver).config), sort_keys=True)
        if owner_type.__name__ in {"Transformer", "AttentionCore", "MLPBlockCore"}
        else None
    )
    role = _PreparedMethod(
        owner_type,
        function,
        prepared,
        record,
        schema,
        _owner_metadata(owner_type),
        config_json,
    )
    role.assert_receiver(receiver)
    return role


def _helper(
    function: FunctionType, name: str, prepared: FunctionType, record: PreparationRecord
) -> HelperBinding:
    return HelperBinding(
        BindingRef("global", name, function.__globals__[name]), prepared, record
    )


def _prepare_helper(
    function: FunctionType,
    helpers: tuple[HelperBinding, ...] = (),
    *,
    softmax: bool = False,
) -> tuple[FunctionType, PreparationRecord]:
    preimage = callable_preimage(function)
    sites: tuple[DtypeSite, ...] = ()
    if softmax:
        definition = _definition(function, preimage.source)
        matches = [
            (p, n)
            for p, n in _paths(definition)
            if isinstance(n, ast.Attribute) and ast.unparse(n) == "jnp.float32"
        ]
        if len(matches) != 1:
            raise SourcePreparationError("original selected softmax dtype site changed")
        path, node = matches[0]
        sites = (
            DtypeSite(
                path,
                _sha(ast.dump(node, include_attributes=False).encode()),
                BindingRef("global", "jnp", function.__globals__["jnp"]),
                jnp.float32,
                jnp.float64,
            ),
        )
    return prepare_callable(
        function,
        CallableRecipe(
            function,
            preimage.source_sha256,
            preimage.code_sha256,
            sites,
            len(sites),
            _INTENT,
            helpers,
        ),
    )


@dataclass(frozen=True)
class NnxModelPreparationRecord:
    _json: str
    entry: PreparationRecord = field(repr=False, compare=False)
    root: _PreparedMethod = field(repr=False, compare=False)
    prepared_owner: Any = field(repr=False, compare=False)
    original_state: tuple[Any, ...] = field(repr=False, compare=False)

    def to_json(self) -> str:
        return self._json

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self._json))

    def assert_current(self) -> None:
        self.entry.assert_current()
        self.root.assert_receiver(self.prepared_owner)
        state = self.root.schema._owner_state(self.prepared_owner)
        if len(state) != len(self.original_state) or any(
            a is not b for a, b in zip(state, self.original_state, strict=True)
        ):
            raise SourcePreparationError("prepared Transformer stored state changed")


def prepare_nnx_transformer_softmax64(
    function: FunctionType,
) -> tuple[FunctionType, NnxModelPreparationRecord]:
    """Prepare the existing full original Transformer; execute no model body.

    The caller enables x64. Only the selected helper's FP64 accumulation changes;
    its original FP32 cast-back, all other source casts, and stored values remain.
    """
    from jax2onnx.plugins.examples.nnx import gpt_oss_flax as source

    if not jax.config.x64_enabled:
        raise SourcePreparationError("prepared FP64 softmax requires effective JAX x64")
    source_filename = vars(source).get("__file__")
    if type(source_filename) is not str:
        raise SourcePreparationError("original Transformer source filename unavailable")
    source_path = Path(source_filename)
    source_bytes = source_path.read_bytes()
    if _sha(source_bytes) != _SOURCE_SHA:
        raise SourcePreparationError("original full Transformer source file changed")
    original_roles = _pinned_original_roles(source, source_bytes.decode())
    if (
        type(function) is not FunctionType
        or function.__globals__ is not vars(source)
        or function.__module__ != source.__name__
        or function.__code__.co_qualname
        != "_build_transformer_model_apply.<locals>._apply"
        or Path(function.__code__.co_filename).resolve() != source_path.resolve()
    ):
        raise SourcePreparationError(
            "callable is not the original Transformer factory entry"
        )
    preimage = callable_preimage(function)
    cells = _cells(function)
    if tuple(cells) != ("module",):
        raise SourcePreparationError("original Transformer factory closure changed")
    original = cells["module"].cell_contents
    if type(original) is not source.Transformer:
        raise SourcePreparationError("original factory did not capture a Transformer")
    if (
        original.config is not source._MODEL_CONFIG
        or type(original.sequence_length) is not int
        or original.sequence_length != 3
    ):
        raise SourcePreparationError("original Transformer configuration changed")
    if type(original.blocks) is not nnx.Dict or tuple(original.blocks) != (
        "block_0",
        "block_1",
    ):
        raise SourcePreparationError("original Transformer block graph changed")
    definition = _definition(function, preimage.source)
    if (
        ast.unparse(definition.body[0]) != "return module(tokens)"
        or len(definition.body) != 1
    ):
        raise SourcePreparationError("original factory entry body changed")
    imported = _untouched(function, definition)
    original_descriptor = prepare_nnx_function_state(
        original, _function(source.Transformer.__call__), 0
    )
    if original_descriptor is None:
        raise SourcePreparationError("original Transformer state unavailable")
    original_schema, original_state = original_descriptor
    # Reuse the existing exact native graph protocol, preserving aliases even
    # when the caller's ambient NNX mode would otherwise request a tree clone.
    owner = original_schema.reconstruct(original_state)
    softmax, softmax_record = _prepare_helper(
        _function(source._softmax_torch_approx), softmax=True
    )
    sdpa_impl, sdpa_record = _prepare_helper(
        _function(source._sdpa_impl),
        (
            _helper(
                _function(source._sdpa_impl),
                "_softmax_torch_approx",
                softmax,
                softmax_record,
            ),
        ),
    )
    sdpa, sdpa_entry_record = _prepare_helper(
        _function(source.sdpa),
        (_helper(_function(source.sdpa), "_sdpa_impl", sdpa_impl, sdpa_record),),
    )
    # These roles describe structure/methods, not the first block's live owner.
    # The second block and every nnx.merge receiver validate the same schema.
    block = owner.blocks["block_0"]
    if (
        type(block) is not source.TransformerBlock
        or type(block.attention) is not source.AttentionCore
        or type(block.mlp) is not source.MLPBlockCore
    ):
        raise SourcePreparationError("original nested Transformer owner types changed")
    attention = _prepare_method(
        block.attention,
        original_roles=original_roles,
        helpers=(
            _helper(
                _function(source.AttentionCore.__call__),
                "sdpa",
                sdpa,
                sdpa_entry_record,
            ),
        ),
    )
    mlp = _prepare_method(
        block.mlp,
        original_roles=original_roles,
        helpers=(
            _helper(
                _function(source.MLPBlockCore.__call__),
                "_softmax_torch_approx",
                softmax,
                softmax_record,
            ),
        ),
    )
    block_role = _prepare_method(
        block,
        original_roles=original_roles,
        children=(("self.attention", attention), ("self.mlp", mlp)),
    )
    # Different masks are ordinary numeric state; their graph/spec schema must
    # match without capturing either block's actual payload in the dispatch role.
    for candidate in owner.blocks.values():
        block_role.assert_receiver(candidate)
        attention.assert_receiver(candidate.attention)
        mlp.assert_receiver(candidate.mlp)
    root = replace(
        _prepare_method(
            owner, original_roles=original_roles, children=(("block", block_role),)
        ),
        original_roles=original_roles,
    )
    if (
        original_schema.graphdef != root.schema.graphdef
        or original_schema.entries != root.schema.entries
        or original_schema.specs != root.schema.specs
        or original_schema.carriers != root.schema.carriers
        or original_schema.native_aliases != root.schema.native_aliases
    ):
        raise SourcePreparationError("NNX cloning changed graph/state metadata")
    cloned_state = root.schema._owner_state(owner)
    if any(a is not b for a, b in zip(original_state, cloned_state, strict=True)):
        raise SourcePreparationError("NNX cloning changed parameter storage/aliases")
    namespace = dict(function.__globals__)
    namespace["__builtins__"] = function.__builtins__
    private = "__jax2onnx_original_transformer_root"
    while private in namespace:
        private += "_"
    namespace[private] = root.invoke
    changed = copy.deepcopy(definition)
    call = cast(ast.Call, cast(ast.Return, changed.body[0]).value)
    call.args.insert(0, call.func)
    call.func = ast.copy_location(ast.Name(id=private, ctx=ast.Load()), call.func)
    ast.fix_missing_locations(changed)
    prepared = _copy_function(
        function, changed, namespace, {"module": CellType(owner)}, imported
    )
    entry_payload = dict(
        kind="original-nnx-transformer-entry-v1",
        source_sha256=preimage.source_sha256,
        acceptance="UNVERIFIED_PREPARATION_ONLY",
    )
    entry = PreparationRecord(
        json.dumps(entry_payload, sort_keys=True),
        _FunctionState.capture(function, watch_globals=False),
        _FunctionState.capture(prepared, watch_globals=True),
        (root.record,),
    )
    payload = dict(
        schema_version=1,
        kind="original-nnx-transformer-softmax64-v1",
        intent=_INTENT,
        original_source_sha256=_SOURCE_SHA,
        original_config=asdict(source._MODEL_CONFIG),
        original_role_definitions=[
            dict(
                role=role.state.function.__qualname__,
                source_ast_sha256=role.source_ast_sha256,
            )
            for role in original_roles
        ],
        sequence_length=original.sequence_length,
        construction_history="not inferred from the returned closure owner",
        parameter_storage="retained",
        method_records=[
            role.record.to_dict() for role in (root, block_role, attention, mlp)
        ],
        selected_softmax=softmax_record.to_dict(),
        acceptance="UNVERIFIED_PREPARATION_ONLY",
    )
    record = NnxModelPreparationRecord(
        json.dumps(payload, sort_keys=True), entry, root, owner, tuple(original_state)
    )
    record.assert_current()
    return prepared, record
