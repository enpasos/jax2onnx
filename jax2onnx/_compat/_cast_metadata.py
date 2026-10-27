# jax2onnx/_compat/_cast_metadata.py

"""Bounded CPU target and options acquisition for the two audited JAX epochs.

An old binding needs one fixed empty-program compile/load/export. Neither path
executes an executable. Constructed options belong to acquisition, not to the
original source compilation; no original MLIR or resolved shardings are given.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import ast
import gc
import functools
import hashlib
import itertools
import os
from pathlib import Path
import stat
import sys
import threading
from types import CodeType, FunctionType, ModuleType
from typing import Any, cast

_TEXT = 65536
_BINARY = 8 * 1024 * 1024
_PROGRAM = """module @source_target_metadata_query attributes {
  mhlo.num_partitions = 1 : i32,
  mhlo.num_replicas = 1 : i32
} {
  func.func public @main() {
    return
  }
}
"""
_SOURCES = {
    "0.10.2": (
        "f61ad41b2b3399e036f431ab1ac8446eecdc431c8de3fafb8fdcd37a03f963fb",
        "a26760846068a3c2f2449aad0ae84ec2d31856d27bcc5a1ed838c842e30cdf43",
    ),
    "0.12.0.1": (
        "2fd500e123543f69cd21540a87cd0a34c014ff9dd7a6cec71b2467235d2cef25",
        "78aa89c3a410591ff240bbb46e0cf7ab0b7c76773a9430fbfaf1fa9befc07986",
    ),
}
_CAST_SOURCES = {
    "0.10.2": "aa44c9d0532e9b939d6b0e8d3fbe46fd6a6cb7b1fdaa1e072e6e611fab9919b1",
    "0.12.0.1": "017f9cbce9fb78726431b883ede7310ec91dcc06daebf03ee11384bef24c0c31",
}


def _backend_function(bound: Any) -> FunctionType:
    import jax

    if jax.__version__ == "0.10.2":
        cache_type = type(functools.lru_cache(maxsize=None)(str))
        _require(type(bound) is cache_type, "native backend cache wrapper differs")
        function = vars(bound).get("__wrapped__")
    else:
        from jax._src.lib import weakref_lru_cache

        cache_type = weakref_lru_cache.StrongLRUCache
        _require(
            type(bound) is cache_type, "native strong backend cache wrapper differs"
        )
        descriptor = vars(cache_type).get("__wrapped__")
        _require(
            type(descriptor) is property
            and type(descriptor.fget).__module__ == "nanobind",
            "native strong cache getter differs",
        )
        function = descriptor.__get__(bound, cache_type)
    _require(type(function) is FunctionType, "native backend wrapped function differs")
    return cast(FunctionType, function)


def _native_global_reads(code: CodeType) -> tuple[str, ...]:
    """Include genuine nested native code without traversing foreign constants."""
    from jax2onnx._source_callable_preparation import _global_reads

    pending = [(code, 0)]
    seen: set[int] = set()
    names: dict[str, None] = {}
    size = 0
    while pending:
        current, depth = pending.pop()
        _require(
            type(current) is CodeType and depth <= 32,
            "native code nesting exceeds bound",
        )
        if id(current) in seen:
            continue
        seen.add(id(current))
        size += len(current.co_code)
        _require(
            len(seen) <= 1024 and size <= 2 * 1024 * 1024,
            "native code extent exceeds bound",
        )
        for name in _global_reads(current):
            names[name] = None
        _require(len(names) <= 4096, "native global-read set exceeds bound")
        pending.extend(
            (value, depth + 1)
            for value in reversed(current.co_consts)
            if type(value) is CodeType
        )
    return tuple(names)


def _pin_function(module: ModuleType, name: str, *, cached: bool = False) -> Any:
    """Compile pinned physical source code constants, never invoke a body."""
    from jax2onnx._source_callable_preparation import (
        _FunctionState,
        _compile_body,
        _code_contract,
        _MISSING,
    )

    bound = vars(module).get(name)
    function = bound
    if cached:
        function = _backend_function(bound)
    _require(
        type(function) is FunctionType and function.__globals__ is vars(module),
        "native Python function binding differs",
    )
    function = cast(FunctionType, function)
    filename = vars(module).get("__file__")
    _require(type(filename) is str, "native module source path unavailable")
    source = Path(cast(str, filename))
    _require(
        function.__code__.co_filename == str(source),
        "native function source path differs",
    )
    with source.open("rb") as stream:
        text = stream.read(2 * 1024 * 1024 + 1)
    _require(len(text) <= 2 * 1024 * 1024, "native Python source exceeds bound")
    definitions = [
        node
        for node in ast.parse(text).body
        if type(node) is ast.FunctionDef and node.name == name
    ]
    _require(len(definitions) == 1, "native source definition differs")
    compiled = _compile_body(function, definitions[0], str(source))
    if _code_contract(compiled) != _code_contract(function.__code__):
        imported = tuple(
            key
            for key in _native_global_reads(function.__code__)
            if type(function.__globals__.get(key)) is ModuleType
        )
        compiled = _compile_body(
            function, definitions[0], str(source), imported_globals=imported
        )
    _require(
        _code_contract(compiled) == _code_contract(function.__code__),
        "native live/source code differs",
    )
    state = _FunctionState.capture(function, watch_globals=True)
    # The generic preparation helper intentionally watches its own supported
    # user-body grammar. Native constructors can contain genexprs/lambdas, so
    # retain those extra actual global bindings in this native-only adapter.
    return replace(
        state,
        globals=tuple(
            (key, function.__globals__.get(key, _MISSING))
            for key in _native_global_reads(function.__code__)
        ),
    )


class NativeCastPins:
    def __init__(self) -> None:
        import jax
        from jax._src.lax import lax
        from jax._src import dtypes
        from .jax import Primitive

        _require(jax.__version__ in _CAST_SOURCES, "unaudited native conversion epoch")
        self._source = Path(lax.__file__)
        self._identity = _file(self._source)
        _require(
            self._identity[1] == _CAST_SOURCES[jax.__version__],
            "native conversion source differs",
        )
        self._states = [
            _pin_function(lax, name)
            for name in ("convert_element_type", "_convert_element_type")
        ]
        self._states.append(_pin_function(dtypes, "check_and_canonicalize_user_dtype"))
        primitive = lax.convert_element_type_p
        _require(
            type(primitive) is Primitive
            and jax.lax.convert_element_type is lax.convert_element_type,
            "native public conversion or primitive differs",
        )
        self._members = [
            (vars(jax.lax), "convert_element_type", lax.convert_element_type),
            (vars(lax), "_convert_element_type", lax._convert_element_type),
            (vars(lax), "convert_element_type_p", primitive),
            (vars(lax), "dtypes", dtypes),
            (
                vars(dtypes),
                "check_and_canonicalize_user_dtype",
                dtypes.check_and_canonicalize_user_dtype,
            ),
        ]
        primitive_storage = vars(primitive)
        self._primitive = (primitive_storage, tuple(dict.items(primitive_storage)))

    def assert_current(self) -> None:
        _require(
            _file(self._source) == self._identity, "native conversion source changed"
        )
        for state in self._states:
            state.assert_current()
        _require(
            all(
                namespace.get(name) is expected
                for namespace, name, expected in self._members
            ),
            "native conversion module member changed",
        )
        namespace, entries = self._primitive
        actual = tuple(dict.items(namespace))
        _require(
            len(actual) == len(entries)
            and all(
                a is c and b is d
                for (a, b), (c, d) in zip(actual, entries, strict=True)
            ),
            "native conversion primitive implementation changed",
        )

    def close(self) -> None:
        self._states.clear()
        self._members.clear()
        self._primitive = ({}, ())


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file(path: Path) -> tuple[tuple[int, ...], str]:
    before = path.stat()
    keys = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")

    def identity(value: os.stat_result) -> tuple[int, ...]:
        return tuple(getattr(value, key) for key in keys)

    _require(
        stat.S_ISREG(before.st_mode) and before.st_size <= 512 * 1024 * 1024,
        "native provenance file exceeds bound",
    )
    digest, count = hashlib.sha256(), 0
    with path.open("rb") as stream:
        _require(
            identity(os.fstat(stream.fileno())) == identity(before),
            "native provenance open changed",
        )
        while block := stream.read(1024 * 1024):
            count += len(block)
            _require(count <= before.st_size, "native provenance file grew")
            digest.update(block)
        _require(
            count == before.st_size
            and identity(os.fstat(stream.fileno()))
            == identity(before)
            == identity(path.stat()),
            "native provenance read changed",
        )
    return identity(before), digest.hexdigest()


def _varint(data: bytes, offset: int) -> tuple[int, int]:
    start, value = offset, 0
    for index in range(10):
        _require(offset < len(data), "truncated protobuf varint")
        byte = data[offset]
        offset += 1
        _require(index != 9 or byte <= 1, "protobuf integer exceeds bound")
        value |= (byte & 127) << (7 * index)
        if byte < 128:
            _require(offset - start == 1 or byte != 0, "noncanonical protobuf integer")
            return value, offset
    raise ValueError("unterminated protobuf integer")


def _fields(data: bytes) -> dict[int, list[Any]]:
    _require(type(data) is bytes and len(data) <= _BINARY, "protobuf exceeds bound")
    result: dict[int, list[Any]] = {}
    offset, count = 0, 0
    while offset < len(data):
        tag, offset = _varint(data, offset)
        field, wire = tag >> 3, tag & 7
        count += 1
        _require(
            0 < field < 2**29 and count <= 4096,
            "protobuf field count or number differs",
        )
        value: Any
        if wire == 0:
            value, offset = _varint(data, offset)
        elif wire in (1, 2, 5):
            if wire == 2:
                size, offset = _varint(data, offset)
            else:
                size = 8 if wire == 1 else 4
            _require(size <= len(data) - offset, "truncated protobuf field")
            value = (
                data[offset : offset + size]
                if wire == 2
                else ("opaque-fixed", wire, data[offset : offset + size])
            )
            offset += size
        else:
            raise ValueError("unsupported protobuf wire type")
        # Relevant fields use only varint/length-delimited; fixed fields remain
        # opaque in complete options bytes and cannot supply interpreted proof.
        result.setdefault(field, []).append(value)
    return result


def _one(fields: dict[int, list[Any]], key: int, default: Any = None) -> Any:
    if key not in fields:
        return default
    _require(len(fields[key]) == 1, "duplicate singular protobuf field")
    return fields[key][0]


def _blob(
    fields: dict[int, list[Any]], key: int, default: bytes | None = None
) -> bytes:
    value = _one(fields, key, default)
    _require(type(value) is bytes, "protobuf message/string wire type differs")
    return cast(bytes, value)


def _target(payload: bytes) -> tuple[tuple[str, str], ...]:
    fields = _fields(payload)
    _require(set(fields) <= {1, 2, 3}, "unknown target field")
    result = []
    for key, name in ((1, "triple"), (2, "cpu"), (3, "features")):
        raw = _blob(fields, key, b"")
        _require(len(raw) <= _TEXT and b"\0" not in raw, "target text exceeds bound")
        result.append((name, raw.decode("utf-8", errors="strict")))
    _require(bool(result[0][1]), "target triple unavailable")
    return tuple(result)


def _retained_target(payload: bytes) -> bytes:
    outer = _fields(payload)
    _require(
        set(outer) <= {1, 2, 3, 4, 9}
        and _blob(outer, 2) == _blob(outer, 3) == b"cpu"
        and type(_one(outer, 4, 0)) is int
        and _one(outer, 4, 0) == 0,
        "retained CPU topology envelope differs",
    )
    wrapped = _fields(_blob(outer, 9))
    _require(
        set(wrapped) == {1, 2}
        and _blob(wrapped, 1) == b"type.googleapis.com/xla.CpuTopologyProto",
        "CPU topology Any differs",
    )
    cpu = _fields(_blob(wrapped, 2))
    _require(
        set(cpu) <= {1, 5} and len(cpu.get(1, [])) == 1,
        "CPU topology device membership differs",
    )
    return _blob(cpu, 5)


def _query_target(payload: bytes) -> tuple[bytes, int]:
    size, begin = _varint(payload, 0)
    _require(
        begin <= 5 and 0 < size <= _TEXT and begin + size < len(payload),
        "IFRT framing differs",
    )
    metadata = _fields(payload[begin : begin + size])
    _require(
        set(metadata) <= {1, 2, 5, 6, 7, 8, 9, 10}
        and _blob(metadata, 2) == b"pjrt_ifrt"
        and not metadata.get(5)
        and not metadata.get(6),
        "empty-query IFRT signature differs",
    )
    pjrt = _fields(payload[begin + size :])
    _require(set(pjrt) <= {1, 2, 3}, "query PJRT schema differs")
    cpu = _fields(_blob(pjrt, 1))
    _require(
        set(cpu) <= {1, 2, 3, 5, 6, 7, 8, 9, 10}
        and type(_one(cpu, 5)) is int
        and _one(cpu, 5) == 2,
        "query CPU mode differs",
    )
    sequence = _fields(_blob(cpu, 6))
    _require(
        not sequence and not cpu.get(7) and len(cpu.get(8, [])) <= 32,
        "empty query has thunks, constants or compiled symbols",
    )
    return _blob(cpu, 9), len(cpu.get(8, []))


def project_cpu_cast_options(payload: bytes) -> tuple[tuple[str, str], ...]:
    """A finite projection; complete bytes are retained separately."""
    _require(type(payload) is bytes and len(payload) <= _TEXT, "options exceed bound")
    options = _fields(payload)
    build = _fields(_blob(options, 3))
    debug = _fields(_blob(build, 3))
    fast_math = _one(debug, 99, 0)
    _require(type(fast_math) is int and fast_math in (0, 1), "fast-math field differs")
    _require(
        fast_math == 0
        and not debug.get(30)
        and not debug.get(124)
        and not debug.get(500)
        and not options.get(7)
        and not _blob(build, 14, b""),
        "nondefault compiler overrides unavailable",
    )
    # Honor-inf/nan/division/function flags are irrelevant when fast_math=false.
    return (
        ("options_sha256", _sha(payload)),
        ("options_scope", "acquisition-construction-not-original-source"),
        ("options_fast_math", "false"),
        ("options_custom_passes", "empty"),
        ("options_backend_extra_options", "empty"),
        ("options_env_overrides", "empty"),
        ("original_logging", "unavailable"),
        ("original_resolved_shardings", "unavailable"),
    )


def _validate_program(ir: Any, module: Any) -> None:
    _require(type(module) is ir.Module, "query MLIR module type differs")
    root = module.operation
    _require(
        root.name == "builtin.module"
        and len(root.regions) == 1
        and len(root.regions[0].blocks) == 1
        and not root.operands
        and not root.results,
        "query module structure differs",
    )
    attrs = root.attributes
    _require(
        len(attrs) == 3
        and all(
            name in attrs
            for name in ("sym_name", "mhlo.num_partitions", "mhlo.num_replicas")
        )
        and ir.StringAttr(attrs["sym_name"]).value == "source_target_metadata_query",
        "query module attributes differ",
    )
    for name in ("mhlo.num_partitions", "mhlo.num_replicas"):
        attr = ir.IntegerAttr(attrs[name])
        _require(attr.value == 1 and str(attr.type) == "i32", "query partitions differ")
    functions = list(itertools.islice(module.body.operations, 2))
    _require(len(functions) == 1, "query function count differs")
    fn = functions[0].operation
    attrs = fn.attributes
    _require(
        fn.name == "func.func"
        and len(fn.regions) == 1
        and len(fn.regions[0].blocks) == 1
        and not fn.operands
        and not fn.results
        and len(attrs) == 3
        and all(
            name in attrs for name in ("sym_name", "sym_visibility", "function_type")
        )
        and ir.StringAttr(attrs["sym_name"]).value == "main"
        and ir.StringAttr(attrs["sym_visibility"]).value == "public",
        "query function differs",
    )
    signature = ir.FunctionType(ir.TypeAttr(attrs["function_type"]).value)
    _require(not signature.inputs and not signature.results, "query signature differs")
    block = fn.regions[0].blocks[0]
    operations = list(itertools.islice(block.operations, 2))
    _require(not block.arguments and len(operations) == 1, "query body differs")
    returned = operations[0].operation
    _require(
        returned.name == "func.return"
        and not returned.operands
        and not returned.results
        and not returned.regions
        and not returned.attributes
        and module.operation.verify(),
        "query return/verification differs",
    )


@dataclass(frozen=True, slots=True)
class CpuCastMetadata:
    fields: tuple[tuple[str, str], ...]
    options: bytes
    target_payload: bytes


class CpuCastContext:
    def __init__(self) -> None:
        import jax
        import jaxlib
        from jax._src import (
            compiler,
            mesh,
            core,
            config,
            xla_bridge,
            compilation_cache,
            lib,
        )
        from jax._src.interpreters import pxla
        from jaxlib import xla_client as xc
        from jaxlib import _jax as native
        import numpy as np
        from .jax import native_context_snapshot

        _require(
            jax.__version__ == jaxlib.__version__ and jax.__version__ in _SOURCES,
            "unaudited JAX/JAXLIB pair",
        )
        backend_state = _pin_function(xla_bridge, "get_backend", cached=True)
        uncached_state = _pin_function(xla_bridge, "_get_backend_uncached")
        # The authenticated explicit 'cpu' branch never reads this global.
        # Initialization itself legitimately assigns it. Every other captured
        # function attribute/global remains under the generic unchanged fence.
        _require(
            sum(name == "_default_backend" for name, _ in uncached_state.globals) == 1,
            "native backend initialization contract differs",
        )
        uncached_state = replace(
            uncached_state,
            globals=tuple(
                entry
                for entry in uncached_state.globals
                if entry[0] != "_default_backend"
            ),
        )
        self._backend_function: Any = backend_state.function
        self._backend: Any = xla_bridge.get_backend
        self._backend_namespace: Any = vars(xla_bridge)
        self._backend_default_initial: Any = vars(xla_bridge).get("_default_backend")
        self._backend_default_seen: Any = self._backend_default_initial
        self._backend_client_type: Any = native.Client
        self._backend_platform: Any = vars(native.Client).get("platform")
        _require(
            self._backend_platform is not None
            and type(vars(xla_bridge).get("_backends")) is dict,
            "native backend registry unavailable",
        )
        self._backend_registry: Any = vars(xla_bridge)["_backends"]
        self._states = [
            _pin_function(core, "trace_state_clean"),
            _pin_function(mesh, "get_abstract_mesh"),
            _pin_function(mesh, "get_concrete_mesh"),
            backend_state,
            uncached_state,
            _pin_function(compilation_cache, "is_persistent_cache_enabled"),
        ]
        self._members = [
            (vars(compiler), "xc", xc),
            (vars(compiler), "np", np),
            (vars(compiler), "config", config),
            (vars(compiler), "compilation_cache", compilation_cache),
            (vars(compiler), "lib", lib),
            (vars(xc), "CompileOptions", native.CompileOptions),
            (vars(xc), "Client", native.Client),
            (vars(xc), "DeviceAssignment", native.DeviceAssignment),
            (vars(xla_bridge), "get_backend", xla_bridge.get_backend),
            (
                vars(compilation_cache),
                "is_persistent_cache_enabled",
                compilation_cache.is_persistent_cache_enabled,
            ),
        ]
        _require(
            all(
                namespace.get(name) is expected
                for namespace, name, expected in self._members
            ),
            "native invoked constructor globals differ",
        )
        _require(
            core.trace_state_clean()
            and jax.config.x64_enabled
            and mesh.get_abstract_mesh().empty
            and mesh.get_concrete_mesh().empty,
            "ordinary clean x64 CPU context unavailable",
        )
        self._closed = False
        self._used = False
        self._acquired_client_identity: int | None = None
        self._profiles: Any = (sys.getprofile(), threading.getprofile())
        self._environment = dict(os.environ)
        _require(
            len(self._environment) <= 2048
            and sum(len(k) + len(v) for k, v in self._environment.items()) <= _TEXT,
            "environment exceeds bound",
        )
        self._snapshot: Any = native_context_snapshot()
        self._functions: list[tuple[dict[str, Any], str, FunctionType, CodeType]] = []
        self._sources = {
            Path(compiler.__file__): _file(Path(compiler.__file__)),
            Path(pxla.__file__): _file(Path(pxla.__file__)),
        }
        _require(
            tuple(value[1] for value in self._sources.values())
            == _SOURCES[jax.__version__],
            "installed compiler source differs",
        )
        for module, function_names in (
            (compiler, ("get_compile_options", "get_latest_profile_version")),
            (pxla, ("create_compile_options",)),
        ):
            for name in function_names:
                function = vars(module).get(name)
                _require(
                    type(function) is FunctionType
                    and function.__globals__ is vars(module),
                    "native Python constructor binding differs",
                )
                self._functions.append(
                    (vars(module), name, function, function.__code__)
                )
                self._states.append(_pin_function(module, name))
        from jax2onnx._source_callable_preparation import (
            _compile_body,
            _code_contract,
        )

        for _, _, function, code in self._functions:
            # The inspected native constructors intentionally write native
            # option fields and contain a lambda. The callable-preparation
            # grammar must not be widened to admit these. Compile only this
            # already SHA-pinned module's exact direct definition constants.
            source_path = next(
                path
                for path in self._sources
                if str(path) == function.__globals__.get("__file__")
            )
            _require(
                function.__code__.co_filename == str(source_path),
                "native constructor source path differs",
            )
            text = source_path.read_text()
            definitions = [
                node
                for node in ast.parse(text).body
                if type(node) is ast.FunctionDef and node.name == function.__name__
            ]
            _require(
                len(definitions) == 1, "native constructor source definition differs"
            )
            definition = definitions[0]
            compiled = _compile_body(
                function, definition, function.__code__.co_filename
            )
            if _code_contract(compiled) != _code_contract(code):
                imported = tuple(
                    name
                    for name in _native_global_reads(function.__code__)
                    if type(function.__globals__.get(name)) is ModuleType
                )
                compiled = _compile_body(
                    function,
                    definition,
                    function.__code__.co_filename,
                    imported_globals=imported,
                )
            _require(
                _code_contract(compiled) == _code_contract(code),
                "native constructor live/source body differs",
            )
        with Path(compiler.__file__).open("rb") as stream:
            profile_source = stream.read(2 * 1024 * 1024 + 1)
        _require(
            len(profile_source) <= 2 * 1024 * 1024,
            "native profile source exceeds bound",
        )
        profile_definition = next(
            node
            for node in ast.parse(profile_source).body
            if type(node) is ast.FunctionDef
            and node.name == "get_latest_profile_version"
        )
        profile_args = profile_definition.args
        profile_body = profile_definition.body
        _require(
            not profile_definition.decorator_list
            and not profile_args.posonlyargs
            and len(profile_args.args) == 1
            and profile_args.args[0].arg == "backend"
            and not profile_args.kwonlyargs
            and profile_args.vararg is None
            and profile_args.kwarg is None
            and not profile_args.defaults
            and not profile_args.kw_defaults
            and len(profile_body) == 2
            and type(profile_body[0]) is ast.Delete
            and len(profile_body[0].targets) == 1
            and type(profile_body[0].targets[0]) is ast.Name
            and profile_body[0].targets[0].id == "backend"
            and type(profile_body[1]) is ast.Return
            and type(profile_body[1].value) is ast.UnaryOp
            and type(profile_body[1].value.op) is ast.USub
            and type(profile_body[1].value.operand) is ast.Constant
            and type(profile_body[1].value.operand.value) is int
            and profile_body[1].value.operand.value == 1,
            "replaced profile-version hook unavailable",
        )
        names: tuple[str, ...] = (
            "enable_pgle",
            "pgle_profiling_runs",
            "jax_xla_profile_version",
            "enable_remat_opt_pass",
            "optimization_level",
            "memory_fitting_level",
            "use_shardy_partitioner",
            "default_device",
            "compilation_cache_dir",
            "enable_compilation_cache",
        )
        if jax.__version__ == "0.10.2":
            names += ("exec_time_optimization_effort", "memory_fitting_effort")
        _require(
            all(type(vars(config).get(name)) is config.State for name in names)
            and type(compiler._DISABLE_MOST_OPTIMIZATIONS) is config.Flag,
            "native compiler configuration holders differ",
        )
        self._configs: tuple[tuple[Any, Any], ...] = tuple(
            (vars(config)[name], vars(config)[name].value) for name in names
        ) + (
            (
                compiler._DISABLE_MOST_OPTIMIZATIONS,
                compiler._DISABLE_MOST_OPTIMIZATIONS.value,
            ),
        )
        _require(
            not (config.enable_pgle.value and config.pgle_profiling_runs.value > 0)
            and config.jax_xla_profile_version.value <= 0
            and config.compilation_cache_dir.value is None
            and config.enable_remat_opt_pass.value
            and not compiler._DISABLE_MOST_OPTIMIZATIONS.value,
            "nondefault compilation pipeline unavailable",
        )
        self.version = jax.__version__

    def assert_current(self) -> None:
        from .jax import require_native_context_restored

        _require(
            not self._closed and dict(os.environ) == self._environment,
            "native acquisition environment changed",
        )
        self.require_profiles_current()
        require_native_context_restored(self._snapshot)
        for state in self._states:
            state.assert_current()
        _require(
            _backend_function(self._backend) is self._backend_function,
            "native backend wrapped function changed",
        )
        self._check_backend_initialization()
        _require(
            all(
                namespace.get(name) is expected
                for namespace, name, expected in self._members
            ),
            "native invoked constructor globals changed",
        )
        _require(
            all(
                namespace.get(name) is fn and fn.__code__ is code
                for namespace, name, fn, code in self._functions
            ),
            "native constructor binding changed",
        )
        _require(
            all(configuration.value is value for configuration, value in self._configs),
            "native compiler configuration changed",
        )
        _require(
            all(_file(path) == identity for path, identity in self._sources.items()),
            "native compiler source changed",
        )

    def _check_backend_initialization(self, selected: Any = None) -> None:
        namespace = self._backend_namespace
        current = dict.get(namespace, "_default_backend")
        _require(
            dict.get(namespace, "_backends") is self._backend_registry
            and vars(self._backend_client_type).get("platform")
            is self._backend_platform,
            "native backend registry/platform changed",
        )
        if self._backend_default_seen is not None:
            _require(
                current is self._backend_default_seen,
                "native default backend identity changed",
            )
        elif current is not None:
            _require(
                type(current) is self._backend_client_type
                and dict.get(self._backend_registry, "cpu") is current
                and self._backend_platform.__get__(current, self._backend_client_type)
                == "cpu",
                "native first-backend transition is not registered CPU",
            )
            self._backend_default_seen = current
        elif selected is not None:
            raise ValueError("native CPU initialization did not establish a default")
        if self._backend_default_initial is None and selected is not None:
            _require(
                current is selected,
                "initialized CPU default differs from acquired client",
            )

    def require_profiles_current(self) -> None:
        _require(
            self._profiles is not None
            and sys.getprofile() is self._profiles[0]
            and threading.getprofile() is self._profiles[1],
            "ambient profile identities changed",
        )

    def acquire(self) -> CpuCastMetadata:
        import jax
        from jax._src import compiler, xla_bridge
        from jax._src.interpreters import mlir
        from jaxlib import _jax as native
        from jaxlib import xla_client as xc
        from jaxlib.mlir import ir
        import numpy as np

        self.assert_current()
        _require(not self._used, "native acquisition is single-use")
        self._used = True
        self.require_profiles_current()
        _require(
            xc.Client is native.Client
            and xc.CompileOptions is native.CompileOptions
            and xc.DeviceList is native.DeviceList,
            "native aliases differ",
        )
        methods = {}
        for cls, names in (
            (
                native.Client,
                (
                    "devices",
                    "local_devices",
                    "live_executables",
                    "compile_and_load",
                    "serialize_executable",
                ),
            ),
            (native.CompileOptions, ("SerializeAsString",)),
        ):
            for name in names:
                method = vars(cls).get(name)
                _require(
                    type(method).__module__ == "nanobind"
                    and type(method).__name__ in ("nb_method", "nb_func"),
                    "native method descriptor differs",
                )
                methods[(cls, name)] = method
        descriptors = {
            (cls, name): vars(cls).get(name)
            for cls, name in (
                (native.Client, "platform"),
                (native.Device, "client"),
                (native.LoadedExecutable, "client"),
            )
        }
        _require(
            all(value is not None for value in descriptors.values()),
            "native identity descriptors unavailable",
        )
        client = devices = local = executable = options = context = module = (
            topology
        ) = device_list = None
        query_identity: int | None = None

        def require_query_released() -> None:
            values = methods[(native.Client, "live_executables")](client)
            try:
                # The native API materializes its existing wrappers. Suite
                # history is not a new admission denominator: inspect only the
                # exact temporary query owner's absence, never clear caches or
                # require unrelated executables to remain alive after GC.
                _require(
                    type(values) is list
                    and all(type(value) is native.LoadedExecutable for value in values),
                    "live executable snapshot differs",
                )
                _require(
                    all(id(value) != query_identity for value in values),
                    "temporary query executable ownership not released",
                )
            finally:
                values = None

        try:
            client = xla_bridge.get_backend("cpu")
            self._check_backend_initialization(client)
            _require(
                type(client) is native.Client
                and descriptors[(native.Client, "platform")].__get__(
                    client, native.Client
                )
                == "cpu",
                "actual CPU client unavailable",
            )
            self._acquired_client_identity = id(client)
            devices = methods[(native.Client, "devices")](client)
            local = methods[(native.Client, "local_devices")](client)
            _require(
                type(devices) is list
                and type(local) is list
                and len(devices) == len(local) == 1
                and devices[0] is local[0]
                and type(devices[0]) is native.Device
                and descriptors[(native.Device, "client")].__get__(
                    devices[0], native.Device
                )
                is client,
                "singleton same-client CPU assignment unavailable",
            )
            from jax._src import config

            _require(
                config.default_device.value is None
                or config.default_device.value is devices[0],
                "default-device identity differs",
            )
            # This is an explicit acquisition logging choice, not a claim about
            # the unavailable original MLIR computation or its sharding flags.
            options = compiler.get_compile_options(
                1,
                1,
                device_assignment=np.array(devices, dtype=object).reshape((1, 1)),
                env_options_overrides={},
                fdo_profile=None,
                detailed_logging=False,
                backend=client,
            )
            _require(
                type(options) is native.CompileOptions,
                "native options constructor result differs",
            )
            options.parameter_is_tupled_arguments = False
            options.executable_build_options.allow_spmd_sharding_propagation_to_parameters = (
                []
            )
            options.executable_build_options.allow_spmd_sharding_propagation_to_output = (
                []
            )
            raw_options = methods[(native.CompileOptions, "SerializeAsString")](options)
            projection = project_cpu_cast_options(raw_options)
            serializer = vars(native.DeviceTopology).get("serialize")
            if serializer is not None:
                _require(
                    self.version == "0.12.0.1"
                    and type(serializer).__module__ == "nanobind"
                    and xc.get_topology_for_devices
                    is vars(native).get("get_topology_for_devices"),
                    "retained topology binding differs",
                )
                topology = xc.get_topology_for_devices(devices)
                _require(
                    type(topology) is native.DeviceTopology,
                    "retained topology type differs",
                )
                retained = serializer(topology)
                _require(
                    type(retained) is bytes and len(retained) <= _TEXT,
                    "retained topology exceeds bound",
                )
                target_payload = _retained_target(retained)
                acquisition = "retained-topology"
                work = "compile=0,serialize-executable=0,execute=0"
                objects = "not-queried"
            else:
                _require(self.version == "0.10.2", "unknown topology API unavailable")
                device_list = native.DeviceList(tuple(devices))
                context = ir.Context()
                context.append_dialect_registry(mlir.upstream_dialects)
                with context:
                    module = ir.Module.parse(_PROGRAM)
                    _validate_program(ir, module)
                    executable = methods[(native.Client, "compile_and_load")](
                        client,
                        module,
                        executable_devices=device_list,
                        compile_options=options,
                    )
                    query_identity = id(executable)
                    _require(
                        type(executable) is native.LoadedExecutable
                        and descriptors[(native.LoadedExecutable, "client")].__get__(
                            executable, native.LoadedExecutable
                        )
                        is client,
                        "empty-query executable/client differs",
                    )
                    _validate_program(ir, module)
                binary = methods[(native.Client, "serialize_executable")](
                    client, executable
                )
                _require(
                    type(binary) is bytes and 0 < len(binary) <= _BINARY,
                    "empty-query executable exceeds bound",
                )
                target_payload, object_count = _query_target(binary)
                acquisition = "same-client-empty-query"
                work = "compile=1,serialize-executable=1,execute=0"
                objects = str(object_count)
            _require(
                methods[(native.CompileOptions, "SerializeAsString")](options)
                == raw_options,
                "acquisition options mutated",
            )
            native_file = _file(Path(native.__file__))
            fields = (
                ("acquisition", acquisition),
                ("client_identity", str(id(client))),
                ("jax", jax.__version__),
                ("jaxlib", self.version),
                ("native_sha256", native_file[1]),
                ("native_build_authority", "unavailable-observed-file-digest-only"),
                ("work", work),
                ("query_object_count", objects),
                ("options_detailed_logging", "false-acquisition-choice"),
                ("options_propagation", "empty-query-ports-not-original-shardings"),
                *projection,
                *_target(target_payload),
            )
            self.assert_current()
            _require(
                all(
                    vars(cls).get(name) is method
                    for (cls, name), method in methods.items()
                )
                and all(
                    vars(cls).get(name) is descriptor
                    for (cls, name), descriptor in descriptors.items()
                ),
                "native acquisition bindings/profile changed",
            )
            self.require_profiles_current()
            return CpuCastMetadata(tuple(fields), raw_options, target_payload)
        finally:
            executable = options = module = context = topology = device_list = (
                devices
            ) = local = None
            gc.collect()
            try:
                if query_identity is not None:
                    require_query_released()
            finally:
                client = None
                self.require_profiles_current()

    def close(self) -> bool:
        held = (
            self._profiles is not None
            and sys.getprofile() is self._profiles[0]
            and threading.getprofile() is self._profiles[1]
        )
        self._functions.clear()
        self._states.clear()
        self._members.clear()
        self._sources.clear()
        self._configs = ()
        self._snapshot = None
        self._backend = self._backend_function = None
        self._backend_namespace = self._backend_registry = None
        self._backend_default_initial = self._backend_default_seen = None
        self._backend_client_type = self._backend_platform = None
        self._environment.clear()
        held = bool(
            held
            and sys.getprofile() is self._profiles[0]
            and threading.getprofile() is self._profiles[1]
        )
        self._profiles = None
        self._acquired_client_identity = None
        self._closed = True
        return held
