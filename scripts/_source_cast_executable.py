# scripts/_source_cast_executable.py

"""Bounded original-executable evidence; never compile or execute a replacement.

The ordinary observer supplies its original profile events. Native serialization
is deferred until the runner has durably recorded the original call outcome.
CPU serialization performs additional object linking and native allocation.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import marshal
import os
from pathlib import Path
import stat
import sys
import threading
import types
from typing import Any

NODE_ID = (
    "tests/extra_tests/converter/test_source_conversion_semantics.py::"
    "test_floating_conversion_rounding_and_strict_special_value_gates"
    "[ties-float16-float64]"
)
DIRECTORY = "source-f64-f16-cast-executables"
_TEST = "test_floating_conversion_rounding_and_strict_special_value_gates"
_MAX_TEXT = 64 * 1024
_MAX_BINARY = 8 * 1024 * 1024
_MAX_TOTAL = 16 * 1024 * 1024
_MAX_JSON = 256 * 1024


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _error(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__module__ + "." + type(error).__qualname__}


def _module(name: str) -> types.ModuleType:
    value = sys.modules.get(name)
    _require(type(value) is types.ModuleType, "required native module not loaded")
    return value


def _functions() -> dict[str, tuple[types.ModuleType, str, types.FunctionType]]:
    result = {}
    for role, module_name, name in (
        ("compiler", "jax._src.compiler", "compile_or_get_cached"),
        ("dispatch", "jax._src.pjit", "_pjit_call_impl_python"),
    ):
        module = _module(module_name)
        function = vars(module).get(name)
        _require(
            type(function) is types.FunctionType, "native function binding differs"
        )
        _require(function.__globals__ is vars(module), "native function globals differ")
        result[role] = (module, name, function)
    return result


def _native_types() -> dict[str, type]:
    native = _module("jaxlib._jax")
    pxla = _module("jax._src.interpreters.pxla")
    result = {
        "executable": vars(native).get("LoadedExecutable"),
        "client": vars(native).get("Client"),
        "jit": vars(native).get("PjitFunction"),
        "mesh": vars(pxla).get("MeshExecutable"),
        "executor": vars(pxla).get("ExecuteReplicated"),
    }
    _require(
        all(isinstance(value, type) for value in result.values()),
        "native types unavailable",
    )
    return result


def _execution_methods(native: dict[str, Any]) -> dict[str, Any]:
    """Pin the two known native Python adapters without calling either."""
    pxla, profiler = _module("jax._src.interpreters.pxla"), _module("jax._src.profiler")
    _require(
        vars(pxla).get("MeshExecutable") is native["mesh"]
        and vars(pxla).get("ExecuteReplicated") is native["executor"],
        "native executor classes differ",
    )
    getter = vars(native["mesh"]).get("unsafe_call")
    wrapper = vars(native["executor"]).get("__call__")
    annotator = vars(profiler).get("annotate_function")
    _require(
        type(getter) is property
        and type(getter.fget) is types.FunctionType
        and getter.fget.__globals__ is vars(pxla),
        "native unsafe-call property differs",
    )
    _require(
        type(wrapper) is types.FunctionType
        and type(annotator) is types.FunctionType
        and wrapper.__globals__ is annotator.__globals__ is vars(profiler),
        "native executor profiler wrapper differs",
    )
    _require(
        any(value is wrapper.__code__ for value in annotator.__code__.co_consts)
        and wrapper.__code__.co_freevars == ("decorator_kwargs", "func", "name"),
        "native executor wrapper code differs",
    )
    closure = wrapper.__closure__
    _require(
        type(closure) is tuple
        and len(closure) == 3
        and all(type(cell) is types.CellType for cell in closure),
        "native executor wrapper cells differ",
    )
    kwargs, body, name = (cell.cell_contents for cell in closure)
    _require(
        type(kwargs) is dict
        and not kwargs
        and type(body) is types.FunctionType
        and body.__globals__ is vars(pxla)
        and body.__qualname__ == "ExecuteReplicated.__call__"
        and wrapper.__dict__.get("__wrapped__") is body
        and type(name) is str
        and name == "ExecuteReplicated.__call__",
        "native executor wrapper target differs",
    )
    return {
        "pxla": pxla,
        "profiler": profiler,
        "property": getter,
        "getter": getter.fget,
        "wrapper": wrapper,
        "annotator": annotator,
        "body": body,
        "closure": closure,
        "kwargs": kwargs,
        "name": name,
    }


def _methods_held(methods: dict[str, Any], native: dict[str, Any]) -> bool:
    pxla, profiler = methods["pxla"], methods["profiler"]
    closure, wrapper = methods["closure"], methods["wrapper"]
    return (
        sys.modules.get("jax._src.interpreters.pxla") is pxla
        and sys.modules.get("jax._src.profiler") is profiler
        and vars(pxla).get("MeshExecutable") is native["mesh"]
        and vars(pxla).get("ExecuteReplicated") is native["executor"]
        and vars(native["mesh"]).get("unsafe_call") is methods["property"]
        and methods["property"].fget is methods["getter"]
        and vars(native["executor"]).get("__call__") is wrapper
        and vars(profiler).get("annotate_function") is methods["annotator"]
        and wrapper.__closure__ is closure
        and closure[0].cell_contents is methods["kwargs"]
        and not methods["kwargs"]
        and closure[1].cell_contents is methods["body"]
        and closure[2].cell_contents is methods["name"]
        and wrapper.__dict__.get("__wrapped__") is methods["body"]
    )


def _program(computation: Any) -> dict[str, Any]:
    """Inspect the actual compiler input, using only loaded native MLIR APIs."""
    ir = _module("jaxlib.mlir.ir")
    _require(
        type(computation) is vars(ir).get("Module"), "compiler input is not native MLIR"
    )
    functions = list(itertools.islice(computation.body.operations, 2))
    _require(len(functions) == 1, "cast module must contain one function")
    function = functions[0].operation
    _require(
        type(function) is vars(ir).get("Operation"), "native function operation differs"
    )
    _require(function.name == "func.func", "cast module function differs")
    _require(
        len(function.regions) == 1 and len(function.regions[0].blocks) == 1,
        "cast function regions differ",
    )
    block = function.regions[0].blocks[0]
    _require(len(block.arguments) == 1, "cast input count differs")
    operations = list(itertools.islice(block.operations, 3))
    _require(len(operations) == 2, "cast function must have only conversion and return")
    convert, returned = (value.operation for value in operations)
    _require(
        type(convert) is vars(ir).get("Operation")
        and type(returned) is vars(ir).get("Operation"),
        "cast operation classes differ",
    )
    _require(
        convert.name == "stablehlo.convert" and returned.name == "func.return",
        "cast operations differ",
    )
    _require(
        len(convert.attributes) == 0 and len(convert.regions) == 0,
        "cast has extra attributes or regions",
    )
    _require(
        len(convert.operands) == len(convert.results) == len(returned.operands) == 1
        and len(returned.results)
        == len(returned.regions)
        == len(returned.attributes)
        == 0,
        "cast ports differ",
    )
    _require(
        convert.operands[0] == block.arguments[0]
        and returned.operands[0] == convert.results[0],
        "cast SSA edges differ",
    )
    _require(
        str(block.arguments[0].type) == "tensor<6xf64>"
        and str(convert.results[0].type) == "tensor<6xf16>",
        "cast typed program differs",
    )
    text = str(computation)
    _require(
        type(text) is str and len(text.encode()) <= _MAX_TEXT, "native IR exceeds bound"
    )
    return {
        "input": {"dtype": "float64", "shape": [6]},
        "output": {"dtype": "float16", "shape": [6]},
        "operations": ["stablehlo.convert", "func.return"],
        "ir": text,
        "ir_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "source": "Actual original compile_or_get_cached computation argument",
    }


def _source(path: Path) -> dict[str, Any]:
    fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    before = path.lstat()
    _require(
        stat.S_ISREG(before.st_mode)
        and path.resolve(strict=True) == path
        and before.st_size <= 2 * 1024 * 1024,
        "diagnostic source path or size differs",
    )
    descriptor = os.open(
        path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    )
    try:
        opened = os.fstat(descriptor)
        _require(
            all(getattr(before, name) == getattr(opened, name) for name in fields),
            "diagnostic source identity changed before read",
        )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read(2 * 1024 * 1024 + 1)
        after, end = os.fstat(descriptor), path.lstat()
        _require(
            len(data) <= 2 * 1024 * 1024
            and all(
                getattr(before, name) == getattr(after, name) == getattr(end, name)
                for name in fields
            )
            and len(data) == before.st_size,
            "diagnostic source changed or exceeded bound",
        )
    finally:
        os.close(descriptor)
    return {
        "path": str(path.resolve()),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _function_record(function: types.FunctionType) -> dict[str, Any]:
    code = function.__code__
    payload = marshal.dumps(code)
    _require(len(payload) <= _MAX_TEXT, "native Python code exceeds bound")
    return {
        "function_identity": id(function),
        "code_identity": id(code),
        "marshal_sha256": hashlib.sha256(payload).hexdigest(),
        "filename": code.co_filename,
        "firstlineno": code.co_firstlineno,
        "name": code.co_name,
    }


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class _Capture:
    def __init__(
        self,
        folder: Path,
        owner: types.ModuleType,
        artifact: types.FunctionType,
        array_record: Any,
    ) -> None:
        self.folder = folder / DIRECTORY
        self.array_record = array_record
        self.owner: Any = owner
        self.artifact: Any = artifact
        self.reference: Any = None
        self.test: Any = None
        self.bindings: dict[str, Any] = {}
        self.codes: dict[str, types.CodeType] = {}
        self.methods: dict[str, Any] = {}
        self.native: dict[str, Any] = {}
        self.source_function: Any = None
        self.source_code: Any = None
        self.source_closure: Any = None
        self.sources: dict[str, Any] = {}
        self.roles: list[dict[str, Any]] = []
        self.artifact_identity: int | None = None
        self.stopped = False
        self.closed = False
        self.owned_folder = False
        self.report: dict[str, Any] = {
            "schema": "source-cast-original-executables-v1",
            "nodeid": NODE_ID,
            "status": "UNVERIFIED",
            "roles": [],
            "errors": [],
            "source_or_numeric_calls_added": False,
            "replacement_compilation_or_execution": False,
            "serialization_scope": "Original stored object export; additional native linking and executable-memory allocation, not original live relocation addresses",
            "serialization_attempts": 0,
            "native_owners_cleared": False,
            "bounds": {
                "roles": 2,
                "ancestor_frames": 64,
                "text_bytes": _MAX_TEXT,
                "binary_bytes_per_role": _MAX_BINARY,
                "binary_bytes_total": _MAX_TOTAL,
                "json_bytes": _MAX_JSON,
            },
        }

    def _fail(
        self, stage: str, error: BaseException, role: dict[str, Any] | None = None
    ) -> None:
        target = self.report if role is None else role["record"]
        if len(target.setdefault("errors", [])) < 16:
            target["errors"].append({"stage": stage, **_error(error)})
        target["status"] = "UNVERIFIED"
        if role is not None:
            role.pop("executable", None)
            role.pop("client", None)

    def _write(self) -> None:
        if not self.owned_folder:
            return
        payload = (
            json.dumps(self.report, indent=2, ensure_ascii=False) + "\n"
        ).encode()
        _require(len(payload) <= _MAX_JSON, "executable report exceeds bound")
        temporary = self.folder / "report.json.tmp"
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.folder / "report.json")
        _sync_directory(self.folder)

    def start(self) -> None:
        _require(
            type(self.owner) is types.ModuleType
            and type(self.artifact) is types.FunctionType,
            "original owner or artifact binding differs",
        )
        self.test = vars(self.owner).get(_TEST)
        self.reference = self.artifact.__globals__.get("_reference")
        _require(
            type(self.test) is types.FunctionType
            and type(self.reference) is types.FunctionType,
            "original test/reference binding differs",
        )
        _require(
            self.test.__globals__ is vars(self.owner), "original test globals differ"
        )
        self.bindings = _functions()
        self.native = _native_types()
        self.native["client_descriptor"] = vars(self.native["executable"]).get("client")
        self.native["serializer"] = vars(self.native["client"]).get(
            "serialize_executable"
        )
        self.native["platform_descriptor"] = vars(self.native["client"]).get("platform")
        self.native["mesh_executable_slot"] = vars(self.native["mesh"]).get(
            "xla_executable"
        )
        for key, cls, name in (
            ("cached_call_slot", self.native["mesh"], "_unsafe_call"),
            ("executor_executable_slot", self.native["executor"], "xla_executable"),
            ("executor_backend_slot", self.native["executor"], "backend"),
        ):
            descriptor = vars(cls).get(name)
            _require(
                type(descriptor) is types.MemberDescriptorType,
                "native executor slot differs",
            )
            self.native[key] = descriptor
        _require(
            self.native["client_descriptor"] is not None
            and self.native["serializer"] is not None
            and self.native["platform_descriptor"] is not None,
            "native serializer/platform binding unavailable",
        )
        _require(
            type(self.native["mesh_executable_slot"]) is types.MemberDescriptorType,
            "native mesh executable slot differs",
        )
        self.report["native_type_provenance"] = {
            key: {
                "module": value.__module__,
                "name": value.__qualname__,
                "identity": id(value),
            }
            for key, value in self.native.items()
            if isinstance(value, type)
        }
        self.report["native_descriptor_identities"] = {
            key: id(self.native[key])
            for key in (
                "client_descriptor",
                "serializer",
                "platform_descriptor",
                "mesh_executable_slot",
                "cached_call_slot",
                "executor_executable_slot",
                "executor_backend_slot",
            )
        }
        self.methods = _execution_methods(self.native)
        self.bindings.update(
            {
                "test": (self.owner, _TEST, self.test),
                "artifact": (self.owner, "_artifact", self.artifact),
                "reference": (self.artifact.__globals__, "_reference", self.reference),
            }
        )
        for role, (_, _, function) in self.bindings.items():
            self.codes[role] = function.__code__
            self.report.setdefault("function_provenance", {})[role] = _function_record(
                function
            )
        for role in ("getter", "wrapper", "annotator", "body"):
            function = self.methods[role]
            self.codes[role] = function.__code__
            self.report["function_provenance"]["execution_" + role] = _function_record(
                function
            )
        paths = {
            Path(function.__code__.co_filename).resolve()
            for _, _, function in self.bindings.values()
        }
        paths.update(
            Path(self.methods[role].__code__.co_filename).resolve()
            for role in ("getter", "wrapper", "annotator", "body")
        )
        paths.add(Path(__file__).resolve())
        self.sources = {str(path): _source(path) for path in sorted(paths)}
        self.report["source_files_before"] = self.sources
        self.folder.mkdir()
        self.owned_folder = True
        _sync_directory(self.folder.parent)
        self._write()

    def _held(self) -> bool:
        for role, (module, name, function) in self.bindings.items():
            namespace = module if type(module) is dict else vars(module)
            if (
                namespace.get(name) is not function
                or function.__code__ is not self.codes[role]
            ):
                return False
            if (
                type(module) is types.ModuleType
                and sys.modules.get(module.__name__) is not module
            ):
                return False
        if self.source_function is not None:
            function = self.source_function
            if not (
                function.__code__ is self.source_code
                and function.__globals__ is vars(self.owner)
                and function.__closure__ is self.source_closure
                and function.__defaults__ is None
                and function.__kwdefaults__ is None
                and type(self.source_closure[0].cell_contents) is str
                and self.source_closure[0].cell_contents == "float16"
            ):
                return False
        if not _methods_held(self.methods, self.native) or any(
            self.methods[role].__code__ is not self.codes[role]
            for role in ("getter", "wrapper", "annotator", "body")
        ):
            return False
        return (
            vars(self.native["executable"]).get("client")
            is self.native["client_descriptor"]
            and vars(self.native["client"]).get("serialize_executable")
            is self.native["serializer"]
            and vars(self.native["client"]).get("platform")
            is self.native["platform_descriptor"]
            and vars(self.native["mesh"]).get("xla_executable")
            is self.native["mesh_executable_slot"]
            and vars(self.native["mesh"]).get("_unsafe_call")
            is self.native["cached_call_slot"]
            and vars(self.native["executor"]).get("xla_executable")
            is self.native["executor_executable_slot"]
            and vars(self.native["executor"]).get("backend")
            is self.native["executor_backend_slot"]
        )

    def _ancestors(self, frame: Any) -> list[Any]:
        result = []
        for _ in range(64):
            if frame is None:
                return result
            result.append(frame)
            frame = frame.f_back
        return result

    def _reference_start(self, frame: Any) -> None:
        _require(
            frame.f_globals is self.reference.__globals__,
            "original reference globals differ",
        )
        parents = self._ancestors(frame.f_back)
        artifacts = [
            parent for parent in parents if parent.f_code is self.artifact.__code__
        ]
        tests = [parent for parent in parents if parent.f_code is self.test.__code__]
        _require(
            len(artifacts) == len(tests) == 1, "original reference ancestry differs"
        )
        artifact, test = artifacts[0], tests[0]
        _require(
            artifact.f_globals is self.artifact.__globals__
            and test.f_globals is self.test.__globals__,
            "original frame globals differ",
        )
        if self.artifact_identity is None:
            self.artifact_identity = id(artifact)
        _require(
            id(artifact) == self.artifact_identity,
            "later artifact cannot substitute for original",
        )
        if len(self.roles) >= 2:
            return  # Activated/export-related references are deliberately excluded.
        local = test.f_locals
        parameters = tuple(local.get(name) for name in ("source", "target", "group"))
        _require(
            all(type(value) is str for value in parameters)
            and parameters == ("float64", "float16", "ties"),
            "original test parameters differ",
        )
        source_fn = artifact.f_locals.get("fn")
        _require(
            type(source_fn) is types.FunctionType and source_fn is local.get("fn"),
            "original callable identity differs",
        )
        if self.source_function is None:
            code, closure = source_fn.__code__, source_fn.__closure__
            _require(
                source_fn.__globals__ is vars(self.owner)
                and any(value is code for value in self.test.__code__.co_consts)
                and code.co_name == "fn"
                and code.co_freevars == ("target",)
                and type(closure) is tuple
                and len(closure) == 1
                and type(closure[0]) is types.CellType
                and type(closure[0].cell_contents) is str
                and closure[0].cell_contents == "float16"
                and source_fn.__defaults__ is None
                and source_fn.__kwdefaults__ is None,
                "original inner callable code/globals/target closure differs",
            )
            self.source_function, self.source_code, self.source_closure = (
                source_fn,
                code,
                closure,
            )
            self.report["original_callable_pin"] = {
                **_function_record(source_fn),
                "globals_identity": id(source_fn.__globals__),
                "closure_identity": id(closure),
                "target_cell_identity": id(closure[0]),
                "target": "float16",
            }
        _require(
            source_fn is self.source_function and self._held(),
            "original callable changed between reference roles",
        )
        feeds = artifact.f_locals.get("feeds")
        values = frame.f_locals.get("values")
        _require(
            type(feeds) in (list, tuple)
            and len(feeds) == 1
            and type(feeds[0]) is tuple
            and len(feeds[0]) == 1,
            "original feed grouping differs",
        )
        _require(
            values is feeds[0] and values[0] is local.get("value"),
            "original reference feed identity differs",
        )
        reference_fn = frame.f_locals.get("fn")
        if not self.roles:
            _require(reference_fn is source_fn, "first original reference is not eager")
        else:
            _require(
                type(reference_fn) is self.native["jit"],
                "second original reference is not native jit",
            )
            storage = object.__getattribute__(reference_fn, "__dict__")
            _require(
                type(storage) is dict and storage.get("_fun") is source_fn,
                "fresh jit source binding differs",
            )
        record: dict[str, Any] = {
            "role": "native_eager" if not self.roles else "native_fresh_jit",
            "status": "UNVERIFIED",
            "errors": [],
            "reference_frame_identity": id(frame),
            "artifact_frame_identity": id(artifact),
            "original_callable": _function_record(source_fn),
            "original_feed_identity": id(values[0]),
        }
        self.roles.append({"frame_identity": id(frame), "record": record})
        self.report["roles"].append(record)

    def _role(self, frame: Any) -> dict[str, Any] | None:
        identities = {id(parent) for parent in self._ancestors(frame)}
        return next(
            (
                role
                for role in self.roles
                if role["frame_identity"] in identities and not role.get("returned")
            ),
            None,
        )

    def observe(self, frame: Any, event: str, argument: Any) -> None:
        if self.stopped or self.closed or event not in ("call", "return"):
            return
        role = None
        try:
            if frame.f_code is self.reference.__code__ and event == "call":
                self._reference_start(frame)
                return
            if event != "return":
                return
            observed = (
                self.reference,
                self.bindings["compiler"][2],
                self.bindings["dispatch"][2],
            )
            function = next(
                (value for value in observed if frame.f_code is value.__code__), None
            )
            if function is None:
                return
            role = self._role(frame)
            if role is None:
                return
            _require(self._held(), "observed function or serializer binding changed")
            _require(
                frame.f_globals is function.__globals__,
                "observed native function globals differ",
            )
            if frame.f_code is self.bindings["compiler"][2].__code__:
                _require(
                    "executable" not in role and not role.get("compiler_seen"),
                    "ambiguous compiler returns in original reference",
                )
                role["compiler_seen"] = True
                _require(
                    type(argument) is self.native["executable"],
                    "compiler did not return exact native executable",
                )
                callbacks = frame.f_locals.get("host_callbacks")
                _require(
                    type(callbacks) in (tuple, list) and not callbacks,
                    "cast compiler has callbacks or unknown callback state",
                )
                client = self.native["client_descriptor"].__get__(
                    argument, self.native["executable"]
                )
                _require(
                    type(client) is self.native["client"]
                    and frame.f_locals.get("backend") is client,
                    "original native client identity differs",
                )
                platform = self.native["platform_descriptor"].__get__(
                    client, self.native["client"]
                )
                _require(
                    type(platform) is str and platform == "cpu",
                    "original executable client is not proven CPU",
                )
                program = _program(frame.f_locals.get("computation"))
                role["record"].update(
                    program=program,
                    executable_identity=id(argument),
                    client_identity=id(client),
                    client_platform=platform,
                )
                role["executable"], role["client"] = argument, client
            elif frame.f_code is self.bindings["dispatch"][2].__code__:
                _require(
                    type(argument) is tuple and len(argument) == 4,
                    "original dispatch return differs",
                )
                compiled = frame.f_locals.get("compiled")
                _require(
                    type(compiled) is self.native["mesh"] and argument[1] is compiled,
                    "original dispatch mesh identity differs",
                )
                constants = frame.f_locals.get("sharded_const_args")
                _require(
                    type(constants) in (tuple, list)
                    and not constants
                    and argument[3] is constants,
                    "cast dispatch has constants or unknown constant state",
                )
                executable = self.native["mesh_executable_slot"].__get__(
                    compiled, self.native["mesh"]
                )
                _require(
                    role.get("executable") is not None
                    and executable is role["executable"],
                    "compiled object was not the original dispatched executable",
                )
                executor = self.native["cached_call_slot"].__get__(
                    compiled, self.native["mesh"]
                )
                _require(
                    type(executor) is self.native["executor"],
                    "native cached executor differs",
                )
                executed = self.native["executor_executable_slot"].__get__(
                    executor, self.native["executor"]
                )
                backend = self.native["executor_backend_slot"].__get__(
                    executor, self.native["executor"]
                )
                _require(
                    executed is executable and backend is role["client"],
                    "cached executor executable/backend differs from original compiler return",
                )
                _require(
                    not role.get("dispatched"), "ambiguous original dispatch returns"
                )
                role["dispatched"] = True
                role["record"]["dispatch_mesh_identity"] = id(compiled)
                role["record"]["compiler_return_is_dispatched_executable"] = True
                role["record"]["cached_executor_identity"] = id(executor)
            elif frame.f_code is self.reference.__code__:
                role["returned"] = True
                _require(
                    type(argument) is list and len(argument) == 1,
                    "original reference output grouping differs",
                )
                value = self.array_record(argument[0])
                _require(
                    value["dtype"] == "float16"
                    and value["shape"] == [6]
                    and len(value["storage_hex"]) == 24,
                    "original reference typed words differ",
                )
                role["record"]["reference"] = value
                _require(
                    role.get("dispatched") is True,
                    "original compiler/dispatch frames unavailable",
                )
                _require(
                    not role["record"]["errors"], "original role has a provenance gap"
                )
                role["record"]["status"] = "ORIGINAL_DISPATCH_BOUND"
        except BaseException as error:
            self._fail("profile-" + event, error, role)

    def bind_artifact(self, records: dict[str, Any]) -> None:
        try:
            _require(len(self.roles) == 2, "two original reference roles unavailable")
            for role in self.roles:
                expected = records.get(role["record"]["role"])
                _require(
                    type(expected) is list
                    and len(expected) == 1
                    and expected[0] == role["record"].get("reference"),
                    "original artifact reference words or identity differ",
                )
                role["artifact_bound"] = True
                role["record"]["artifact_reference_held"] = True
        except BaseException as error:
            self._fail("artifact-reference-binding", error)

    def stop(self) -> None:
        self.stopped = True

    def after_report(self, phase: dict[str, Any]) -> None:
        try:
            _require(
                self.stopped
                and not self.closed
                and sys.getprofile() is None
                and threading.getprofile() is None,
                "serialization requires stopped original observer and absent profiles",
            )
            _require(
                type(phase) is dict
                and phase.get("nodeid") == NODE_ID
                and phase.get("when") == "call"
                and phase.get("outcome") in ("passed", "failed", "skipped")
                and phase.get("durable_phase_recorded") is True,
                "original durable call outcome unavailable",
            )
            self.report["original_call_phase"] = phase
            _require(self._held(), "native/code bindings changed before serialization")
            self._write()
            total = 0
            for role in self.roles:
                record = role["record"]
                if (
                    record["status"] != "ORIGINAL_DISPATCH_BOUND"
                    or not role.get("artifact_bound")
                    or self.report["errors"]
                ):
                    continue
                try:
                    _require(
                        self._held(),
                        "native/code bindings changed before role serialization",
                    )
                    platform = self.native["platform_descriptor"].__get__(
                        role["client"], self.native["client"]
                    )
                    _require(
                        type(platform) is str and platform == "cpu",
                        "original client CPU scope changed before serialization",
                    )
                    self.report["serialization_attempts"] += 1
                    record["serialization_attempted"] = True
                    record["additional_native_linking_and_allocation"] = True
                    self._write()
                    payload = self.native["serializer"](
                        role["client"], role["executable"]
                    )
                    _require(
                        type(payload) is bytes
                        and 0 < len(payload) <= _MAX_BINARY
                        and total + len(payload) <= _MAX_TOTAL,
                        "original executable bytes exceed bound or differ in type",
                    )
                    _require(
                        self._held(),
                        "native/code bindings changed during serialization",
                    )
                    platform = self.native["platform_descriptor"].__get__(
                        role["client"], self.native["client"]
                    )
                    _require(
                        type(platform) is str and platform == "cpu",
                        "original client CPU scope changed during serialization",
                    )
                    record["client_platform_end"] = platform
                    _require(
                        sys.getprofile() is None and threading.getprofile() is None,
                        "serialization changed profile ownership",
                    )
                    path = self.folder / (record["role"] + ".bin")
                    with path.open("xb") as stream:
                        stream.write(payload)
                        stream.flush()
                        os.fsync(stream.fileno())
                    total += len(payload)
                    record.update(
                        status="ORIGINAL_EXECUTABLE_SERIALIZED",
                        binary={
                            "path": path.name,
                            "bytes": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest(),
                        },
                    )
                except BaseException as error:
                    self._fail("serialization", error, role)
                finally:
                    role.pop("executable", None)
                    role.pop("client", None)
                    self._write()
            self.report["source_files_end"] = {
                path: _source(Path(path)) for path in self.sources
            }
            self.report["source_fence_held"] = (
                self.report["source_files_end"] == self.sources
            )
            self.report["code_binding_fence_held"] = self._held()
            self.report["profiles_remained_absent"] = (
                sys.getprofile() is None and threading.getprofile() is None
            )
            if (
                len(self.roles) == 2
                and all(
                    role["record"]["status"] == "ORIGINAL_EXECUTABLE_SERIALIZED"
                    for role in self.roles
                )
                and not self.report["errors"]
                and self.report["source_fence_held"]
                and self.report["code_binding_fence_held"]
                and self.report["profiles_remained_absent"]
            ):
                self.report["status"] = "ORIGINAL_EXECUTABLES_SERIALIZED"
        except BaseException as error:
            self._fail("post-report", error)
        finally:
            self.close()

    def close(self) -> None:
        for role in self.roles:
            role.pop("executable", None)
            role.pop("client", None)
        self.roles.clear()
        self.bindings.clear()
        self.codes.clear()
        self.methods.clear()
        self.native.clear()
        self.source_function = self.source_code = self.source_closure = None
        self.owner = self.artifact = self.reference = self.test = self.array_record = (
            None
        )
        self.closed = True
        self.stopped = True
        self.report["native_owners_cleared"] = True
        self._write()


def start_capture(
    folder: Path,
    owner: types.ModuleType,
    artifact: types.FunctionType,
    array_record: Any,
) -> _Capture:
    capture = _Capture(folder, owner, artifact, array_record)
    try:
        capture.start()
    except BaseException as error:
        capture._fail("start", error)
        capture.close()
    return capture
