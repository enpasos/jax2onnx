# scripts/_source_bf16_topk_capture.py

"""Retain one original finite-inline BF16 TopK call after its durable report.

This helper imports no framework, evaluates no model, and claims no numerical
acceptance. Missing evidence never changes the original pytest outcome.
"""

from __future__ import annotations

import ast
import base64
import csv
from email.parser import Parser
import hashlib
import importlib.metadata
import inspect
import io
import json
import marshal
import os
from pathlib import Path
import stat
import struct
import sys
import threading
import types
from typing import Any
import zipfile

OWNER = "tests/extra_tests/converter/test_source_native_top_k.py"
TEST = "test_source_bfloat16_top_k_portable_words_and_native_equivalence"
NODE_ID = OWNER + "::" + TEST + "[finite-inline]"
DIRECTORY = "source-bf16-topk-native"
SCHEMA = "source-bf16-topk-native-capture-v1"
_ROOT = Path(__file__).resolve().parents[1]
_OWNER_SHA256 = "1c40b158336f2b3b132085e36005ffb7dff9977adb2e8d6e43be785e219afabf"
_FORMALS = (
    "backend",
    "computation",
    "devices",
    "compile_options",
    "host_callbacks",
    "executable_devices",
    "pgle_profiler",
)
_COMPILER_SCHEMAS = {
    "f61ad41b2b3399e036f431ab1ac8446eecdc431c8de3fafb8fdcd37a03f963fb": (
        "a26760846068a3c2f2449aad0ae84ec2d31856d27bcc5a1ed838c842e30cdf43",
        "aa44c9d0532e9b939d6b0e8d3fbe46fd6a6cb7b1fdaa1e072e6e611fab9919b1",
    ),
    "2fd500e123543f69cd21540a87cd0a34c014ff9dd7a6cec71b2467235d2cef25": (
        "78aa89c3a410591ff240bbb46e0cf7ab0b7c76773a9430fbfaf1fa9befc07986",
        "017f9cbce9fb78726431b883ede7310ec91dcc06daebf03ee11384bef24c0c31",
    ),
}
_SOURCES = (
    OWNER,
    "scripts/run_pytest_batches.py",
    "scripts/_source_bf16_topk_capture.py",
    "jax2onnx/_compat/jax.py",
    "jax2onnx/precision.py",
    "jax2onnx/plugins/jax/lax/top_k.py",
    "jax2onnx/converter/source_top_k.py",
    "jax2onnx/converter/source_profile.py",
    "tests/extra_tests/converter/test_source_model_native_bindings.py",
    "tests/extra_tests/converter/test_source_mlp_runtime_acceptance.py",
    "tests/extra_tests/converter/source_precision_diagnostics.py",
)
_CODE_FIELDS = (
    "co_argcount",
    "co_posonlyargcount",
    "co_kwonlyargcount",
    "co_nlocals",
    "co_stacksize",
    "co_flags",
    "co_code",
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
_ROLES = ("fresh", "warmed", "eager")
_FILE_LIMIT = 1024 * 1024
_DATA_LIMIT = 4 * _FILE_LIMIT
_HLO_LIMIT = 4 * _FILE_LIMIT
_TOTAL_LIMIT = 8 * _FILE_LIMIT
_STAT_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_nlink",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=True) + "\n").encode()


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        _require(name not in result, "duplicate JSON key")
        result[name] = value
    return result


def _stat(
    path: Path, *, directory: bool = False, single: bool = False
) -> tuple[int, ...]:
    result = path.lstat()
    _require(
        stat.S_ISDIR(result.st_mode) if directory else stat.S_ISREG(result.st_mode),
        "non-regular evidence path",
    )
    _require(not single or result.st_nlink == 1, "hardlinked evidence")
    return tuple(getattr(result, field) for field in _STAT_FIELDS)


def _parents(path: Path) -> None:
    _require(
        path.is_absolute() and ".." not in path.parts, "noncanonical evidence path"
    )
    for parent in path.parents:
        _stat(parent, directory=True)


def _read(path: Path, limit: int, *, single: bool = True) -> bytes:
    _parents(path)
    before = _stat(path, single=single)
    _require(before[4] <= limit, "evidence file exceeds finite bound")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        _require(
            tuple(getattr(os.fstat(descriptor), field) for field in _STAT_FIELDS)
            == before,
            "evidence file changed before read",
        )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read(limit + 1)
        _require(len(payload) == before[4], "evidence size changed")
        _require(
            tuple(getattr(os.fstat(descriptor), field) for field in _STAT_FIELDS)
            == before
            and _stat(path, single=single) == before,
            "evidence file changed during read",
        )
        return payload
    finally:
        os.close(descriptor)


def _hash_file(
    path: Path, limit: int, expected: tuple[int, ...] | None = None
) -> dict[str, Any]:
    _parents(path)
    before = _stat(path)
    _require(expected is None or before == expected, "pinned file changed")
    _require(before[4] <= limit, "hashed file exceeds bound")
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        _require(
            tuple(getattr(os.fstat(descriptor), f) for f in _STAT_FIELDS) == before,
            "file identity changed",
        )
        count = 0
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            while chunk := stream.read(_FILE_LIMIT):
                count += len(chunk)
                _require(count <= limit, "hashed file grew beyond bound")
                digest.update(chunk)
        _require(count == before[4], "hashed file size changed")
        _require(
            tuple(getattr(os.fstat(descriptor), f) for f in _STAT_FIELDS) == before
            and _stat(path) == before,
            "hashed file changed during read",
        )
        return {
            "path": str(path),
            "bytes": count,
            "sha256": digest.hexdigest(),
            "stat": before,
        }
    finally:
        os.close(descriptor)


def _encoded(value: Any, depth: int = 0, counter: list[int] | None = None) -> Any:
    """The reviewed v3 typed CodeType encoding, including slice constants."""
    if counter is None:
        counter = [0]
    counter[0] += 1
    _require(depth <= 32 and counter[0] <= 4096, "code fingerprint exceeds bound")

    def child(item: Any) -> Any:
        return _encoded(item, depth + 1, counter)

    kind = type(value)
    if value is None:
        return ["none"]
    if value is Ellipsis:
        return ["ellipsis"]
    if kind is bool:
        return ["bool", value]
    if kind is int:
        _require(value.bit_length() <= 65536, "integer constant exceeds bound")
        return ["int", hex(value)]
    if kind is float:
        return ["float64bits", struct.pack(">d", value).hex()]
    if kind is complex:
        return ["complex128bits", struct.pack(">dd", value.real, value.imag).hex()]
    if kind is str:
        _require(len(value) <= _FILE_LIMIT, "string constant exceeds bound")
        return ["str", value]
    if kind is bytes:
        _require(len(value) <= _FILE_LIMIT, "bytes constant exceeds bound")
        return ["bytes", value.hex()]
    if kind is tuple:
        return ["tuple", [child(item) for item in value]]
    if kind is frozenset:
        items = [child(item) for item in value]
        return [
            "frozenset",
            sorted(
                items,
                key=lambda item: json.dumps(
                    item, sort_keys=True, separators=(",", ":")
                ),
            ),
        ]
    if kind is slice:
        return ["slice", child(value.start), child(value.stop), child(value.step)]
    if kind is types.CodeType:
        return [
            "code",
            {field: child(getattr(value, field)) for field in _CODE_FIELDS},
            ["co_consts", child(value.co_consts)],
        ]
    raise TypeError("unsupported code constant")


def _fingerprint(code: types.CodeType) -> str:
    _require(type(code) is types.CodeType, "code must be exact CodeType")
    payload = json.dumps(
        _encoded(code), sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode()
    _require(len(payload) <= _FILE_LIMIT, "encoded code exceeds bound")
    return _sha(payload)


def _npy(payload: bytes, batch: int, index: int) -> dict[str, Any]:
    _require(
        len(payload) <= _FILE_LIMIT and payload[:6] == b"\x93NUMPY",
        "invalid NPY magic or size",
    )
    version = payload[6:8]
    _require(version in (b"\x01\x00", b"\x02\x00"), "unsupported NPY version")
    offset = 10 if version == b"\x01\x00" else 12
    _require(len(payload) >= offset, "short NPY header")
    length = int.from_bytes(payload[8:offset], "little")
    _require(
        length <= 4096 and offset + length <= len(payload), "NPY header exceeds bound"
    )
    tree = ast.parse(
        payload[offset : offset + length].decode("ascii").strip(), mode="eval"
    )
    _require(
        sum(1 for _ in ast.walk(tree)) <= 64, "NPY header complexity exceeds bound"
    )
    _require(
        isinstance(tree.body, ast.Dict) and len(tree.body.keys) == 3,
        "NPY header fields repeat",
    )
    header = ast.literal_eval(tree)
    _require(
        type(header) is dict and set(header) == {"descr", "shape", "fortran_order"},
        "NPY header fields differ",
    )
    _require(header["fortran_order"] is False, "unexpected NPY layout")
    shape = header["shape"]
    _require(
        type(shape) is tuple
        and all(type(d) is int for d in shape)
        and shape == (batch, 8, 2),
        "NPY shape differs",
    )
    allowed = ("<V2", "|V2") if index == 0 else ("<i4",)
    _require(
        type(header["descr"]) is str and header["descr"] in allowed,
        "NPY storage dtype differs",
    )
    _require(sys.byteorder == "little", "unreviewed host byteorder")
    raw = payload[offset + length :]
    _require(
        len(raw) == batch * 8 * 2 * (2 if index == 0 else 4), "NPY payload size differs"
    )
    return {
        "storage_dtype": header["descr"],
        "shape": shape,
        "raw_sha256": _sha(raw),
        "raw_hex": raw.hex(),
    }


def _npz(payload: bytes, batch: int) -> list[dict[str, Any]]:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        entries = archive.infolist()
        _require(
            [entry.filename for entry in entries] == ["leaf_0.npy", "leaf_1.npy"],
            "NPZ members differ or repeat",
        )
        result = []
        for index, entry in enumerate(entries):
            _require(
                entry.file_size <= _FILE_LIMIT
                and entry.compress_type == zipfile.ZIP_STORED
                and not entry.flag_bits & 1,
                "NPZ member exceeds schema",
            )
            raw = archive.read(entry)  # ZipFile also verifies each stored CRC.
            record = _npy(raw, batch, index)
            record.update(member=entry.filename, crc32=entry.CRC)
            result.append(record)
        return result


def _module(name: str) -> types.ModuleType:
    module = sys.modules.get(name)
    _require(
        type(module) is types.ModuleType, "required original module not already loaded"
    )
    return module


class _Capture:
    def __init__(self, folder: Path, item: Any, context: dict[str, Any]) -> None:
        self.folder, self.item, self.context_before = folder, item, context
        self.events: list[dict[str, Any]] = []
        self.hosts: dict[int, dict[str, Any]] = {}
        self.failed: dict[str, Any] | None = None
        self.original_report: Any = None
        self.compiler: Any = None
        self.original: Any = None
        self.wrapper: Any = None
        self.installed = False
        self.stopped = False
        self.closed = False
        self.files: list[dict[str, Any]] = []
        self.file_count = 0
        self.used = {"data": 0, "hlo": 0}
        self.report: dict[str, Any] = {
            "schema": SCHEMA,
            "status": "UNVERIFIED",
            "nodeid": NODE_ID,
            "original_test_outcome_authoritative": True,
            "numerical_acceptance": "NOT_ASSERTED",
            "source_or_runtime_called_by_capture": False,
            "profile_installed": False,
            "native_references_recomputed": False,
            "errors": [],
            "files": self.files,
            "compiler_events": [],
            "roles": [
                {
                    "batch": batch,
                    "role": role,
                    "status": "UNAVAILABLE",
                    "independent_compiler_return_claimed": False,
                }
                for batch in (2, 5)
                for role in _ROLES
            ],
            "raw_marshal_journal": [],
            "pytest_current_test_journal": [],
        }

    def error(self, stage: str, error: BaseException) -> None:
        if len(self.report["errors"]) < 32:
            self.report["errors"].append({"stage": stage, "type": type(error).__name__})
        self.report["status"] = "UNVERIFIED"

    def _function_state(self) -> dict[str, Any]:
        result = {}
        for name, function in self.functions.items():
            _require(
                type(function) is types.FunctionType, "non-Python original function"
            )
            code = function.__code__
            result[name] = {
                "function_id": id(function),
                "code_id": id(code),
                "semantic_sha256": _fingerprint(code),
            }
            raw = marshal.dumps(code)
            _require(
                len(raw) <= _FILE_LIMIT
                and len(self.report["raw_marshal_journal"]) < 64,
                "marshal journal exceeds bound",
            )
            self.report["raw_marshal_journal"].append(
                {
                    **result[name],
                    "raw_marshal_sha256": _sha(raw),
                    "authoritative_for_identity_equality": False,
                }
            )
        return result

    def _source_state(self) -> dict[str, Any]:
        result = {}
        for name, path in self.source_paths.items():
            raw = _read(path, 2 * _FILE_LIMIT, single=False)
            result[name] = {"bytes": len(raw), "sha256": _sha(raw)}
        return result

    def _context(self) -> dict[str, Any]:
        environment = dict(os.environ)
        _require(
            len(environment) <= 1024
            and sum(len(k) + len(v) for k, v in environment.items()) <= _FILE_LIMIT,
            "environment exceeds bound",
        )
        full_hash = _sha(_json(sorted(environment.items())))
        current_test = environment.pop("PYTEST_CURRENT_TEST", None)
        _require(
            current_test is None
            or (type(current_test) is str and len(current_test) <= 4096),
            "pytest phase value exceeds bound",
        )
        return {
            "environment_sha256": full_hash,
            "environment_without_pytest_current_test_sha256": _sha(
                _json(sorted(environment.items()))
            ),
            "pytest_current_test": current_test,
            "affinity": (
                sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None
            ),
            "sys_profile_id": id(sys.getprofile()),
            "threading_profile_id": id(threading.getprofile()),
        }

    def _observe_context(
        self, boundary: str, *, initial: bool = False, completed_call: bool = True
    ) -> dict[str, Any]:
        current = self._context()
        journal = self.report["pytest_current_test_journal"]
        _require(len(journal) < 12, "pytest phase journal exceeds bound")
        entry = {
            "boundary": boundary,
            "value": current["pytest_current_test"],
            "environment_sha256": current["environment_sha256"],
            "other_environment_sha256": current[
                "environment_without_pytest_current_test_sha256"
            ],
            "accepted": False,
            "completed_call_boundary": False,
            "requested_completed_call_boundary": completed_call and not initial,
        }
        journal.append(entry)
        setup, call = f"{NODE_ID} (setup)", f"{NODE_ID} (call)"
        value = current["pytest_current_test"]
        if initial:
            _require(
                value == setup, "capture acquisition is not the exact target setup"
            )
            self.phase_seen = "setup"
            self.context_pinned = current
        else:
            if completed_call:
                _require(value == call, "capture boundary is not the exact target call")
            else:
                _require(
                    value == call or (value == setup and self.phase_seen == "setup"),
                    "cleanup phase is not the same target setup or call",
                )
            excluded = {"environment_sha256", "pytest_current_test"}
            _require(
                {key: item for key, item in current.items() if key not in excluded}
                == {
                    key: item
                    for key, item in self.context_pinned.items()
                    if key not in excluded
                },
                "original readonly context changed beyond the exact pytest phase",
            )
            if value == call:
                self.phase_seen = "call"
        entry["accepted"] = True
        entry["completed_call_boundary"] = completed_call and not initial
        return current

    def start(self) -> None:
        _require(
            type(self.context_before) is dict
            and self.context_before.get("status") == "OBSERVED"
            and self.context_before.get("errors") == [],
            "original worker context unavailable",
        )
        _require(self.item.nodeid == NODE_ID, "unexpected capture node")
        _require(
            type(self.item) is vars(_module("pytest"))["Function"],
            "unexpected pytest item type",
        )
        self.owner = self.item.module
        _require(type(self.owner) is types.ModuleType, "original owner module differs")
        _require(
            Path(vars(self.owner)["__file__"]) == _ROOT / OWNER, "owner path differs"
        )
        _require(
            _sha(_read(_ROOT / OWNER, 2 * _FILE_LIMIT)) == _OWNER_SHA256,
            "owner source differs",
        )
        self.tmp = self.item.funcargs["tmp_path"]
        _require(
            type(self.tmp) is type(Path()) and self.tmp.is_absolute(),
            "tmp_path fixture differs",
        )
        _parents(self.tmp)
        self.tmp_stat = _stat(self.tmp, directory=True)
        self.numpy = _module("numpy")
        self.array_type = vars(self.numpy)["ndarray"]
        self.array_bytes = vars(self.array_type)["tobytes"]
        self.compiler = _module("jax._src.compiler")
        self.pxla = _module("jax._src.interpreters.pxla")
        self.native_lax = _module("jax._src.lax.lax")
        _require(
            vars(self.pxla).get("compiler") is self.compiler,
            "pxla compiler binding differs",
        )
        self.original = vars(self.compiler)["compile_or_get_cached"]
        self.functions = {
            name: vars(self.owner)[name]
            for name in (TEST, "_bf16_native", "_portable_relation", "_retain")
        }
        _require(self.item.obj is self.functions[TEST], "original test binding differs")
        self.functions["compiler"] = self.original
        self.functions["native_lowerer"] = vars(self.native_lax)["_top_k_lower"]
        self.codes = {
            name: function.__code__ for name, function in self.functions.items()
        }
        code = self.original.__code__
        _require(
            type(self.original) is types.FunctionType
            and self.original.__globals__ is vars(self.compiler)
            and code.co_argcount == 7
            and code.co_posonlyargcount == code.co_kwonlyargcount == 0
            and code.co_varnames[:7] == _FORMALS
            and not code.co_flags & 0x0C
            and self.original.__defaults__ == (None,)
            and self.original.__kwdefaults__ is None
            and self.original.__closure__ is None,
            "unreviewed compiler callable schema",
        )
        self.signature = inspect.signature(
            self.original, follow_wrapped=False, eval_str=False
        )
        _require(
            tuple(self.signature.parameters) == _FORMALS
            and all(
                p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
                for p in self.signature.parameters.values()
            ),
            "compiler signature differs",
        )
        self.source_paths = {name: _ROOT / name for name in _SOURCES}
        for name, module in (
            ("compiler", self.compiler),
            ("pxla", self.pxla),
            ("lax", self.native_lax),
        ):
            self.source_paths[name] = Path(vars(module)["__file__"])
        self.sources_before = self._source_state()
        schema = _COMPILER_SCHEMAS[self.sources_before["compiler"]["sha256"]]
        _require(
            (
                self.sources_before["pxla"]["sha256"],
                self.sources_before["lax"]["sha256"],
            )
            == schema,
            "native source epochs differ",
        )
        self.functions_before = self._function_state()
        self._observe_context("acquire", initial=True)
        _require(
            sys.getprofile() is None and threading.getprofile() is None,
            "preexisting profile preserved",
        )
        self.native_paths: dict[Path, tuple[int, ...]] = {}
        for name in (
            "jaxlib._jax",
            "jaxlib._xla",
            "jaxlib.xla_extension",
            "numpy._core._multiarray_umath",
            "numpy.core._multiarray_umath",
            "ml_dtypes._ml_dtypes_ext",
        ):
            module = sys.modules.get(name)
            if module is not None:
                _require(type(module) is types.ModuleType, "foreign native module")
                origin = vars(module).get("__file__")
                if type(origin) is str and origin.endswith((".so", ".pyd")):
                    path = Path(origin)
                    self.native_paths[path] = _stat(path)
        jaxlib = _module("jaxlib")
        for basename in ("_jax.so", "_xla.so", "libjax_common.so"):
            path = Path(vars(jaxlib)["__file__"]).parent / basename
            try:
                self.native_paths[path] = _stat(path)
            except FileNotFoundError:
                pass
        _require(0 < len(self.native_paths) <= 8, "native identity paths unavailable")
        self.module_bindings = {
            name: _module(name)
            for name in (
                "numpy",
                "jax",
                "jaxlib",
                "jax._src.compiler",
                "jax._src.interpreters.pxla",
                "jax._src.lax.lax",
            )
        }
        self.report.update(
            context_before=self.context_before,
            readonly_context_before=self.context_pinned,
            source_before=self.sources_before,
            functions_before=self.functions_before,
            module_origins={
                name: vars(module).get("__file__")
                for name, module in self.module_bindings.items()
            },
            compiler_formals=_FORMALS,
        )

        def passthrough(*args: Any, **kwargs: Any) -> Any:
            executable = self.original(*args, **kwargs)
            # The original call is deliberately outside diagnostic handling.
            try:
                _require(len(self.events) < 16, "compiler event count exceeds bound")
                event: dict[str, Any] = {
                    "ordinal": len(self.events),
                    "executable": executable,
                    "executable_id": id(executable),
                    "argument_ids": [id(v) for v in args],
                    "keyword_argument_ids": {name: id(v) for name, v in kwargs.items()},
                    "returned_exact_original_object": True,
                    "native_frame_binding": None,
                }
                self.events.append(event)  # Retain actual return before metadata.
                parameters = self.signature.bind(*args, **kwargs)
                parameters.apply_defaults()
                _require(
                    tuple(parameters.arguments) == _FORMALS, "compiler binding differs"
                )
                event["parameters"] = dict(parameters.arguments)
                event["formal_argument_ids"] = {
                    name: id(v) for name, v in parameters.arguments.items()
                }
                event["native_frame_binding"] = self._compiler_frames()
            except BaseException as error:
                self.error("compiler-return-observation", error)
            return executable

        self.wrapper = passthrough
        _require(
            vars(self.compiler).get("compile_or_get_cached") is self.original,
            "compiler changed before install",
        )
        self.compiler.compile_or_get_cached = self.wrapper
        self.installed = True

    def _array(
        self,
        value: Any,
        batch: int,
        *,
        input_value: bool = False,
        index: bool = False,
        copy: bool = False,
    ) -> dict[str, Any]:
        _require(
            type(value) is self.array_type, "host value is not exact original ndarray"
        )
        shape = (batch, 8, 4 if input_value else 2)
        dtype = "int32" if index else "bfloat16"
        _require(
            value.shape == shape
            and value.dtype.name == dtype
            and value.nbytes == batch * 8 * shape[-1] * (4 if index else 2),
            "host dtype or shape differs",
        )
        result = {
            "array_id": id(value),
            "dtype_name": dtype,
            "shape": shape,
            "bytes": value.nbytes,
        }
        if copy:
            raw = self.array_bytes(value)
            _require(
                type(raw) is bytes and len(raw) == value.nbytes, "host bytes differ"
            )
            result.update(raw_hex=raw.hex(), raw_sha256=_sha(raw))
        return result

    def _bind(self, test: dict[str, Any], native: dict[str, Any]) -> dict[str, Any]:
        label, directory, data = native["label"], native["directory"], native["data"]
        batch = test["size"]
        _require(
            type(batch) is int
            and batch in (2, 5)
            and type(label) is str
            and label in _ROLES,
            "native role or batch differs",
        )
        _require(
            type(directory) is type(Path())
            and directory == self.tmp / "native-default" / f"B{batch}",
            "native directory differs",
        )
        _require(
            test["scope"] == "inline"
            and test["contains_nan"] is False
            and test["selected"] is data
            and test["inputs"] is native["inputs"],
            "native input owner differs",
        )
        inputs, words, references = (
            native["inputs"],
            test["original_words"],
            native["references"],
        )
        _require(
            type(inputs) is tuple and len(inputs) == 1 and inputs[0] is data,
            "native input tuple differs",
        )
        _require(
            type(words) is tuple
            and len(words) == 1
            and type(words[0]) is bytes
            and len(words[0]) == batch * 8 * 4 * 2,
            "original pre-call words unavailable",
        )
        _require(
            type(references) is list and len(references) <= 3,
            "native references differ",
        )
        self._array(data, batch, input_value=True)
        binding = {
            "batch": batch,
            "role": label,
            "directory": f"native-default/B{batch}",
            "data_id": id(data),
            "inputs_id": id(inputs),
            "original_words_id": id(words),
            "jit_wrapper_id": id(native["compiled"]),
            "operation_id": id(native["operation"]),
            "references_id": id(references),
        }
        prior = self.hosts.get(batch)
        if prior is not None:
            _require(
                prior["data"] is data
                and prior["references"] is references
                and prior["words"] is words,
                "native batch replay differs",
            )
        else:
            self.hosts[batch] = {
                "data": data,
                "inputs": inputs,
                "words": words,
                "references": references,
                "binding": binding,
            }
        return binding

    def _compiler_frames(self) -> dict[str, Any] | None:
        frame = sys._getframe(1)
        found: dict[str, list[dict[str, Any]]] = {TEST: [], "_bf16_native": []}
        try:
            for _ in range(64):
                if frame is None:
                    break
                for name in found:
                    if frame.f_code is self.codes[name]:
                        found[name].append(frame.f_locals)
                frame = frame.f_back
            _require(frame is None, "source frame walk exceeds bound")
            if not found["_bf16_native"]:
                return None  # An unrelated original compilation is not TopK proof.
            _require(
                all(len(matches) == 1 for matches in found.values()),
                "source frames missing or ambiguous",
            )
            return self._bind(found[TEST][0], found["_bf16_native"][0])
        finally:
            del frame

    def finish(
        self, context: dict[str, Any] | None = None, *, completed_call: bool = True
    ) -> None:
        if self.stopped:
            return
        self.stopped = True
        try:
            if self.installed:
                if vars(self.compiler).get("compile_or_get_cached") is not self.wrapper:
                    self.installed = False
                    raise ValueError("foreign compiler replacement preserved")
                self.compiler.compile_or_get_cached = self.original
                self.installed = False
            self._fences(
                "call-finish" if completed_call else "cleanup",
                completed_call=completed_call,
            )
            if context is not None:
                self.finish_context(context)
        except BaseException as error:
            self.error("original-call-final-fences", error)

    def _fences(
        self, boundary: str = "call-fence", *, completed_call: bool = True
    ) -> None:
        _require(
            vars(self.compiler).get("compile_or_get_cached") is self.original,
            "original compiler reference not restored",
        )
        _require(
            all(
                sys.modules.get(name) is module
                for name, module in self.module_bindings.items()
            ),
            "original module binding changed",
        )
        _require(
            vars(self.pxla).get("compiler") is self.compiler,
            "pxla compiler binding changed",
        )
        _require(
            all(
                vars(self.owner).get(name) is function
                for name, function in self.functions.items()
                if name not in ("compiler", "native_lowerer")
            ),
            "owner function binding changed",
        )
        _require(
            vars(self.native_lax).get("_top_k_lower")
            is self.functions["native_lowerer"],
            "native lowerer binding changed",
        )
        current = self._function_state()
        _require(
            current == self.functions_before,
            "original function or stable code fingerprint changed",
        )
        _require(self._source_state() == self.sources_before, "original source changed")
        self._observe_context(boundary, completed_call=completed_call)
        _require(
            vars(self.array_type).get("tobytes") is self.array_bytes,
            "original ndarray bytes binding changed",
        )
        _require(
            _stat(self.tmp, directory=True)[:3] == self.tmp_stat[:3],
            "original pytest tmp identity changed",
        )
        self.report[
            "original_fences_held" if completed_call else "cleanup_fences_held"
        ] = True

    def finish_context(self, context: dict[str, Any]) -> None:
        self.report["context_after"] = context
        try:
            current = self._observe_context("worker-context-after")
            before = self.context_before.get("environment")
            after = context.get("environment")
            _require(
                type(before) is dict and type(after) is dict,
                "worker environment missing",
            )
            self.report["worker_pytest_current_test"] = {
                "before": before.get("PYTEST_CURRENT_TEST"),
                "after": after.get("PYTEST_CURRENT_TEST"),
                "accepted": False,
            }
            _require(
                before.get("PYTEST_CURRENT_TEST")
                == self.context_pinned["pytest_current_test"]
                and after.get("PYTEST_CURRENT_TEST") == current["pytest_current_test"],
                "worker pytest phase does not match the exact observed target",
            )
            _require(
                {
                    key: value
                    for key, value in before.items()
                    if key != "PYTEST_CURRENT_TEST"
                }
                == {
                    key: value
                    for key, value in after.items()
                    if key != "PYTEST_CURRENT_TEST"
                }
                and all(
                    context.get(name) == self.context_before.get(name)
                    for name in ("cpu", "loaded_jax_config")
                ),
                "target worker context changed beyond the exact pytest phase",
            )
            self.report["worker_pytest_current_test"]["accepted"] = True
        except BaseException as error:
            self.error("target-worker-context", error)

    def retain_report(self, call: Any, report: Any) -> None:
        """Keep only existing owners after pytest created its original report."""
        try:
            self._observe_context("original-report")
            _require(
                type(report) is vars(_module("_pytest.reports"))["TestReport"],
                "foreign original report",
            )
            _require(
                report.nodeid == NODE_ID and report.when == "call",
                "original report scope differs",
            )
            self.original_report = report
            if report.outcome != "failed":
                return
            _require(
                type(call) is vars(_module("_pytest.runner"))["CallInfo"],
                "foreign original call",
            )
            excinfo = vars(call)["excinfo"]
            _require(
                type(excinfo) is vars(_module("_pytest._code.code"))["ExceptionInfo"],
                "foreign exception info",
            )
            triple = vars(excinfo)["_excinfo"]
            _require(
                type(triple) is tuple and len(triple) == 3,
                "original exception tuple differs",
            )
            traceback = triple[2]
            found: dict[str, list[dict[str, Any]]] = {
                TEST: [],
                "_bf16_native": [],
                "_portable_relation": [],
            }
            try:
                for _ in range(64):
                    if traceback is None:
                        break
                    _require(
                        type(traceback) is types.TracebackType, "foreign traceback"
                    )
                    for name in found:
                        if traceback.tb_frame.f_code is self.codes[name]:
                            found[name].append(traceback.tb_frame.f_locals)
                    traceback = traceback.tb_next
                _require(traceback is None, "traceback exceeds bound")
                if not found["_bf16_native"]:
                    return
                _require(
                    len(found[TEST]) == len(found["_bf16_native"]) == 1
                    and len(found["_portable_relation"]) <= 1,
                    "failed source frame ambiguity",
                )
                binding = self._bind(found[TEST][0], found["_bf16_native"][0])
                actual = found["_bf16_native"][0].get("actual")
                self.failed = {
                    "binding": binding,
                    "actual": actual,
                    "relation": (
                        {
                            name: found["_portable_relation"][0].get(name)
                            for name in (
                                "position",
                                "row",
                                "keys",
                                "chosen",
                                "selected_keys",
                            )
                        }
                        if found["_portable_relation"]
                        else None
                    ),
                }
            finally:
                del traceback
        except BaseException as error:
            self.error("original-report-pin", error)

    def _write(
        self,
        relative: str,
        payload: bytes,
        *,
        hlo: bool = False,
        manifest: bool = False,
    ) -> dict[str, Any]:
        kind = "hlo" if hlo else "data"
        _require(
            type(payload) is bytes and len(payload) <= _FILE_LIMIT,
            "output file exceeds bound",
        )
        _require(
            self.file_count < (64 if manifest else 63),
            "output file count exceeds bound",
        )
        limit = (
            _HLO_LIMIT if hlo else _DATA_LIMIT - (0 if manifest else _FILE_LIMIT // 2)
        )
        _require(
            self.used[kind] + len(payload) <= limit
            and sum(self.used.values()) + len(payload) <= _TOTAL_LIMIT,
            "aggregate output exceeds bound",
        )
        # Failed writes still consume the finite allocation and file allowance.
        self.file_count += 1
        self.used[kind] += len(payload)
        path = self.output / relative
        _require(
            not Path(relative).is_absolute() and ".." not in Path(relative).parts,
            "invalid output path",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        _parents(path)
        descriptor = os.open(
            path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
        )
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(payload)
            _require(os.fstat(descriptor).st_size == len(payload), "short output write")
        finally:
            os.close(descriptor)
        record = {
            "path": relative,
            "bytes": len(payload),
            "sha256": _sha(payload),
            "kind": kind,
        }
        if not manifest:
            self.files.append(record)
        return record

    def _host_bytes(self) -> dict[tuple[int, str], tuple[dict[str, Any], ...]]:
        returned = {}
        for batch, host in self.hosts.items():
            data = self._array(host["data"], batch, input_value=True, copy=True)
            _require(
                data["raw_hex"] == host["words"][0].hex(),
                "original input words changed",
            )
            self._write(
                f"B{batch}-original-input.json",
                _json(
                    {
                        "binding": host["binding"],
                        "input": data,
                        "original_pre_call_words_hex": host["words"][0].hex(),
                    }
                ),
            )
            references = host["references"]
            _require(
                type(references) is list and len(references) <= 3,
                "retained original references changed",
            )
            for role, values in zip(_ROLES, references, strict=False):
                _require(
                    type(values) is tuple and len(values) == 2,
                    "original retained outputs differ",
                )
                returned[batch, role] = tuple(
                    self._array(value, batch, index=index == 1, copy=True)
                    for index, value in enumerate(values)
                )
        if self.failed is not None:
            binding, actual = self.failed["binding"], self.failed["actual"]
            if actual is not None:
                _require(
                    type(actual) is tuple and len(actual) == 2,
                    "failed original outputs differ",
                )
                returned[binding["batch"], binding["role"]] = tuple(
                    self._array(value, binding["batch"], index=index == 1, copy=True)
                    for index, value in enumerate(actual)
                )
            relation = self.failed["relation"]
            if relation is not None:
                for value in relation.values():
                    _require(
                        type(value) in (tuple, list)
                        and len(value) <= 4
                        and all(type(item) is int for item in value),
                        "failed relation literals differ",
                    )
            self.report["failed_original_frame"] = {
                "binding": binding,
                "relation": relation,
            }
        return returned

    def _existing_files(
        self, returned: dict[tuple[int, str], tuple[dict[str, Any], ...]]
    ) -> None:
        copied = 0
        seen_inodes = set()
        missing = False
        prefix_ended = False
        for batch in (2, 5):
            base = self.tmp / "native-default" / f"B{batch}"
            observations = None
            try:
                raw = _read(base / "native-word-observations.json", _FILE_LIMIT)
                observations = json.loads(raw, object_pairs_hook=_unique)
                _require(
                    type(observations) is dict
                    and observations.get("contains_nan") is False,
                    "native observations differ",
                )
                entries = observations["native_roles"]
                _require(
                    type(entries) is list
                    and len(entries) <= 3
                    and [entry["native_role"] for entry in entries]
                    == list(_ROLES[: len(entries)]),
                    "native role order differs",
                )
                self._write(
                    f"native-default/B{batch}/native-word-observations.json", raw
                )
                copied += 1
            except FileNotFoundError:
                entries = []
                missing = True
            for index, role in enumerate(_ROLES):
                record = self.report["roles"][(0 if batch == 2 else 3) + index]
                try:
                    path = base / f"{role}.npz"
                    payload = _read(path, _FILE_LIMIT)
                    _require(
                        not prefix_ended,
                        "native archives are not an original role prefix",
                    )
                    identity = _stat(path, single=True)[:2]
                    _require(
                        identity not in seen_inodes, "replayed native payload inode"
                    )
                    seen_inodes.add(identity)
                    arrays = _npz(payload, batch)
                    event = _read(base / f"{role}.json", _FILE_LIMIT)
                    archived = json.loads(event, object_pairs_hook=_unique)
                    _require(
                        archived == {"status": "archived", "stage": role, "leaves": 2},
                        "native archive event differs",
                    )
                    _require(index < len(entries), "native observation role missing")
                    observation = entries[index]
                    _require(
                        observation["retained_file"] == path.name
                        and observation["retained_sha256"] == _sha(payload),
                        "native retained digest differs",
                    )
                    outputs = returned.get((batch, role))
                    _require(
                        outputs is not None,
                        "original host dtype/source binding missing",
                    )
                    for array, output in zip(arrays, outputs, strict=True):
                        _require(
                            array["raw_hex"] == output["raw_hex"],
                            "NPZ and exact host bytes differ",
                        )
                        array["actual_dtype_name"] = output["dtype_name"]
                        array["original_array_id"] = output["array_id"]
                    self._write(f"native-default/B{batch}/{role}.npz", payload)
                    self._write(f"native-default/B{batch}/{role}.json", event)
                    self._write(
                        f"native-default/B{batch}/{role}-raw.json",
                        _json(
                            {
                                "arrays": arrays,
                                "source_binding": self.hosts[batch]["binding"],
                            }
                        ),
                    )
                    copied += 2
                    record.update(
                        status="ORIGINAL_HOST_BYTES_AND_ARCHIVE_MATCH",
                        npz_sha256=_sha(payload),
                        jit_wrapper_id=self.hosts[batch]["binding"]["jit_wrapper_id"],
                    )
                except FileNotFoundError:
                    missing = True
                    prefix_ended = True
                    if index < len(entries) or (batch, role) in returned:
                        self.error(
                            f"missing-original-archive-B{batch}-{role}",
                            ValueError("observed original role file missing"),
                        )
                except BaseException as error:
                    self.error(f"existing-native-file-B{batch}-{role}", error)
        _require(copied <= 14, "native file allowlist exceeds bound")
        self.report["six_role_archive_complete"] = (
            not missing
            and len(self.report["roles"]) == 6
            and all(
                role["status"] == "ORIGINAL_HOST_BYTES_AND_ARCHIVE_MATCH"
                for role in self.report["roles"]
            )
        )

    def _hlo(self) -> None:
        observed: dict[int, list[dict[str, Any]]] = {}
        for event in self.events:
            record = {
                name: value
                for name, value in event.items()
                if name not in ("executable", "parameters")
            }
            record["hlo"] = {"status": "UNAVAILABLE"}
            self.report["compiler_events"].append(record)
            binding = event["native_frame_binding"]
            if binding is None or "parameters" not in event:
                record["classification"] = "UNCATEGORIZED_ORIGINAL_COMPILATION"
                continue
            record["classification"] = "EXACT_ORIGINAL_NATIVE_FRAME_BOUND_RETURN"
            try:
                parameters = event["parameters"]
                record["actual_module_symbol"] = str(
                    parameters["computation"].operation.attributes["sym_name"]
                )
                _require(
                    len(record["actual_module_symbol"]) <= 256,
                    "module symbol exceeds bound",
                )
                record["backend_platform"] = parameters["backend"].platform
                _require(
                    type(record["backend_platform"]) is str
                    and len(record["backend_platform"]) <= 256,
                    "backend metadata differs",
                )
                identity = event["executable_id"]
                if identity not in observed:
                    modules = event["executable"].hlo_modules()
                    _require(
                        type(modules) in (list, tuple) and 0 < len(modules) <= 4,
                        "HLO module count differs",
                    )
                    files = []
                    for index, module in enumerate(modules):
                        text = module.to_string()
                        _require(
                            type(text) is str and len(text) <= _FILE_LIMIT,
                            "HLO text exceeds bound",
                        )
                        files.append(
                            self._write(
                                f"hlo/compile-{event['ordinal']}-module-{index}.hlo.txt",
                                text.encode(),
                                hlo=True,
                            )
                        )
                    observed[identity] = files
                record["hlo"] = {
                    "status": "CAPTURED_FROM_SAME_ACTUAL_COMPILER_RETURN",
                    "files": observed[identity],
                }
            except BaseException as error:
                self.error("actual-executable-hlo", error)
        self.report["actual_native_hlo_observed"] = bool(observed)

    def _native_builds(self) -> None:
        records, total = [], 0
        for path, before in self.native_paths.items():
            total += before[4]
            _require(total <= 768 * _FILE_LIMIT, "native hash aggregate exceeds bound")
            records.append(_hash_file(path, 512 * _FILE_LIMIT, before))
        distributions = {}
        for name in ("jaxlib", "numpy", "ml-dtypes"):
            distribution = importlib.metadata.distribution(name)
            _require(
                type(distribution) is importlib.metadata.PathDistribution,
                "foreign installed distribution",
            )
            metadata = vars(distribution).get("_path")
            _require(
                type(metadata) is type(Path()) and metadata.name.endswith(".dist-info"),
                "installed metadata path differs",
            )
            root = distribution.locate_file("")
            _require(
                type(root) is type(Path()) and root == metadata.parent,
                "installed root differs",
            )
            metadata_bytes = _read(metadata / "METADATA", 2 * _FILE_LIMIT, single=False)
            fields = Parser().parsestr(metadata_bytes.decode(), headersonly=True)
            names, versions = fields.get_all("Name", []), fields.get_all("Version", [])
            _require(
                len(names) == len(versions) == 1
                and all(
                    type(value) is str and len(value) <= 256
                    for value in (*names, *versions)
                )
                and names[0].lower().replace("_", "-") == name,
                "installed metadata name or version differs",
            )
            payload = _read(metadata / "RECORD", 2 * _FILE_LIMIT, single=False)
            matches = []
            seen = set()
            for index, row in enumerate(csv.reader(io.StringIO(payload.decode()))):
                _require(
                    index < 20000
                    and len(row) == 3
                    and all(len(field) <= 4096 for field in row),
                    "installed RECORD exceeds schema bounds",
                )
                relative, digest, size = row
                _require(relative not in seen, "duplicate installed RECORD entry")
                seen.add(relative)
                if not relative.endswith((".so", ".pyd")):
                    continue
                parts = relative.split("/")
                _require(
                    not Path(relative).is_absolute()
                    and all(part not in ("", ".", "..") for part in parts),
                    "unsafe installed native path",
                )
                located = root.joinpath(*parts)
                for record in records:
                    if located == Path(record["path"]):
                        expected = "sha256=" + base64.urlsafe_b64encode(
                            bytes.fromhex(record["sha256"])
                        ).decode().rstrip("=")
                        _require(
                            digest == expected and size == str(record["bytes"]),
                            "native build disagrees with RECORD",
                        )
                        matches.append(record["path"])
            distributions[name] = {
                "version": versions[0],
                "metadata_sha256": _sha(metadata_bytes),
                "record_sha256": _sha(payload),
                "record_bytes": len(payload),
                "matched_native_paths": matches,
            }
        _require(
            all(
                any(
                    record["path"] in item["matched_native_paths"]
                    for item in distributions.values()
                )
                for record in records
            ),
            "native file has no installed RECORD proof",
        )
        self.report.update(native_builds=records, installed_distributions=distributions)
        _require(
            all(_stat(path) == before for path, before in self.native_paths.items()),
            "native file changed after hash",
        )

    def after_report(self, phase: dict[str, Any]) -> None:
        try:
            _require(
                self.stopped
                and phase["nodeid"] == NODE_ID
                and phase["when"] == "call"
                and phase["durable_phase_recorded"] is True,
                "original durable phase missing",
            )
            self.output = self.folder / DIRECTORY
            self.output.mkdir(exist_ok=False)
            longrepr = phase.get("longrepr", "")
            _require(
                phase["outcome"] in ("passed", "failed", "skipped")
                and type(longrepr) is str
                and len(longrepr) <= _FILE_LIMIT,
                "runner-observed original phase exceeds bounds",
            )
            self.report["runner_observed_original_call"] = {
                "nodeid": NODE_ID,
                "when": "call",
                "outcome": phase["outcome"],
                "longrepr_sha256": _sha(longrepr.encode()),
                "durable_phase_recorded": True,
                "validation": "RUNNER_OBSERVED_NOT_CAPTURE_VALIDATED",
            }
            _require(
                self.original_report is not None
                and self.original_report.outcome == phase["outcome"],
                "original report binding missing",
            )
            if phase["outcome"] == "failed":
                _require(
                    str(self.original_report.longrepr) == phase.get("longrepr", ""),
                    "original failure longrepr binding differs",
                )
            self._observe_context("durable-original-report")
            target_phases = phase["receipt_target_phases"]
            _require(
                type(target_phases) is list
                and len(target_phases) == 2
                and [entry["when"] for entry in target_phases] == ["setup", "call"],
                "original receipt phase projection differs",
            )
            _require(
                target_phases[-1]
                == {
                    name: phase[name]
                    for name in ("when", "outcome", "longrepr", "wasxfail")
                    if name in phase
                },
                "original phase and durable receipt projection differ",
            )
            projection = self._write(
                "original-phase.json",
                _json({"nodeid": NODE_ID, "phases": target_phases}),
            )
            self.report["original_phase"] = {
                "nodeid": NODE_ID,
                "when": "call",
                "outcome": phase["outcome"],
                "longrepr_sha256": _sha(phase.get("longrepr", "").encode()),
                "durable_phase_recorded": True,
                "receipt": _hash_file(self.folder / "receipt.json", 16 * _FILE_LIMIT),
                "events": _hash_file(self.folder / "events.jsonl", 16 * _FILE_LIMIT),
                "receipt_target_projection": projection,
                "receipt_sha_scope": "Durable call boundary; receipt is rewritten by later original phases. Match original-phase.json against the final target phase prefix.",
                "events_sha_scope": "Durable byte prefix of the original append-only events.jsonl; later phases append further bytes.",
            }
            self._fences("before-native-copy")
            returned = self._host_bytes()
            self._existing_files(returned)
            self._hlo()
            self._native_builds()
            self._fences("after-native-copy")
            _require(
                self.report.get("actual_native_hlo_observed") is True,
                "actual native executable unavailable",
            )
            if not self.report["errors"]:
                self.report["status"] = (
                    "READY" if self.report["six_role_archive_complete"] else "PARTIAL"
                )
        except BaseException as error:
            self.error("post-original-report-capture", error)
        finally:
            try:
                self.close()
            finally:
                if hasattr(self, "output"):
                    try:
                        self._write("manifest.json", _json(self.report), manifest=True)
                    except BaseException as error:
                        self.error("manifest-write", error)

    def close(self) -> None:
        if self.closed:
            return
        try:
            self.finish(completed_call=False)
        finally:
            self.closed = True
            self.events.clear()
            self.hosts.clear()
            self.failed = self.original_report = self.item = None
            self.report["native_owners_cleared"] = True


def start_observation(folder: Path, item: Any, context: dict[str, Any]) -> _Capture:
    capture = _Capture(folder, item, context)
    try:
        capture.start()
    except BaseException as error:
        try:
            capture.error("capture-start", error)
        finally:
            capture.finish(completed_call=False)
    return capture
