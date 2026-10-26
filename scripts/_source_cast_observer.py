# scripts/_source_cast_observer.py

"""Passive words from one unchanged cast test inside its original batch.

This diagnostic never executes the test, a model, an oracle, or another sample.
The batch runner owns selection, pytest phases, original outcomes and uploads.
"""

from __future__ import annotations

import base64
import csv
from email.parser import Parser
import hashlib
import importlib.metadata
import io
import json
import marshal
import os
from pathlib import Path
import re
import stat
import sys
import sysconfig
import types
from typing import Any

NODE_ID = (
    "tests/extra_tests/converter/test_source_conversion_semantics.py::"
    "test_floating_conversion_rounding_and_strict_special_value_gates"
    "[ties-float16-float64]"
)
REPORT_NAME = "source-f64-f16-cast-observation.json"
MODEL_NAME = "source-f64-f16-cast-artifact.onnx"
_ROOT = Path(__file__).resolve().parent.parent
_SOURCE_FILES = (
    "scripts/run_pytest_batches.py",
    "scripts/_source_cast_observer.py",
    "tests/extra_tests/converter/test_source_conversion_semantics.py",
    "tests/extra_tests/converter/test_source_pointwise_mul_exp.py",
    "tests/extra_tests/converter/test_source_predicate_families.py",
    "jax2onnx/precision.py",
    "jax2onnx/_compat/jax.py",
    "jax2onnx/plugins/jax/lax/convert_element_type.py",
)
_MAX_FILE_BYTES = 2 * 1024 * 1024
_MAX_REPORT_BYTES = 4 * 1024 * 1024
_NATIVE_DISTRIBUTIONS = ("jaxlib", "onnxruntime", "numpy", "ml-dtypes")
_MAX_NATIVE_RECORD_BYTES = 8 * 1024 * 1024
_MAX_NATIVE_RECORD_ROWS = 20000
_MAX_NATIVE_FILES = 128
_MAX_NATIVE_FILE_BYTES = 512 * 1024 * 1024
_MAX_NATIVE_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
_MAX_NATIVE_MANIFEST_BYTES = 256 * 1024
_NATIVE_CHUNK_BYTES = 1024 * 1024
_NATIVE_PROGRESS_BYTES = 64 * 1024 * 1024


def _installed_stat(path: Path) -> dict[str, int]:
    observed = path.lstat()
    _require(stat.S_ISREG(observed.st_mode), "installed proof requires a regular file")
    _require(path.resolve(strict=True) == path, "installed proof path is not canonical")
    return _stat_fields(observed)


def _stat_fields(observed: os.stat_result) -> dict[str, int]:
    return {
        name: getattr(observed, name)
        for name in (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
    }


def _read_installed_file(
    path: Path,
    bound: int,
    proof: dict[str, Any],
    *,
    collect: bool = False,
    progress: Any = None,
    checkpoint: Any = None,
) -> bytes:
    before = _installed_stat(path)
    proof["stat_before"] = before
    _require(before["st_size"] <= bound, "installed file exceeds byte bound")
    descriptor = os.open(
        path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC
    )
    digest = hashlib.sha256()
    parts = []
    count = 0
    next_checkpoint = _NATIVE_PROGRESS_BYTES
    try:
        _require(
            _stat_fields(os.fstat(descriptor)) == before,
            "installed file changed before open",
        )
        while True:
            chunk = os.read(descriptor, min(_NATIVE_CHUNK_BYTES, bound - count + 1))
            if not chunk:
                break
            count += len(chunk)
            proof["bytes_hashed"] = count
            _require(count <= bound, "installed file grew beyond byte bound")
            digest.update(chunk)
            if collect:
                parts.append(chunk)
            if progress is not None:
                progress(len(chunk))
            if checkpoint is not None and count >= next_checkpoint:
                checkpoint()
                next_checkpoint += _NATIVE_PROGRESS_BYTES
        after = _stat_fields(os.fstat(descriptor))
        proof["stat_after"] = _installed_stat(path)
        _require(
            before == after == proof["stat_after"] and count == before["st_size"],
            "installed file identity or stat changed while hashing",
        )
        proof.update(sha256=digest.hexdigest(), file_identity_held=True)
    finally:
        os.close(descriptor)
    return b"".join(parts)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _error(error: BaseException) -> dict[str, Any]:
    args = BaseException.__dict__["args"].__get__(error, type(error))
    return {
        "type": type(error).__module__ + "." + type(error).__qualname__,
        "literal_args": [
            (
                value
                if type(value) in (str, int, bool) or value is None
                else {"type_identity": id(type(value))}
            )
            for value in args[:8]
        ],
    }


def _file(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    _require(size <= _MAX_FILE_BYTES, "diagnostic source file exceeds finite bound")
    payload = path.read_bytes()
    _require(len(payload) == size, "diagnostic source file changed while reading")
    return {
        "path": str(path.resolve()),
        "bytes": size,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _literal(value: Any, depth: int = 0) -> Any:
    _require(depth <= 4, "diagnostic literal depth exceeds bound")
    if value is None or type(value) in (bool, int, float):
        return value
    if type(value) is str:
        _require(len(value) <= 4096, "diagnostic string exceeds bound")
        return value
    if type(value) in (tuple, list):
        _require(len(value) <= 32, "diagnostic sequence exceeds bound")
        return [_literal(item, depth + 1) for item in value]
    raise ValueError("diagnostic return requires exact literal storage")


def _dtype_promotion_enum(value: Any) -> dict[str, Any]:
    # Newer JAX exposes this setting as a StrEnum. Accept only its actual,
    # already-loaded native members; str()/equality on foreign values can run
    # user code and would cease to be a passive observation.
    native = sys.modules.get("jax._src.config")
    _require(
        type(native) is types.ModuleType,
        "original JAX config enum module was not loaded",
    )
    enum_type = vars(native).get("NumpyDtypePromotion")
    _require(
        type(value) is enum_type,
        "original JAX config requires its exact native dtype-promotion enum",
    )
    namespace = vars(enum_type)
    member = next(
        (name for name in ("STANDARD", "STRICT") if value is namespace.get(name)),
        None,
    )
    _require(member is not None, "original JAX config enum member differs")
    storage = object.__getattribute__(value, "__dict__")
    _require(type(storage) is dict, "original JAX config enum storage differs")
    name, literal = storage.get("_name_"), storage.get("_value_")
    _require(
        type(name) is str
        and type(literal) is str
        and name == member
        and literal == {"STANDARD": "standard", "STRICT": "strict"}[member],
        "original JAX config enum literal storage differs",
    )
    return {
        "kind": "native-jax-enum",
        "type_module": "jax._src.config",
        "type_name": "NumpyDtypePromotion",
        "type_identity": id(enum_type),
        "member": name,
        "member_identity": id(value),
        "value": literal,
    }


def _config_values(config: Any) -> dict[str, Any]:
    values = {
        name: getattr(config, name)
        for name in (
            "x64_enabled",
            "jax_default_matmul_precision",
            "jax_platforms",
            "jax_numpy_dtype_promotion",
            "jax_numpy_rank_promotion",
        )
    }
    _require(
        type(values["x64_enabled"]) is bool, "original x64 config is not an exact bool"
    )
    for name, value in values.items():
        if name != "x64_enabled":
            if value is None or (type(value) is str and len(value) <= 256):
                continue
            _require(
                name == "jax_numpy_dtype_promotion",
                "original JAX config requires literal string or None",
            )
            values[name] = _dtype_promotion_enum(value)
    return values


class _Observation:
    def __init__(self, folder: Path, nodeid: str, owner: types.ModuleType) -> None:
        self.folder = folder
        self.owner = owner
        self.callback = self._profile
        self.active = False
        self.previous: Any = None
        self.codes: dict[str, Any] = {}
        self.functions: dict[str, Any] = {}
        self.retained: dict[int, Any] = {}
        self.before_files: dict[str, Any] = {}
        self.settings: Any = None
        self.context: Any = None
        self.jax: Any = None
        self.config: Any = None
        self.compat: Any = None
        self.numpy: Any = None
        self.model_type: Any = None
        self.criteria_type: Any = None
        self.report: dict[str, Any] = {
            "status": "UNVERIFIED",
            "nodeid": nodeid,
            "scope": "Passive return observations during the first original batch call",
            "parameter_order": {
                "group": "ties",
                "target": "float16",
                "source": "float64",
            },
            "artifact_events": [],
            "runtime_events": [],
            "comparison_events": [],
            "observer_errors": [],
            "native_references_recomputed_by_observer": False,
            "inputs_criteria_or_assertions_changed": False,
            "original_phase_receipt": "receipt.json",
            "original_phase_events": "events.jsonl",
            "original_junit": "junit.xml",
            "numerical_acceptance_claimed_by_observer": False,
        }

    def error(self, stage: str, error: BaseException) -> None:
        if len(self.report["observer_errors"]) < 16:
            self.report["observer_errors"].append({"stage": stage, **_error(error)})
        self.report["status"] = "UNVERIFIED"

    def write(self) -> None:
        try:
            payload = json.dumps(self.report, indent=2, ensure_ascii=False) + "\n"
            _require(
                len(payload.encode()) <= _MAX_REPORT_BYTES,
                "diagnostic JSON exceeds finite bound",
            )
            (self.folder / REPORT_NAME).write_text(payload)
        except BaseException as error:
            # Diagnostics cannot replace the original test outcome, including
            # when the filesystem or output stream itself is unavailable.
            try:
                print("[source-cast-observer] UNVERIFIED: " + type(error).__name__)
            except BaseException:
                pass

    def start(self) -> None:
        _require(self.report["nodeid"] == NODE_ID, "unexpected diagnostic node")
        _require(self.folder.is_dir(), "batch folder must already exist")
        self.previous = sys.getprofile()
        if self.previous is not None:
            self.report["reason"] = "preexisting-profile-preserved"
            self.write()
            return
        modules = sys.modules
        self.numpy = modules.get("numpy")
        precision = modules.get("jax2onnx.precision")
        self.jax = modules.get("jax")
        self.compat = modules.get("jax2onnx._compat.jax")
        for module in (self.numpy, precision, self.jax, self.compat, self.owner):
            _require(
                type(module) is types.ModuleType,
                "required original module was not already loaded",
            )
        _require(
            Path(vars(self.owner)["__file__"]).resolve()
            == (_ROOT / NODE_ID.split("::", 1)[0]).resolve(),
            "original owner module source differs",
        )
        onnx = modules.get("onnx")
        _require(type(onnx) is types.ModuleType, "original ONNX module was not loaded")
        self.model_type = vars(onnx)["ModelProto"]
        self.criteria_type = vars(precision)["SampleCriteria"]
        self.functions = {
            "artifact": vars(self.owner)["_artifact"],
            "runtime": vars(precision)["_run_cpu_samples"],
            "compare": vars(precision)["_compare"],
        }
        self.report["function_code_provenance"] = {}
        for role, function in self.functions.items():
            _require(
                type(function) is types.FunctionType,
                "observed binding is not an exact Python function",
            )
            code = function.__code__
            payload = marshal.dumps(code)
            _require(len(payload) <= _MAX_FILE_BYTES, "function code exceeds bound")
            self.codes[role] = code
            self.retained[id(function)] = function
            self.retained[id(code)] = code
            self.report["function_code_provenance"][role] = {
                "function_identity": id(function),
                "code_identity": id(code),
                "filename": code.co_filename,
                "firstlineno": code.co_firstlineno,
                "name": code.co_name,
                "marshal_sha256": hashlib.sha256(payload).hexdigest(),
            }
        self.before_files = {
            relative: _file(_ROOT / relative) for relative in _SOURCE_FILES
        }
        self.config = vars(self.jax)["config"]
        self.context = vars(self.compat)["native_context_snapshot"]()
        self.settings = self._settings()
        self.report.update(
            status="OBSERVING",
            python=sys.version,
            executable=sys.executable,
            prefix=sys.prefix,
            base_prefix=sys.base_prefix,
            soabi=sysconfig.get_config_var("SOABI"),
            byteorder=sys.byteorder,
            source_files_before=self.before_files,
            jax_config_before=self.settings,
            event_commit=os.environ.get("GITHUB_SHA"),
            head_ref=os.environ.get("GITHUB_HEAD_REF"),
            runtime_environment={
                name: os.environ.get(name)
                for name in (
                    "XLA_FLAGS",
                    "JAX_PLATFORMS",
                    "JAX_ENABLE_X64",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NPY_DISABLE_CPU_FEATURES",
                )
            },
        )
        cpu_path = Path("/proc/cpuinfo")
        if cpu_path.is_file():
            with cpu_path.open() as stream:
                raw = stream.read(_MAX_FILE_BYTES + 1)
            _require(len(raw.encode()) <= _MAX_FILE_BYTES, "CPU metadata exceeds bound")
            cpu = {}
            for line in raw.splitlines():
                key, separator, value = line.partition(":")
                key = key.strip()
                if separator and key in ("model name", "flags") and key not in cpu:
                    _require(len(value) <= 8192, "CPU capability text exceeds bound")
                    cpu[key] = value.strip()
            self.report["cpu_observation"] = cpu
        # Recheck after lazy diagnostic setup. Never displace another owner.
        if sys.getprofile() is not self.previous:
            self.report.update(
                status="UNVERIFIED", reason="profile-changed-before-start"
            )
            self.write()
            return
        self.write()
        sys.setprofile(self.callback)
        self.active = True

    def _settings(self) -> dict[str, Any]:
        _require(
            sys.modules.get("jax") is self.jax
            and vars(self.jax).get("config") is self.config,
            "original JAX module or config binding changed",
        )
        return _config_values(self.config)

    def _array(self, value: Any) -> dict[str, Any]:
        _require(
            type(value) is vars(self.numpy)["ndarray"],
            "observed value requires exact NumPy storage",
        )
        _require(
            value.dtype.kind in "fiu" and value.dtype.itemsize in (2, 4, 8),
            "unexpected observed array dtype",
        )
        _require(
            value.size <= 64 and value.nbytes <= 512,
            "observed array exceeds original-input bound",
        )
        self.retained[id(value)] = value
        _require(len(self.retained) <= 128, "retained observation count exceeds bound")
        payload = value.tobytes(order="C")
        width = value.dtype.itemsize
        return {
            "identity": id(value),
            "dtype": value.dtype.name,
            "dtype_str": value.dtype.str,
            "shape": list(value.shape),
            "storage_hex": payload.hex(),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "element_storage_hex": [
                payload[index : index + width].hex()
                for index in range(0, len(payload), width)
            ],
        }

    def _labels(self, frame: Any) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for _ in range(16):
            if frame is None:
                break
            for name in (
                "source",
                "target",
                "group",
                "optimization",
                "reference_index",
            ):
                value = frame.f_locals.get(name)
                if name not in result and type(value) in (str, bool, int):
                    result[name] = _literal(value)
            frame = frame.f_back
        return result

    def _profile(self, frame: Any, event: str, argument: Any) -> None:
        if event != "return":
            return
        try:
            if frame.f_code is self.codes["compare"]:
                events = self.report["comparison_events"]
                _require(len(events) < 8, "comparison event count exceeds bound")
                local = frame.f_locals
                actual, reference = self._array(local["actual"]), self._array(
                    local["reference"]
                )
                _require(
                    actual["dtype"] == reference["dtype"]
                    and actual["shape"] == reference["shape"],
                    "original comparison did not have equal typed shapes",
                )
                criteria = local["criteria"]
                _require(
                    type(criteria) is self.criteria_type,
                    "original criteria class differs",
                )
                events.append(
                    {
                        "labels": self._labels(frame),
                        "criteria": {
                            name: _literal(getattr(criteria, name))
                            for name in ("mode", "rtol", "atol", "equal_nan")
                        },
                        "actual": actual,
                        "reference": reference,
                        "different_indices": [
                            index
                            for index, (left, right) in enumerate(
                                zip(
                                    actual["element_storage_hex"],
                                    reference["element_storage_hex"],
                                    strict=True,
                                )
                            )
                            if left != right
                        ],
                        "original_result": _literal(argument),
                    }
                )
            elif frame.f_code is self.codes["runtime"]:
                events = self.report["runtime_events"]
                _require(
                    len(events) < 8 and type(argument) is tuple,
                    "unexpected original runtime return",
                )
                local = frame.f_locals
                feeds = local["feeds"]
                _require(
                    type(feeds) is dict and len(feeds) == 1,
                    "unexpected original runtime feeds",
                )
                _require(len(argument) <= 8, "runtime output count exceeds bound")
                ort = sys.modules.get("onnxruntime")
                _require(
                    type(ort) is types.ModuleType
                    and type(local["session"]) is vars(ort)["InferenceSession"],
                    "original runtime session class differs",
                )
                providers = local["session"].get_providers()
                _require(len(providers) <= 8, "provider count exceeds bound")
                events.append(
                    {
                        "labels": self._labels(frame),
                        "inputs": {
                            name: self._array(value) for name, value in feeds.items()
                        },
                        "outputs": [self._array(value) for value in argument],
                        "providers": _literal(providers),
                    }
                )
            elif frame.f_code is self.codes["artifact"]:
                events = self.report["artifact_events"]
                _require(
                    not events and type(argument) is tuple and len(argument) == 4,
                    "unexpected original artifact return",
                )
                proto, references = argument[1], argument[3]
                _require(type(proto) is self.model_type, "original model class differs")
                _require(
                    type(references) is list and len(references) == 1,
                    "unexpected original native-reference groups",
                )
                _require(
                    type(references[0]) is tuple
                    and len(references[0]) == 2
                    and all(
                        type(values) is list and len(values) <= 8
                        for values in references[0]
                    ),
                    "original native-reference output count differs",
                )
                _require(
                    len(proto.graph.node) <= 64, "artifact node count exceeds bound"
                )
                for node in proto.graph.node:
                    _require(
                        len(node.op_type) <= 128
                        and len(node.input) <= 16
                        and len(node.output) <= 16
                        and len(node.attribute) <= 16,
                        "artifact node metadata exceeds bound",
                    )
                    _require(
                        all(len(name) <= 256 for name in (*node.input, *node.output))
                        and all(len(attr.name) <= 128 for attr in node.attribute),
                        "artifact symbol metadata exceeds bound",
                    )
                payload = proto.SerializeToString()
                _require(len(payload) <= 1024 * 1024, "original model exceeds bound")
                (self.folder / MODEL_NAME).write_bytes(payload)
                events.append(
                    {
                        "labels": self._labels(frame),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "native_eager": [
                            self._array(value) for value in references[0][0]
                        ],
                        "native_fresh_jit": [
                            self._array(value) for value in references[0][1]
                        ],
                        "graph_ops": [
                            {
                                "op_type": node.op_type,
                                "input": list(node.input),
                                "output": list(node.output),
                                "attributes": [
                                    {
                                        "name": attr.name,
                                        "type": attr.type,
                                        "int": attr.i,
                                    }
                                    for attr in node.attribute
                                ],
                            }
                            for node in proto.graph.node
                        ],
                    }
                )
        except BaseException as error:
            self.error("return-observation", error)

    def _native_module(self, name: str) -> dict[str, Any]:
        module = sys.modules.get(name)
        _require(
            type(module) is types.ModuleType,
            "runtime module was not loaded: " + name,
        )
        dist = importlib.metadata.distribution(name)
        filename = Path(vars(module)["__file__"]).resolve()
        files = dist.files or ()
        _require(len(files) <= 20000, "distribution RECORD entry count exceeds bound")
        matches = [
            entry
            for entry in files
            if Path(dist.locate_file(entry)).resolve() == filename
        ]
        _require(len(matches) == 1, "loaded module lacks unique RECORD entry")
        entry = matches[0]
        observed = _file(filename)
        _require(
            entry.hash is not None and entry.hash.mode == "sha256",
            "loaded module lacks a SHA256 RECORD digest",
        )
        digest = bytes.fromhex(observed["sha256"])
        actual = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
        _require(actual == entry.hash.value, "loaded module differs from RECORD")
        return {
            "version": dist.version,
            **observed,
            "record_digest_verified": True,
            "wheel_archive_hash_claimed": False,
        }

    def _native_modules(self) -> dict[str, Any]:
        names = ("jax", "jaxlib", "numpy", "scipy", "onnxruntime")
        snapshot = {
            name: {
                "sys_modules_contains_name": name in sys.modules,
                "exact_module_type": type(sys.modules.get(name)) is types.ModuleType,
                "binding_identity": (
                    id(sys.modules[name]) if name in sys.modules else None
                ),
            }
            for name in names
        }
        self.report["runtime_module_snapshot"] = snapshot
        modules: dict[str, Any] = {}
        # Retain successful proofs and the complete presence snapshot even if a
        # later required module cannot be qualified.
        self.report["loaded_native_modules"] = modules
        for name in names:
            if name == "scipy" and not snapshot[name]["sys_modules_contains_name"]:
                # A cast need not import SciPy. Installed distribution metadata
                # is reported separately; do not import code to manufacture a
                # loaded-module proof or claim its RECORD has been verified.
                modules[name] = {
                    "status": "NOT_LOADED",
                    "record_digest_verified": False,
                    "wheel_archive_hash_claimed": False,
                }
                continue
            _require(
                snapshot[name]["exact_module_type"] is True,
                "runtime module was not loaded: " + name,
            )
            _require(
                id(sys.modules.get(name)) == snapshot[name]["binding_identity"],
                "runtime module binding changed: " + name,
            )
            modules[name] = {"status": "LOADED", **self._native_module(name)}
        return modules

    def _installed_native_files(self) -> None:
        manifest: dict[str, Any] = {
            "status": "UNVERIFIED",
            "evidence_kind": "INSTALLED_RECORD_FILE",
            "native_modules_imported_by_observer": False,
            "loaded_native_binary_claimed": False,
            "executed_kernel_claimed": False,
            "wheel_archive_hash_claimed": False,
            "files_started": 0,
            "declared_bytes": 0,
            "bytes_hashed": 0,
            "distributions": {},
        }
        self.report["installed_native_files"] = manifest

        def checkpoint() -> None:
            _require(
                len(json.dumps(manifest).encode()) <= _MAX_NATIVE_MANIFEST_BYTES,
                "installed manifest exceeds report bound",
            )
            self.write()

        def progress(amount: int) -> None:
            manifest["bytes_hashed"] += amount
            _require(
                manifest["bytes_hashed"] <= _MAX_NATIVE_TOTAL_BYTES,
                "installed files exceed aggregate actual byte bound",
            )

        for name in _NATIVE_DISTRIBUTIONS:
            item: dict[str, Any] = {"status": "UNVERIFIED", "files": []}
            manifest["distributions"][name] = item
            try:
                dist = importlib.metadata.distribution(name)
                _require(
                    type(dist) is importlib.metadata.PathDistribution,
                    "installed proof requires an exact filesystem distribution",
                )
                native_path = vars(dist).get("_path")
                _require(
                    type(native_path) is type(Path()),
                    "installed distribution path requires an exact native Path",
                )
                located = dist.locate_file("")
                _require(
                    type(located) is type(Path()), "installed root is not a native Path"
                )
                root, metadata = located.resolve(strict=True), native_path.resolve(
                    strict=True
                )
                _require(
                    metadata.parent == root,
                    "installed metadata escapes distribution root",
                )
                metadata_path, record_path = metadata / "METADATA", metadata / "RECORD"
                item.update(root=str(root), metadata_file={}, record_file={})
                payload = _read_installed_file(
                    metadata_path, _MAX_FILE_BYTES, item["metadata_file"], collect=True
                )
                fields = Parser().parsestr(payload.decode("utf-8"), headersonly=True)
                names, versions = fields.get_all("Name", []), fields.get_all(
                    "Version", []
                )
                _require(
                    len(names) == len(versions) == 1
                    and type(names[0]) is str
                    and type(versions[0]) is str
                    and len(names[0]) <= 256
                    and len(versions[0]) <= 256
                    and names[0].lower().replace("_", "-") == name,
                    "installed distribution metadata differs",
                )
                item["version"] = versions[0]
                payload = _read_installed_file(
                    record_path,
                    _MAX_NATIVE_RECORD_BYTES,
                    item["record_file"],
                    collect=True,
                )
                seen: set[str] = set()
                item["record_entries"] = 0
                for row in csv.reader(io.StringIO(payload.decode("utf-8"))):
                    item["record_entries"] += 1
                    _require(
                        item["record_entries"] <= _MAX_NATIVE_RECORD_ROWS,
                        "installed RECORD row bound exceeded",
                    )
                    _require(
                        len(row) == 3 and all(len(field) <= 4096 for field in row),
                        "installed RECORD fields exceed schema bounds",
                    )
                    relative, recorded_digest, recorded_size = row
                    if re.search(r"\.so(?:\.[0-9]+)*$", relative) is None:
                        continue
                    parts = relative.split("/")
                    _require(
                        len(relative) <= 1024
                        and parts
                        and all(part not in ("", ".", "..") for part in parts)
                        and not Path(relative).is_absolute()
                        and relative not in seen,
                        "installed native RECORD path is unsafe or duplicated",
                    )
                    seen.add(relative)
                    path = root.joinpath(*parts)
                    _require(
                        path.resolve(strict=True).is_relative_to(root),
                        "installed native file escapes distribution root",
                    )
                    _require(
                        len(str(path)) <= 4096, "installed native path exceeds bound"
                    )
                    _require(
                        recorded_digest.startswith("sha256=")
                        and len(recorded_digest) == 50,
                        "installed native file lacks SHA256 RECORD digest",
                    )
                    encoded = recorded_digest[7:]
                    expected = base64.b64decode(
                        encoded + "=", altchars=b"-_", validate=True
                    )
                    _require(
                        base64.urlsafe_b64encode(expected).rstrip(b"=").decode()
                        == encoded
                        and len(expected) == 32,
                        "installed RECORD digest is not canonical SHA256",
                    )
                    _require(
                        recorded_size.isascii()
                        and recorded_size.isdecimal()
                        and len(recorded_size) <= 20,
                        "installed native file lacks exact RECORD size",
                    )
                    size = int(recorded_size)
                    _require(
                        size <= _MAX_NATIVE_FILE_BYTES,
                        "installed native file exceeds per-file bound",
                    )
                    _require(
                        manifest["files_started"] < _MAX_NATIVE_FILES,
                        "installed native file count exceeds bound",
                    )
                    _require(
                        manifest["declared_bytes"] + size <= _MAX_NATIVE_TOTAL_BYTES,
                        "installed files exceed aggregate declared byte bound",
                    )
                    proof: dict[str, Any] = {
                        "status": "UNVERIFIED",
                        "evidence_kind": "INSTALLED_RECORD_FILE",
                        "distribution_path": relative,
                        "path": str(path),
                        "record_declared_bytes": size,
                        "record_declared_sha256": expected.hex(),
                        "bytes_hashed": 0,
                    }
                    _require(
                        len(json.dumps(manifest).encode())
                        + len(json.dumps(proof).encode())
                        + 4096
                        <= _MAX_NATIVE_MANIFEST_BYTES,
                        "installed manifest exceeds report bound",
                    )
                    item["files"].append(proof)
                    manifest["files_started"] += 1
                    manifest["declared_bytes"] += size
                    checkpoint()
                    _read_installed_file(
                        path, size, proof, progress=progress, checkpoint=checkpoint
                    )
                    _require(
                        proof["bytes_hashed"] == size,
                        "installed native size differs from RECORD",
                    )
                    _require(
                        proof["sha256"] == expected.hex(),
                        "installed native digest differs from RECORD",
                    )
                    proof.update(
                        status="VERIFIED",
                        record_digest_verified=True,
                        record_size_verified=True,
                    )
                    checkpoint()
                _require(
                    bool(item["files"]),
                    "installed distribution has no native RECORD files",
                )
                for path, key in (
                    (metadata_path, "metadata_file"),
                    (record_path, "record_file"),
                ):
                    item[key]["stat_end"] = _installed_stat(path)
                    _require(
                        item[key]["stat_end"] == item[key]["stat_before"],
                        "installed metadata or RECORD changed during audit",
                    )
                item["status"] = "VERIFIED"
                checkpoint()
            except BaseException as error:
                item["error"] = _error(error)
                self.write()
                raise
        manifest["status"] = "VERIFIED"
        checkpoint()

    def _end_fences(self) -> None:
        self.report["previous_profile_restored"] = (
            self.report.get("previous_profile_restored") is True
            and sys.getprofile() is self.previous
        )
        for name in (
            "native_context_restored",
            "source_settings_restored",
            "source_fence_held",
            "code_identity_fence_held",
        ):
            self.report[name] = False
        vars(self.compat)["require_native_context_restored"](self.context)
        self.report["native_context_restored"] = True
        self.report["jax_config_end"] = self._settings()
        self.report["source_settings_restored"] = (
            self.report["jax_config_end"] == self.settings
        )
        self.report["source_files_end"] = {
            relative: _file(_ROOT / relative) for relative in _SOURCE_FILES
        }
        self.report["source_fence_held"] = (
            self.report["source_files_end"] == self.before_files
        )
        self.report["code_identity_fence_held"] = all(
            self.functions[role].__code__ is code for role, code in self.codes.items()
        )

    def finish(self) -> None:
        if not self.active:
            self.write()
            return
        try:
            if sys.getprofile() is self.callback:
                sys.setprofile(self.previous)
                self.report["previous_profile_restored"] = (
                    sys.getprofile() is self.previous
                )
            else:
                self.report["previous_profile_restored"] = False
                self.report["reason"] = "original-call-changed-profile-preserved"
                self.report["status"] = "UNVERIFIED"
            self.active = False
            vars(self.compat)["require_native_context_restored"](self.context)
            self.report["native_context_restored"] = True
            self.report["jax_config_end"] = self._settings()
            self.report["source_settings_restored"] = (
                self.report["jax_config_end"] == self.settings
            )
            self.report["source_files_end"] = {
                relative: _file(_ROOT / relative) for relative in _SOURCE_FILES
            }
            self.report["source_fence_held"] = (
                self.report["source_files_end"] == self.before_files
            )
            self.report["code_identity_fence_held"] = all(
                self.functions[role].__code__ is code
                for role, code in self.codes.items()
            )
            self.report["distributions"] = {
                name: importlib.metadata.version(name)
                for name in (
                    "jax",
                    "jaxlib",
                    "numpy",
                    "scipy",
                    "flax",
                    "onnxruntime",
                    "onnx",
                    "onnx-ir",
                    "equinox",
                    "ml-dtypes",
                    "jax2onnx",
                )
            }
            self._native_modules()
            self._installed_native_files()
            complete = (
                len(self.report["artifact_events"]) == 1
                and self.report["runtime_events"]
                and self.report["comparison_events"]
            )
            fences = all(
                self.report.get(name) is True
                for name in (
                    "previous_profile_restored",
                    "native_context_restored",
                    "source_settings_restored",
                    "source_fence_held",
                    "code_identity_fence_held",
                )
            )
            if complete and fences and not self.report["observer_errors"]:
                self.report["status"] = "ORIGINAL_CALL_WORDS_OBSERVED"
            else:
                self.report["status"] = "UNVERIFIED"
        except BaseException as error:
            self.error("finish", error)
        finally:
            try:
                self._end_fences()
                if not all(
                    self.report.get(name) is True
                    for name in (
                        "previous_profile_restored",
                        "native_context_restored",
                        "source_settings_restored",
                        "source_fence_held",
                        "code_identity_fence_held",
                    )
                ):
                    self.report["status"] = "UNVERIFIED"
            except BaseException as error:
                self.error("finish-final-fences", error)
            self.write()


def start_observation(
    folder: Path, nodeid: str, owner: types.ModuleType
) -> _Observation:
    observation = _Observation(folder, nodeid, owner)
    try:
        observation.start()
    except BaseException as error:
        observation.error("start", error)
        # Setup errors cannot leak a profile that was installed just before an
        # unexpected observer-side error.
        if sys.getprofile() is observation.callback:
            try:
                sys.setprofile(observation.previous)
            except BaseException as restore_error:
                observation.error("start-profile-restore", restore_error)
        observation.write()
    return observation
