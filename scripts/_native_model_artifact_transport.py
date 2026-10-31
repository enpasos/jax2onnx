# scripts/_native_model_artifact_transport.py

"""Passive, stdlib-only retention of the existing original-model payloads.

This proves file transport, never model semantics or numerical acceptance.
The original worker outcome remains authoritative when transport is incomplete.
"""

from __future__ import annotations

import ast
from contextlib import contextmanager
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any
import xml.etree.ElementTree as ET


ENVIRONMENT_KEY = "JAX2ONNX_NATIVE_BINDINGS_EVIDENCE_DIR"
OWNER = "tests/extra_tests/converter/test_source_model_native_bindings_exports.py"
_ROLES = ("eager", "jit-first-for-sample", "same-jit-warmed")
_INDICES = (0, 1, 3, 4, 20)
_PREFIX = OWNER + "::test_original_full_model_native_bindings_cpu_gate"
_PAYLOADS = {
    f"{_PREFIX}[raw-x64-{flag}-{index}]": f"{index:02d}-x64-{flag}"
    for flag in ("off", "on")
    for index in _INDICES
}
_CONTROLS = (
    OWNER + "::test_frozen_five_owner_output_contract_rejects_corruption",
    OWNER + "::test_reference_scaled_direction_and_all_native_roles_are_required",
)
_CASE_RAW_OUTPUT_BYTES = {
    0: 901447680,
    1: 220200960,
    3: 220200960,
    4: 220200960,
    20: 560,
}
_CRITERION = "67cd821a4c40e0464f09a450e7c66afac99a0cef26b8569b17bd842f51c60c51"
_CHUNK = 1024 * 1024
_FILE_LIMIT = 4 * 1024**3
_TOTAL_LIMIT = 64 * 1024**3
_JSON_LIMIT = 64 * 1024**2
_HEADER_LIMIT = 65536
_MANIFEST_LIMIT = 16 * 1024**2
_ENTRY_LIMIT = 8192
_FLOOR = 4 * 1024**3
_STABLE = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_uid",
    "st_gid",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
    "st_nlink",
)


class ArtifactResourceCensor(RuntimeError):
    """The complete planned run is retained without starting another worker."""


def _resource_error(error: BaseException) -> bool:
    return isinstance(error, (ArtifactResourceCensor, MemoryError)) or (
        isinstance(error, OSError)
        and error.errno in (errno.ENOSPC, errno.EDQUOT, errno.ENOMEM)
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _digest(value: Any) -> bool:
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


def _identity(value: os.stat_result) -> dict[str, int]:
    return {name: int(getattr(value, name)) for name in _STABLE}


def _free(path: Path) -> int:
    value = os.statvfs(path)
    return value.f_bavail * value.f_frsize


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        _require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _json(raw: bytes) -> Any:
    def invalid(value: str) -> Any:
        raise ValueError("nonfinite JSON constant")

    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=invalid)


def _array(value: Any) -> dict[str, Any]:
    _require(
        type(value) is dict
        and set(value) == {"dtype", "shape", "weak_type", "bytes", "sha256"},
        "array metadata schema differs",
    )
    _require(
        type(value["dtype"]) is str
        and type(value["shape"]) is list
        and len(value["shape"]) <= 64,
        "array metadata types differ",
    )
    _require(
        all(type(dim) is int and 0 <= dim <= 2**31 - 1 for dim in value["shape"]),
        "array dimensions differ",
    )
    _require(
        type(value["weak_type"]) is bool
        and type(value["bytes"]) is int
        and 0 <= value["bytes"] <= _FILE_LIMIT
        and _digest(value["sha256"]),
        "array storage metadata differs",
    )
    return value


def _npy_header(read: Any) -> dict[str, Any]:
    prefix = read(8)
    _require(len(prefix) == 8 and prefix[:6] == b"\x93NUMPY", "NPY magic differs")
    version = tuple(prefix[6:])
    _require(version in ((1, 0), (2, 0), (3, 0)), "NPY version unsupported")
    width = 2 if version == (1, 0) else 4
    length_raw = read(width)
    _require(len(length_raw) == width, "NPY header length missing")
    length = int.from_bytes(length_raw, "little")
    _require(0 < length <= _HEADER_LIMIT, "NPY header exceeds bound")
    raw = read(length)
    _require(len(raw) == length and raw.endswith(b"\n"), "NPY header truncated")
    tree = ast.parse(
        raw.decode("utf-8" if version == (3, 0) else "latin1").strip(), mode="eval"
    )
    _require(
        type(tree.body) is ast.Dict
        and all(
            type(key) is ast.Constant and type(key.value) is str
            for key in tree.body.keys
        ),
        "NPY header keys differ",
    )
    names = [key.value for key in tree.body.keys]
    _require(len(names) == len(set(names)), "duplicate NPY header key")
    header = ast.literal_eval(tree)
    _require(
        type(header) is dict and set(header) == {"descr", "fortran_order", "shape"},
        "NPY header schema differs",
    )
    _require(
        type(header["descr"]) is str
        and type(header["fortran_order"]) is bool
        and header["fortran_order"] is False,
        "NPY object/Fortran storage unsupported",
    )
    match = re.fullmatch(r"[<>=|]([biufc])([0-9]+)", header["descr"])
    _require(match is not None, "NPY nonprimitive dtype unsupported")
    kind, width_text = match.groups()
    itemsize = int(width_text)
    allowed = {
        "b": (1,),
        "i": (1, 2, 4, 8),
        "u": (1, 2, 4, 8),
        "f": (2, 4, 8, 16),
        "c": (8, 16, 32),
    }
    _require(itemsize in allowed[kind], "NPY dtype width unsupported")
    shape = header["shape"]
    _require(
        type(shape) is tuple
        and len(shape) <= 64
        and all(type(dim) is int and 0 <= dim <= 2**31 - 1 for dim in shape),
        "NPY shape differs",
    )
    size = math.prod(shape) * itemsize
    _require(size <= _FILE_LIMIT, "NPY data exceeds bound")
    return {
        "dtype": header["descr"],
        "shape": list(shape),
        "bytes": size,
        "header_bytes": 8 + len(length_raw) + length,
        "fortran_order": False,
    }


@contextmanager
def _parent(path: Path) -> Any:
    _require(
        path.is_absolute() and ".." not in path.parts,
        "file path is not absolute and confined",
    )
    descriptor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in path.parts[1:-1]:
            following = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = following
        yield descriptor, path.name
    finally:
        os.close(descriptor)


def _path_stat(path: Path) -> os.stat_result:
    with _parent(path) as (descriptor, name):
        return os.stat(name, dir_fd=descriptor, follow_symlinks=False)


def _children(path: Path) -> list[Path]:
    result = []
    with _parent(path / "_transport_directory_anchor") as (descriptor, _):
        with os.scandir(descriptor) as entries:
            for entry in entries:
                _require(
                    len(result) < _ENTRY_LIMIT,
                    "directory entry bound exceeded; inventory incomplete",
                )
                result.append(path / _name(entry.name))
    return sorted(result, key=lambda value: value.name)


def _file(
    path: Path,
    *,
    collect: bool = False,
    array: dict[str, Any] | None = None,
    file_sha256: str | None = None,
) -> tuple[dict[str, Any], bytes | None]:
    before = _path_stat(path)
    _require(
        stat.S_ISREG(before.st_mode)
        and before.st_uid == os.getuid()
        and before.st_nlink == 1,
        "file is not an owned unique regular file",
    )
    limit = _JSON_LIMIT if collect else _FILE_LIMIT
    _require(before.st_size <= limit, "file exceeds declared bound")
    with _parent(path) as (parent, name):
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
    proof: dict[str, Any] = {
        "bytes": before.st_size,
        "before": _identity(before),
        "errors": [],
    }
    chunks: list[bytes] = []
    count = 0
    file_hash = hashlib.sha256()
    data_hash = hashlib.sha256()
    header = None
    with os.fdopen(fd, "rb") as stream:
        _require(
            _identity(os.fstat(stream.fileno())) == proof["before"],
            "opened file identity differs",
        )

        def read(length: int) -> bytes:
            nonlocal count
            value = stream.read(length)
            count += len(value)
            _require(
                count <= before.st_size, "file grew beyond its observed byte bound"
            )
            file_hash.update(value)
            if collect:
                chunks.append(value)
            return value

        if path.suffix == ".npy":
            try:
                header = _npy_header(read)
                _require(
                    header["header_bytes"] + header["bytes"] == before.st_size,
                    "NPY file/data lengths differ",
                )
            except (ValueError, SyntaxError, UnicodeError, RecursionError) as error:
                proof["errors"].append(type(error).__name__ + ":NPY-header")
        while value := read(_CHUNK):
            if header is not None:
                data_hash.update(value)
        after_fd = _identity(os.fstat(stream.fileno()))
    try:
        after_path = _identity(_path_stat(path))
    except OSError as error:
        if _resource_error(error):
            raise
        after_path = None
        proof["errors"].append("file-path-unavailable-after-read")
    proof.update(
        sha256=file_hash.hexdigest(),
        after=after_path,
        identity_held=proof["before"] == after_fd == after_path
        and count == before.st_size,
    )
    if not proof["identity_held"]:
        proof["errors"].append("file-identity-changed")
    if file_sha256 is not None and proof["sha256"] != file_sha256:
        proof["errors"].append("whole-file-sha256-differs")
    if header is not None:
        proof["array"] = dict(header, sha256=data_hash.hexdigest())
        if array is None:
            proof["errors"].append("NPY-array-reference-unavailable")
        else:
            expected = _array(array)
            if any(
                proof["array"][key] != expected[key]
                for key in ("dtype", "shape", "bytes", "sha256")
            ):
                proof["errors"].append("NPY-array-reference-differs")
    elif array is not None:
        proof["errors"].append("NPY-array-data-unavailable")
    return proof, b"".join(chunks) if collect else None


def _name(value: Any) -> str:
    _require(
        type(value) is str
        and 0 < len(value) <= 255
        and value not in (".", "..")
        and "/" not in value
        and "\\" not in value
        and "\0" not in value,
        "unsafe relative payload name",
    )
    return value


def _requirements(record: Any, index: int) -> dict[str, tuple[Any, Any]]:
    _require(
        type(record) is dict
        and record.get("status") == "PASS_FINITE_ORIGINAL_FIVE_OWNER_SUBGATE",
        "original model receipt is incomplete or failed",
    )
    _require(
        record.get("raw_criteria", "absent") is None
        and record.get("criterion_sha256") == _CRITERION
        and record.get("artifact_sidecars") == [],
        "original criterion/inline artifact assertion differs",
    )
    _require(
        _digest(record.get("contract_sha256"))
        and _sha(
            json.dumps(
                record["contract"],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        )
        == record["contract_sha256"],
        "original contract digest differs",
    )
    required: dict[str, tuple[Any, Any]] = {
        "receipt.json": (None, None),
        "model.onnx": (None, None),
        "original-report.json": (None, None),
        "original-native-symbolic-trace.json": (None, None),
        "activated-full-world-symbolic-trace.json": (None, None),
    }
    if index != 20:
        required["activated-inlined-original-symbolic-trace.json"] = (None, None)
    definitions = record["function_definitions"]
    _require(
        type(definitions) is list and len(definitions) <= 1024,
        "function definition count differs",
    )
    for number in range(1, len(definitions) + 1):
        required[f"source-function-body-{number}.json"] = (None, None)
        required[f"source-function-definition-{number}.json"] = (None, None)
    feeds = record["feeds"]
    _require(
        type(feeds) is dict and set(feeds) == {"2", "5"}, "original B2/B5 feeds missing"
    )
    for batch in (2, 5):
        _require(
            type(feeds[str(batch)]) is list and 0 < len(feeds[str(batch)]) <= 64,
            "original feed arity differs",
        )
        for port, value in enumerate(feeds[str(batch)]):
            required[f"feed-B{batch}-{port}.npy"] = (_array(value), None)
    keys = [
        f"B{batch}-{flag}"
        for batch in (2, 5)
        for flag in (("staticFalse",) if index == 20 else ("True", "False"))
    ]
    _require(
        type(record["native"]) is dict and set(record["native"]) == set(keys),
        "original native sample coverage differs",
    )
    expected_stages = ["native-" + key + "-" + role for key in keys for role in _ROLES]
    expected_stages += ["export-original-source-callable"]
    for mode in ("disabled", "all"):
        expected_stages += ["session-" + mode] + [
            "cpu-" + mode + "-" + key for key in keys
        ]
    events = record["events"]
    _require(
        type(events) is list
        and [event["stage"] for event in events] == expected_stages,
        "original event order/coverage differs",
    )
    artifact_sha = None
    for number, event in enumerate(events, 1):
        stage = event["stage"]
        _require(
            event.get("status") == "ACTUAL_RETURN"
            and not any(
                key in event
                for key in ("postcheck_failure", "precheck_failure", "exception_type")
            ),
            "original event did not return successfully",
        )
        if stage == "export-original-source-callable":
            _require(
                event.get("artifact_path") == "model.onnx"
                and _digest(event.get("artifact_sha256")),
                "original model file binding missing",
            )
            artifact_sha = event["artifact_sha256"]
            required["model.onnx"] = (None, artifact_sha)
        elif stage.startswith(("native-", "cpu-")):
            arrays = event["arrays"]
            _require(
                type(arrays) is list and len(arrays) == 1,
                "original single-array return missing",
            )
            value = arrays[0]
            path = _name(value["path"])
            _require(
                path == f"event-{number:03d}-array-0.npy" and _digest(value["sha256"]),
                "original return file order/hash differs",
            )
            _require(path not in required, "duplicate original payload reference")
            required[path] = (_array(value["actual"]), value["sha256"])
    _require(
        type(record["cpu"]) is dict and set(record["cpu"]) == {"disabled", "all"},
        "original target profiles differ",
    )
    for key in keys:
        sample = record["native"][key]
        _require(set(sample["roles"]) == set(_ROLES), "original native roles differ")
        for role in _ROLES:
            event = events[expected_stages.index("native-" + key + "-" + role)]
            _require(
                _array(sample["roles"][role]["actual"]) == event["arrays"][0]["actual"],
                "native role/raw return binding differs",
            )
    for mode in ("disabled", "all"):
        cpu = record["cpu"][mode]
        _require(
            cpu["artifact_sha256"] == artifact_sha and set(cpu["runs"]) == set(keys),
            "target/model sample binding differs",
        )
        for key in keys:
            run = cpu["runs"][key]
            comparisons = run["comparisons"]
            _require(
                set(comparisons) == set(_ROLES)
                and all(
                    value.get("criterion_sha256") == _CRITERION
                    and value.get("all_components") is True
                    for value in comparisons.values()
                ),
                "original comparison role/criterion coverage differs",
            )
            actual_feeds = [_array(value) for value in run["feed"].values()]
            batch = key[1 : key.index("-")]
            for original in feeds[batch]:
                _require(
                    original in actual_feeds,
                    "target feed does not bind original input bytes",
                )
                actual_feeds.remove(original)
            if index == 20:
                _require(not actual_feeds, "static bridge has extra target feeds")
            else:
                expected_bool = key.endswith("-True")
                _require(
                    len(actual_feeds) == 1
                    and actual_feeds[0]
                    == {
                        "dtype": "|b1",
                        "shape": [],
                        "weak_type": False,
                        "bytes": 1,
                        "sha256": _sha(bytes([int(expected_bool)])),
                    },
                    "dynamic target control feed differs",
                )
    _require(
        record["native_input_usage_correspondence"]["artifact_sha256"] == artifact_sha,
        "native correspondence artifact differs",
    )
    return required


class NativeModelTransport:
    """Run-owned observation; exceptions never rewrite a closed pytest result."""

    def __init__(self, root: Path, checkout: str, environment: dict[str, str]) -> None:
        _require(
            type(environment) is dict
            and all(
                type(key) is type(value) is str for key, value in environment.items()
            ),
            "child environment schema differs",
        )
        self.root = root
        root_stat = _path_stat(root)
        _require(
            stat.S_ISDIR(root_stat.st_mode) and root_stat.st_uid == os.getuid(),
            "run root is not owned",
        )
        self.root_identity = (root_stat.st_dev, root_stat.st_ino)
        self.path = root / "native-model-artifact-transport.json"
        self.evidence = root / "native-model-artifacts"
        self.environment = dict(environment)
        self.external = ENVIRONMENT_KEY in environment
        self.executed: set[str] = set()
        self.held: dict[str, dict[str, int]] = {}
        self.worker_held: dict[str, dict[str, int]] = {}
        self.worker_absent: set[str] = set()
        self.total = 0
        source = Path(__file__).resolve().parents[1] / OWNER
        source_proof, _ = _file(source)
        _require(not source_proof["errors"], "original source owner identity changed")
        self.source = source
        self.sources = {str(source): source_proof}
        for path in (
            Path(__file__).resolve(),
            Path(__file__).with_name("run_pytest_batches.py").resolve(),
        ):
            proof, _ = _file(path)
            _require(not proof["errors"], "transport source file identity changed")
            self.sources[str(path)] = proof
        if not self.external:
            try:
                _path_stat(self.evidence)
            except FileNotFoundError as error:
                if _resource_error(error):
                    raise
                pass
            else:
                raise ValueError("artifact destination already exists")
            with _parent(self.evidence) as (descriptor, name):
                current = os.fstat(descriptor)
                _require(
                    (current.st_dev, current.st_ino) == self.root_identity,
                    "run root changed before artifact directory creation",
                )
                os.mkdir(name, mode=0o700, dir_fd=descriptor)
                os.fsync(descriptor)
            self.environment[ENVIRONMENT_KEY] = str(self.evidence)
            evidence_stat = _path_stat(self.evidence)
            self.evidence_identity = (evidence_stat.st_dev, evidence_stat.st_ino)
        self.record: dict[str, Any] = {
            "schema": "native-model-artifact-transport-v1",
            "status": "EXTERNAL_UNVERIFIED" if self.external else "PARTIAL",
            "environment_mode": "EXTERNAL" if self.external else "RUN_OWNED",
            "environment_key": ENVIRONMENT_KEY,
            "environment_value": self.environment[ENVIRONMENT_KEY],
            "parent_environment_mutated": False,
            "checkout": checkout,
            "source_owner": dict(source_proof, path=OWNER),
            "workers": [],
            "cases": {},
            "source_files_before": self.sources,
            "full_collection_ids": None,
            "expected_owner_ids": [],
            "errors": [],
            "resource_observations": [],
            "bounds": {
                "file": _FILE_LIMIT,
                "aggregate": _TOTAL_LIMIT,
                "json": _JSON_LIMIT,
                "npy_header": _HEADER_LIMIT,
                "entries": _ENTRY_LIMIT,
                "manifest": _MANIFEST_LIMIT,
            },
            "numerical_acceptance": "NOT_ASSERTED",
            "full_B_to_F_acceptance": "UNVERIFIED",
            "capacity_scope": "known raw-output lower bound only; model/feed/metadata/staging footprint remains unmeasured",
        }
        self._save()

    def _owned(self) -> None:
        current = _path_stat(self.root)
        _require(
            stat.S_ISDIR(current.st_mode)
            and (current.st_dev, current.st_ino) == self.root_identity,
            "run root identity changed",
        )
        if not self.external:
            current = _path_stat(self.evidence)
            _require(
                stat.S_ISDIR(current.st_mode)
                and current.st_uid == os.getuid()
                and (current.st_dev, current.st_ino) == self.evidence_identity,
                "owned evidence root identity changed",
            )

    def _error(self, stage: str, error: BaseException) -> None:
        self.record["errors"].append(
            {"stage": stage, "type": type(error).__name__, "message": str(error)[:256]}
        )
        self.record["status"] = "EXTERNAL_UNVERIFIED" if self.external else "UNVERIFIED"

    def _save(self) -> None:
        self._owned()
        raw = (
            json.dumps(self.record, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode()
        _require(len(raw) <= _MANIFEST_LIMIT, "transport manifest exceeds bound")
        if _free(self.root) < _FLOOR + len(raw):
            raise ArtifactResourceCensor("transport manifest disk floor")
        temporary = self.path.with_suffix(".json.tmp")
        with _parent(temporary) as (descriptor, temporary_name):
            current = os.fstat(descriptor)
            _require(
                (current.st_dev, current.st_ino) == self.root_identity,
                "run root changed before manifest write",
            )
            fd = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=descriptor,
            )
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(
                temporary_name,
                self.path.name,
                src_dir_fd=descriptor,
                dst_dir_fd=descriptor,
            )
            os.fsync(descriptor)

    def collection(self, nodeids: list[str]) -> None:
        self.record["full_collection_ids"] = list(nodeids)
        owner = [node for node in nodeids if node.startswith(OWNER + "::")]
        self.record["expected_owner_ids"] = owner
        unknown = set(owner) - set(_PAYLOADS) - set(_CONTROLS)
        if unknown or len(nodeids) != len(set(nodeids)):
            self._error(
                "collection", ValueError("unknown or duplicate original owner IDs")
            )
        self._save()

    def before_worker(self, folder: Path, request: dict[str, Any]) -> None:
        remaining = sum(
            _CASE_RAW_OUTPUT_BYTES[int(node.rsplit("-", 1)[1][:-1])]
            for node in self.record["expected_owner_ids"]
            if node in _PAYLOADS and node not in self.executed
        )
        try:
            self._owned()
            free = _free(self.root)
        except Exception as error:
            if _resource_error(error):
                raise
            raise ArtifactResourceCensor(
                "owned artifact root or disk observation unavailable; no worker started"
            ) from error
        needed = _FLOOR + 4 * _MANIFEST_LIMIT + (0 if self.external else remaining)
        observation = {
            "folder": folder.name,
            "free_bytes": free,
            "necessary_minimum_bytes": needed,
            "remaining_known_raw_output_bytes": remaining,
            "complete_capacity_proof": False,
        }
        self.record["resource_observations"].append(observation)
        if free < needed:
            self.record["resource_censored"] = {
                "folder": folder.name,
                "unexecuted_owner_ids": [
                    node
                    for node in self.record["expected_owner_ids"]
                    if node not in self.executed
                ],
                **observation,
            }
            try:
                self._save()
            finally:
                raise ArtifactResourceCensor(
                    "native artifact disk lower bound cannot be met; no worker started"
                )
        self._save()

    def _checkpoint(
        self, folder: Path, worker: dict[str, Any], owner_ids: list[str]
    ) -> None:
        document = {
            "schema": "native-model-closed-worker-transport-v1",
            "environment_mode": self.record["environment_mode"],
            "environment_value": self.environment[ENVIRONMENT_KEY],
            "checkout": self.record["checkout"],
            "worker": worker,
            "cases": {
                node: self.record["cases"][node]
                for node in owner_ids
                if node in self.record["cases"]
            },
            "scope": "transport only; original worker outcomes authoritative",
        }
        raw = (
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode()
        _require(len(raw) <= _MANIFEST_LIMIT, "closed-worker transport manifest bound")
        if _free(self.root) < _FLOOR + len(raw):
            raise ArtifactResourceCensor("closed-worker transport manifest disk floor")
        path = folder / "native-model-artifact-transport.json"
        with _parent(path) as (descriptor, name):
            fd = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=descriptor,
            )
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.fsync(descriptor)
        proof, _ = _file(path)
        self.worker_held[str(path)] = proof["after"]
        worker["transport_checkpoint"] = {
            "path": path.name,
            "sha256": proof["sha256"],
            "bytes": proof["bytes"],
        }

    def _read_json(self, path: Path) -> tuple[Any, dict[str, Any]]:
        proof, raw = _file(path, collect=True)
        _require(not proof["errors"], "JSON file identity changed")
        return _json(raw), proof

    def _case(self, node: str, phases: Any, returncode: Any) -> None:
        name = _PAYLOADS[node]
        folder = self.evidence / name
        result: dict[str, Any] = {
            "nodeid": node,
            "folder": name,
            "original_worker_returncode": returncode,
            "original_phases": phases,
            "status": "UNVERIFIED",
            "files": {},
            "errors": [],
        }
        self.record["cases"][node] = result
        self._save()
        self._owned()
        folder_stat = _path_stat(folder)
        _require(
            stat.S_ISDIR(folder_stat.st_mode) and folder_stat.st_uid == os.getuid(),
            "owned model case directory missing",
        )
        entries = _children(folder)
        result["actual_file_names"] = [path.name for path in entries]
        unsupported = [
            path.name for path in entries if not stat.S_ISREG(_path_stat(path).st_mode)
        ]
        if unsupported:
            result["errors"].append({"stage": "paths", "unsupported": unsupported})
        required: dict[str, tuple[Any, Any]] = {}
        try:
            record, receipt_proof = self._read_json(folder / "receipt.json")
            result["files"]["receipt.json"] = receipt_proof
            self.total += receipt_proof["bytes"]
            _require(self.total <= _TOTAL_LIMIT, "aggregate payload exceeds bound")
            self.held[str(folder / "receipt.json")] = receipt_proof["after"]
            required = _requirements(record, int(node.rsplit("-", 1)[1][:-1]))
            _require(
                set(required) == set(result["actual_file_names"]),
                "missing or extra original payload/trace files",
            )
        except Exception as error:
            if _resource_error(error):
                raise
            result["errors"].append(
                {
                    "stage": "receipt",
                    "type": type(error).__name__,
                    "message": str(error)[:256],
                }
            )
        for path in entries:
            if path.name in result["files"]:
                continue
            try:
                _require(
                    stat.S_ISREG(_path_stat(path).st_mode),
                    "unsupported payload path left untouched",
                )
                self.total += _path_stat(path).st_size
                _require(self.total <= _TOTAL_LIMIT, "aggregate payload exceeds bound")
                array, file_sha = required.get(path.name, (None, None))
                proof, raw = _file(
                    path,
                    collect=path.suffix == ".json",
                    array=array,
                    file_sha256=file_sha,
                )
                result["files"][path.name] = proof
                self.held[str(path)] = proof["after"]
                if path.suffix == ".json":
                    value = _json(raw)
                    _require(
                        type(value) in (dict, list),
                        "original trace/report JSON schema differs",
                    )
                    if path.name.startswith("source-function-definition-") and required:
                        number = int(path.stem.rsplit("-", 1)[1])
                        pair = record["function_definitions"][number - 1]
                        _require(
                            [value["domain"], value["name"]] == pair,
                            "function definition trace binding differs",
                        )
                if proof["errors"]:
                    result["errors"].append(
                        {"stage": path.name, "errors": proof["errors"]}
                    )
            except Exception as error:
                if _resource_error(error):
                    raise
                result["errors"].append(
                    {
                        "stage": path.name,
                        "type": type(error).__name__,
                        "message": str(error)[:256],
                    }
                )
            self._save()
        _require(
            returncode == 0
            and [(value["when"], value["outcome"]) for value in phases]
            == [(when, "passed") for when in ("setup", "call", "teardown")],
            "original model pytest outcome is not complete PASS",
        )
        _require(
            required and not result["errors"],
            "original model payload transport incomplete",
        )
        result["status"] = "COMPLETE_TRANSPORT_ONLY"

    def closed_worker(
        self, folder: Path, request: dict[str, Any], returncode: Any
    ) -> None:
        worker: dict[str, Any] = {
            "folder": folder.name,
            "kind": request["kind"],
            "assigned": request.get("assigned", []),
            "original_returncode": returncode,
            "files": {},
            "errors": [],
        }
        self.record["workers"].append(worker)
        owner_ids = [
            node
            for node in request.get("assigned", [])
            if node.startswith(OWNER + "::")
        ]
        duplicates = self.executed.intersection(owner_ids)
        if duplicates:
            self._error(
                folder.name, ValueError("original owner case executed more than once")
            )
            worker["duplicate_owner_ids"] = sorted(duplicates)
        self.executed.update(owner_ids)
        receipt: dict[str, Any] = {}
        documents: dict[str, bytes] = {}
        collection_events_absent = False
        for name in (
            "request.json",
            "receipt.json",
            "events.jsonl",
            "junit.xml",
            "output.log",
        ):
            try:
                self._owned()
                proof, raw = _file(folder / name, collect=name != "output.log")
                worker["files"][name] = proof
                self.worker_held[str(folder / name)] = proof["after"]
                _require(not proof["errors"], "original worker file identity changed")
                if raw is not None:
                    documents[name] = raw
            except Exception as error:
                if _resource_error(error):
                    raise
                if (
                    name == "events.jsonl"
                    and request["kind"] == "collection"
                    and isinstance(error, FileNotFoundError)
                    and error.errno == errno.ENOENT
                ):
                    # The canonical worker creates this file only for a phase
                    # or collection diagnostic. Validate the clean collect-only
                    # receipt below before accepting its absence.
                    collection_events_absent = True
                    continue
                worker["errors"].append(
                    {
                        "stage": name,
                        "type": type(error).__name__,
                        "message": str(error)[:256],
                    }
                )
                self._error(folder.name + "/" + name, error)
        try:
            saved = _json(documents["request.json"])
            _require(saved == request, "original worker request changed")
            receipt = _json(documents["receipt.json"])
            _require(
                receipt["kind"] == request["kind"]
                and receipt["supplemental"] is False
                and receipt["native_source_recomputed"] is False,
                "worker is not the canonical original call",
            )
            if request["kind"] == "batch":
                _require(
                    receipt["assigned_ids"]
                    == receipt["selected_ids"]
                    == receipt["completed"]
                    == request["assigned"],
                    "original assigned/executed item order differs",
                )
                _require(
                    receipt["full_preselection_ids"]
                    == request["full_preselection_ids"],
                    "original full collection differs",
                )
            _require(
                receipt["startup_context"]["environment"].get(ENVIRONMENT_KEY)
                == self.environment[ENVIRONMENT_KEY],
                "canonical worker evidence environment differs",
            )
            junit = ET.fromstring(documents["junit.xml"])
            cases = list(junit.iter("testcase"))
            if collection_events_absent:
                args = request["pytest_args"]
                full = receipt["full_preselection_ids"]
                _require(
                    "assigned" not in request
                    and type(args) is list
                    and all(type(arg) is str for arg in args)
                    and bool(args)
                    and args[-1] == "--collect-only"
                    and type(returncode) is int
                    and returncode == 0
                    and type(receipt["exitstatus"]) is int
                    and receipt["exitstatus"] == 0
                    and receipt["full_collection_checked"] is True
                    and receipt["assigned_ids"] == receipt["completed"] == []
                    and receipt["phases"] == {}
                    and receipt["collection_errors"]
                    == receipt["collection_skips"]
                    == []
                    and type(full) is list
                    and bool(full)
                    and all(type(node) is str for node in full)
                    and len(full) == len(set(full))
                    and receipt["selected_ids"] == full
                    and [item["nodeid"] for item in receipt["items"]] == full
                    and [item["nodeid"] for item in receipt["full_items"]] == full,
                    "missing events require a complete clean collect-only receipt",
                )
                suites = list(junit.iter("testsuite"))
                _require(
                    bool(suites)
                    and not cases
                    and all(
                        suite.attrib.get(name) == "0"
                        for suite in suites
                        for name in ("tests", "errors", "failures", "skipped")
                    ),
                    "clean collect-only JUnit must contain no test outcomes",
                )
                worker["phase_events"] = {
                    "status": "NOT_APPLICABLE_CLEAN_COLLECTION",
                    "path": "events.jsonl",
                    "reason": "canonical collect-only emitted no phases or collection diagnostics",
                }
                self.worker_absent.add(str(folder / "events.jsonl"))
            else:
                events = [
                    _json(line) for line in documents["events.jsonl"].splitlines()
                ]
                for node in owner_ids:
                    phases = [
                        event
                        for event in events
                        if event.get("kind") == "phase" and event.get("nodeid") == node
                    ]
                    _require(
                        receipt["phases"].get(node, [])
                        == [
                            {
                                key: value
                                for key, value in phase.items()
                                if key not in ("kind", "nodeid")
                            }
                            for phase in phases
                        ],
                        "original phase receipt/event binding differs",
                    )
                    expected_class, expected_name = node.split("::", 1)
                    matching = [
                        case
                        for case in cases
                        if case.attrib.get("classname")
                        == expected_class[:-3].replace("/", ".")
                        and case.attrib.get("name") == expected_name
                    ]
                    _require(
                        len(matching) == 1
                        and not any(
                            matching[0].find(tag) is not None
                            for tag in ("failure", "error", "skipped")
                        ),
                        "original model/control JUnit outcome differs",
                    )
        except Exception as error:
            if _resource_error(error):
                raise
            worker["errors"].append(
                {"type": type(error).__name__, "message": str(error)[:256]}
            )
            self._error(folder.name, error)
        if not self.external:
            for node in owner_ids:
                if node in duplicates:
                    continue
                try:
                    if node in _PAYLOADS:
                        self._case(
                            node, receipt.get("phases", {}).get(node, []), returncode
                        )
                    elif node in _CONTROLS:
                        phases = receipt.get("phases", {}).get(node, [])
                        _require(
                            returncode == 0
                            and [(p["when"], p["outcome"]) for p in phases]
                            == [
                                (when, "passed")
                                for when in ("setup", "call", "teardown")
                            ],
                            "original guard control did not pass",
                        )
                        self.record["cases"][node] = {
                            "status": "COMPLETE_NO_PAYLOAD_CONTROL",
                            "original_phases": phases,
                        }
                    else:
                        raise ValueError("unknown original owner ID")
                except Exception as error:
                    if _resource_error(error):
                        raise
                    worker["errors"].append(
                        {"type": type(error).__name__, "message": str(error)[:256]}
                    )
                    self._error(node, error)
        self.record["resource_observations"].append(
            {"folder": folder.name, "after_worker_free_bytes": _free(self.root)}
        )
        try:
            self._checkpoint(folder, worker, owner_ids)
        except Exception as error:
            if _resource_error(error):
                raise
            self._error(folder.name + ":checkpoint", error)
        self._save()

    def finish(self, original_status: str) -> None:
        try:
            self.record["original_run_status"] = original_status
            _require(
                type(self.record["checkout"]) is str
                and re.fullmatch("[0-9a-f]{40}", self.record["checkout"]) is not None,
                "exact checkout identity unavailable",
            )
            end = {}
            self.record["source_files_end"] = end
            for path, before in self.sources.items():
                proof, _ = _file(Path(path))
                end[path] = proof
                _require(
                    not proof["errors"]
                    and proof["sha256"] == before["sha256"]
                    and proof["before"] == before["before"],
                    "original owner/transport source changed",
                )
            if not self.external:
                self._owned()
                actual: set[str] = set()
                expected_directories = {
                    _PAYLOADS[node]
                    for node in self.record["cases"]
                    if node in _PAYLOADS
                }
                folders = _children(self.evidence)
                _require(
                    {path.name for path in folders} == expected_directories,
                    "final owned case directory membership differs",
                )
                for folder in folders:
                    folder_stat = _path_stat(folder)
                    _require(
                        stat.S_ISDIR(folder_stat.st_mode)
                        and folder_stat.st_uid == os.getuid(),
                        "owned case directory changed",
                    )
                    for path in _children(folder):
                        value = _path_stat(path)
                        _require(
                            stat.S_ISREG(value.st_mode) and value.st_uid == os.getuid(),
                            "owned artifact tree contains foreign path",
                        )
                        actual.add(str(path))
                        _require(
                            len(actual) <= _ENTRY_LIMIT, "owned artifact entry bound"
                        )
                _require(
                    actual == set(self.held), "final whole-tree file membership differs"
                )
                for path, expected in self.held.items():
                    _require(
                        _identity(_path_stat(Path(path))) == expected,
                        "closed original payload changed",
                    )
                for path in self.worker_absent:
                    try:
                        _path_stat(Path(path))
                    except FileNotFoundError as error:
                        if error.errno != errno.ENOENT:
                            raise
                    else:
                        raise ValueError(
                            "clean collect-only events appeared after closure"
                        )
                for path, expected in self.worker_held.items():
                    _require(
                        _identity(_path_stat(Path(path))) == expected,
                        "closed original worker receipt changed",
                    )
                expected = self.record["expected_owner_ids"]
                full = (
                    set(expected) == set(_PAYLOADS) | set(_CONTROLS)
                    and len(expected) == 12
                )
                self.record["whole_original_owner_collected"] = full
                _require(
                    set(self.record["cases"]) == set(expected),
                    "original owner execution coverage incomplete",
                )
                _require(
                    all(
                        value["status"]
                        in ("COMPLETE_TRANSPORT_ONLY", "COMPLETE_NO_PAYLOAD_CONTROL")
                        for value in self.record["cases"].values()
                    ),
                    "original owner transport incomplete",
                )
                _require(
                    not self.record["errors"] and original_status == "PASS",
                    "original run or transport has failures",
                )
                self.record["status"] = (
                    "COMPLETE_ORIGINAL_OWNER_TRANSPORT_ONLY"
                    if full
                    else "SUBSET_TRANSPORT_ONLY"
                )
            self.record["source_owner_held"] = True
        except Exception as error:
            if _resource_error(error):
                raise
            self._error("finish", error)
        self._save()
