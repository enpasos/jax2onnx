#!/usr/bin/env python3
# scripts/run_pytest_batches.py

"""Run the entire selected suite in bounded, fresh, serial pytest processes.

Pytest options follow ``--``; optional collection paths use ``--test-path``.
Collection, raw output, phase reports, JUnit and passive resource observations
are retained in RUNNER_TEMP/jax2onnx-pytest-batches (a fresh temp dir locally).
No retry or resource-limit change can turn an incomplete run into success.
"""

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import shlex
import signal
import stat
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
from typing import Any
import xml.etree.ElementTree as ET

_MAX_BATCH_SIZE = 500
_ISOLATED_OWNER = "test_source_model_native_bindings_exports.py"
_CAST_OBSERVER_NODE = (
    "tests/extra_tests/converter/test_source_conversion_semantics.py::"
    "test_floating_conversion_rounding_and_strict_special_value_gates"
    "[ties-float16-float64]"
)
_CAST_DIAGNOSTIC_NAME = "source-f64-f16-cast-observation.json"
_BF16_CAPTURE_NODE = (
    "tests/extra_tests/converter/test_source_native_top_k.py::"
    "test_source_bfloat16_top_k_portable_words_and_native_equivalence[finite-inline]"
)
_CONTROL_SCHEMA = "source-cast-observer-off-v1"
_REQUEST_FIELDS = (
    "kind",
    "assigned",
    "full_preselection_ids",
    "collection_skips",
    "pytest_args",
)
_WORKER_ENV_NAMES = (
    "PYTHONPATH",
    "PYTHONHASHSEED",
    "PYTEST_ADDOPTS",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
    "PYTHONHOME",
    "PYTHONNOUSERSITE",
    "PYTHONSAFEPATH",
    "PYTHONPYCACHEPREFIX",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OMP_DYNAMIC",
    "MKL_NUM_THREADS",
    "MKL_DYNAMIC",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TF_NUM_INTEROP_THREADS",
    "TF_NUM_INTRAOP_THREADS",
    "JAX_PLATFORMS",
    "JAX_ENABLE_X64",
    "XLA_FLAGS",
    "NPY_DISABLE_CPU_FEATURES",
    "JAX2ONNX_RUN_MAXTEXT",
    "JAX2ONNX_MAXTEXT_SRC",
    "JAX2ONNX_MAXTEXT_MODELS",
    "JAX2ONNX_MAXTEXT_REF",
    "JAX2ONNX_RUN_MAXDIFFUSION",
    "JAX2ONNX_MAXDIFFUSION_SRC",
    "JAX2ONNX_MAXDIFFUSION_MODELS",
    "JAX2ONNX_MAXDIFFUSION_REF",
)
_WORKER_ENV_PREFIXES = (
    "JAX_",
    "JAX2ONNX_",
    "XLA_",
    "PYTEST_",
    "OMP_",
    "MKL_",
    "OPENBLAS_",
    "TF_",
    "NPY_",
    "NUMEXPR_",
)
_CI_VOLATILE_ENV_NAMES = frozenset(
    (
        "PYTEST_CURRENT_TEST",
        "PYTEST_XDIST_WORKER",
        "PYTEST_XDIST_WORKER_COUNT",
        "PYTEST_XDIST_TESTRUNUID",
    )
)
_WORKER_SOURCE_FILES = (
    "scripts/run_pytest_batches.py",
    "scripts/_source_cast_observer.py",
    "scripts/_source_cast_postmortem.py",
    "tests/extra_tests/converter/test_source_conversion_semantics.py",
    "tests/extra_tests/converter/test_source_pointwise_mul_exp.py",
    "tests/extra_tests/converter/test_source_predicate_families.py",
    "jax2onnx/precision.py",
    "jax2onnx/_compat/jax.py",
    "jax2onnx/plugins/jax/lax/convert_element_type.py",
)


def _worker_environment_names(environment: Any = None) -> list[str]:
    source = os.environ if environment is None else environment
    names = set(_WORKER_ENV_NAMES)
    for name in source:
        if type(name) is not str:
            raise ValueError("worker environment names must be exact strings")
        if name.startswith(_WORKER_ENV_PREFIXES):
            names.add(name)
    if len(names) > 256 or any(len(name) > 256 for name in names):
        raise ValueError("worker environment names exceed bounds")
    return sorted(names)


def _context_read(path: Path, limit: int = 2 * 1024 * 1024) -> bytes:
    with path.open("rb") as stream:
        payload = stream.read(limit + 1)
    if len(payload) > limit:
        raise ValueError("worker context file exceeds bound")
    return payload


def _context_json(path: Path, limit: int, *, label: str) -> tuple[Any, bytes]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate {label} key")
            result[key] = value
        return result

    raw = _context_read(path, limit)
    return json.loads(raw, object_pairs_hook=object_pairs), raw


def _worker_environment() -> dict[str, str | None]:
    environment = {name: os.environ.get(name) for name in _worker_environment_names()}
    if (
        any(
            value is not None and (type(value) is not str or len(value) > 8192)
            for value in environment.values()
        )
        or len(json.dumps(environment).encode()) > 65536
    ):
        raise ValueError("worker environment values exceed bounds")
    return environment


def _worker_jax_config() -> dict[str, Any]:
    # Metadata only: inspect the already loaded official module, never import it.
    if "jax" not in sys.modules:
        return {"status": "NOT_LOADED", "values": {}}
    import types

    jax, native = sys.modules["jax"], sys.modules.get("jax._src.config")
    if type(jax) is not types.ModuleType or type(native) is not types.ModuleType:
        raise ValueError("loaded JAX module context is unsupported")
    config = vars(jax).get("config")
    if type(config) is not vars(native).get("Config"):
        raise ValueError("loaded JAX config context is unsupported")
    values = {}
    for name in (
        "x64_enabled",
        "jax_disable_jit",
        "jax_default_matmul_precision",
        "jax_platforms",
        "jax_numpy_dtype_promotion",
        "jax_numpy_rank_promotion",
    ):
        value = getattr(config, name)
        if value is None or type(value) is bool or type(value) is str:
            if type(value) is str and len(value) > 256:
                raise ValueError("loaded JAX setting exceeds bound")
            values[name] = value
        elif name == "jax_numpy_dtype_promotion" and type(value) is vars(native).get(
            "NumpyDtypePromotion"
        ):
            member = next(
                (
                    key
                    for key in ("STANDARD", "STRICT")
                    if value is vars(type(value)).get(key)
                ),
                None,
            )
            if member is None:
                raise ValueError("loaded JAX enum member differs")
            storage = object.__getattribute__(value, "__dict__")
            literal = storage.get("_value_") if type(storage) is dict else None
            if (
                type(literal) is not str
                or literal != {"STANDARD": "standard", "STRICT": "strict"}[member]
            ):
                raise ValueError("loaded JAX enum storage differs")
            values[name] = {
                "type": "jax._src.config.NumpyDtypePromotion",
                "member": member,
                "value": literal,
            }
        else:
            raise ValueError("loaded JAX setting is not literal metadata")
    return {"status": "OBSERVED", "values": values}


def _worker_context() -> dict[str, Any]:
    # Metadata only: never import JAX or another framework to complete a snapshot.
    result: dict[str, Any] = {
        "status": "OBSERVED",
        "python": sys.version,
        "executable": sys.executable,
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "soabi": sysconfig.get_config_var("SOABI"),
        "byteorder": sys.byteorder,
        "checkout": {},
        "cpu": {},
        "environment": {},
        "source_files": {},
        "sys_profile_is_none": sys.getprofile() is None,
        "threading_profile_is_none": threading.getprofile() is None,
        "loaded_jax_config": {"status": "NOT_LOADED", "values": {}},
        "errors": [],
    }

    def error(stage: str, exc: BaseException) -> None:
        result["status"] = "UNVERIFIED"
        result["errors"].append({"stage": stage, "error_type": type(exc).__name__})

    root = Path(__file__).resolve().parents[1]
    try:
        if any(
            type(result[name]) is not str or len(result[name]) > 4096
            for name in ("python", "executable", "prefix", "base_prefix", "byteorder")
        ) or (
            result["soabi"] is not None
            and (type(result["soabi"]) is not str or len(result["soabi"]) > 256)
        ):
            raise ValueError("worker interpreter metadata exceeds bounds")
    except BaseException as exc:
        for name in (
            "python",
            "executable",
            "prefix",
            "base_prefix",
            "byteorder",
            "soabi",
        ):
            result[name] = None
        error("interpreter", exc)
    try:
        result["environment"] = _worker_environment()
    except BaseException as exc:
        error("environment", exc)
    try:
        directory = root / ".git"
        if directory.is_file():
            pointer = _context_read(directory, 4096).decode().strip()
            if not pointer.startswith("gitdir: "):
                raise ValueError("invalid Git pointer")
            directory = (root / pointer[8:]).resolve()
        head = _context_read(directory / "HEAD", 4096).decode().strip()
        reference = ""
        if head.startswith("ref: "):
            reference = head[5:]
            if not reference.startswith("refs/") or ".." in reference.split("/"):
                raise ValueError("invalid Git reference")
            head = _context_read(directory / reference, 4096).decode().strip()
        if len(head) != 40 or any(char not in "0123456789abcdef" for char in head):
            raise ValueError("invalid Git head")
        result["checkout"] = {"head": head, "symbolic_ref": reference}
    except BaseException as exc:
        error("checkout", exc)
    try:
        cpu = {}
        for line in _context_read(Path("/proc/cpuinfo")).decode().splitlines():
            key, separator, value = line.partition(":")
            key = key.strip()
            if (
                separator
                and key
                in (
                    "vendor_id",
                    "cpu family",
                    "model",
                    "model name",
                    "stepping",
                    "microcode",
                    "flags",
                )
                and key not in cpu
            ):
                if len(value) > 8192:
                    raise ValueError("CPU context exceeds bound")
                cpu[key] = value.strip()
        result["cpu"] = cpu
    except BaseException as exc:
        error("cpu", exc)
    for relative in _WORKER_SOURCE_FILES:
        try:
            payload = _context_read(root / relative)
            result["source_files"][relative] = {
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        except BaseException as exc:
            error("source:" + relative, exc)
    try:
        result["loaded_jax_config"] = _worker_jax_config()
    except BaseException as exc:
        result["loaded_jax_config"] = {"status": "UNVERIFIED", "values": {}}
        error("loaded_jax_config", exc)
    return result


def _diagnostic_control(request: dict[str, Any]) -> dict[str, Any] | None:
    if type(request) is not dict:
        raise ValueError("worker request must be an exact JSON object")
    if "diagnostic_control" not in request:
        return None
    descriptor = request["diagnostic_control"]
    keys = {
        "schema",
        "nodeid",
        "observer_enabled",
        "original_request_sha256",
        "original_receipt_sha256",
        "original_observer_sha256",
    }
    if (
        type(descriptor) is not dict
        or any(type(key) is not str for key in descriptor)
        or set(descriptor) != keys
        or set(request) != {*_REQUEST_FIELDS, "diagnostic_control"}
    ):
        raise ValueError("invalid observer-off descriptor schema")
    if (
        type(request["kind"]) is not str
        or type(descriptor["schema"]) is not str
        or type(descriptor["nodeid"]) is not str
        or type(request["assigned"]) is not list
        or any(type(node) is not str for node in request["assigned"])
    ):
        raise ValueError("observer-off descriptor requires exact JSON types")
    if (
        request["kind"] != "batch"
        or descriptor["schema"] != _CONTROL_SCHEMA
        or descriptor["nodeid"] != _CAST_OBSERVER_NODE
        or descriptor["observer_enabled"] is not False
        or _CAST_OBSERVER_NODE not in request["assigned"]
    ):
        raise ValueError("invalid observer-off descriptor scope")
    for name in (
        "original_request_sha256",
        "original_receipt_sha256",
        "original_observer_sha256",
    ):
        digest = descriptor[name]
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("invalid observer-off evidence digest")
    original = {
        name: value for name, value in request.items() if name != "diagnostic_control"
    }
    payload = (json.dumps(original, indent=2, ensure_ascii=False) + "\n").encode()
    if hashlib.sha256(payload).hexdigest() != descriptor["original_request_sha256"]:
        raise ValueError("observer-off original request binding differs")
    return descriptor


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, value: Any) -> None:
    # Unique temporary names also make a signal-time checkpoint independent
    # of an ordinary checkpoint it may have interrupted.
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_text(path: Path) -> str:
    try:
        return path.read_text()[:16384]
    except (OSError, UnicodeError) as exc:
        return "unavailable: " + type(exc).__name__


def _resources(pid: int | None) -> dict[str, Any]:
    memory = _read_text(Path("/proc/meminfo"))
    process = (
        _read_text(Path(f"/proc/{pid}/status")) if pid else "unavailable: no child"
    )
    try:
        disk = shutil.disk_usage(Path.cwd())
        disk_state: dict[str, Any] = {
            "total": disk.total,
            "used": disk.used,
            "free": disk.free,
        }
    except OSError as exc:
        disk_state = {"unavailable": type(exc).__name__}
    membership = _read_text(Path("/proc/self/cgroup"))
    cgroup = Path("/sys/fs/cgroup")
    cgroup_selection = "root fallback; process cgroup files unavailable"
    for line in membership.splitlines():
        if line.startswith("0::"):
            candidate = cgroup / line[3:].lstrip("/")
            try:
                if (candidate / "memory.current").is_file():
                    cgroup = candidate
                    cgroup_selection = "process cgroup"
            except OSError:
                pass
            break
    return {
        "time": _now(),
        "host_memory": [
            line
            for line in memory.splitlines()
            if line.startswith(
                ("MemTotal:", "MemAvailable:", "SwapFree:", "unavailable:")
            )
        ],
        "process_memory": [
            line
            for line in process.splitlines()
            if line.startswith(("VmRSS:", "VmHWM:", "VmSize:", "unavailable:"))
        ],
        "cgroup": {
            "membership": membership,
            "path": str(cgroup),
            "selection": cgroup_selection,
            **{
                name: _read_text(cgroup / name)
                for name in (
                    "memory.current",
                    "memory.peak",
                    "memory.max",
                    "memory.events",
                )
            },
        },
        "disk": disk_state,
    }


def _plan(
    items: list[dict[str, Any]],
    limit: int,
    *,
    isolated_owners: frozenset[str] = frozenset(),
) -> list[list[str]]:
    """Pack consecutive classes; keep isolated owners in their own batches."""
    groups: list[tuple[list[str], bool]] = []
    isolated_seen: set[str] = set()
    index = 0
    while index < len(items):
        end = index + 1
        module = items[index]["module"]
        while end < len(items) and items[end]["module"] == module:
            end += 1
        segment = items[index:end]
        if Path(module).name == _ISOLATED_OWNER:
            groups.extend(([item["nodeid"]], True) for item in segment)
        elif module in isolated_owners:
            if module in isolated_seen:
                raise ValueError("isolated whole owner is not contiguous")
            if len(segment) > limit:
                raise ValueError("isolated whole owner exceeds the batch bound")
            isolated_seen.add(module)
            groups.append(([item["nodeid"] for item in segment], True))
        else:
            offset = 0
            while offset < len(segment):
                stop = offset + 1
                scope = segment[offset]["class"]
                while stop < len(segment) and segment[stop]["class"] == scope:
                    stop += 1
                scoped = segment[offset:stop]
                for start in range(0, len(scoped), limit):
                    groups.append(
                        (
                            [item["nodeid"] for item in scoped[start : start + limit]],
                            False,
                        )
                    )
                offset = stop
        index = end
    batches: list[list[str]] = []
    pending: list[str] = []
    for group, isolated in groups:
        if pending and (isolated or len(pending) + len(group) > limit):
            batches.append(pending)
            pending = []
        if isolated:
            batches.append(group)
        else:
            pending.extend(group)
    if pending:
        batches.append(pending)
    return batches


def _completed(phases: dict[str, list[dict[str, Any]]]) -> list[str]:
    completed = []
    for nodeid, reports in phases.items():
        names = [report["when"] for report in reports]
        if names == ["setup", "call", "teardown"] or (
            names == ["setup", "teardown"]
            and reports[0]["outcome"] in {"skipped", "failed"}
        ):
            completed.append(nodeid)
    return completed


def _validate(
    receipt: dict[str, Any], assigned: list[str], full: list[str], skips: list[Any]
) -> None:
    if (
        receipt.get("supplemental", False) is not False
        or receipt.get("native_source_recomputed", False) is not False
        or receipt.get("cast_observer_enabled", True) is not True
        or "diagnostic_control" in receipt
    ):
        raise ValueError(
            "supplemental diagnostic receipt cannot validate a canonical batch"
        )
    actual = [item["nodeid"] for item in receipt["items"]]
    if (
        actual != assigned
        or receipt["selected_ids"] != assigned
        or receipt["assigned_ids"] != assigned
        or receipt["full_preselection_ids"] != full
        or receipt["collection_skips"] != skips
        or not receipt["full_collection_checked"]
        or len(set(actual)) != len(actual)
    ):
        raise ValueError("batch collection differs from its assigned node IDs")
    if receipt["collection_errors"] or receipt.get("exitstatus") != 0:
        raise ValueError("pytest did not finish successfully")
    phases = receipt["phases"]
    completed = _completed(phases)
    if Counter(completed) != Counter(assigned) or set(phases) != set(assigned):
        raise ValueError(
            "batch has missing, duplicate or unexpected completed node IDs"
        )
    if receipt.get("completed") != completed:
        raise ValueError("completed receipt differs from the recorded phase outcomes")
    for reports in phases.values():
        if any(report["outcome"] not in {"passed", "skipped"} for report in reports):
            raise ValueError("batch contains a failed setup, call or teardown")


def _worker(request_path: Path) -> int:
    startup_context = _worker_context()

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate worker request JSON key")
            result[key] = value
        return result

    request_payload = request_path.read_bytes()
    request = json.loads(request_payload, object_pairs_hook=unique)
    control = _diagnostic_control(request)
    expected_keys = (
        {"kind", "pytest_args"}
        if request.get("kind") == "collection"
        else set(_REQUEST_FIELDS)
    )
    if control is not None:
        expected_keys.add("diagnostic_control")
    if (
        request.get("kind") not in {"collection", "batch"}
        or set(request) != expected_keys
        or type(request.get("pytest_args")) is not list
        or any(type(argument) is not str for argument in request["pytest_args"])
    ):
        raise ValueError("invalid canonical worker request schema")
    authority = _ci_lane_worker_authority(request["pytest_args"])
    lane_context = None
    if authority is not None:
        lane_context = _CiLaneContext(
            Path(authority[_CI_PRIVATE_OPTIONS[1]]),
            Path.cwd().resolve(),
            environment=startup_context["environment"],
            expected_sha256=authority[_CI_PRIVATE_OPTIONS[2]],
        )
        lane_context.observe_node(startup_context)
        lane_context.check_snapshot(startup_context, stage="startup")
    # Import pytest only inside disposable collection/execution children.
    import pytest

    folder = request_path.parent
    receipt: dict[str, Any] = {
        "kind": request["kind"],
        "started": _now(),
        "pid": os.getpid(),
        "pytest_args": request["pytest_args"],
        "items": [],
        "full_items": [],
        "full_preselection_ids": [],
        "assigned_ids": request.get("assigned", []),
        "selected_ids": [],
        "full_collection_checked": False,
        "collection_errors": [],
        "collection_skips": [],
        "phases": {},
        "phase_durations": {},
        "exitstatus": None,
        "startup_context": startup_context,
        "supplemental": control is not None,
        "native_source_recomputed": False,
        "cast_observer_enabled": control is None,
    }
    if control is not None:
        receipt["diagnostic_control"] = {
            **control,
            "request_binding_verified_by_worker": True,
            "evidence_sha_scope": "Coordinator-verified references; worker verifies request binding only.",
        }

    def checkpoint() -> None:
        receipt["completed"] = _completed(receipt["phases"])
        _write(folder / "receipt.json", receipt)

    def append_event(event: dict[str, Any], *extra_events: dict[str, Any]) -> None:
        # Append each observation before the next test phase. Avoid repeatedly
        # rewriting all prior phases while retaining partial interrupted runs.
        with (folder / "events.jsonl").open("a") as output:
            for value in (event, *extra_events):
                output.write(json.dumps(value, ensure_ascii=False) + "\n")

    def describe(item: Any) -> dict[str, Any]:
        module = item.getparent(pytest.Module)
        cls = item.getparent(pytest.Class)
        return {
            "nodeid": item.nodeid,
            "module": (
                module.nodeid if module is not None else item.nodeid.split("::", 1)[0]
            ),
            "class": cls.nodeid if cls is not None else None,
        }

    class Report:
        postmortem_pin: Any = None
        postmortem_module: Any = None
        cast_observer: Any = None
        bf16_capture: Any = None
        lane_context: Any = None
        config: Any = None
        session: Any = None
        full_items: Any = ()
        ordered_collection: list[str] | None = None
        selected: list[str] | None = None
        serial_position = 0
        projecting = False

        def lane_error(self, message: str, *, terminal: bool = False) -> None:
            receipt.setdefault("ci_lane", {"errors": []})["errors"].append(message)
            if not terminal:
                checkpoint()
                pytest.exit(message, returncode=4)

        def audit_lane(
            self,
            *,
            terminal: bool = False,
            protocol: bool = False,
            post_pytest: bool = False,
        ) -> None:
            if self.lane_context is None:
                return
            try:
                if self.config.option.collectonly is not (
                    request["kind"] == "collection"
                ):
                    raise ValueError(
                        "canonical lane collection mode changed after qualification"
                    )
                if request["kind"] == "collection" and not _ci_lane_phase_free(receipt):
                    raise ValueError(
                        "canonical lane collection recorded test execution"
                    )
                if {
                    name: self.config.getoption(name) for name in _CI_PRIVATE_OPTIONS
                } != authority:
                    raise ValueError(
                        "canonical CI lane authority changed after qualification"
                    )
                if any(
                    self.config.getoption(name, default=None)
                    for name in ("ci_lane", "ci_main_baseline", "ci_lane_report")
                ):
                    raise ValueError(
                        "independent CI projector conflicts with canonical authority"
                    )
                if protocol:
                    self.lane_context.check_collected()
                    return
                if terminal and self.lane_context.transition is not None:
                    if post_pytest:
                        self.lane_context.check_post_pytest(receipt["final_context"])
                    else:
                        self.lane_context.check_collected()
                bindings = self.lane_context.check(
                    self.config, self.full_items, post_pytest=post_pytest
                )
                receipt["ci_lane"][
                    "source_files_after" if terminal else "source_files"
                ] = bindings
            except (OSError, ValueError, LookupError) as exc:
                self.lane_error(str(exc), terminal=terminal)

        def verify_lane(self, session: Any, *, terminal: bool = False) -> None:
            if authority is not None and (
                self.selected is None
                or [item.nodeid for item in session.items] != self.selected
            ):
                self.lane_error(
                    "canonical lane executable selection changed", terminal=terminal
                )

        def pytest_addoption(self, parser: Any) -> None:
            for option in _CI_PRIVATE_OPTIONS:
                parser.addoption(option, action=_CiAuthority, default=None)

        @pytest.hookimpl(hookwrapper=True, tryfirst=True)
        def pytest_collection_modifyitems(self, items: Any) -> Any:
            before = [item.nodeid for item in items]
            yield
            if authority is not None:
                after = [item.nodeid for item in items]
                if len(set(after)) != len(after) or Counter(after) != Counter(before):
                    self.lane_error(
                        "ordinary collection hooks changed inventory instead of ordering"
                    )
                self.ordered_collection = after

        def pytest_deselected(self, items: Any) -> None:
            if authority is not None and items and not self.projecting:
                self.lane_error(
                    "foreign deselection prevents full canonical lane collection"
                )

        def pytest_fixture_setup(self, fixturedef: Any) -> None:
            if self.lane_context is not None:
                import inspect

                try:
                    self.lane_context.check_source(
                        inspect.getsourcefile(fixturedef.func)
                    )
                except (OSError, ValueError) as exc:
                    self.lane_error(str(exc))

        @pytest.hookimpl(hookwrapper=True, tryfirst=True)
        def pytest_runtestloop(self, session: Any) -> Any:
            if authority is not None:
                self.verify_lane(session)
                if self.lane_context.transition is not None:
                    try:
                        self.lane_context.check_collected()
                    except ValueError as exc:
                        self.lane_error(str(exc))
                self.audit_lane()
            yield
            if authority is not None:
                self.verify_lane(session)

        @pytest.hookimpl(tryfirst=True)
        def pytest_runtest_protocol(self, item: Any) -> None:
            if request["kind"] == "collection":
                self.lane_error("canonical collection cannot execute a test protocol")
            if authority is not None:
                if self.lane_context.transition is not None:
                    self.audit_lane(protocol=True)
                if self.serial_position == 0:
                    self.verify_lane(item.session)
                if (
                    self.selected is None
                    or self.serial_position >= len(self.selected)
                    or item.nodeid != self.selected[self.serial_position]
                ):
                    self.lane_error(
                        "canonical lane execution differs from its assignment"
                    )
                self.serial_position += 1

        def release_bf16_capture(self) -> None:
            capture, self.bf16_capture = self.bf16_capture, None
            if capture is not None:
                try:
                    capture.close()
                except BaseException as error:
                    receipt["bf16_capture_cleanup"] = {
                        "status": "UNVERIFIED",
                        "error_type": type(error).__name__,
                    }

        def release_capture(self) -> None:
            observer, self.cast_observer = self.cast_observer, None
            if observer is not None:
                try:
                    observer.close()
                except BaseException as error:
                    receipt["cast_executable_cleanup"] = {
                        "status": "UNVERIFIED",
                        "error_type": type(error).__name__,
                    }

        def pytest_configure(self, config: Any) -> None:
            self.config = config
            if getattr(config.option, "numprocesses", None) not in (None, 0):
                raise pytest.UsageError("pytest batches require serial child processes")
            effective = {name: config.getoption(name) for name in _CI_PRIVATE_OPTIONS}
            if (
                any(value is not None for value in effective.values())
                and effective != authority
            ):
                raise pytest.UsageError(
                    "unbound or conflicting canonical CI lane authority"
                )
            if authority is not None:
                receipt["ci_lane"] = {
                    "lane": authority[_CI_PRIVATE_OPTIONS[0]],
                    "manifest_path": authority[_CI_PRIVATE_OPTIONS[1]],
                    "manifest_sha256": authority[_CI_PRIVATE_OPTIONS[2]],
                    "errors": [],
                }
                try:
                    if any(
                        config.getoption(name, default=None)
                        for name in ("ci_lane", "ci_main_baseline", "ci_lane_report")
                    ):
                        raise ValueError(
                            "independent CI projector conflicts with canonical authority"
                        )
                    self.lane_context = lane_context
                    receipt["ci_lane"].update(
                        environment=self.lane_context.manifest["environment"],
                        source_files=self.lane_context.sources,
                    )
                    self.audit_lane()
                    if self.lane_context.manifest.get("collection_args", []):
                        receipt["collection_args"] = list(config.option.file_or_dir)
                        receipt["qualified_collection_args"] = list(
                            self.lane_context.manifest["collection_args"]
                        )
                except (OSError, ValueError, LookupError) as exc:
                    self.lane_error(str(exc), terminal=True)
                    raise pytest.UsageError(str(exc)) from exc

        def pytest_collectreport(self, report: Any) -> None:
            if report.failed or report.skipped:
                key = "collection_errors" if report.failed else "collection_skips"
                receipt[key].append(
                    {"nodeid": report.nodeid, "longrepr": str(report.longrepr)}
                )
                append_event({"kind": key, "report": receipt[key][-1]})
                checkpoint()
                if self.lane_context is not None:
                    try:
                        self.lane_context.check_source(
                            self.lane_context.root / report.nodeid.split("::", 1)[0]
                        )
                    except (OSError, ValueError) as exc:
                        self.lane_error(str(exc))

        @pytest.hookimpl(hookwrapper=True, tryfirst=True)
        def pytest_collection_finish(self, session: Any) -> Any:
            # Resume after all ordinary collection/order/finish hooks. Every
            # batch imports the same full suite before execution is selected.
            yield
            full_items = list(session.items)
            self.session = session
            self.full_items = full_items
            receipt["collection_context"] = _worker_context()
            if self.lane_context is not None:
                self.lane_context.observe_node(receipt["collection_context"])
            full_ids = [item.nodeid for item in full_items]
            receipt["full_items"] = [describe(item) for item in full_items]
            receipt["full_preselection_ids"] = full_ids
            checkpoint()
            if authority is not None:
                if full_ids != self.ordered_collection:
                    self.lane_error("canonical full collection changed after ordering")
                if receipt["collection_errors"]:
                    self.lane_error("canonical lane collection contains errors")
                try:
                    self.lane_context.complete_collection(receipt["collection_context"])
                    if self.lane_context.transition is not None:
                        receipt["collection_transition_completed"] = True
                except ValueError as exc:
                    self.lane_error(str(exc))
                self.audit_lane()
                try:
                    standard, extra = _ci_lane_partition(
                        full_ids, self.lane_context.manifest
                    )
                    lane_ids = (
                        standard
                        if authority[_CI_PRIVATE_OPTIONS[0]] == "standard"
                        else extra
                    )
                    assigned_set = set(request.get("assigned", []))
                    if request["kind"] == "batch" and (
                        len(assigned_set) != len(request["assigned"])
                        or [nodeid for nodeid in lane_ids if nodeid in assigned_set]
                        != request["assigned"]
                    ):
                        raise ValueError(
                            "batch assignment is outside its qualified lane/order"
                        )
                except ValueError as exc:
                    self.lane_error(str(exc))
            if len(set(full_ids)) != len(full_ids):
                pytest.exit("duplicate collected node IDs", returncode=4)
            if request["kind"] == "batch":
                if full_ids != request["full_preselection_ids"]:
                    pytest.exit(
                        "full collection differs from baseline node IDs", returncode=4
                    )
                if receipt["collection_skips"] != request["collection_skips"]:
                    pytest.exit("collection skips differ from baseline", returncode=4)
                assigned = set(request["assigned"])
                selected = [item for item in full_items if item.nodeid in assigned]
                deselected = [
                    item for item in full_items if item.nodeid not in assigned
                ]
                session.items[:] = selected
                session.testscollected = len(selected)
                self.projecting = True
                try:
                    session.config.hook.pytest_deselected(items=deselected)
                finally:
                    self.projecting = False
            self.selected = [item.nodeid for item in session.items]
            receipt["full_collection_checked"] = True
            receipt["items"] = [describe(item) for item in session.items]
            receipt["selected_ids"] = [item.nodeid for item in session.items]
            checkpoint()
            if (
                request["kind"] == "batch"
                and receipt["selected_ids"] != request["assigned"]
            ):
                pytest.exit(
                    "batch collection differs from assigned node IDs", returncode=4
                )

        @pytest.hookimpl(hookwrapper=True)
        def pytest_runtest_call(self, item: Any) -> Any:
            if (
                request["kind"] == "batch"
                and control is None
                and item.nodeid == _BF16_CAPTURE_NODE
            ):
                capture: Any = None
                try:
                    from importlib.util import module_from_spec, spec_from_file_location

                    receipt["bf16_context_before"] = _worker_context()
                    path = Path(__file__).with_name("_source_bf16_topk_capture.py")
                    spec = spec_from_file_location("_source_bf16_topk_capture", path)
                    if spec is None or spec.loader is None:
                        raise ValueError("BF16 capture loader unavailable")
                    module = module_from_spec(spec)
                    spec.loader.exec_module(module)
                    capture = module.start_observation(
                        folder, item, receipt["bf16_context_before"]
                    )
                    self.bf16_capture = capture
                except BaseException as error:
                    receipt["bf16_capture"] = {
                        "status": "UNVERIFIED",
                        "stage": "start",
                        "error_type": type(error).__name__,
                    }
                try:
                    yield
                finally:
                    if capture is not None:
                        try:
                            capture.finish()
                            receipt["bf16_context_after"] = _worker_context()
                            capture.finish_context(receipt["bf16_context_after"])
                        except BaseException as error:
                            receipt["bf16_capture"] = {
                                "status": "UNVERIFIED",
                                "stage": "finish",
                                "error_type": type(error).__name__,
                            }
                return
            if request["kind"] != "batch" or item.nodeid != _CAST_OBSERVER_NODE:
                yield
                return
            receipt["target_context_before"] = _worker_context()
            checkpoint()
            if control is not None:
                before = receipt["target_context_before"]
                if (
                    not before["sys_profile_is_none"]
                    or not before["threading_profile_is_none"]
                ):
                    receipt["target_context_after"] = _worker_context()
                    checkpoint()
                    raise pytest.UsageError(
                        "observer-off control requires absent profile owners"
                    )
                receipt["native_source_recomputed"] = True
                try:
                    from importlib.util import module_from_spec, spec_from_file_location

                    path = Path(__file__).with_name("_source_cast_postmortem.py")
                    spec = spec_from_file_location("_cast_off_postmortem", path)
                    if spec is None or spec.loader is None:
                        raise ValueError("postmortem loader unavailable")
                    self.postmortem_module = module_from_spec(spec)
                    spec.loader.exec_module(self.postmortem_module)
                    self.postmortem_pin = self.postmortem_module.pin_postmortem(
                        item, receipt["target_context_before"]
                    )
                except BaseException as error:
                    receipt["off_postmortem"] = {
                        "schema": "source-cast-off-postmortem-v1",
                        "status": "UNVERIFIED",
                        "nodeid": item.nodeid,
                        "errors": [
                            {
                                "stage": "pre-call-pin",
                                "error_type": type(error).__name__,
                            }
                        ],
                    }
                try:
                    yield
                finally:
                    receipt["target_context_after"] = _worker_context()
                    checkpoint()
                return
            observer: Any = None

            def diagnostic_error(
                stage: str, error: BaseException | None = None
            ) -> None:
                record = {
                    "status": "UNVERIFIED",
                    "nodeid": item.nodeid,
                    "reason": stage,
                    "original_test_outcome_authoritative": True,
                }
                if error is not None:
                    record["error_type"] = type(error).__name__
                try:
                    _write(folder / _CAST_DIAGNOSTIC_NAME, record)
                except BaseException:
                    # An unavailable diagnostic must not change pytest outcome.
                    pass

            if sys.getprofile() is not None:
                diagnostic_error("preexisting-profile-preserved")
            else:
                try:
                    # Load only at this original call, after ordinary complete
                    # collection and selection. No framework import or retry.
                    from importlib.util import module_from_spec, spec_from_file_location

                    path = Path(__file__).with_name("_source_cast_observer.py")
                    spec = spec_from_file_location("_jax2onnx_cast_observer", path)
                    if spec is None or spec.loader is None:
                        raise ValueError("cast observer module loader unavailable")
                    module = module_from_spec(spec)
                    spec.loader.exec_module(module)
                    observer = module.start_observation(
                        folder, item.nodeid, item.module
                    )
                    self.cast_observer = observer
                except BaseException as error:
                    diagnostic_error("observer-start-error", error)
            try:
                yield
            finally:
                if observer is not None:
                    try:
                        observer.finish()
                    except BaseException as error:
                        diagnostic_error("observer-finish-error", error)
                receipt["target_context_after"] = _worker_context()
                checkpoint()

        @pytest.hookimpl(hookwrapper=True)
        def pytest_runtest_makereport(self, item: Any, call: Any) -> Any:
            outcome = yield
            if (
                request["kind"] == "batch"
                and control is None
                and item.nodeid == _BF16_CAPTURE_NODE
                and call.when == "call"
            ):
                if self.bf16_capture is not None:
                    try:
                        self.bf16_capture.retain_report(call, outcome.get_result())
                    except BaseException as error:
                        receipt["bf16_capture"] = {
                            "status": "UNVERIFIED",
                            "stage": "original-report",
                            "error_type": type(error).__name__,
                        }
                return
            if (
                control is None
                or item.nodeid != _CAST_OBSERVER_NODE
                or call.when != "call"
            ):
                return
            try:
                original_report = outcome.get_result()
                if (
                    self.postmortem_pin is not None
                    and self.postmortem_module is not None
                ):
                    if original_report.outcome == "failed":
                        receipt["off_postmortem"] = (
                            self.postmortem_module.capture_postmortem(
                                self.postmortem_pin,
                                call,
                                original_report,
                                receipt["target_context_after"],
                            )
                        )
                    else:
                        receipt["off_postmortem"] = {
                            "schema": "source-cast-off-postmortem-v1",
                            "status": "UNVERIFIED",
                            "nodeid": item.nodeid,
                            "errors": [],
                            "reason": "target-call-did-not-fail",
                        }
            except BaseException as error:
                receipt["off_postmortem"] = {
                    "schema": "source-cast-off-postmortem-v1",
                    "status": "UNVERIFIED",
                    "nodeid": item.nodeid,
                    "errors": [
                        {
                            "stage": "post-report-copy",
                            "error_type": type(error).__name__,
                        }
                    ],
                }
            finally:
                self.postmortem_pin = self.postmortem_module = None
                try:
                    checkpoint()
                except BaseException:
                    # A post-report diagnostic write cannot replace the report.
                    pass

        def pytest_runtest_logreport(self, report: Any) -> None:
            entry = {"when": report.when, "outcome": report.outcome}
            if hasattr(report, "wasxfail"):
                entry["wasxfail"] = str(report.wasxfail)
            if report.failed or report.skipped:
                entry["longrepr"] = str(report.longrepr)
            receipt["phases"].setdefault(report.nodeid, []).append(entry)
            timing = {"when": report.when, "seconds": report.duration}
            receipt["phase_durations"].setdefault(report.nodeid, []).append(timing)
            # Timings vary between runs; keep canonical outcome comparisons
            # unchanged for cast observer controls and replay consumers. Leave
            # the semantic phase last for existing post-report checkpoints.
            append_event(
                {"kind": "phase-duration", "nodeid": report.nodeid, **timing},
                {"kind": "phase", "nodeid": report.nodeid, **entry},
            )
            if (
                request["kind"] == "batch"
                and control is None
                and report.nodeid == _BF16_CAPTURE_NODE
                and report.when == "call"
                and self.bf16_capture is not None
            ):
                try:
                    checkpoint()
                    for name in ("events.jsonl", "receipt.json"):
                        with (folder / name).open("rb") as stream:
                            os.fsync(stream.fileno())
                    descriptor = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                    self.bf16_capture.after_report(
                        {
                            "nodeid": report.nodeid,
                            **entry,
                            "durable_phase_recorded": True,
                            "receipt_target_phases": list(
                                receipt["phases"][report.nodeid]
                            ),
                        }
                    )
                    receipt["bf16_capture"] = {
                        "status": self.bf16_capture.report["status"],
                        "manifest": "source-bf16-topk-native/manifest.json",
                        "errors": self.bf16_capture.report["errors"],
                        "original_test_outcome_authoritative": True,
                    }
                except BaseException as error:
                    receipt["bf16_capture"] = {
                        "status": "UNVERIFIED",
                        "stage": "after-report",
                        "error_type": type(error).__name__,
                        "original_test_outcome_authoritative": True,
                    }
                finally:
                    self.release_bf16_capture()
            if (
                report.nodeid == _CAST_OBSERVER_NODE
                and report.when == "call"
                and control is None
                and self.cast_observer is not None
            ):
                try:
                    # Exporting an existing CPU executable performs additional
                    # native linking. Record the authoritative original outcome
                    # first, on disk, after the call hook restored its profile.
                    checkpoint()
                    for name in ("events.jsonl", "receipt.json"):
                        with (folder / name).open("rb") as stream:
                            os.fsync(stream.fileno())
                    descriptor = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                    self.cast_observer.after_report(
                        {
                            "nodeid": report.nodeid,
                            **entry,
                            "durable_phase_recorded": True,
                            "phase_receipt": "receipt.json",
                            "phase_events": "events.jsonl",
                        }
                    )
                except BaseException as error:
                    receipt["cast_executable_capture"] = {
                        "status": "UNVERIFIED",
                        "original_test_outcome_authoritative": True,
                        "error_type": type(error).__name__,
                    }
                finally:
                    self.release_capture()

        def pytest_sessionfinish(self, session: Any, exitstatus: int) -> None:
            if authority is not None:
                self.session = session
                receipt.setdefault("ci_lane", {"errors": []})["native_exitstatus"] = (
                    int(exitstatus)
                )
                self.audit_lane(terminal=True)
                self.verify_lane(session, terminal=True)
                if receipt.get("ci_lane", {}).get("errors") and session.exitstatus == 0:
                    session.exitstatus = 1
                exitstatus = session.exitstatus
            receipt["exitstatus"] = int(exitstatus)
            receipt["finished"] = _now()
            checkpoint()

        @pytest.hookimpl(hookwrapper=True, tryfirst=True)
        def pytest_unconfigure(self, config: Any) -> Any:
            yield
            if (
                authority is not None
                and self.lane_context is not None
                and self.lane_context.transition is not None
            ):
                # Pytest still owns E1 here. Its outer finally may restore E2
                # after this hook, so retain any E1 failure for the final result.
                self.audit_lane(terminal=True)

    def interrupted(number: int, frame: Any) -> None:
        receipt["interrupted_signal"] = number
        checkpoint()
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    checkpoint()
    plugin = Report()
    try:
        result = int(
            pytest.main(
                [*request["pytest_args"], "--junitxml", str(folder / "junit.xml")],
                plugins=[plugin],
            )
        )
    except BaseException as exc:
        receipt["worker_exception"] = type(exc).__name__
        checkpoint()
        raise
    finally:
        plugin.release_bf16_capture()
        plugin.release_capture()
        receipt["final_context"] = _worker_context()
        if lane_context is not None:
            lane_context.observe_node(receipt["final_context"])
        checkpoint()
    if authority is not None:
        # pytest_unconfigure and observer cleanup have now also returned.
        if plugin.lane_context is not None:
            try:
                plugin.lane_context.complete_post_pytest(receipt["final_context"])
                if plugin.lane_context.transition is not None:
                    receipt["post_pytest_completed"] = True
            except (OSError, ValueError, LookupError) as exc:
                plugin.lane_error(str(exc), terminal=True)
        plugin.audit_lane(terminal=True, post_pytest=True)
        if plugin.session is not None:
            plugin.verify_lane(plugin.session, terminal=True)
        if receipt.get("ci_lane", {}).get("errors") and result == 0:
            result = 1
    receipt["exitstatus"] = result
    receipt["finished"] = _now()
    checkpoint()
    if authority is not None:
        _write(
            folder / "lane-terminal.json",
            {
                "receipt_sha256": hashlib.sha256(
                    (folder / "receipt.json").read_bytes()
                ).hexdigest(),
                "request_sha256": hashlib.sha256(request_payload).hexdigest(),
                "manifest_sha256": authority[_CI_PRIVATE_OPTIONS[2]],
                "exitstatus": result,
            },
        )
    return result


def _child(
    folder: Path,
    request: dict[str, Any],
    run: dict[str, Any],
    root: Path,
    *,
    environment: dict[str, str] | None = None,
    timeout_seconds: float | None = None,
) -> int:
    if environment is not None and (
        type(environment) is not dict
        or len(environment) > 1024
        or any(
            type(key) is not str or type(value) is not str
            for key, value in environment.items()
        )
        or sum(len(key) + len(value) for key, value in environment.items())
        > 1024 * 1024
    ):
        raise ValueError("explicit child environment exceeds schema bounds")
    if timeout_seconds is not None and (
        type(timeout_seconds) not in (int, float) or not 0 < timeout_seconds <= 240
    ):
        raise ValueError("child diagnostic timeout must be within 240 seconds")
    folder.mkdir()
    _write(folder / "request.json", request)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        str(folder / "request.json"),
    ]
    event: dict[str, Any] = {
        "kind": request["kind"],
        "folder": folder.name,
        "command": command,
        "started": _now(),
        "before": _resources(None),
        "samples": [],
        "returncode": None,
    }
    run["processes"].append(event)
    _write(root / "run.json", run)
    start = time.monotonic()
    child = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=environment,
    )
    event["pid"] = child.pid
    event["child_started"] = _resources(child.pid)
    _write(root / "run.json", run)

    def copy_output() -> None:
        assert child.stdout is not None
        with (folder / "output.log").open("wb") as output:
            for line in iter(child.stdout.readline, b""):
                output.write(line)
                output.flush()
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()

    reader = threading.Thread(target=copy_output, daemon=True)
    reader.start()

    def interrupted(number: int, frame: Any) -> None:
        event["interrupted_signal"] = number
        run["status"] = "FAIL"
        run["failure"] = {"type": "signal", "number": number}
        _write(root / "run.json", run)
        if child.poll() is None:
            try:
                os.killpg(child.pid, number)
            except ProcessLookupError:
                pass
        # Enter the bounded cleanup below even if native child code defers its
        # Python signal handler. An interrupted parent never continues batches.
        raise KeyboardInterrupt(f"parent received signal {number}")

    prior = {
        number: signal.signal(number, interrupted)
        for number in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        while True:
            try:
                wait = (
                    60
                    if timeout_seconds is None
                    else min(
                        60, max(0.001, timeout_seconds - (time.monotonic() - start))
                    )
                )
                result = child.wait(timeout=wait)
                if (
                    timeout_seconds is not None
                    and time.monotonic() - start >= timeout_seconds
                ):
                    event["diagnostic_timeout_seconds"] = timeout_seconds
                    run["status"] = "FAIL"
                    run["failure"] = {"type": "diagnostic-timeout"}
                    raise TimeoutError("diagnostic child exceeded its fixed timeout")
                break
            except subprocess.TimeoutExpired:
                if (
                    timeout_seconds is not None
                    and time.monotonic() - start >= timeout_seconds
                ):
                    event["diagnostic_timeout_seconds"] = timeout_seconds
                    run["status"] = "FAIL"
                    run["failure"] = {"type": "diagnostic-timeout"}
                    raise TimeoutError("diagnostic child exceeded its fixed timeout")
                sample = _resources(child.pid)
                event["samples"].append(sample)
                _write(root / "run.json", run)
                print(
                    f"[pytest-batches] {folder.name} pid={child.pid} "
                    f"elapsed={time.monotonic() - start:.0f}s "
                    f"{sample['host_memory']} {sample['process_memory']} "
                    f"cgroup.current={sample['cgroup']['memory.current'].strip()} "
                    f"disk={sample['disk']}",
                    flush=True,
                )
    finally:
        for number, handler in prior.items():
            signal.signal(number, handler)
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
        reader.join(timeout=10)
        event.update(
            returncode=child.returncode,
            finished=_now(),
            elapsed_seconds=time.monotonic() - start,
            after=_resources(child.pid),
            output_complete=not reader.is_alive(),
        )
        _write(root / "run.json", run)
    if event.get("interrupted_signal"):
        raise ValueError(
            f"{folder.name} was interrupted by signal {event['interrupted_signal']}"
        )
    if not event["output_complete"]:
        raise ValueError(f"{folder.name} output stream did not close")
    return result


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=_MAX_BATCH_SIZE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--test-path", action="append", default=[])
    parser.add_argument("--ci-lane", choices=("standard", "extra"), action=_CiAuthority)
    parser.add_argument("--ci-main-baseline", type=Path, action=_CiAuthority)
    campaign_authority = parser.add_mutually_exclusive_group()
    campaign_authority.add_argument(
        "--ci-campaign-contract", type=Path, action=_CiAuthority
    )
    campaign_authority.add_argument(
        "--_ci-provision-campaign",
        type=Path,
        action=_CiAuthority,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= _MAX_BATCH_SIZE:
        parser.error("--batch-size must be between 1 and 500")
    options = args.pytest_args
    if options[:1] == ["--"]:
        options = options[1:]
    if bool(args.ci_lane) != bool(args.ci_main_baseline):
        parser.error(
            "canonical CI selection requires both --ci-lane and --ci-main-baseline"
        )
    if args.ci_campaign_contract or args._ci_provision_campaign:
        if args.ci_lane != "standard":
            parser.error("a campaign contract requires canonical --ci-lane standard")
        try:
            manifest, _ = _context_json(
                args.ci_main_baseline, 16 * 1024 * 1024, label="baseline manifest"
            )
            if type(manifest) is not dict:
                raise ValueError("campaign requires a baseline manifest")
            _ci_campaign_arguments(
                args.test_path, options, manifest.get("collection_args", [])
            )
        except (OSError, ValueError) as error:
            parser.error(str(error))
    if (
        args._ci_provision_campaign
        and "JAX2ONNX_NATIVE_BINDINGS_EVIDENCE_DIR" in os.environ
    ):
        parser.error("provisioning requires the parent native evidence key absent")
    if any(
        argument == name or argument.startswith(name + "=")
        for argument in [*options, *args.test_path]
        for name in _CI_PRIVATE_OPTIONS
    ):
        parser.error("private CI lane authority cannot be supplied by callers")
    if args.ci_lane and any(
        argument == name or argument.startswith(name + "=")
        for argument in options
        for name in ("--ci-lane", "--ci-main-baseline", "--ci-lane-report")
    ):
        parser.error("independent CI projector conflicts with canonical lane selection")
    if args.output_dir is not None:
        root = args.output_dir.resolve()
        root.mkdir(parents=True, exist_ok=False)
    elif os.environ.get("RUNNER_TEMP"):
        root = Path(os.environ["RUNNER_TEMP"]) / "jax2onnx-pytest-batches"
        root.mkdir(parents=True, exist_ok=False)
    else:
        root = Path(tempfile.mkdtemp(prefix="jax2onnx-pytest-batches-"))
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
        checked_out_commit = (
            commit.stdout.strip() if commit.returncode == 0 else "unavailable"
        )
    except OSError:
        checked_out_commit = "unavailable"
    run: dict[str, Any] = {
        "status": "RUNNING",
        "started": _now(),
        "checked_out_commit": checked_out_commit,
        "event_commit": os.environ.get("GITHUB_SHA", "unavailable"),
        "interpreter": sys.executable,
        "python": sys.version,
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "pytest_addopts": os.environ.get("PYTEST_ADDOPTS", ""),
        "batch_size": args.batch_size,
        "processes": [],
        "completed": [],
    }
    _write(root / "run.json", run)
    print(f"[pytest-batches] receipts: {root}", flush=True)
    transport = None
    helper = None
    setup_failure = None
    lane_context = None
    campaign = None
    provision = None
    lane_environment = None
    lane_closed: list[tuple[Path, dict[str, Any], dict[str, str]]] = []
    run["native_model_artifact_transport"] = "native-model-artifact-transport.json"

    def write_setup_failure() -> None:
        path = root / run["native_model_artifact_transport"]
        _write(path, setup_failure)
        # A failed optional setup still has a durable, explicit receipt. Any
        # inability to retain that receipt is terminal, never silent fallback.
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def resource_error(error: Exception) -> bool:
        import errno

        censor = None if helper is None else vars(helper).get("ArtifactResourceCensor")
        return (
            isinstance(error, MemoryError)
            or (
                isinstance(error, OSError)
                and error.errno in (errno.ENOSPC, errno.EDQUOT, errno.ENOMEM)
            )
            or (
                isinstance(censor, type)
                and issubclass(censor, Exception)
                and isinstance(error, censor)
            )
        )

    def record_resource_error(stage: str, error: Exception, **original: Any) -> None:
        run.setdefault("native_model_transport_resource_errors", []).append(
            {
                "stage": stage,
                "type": type(error).__name__,
                "errno": error.errno if isinstance(error, OSError) else None,
                **original,
            }
        )

    try:
        # Load only the adjacent stdlib helper. The script is also imported by
        # file location in tests, so this must not depend on a scripts namespace
        # being installed or on changing the parent's import path/environment.
        from importlib.util import module_from_spec, spec_from_file_location

        try:
            spec = spec_from_file_location(
                "_native_model_transport",
                Path(__file__).with_name("_native_model_artifact_transport.py"),
            )
            if spec is None or spec.loader is None:
                raise ValueError("native model transport helper is unavailable")
            helper = module_from_spec(spec)
            spec.loader.exec_module(helper)
            transport = helper.NativeModelTransport(
                root, checked_out_commit, dict(os.environ)
            )
        except Exception as error:
            terminal_resource_error = resource_error(error)
            run["native_model_artifact_transport"] = (
                "native-model-artifact-transport-setup-error.json"
            )
            setup_failure = {
                "schema": "native-model-artifact-transport-v1",
                "status": "UNVERIFIED",
                "setup_status": (
                    "RESOURCE_CENSORED" if terminal_resource_error else "UNAVAILABLE"
                ),
                "environment_mode": "ORIGINAL_INHERITED",
                "environment_key": "JAX2ONNX_NATIVE_BINDINGS_EVIDENCE_DIR",
                "environment_value": os.environ.get(
                    "JAX2ONNX_NATIVE_BINDINGS_EVIDENCE_DIR"
                ),
                "parent_environment_mutated": False,
                "checkout": checked_out_commit,
                "errors": [
                    {
                        "stage": "setup",
                        "type": type(error).__name__,
                        "message": str(error)[:256],
                    }
                ],
                "workers": [],
                "full_collection_ids": None,
                "original_run_status": "RUNNING",
                "numerical_acceptance": "NOT_ASSERTED",
                "full_B_to_F_acceptance": "UNVERIFIED",
                "partial_setup_artifacts_preserved": True,
                "partial_transport_manifest": "native-model-artifact-transport.json",
            }
            run.setdefault("native_model_transport_errors", []).append(
                {"stage": "setup", "type": type(error).__name__}
            )
            write_setup_failure()
            if terminal_resource_error:
                record_resource_error("setup", error)
                raise

        def canonical_child(folder: Path, request: dict[str, Any]) -> int:
            if transport is None:
                # Preserve the original inheritance contract exactly; an
                # unavailable observer must not install even a copied env map.
                return _child(folder, request, run, root)
            try:
                transport.before_worker(folder, request)
            except Exception as error:
                if resource_error(error):
                    record_resource_error(folder.name + ":before", error)
                    raise
                run.setdefault("native_model_transport_errors", []).append(
                    {"stage": folder.name + ":before", "type": type(error).__name__}
                )
            result = None
            try:
                result = _child(
                    folder, request, run, root, environment=transport.environment
                )
                return result
            finally:
                original_exception = sys.exc_info()[1]
                try:
                    transport.closed_worker(folder, request, result)
                except Exception as error:
                    run.setdefault("native_model_transport_errors", []).append(
                        {"stage": folder.name + ":closed", "type": type(error).__name__}
                    )
                    if resource_error(error):
                        record_resource_error(
                            folder.name + ":closed",
                            error,
                            original_worker_returncode=result,
                            original_worker_exception_type=(
                                type(original_exception).__name__
                                if original_exception is not None
                                else None
                            ),
                        )
                        # Retain an active original exception/nonzero result.
                        # Otherwise stop orchestration before another worker.
                        if original_exception is None and result == 0:
                            raise

        worker_options = list(options)
        if args.ci_lane:
            lane_environment = (
                transport.environment if transport is not None else dict(os.environ)
            )
            lane_context = _CiLaneContext(
                args.ci_main_baseline,
                Path.cwd().resolve(),
                environment=lane_environment,
            )
            worker_options.extend(
                f"{name}={value}"
                for name, value in zip(
                    _CI_PRIVATE_OPTIONS,
                    (args.ci_lane, str(lane_context.path), lane_context.sha256),
                    strict=True,
                )
            )
            (root / "ci-main-baseline.json").write_bytes(lane_context.path.read_bytes())
            run["ci_lane"] = {
                "lane": args.ci_lane,
                "manifest_path": str(lane_context.path),
                "manifest_sha256": lane_context.sha256,
                "environment": lane_context.manifest["environment"],
                "source_files": lane_context.sources,
                "acceptance_scope": "chosen-lane-native-pytest-outcomes-only",
                "outside_lane_source_native_acceptance": "UNVERIFIED",
                "closed_workers": {},
            }
            _write(root / "run.json", run)

        if args.ci_campaign_contract:
            campaign = _CiCampaignContext(args.ci_campaign_contract, lane_context)
            run["ci_campaign"] = {
                "contract_path": str(campaign.path),
                "contract_sha256": campaign.sha256,
                campaign.inventory_name
                + "_path": str(campaign.references[campaign.inventory_name][0]),
                campaign.inventory_name + "_sha256": campaign.inventory_sha256,
            }
            _write(root / "run.json", run)

        if args._ci_provision_campaign:
            provision = _CiCampaignProvision(
                args._ci_provision_campaign, lane_context, root
            )

        def closed_lane_worker(folder: Path, request: dict[str, Any]) -> dict[str, Any]:
            if lane_context is None:
                return json.loads((folder / "receipt.json").read_text())
            lane_context.recheck(environment=lane_environment)
            receipt = _ci_lane_closed(folder, request, lane_context)
            names = ("request.json", "receipt.json", "lane-terminal.json")
            if campaign is not None and request["kind"] == "batch":
                names += ("junit.xml",)
            hashes = {
                name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                for name in names
            }
            lane_closed.append((folder, request, hashes))
            run["ci_lane"]["closed_workers"][folder.name] = hashes
            return receipt

        collection = root / "collection"
        request = {
            "kind": "collection",
            "pytest_args": [*worker_options, *args.test_path, "--collect-only"],
        }
        if canonical_child(collection, request) != 0:
            raise ValueError(
                "full-suite collection process did not finish successfully"
            )
        baseline = closed_lane_worker(collection, request)
        items = baseline["items"]
        nodeids = [item["nodeid"] for item in items]
        if baseline["collection_errors"] or baseline["exitstatus"] != 0:
            raise ValueError("full-suite collection contains errors")
        if lane_context is not None and not _ci_lane_phase_free(baseline):
            raise ValueError("canonical lane collection recorded test execution")
        if (
            not baseline["full_collection_checked"]
            or baseline["full_preselection_ids"] != nodeids
            or baseline["selected_ids"] != nodeids
        ):
            raise ValueError("full-suite collection receipt is incomplete")
        if not nodeids or len(set(nodeids)) != len(nodeids):
            raise ValueError("full-suite collection is empty or has duplicate node IDs")
        if provision is not None:
            # Freeze the independently reviewed allowances before collecting;
            # bind only the untouched, closed first collection afterward.
            provision.recheck()
            campaign = provision.campaign(collection)
            run["ci_campaign"] = {
                "contract_path": str(campaign.path),
                "contract_sha256": campaign.sha256,
                campaign.inventory_name + "_path": str(collection / "receipt.json"),
                campaign.inventory_name + "_sha256": campaign.inventory_sha256,
                "provision_source": str(provision.path),
                "provision_sha256": provision.sha256,
            }
        if campaign is not None:
            campaign.collection(baseline)
        collection_skips = baseline["collection_skips"]
        lane_ids = nodeids
        if lane_context is not None:
            standard, extra = _ci_lane_partition(nodeids, lane_context.manifest)
            lane_ids = standard if args.ci_lane == "standard" else extra
            chosen = set(lane_ids)
            items = [item for item in items if item["nodeid"] in chosen]
            del chosen
            run["lane_nodeids"] = lane_ids
            run["ci_lane"].update(standard_ids=standard, extra_ids=extra)
        batches = _plan(
            items,
            args.batch_size,
            isolated_owners=(
                _CI_STANDARD_OWNERS if args.ci_lane == "standard" else frozenset()
            ),
        )
        if [nodeid for batch in batches for nodeid in batch] != lane_ids:
            raise ValueError("batch plan differs from the ordered chosen lane")
        del baseline, items
        run["nodeids"] = nodeids
        run["batches"] = batches
        _write(root / "run.json", run)
        if transport is not None:
            try:
                transport.collection(nodeids)
            except Exception as error:
                if resource_error(error):
                    record_resource_error("collection", error)
                    raise
                run.setdefault("native_model_transport_errors", []).append(
                    {"stage": "collection", "type": type(error).__name__}
                )
        for index, assigned in enumerate(batches, 1):
            folder = root / f"batch-{index:04d}"
            print(
                f"[pytest-batches] batch {index}/{len(batches)}: {len(assigned)} items",
                flush=True,
            )
            request = {
                "kind": "batch",
                "assigned": assigned,
                "full_preselection_ids": nodeids,
                "collection_skips": collection_skips,
                "pytest_args": [*worker_options, *args.test_path],
            }
            if canonical_child(folder, request) != 0:
                raise ValueError(f"{folder.name} process did not finish successfully")
            receipt = closed_lane_worker(folder, request)
            _validate(receipt, assigned, nodeids, collection_skips)
            if lane_context is not None and receipt["completed"] != assigned:
                raise ValueError("completed batch order differs from its assignment")
            junit = ET.parse(folder / "junit.xml")
            if len(junit.findall(".//testcase")) != len(assigned) + len(
                collection_skips
            ):
                raise ValueError(f"{folder.name} JUnit does not account for every item")
            if campaign is not None:
                campaign.batch(receipt, assigned, folder / "junit.xml")
            run["completed"].extend(receipt["completed"])
            _write(root / "run.json", run)
            del receipt, junit
        if Counter(run["completed"]) != Counter(lane_ids) or (
            lane_context is not None and run["completed"] != lane_ids
        ):
            raise ValueError(
                "completed batch union/order differs from its assigned scope"
            )
        run["status"] = "PASS"
    except BaseException as exc:
        run["status"] = "FAIL"
        run["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        print(f"[pytest-batches] FAIL: {exc}", file=sys.stderr, flush=True)
    if transport is not None:
        original_pytest_status = run["status"]
        try:
            transport.finish(original_pytest_status)
        except Exception as error:
            run.setdefault("native_model_transport_errors", []).append(
                {"stage": "finish", "type": type(error).__name__}
            )
            if resource_error(error):
                record_resource_error(
                    "finish",
                    error,
                    original_pytest_status=original_pytest_status,
                    original_completed_ids=list(run["completed"]),
                )
                run["status"] = "FAIL"
                run.setdefault(
                    "failure", {"type": type(error).__name__, "message": str(error)}
                )
    if lane_context is not None:
        try:
            lane_context.recheck(environment=lane_environment)
            for folder, request, hashes in lane_closed:
                if any(
                    hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected
                    for name, expected in hashes.items()
                ):
                    raise ValueError(
                        "closed canonical lane evidence changed after validation"
                    )
                _ci_lane_closed(folder, request, lane_context)
            if campaign is not None:
                campaign.recheck()
                for folder, request, _ in lane_closed:
                    receipt = _ci_lane_closed(folder, request, lane_context)
                    if request["kind"] == "collection":
                        campaign.collection(receipt)
                    else:
                        campaign.batch(
                            receipt, request["assigned"], folder / "junit.xml"
                        )
            if provision is not None:
                provision.recheck()
        except (OSError, ValueError, LookupError) as error:
            run["status"] = "FAIL"
            run["failure"] = {"type": type(error).__name__, "message": str(error)}
    if setup_failure is not None:
        setup_failure["original_run_status"] = run["status"]
        setup_failure["full_collection_ids"] = run.get("nodeids")
        setup_failure["completed_ids"] = list(run["completed"])
        write_setup_failure()
    run["finished"] = _now()
    _write(root / "run.json", run)
    if run["status"] == "PASS":
        scope = (
            "collected items" if lane_context is None else f"{args.ci_lane} lane items"
        )
        print(
            f"[pytest-batches] PASS: all {len(run['completed'])} {scope} completed once",
            flush=True,
        )
        return 0
    return 1


# Shared lane qualification and the optional direct pytest projector. Canonical
# workers reuse these audits while Report owns their sole assignment projection.
def _ci_lane_environment(environment: Any = None) -> dict[str, str | None]:
    source = os.environ if environment is None else environment
    return {
        name: source.get(name)
        for name in _worker_environment_names(source)
        if name not in _CI_VOLATILE_ENV_NAMES
    }


def _ci_lane_jax_config(snapshot: Any) -> None:
    if (
        type(snapshot) is not dict
        or set(snapshot) != {"status", "values"}
        or type(snapshot["status"]) is not str
        or type(snapshot["values"]) is not dict
    ):
        raise ValueError("unsupported collection-transition JAX snapshot")
    values = snapshot["values"]
    if snapshot["status"] == "NOT_LOADED" and not values:
        return
    if snapshot["status"] != "OBSERVED" or set(values) != {
        "x64_enabled",
        "jax_disable_jit",
        "jax_default_matmul_precision",
        "jax_platforms",
        "jax_numpy_dtype_promotion",
        "jax_numpy_rank_promotion",
    }:
        raise ValueError("unsupported collection-transition JAX snapshot")
    if any(
        type(values[name]) is not bool for name in ("x64_enabled", "jax_disable_jit")
    ):
        raise ValueError("collection-transition JAX flags need exact booleans")
    for name in ("jax_default_matmul_precision", "jax_platforms"):
        value = values[name]
        if value is not None and (type(value) is not str or len(value) > 256):
            raise ValueError("collection-transition JAX settings need bounded literals")
    if type(values["jax_numpy_rank_promotion"]) is not str or values[
        "jax_numpy_rank_promotion"
    ] not in {"allow", "warn", "raise"}:
        raise ValueError("unsupported collection-transition JAX rank promotion")
    promotion = values["jax_numpy_dtype_promotion"]
    if type(promotion) is str and promotion in {"standard", "strict"}:
        return
    if (
        type(promotion) is not dict
        or set(promotion) != {"type", "member", "value"}
        or any(type(value) is not str for value in promotion.values())
        or promotion["type"] != "jax._src.config.NumpyDtypePromotion"
        or promotion["member"] not in {"STANDARD", "STRICT"}
        or promotion["value"]
        != {"STANDARD": "standard", "STRICT": "strict"}[promotion["member"]]
    ):
        raise ValueError("unsupported collection-transition JAX dtype promotion")


def _ci_lane_transition(transition: Any) -> None:
    if type(transition) is not dict or set(transition) != {
        "startup_jax_config",
        "collected_environment",
        "collected_jax_config",
        "post_pytest_environment",
        "post_pytest_jax_config",
    }:
        raise ValueError("unsupported collection_transition schema")
    for name in (
        "startup_jax_config",
        "collected_jax_config",
        "post_pytest_jax_config",
    ):
        _ci_lane_jax_config(transition[name])
    for name in ("collected_environment", "post_pytest_environment"):
        environment = transition[name]
        if (
            type(environment) is not dict
            or environment != _ci_lane_environment(environment)
            or any(
                value is not None and (type(value) is not str or len(value) > 8192)
                for value in environment.values()
            )
            or len(json.dumps(environment).encode()) > 65536
        ):
            raise ValueError(
                "collection_transition needs a complete bounded environment"
            )


_CI_NODE_PATH_ENV = frozenset(
    {"NPM_CONFIG_CACHE", "NPM_CONFIG_USERCONFIG", "NPM_CONFIG_GLOBALCONFIG"}
)
_CI_NODE_SOURCE_FILES = frozenset(
    {
        "package.json",
        "package-lock.json",
        "scripts/run_onnxruntime_web_smoke.sh",
        "scripts/validate_onnxruntime_web.mjs",
        "scripts/onnxruntime_web_validation_utils.mjs",
    }
)
_CI_NODE_ENV_NAMES = frozenset(
    {
        "NODE_OPTIONS",
        "NODE_PATH",
        "NODE_ENV",
        "NODE_EXTRA_CA_CERTS",
        "NODE_ICU_DATA",
        "NPM_CONFIG_NODE_OPTIONS",
        "NPM_CONFIG_PREFIX",
        "NPM_CONFIG_PRODUCTION",
        "NPM_CONFIG_OMIT",
        "NPM_CONFIG_INCLUDE",
        "NPM_CONFIG_IGNORE_SCRIPTS",
        *_CI_NODE_PATH_ENV,
    }
)
_CI_NODE_ENV_NAMES |= frozenset(name.lower() for name in _CI_NODE_ENV_NAMES)


def _ci_node_relative(value: Any) -> str:
    if (
        type(value) is not str
        or not value
        or value == "."
        or Path(value).is_absolute()
        or ".." in Path(value).parts
        or Path(value).as_posix() != value
        or any(character in value for character in "\\\x00\n\r")
    ):
        raise ValueError("Node bindings need canonical relative paths")
    return value


def _ci_node_digest(value: Any) -> bool:
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


def _ci_node_reference(reference: Any) -> None:
    """Validate reviewed data, never manufacture expectations from this host."""
    if (
        type(reference) is not dict
        or set(reference)
        != {
            "schema",
            "archive_sha256",
            "node_version",
            "npm_version",
            "binary",
            "npm",
            "npm_links",
            "node_modules",
        }
        or reference["schema"] != "jax2onnx-ci-node-runtime-reference-v1"
        or not _ci_node_digest(reference["archive_sha256"])
        or type(reference["node_version"]) is not str
        or re.fullmatch(r"v22\.\d+\.\d+", reference["node_version"]) is None
        or type(reference["npm_version"]) is not str
        or re.fullmatch(r"\d+\.\d+\.\d+", reference["npm_version"]) is None
    ):
        raise ValueError("unsupported reviewed Node runtime reference")

    def regular(value: Any) -> bool:
        return (
            type(value) is dict
            and set(value) == {"bytes", "sha256"}
            and type(value["bytes"]) is int
            and 0 <= value["bytes"] <= 1024**3
            and _ci_node_digest(value["sha256"])
        )

    if not regular(reference["binary"]):
        raise ValueError("Node binary needs an independent size/SHA binding")
    for name in ("npm", "node_modules"):
        tree = reference[name]
        if (
            type(tree) is not dict
            or set(tree) != {"files", "links"}
            or type(tree["files"]) is not dict
            or not tree["files"]
            or type(tree["links"]) is not dict
            or len(tree["files"]) + len(tree["links"]) > 20000
            or set(tree["files"]) & set(tree["links"])
        ):
            raise ValueError("Node tree needs an exact regular-file/link inventory")
        for path, binding in tree["files"].items():
            _ci_node_relative(path)
            if not regular(binding):
                raise ValueError("Node tree regular files need size/SHA bindings")
        for path, target in tree["links"].items():
            _ci_node_relative(path)
            _ci_node_relative(target)
    if (
        reference["npm_links"]
        != {
            "bin/npm": "lib/node_modules/npm/bin/npm-cli.js",
            "bin/npx": "lib/node_modules/npm/bin/npx-cli.js",
        }
        or not {"package.json", "bin/npm-cli.js", "bin/npx-cli.js"}
        <= reference["npm"]["files"].keys()
    ):
        raise ValueError("Node reference needs the official npm/npx CLI links")


def _ci_node_environment(expected: Any, environment: Any = None) -> dict[str, Any]:
    if type(expected) is not dict or set(expected) != _CI_NODE_ENV_NAMES:
        raise ValueError("Node profile needs explicit overlay absence/configuration")
    for name, value in expected.items():
        if value is not None and (
            name.upper() not in _CI_NODE_PATH_ENV
            or type(value) is not str
            or not Path(value).is_absolute()
            or "\x00" in value
        ):
            raise ValueError("Node overlays only allow declared portable config paths")
    env = os.environ if environment is None else environment
    names = _CI_NODE_ENV_NAMES | {
        name for name in env if name.lower().startswith(("node_", "npm_config_"))
    }
    observed = {name: env.get(name) for name in sorted(names)}
    if observed != expected:
        raise ValueError(
            "Node/npm execution overlay differs from reviewed absence/paths"
        )
    return observed


def _ci_node_authority(authority: Any) -> None:
    if (
        type(authority) is not dict
        or set(authority) != {"reference", "prefix", "environment"}
        or type(authority["prefix"]) is not str
        or not Path(authority["prefix"]).is_absolute()
        or type(authority["reference"]) is not dict
        or set(authority["reference"]) != {"path", "sha256"}
    ):
        raise ValueError("unsupported Node runtime authority")
    _ci_node_relative(authority["reference"]["path"])
    if not _ci_node_digest(authority["reference"]["sha256"]):
        raise ValueError("Node reference needs a SHA-256 binding")
    _ci_node_environment(authority["environment"], authority["environment"])


def _ci_node_load(
    root: Path, authority: Any, sources: dict[str, str]
) -> dict[str, Any]:
    _ci_node_authority(authority)
    if not _CI_NODE_SOURCE_FILES <= sources.keys():
        raise ValueError(
            "Node support must bind package/lock/smoke/validator/helper sources"
        )
    binding = authority["reference"]
    relative = _ci_node_relative(binding["path"])
    if (
        not _ci_node_digest(binding["sha256"])
        or sources.get(relative) != binding["sha256"]
    ):
        raise ValueError("Node expectations must be independently source-bound")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("Node reference escapes the checkout")
    reference, raw = _context_json(
        path, 4 * 1024 * 1024, label="Node runtime reference"
    )
    if hashlib.sha256(raw).hexdigest() != binding["sha256"]:
        raise ValueError("reviewed Node reference bytes changed")
    _ci_node_reference(reference)
    return reference


def _ci_node_file(path: Path, expected_bytes: int | None = None) -> dict[str, Any]:
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"Node regular-file binding changed kind: {path}")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise ValueError(f"Node regular-file size changed: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
            if expected_bytes is not None and size > expected_bytes:
                raise ValueError(f"Node regular-file grew during its audit: {path}")
    return {"bytes": size, "sha256": digest.hexdigest()}


def _ci_node_tree(root: Path, expected: dict[str, Any]) -> dict[str, Any]:
    """Stream each regular file once; never follow installed directory links."""
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Node tree root is not a directory")
    files, links = {}, {}
    pending = [root]
    while pending:
        for path in pending.pop().iterdir():
            relative = path.relative_to(root).as_posix()
            _ci_node_relative(relative)
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                if relative not in expected["links"]:
                    raise ValueError(f"unreviewed Node installed link: {relative}")
                target = path.resolve(strict=True)
                if not target.is_relative_to(root):
                    raise ValueError(
                        f"Node installed link escapes its tree: {relative}"
                    )
                links[relative] = target.relative_to(root).as_posix()
            elif stat.S_ISDIR(mode):
                pending.append(path)
            elif stat.S_ISREG(mode):
                if relative not in expected["files"]:
                    raise ValueError(f"unreviewed Node installed file: {relative}")
                files[relative] = _ci_node_file(
                    path, expected["files"][relative]["bytes"]
                )
            else:
                raise ValueError(f"unsupported Node installed file kind: {relative}")
            if len(files) + len(links) > 20000:
                raise ValueError("Node installed inventory exceeds its bound")
    return {"files": dict(sorted(files.items())), "links": dict(sorted(links.items()))}


def _ci_node_check(
    root: Path,
    authority: dict[str, Any],
    reference: dict[str, Any],
    *,
    full: bool,
    environment: Any = None,
) -> dict[str, Any]:
    """Trusted-loader observations, not dynamic-library or TOCTOU attestation."""
    env = os.environ if environment is None else environment
    observed = _ci_node_environment(authority["environment"], env)
    prefix = Path(authority["prefix"]).resolve(strict=True)
    if str(prefix) != authority["prefix"]:
        raise ValueError("Node prefix canonical target changed after binding")
    binary = prefix / "bin/node"
    selected = shutil.which("node", path=env.get("PATH", os.defpath))
    if (
        selected is None
        or not stat.S_ISREG(binary.lstat().st_mode)
        or not os.access(binary, os.X_OK)
        or Path(selected).resolve(strict=True) != binary
    ):
        raise ValueError("bare node does not resolve to the bound executable")
    result = {
        "status": "OBSERVED",
        "reference_sha256": authority["reference"]["sha256"],
        "node_version": reference["node_version"],
        "npm_version": reference["npm_version"],
        "executable": str(binary),
        "environment": observed,
    }
    if full:
        if _ci_node_file(binary, reference["binary"]["bytes"]) != reference["binary"]:
            raise ValueError("independent Node binary size/SHA binding changed")
        for relative, target in reference["npm_links"].items():
            link = prefix / relative
            if not link.is_symlink() or link.resolve(strict=True) != prefix / target:
                raise ValueError("official Node npm/npx CLI link changed")
        for name, path in (
            ("npm", prefix / "lib/node_modules/npm"),
            ("node_modules", root / "node_modules"),
        ):
            if (
                path.is_symlink()
                or (
                    name == "npm"
                    and not path.resolve(strict=True).is_relative_to(prefix)
                )
                or _ci_node_tree(path, reference[name]) != reference[name]
            ):
                raise ValueError(
                    f"exact Node {name} regular-file/link inventory changed"
                )
        package, _ = _context_json(
            prefix / "lib/node_modules/npm/package.json",
            65536,
            label="bound npm package",
        )
        if (
            type(package) is not dict
            or package.get("version") != reference["npm_version"]
        ):
            raise ValueError(
                "bound npm package metadata differs from its archive label"
            )
    return result


def _ci_lane_manifest(
    path: Path, *, environment: Any = None
) -> tuple[dict[str, Any], bytes]:
    manifest, raw = _context_json(path, 16 * 1024 * 1024, label="baseline manifest")
    required = {
        "schema",
        "baseline_commit",
        "python_minor",
        "nodeids",
        "source_files",
        "environment",
    }
    if (
        type(manifest) is not dict
        or not required <= manifest.keys()
        or manifest.keys()
        - required
        - {
            "packages",
            "collection_args",
            "collection_transition",
            "support_role",
            "node_runtime",
        }
        or manifest["schema"] != "jax2onnx-ci-baseline-v1"
    ):
        raise ValueError("unsupported baseline manifest schema")
    if "collection_transition" in manifest:
        _ci_lane_transition(manifest["collection_transition"])
    if "node_runtime" in manifest or "support_role" in manifest:
        if (
            manifest.get("support_role") != "node-smoke"
            or manifest["python_minor"] != "3.12"
            or "node_runtime" not in manifest
            or "collection_transition" not in manifest
            or not manifest.get("collection_args")
        ):
            raise ValueError(
                "Node support baseline needs its complete Python3.12 variant"
            )
        _ci_node_authority(manifest["node_runtime"])

    def digest(value: Any, length: int) -> bool:
        return (
            type(value) is str
            and len(value) == length
            and all(character in "0123456789abcdef" for character in value)
        )

    if not digest(manifest["baseline_commit"], 40):
        raise ValueError("baseline_commit must be a bare forty-character SHA")
    minor = f"{sys.version_info.major}.{sys.version_info.minor}"
    if (
        minor not in {"3.11", "3.12", "3.13", "3.14"}
        or manifest["python_minor"] != minor
    ):
        raise ValueError("unqualified Python cohort")
    ids = manifest["nodeids"]
    if (
        type(ids) is not list
        or not ids
        or any(type(nodeid) is not str or not nodeid for nodeid in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("baseline nodeids must be nonempty, unique ordered strings")
    collection_args = manifest.get("collection_args", [])
    if (
        type(collection_args) is not list
        or any(
            type(argument) is not str or not argument for argument in collection_args
        )
        or len(set(collection_args)) != len(collection_args)
    ):
        raise ValueError(
            "baseline collection_args need unique ordered literal selectors"
        )
    sources = manifest["source_files"]
    if (
        type(sources) is not dict
        or not sources
        or any(
            type(name) is not str
            or not name
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or Path(name).as_posix() != name
            or not digest(value, 64)
            for name, value in sources.items()
        )
    ):
        raise ValueError(
            "baseline source_files need relative paths and SHA-256 bindings"
        )
    if type(manifest["environment"]) is not dict or manifest[
        "environment"
    ] != _ci_lane_environment(environment):
        raise ValueError("baseline environment differs from the qualified context")
    packages = manifest.get("packages", {})
    if type(packages) is not dict or any(
        type(name) is not str or not name or type(version) is not str or not version
        for name, version in packages.items()
    ):
        raise ValueError("baseline packages need exact distribution versions")
    from importlib.metadata import PackageNotFoundError, version

    try:
        actual = {name: version(name) for name in packages}
    except PackageNotFoundError as exc:
        raise ValueError("baseline distribution is not installed") from exc
    if actual != packages:
        raise ValueError("baseline package versions differ from the qualified cohort")
    return manifest, raw


def _ci_lane_sources(root: Path, sources: dict[str, str]) -> dict[str, Any]:
    bindings = {}
    for relative, expected in sources.items():
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"baseline source escapes the checkout: {relative}")
        payload = path.read_bytes()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            raise ValueError(f"baseline source binding changed: {relative}")
        bindings[relative] = {"sha256": actual, "bytes": len(payload)}
    return bindings


def _ci_lane_checkout(root: Path) -> dict[str, Any]:
    # Observations only: a synthetic source archive can lack Git metadata.
    result: dict[str, Any] = {"head": None, "dirty": None, "status": "UNAVAILABLE"}
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        status = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if head.returncode == status.returncode == 0:
            result.update(
                head=head.stdout.strip(),
                dirty=bool(status.stdout),
                source_status=status.stdout.splitlines(),
                status="OBSERVED",
            )
    except (OSError, subprocess.TimeoutExpired):
        pass
    return result


class _CiAuthority(argparse.Action):
    def __call__(
        self, parser: Any, namespace: Any, values: Any, option_string: Any = None
    ) -> None:
        if getattr(namespace, self.dest, None) is not None:
            raise argparse.ArgumentError(self, "duplicate CI lane authority")
        setattr(namespace, self.dest, values)


_CI_PRIVATE_OPTIONS: tuple[str, ...] = (
    "--_j2o-ci-lane",
    "--_j2o-ci-main-baseline",
    "--_j2o-ci-main-sha256",
)
_CI_STANDARD_OWNERS: frozenset[str] = frozenset(
    (
        "tests/examples/test_eqx_dino.py",
        "tests/examples/test_maxdiffusion.py",
        "tests/examples/test_maxtext.py",
        "tests/examples/test_nnx_dino.py",
    )
)


def _ci_lane_qualify(
    config: Any, root: Path, manifest: dict[str, Any], *, recheck: bool = False
) -> None:
    from fnmatch import fnmatchcase

    if Path(config.rootpath).resolve() != root:
        raise ValueError("CI lane checkout root differs from its qualified context")
    for name in (
        "keyword",
        "markexpr",
        "ignore",
        "ignore_glob",
        "deselect",
        "lf",
        "failedfirst",
        "newfirst",
        "noconftest",
        "confcutdir",
        "override_ini",
        "pyargs",
        "inifilename",
        "rootdir",
    ):
        if getattr(config.option, name, None):
            raise ValueError(
                f"CI lane selection rejects effective prefilter/context option: {name}"
            )
    if getattr(config.option, "importmode", "prepend") != "prepend" or config.getini(
        "pythonpath"
    ):
        raise ValueError(
            "CI lane selection requires the qualified default import context"
        )
    collection_args = manifest.get("collection_args", [])
    if list(getattr(config.option, "file_or_dir", [])) != collection_args:
        raise ValueError(
            "effective collection arguments changed after qualification"
            if recheck
            else "effective collection arguments differ from the qualified context"
        )
    if collection_args and Path(config.invocation_params.dir).resolve() != root:
        raise ValueError(
            "qualified collection invocation directory changed"
            if recheck
            else "qualified collection selectors require checkout-root invocation"
        )
    for argument in collection_args:
        relative, *case = argument.split("::")
        owner = Path(relative)
        if (
            not relative
            or owner.is_absolute()
            or owner.as_posix() != relative
            or ".." in owner.parts
            or relative.startswith("-")
            or any(character in relative for character in "*?[]\\")
            or any(character in argument for character in "\x00\n\r")
            or owner.suffix != ".py"
            or not any(
                fnmatchcase(owner.name, pattern)
                for pattern in config.getini("python_files")
            )
            or any(not component for component in case)
        ):
            raise ValueError(f"unsupported literal collection selector: {argument}")
        resolved = (root / owner).resolve(strict=True)
        if (
            not resolved.is_relative_to(root)
            or not resolved.is_file()
            or relative not in manifest["source_files"]
        ):
            raise ValueError(
                "collection selector needs a contained, source-bound test owner: "
                f"{argument}"
            )


def _ci_lane_partition(
    full: list[str], manifest: dict[str, Any]
) -> tuple[list[str], list[str]]:
    if len(set(full)) != len(full):
        raise ValueError("duplicate collected node IDs")
    if any(
        argument not in full
        for argument in manifest.get("collection_args", [])
        if "::" in argument
    ):
        raise ValueError("qualified case selector is not an exact collected node ID")
    baseline = manifest["nodeids"]
    known = set(baseline)
    standard = [nodeid for nodeid in full if nodeid in known]
    if standard != baseline:
        raise ValueError("baseline IDs are missing or their relative order changed")
    return standard, [nodeid for nodeid in full if nodeid not in known]


class _CiLaneContext:
    """The shared manifest/source boundary; pytest projection stays with callers."""

    def __init__(
        self,
        path: Path,
        root: Path,
        *,
        environment: Any = None,
        expected_sha256: str | None = None,
    ) -> None:
        self.path = path.resolve(strict=True)
        self.root = root
        self.manifest, raw = _ci_lane_manifest(self.path, environment=environment)
        self.sha256 = hashlib.sha256(raw).hexdigest()
        if expected_sha256 is not None and self.sha256 != expected_sha256:
            raise ValueError("baseline manifest SHA changed")
        self.sources = _ci_lane_sources(root, self.manifest["source_files"])
        self.node_authority = self.manifest.get("node_runtime")
        self.node_reference = (
            _ci_node_load(root, self.node_authority, self.manifest["source_files"])
            if self.node_authority is not None
            else None
        )
        if self.node_reference is not None:
            _ci_node_check(root, self.node_authority, self.node_reference, full=True)
        self.transition = self.manifest.get("collection_transition")
        self.collection_completed = False
        self.post_pytest_completed = False

    def observe_node(self, snapshot: dict[str, Any]) -> None:
        if self.node_reference is not None:
            try:
                snapshot["node_runtime"] = _ci_node_check(
                    self.root, self.node_authority, self.node_reference, full=False
                )
            except (OSError, ValueError) as exc:
                # Preserve original worker exceptions while final evidence fails
                # closed. This adds nothing to ordinary/global raw contexts.
                snapshot["node_runtime"] = {"status": "UNVERIFIED", "error": str(exc)}

    def check_snapshot(self, snapshot: Any, *, stage: str) -> None:
        if self.node_reference is not None and (
            type(snapshot) is not dict
            or snapshot.get("node_runtime")
            != _ci_node_check(
                self.root, self.node_authority, self.node_reference, full=False
            )
        ):
            raise ValueError("Node runtime context observation is missing or changed")
        if self.transition is None:
            return
        if stage not in {"startup", "collected", "post_pytest"}:
            raise ValueError("unsupported collection-transition stage")
        if (
            type(snapshot) is not dict
            or snapshot.get("status") != "OBSERVED"
            or snapshot.get("errors") != []
            or type(snapshot.get("environment")) is not dict
        ):
            raise ValueError("collection-transition context snapshot is unverified")
        _ci_lane_jax_config(snapshot.get("loaded_jax_config"))
        observed_environment = _ci_lane_environment(snapshot["environment"])
        if {
            name: value
            for name, value in snapshot["environment"].items()
            if name not in _CI_VOLATILE_ENV_NAMES
        } != observed_environment:
            raise ValueError("collection-transition environment snapshot is incomplete")
        environment = (
            self.manifest["environment"]
            if stage == "startup"
            else self.transition[stage + "_environment"]
        )
        jax_config = self.transition[stage + "_jax_config"]
        if observed_environment != environment:
            raise ValueError("collection-transition environment differs")
        if snapshot["loaded_jax_config"] != jax_config:
            raise ValueError("collection-transition loaded JAX configuration differs")

    def complete_collection(self, snapshot: Any) -> None:
        if self.transition is not None:
            if self.collection_completed:
                raise ValueError("collection-transition stage was already completed")
            self.check_snapshot(snapshot, stage="collected")
            self.collection_completed = True

    def complete_post_pytest(self, snapshot: Any) -> None:
        if self.transition is not None:
            if not self.collection_completed or self.post_pytest_completed:
                raise ValueError(
                    "collection-transition post-pytest stage cannot advance"
                )
            self.check_snapshot(snapshot, stage="post_pytest")
            self.post_pytest_completed = True

    def check_collected(self, snapshot: Any = None) -> None:
        if self.node_reference is not None:
            # Protocol checks never rehash the binary/npm/WASM trees.
            _ci_node_check(
                self.root, self.node_authority, self.node_reference, full=False
            )
        if self.transition is not None:
            if not self.collection_completed or self.post_pytest_completed:
                raise ValueError("collection-transition stage is incomplete")
            if snapshot is None:
                snapshot = {
                    "status": "OBSERVED",
                    "errors": [],
                    "environment": _worker_environment(),
                    "loaded_jax_config": _worker_jax_config(),
                }
                self.observe_node(snapshot)
            self.check_snapshot(snapshot, stage="collected")

    def check_post_pytest(self, snapshot: Any) -> None:
        if self.transition is not None:
            if not self.post_pytest_completed:
                raise ValueError(
                    "collection-transition post-pytest stage is incomplete"
                )
            self.check_snapshot(snapshot, stage="post_pytest")

    def check_source(self, path: Any) -> Path | None:
        if path is None:
            return
        resolved = Path(path).resolve()
        if resolved.is_relative_to(self.root) and resolved.suffix == ".py":
            relative = resolved.relative_to(self.root).as_posix()
            if relative not in self.manifest["source_files"]:
                raise ValueError(
                    f"unbound collected owner or dependency source: {relative}"
                )
        return resolved

    def recheck(self, *, environment: Any = None) -> dict[str, Any]:
        _, raw = _ci_lane_manifest(self.path, environment=environment)
        if hashlib.sha256(raw).hexdigest() != self.sha256:
            raise ValueError("baseline manifest SHA changed")
        bindings = _ci_lane_sources(self.root, self.manifest["source_files"])
        if self.node_reference is not None:
            _ci_node_check(
                self.root, self.node_authority, self.node_reference, full=True
            )
        return bindings

    def check(
        self, config: Any, items: Any = (), *, post_pytest: bool = False
    ) -> dict[str, Any]:
        _ci_lane_qualify(config, self.root, self.manifest, recheck=True)
        # Ordinary imports may legitimately be partway through E0 -> E1. The
        # startup snapshot proves E0; complete_collection alone advances stage.
        bindings = self.recheck(
            environment=(
                self.manifest["environment"] if self.transition is not None else None
            )
        )
        if post_pytest and self.transition is not None:
            if not self.post_pytest_completed:
                raise ValueError(
                    "collection-transition post-pytest stage is incomplete"
                )
        elif self.collection_completed:
            self.check_collected()
        if config.inipath is not None:
            relative = Path(config.inipath).resolve().relative_to(self.root).as_posix()
            if relative not in self.manifest["source_files"]:
                raise ValueError(f"unbound pytest configuration: {relative}")
        checked_paths: dict[str | bytes, Path] = {}

        def checked_source(path: Any) -> Path:
            key = os.fspath(path)
            if key not in checked_paths:
                resolved = self.check_source(path)
                assert resolved is not None
                checked_paths[key] = resolved
            return checked_paths[key]

        for item in items:
            if not checked_source(item.path).is_relative_to(self.root):
                raise ValueError("collected test owner resolves outside the checkout")
        for name, module in list(sys.modules.items()):
            namespace = vars(module) if module is not None else {}
            path = namespace.get("__file__")
            if type(path) is not str:
                continue
            resolved = checked_source(path)
            if name == "jax2onnx" or name.startswith(("jax2onnx.", "tests.")):
                if not resolved.is_relative_to(self.root):
                    raise ValueError(
                        f"test/converter module resolves outside the checkout: {name}"
                    )
        return bindings


def _ci_campaign_arguments(
    test_paths: list[str], options: list[str], collection_args: list[str]
) -> None:
    """Collect the complete authority command with reporting options only."""
    if test_paths != collection_args:
        raise ValueError("campaign --test-path selectors differ from authority")
    flags = {
        "-q",
        "--quiet",
        "-v",
        "--verbose",
        "-x",
        "--exitfirst",
        "--disable-warnings",
        "--showlocals",
        "--no-header",
        "--no-summary",
    }
    values = {
        "--durations",
        "--durations-min",
        "--tb",
        "--color",
        "--capture",
        "--maxfail",
    }
    for arguments in (options, shlex.split(os.environ.get("PYTEST_ADDOPTS", ""))):
        pending = False
        for argument in arguments:
            if pending:
                if not argument or argument.startswith("-"):
                    raise ValueError("campaign reporting option needs a value")
                pending = False
            elif argument in flags:
                continue
            elif argument in values:
                pending = True
            elif "=" in argument and argument.split("=", 1)[0] in values:
                if not argument.split("=", 1)[1]:
                    raise ValueError("campaign reporting option needs a value")
            else:
                raise ValueError(
                    f"campaign rejects selection/context override: {argument}"
                )
        if pending:
            raise ValueError("campaign reporting option needs a value")


class _CiCampaignProvision:
    """Source-bound allowances for the existing first canonical collection."""

    def __init__(self, path: Path, context: _CiLaneContext, run: Path) -> None:
        from importlib.util import module_from_spec, spec_from_file_location

        self.path = path.resolve(strict=True)
        self.context = context
        self.run = run
        spec = spec_from_file_location(
            "_ci_standard_preparation",
            Path(__file__).with_name("prepare_ci_standard.py"),
        )
        if spec is None or spec.loader is None:
            raise ValueError("STANDARD authority renderer is unavailable")
        self.helper = module_from_spec(spec)
        spec.loader.exec_module(self.helper)
        self.data, baseline, self.bindings = self.helper._cohort(
            self.path, _context_json
        )
        self.sha256 = self.bindings[self.path]
        node = context.manifest.get("node_runtime")
        expected_node = self.helper._node_binding(
            self.data,
            self.path,
            context.root,
            run,
            Path(node["prefix"]) if node is not None else None,
        )
        if (
            baseline["nodeids"] != context.manifest["nodeids"]
            or self.data["python_minor"] != context.manifest["python_minor"]
            or self.data["baseline_commit"] != context.manifest["baseline_commit"]
            or self.data["packages"] != context.manifest.get("packages")
            or self.data.get("collection_args", [])
            != context.manifest.get("collection_args", [])
            or expected_node != node
            or (
                node is not None
                and context.manifest.get("support_role") != "node-smoke"
            )
            or any(
                context.manifest["source_files"].get(name) != digest
                for name, digest in baseline["generated"].items()
            )
        ):
            raise ValueError("provisioned cohort differs from rendered main authority")
        slots = {
            "${CHECKOUT}": str(context.root),
            "${RUN}/native-model-artifacts": str(run / "native-model-artifacts"),
            **{
                f"${{{name}_SOURCE}}": context.manifest["environment"].get(
                    f"JAX2ONNX_{name}_SRC"
                )
                for name in ("MAXTEXT", "MAXDIFFUSION")
            },
        }
        expected_transition = dict(self.data["collection_transition"])
        for name in ("collected_environment", "post_pytest_environment"):
            expected_transition[name] = self.helper._environment(
                expected_transition[name], slots
            )
        if (
            self.helper._environment(self.data["environment"], slots)
            != context.manifest["environment"]
            or expected_transition != context.transition
        ):
            raise ValueError("provisioned context differs from reviewed template")
        # Validate allowance shape before C; final membership and outcome checks
        # remain the responsibility of the unchanged campaign validator.
        _ci_campaign_skips(self.data["allowed_skips"], context)
        self.recheck()

    def recheck(self) -> None:
        if any(
            self.context.manifest["source_files"].get(name) != digest
            for name, digest in self.data["bindings"].items()
        ) or set(self.data["bindings"]) != self.helper._binding_names(self.data):
            raise ValueError("provisioned lock/project/installer binding differs")
        for path, digest in self.bindings.items():
            relative = path.relative_to(self.context.root).as_posix()
            if (
                self.context.manifest["source_files"].get(relative) != digest
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest
            ):
                raise ValueError("reviewed provision source changed")
        self.helper._installation(self.data, self.context.root)

    def campaign(self, collection: Path) -> "_CiCampaignContext":
        support = self.data["schema"] == "jax2onnx-ci-support-cohort-v1"
        payload = {
            "schema": (
                "jax2onnx-ci-support-campaign-v1"
                if support
                else "jax2onnx-ci-campaign-v1"
            ),
            "python_minor": self.context.manifest["python_minor"],
            "main_manifest": {
                "path": str(self.context.path),
                "sha256": self.context.sha256,
            },
            "command_inventory" if support else "global_inventory": {
                "path": str(collection / "receipt.json"),
                "sha256": hashlib.sha256(
                    (collection / "receipt.json").read_bytes()
                ).hexdigest(),
            },
            "allowed_skips": self.data["allowed_skips"],
        }
        path = self.run / "ci-provisioned-campaign.json"
        with path.open("x") as stream:
            stream.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        return _CiCampaignContext(path, self.context)


def _ci_campaign_skips(skips: Any, context: _CiLaneContext) -> dict[str, Any]:
    allowed = {}
    if type(skips) is not list:
        raise ValueError("campaign allowed_skips must be a list")
    for skip in skips:
        if (
            type(skip) is not dict
            or set(skip) != {"nodeid", "phases", "source", "line", "reason"}
            or type(skip["nodeid"]) is not str
            or skip["nodeid"] not in context.manifest["nodeids"]
            or skip["nodeid"] in allowed
            or skip["phases"]
            not in (
                ["setup:skipped", "teardown:passed"],
                ["setup:passed", "call:skipped", "teardown:passed"],
            )
            or type(skip["source"]) is not str
            or skip["source"] not in context.manifest["source_files"]
            or type(skip["line"]) is not int
            or skip["line"] < 1
            or type(skip["reason"]) is not str
            or not skip["reason"]
        ):
            raise ValueError("campaign skip allowance is not exact/source-bound")
        allowed[skip["nodeid"]] = skip
    return allowed


class _CiCampaignContext:
    """Optional strict outcomes around the existing canonical lane machinery."""

    def __init__(self, path: Path, lane_context: _CiLaneContext) -> None:
        self.path = path.resolve(strict=True)
        self.context = lane_context
        contract, raw = _context_json(self.path, 65536, label="campaign contract")
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.support = (
            type(contract) is dict
            and contract.get("schema") == "jax2onnx-ci-support-campaign-v1"
        )
        self.inventory_name = (
            "command_inventory" if self.support else "global_inventory"
        )
        if (
            type(contract) is not dict
            or set(contract)
            != {
                "schema",
                "python_minor",
                "main_manifest",
                self.inventory_name,
                "allowed_skips",
            }
            or contract["schema"]
            not in {"jax2onnx-ci-campaign-v1", "jax2onnx-ci-support-campaign-v1"}
            or contract["python_minor"] != lane_context.manifest["python_minor"]
            or bool(lane_context.manifest.get("collection_args", [])) != self.support
        ):
            raise ValueError("unsupported campaign contract or full-global cohort")
        if lane_context.transition is None:
            raise ValueError(
                "campaign requires independently observed collection_transition"
            )
        from importlib.metadata import version

        if version("pytest") != "9.1.1":
            raise ValueError(
                "campaign JUnit serialization requires qualified pytest 9.1.1"
            )
        self.references = {}
        for name in ("main_manifest", self.inventory_name):
            reference = contract[name]
            if (
                type(reference) is not dict
                or set(reference) != {"path", "sha256"}
                or type(reference["path"]) is not str
                or not reference["path"]
                or type(reference["sha256"]) is not str
                or len(reference["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in reference["sha256"])
            ):
                raise ValueError("campaign reference needs a path and SHA-256")
            resolved = (self.path.parent / reference["path"]).resolve(strict=True)
            self.references[name] = (resolved, reference["sha256"])
        if self.references["main_manifest"] != (lane_context.path, lane_context.sha256):
            raise ValueError("campaign main manifest binding differs")
        inventory_path, self.inventory_sha256 = self.references[self.inventory_name]
        self.inventory, payload = _context_json(
            inventory_path, 16 * 1024 * 1024, label=self.inventory_name
        )
        if hashlib.sha256(payload).hexdigest() != self.inventory_sha256:
            raise ValueError("campaign global inventory SHA differs")
        self.standard, self.extra = self._inventory(self.inventory)
        self.allowed_skips = _ci_campaign_skips(contract["allowed_skips"], lane_context)
        self.recheck()

    def _inventory(self, receipt: Any) -> tuple[list[str], list[str]]:
        if (
            type(receipt) is not dict
            or receipt.get("kind") != "collection"
            or not _ci_lane_phase_free(receipt)
            or receipt.get("full_collection_checked") is not True
            or receipt.get("collection_errors") != []
            or type(receipt.get("exitstatus")) is not int
            or receipt["exitstatus"] != 0
            or type(receipt.get("finished")) is not str
            or not receipt["finished"]
            or receipt.get("assigned_ids") != []
            or receipt.get("supplemental", False) is not False
            or receipt.get("native_source_recomputed", False) is not False
            or receipt.get("cast_observer_enabled", True) is not True
            or "diagnostic_control" in receipt
        ):
            raise ValueError("campaign inventory needs a closed phase-free collection")
        self._selectors(receipt)
        full = receipt.get("full_preselection_ids")
        items = receipt.get("full_items")
        skips = receipt.get("collection_skips")
        if (
            type(full) is not list
            or not full
            or any(type(node) is not str or not node for node in full)
            or len(set(full)) != len(full)
            or type(items) is not list
            or any(
                type(item) is not dict
                or set(item) != {"nodeid", "module", "class"}
                or type(item["module"]) is not str
                or not item["module"]
                or (item["class"] is not None and type(item["class"]) is not str)
                for item in items
            )
            or [item["nodeid"] for item in items] != full
            or receipt.get("items") != items
            or receipt.get("selected_ids") != full
            or type(skips) is not list
            or any(
                type(skip) is not dict
                or set(skip) != {"nodeid", "longrepr"}
                or any(type(skip[key]) is not str for key in skip)
                for skip in skips
            )
        ):
            raise ValueError(
                "campaign inventory has incomplete C/items/collection skips"
            )
        standard, extra = _ci_lane_partition(full, self.context.manifest)
        for key, expected in (("full", full), ("standard", standard), ("extra", extra)):
            if key in receipt and receipt[key] != expected:
                raise ValueError("campaign stored C/S/A projection differs")
        lane = receipt.get("ci_lane", {})
        if type(lane) is not dict or any(
            key in lane and lane[key] != expected
            for key, expected in (("standard_ids", standard), ("extra_ids", extra))
        ):
            raise ValueError("campaign stored S/A projection differs")
        for key, stage in (
            ("startup_context", "startup"),
            ("collection_context", "collected"),
            ("final_context", "post_pytest"),
        ):
            self.context.check_snapshot(receipt.get(key), stage=stage)
        return standard, extra

    def _selectors(self, receipt: dict[str, Any]) -> None:
        if self.support and any(
            receipt.get(name) != self.context.manifest["collection_args"]
            for name in ("collection_args", "qualified_collection_args")
        ):
            raise ValueError("support campaign receipt selectors differ from authority")

    def collection(self, receipt: dict[str, Any]) -> tuple[list[str], list[str]]:
        partition = self._inventory(receipt)
        if any(
            receipt[key] != self.inventory[key]
            for key in (
                "full_preselection_ids",
                "full_items",
                "items",
                "selected_ids",
                "collection_skips",
            )
        ):
            raise ValueError("campaign fresh global collection differs from inventory")
        return partition

    @staticmethod
    def _junit_key(nodeid: str) -> tuple[str, str]:
        from _pytest.junitxml import bin_xml_escape

        # Match pytest's public JUnit names, including parameters containing ::.
        head, bracket, parameters = nodeid.partition("[")
        names = head.split("::")
        names[0] = names[0].replace("/", ".").removesuffix(".py")
        names[-1] += bracket + parameters
        return ".".join(names[:-1]), bin_xml_escape(names[-1])

    def batch(
        self, receipt: dict[str, Any], assigned: list[str], junit_path: Path
    ) -> None:
        from _pytest.junitxml import bin_xml_escape

        full = self.inventory["full_preselection_ids"]
        skips = self.inventory["collection_skips"]
        self._selectors(receipt)
        _validate(receipt, assigned, full, skips)
        assigned_set = set(assigned)
        if (
            not assigned
            or len(assigned_set) != len(assigned)
            or [node for node in self.standard if node in assigned_set] != assigned
            or receipt["completed"] != assigned
        ):
            raise ValueError("campaign batch differs from ordered S assignment")
        expected = {}
        for nodeid in assigned:
            reports = receipt["phases"][nodeid]
            phases = [f"{report['when']}:{report['outcome']}" for report in reports]
            if any("wasxfail" in report for report in reports):
                raise ValueError("campaign rejects any wasxfail presence")
            allowance = None
            if phases != ["setup:passed", "call:passed", "teardown:passed"]:
                allowance = self.allowed_skips.get(nodeid)
                if allowance is None or phases != allowance["phases"]:
                    raise ValueError(
                        "campaign has unexpected or incomplete phases/skip"
                    )
                report = next(
                    report for report in reports if report["outcome"] == "skipped"
                )
                literal = report.get("longrepr")
                if type(literal) is not str or len(literal) > 65536:
                    raise ValueError("campaign skipped report needs an exact location")
                try:
                    location = ast.literal_eval(literal)
                except (ValueError, SyntaxError, RecursionError) as error:
                    raise ValueError(
                        "campaign skipped location is malformed"
                    ) from error
                if type(location) is not tuple or len(location) != 3:
                    raise ValueError("campaign skipped location is malformed")
                source, line, reason = location
                if type(source) is not str:
                    raise ValueError("campaign skipped source is malformed")
                source_path = Path(source)
                if source_path.is_absolute():
                    try:
                        source = source_path.relative_to(self.context.root).as_posix()
                    except ValueError as error:
                        raise ValueError(
                            "campaign skipped source is outside checkout"
                        ) from error
                if (
                    source != allowance["source"]
                    or type(line) is not int
                    or line != allowance["line"]
                    or reason != "Skipped: " + allowance["reason"]
                ):
                    raise ValueError("campaign skipped source/line/reason differs")
            key = self._junit_key(nodeid)
            if key in expected:
                raise ValueError("campaign JUnit identities are ambiguous")
            expected[key] = ("case", allowance)
        for skip in skips:
            key = self._junit_key(skip["nodeid"])
            if key in expected:
                raise ValueError("campaign JUnit collection identities are ambiguous")
            expected[key] = ("collection", skip)
        tree = ET.fromstring(_context_read(junit_path, 16 * 1024 * 1024))
        cases = tree.findall(".//testcase")
        keys = [(case.get("classname", ""), case.get("name", "")) for case in cases]
        if (
            Counter(keys) != Counter(expected.keys())
            or tree.findall(".//failure")
            or tree.findall(".//error")
        ):
            raise ValueError("campaign JUnit identities/outcomes differ")
        for case, key in zip(cases, keys, strict=True):
            kind, allowance = expected[key]
            skipped = case.findall("skipped")
            if allowance is None:
                if skipped:
                    raise ValueError("campaign JUnit has an unexpected skip")
            elif (
                len(skipped) != 1
                or (
                    kind == "case"
                    and (
                        skipped[0].get("type") != "pytest.skip"
                        or skipped[0].get("message")
                        != bin_xml_escape(allowance["reason"])
                    )
                )
                or (
                    kind == "collection"
                    and (
                        "type" in skipped[0].attrib
                        or skipped[0].get("message") != "collection skipped"
                        or skipped[0].text != bin_xml_escape(allowance["longrepr"])
                    )
                )
            ):
                raise ValueError("campaign JUnit skipped outcome differs")

    def recheck(self) -> None:
        if hashlib.sha256(_context_read(self.path, 65536)).hexdigest() != self.sha256:
            raise ValueError("campaign contract SHA changed")
        for path, expected in self.references.values():
            if (
                hashlib.sha256(_context_read(path, 16 * 1024 * 1024)).hexdigest()
                != expected
            ):
                raise ValueError("campaign referenced bytes changed")
        self.context.recheck(environment=self.context.manifest["environment"])


def _ci_lane_worker_authority(options: list[str]) -> dict[str, str] | None:
    authority = {}
    for argument in options:
        for name in _CI_PRIVATE_OPTIONS:
            if argument == name or argument.startswith(name + "="):
                if (
                    name in authority
                    or not argument.startswith(name + "=")
                    or not argument[len(name) + 1 :]
                ):
                    raise ValueError("duplicate or malformed private CI lane authority")
                authority[name] = argument[len(name) + 1 :]
    if not authority:
        return None
    if set(authority) != set(_CI_PRIVATE_OPTIONS) or authority[
        _CI_PRIVATE_OPTIONS[0]
    ] not in {"standard", "extra"}:
        raise ValueError("incomplete private CI lane authority")
    digest = authority[_CI_PRIVATE_OPTIONS[2]]
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("invalid private CI manifest SHA")
    return authority


def _ci_lane_phase_free(receipt: dict[str, Any]) -> bool:
    return (
        receipt.get("phases") == {}
        and receipt.get("completed") == []
        and receipt.get("phase_durations") == {}
    )


def _ci_lane_closed(
    folder: Path, request: dict[str, Any], context: _CiLaneContext
) -> dict[str, Any]:
    payload = (folder / "receipt.json").read_bytes()
    terminal = json.loads((folder / "lane-terminal.json").read_text())
    request_payload = (
        json.dumps(request, indent=2, ensure_ascii=False) + "\n"
    ).encode()
    if (
        terminal
        != {
            "receipt_sha256": hashlib.sha256(payload).hexdigest(),
            "request_sha256": hashlib.sha256(request_payload).hexdigest(),
            "manifest_sha256": context.sha256,
            "exitstatus": 0,
        }
        or (folder / "request.json").read_bytes() != request_payload
    ):
        raise ValueError("closed canonical lane receipt/request binding differs")
    receipt = json.loads(payload)
    selectors = context.manifest.get("collection_args", [])
    if selectors and any(
        receipt.get(name) != selectors
        for name in ("collection_args", "qualified_collection_args")
    ):
        raise ValueError("closed canonical lane support receipt selectors differ")
    if context.transition is not None:
        if (
            receipt.get("collection_transition_completed") is not True
            or receipt.get("post_pytest_completed") is not True
        ):
            raise ValueError("closed collection-transition stage is incomplete")
        context.check_snapshot(receipt.get("startup_context"), stage="startup")
        context.check_snapshot(receipt.get("collection_context"), stage="collected")
        context.check_snapshot(receipt.get("final_context"), stage="post_pytest")
    if request["kind"] == "collection" and not _ci_lane_phase_free(receipt):
        raise ValueError("canonical lane collection recorded test execution")
    lane = receipt.get("ci_lane", {})
    authority = _ci_lane_worker_authority(request["pytest_args"])
    if (
        authority is None
        or authority[_CI_PRIVATE_OPTIONS[1]] != str(context.path)
        or authority[_CI_PRIVATE_OPTIONS[2]] != context.sha256
        or lane.get("lane") != authority[_CI_PRIVATE_OPTIONS[0]]
        or receipt.get("kind") != request["kind"]
        or receipt.get("pytest_args") != request["pytest_args"]
        or receipt.get("exitstatus") != 0
        or not receipt.get("finished")
        or lane.get("errors") != []
        or lane.get("manifest_sha256") != context.sha256
        or lane.get("manifest_path") != str(context.path)
        or lane.get("environment") != context.manifest["environment"]
        or lane.get("source_files") != context.sources
        or lane.get("source_files_after") != context.sources
    ):
        raise ValueError("canonical lane terminal audit is incomplete")
    return receipt


def pytest_addoption(parser: Any) -> None:
    group = parser.getgroup("reviewed CI lane selection")
    group.addoption("--ci-lane", choices=("standard", "extra"), default=None)
    group.addoption("--ci-main-baseline", default=None)
    group.addoption("--ci-lane-report", default=None)


def pytest_configure(config: Any) -> None:
    lane = config.getoption("ci_lane")
    baseline = config.getoption("ci_main_baseline")
    destination = config.getoption("ci_lane_report")
    if hasattr(config.option, "_j2o_ci_lane"):
        if any((lane, baseline, destination)):
            import pytest

            raise pytest.UsageError(
                "independent CI projector conflicts with canonical authority"
            )
        return
    if not any((lane, baseline, destination)):
        return
    import inspect
    import pytest

    if not all((lane, baseline, destination)):
        raise pytest.UsageError("CI lane selection needs all three explicit options")
    root = Path(config.rootpath).resolve()
    requested_collectonly = bool(config.option.collectonly)
    folder = Path(destination).resolve()
    worker = getattr(config, "workerinput", None)
    try:
        if folder.is_relative_to(root):
            raise ValueError(
                "CI lane report must be fresh and external to the checkout"
            )
        if worker is None:
            folder.mkdir(parents=True, exist_ok=False)
            (folder / "workers").mkdir()
        else:
            authorization = worker.get("ci_lane")
            if not authorization or authorization["report_root"] != str(folder):
                raise ValueError("worker has no controller CI lane authorization")
            identifier = worker["workerid"]
            if not identifier or any(
                c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in identifier
            ):
                raise ValueError("unsupported worker identity")
            folder = folder / "workers" / identifier
            folder.mkdir(exist_ok=False)
    except (OSError, ValueError) as exc:
        raise pytest.UsageError(str(exc)) from exc
    receipt: dict[str, Any] = {
        "schema": "jax2onnx-ci-lane-v1",
        "status": "FAIL",
        "finished": None,
        "acceptance_scope": "selection-and-native-phase-accounting-only",
        "lane": lane,
        "role": "worker" if worker is not None else "serial",
        "requested_collectonly": requested_collectonly,
        "started": _now(),
        "checkout": _ci_lane_checkout(root),
        "collection_args": list(getattr(config.option, "file_or_dir", [])),
        "qualified_collection_args": None,
        "full_preselection_ids": [],
        "selected_ids": [],
        "deselected_ids": [],
        "collection_skips": [],
        "collection_errors": [],
        "errors": [],
        "phases": {},
        "phase_durations": {},
        "completed": [],
        "workers": {},
        "crashes": [],
        "accounting_complete": False,
        "selector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }

    def checkpoint() -> None:
        _write(folder / "receipt.json", receipt)

    try:
        manifest, raw = _ci_lane_manifest(Path(baseline))
        if "collection_transition" in manifest:
            raise ValueError(
                "collection_transition requires canonical pre-pytest startup evidence"
            )
        collection_args = manifest.get("collection_args", [])
        receipt["qualified_collection_args"] = collection_args
        _ci_lane_qualify(config, root, manifest)
        raw_hash = hashlib.sha256(raw).hexdigest()
        if worker is not None and (
            worker["ci_lane"]["manifest_sha256"] != raw_hash
            or any(
                worker["ci_lane"][key] != receipt[key]
                for key in ("collection_args", "qualified_collection_args")
            )
        ):
            raise ValueError("worker baseline context differs from its controller")
        receipt.update(
            manifest_sha256=raw_hash,
            baseline_commit=manifest["baseline_commit"],
            python_minor=manifest["python_minor"],
            packages=manifest.get("packages", {}),
            environment=_ci_lane_environment(),
            source_files=_ci_lane_sources(root, manifest["source_files"]),
        )
        if worker is None:
            (folder / "baseline.json").write_bytes(raw)
    except (OSError, ValueError, LookupError) as exc:
        receipt["errors"].append(str(exc))
        checkpoint()
        raise pytest.UsageError(str(exc)) from exc
    checkpoint()
    context = _CiLaneContext(Path(baseline), root, expected_sha256=raw_hash)

    class LaneSelection:
        projecting = False
        selected: list[str] | None = None
        selected_set: set[str] = set()
        protocol_started = False
        serial_position = 0
        distributed = worker is None and getattr(
            config.option, "numprocesses", None
        ) not in (None, 0)

        def fail(self, message: str, *, terminal: bool = False) -> None:
            receipt["errors"].append(message)
            if not terminal:
                checkpoint()
                pytest.exit(message, returncode=4)

        def check_source(self, path: Any, *, terminal: bool = False) -> None:
            try:
                context.check_source(path)
            except (OSError, ValueError) as exc:
                self.fail(str(exc), terminal=terminal)

        def check_context(self, items: Any = (), *, terminal: bool = False) -> None:
            try:
                if config.option.collectonly is not requested_collectonly:
                    raise ValueError(
                        "CI lane collection mode changed after qualification"
                    )
                if requested_collectonly and not _ci_lane_phase_free(receipt):
                    raise ValueError("requested CI collection recorded test execution")
                if any(
                    config.getoption(name) != expected
                    for name, expected in (
                        ("ci_lane", lane),
                        ("ci_main_baseline", baseline),
                        ("ci_lane_report", destination),
                    )
                ):
                    raise ValueError(
                        "explicit CI lane authority changed after qualification"
                    )
                key = "source_files_after" if terminal else "source_files"
                receipt[key] = context.check(config, items)
            except (OSError, ValueError) as exc:
                self.fail(str(exc), terminal=terminal)

        def verify(self, session: Any) -> None:
            if (
                self.selected is None
                or [item.nodeid for item in session.items] != self.selected
            ):
                self.fail("CI lane executable selection changed after projection")

        @pytest.hookimpl(hookwrapper=True, tryfirst=True)
        def pytest_collection_modifyitems(self, session: Any, items: Any) -> Any:
            before = [item.nodeid for item in items]
            outcome = yield
            outcome.get_result()
            full = [item.nodeid for item in items]
            receipt["full_preselection_ids"] = full
            if len(set(full)) != len(full) or Counter(full) != Counter(before):
                self.fail(
                    "ordinary collection hooks changed inventory instead of ordering"
                )
            if receipt["collection_errors"]:
                self.fail("CI lane collection contains errors")
            self.check_context(items)
            try:
                baseline_ids, _ = _ci_lane_partition(full, manifest)
            except ValueError as exc:
                self.fail(str(exc))
            baseline_set = set(baseline_ids)
            selected = [
                item
                for item in items
                if (item.nodeid in baseline_set) == (lane == "standard")
            ]
            deselected = [
                item
                for item in items
                if (item.nodeid in baseline_set) != (lane == "standard")
            ]
            self.selected = [item.nodeid for item in selected]
            self.selected_set = set(self.selected)
            receipt["selected_ids"] = self.selected
            receipt["deselected_ids"] = [item.nodeid for item in deselected]
            items[:] = selected
            self.projecting = True
            try:
                config.hook.pytest_deselected(items=deselected)
            finally:
                self.projecting = False
            checkpoint()

        def pytest_deselected(self, items: Any) -> None:
            if items and not self.projecting:
                self.fail("foreign deselection prevents full CI lane collection")

        def pytest_collectreport(self, report: Any) -> None:
            if report.failed or report.skipped:
                key = "collection_errors" if report.failed else "collection_skips"
                receipt[key].append(
                    {"nodeid": report.nodeid, "longrepr": str(report.longrepr)}
                )
                checkpoint()
                self.check_source(root / report.nodeid.split("::", 1)[0])

        @pytest.hookimpl(hookwrapper=True, tryfirst=True)
        def pytest_collection_finish(self, session: Any) -> Any:
            self.verify(session)
            checkpoint()  # xdist publishes only after this outer pre-yield check.
            yield
            self.verify(session)

        @pytest.hookimpl(hookwrapper=True, tryfirst=True)
        def pytest_runtestloop(self, session: Any) -> Any:
            if not self.distributed:
                self.verify(session)
                self.check_context(session.items)
            yield

        @pytest.hookimpl(tryfirst=True)
        def pytest_runtest_protocol(self, item: Any) -> None:
            if requested_collectonly:
                self.fail("requested CI collection cannot execute a test protocol")
            if not self.protocol_started:
                self.verify(item.session)
                self.protocol_started = True
            if item.nodeid not in self.selected_set:
                self.fail("unexpected CI lane test execution")
            if worker is None and not self.distributed:
                if (
                    self.selected is None
                    or self.serial_position >= len(self.selected)
                    or item.nodeid != self.selected[self.serial_position]
                ):
                    self.fail("serial CI lane execution changed selected order")
                self.serial_position += 1

        @pytest.hookimpl(tryfirst=True)
        def pytest_fixture_setup(self, fixturedef: Any) -> None:
            self.check_source(inspect.getsourcefile(fixturedef.func))

        def pytest_runtest_logreport(self, report: Any) -> None:
            phase = {"when": report.when, "outcome": report.outcome}
            if hasattr(report, "wasxfail"):
                phase["wasxfail"] = str(report.wasxfail)
            if report.failed or report.skipped:
                phase["longrepr"] = str(report.longrepr)
            receipt["phases"].setdefault(report.nodeid, []).append(phase)
            timing = {"when": report.when, "seconds": report.duration}
            receipt["phase_durations"].setdefault(report.nodeid, []).append(timing)
            with (folder / "events.jsonl").open("a") as stream:
                stream.write(
                    json.dumps({"nodeid": report.nodeid, **phase, **timing}) + "\n"
                )

        @pytest.hookimpl(optionalhook=True)
        def pytest_configure_node(self, node: Any) -> None:
            receipt["role"] = "controller"
            node.workerinput["ci_lane"] = {
                "report_root": str(folder),
                "manifest_sha256": raw_hash,
                "collection_args": receipt["collection_args"],
                "qualified_collection_args": receipt["qualified_collection_args"],
            }
            receipt["workers"][node.gateway.id] = {
                "receipt": str(folder / "workers" / node.gateway.id / "receipt.json"),
                "published_ids": None,
                "terminal_received": False,
            }
            checkpoint()

        @pytest.hookimpl(optionalhook=True)
        def pytest_xdist_node_collection_finished(
            self, node: Any, ids: list[str]
        ) -> None:
            path = folder / "workers" / node.gateway.id / "receipt.json"
            try:
                inventory = json.loads(path.read_text())
                if (
                    inventory["manifest_sha256"] != raw_hash
                    or inventory["requested_collectonly"] is not requested_collectonly
                    or inventory["collection_args"] != receipt["collection_args"]
                    or inventory["qualified_collection_args"]
                    != receipt["qualified_collection_args"]
                    or inventory["selected_ids"] != ids
                    or inventory["collection_errors"]
                    or inventory["errors"]
                    or inventory["source_files"] != receipt["source_files"]
                ):
                    raise ValueError(
                        "worker inventory differs from its published selection"
                    )
                if self.selected is None:
                    self.selected = list(ids)
                    self.selected_set = set(ids)
                    for key in (
                        "full_preselection_ids",
                        "selected_ids",
                        "deselected_ids",
                        "collection_skips",
                    ):
                        receipt[key] = inventory[key]
                elif any(
                    receipt[key] != inventory[key]
                    for key in (
                        "full_preselection_ids",
                        "selected_ids",
                        "deselected_ids",
                        "collection_skips",
                    )
                ):
                    raise ValueError("worker CI lane inventories disagree")
                receipt["workers"][node.gateway.id]["published_ids"] = list(ids)
                checkpoint()
            except (OSError, ValueError, KeyError) as exc:
                self.fail(str(exc))

        @pytest.hookimpl(optionalhook=True)
        def pytest_testnodedown(self, node: Any, error: Any) -> None:
            if error is not None:
                receipt["crashes"].append(
                    {"worker": node.gateway.id, "error": str(error)}
                )
            try:
                record = receipt["workers"][node.gateway.id]
                payload = Path(record["receipt"]).read_bytes()
                terminal = json.loads(payload)
                if (
                    type(terminal) is not dict
                    or type(terminal.get("phases")) is not dict
                ):
                    raise ValueError(
                        "terminal worker receipt needs native phase records"
                    )
                record.update(
                    terminal_received=True,
                    terminal_sha256=hashlib.sha256(payload).hexdigest(),
                    terminal_status=terminal.get("status"),
                    terminal_exitstatus=terminal.get("exitstatus"),
                    terminal_errors=terminal.get("errors"),
                )
                if (
                    terminal["role"] != "worker"
                    or terminal["requested_collectonly"] is not requested_collectonly
                    or (requested_collectonly and not _ci_lane_phase_free(terminal))
                    or terminal["lane"] != lane
                    or terminal["manifest_sha256"] != raw_hash
                    or terminal["baseline_commit"] != manifest["baseline_commit"]
                    or terminal["collection_args"] != receipt["collection_args"]
                    or terminal["qualified_collection_args"]
                    != receipt["qualified_collection_args"]
                    or terminal["python_minor"] != manifest["python_minor"]
                    or terminal["packages"] != manifest.get("packages", {})
                    or terminal["environment"] != receipt["environment"]
                    or terminal["selector_sha256"] != receipt["selector_sha256"]
                    or terminal["source_files"] != receipt["source_files"]
                    or terminal["source_files_after"] != receipt["source_files"]
                    or any(
                        terminal[key] != receipt[key]
                        for key in (
                            "full_preselection_ids",
                            "selected_ids",
                            "deselected_ids",
                            "collection_skips",
                        )
                    )
                    or terminal["selected_ids"] != record["published_ids"]
                    or terminal["status"] != "WORKER_FINISHED"
                    or not terminal["finished"]
                    or terminal["exitstatus"] != 0
                    or terminal["errors"]
                    or terminal["collection_errors"]
                    or terminal["crashes"]
                    or not terminal["observed_phases_complete"]
                    or terminal["completed"] != _completed(terminal["phases"])
                    or set(terminal["completed"]) != set(terminal["phases"])
                    or set(terminal["phases"]) - self.selected_set
                    or any(
                        phase["outcome"] not in {"passed", "skipped"}
                        for phases in terminal["phases"].values()
                        for phase in phases
                    )
                ):
                    raise ValueError(
                        "terminal worker receipt failed its binding or accounting checks"
                    )
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.fail(f"worker {node.gateway.id}: {exc}", terminal=True)
            checkpoint()

        @pytest.hookimpl(trylast=True)
        def pytest_sessionfinish(self, session: Any, exitstatus: Any) -> None:
            receipt["native_exitstatus"] = int(exitstatus)
            receipt["finished"] = _now()
            receipt["checkout_after"] = _ci_lane_checkout(root)
            # Test bodies may import contract dependencies lazily. Audit that
            # enlarged context before success; a failure here is after-body.
            self.check_context(session.items, terminal=True)
            receipt["completed"] = _completed(receipt["phases"])
            observed_complete = set(receipt["completed"]) == set(
                receipt["phases"]
            ) and all(
                phase["outcome"] in {"passed", "skipped"}
                for phases in receipt["phases"].values()
                for phase in phases
            )
            receipt["observed_phases_complete"] = observed_complete
            if self.distributed and any(
                not record["terminal_received"]
                for record in receipt["workers"].values()
            ):
                self.fail(
                    "configured workers have missing terminal receipts", terminal=True
                )
            clean = not (
                receipt["errors"] or receipt["collection_errors"] or receipt["crashes"]
            )
            if worker is not None:
                # A worker owns only its observed assigned phases, not every ID
                # in its shared collection. Global completion belongs upstairs.
                valid = (
                    observed_complete and set(receipt["phases"]) <= self.selected_set
                )
                receipt["status"] = (
                    "WORKER_FINISHED"
                    if clean and valid and int(exitstatus) == 0
                    else "FAIL"
                )
            elif requested_collectonly:
                receipt["status"] = (
                    "COLLECTED" if clean and int(exitstatus) == 0 else "FAIL"
                )
            else:
                complete = (
                    self.selected is not None
                    and observed_complete
                    and Counter(receipt["completed"])
                    == Counter(receipt["selected_ids"])
                    and set(receipt["phases"]) == set(receipt["selected_ids"])
                    and (not self.distributed or bool(receipt["workers"]))
                )
                receipt["accounting_complete"] = complete
                receipt["status"] = (
                    "PASS" if clean and complete and int(exitstatus) == 0 else "FAIL"
                )
            if receipt["status"] == "FAIL" and int(session.exitstatus) == 0:
                session.exitstatus = 1
            receipt["exitstatus"] = int(session.exitstatus)
            if worker is not None:
                # xdist snapshots exitstatus before yielding to ordinary finish
                # hooks; update that native transport field after our audit.
                config.workeroutput["exitstatus"] = receipt["exitstatus"]
            checkpoint()

    config.pluginmanager.register(LaneSelection(), "jax2onnx-ci-lane-selection")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        raise SystemExit(_worker(Path(sys.argv[2])))
    raise SystemExit(_main())
