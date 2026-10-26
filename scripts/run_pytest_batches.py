#!/usr/bin/env python3
# scripts/run_pytest_batches.py

"""Run the entire selected suite in bounded, fresh, serial pytest processes.

Pytest options follow ``--``; optional collection paths use ``--test-path``.
Collection, raw output, phase reports, JUnit and passive resource observations
are retained in RUNNER_TEMP/jax2onnx-pytest-batches (a fresh temp dir locally).
No retry or resource-limit change can turn an incomplete run into success.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
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
        environment = {
            name: os.environ.get(name) for name in _worker_environment_names()
        }
        if (
            any(
                value is not None and (type(value) is not str or len(value) > 8192)
                for value in environment.values()
            )
            or len(json.dumps(environment).encode()) > 65536
        ):
            raise ValueError("worker environment values exceed bounds")
        result["environment"] = environment
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
    if "jax" in sys.modules:
        try:
            import types

            jax, native = sys.modules["jax"], sys.modules.get("jax._src.config")
            if (
                type(jax) is not types.ModuleType
                or type(native) is not types.ModuleType
            ):
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
                elif name == "jax_numpy_dtype_promotion" and type(value) is vars(
                    native
                ).get("NumpyDtypePromotion"):
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
                        or literal
                        != {"STANDARD": "standard", "STRICT": "strict"}[member]
                    ):
                        raise ValueError("loaded JAX enum storage differs")
                    values[name] = {
                        "type": "jax._src.config.NumpyDtypePromotion",
                        "member": member,
                        "value": literal,
                    }
                else:
                    raise ValueError("loaded JAX setting is not literal metadata")
            result["loaded_jax_config"] = {"status": "OBSERVED", "values": values}
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


def _plan(items: list[dict[str, Any]], limit: int) -> list[list[str]]:
    """Keep consecutive modules/classes together when they fit the bound."""
    groups: list[tuple[list[str], bool]] = []
    index = 0
    while index < len(items):
        end = index + 1
        module = items[index]["module"]
        while end < len(items) and items[end]["module"] == module:
            end += 1
        segment = items[index:end]
        if Path(module).name == _ISOLATED_OWNER:
            groups.extend(([item["nodeid"]], True) for item in segment)
        elif len(segment) <= limit:
            groups.append(([item["nodeid"] for item in segment], False))
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

    request = json.loads(request_path.read_text(), object_pairs_hook=unique)
    control = _diagnostic_control(request)
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

    def append_event(event: dict[str, Any]) -> None:
        # Append each observation before the next test phase. Avoid repeatedly
        # rewriting all prior phases while retaining partial interrupted runs.
        with (folder / "events.jsonl").open("a") as output:
            output.write(json.dumps(event, ensure_ascii=False) + "\n")

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

        def pytest_configure(self, config: Any) -> None:
            if getattr(config.option, "numprocesses", None) not in (None, 0):
                raise pytest.UsageError("pytest batches require serial child processes")

        def pytest_collectreport(self, report: Any) -> None:
            if report.failed or report.skipped:
                key = "collection_errors" if report.failed else "collection_skips"
                receipt[key].append(
                    {"nodeid": report.nodeid, "longrepr": str(report.longrepr)}
                )
                append_event({"kind": key, "report": receipt[key][-1]})
                checkpoint()

        @pytest.hookimpl(hookwrapper=True, tryfirst=True)
        def pytest_collection_finish(self, session: Any) -> Any:
            # Resume after all ordinary collection/order/finish hooks. Every
            # batch imports the same full suite before execution is selected.
            yield
            full_items = list(session.items)
            receipt["collection_context"] = _worker_context()
            full_ids = [item.nodeid for item in full_items]
            receipt["full_items"] = [describe(item) for item in full_items]
            receipt["full_preselection_ids"] = full_ids
            checkpoint()
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
                session.config.hook.pytest_deselected(items=deselected)
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
            append_event({"kind": "phase", "nodeid": report.nodeid, **entry})

        def pytest_sessionfinish(self, session: Any, exitstatus: int) -> None:
            receipt["exitstatus"] = int(exitstatus)
            receipt["finished"] = _now()
            checkpoint()

    def interrupted(number: int, frame: Any) -> None:
        receipt["interrupted_signal"] = number
        checkpoint()
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    checkpoint()
    try:
        result = int(
            pytest.main(
                [*request["pytest_args"], "--junitxml", str(folder / "junit.xml")],
                plugins=[Report()],
            )
        )
    except BaseException as exc:
        receipt["worker_exception"] = type(exc).__name__
        checkpoint()
        raise
    receipt["exitstatus"] = result
    receipt["finished"] = _now()
    checkpoint()
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
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= _MAX_BATCH_SIZE:
        parser.error("--batch-size must be between 1 and 500")
    options = args.pytest_args
    if options[:1] == ["--"]:
        options = options[1:]
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
    try:
        collection = root / "collection"
        request = {
            "kind": "collection",
            "pytest_args": [*options, *args.test_path, "--collect-only"],
        }
        if _child(collection, request, run, root) != 0:
            raise ValueError(
                "full-suite collection process did not finish successfully"
            )
        baseline = json.loads((collection / "receipt.json").read_text())
        items = baseline["items"]
        nodeids = [item["nodeid"] for item in items]
        if baseline["collection_errors"] or baseline["exitstatus"] != 0:
            raise ValueError("full-suite collection contains errors")
        if (
            not baseline["full_collection_checked"]
            or baseline["full_preselection_ids"] != nodeids
            or baseline["selected_ids"] != nodeids
        ):
            raise ValueError("full-suite collection receipt is incomplete")
        if not nodeids or len(set(nodeids)) != len(nodeids):
            raise ValueError("full-suite collection is empty or has duplicate node IDs")
        batches = _plan(items, args.batch_size)
        run["nodeids"] = nodeids
        run["batches"] = batches
        _write(root / "run.json", run)
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
                "collection_skips": baseline["collection_skips"],
                "pytest_args": [*options, *args.test_path],
            }
            if _child(folder, request, run, root) != 0:
                raise ValueError(f"{folder.name} process did not finish successfully")
            receipt = json.loads((folder / "receipt.json").read_text())
            _validate(receipt, assigned, nodeids, baseline["collection_skips"])
            junit = ET.parse(folder / "junit.xml")
            if len(junit.findall(".//testcase")) != len(assigned) + len(
                baseline["collection_skips"]
            ):
                raise ValueError(f"{folder.name} JUnit does not account for every item")
            run["completed"].extend(receipt["completed"])
            _write(root / "run.json", run)
        if Counter(run["completed"]) != Counter(nodeids):
            raise ValueError("completed batch union differs from the full collection")
        run["status"] = "PASS"
    except BaseException as exc:
        run["status"] = "FAIL"
        run["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        print(f"[pytest-batches] FAIL: {exc}", file=sys.stderr, flush=True)
    run["finished"] = _now()
    _write(root / "run.json", run)
    if run["status"] == "PASS":
        print(
            f"[pytest-batches] PASS: all {len(run['completed'])} collected items completed once",
            flush=True,
        )
        return 0
    return 1


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        raise SystemExit(_worker(Path(sys.argv[2])))
    raise SystemExit(_main())
