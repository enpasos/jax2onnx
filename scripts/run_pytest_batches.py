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
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any
import xml.etree.ElementTree as ET


_MAX_BATCH_SIZE = 500
_ISOLATED_OWNER = "test_source_model_native_bindings_exports.py"


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
    # Import pytest only inside disposable collection/execution children.
    import pytest

    request = json.loads(request_path.read_text())
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
    folder: Path, request: dict[str, Any], run: dict[str, Any], root: Path
) -> int:
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
                result = child.wait(timeout=60)
                break
            except subprocess.TimeoutExpired:
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
