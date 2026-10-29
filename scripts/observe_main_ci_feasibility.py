# scripts/observe_main_ci_feasibility.py

"""One fixed hosted main diagnostic; never selects tests or changes JAX settings.

Only this scripts directory is exposed to subject pytest. The driver package is
never installed or placed on sys.path. Worker records prove observed execution
ownership, not scheduler assignment history or loaded JAX configuration.
"""

from __future__ import annotations

import argparse
import ctypes
from collections import defaultdict, deque
import hashlib
from importlib import metadata, util
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import xml.etree.ElementTree as ET


GiB = 2**30
WORKERS = {"gw0", "gw1"}
NATIVE_CONTROLS = (
    "JAX_PLATFORMS",
    "JAX_PLATFORM_NAME",
    "JAX_ENABLE_X64",
    "XLA_FLAGS",
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
    "JAX_COMPILATION_CACHE_DIR",
)


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def _append(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value) + "\n")


def _read(path):
    return json.loads(Path(path).read_text())


def _events(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _phase_sequence(nodeid, events, authority):
    allowance = {item["nodeid"]: item for item in authority["skips"]}.get(nodeid)
    pattern = [("setup", "passed"), ("call", "passed"), ("teardown", "passed")]
    if allowance:
        pattern = (
            [("setup", "skipped"), ("teardown", "passed")]
            if allowance["phase"] == "setup"
            else [("setup", "passed"), ("call", "skipped"), ("teardown", "passed")]
        )
    _require(
        [(event["when"], event["outcome"]) for event in events] == pattern,
        "missing, duplicate, failed or unexpected phase",
    )
    for event in events:
        _require(
            event.get("wasxfail_present") is False and "wasxfail" not in event,
            "wasxfail presence or absent observation",
        )
        _require(
            type(event.get("duration")) in (int, float)
            and 0 <= event["duration"] < float("inf"),
            "invalid native phase duration",
        )
        if event["outcome"] == "skipped":
            location = event.get("skip_location")
            owner = nodeid.split("::", 1)[0]
            _require(
                type(location) is list
                and len(location) == 3
                and type(location[0]) is str
                and (location[0] == owner or location[0].endswith("/" + owner))
                and type(location[1]) is int
                and location[1] == allowance["line"]
                and location[2] == "Skipped: " + allowance["reason"]
                and event.get("longrepr") in (str(tuple(location)), str(location)),
                "wrong exact skip allowance",
            )


def _validate(authority, collections, controller_events, closures):
    _require(
        set(collections) == set(closures) == WORKERS,
        "two native worker closures required",
    )
    expected = authority["collection"]
    for ids in collections.values():
        _require(
            type(ids) is list
            and len(ids) == len(set(ids)) == expected["count"]
            and _digest(ids) == expected["ordered_ids_sha256"],
            "worker full collection differs",
        )
    _require(
        collections["gw0"] == collections["gw1"], "worker collection orders differ"
    )
    for worker, closure in closures.items():
        _require(
            closure.get("worker_id") == worker
            and closure.get("closed") is True
            and closure.get("error") is None
            and type(closure.get("exitstatus")) is int
            and closure["exitstatus"] == 0
            and not closure.get("shouldfail")
            and not closure.get("shouldstop"),
            "missing or failed native worker closure",
        )
    phases, owners = defaultdict(list), {}
    for event in controller_events:
        worker, index = event.get("worker_id"), event.get("item_index")
        _require(
            worker in WORKERS and event.get("transport_worker_id") == worker,
            "native worker differs from trusted transport",
        )
        _require(
            type(index) is int
            and 0 <= index < len(collections[worker])
            and collections[worker][index] == event.get("nodeid"),
            "native collection index differs",
        )
        _require(
            type(event.get("testrun_uid")) is str
            and bool(event["testrun_uid"])
            and event["testrun_uid"] == closures[worker]["testrun_uid"],
            "native run UID differs",
        )
        node = event["nodeid"]
        _require(
            owners.setdefault(node, worker) == worker,
            "cross-worker execution ownership",
        )
        phases[node].append(event)
    _require(set(phases) == set(collections["gw0"]), "missing or extra executed ID")
    for node, events in phases.items():
        _phase_sequence(node, events, authority)
    _require(
        len(controller_events) == expected["phase_count"], "native phase count differs"
    )
    return {
        "count": len(phases),
        "passed": len(phases) - len(authority["skips"]),
        "skipped": len(authority["skips"]),
        "phase_count": len(controller_events),
    }


def _git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args]).decode().strip()


def _source(subject):
    files = {
        name: _sha(subject / name)
        for name in _git(subject, "ls-files", "-z").split("\0")
        if name
    }
    generated = {
        str(path.relative_to(subject)): _sha(path)
        for path in sorted((subject / "tests").rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    return {
        "head": _git(subject, "rev-parse", "HEAD"),
        "files": files,
        "generated": generated,
        "tracked_status": _git(
            subject, "status", "--porcelain=v1", "--untracked-files=no"
        ),
    }


def _check_source(source, authority, *, generated=False):
    expected = authority["subject"]
    _require(
        source["head"] == expected["commit"]
        and not source["tracked_status"]
        and _digest(source["files"]) == expected["source_digest"],
        "tracked subject source drift",
    )
    if generated:
        _require(
            _digest(source["generated"]) == expected["generated_digest"],
            "post-generation source authority differs",
        )


def _binding(subject, authority):
    source = _source(subject)
    distributions = sorted(
        (dist.metadata["Name"], dist.version) for dist in metadata.distributions()
    )
    plugins = sorted(
        (dist.metadata["Name"], ep.name, ep.value)
        for dist in metadata.distributions()
        for ep in dist.entry_points
        if ep.group == "pytest11"
    )
    origins = {
        name: util.find_spec(name).origin for name in ("jax2onnx", "pytest", "xdist")
    }
    _require(
        Path(origins["jax2onnx"]).resolve().is_relative_to(subject),
        "candidate package overlay",
    )
    driver = Path(__file__).resolve().parents[1]
    _require(
        all(Path(entry or ".").resolve() != driver for entry in sys.path),
        "driver root on sys.path",
    )
    native = {
        name: _sha(Path(origins["xdist"]).parent / name)
        for name in authority["xdist_sources"]
    }
    _require(native == authority["xdist_sources"], "native xdist source differs")
    return {
        "source": source,
        "distributions": distributions,
        "plugins": plugins,
        "static_origins": origins,
        "python": sys.executable,
        "version": sys.version,
        "cwd": str(Path.cwd()),
        "observer_sha256": _sha(__file__),
    }


def _environment():
    names = {
        *NATIVE_CONTROLS,
        "PYTHONPATH",
        "PYTEST_ADDOPTS",
        "PYTEST_XDIST_AUTO_NUM_WORKERS",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
        "PYTHONHASHSEED",
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
    }
    names.update(
        name for name in os.environ if name.startswith(("JAX2ONNX_", "PYTEST_XDIST_"))
    )
    return {name: os.environ.get(name) for name in sorted(names)}


def _phase(report):
    event = {
        "nodeid": report.nodeid,
        "when": report.when,
        "outcome": report.outcome,
        "duration": report.duration,
        "wasxfail_present": hasattr(report, "wasxfail"),
    }
    if hasattr(report, "wasxfail"):
        event["wasxfail"] = str(report.wasxfail)
    if report.failed or report.skipped:
        event["longrepr"] = str(report.longrepr)
    if report.skipped and type(report.longrepr) in (tuple, list):
        event["skip_location"] = list(report.longrepr)
    return event


def pytest_configure(config):
    """Only the native controller observes; workers have no file protocol."""
    if hasattr(config, "workerinput"):
        return
    import pytest

    evidence = Path(os.environ["JAX2ONNX_MAIN_EVIDENCE"]) / "pytest"
    authority = _read(os.environ["JAX2ONNX_MAIN_AUTHORITY"])
    evidence.mkdir(parents=True, exist_ok=False)

    class Observation:
        def __init__(self):
            self.record = {
                "collections": {},
                "configured": {},
                "closures": {},
                "errors": [],
                "native_controller_exitstatus": None,
            }

        def save(self):
            _write(evidence / "controller.json", self.record)

        def fail(self, message):
            self.record["errors"].append(str(message))
            self.save()
            pytest.exit(str(message), returncode=1)

        @pytest.hookimpl(optionalhook=True)
        def pytest_configure_node(self, node):
            worker = node.gateway.id
            if worker not in WORKERS or worker in self.record["configured"]:
                self.fail("unexpected or replacement worker")
            self.record["configured"][worker] = node.workerinput["testrunuid"]
            self.save()

        @pytest.hookimpl(optionalhook=True, tryfirst=True)
        def pytest_xdist_node_collection_finished(self, node, ids):
            worker = node.gateway.id
            expected = authority["collection"]
            if (
                worker not in self.record["configured"]
                or worker in self.record["collections"]
                or len(ids) != len(set(ids))
                or len(ids) != expected["count"]
                or _digest(ids) != expected["ordered_ids_sha256"]
            ):
                self.fail("full worker collection rejected before native scheduling")
            self.record["collections"][worker] = list(ids)
            self.save()

        def pytest_collectreport(self, report):
            if report.failed or report.skipped:
                self.fail(
                    "unexpected failed/skipped collector: " + str(report.longrepr)
                )

        def pytest_deselected(self, items):
            if items:
                self.fail("deselection is forbidden")

        def pytest_runtest_logreport(self, report):
            event = _phase(report)
            for name in ("worker_id", "item_index", "testrun_uid"):
                if hasattr(report, name):
                    event[name] = getattr(report, name)
            event["transport_worker_id"] = getattr(
                getattr(getattr(report, "node", None), "gateway", None), "id", None
            )
            _append(evidence / "phases.jsonl", event)
            worker, index = event.get("worker_id"), event.get("item_index")
            collections = self.record["collections"]
            if (
                set(collections) != WORKERS
                or worker not in collections
                or event["transport_worker_id"] != worker
                or type(index) is not int
                or not 0 <= index < len(collections[worker])
                or collections[worker][index] != report.nodeid
                or event.get("testrun_uid") != self.record["configured"][worker]
            ):
                self.fail("native phase transport/index identity differs")

        @pytest.hookimpl(optionalhook=True)
        def pytest_testnodedown(self, node, error):
            worker = node.gateway.id
            if (
                worker not in self.record["configured"]
                or worker in self.record["closures"]
            ):
                self.fail("unexpected or duplicate native worker closure")
            output = getattr(node, "workeroutput", {})
            self.record["closures"][worker] = {
                "worker_id": worker,
                "closed": True,
                "testrun_uid": self.record["configured"][worker],
                "error": str(error) if error is not None else None,
                **{
                    name: output.get(name)
                    for name in ("exitstatus", "shouldfail", "shouldstop")
                },
            }
            self.save()

        @pytest.hookimpl(trylast=True)
        def pytest_sessionfinish(self, session, exitstatus):
            self.record["native_controller_exitstatus"] = int(exitstatus)
            self.save()

    config.pluginmanager.register(Observation(), "frozen-main-feasibility")


def _read_stat(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    result = {
        "parent": int(fields[1]),
        "start": int(fields[19]),
        "rss": int(fields[21]) * os.sysconf("SC_PAGE_SIZE"),
        "state": fields[0],
    }
    _require(result["start"] >= 0 and result["rss"] >= 0, "invalid owned process stat")
    return result


def _children(pid):
    # Children may have been forked by any thread; sessions/process groups do not establish ownership.
    children = set()
    for task in Path(f"/proc/{pid}/task").iterdir():
        try:
            children.update(
                int(child) for child in (task / "children").read_text().split()
            )
        except FileNotFoundError:
            if task.exists():
                raise
    return children


def _after_read_error(pid, expected_start, error):
    """A missing stat is not zero RSS. Only verified exit/reuse permits omission."""
    try:
        fresh = _read_stat(pid)
    except (OSError, ValueError, IndexError):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return None
        except PermissionError:
            pass
    else:
        if fresh["state"] == "Z":
            return fresh
        if expected_start is not None and fresh["start"] != expected_start:
            return None  # The previously owned identity exited; never adopt this replacement blindly.
    raise ValueError(f"owned PID {pid} live/uncertain after telemetry error: {error}")


def _owned_tree(owned, leader):
    guard = os.getpid()
    guard_identity = (guard, _read_stat(guard)["start"])

    def current(identity):
        pid, start = identity
        try:
            return _read_stat(pid)
        except (OSError, ValueError, IndexError) as error:
            return _after_read_error(pid, start, error)

    def same(item, identity):
        return item is not None and item["start"] == identity[1]

    def guard_children():
        _require(
            same(current(guard_identity), guard_identity),
            "persistent guard identity changed",
        )
        children = _children(guard)
        _require(
            same(current(guard_identity), guard_identity),
            "guard identity changed during enumeration",
        )
        return children

    def adopted(pid, item):
        # A vanished/reused numeric parent is never authority. Prove direct adoption independently.
        members = guard_children()
        fresh = current((pid, item["start"]))
        if fresh is None:
            return None
        _require(
            pid in members
            and fresh["start"] == item["start"]
            and fresh["parent"] == guard
            and same(current(guard_identity), guard_identity),
            "child lineage/guard adoption is uncertain",
        )
        return fresh

    pending = deque(
        [
            (leader, guard_identity),
            *((pid, guard_identity) for pid in guard_children()),
            *((pid, None) for pid in owned),
        ]
    )
    table, seen = {}, set()
    while pending:
        pid, parent_identity = pending.popleft()
        if pid in seen:
            continue
        seen.add(pid)
        expected = owned.get(pid)
        try:
            item = _read_stat(pid)
        except (OSError, ValueError, IndexError) as error:
            item = _after_read_error(pid, expected, error)
        if item is None:
            continue
        if expected != item["start"]:
            if parent_identity is None:
                continue  # Reused historical PID, without a newly observed identity-bound edge.
            before = current(parent_identity)
            fresh = current((pid, item["start"]))
            after = current(parent_identity)
            if fresh is None:
                continue
            item = fresh
            if not (
                same(before, parent_identity)
                and same(after, parent_identity)
                and item["parent"] == parent_identity[0]
            ):
                item = adopted(pid, item)
                if item is None:
                    continue
            owned[pid] = item["start"]
        table[pid] = item
        if item["state"] != "Z":
            identity = (pid, item["start"])
            before = current(identity)
            if not same(before, identity):
                _require(
                    before is None, "parent identity replaced before child enumeration"
                )
                table.pop(pid, None)
                continue
            try:
                children = _children(pid)
            except (OSError, ValueError) as error:
                fresh = _after_read_error(pid, item["start"], error)
                if fresh is None or fresh["state"] == "Z":
                    table.pop(pid, None)
                continue
            after = current(identity)
            _require(
                after is None or same(after, identity),
                "parent identity replaced during child enumeration",
            )
            pending.extend((child, identity) for child in children)
            if after is None or after["state"] == "Z":
                table.pop(pid, None)
    return table


def _signal_owned(owned, snapshot, signum):
    uncertainty = []
    for pid, item in sorted(
        snapshot.items(), key=lambda pair: pair[1]["start"], reverse=True
    ):
        try:
            try:
                fresh = _read_stat(pid)
            except (OSError, ValueError, IndexError) as error:
                fresh = _after_read_error(pid, owned.get(pid), error)
            if (
                fresh is not None
                and owned.get(pid) == item["start"] == fresh["start"]
                and fresh["state"] != "Z"
            ):
                os.kill(pid, signum)
        except ProcessLookupError:
            pass
        except (OSError, ValueError) as error:
            uncertainty.append({"pid": pid, "error": str(error)})
    return uncertainty


def _subreaper(value=None):
    # Linux PR_SET/GET_CHILD_SUBREAPER, no runtime/compiler substitution.
    # https://man7.org/linux/man-pages/man2/PR_SET_CHILD_SUBREAPER.2const.html
    libc = ctypes.CDLL(None, use_errno=True)
    setting = ctypes.c_int()
    argument = ctypes.byref(setting) if value is None else ctypes.c_ulong(value)
    result = libc.prctl(
        37 if value is None else 36,
        argument,
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
    )
    if result != 0:
        raise OSError(ctypes.get_errno(), "PR_GET/SET_CHILD_SUBREAPER unavailable")
    return setting.value if value is None else None


def _reap_adopted(owned, leader, table):
    reaped = []
    for pid, item in table.items():
        if pid == leader or item["parent"] != os.getpid() or item["state"] != "Z":
            continue
        fresh = _read_stat(pid)
        if fresh["start"] == owned.get(pid) and fresh["parent"] == os.getpid():
            child, status = os.waitpid(pid, os.WNOHANG)
            if child:
                reaped.append({"pid": child, "start": fresh["start"], "status": status})
    return reaped


def _telemetry(subject):
    memory = {
        line.split()[0].rstrip(":"): int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
    }
    vm = {
        name: int(value)
        for name, value in (
            line.split() for line in Path("/proc/vmstat").read_text().splitlines()
        )
    }
    membership = Path("/proc/self/cgroup").read_text()
    path = next(line[3:] for line in membership.splitlines() if line.startswith("0::"))
    cgroup = Path("/sys/fs/cgroup") / path.lstrip("/")
    maximum = (cgroup / "memory.max").read_text().strip()
    current = int((cgroup / "memory.current").read_text())
    events = dict(
        line.split() for line in (cgroup / "memory.events").read_text().splitlines()
    )
    available = memory["MemAvailable"]
    return {
        "monotonic": time.monotonic(),
        "host_available": available,
        "effective_available": (
            min(available, max(0, int(maximum) - current))
            if maximum != "max"
            else available
        ),
        "disk_free": shutil.disk_usage(subject).free,
        "SwapFree": memory["SwapFree"],
        "pswpin": vm["pswpin"],
        "pswpout": vm["pswpout"],
        "host_oom_kill": vm["oom_kill"],
        "cgroup_oom": int(events["oom"]),
        "cgroup_oom_kill": int(events["oom_kill"]),
        "cgroup_path": str(cgroup),
        "cgroup_current": current,
        "cgroup_max": maximum,
        "cgroup_peak": int((cgroup / "memory.peak").read_text()),
    }


def _stop_reason(sample, baseline, rss, elapsed):
    names = (
        "effective_available",
        "disk_free",
        "SwapFree",
        "pswpin",
        "pswpout",
        "host_oom_kill",
        "cgroup_oom",
        "cgroup_oom_kill",
    )
    if any(
        type(sample.get(name)) is not int
        or sample[name] < 0
        or type(baseline.get(name)) is not int
        or baseline[name] < 0
        for name in names
    ):
        return "missing/invalid safety telemetry"
    if (
        rss > 4 * GiB
        or sample["effective_available"] < 2 * GiB
        or sample["disk_free"] < 4 * GiB
    ):
        return "aggregate RSS/available memory/disk guard"
    if elapsed > 1200:
        return "1200-second workload guard"
    if sample["SwapFree"] < baseline["SwapFree"] or any(
        sample[name] != baseline[name] for name in names[3:]
    ):
        return "host swap/OOM counter activity or regression; cause unassigned"
    return None


def _admit(sample):
    _require(
        _stop_reason(sample, sample, 0, 0) is None
        and sample["effective_available"] >= 6 * GiB,
        "hosted admission requires 6-GiB reserve/disk/telemetry",
    )


def _guard(command, subject, evidence, label, seconds, environment):
    allowed = sorted(os.sched_getaffinity(0))
    _require(
        len(allowed) == 4,
        f"unexpected hosted CPU allocation: {allowed}; affinity unchanged",
    )
    _require(not _children(os.getpid()), "unrelated preexisting child workload")
    prior_subreaper, prior_term = _subreaper(), signal.getsignal(signal.SIGTERM)
    owned, ledger, reaped, uncertainty = {}, {}, [], []
    process, code, baseline, stop, peak = None, None, None, None, 0
    cleanup_started = None
    requested_term = False
    started = time.monotonic()
    _write(evidence / (label + "-command.json"), command)

    def term(_number, _frame):
        nonlocal requested_term
        requested_term = (
            True  # Do not interrupt Popen before its leader handle is assigned.
        )

    def sample():
        nonlocal peak
        table = _owned_tree(owned, process.pid)
        for pid, start in owned.items():
            if ledger.get(pid) != start:
                _append(
                    evidence / (label + "-pid-starts.jsonl"),
                    {"pid": pid, "start": start},
                )
                ledger[pid] = start
        reaped.extend(_reap_adopted(owned, process.pid, table))
        live = {pid: item for pid, item in table.items() if item["state"] != "Z"}
        peak = max(peak, sum(item["rss"] for item in live.values()))
        return live

    def cleanup():
        nonlocal cleanup_started
        if process is None:
            return
        if cleanup_started is None:
            cleanup_started = time.monotonic()
        for signum, limit in ((signal.SIGTERM, 10), (signal.SIGKILL, 15)):
            deadline = cleanup_started + limit
            while time.monotonic() < deadline:
                try:
                    live = sample()
                except (OSError, ValueError, IndexError) as error:
                    uncertainty.append(str(error))
                    live = {pid: {"start": start} for pid, start in owned.items()}
                uncertainty.extend(_signal_owned(owned, live, signum))
                process.poll()  # Popen alone reaps its leader.
                if not live and not _children(os.getpid()):
                    return
                time.sleep(0.2)

    try:
        _subreaper(1)
        _require(_subreaper() == 1, "subreaper setup did not take effect")
        signal.signal(signal.SIGTERM, term)
        baseline = _telemetry(subject)
        _append(
            evidence / (label + "-resources.jsonl"), {"stage": "prelaunch", **baseline}
        )
        _admit(baseline)
        with (evidence / (label + ".log")).open("w") as stream:
            process = subprocess.Popen(
                command,
                cwd=subject,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            while True:
                live = sample()
                rss = sum(item["rss"] for item in live.values())
                peak = max(peak, rss)
                state = _telemetry(subject)
                _append(
                    evidence / (label + "-resources.jsonl"),
                    {"stage": "during", "rss": rss, **state},
                )
                stop = _stop_reason(state, baseline, rss, time.monotonic() - started)
                if requested_term:
                    stop = "SIGTERM received by owned guard"
                if time.monotonic() - started > seconds:
                    stop = label + " time budget"
                code = process.poll()
                if code is not None:
                    live = (
                        sample()
                    )  # The earlier census may still contain the now-reaped leader.
                    if live:
                        stop = stop or "leader exited with adopted/live descendants"
                if stop:
                    cleanup()
                    break
                if code is not None and not live:
                    break
                time.sleep(0.2)
            code = process.wait(timeout=5)
    except BaseException as error:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        stop = f"{type(error).__name__}: {error}"
        cleanup()
        code = process.poll() if process is not None else None
    try:
        state = _telemetry(subject)
        _append(evidence / (label + "-resources.jsonl"), {"stage": "post", **state})
        stop = stop or _stop_reason(
            state, baseline or {}, peak, time.monotonic() - started
        )
    except (OSError, ValueError, IndexError, StopIteration) as error:
        uncertainty.append(str(error))
    try:
        remaining = sample() if process is not None else {}
        clean = not remaining and not _children(os.getpid())
    except (OSError, ValueError, IndexError) as error:
        uncertainty.append(str(error))
        clean = False
    if requested_term:
        stop = stop or "SIGTERM received by owned guard"
    if clean:
        _subreaper(prior_subreaper)
    signal.signal(signal.SIGTERM, prior_term)
    result = {
        "exit": code,
        "stop": stop,
        "seconds": time.monotonic() - started,
        "owned_peak_bytes": peak,
        "owned_pids": owned,
        "allowed_cpus": allowed,
        "affinity_changed": False,
        "reaped_adopted": reaped,
        "cleanup_complete": clean,
        "cleanup_uncertainty": uncertainty,
        "prior_subreaper": prior_subreaper,
        "subreaper_restored": clean,
    }
    _write(evidence / (label + "-result.json"), result)
    _require(
        code == 0 and stop is None and clean and not uncertainty,
        label + " failed/incomplete",
    )
    return result


def _stage(name, command, subject, evidence):
    started = time.monotonic()
    with (evidence / (name + ".log")).open("w") as stream:
        result = subprocess.run(
            command, cwd=subject, stdout=stream, stderr=subprocess.STDOUT
        )
    _write(
        evidence / (name + "-result.json"),
        {
            "command": command,
            "exit": result.returncode,
            "seconds": time.monotonic() - started,
        },
    )
    _require(result.returncode == 0, name + " failed")


def _junit(path, authority):
    """Corroborating totals/exact skips; the controller stream owns ID accounting."""
    tree = ET.parse(path).getroot()
    cases, suites = tree.findall(".//testcase"), tree.findall(".//testsuite")
    _require(
        len(cases) == authority["collection"]["count"]
        and sum(int(suite.get("tests", "0")) for suite in suites) == len(cases)
        and not tree.findall(".//error")
        and not tree.findall(".//failure"),
        "JUnit totals/errors differ",
    )
    skipped = [case for case in cases if case.find("skipped") is not None]
    _require(len(skipped) == len(authority["skips"]), "JUnit skip count differs")
    for allowance in authority["skips"]:
        owner, name = allowance["nodeid"].split("::", 1)
        matches = [
            case
            for case in skipped
            if case.get("name") == name
            and (
                case.get("file") == owner or case.get("file", "").endswith("/" + owner)
            )
        ]
        _require(len(matches) == 1, "JUnit exact skip identity differs")
        skip = matches[0].find("skipped")
        _require(
            skip.get("type") != "pytest.xfail"
            and skip.get("message") == allowance["reason"],
            "JUnit exact skip reason/xfail differs",
        )


def _close(subject, driver, evidence, authority, expected_driver, failure):
    started = time.monotonic()
    verdict = {
        "status": "FAILED_OR_INCOMPLETE",
        "errors": [failure] if failure else [],
        "scope": "One observed hosted main311 feasibility run, not NoIncrease or migration acceptance.",
        "unverified": [
            "unseen scheduler assignments",
            "loaded worker JAX/source/package context",
            "other cohorts/support",
            "paired performance",
            "upload cost until job-log audit",
        ],
        "closure_budget_policy": "Measured retrospective <=300s qualification only; hard35m job envelope.",
    }
    try:
        source = _source(subject)
        _write(evidence / "subject-source-after.json", source)
        _check_source(source, authority)
        _require(
            _git(driver, "rev-parse", "HEAD") == expected_driver
            and not _git(driver, "status", "--porcelain=v1", "--untracked-files=no"),
            "driver drift",
        )
        _require(not failure, failure)
        before, after = _read(evidence / "binding-before.json"), _read(
            evidence / "binding-after.json"
        )
        _check_source(before["source"], authority)
        _check_source(after["source"], authority, generated=True)
        _require(
            {key: value for key, value in before.items() if key != "source"}
            == {key: value for key, value in after.items() if key != "source"},
            "workload package binding drift",
        )
        controller = _read(evidence / "pytest/controller.json")
        _require(
            controller["errors"] == []
            and controller["native_controller_exitstatus"] == 0,
            "native controller did not close successfully",
        )
        verdict["counts"] = _validate(
            authority,
            controller["collections"],
            _events(evidence / "pytest/phases.jsonl"),
            controller["closures"],
        )
        _junit(evidence / "junit.xml", authority)
        verdict["status"] = "FINITE_HOSTED_MAIN_FEASIBILITY_PASS"
    except (OSError, ValueError, KeyError, TypeError, ET.ParseError) as error:
        verdict["errors"].append(str(error))
    payloads = {}
    for root in (
        subject / "onnx",
        evidence.parent / "main-feasibility-caches/pytest-tmp",
    ):
        if root.exists():
            payloads[str(root)] = {
                str(path.relative_to(root)): {
                    "bytes": path.stat().st_size,
                    "sha256": _sha(path),
                }
                for path in sorted(root.rglob("*"))
                if path.is_file()
            }
    _write(evidence / "model-payloads-NOT_RETAINED.json", payloads)
    manifest = {
        str(path.relative_to(evidence)): {
            "bytes": path.stat().st_size,
            "sha256": _sha(path),
        }
        for path in sorted(evidence.rglob("*"))
        if path.is_file()
        and path.name not in {"evidence.tar.gz", "verdict.json", "sha256-manifest.json"}
    }
    _write(evidence / "sha256-manifest.json", manifest)
    with tarfile.open(evidence / "evidence.tar.gz", "w:gz") as archive:
        for name in [*manifest, "sha256-manifest.json"]:
            archive.add(evidence / name, arcname=name, recursive=False)
    size = (evidence / "evidence.tar.gz").stat().st_size
    verdict.update(
        archive_bytes=size,
        archive_sha256=_sha(evidence / "evidence.tar.gz"),
        closure_seconds=time.monotonic() - started,
        model_payloads="NOT_RETAINED; hashes/counts/bytes retained",
    )
    if size > 99 * 2**20 or verdict["closure_seconds"] > 300:
        verdict["status"] = "FAILED_OR_INCOMPLETE"
        verdict["errors"].append(
            "artifact/retrospective closure budget exceeded; full scoped records preserved"
        )
    _write(evidence / "verdict.json", verdict)
    if os.environ.get("GITHUB_OUTPUT"):
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
            stream.write("bundle_ready=true\n")
    return 0 if verdict["status"] == "FINITE_HOSTED_MAIN_FEASIBILITY_PASS" else 1


def _invariants(driver, subject, authority, expected_driver):
    _require(
        os.environ.get("GITHUB_ACTIONS") == "true",
        "hosted-only diagnostic; no local main retry",
    )
    event = _read(os.environ["GITHUB_EVENT_PATH"])
    _require(
        os.environ.get("GITHUB_EVENT_NAME") == "pull_request"
        and event["number"] == 277
        and event["action"] == "synchronize"
        and event["pull_request"]["head"]["sha"] == expected_driver
        and event["pull_request"]["head"]["repo"]["full_name"] == "enpasos/jax2onnx"
        and event["pull_request"]["head"]["ref"] == "precision-migration",
        "unexpected hosted PR authority",
    )
    _require(
        _git(driver, "rev-parse", "HEAD") == expected_driver
        and not _git(driver, "status", "--porcelain=v1", "--untracked-files=no"),
        "driver must equal clean PR event head",
    )
    _check_source(
        _source(subject), authority
    )  # Tracked fence; generation may not have run yet.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--expected-driver", required=True)
    roles = parser.add_mutually_exclusive_group()
    roles.add_argument("--bootstrap", action="store_true")
    roles.add_argument("--workload", action="store_true")
    args = parser.parse_args()
    driver = Path(__file__).resolve().parents[1]
    authority_path = driver / "internal_docs/precision/ci_main_py311_81ad6ca.json"
    authority = _read(authority_path)
    subject, evidence = args.subject.resolve(), args.evidence.resolve()
    mode = "bootstrap" if args.bootstrap else "workload" if args.workload else "outer"
    evidence.mkdir(parents=True, exist_ok=mode != "outer")
    admission = {
        "mode": mode,
        "expected_driver": args.expected_driver,
        "environment": _environment(),
    }
    try:
        admission["source"] = _source(subject)
        _write(evidence / ("admission-" + mode + ".json"), admission)
        _invariants(driver, subject, authority, args.expected_driver)
    except (OSError, ValueError, KeyError) as error:
        admission.update(status="FAILED_ADMISSION_NO_STAGE_LAUNCHED", error=str(error))
        _write(evidence / ("admission-" + mode + ".json"), admission)
        raise
    common = [
        str(Path(__file__).resolve()),
        "--subject",
        str(subject),
        "--evidence",
        str(evidence),
        "--expected-driver",
        args.expected_driver,
    ]
    if args.bootstrap:
        for name, command in (
            (
                "install-poetry",
                [sys.executable, "-m", "pip", "install", "poetry>=2.0,<3.0"],
            ),
            ("poetry-install", ["poetry", "install", "--no-interaction"]),
            ("install-xdist", ["poetry", "run", "pip", "install", "pytest-xdist"]),
        ):
            _stage(name, command, subject, evidence)
        executable = (
            subprocess.check_output(
                ["poetry", "env", "info", "--executable"], cwd=subject
            )
            .decode()
            .strip()
        )
        _write(evidence / "subject-python.json", executable)
        return 0
    if args.workload:
        _write(evidence / "binding-before.json", _binding(subject, authority))
        try:
            for name, command in (
                (
                    "path-markers",
                    [sys.executable, "scripts/ensure_path_markers.py", "--fix"],
                ),
                (
                    "pre-commit",
                    [sys.executable, "-m", "pre_commit", "run", "--all-files"],
                ),
                ("generation", [sys.executable, "scripts/generate_tests.py"]),
            ):
                _stage(name, command, subject, evidence)
                source = _source(subject)
                _write(evidence / (name + "-sources.json"), source)
                _check_source(source, authority, generated=name == "generation")
            _stage(
                "pytest",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-p",
                    "observe_main_ci_feasibility",
                    "-n",
                    "auto",
                    "--dist=loadscope",
                    "--durations=50",
                    "--durations-min=1",
                    "--max-worker-restart=0",
                    "--junitxml=" + str(evidence / "junit.xml"),
                    "-o",
                    "junit_family=xunit1",
                    "--basetemp="
                    + str(evidence.parent / "main-feasibility-caches/pytest-tmp"),
                    "-o",
                    "cache_dir="
                    + str(evidence.parent / "main-feasibility-caches/pytest-cache"),
                ],
                subject,
                evidence,
            )
        finally:
            _write(evidence / "binding-after.json", _binding(subject, authority))
        return 0
    _require(
        "PYTEST_ADDOPTS" not in os.environ
        and not any(name in os.environ for name in NATIVE_CONTROLS),
        "original native/thread/addopts controls must remain unset",
    )
    _require(
        "PYTHONPATH" not in os.environ and not (driver / "scripts/jax2onnx").exists(),
        "driver overlay forbidden",
    )
    _require(
        not any(name.startswith("JAX2ONNX_") for name in os.environ),
        "foreign subject profile",
    )
    _write(evidence / "authority.json", authority)
    _write(evidence / "subject-source-before.json", _source(subject))
    cpuinfo = Path("/proc/cpuinfo").read_text()
    _write(
        evidence / "host-preflight.json",
        {
            "allowed_cpus": sorted(os.sched_getaffinity(0)),
            "cpu_models": sorted(
                {
                    line.split(":", 1)[1].strip()
                    for line in cpuinfo.splitlines()
                    if line.startswith("model name")
                }
            ),
            "cpu_flags": sorted(
                {
                    line.split(":", 1)[1].strip()
                    for line in cpuinfo.splitlines()
                    if line.startswith("flags")
                }
            ),
            "MemTotal_bytes": int(
                next(
                    line.split()[1]
                    for line in Path("/proc/meminfo").read_text().splitlines()
                    if line.startswith("MemTotal:")
                )
            )
            * 1024,
        },
    )
    _write(
        evidence / "driver.json",
        {
            "head": args.expected_driver,
            "observer_sha256": _sha(__file__),
            "authority_sha256": _sha(authority_path),
            "python": sys.version,
            "runner": {
                key: os.environ.get(key)
                for key in (
                    "RUNNER_OS",
                    "RUNNER_ARCH",
                    "ImageOS",
                    "ImageVersion",
                    "RUNNER_TOOL_CACHE",
                )
            },
            "platform": platform.platform(),
            "initial_environment": _environment(),
        },
    )
    environment = os.environ.copy()
    caches = evidence.parent / "main-feasibility-caches"
    for key, value in (
        ("PIP_CACHE_DIR", "pip"),
        ("POETRY_CACHE_DIR", "poetry"),
        ("PRE_COMMIT_HOME", "pre-commit"),
    ):
        _require(not (caches / value).exists(), "fresh cache path already exists")
        environment[key] = str(caches / value)
    environment.update(
        POETRY_VIRTUALENVS_IN_PROJECT="true",
        PYTHONDONTWRITEBYTECODE="1",
        PYTEST_XDIST_AUTO_NUM_WORKERS="2",
        PYTHONPATH=str(driver / "scripts"),
        JAX2ONNX_RUN_MAXTEXT="0",
        JAX2ONNX_MAXTEXT_SRC="",
        JAX2ONNX_MAXTEXT_MODELS="",
        JAX2ONNX_RUN_MAXDIFFUSION="0",
        JAX2ONNX_MAXDIFFUSION_SRC="",
        JAX2ONNX_MAXDIFFUSION_MODELS="",
        JAX2ONNX_MAIN_SUBJECT=str(subject),
        JAX2ONNX_MAIN_EVIDENCE=str(evidence),
        JAX2ONNX_MAIN_AUTHORITY=str(authority_path),
    )
    _require(
        not any(
            name in environment
            for name in (
                "JAX2ONNX_MAXTEXT_REF",
                "JAX2ONNX_MAXDIFFUSION_REF",
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
            )
        ),
        "foreign optional/plugin controls",
    )
    failure = None
    try:
        _guard(
            [sys.executable, *common, "--bootstrap"],
            subject,
            evidence,
            "setup",
            480,
            environment,
        )
        python = _read(evidence / "subject-python.json")
        _guard(
            [python, *common, "--workload"],
            subject,
            evidence,
            "workload",
            1200,
            environment,
        )
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        failure = str(error)
    return _close(subject, driver, evidence, authority, args.expected_driver, failure)


if __name__ == "__main__":
    raise SystemExit(main())
