# scripts/run_with_resources.py

"""Observe one Linux command interval without rewriting its argv or environment.

Sampled owned RSS includes this observer. Host/cgroup counters are shared
observations, not attribution. Unobserved GitHub service actions are outside this
interval; this receipt neither proves whole-job RSS nor the Actions time budget.
"""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from typing import Any, NamedTuple


_INTERVAL = 0.2
_MAX_GAP = 2.0
_MAX_SAMPLES_BYTES = 16 * 1024**2
_MAX_IDENTITIES = 4096
_PAGE = os.sysconf("SC_PAGE_SIZE")


class ObservationError(RuntimeError):
    """Required owned-process or host accounting is incomplete."""


class _Process(NamedTuple):
    pid: int
    parent: int
    start: int
    rss: int
    state: str


def _process(pid: int) -> _Process:
    raw = Path(f"/proc/{pid}/stat").read_text()
    fields = raw.rsplit(")", 1)[1].split()
    result = _Process(
        pid, int(fields[1]), int(fields[19]), int(fields[21]) * _PAGE, fields[0]
    )
    if result.start <= 0 or result.rss < 0:
        raise ValueError("invalid process counters")
    return result


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _live(pid: int, start: int | None = None) -> _Process | None:
    # A disappearing process is ordinary. A known/live unreadable process must
    # never become a zero-RSS contribution; retry and independently check life.
    for _ in range(2):
        try:
            item = _process(pid)
            return item if start is None or item.start == start else None
        except (OSError, ValueError, IndexError):
            if _gone(pid):
                return None
    raise ObservationError(f"live process {pid} has unreadable identity/RSS")


def _children(parent: _Process) -> list[_Process]:
    if _live(parent.pid, parent.start) is None:
        return []
    children: set[int] = set()
    try:
        for task in Path(f"/proc/{parent.pid}/task").iterdir():
            try:
                children.update(
                    int(value) for value in (task / "children").read_text().split()
                )
            except FileNotFoundError:
                if task.exists():
                    raise
    except (OSError, ValueError) as error:
        if _live(parent.pid, parent.start) is None:
            return []
        raise ObservationError("live parent child enumeration is incomplete") from error
    # Validate the SAME parent identity around enumeration and each admission.
    current = _live(parent.pid)
    if current is not None and current.start != parent.start:
        raise ObservationError("parent PID was reused during child enumeration")
    if current is None:
        return []  # Newly adopted children must be rediscovered via the owner.
    result = []
    for pid in sorted(children):
        item = _live(pid)
        fresh = _live(parent.pid)
        if fresh is not None and fresh.start != parent.start:
            raise ObservationError("parent PID was reused before child admission")
        if fresh is None:
            return []
        if item is not None and item.parent == parent.pid:
            result.append(item)
    return result


class _Owned:
    def __init__(self) -> None:
        self.owner = _process(os.getpid())
        self.identities: set[tuple[int, int]] = {(self.owner.pid, self.owner.start)}
        self.wait_outcomes: list[dict[str, Any]] = []

    def sample(self) -> list[_Process]:
        if _live(self.owner.pid, self.owner.start) is None:
            raise ObservationError("persistent observer identity changed")
        pending = list(self.identities)
        visited = set()
        active = []
        while pending:
            identity = pending.pop()
            if identity in visited:
                continue
            visited.add(identity)
            item = _live(*identity)
            if item is None:
                continue
            active.append(item)
            for child in _children(item):
                identity = (child.pid, child.start)
                self.identities.add(identity)
                if len(self.identities) > _MAX_IDENTITIES:
                    raise ObservationError("owned identity ledger exceeds its bound")
                pending.append(identity)
        return active

    def reap(self, leader_pid: int) -> None:
        for pid, start in sorted(self.identities):
            if pid in (self.owner.pid, leader_pid):
                continue
            item = _live(pid, start)
            if item is not None and item.parent == self.owner.pid and item.state == "Z":
                try:
                    reaped, status = os.waitpid(pid, os.WNOHANG)
                    if reaped == pid:
                        code = os.waitstatus_to_exitcode(status)
                        self.wait_outcomes.append(
                            {
                                "pid": pid,
                                "start": start,
                                "returncode": code,
                                "signal": -code if code < 0 else None,
                            }
                        )
                except ChildProcessError:
                    pass


def _subreaper(value: int | None = None) -> int:
    # New standalone Linux ownership mechanism. Orphans are adopted even when
    # their intermediate parent exits between samples (Linux man-pages):
    # https://man7.org/linux/man-pages/man2/PR_SET_CHILD_SUBREAPER.2const.html
    libc = ctypes.CDLL(None, use_errno=True)
    if value is not None and libc.prctl(36, ctypes.c_ulong(value), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER failed")
    result = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(result), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_GET_CHILD_SUBREAPER failed")
    return result.value


def _numbers(path: Path, *, colon: bool = False) -> dict[str, int]:
    result = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) >= 2:
            try:
                result[fields[0].rstrip(":") if colon else fields[0]] = int(fields[1])
            except ValueError:
                continue
    return result


def _kernel() -> dict[str, Any]:
    result: dict[str, Any] = {"scope": "shared-host; not owned-workload attribution"}
    for name, path, keys, colon in (
        ("memory", "/proc/meminfo", ("MemAvailable", "SwapTotal", "SwapFree"), True),
        ("vmstat", "/proc/vmstat", ("pswpin", "pswpout", "oom_kill"), False),
    ):
        try:
            values = _numbers(Path(path), colon=colon)
            result[name] = {key: values.get(key) for key in keys}
        except (OSError, ValueError):
            result[name] = {key: None for key in keys}
    if result["memory"]["MemAvailable"] is None:
        raise ObservationError("required host MemAvailable is unavailable")
    result["cgroup"] = _cgroup()
    return result


def _cgroup() -> dict[str, Any]:
    result: dict[str, Any] = {
        "scope": "shared-cgroup; ownership/exclusivity UNVERIFIED"
    }
    try:
        membership = Path("/proc/self/cgroup").read_text()
        result["membership"] = membership
        row = next(line for line in membership.splitlines() if line.startswith("0::"))
        relative = row[3:]
        if not relative.startswith("/") or ".." in Path(relative).parts:
            raise ValueError("unsupported cgroup membership")
        mounts = []
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            head, tail = line.split(" - ", 1)
            if tail.split()[0] == "cgroup2":
                fields = head.split()

                def unescape(value: str) -> str:
                    return re.sub(
                        r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value
                    )

                mounts.append((unescape(fields[3]), unescape(fields[4])))
        result["cgroup2_mounts"] = mounts
        if len(mounts) != 1 or mounts[0][0] != "/":
            raise ValueError("ambiguous or non-root cgroup2 mapping unsupported")
        mount = Path(mounts[0][1]).resolve(strict=True)
        directory = (mount / relative.lstrip("/")).resolve(strict=True)
        if not directory.is_relative_to(mount):
            raise ValueError("cgroup mapping escapes its observed mount")
        result["path"] = str(directory)
        result["status"] = "OBSERVED_SHARED"
        names = (
            "memory.current",
            "memory.peak",
            "memory.max",
            "memory.events",
            "memory.events.local",
            "memory.swap.current",
            "memory.swap.peak",
            "memory.swap.max",
            "memory.swap.events",
        )
        for name in names:
            try:
                raw = (directory / name).read_text().strip()
                result[name] = (
                    (_numbers(directory / name) or None)
                    if "events" in name
                    else "max" if raw == "max" and name.endswith(".max") else int(raw)
                )
            except (OSError, ValueError):
                result[name] = None
        if not any(result[name] is not None for name in names):
            raise ValueError("mapped cgroup has no readable counters")
    except (OSError, ValueError, IndexError, StopIteration) as error:
        result["status"] = "UNAVAILABLE"
        result["unavailable_reason"] = str(error)
    return result


def _delta(first: Any, last: Any) -> dict[str, Any]:
    if type(first) is not int or type(last) is not int:
        return {"status": "UNAVAILABLE", "value": None}
    if last < first:
        return {"status": "RESET", "value": None}
    return {"status": "OBSERVED", "value": last - first}


def _cgroup_deltas(first: dict[str, Any], last: dict[str, Any]) -> dict[str, Any]:
    if (
        first.get("status") != "OBSERVED_SHARED"
        or last.get("status") != "OBSERVED_SHARED"
        or first.get("path") != last.get("path")
        or first.get("membership") != last.get("membership")
    ):
        return {
            "status": "UNAVAILABLE_OR_MEMBERSHIP_CHANGED",
            "scope": "shared; not owned attribution",
        }
    result: dict[str, Any] = {
        "status": "OBSERVED_SHARED",
        "scope": "shared; not owned attribution",
    }
    for name in ("memory.events", "memory.events.local", "memory.swap.events"):
        before, after = first.get(name), last.get(name)
        result[name] = (
            {
                key: _delta(before.get(key), after.get(key))
                for key in set(before) | set(after)
            }
            if type(before) is dict and type(after) is dict
            else None
        )
    result["swap_growth"] = (
        last["memory.swap.current"] > first["memory.swap.current"]
        if type(first.get("memory.swap.current")) is int
        and type(last.get("memory.swap.current")) is int
        else None
    )
    return result


def _growth(before: dict[str, Any], current: dict[str, Any]) -> list[str]:
    errors = []
    for name in ("pswpin", "pswpout", "oom_kill"):
        first, last = before["vmstat"][name], current["vmstat"][name]
        if first is not None and last is not None:
            if last > first:
                errors.append(f"shared-host {name} growth; cause UNVERIFIED")
    first, last = before["memory"]["SwapFree"], current["memory"]["SwapFree"]
    if first is not None and last is not None and last < first:
        errors.append("shared-host SwapFree decreased; cause UNVERIFIED")
    cgroups = _cgroup_deltas(before.get("cgroup", {}), current.get("cgroup", {}))
    if cgroups.get("swap_growth"):
        errors.append("shared-cgroup swap growth; cause UNVERIFIED")
    for field in ("memory.events", "memory.events.local"):
        for name, value in (cgroups.get(field) or {}).items():
            if (
                name.startswith("oom")
                and value["value"] is not None
                and value["value"] > 0
            ):
                errors.append("shared-cgroup OOM growth; cause UNVERIFIED")
    return errors


def _remaining(tree: _Owned) -> list[_Process]:
    return [
        item
        for item in tree.sample()
        if item.pid != tree.owner.pid and item.state not in {"Z", "X"}
    ]


def _cleanup(
    tree: _Owned,
    leader: subprocess.Popen[Any],
    errors: list[str],
    capture: Any = None,
    signals: Any = None,
) -> list[int]:
    sent = set()
    for number, grace in ((signal.SIGTERM, 5.0), (signal.SIGKILL, 2.0)):
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            try:
                if capture is None:
                    living = _remaining(tree)
                else:
                    active, breaches = capture("cleanup")
                    errors.extend(value for value in breaches if value not in errors)
                    living = [
                        item
                        for item in active
                        if item.pid != tree.owner.pid and item.state not in {"Z", "X"}
                    ]
                tree.reap(leader.pid)
            except (ObservationError, OSError) as error:
                errors.append(str(error))
                # Only signal identities that can still be independently read.
                living = []
                for pid, start in tree.identities:
                    if pid == tree.owner.pid:
                        continue
                    try:
                        item = _live(pid, start)
                        if item is not None:
                            living.append(item)
                    except ObservationError:
                        pass
            pending_leader = leader.poll() is None
            if pending_leader and not any(item.pid == leader.pid for item in living):
                # An unreaped Popen child cannot have its PID reused. This
                # handle remains authoritative if first discovery/stat failed.
                identity = (leader.pid, "unreaped-Popen", number)
                if identity not in sent:
                    try:
                        leader.send_signal(number)
                        sent.add(identity)
                        if signals is not None:
                            signals.append(
                                {
                                    "pid": leader.pid,
                                    "start": None,
                                    "ownership": "unreaped-Popen-handle",
                                    "signal": number,
                                    "monotonic": time.monotonic(),
                                }
                            )
                    except OSError as error:
                        errors.append(str(error))
            if not living and not pending_leader:
                break
            for item in sorted(living, key=lambda value: value.start, reverse=True):
                try:
                    identity = (item.pid, item.start, number)
                    if identity not in sent and _live(item.pid, item.start) is not None:
                        os.kill(item.pid, number)
                        sent.add(identity)
                        if signals is not None:
                            signals.append(
                                {
                                    "pid": item.pid,
                                    "start": item.start,
                                    "signal": number,
                                    "monotonic": time.monotonic(),
                                }
                            )
                except ProcessLookupError:
                    pass
                except (ObservationError, PermissionError) as error:
                    errors.append(str(error))
            time.sleep(_INTERVAL)
    leader.poll()  # Popen alone reaps the main child.
    try:
        tree.reap(leader.pid)
        return sorted(
            {item.pid for item in _remaining(tree)}
            | ({leader.pid} if leader.poll() is None else set())
        )
    except (ObservationError, OSError) as error:
        errors.append(str(error))
        return sorted(
            {
                pid
                for pid, _ in tree.identities
                if pid != tree.owner.pid and not _gone(pid)
            }
            | ({leader.pid} if leader.poll() is None else set())
        )


def _artifacts(output: Path, summary: dict[str, Any]) -> bytes:
    files = [
        path
        for path in output.rglob("*")
        if path.is_file() and not path.is_symlink() and path.name != "summary.json"
    ]
    size = sum(path.stat().st_size for path in files)
    summary["observer_artifacts"] = {
        "files": len(files) + 1,
        "bytes": size,
        "scope": "observer output only; child outputs elsewhere UNMEASURED",
    }
    for _ in range(8):
        payload = (json.dumps(summary, indent=2) + "\n").encode()
        total = size + len(payload)
        if summary["observer_artifacts"]["bytes"] == total:
            return payload
        summary["observer_artifacts"]["bytes"] = total
    raise ObservationError("artifact accounting did not converge")


def _commit_summary(
    output: Path,
    summary: dict[str, Any],
    requested: list[int],
    result: int,
    handlers: dict[Any, Any],
) -> tuple[int, int | None]:
    # Keep handlers installed through receipt construction. Pending termination
    # is consumed and recorded while blocked, rather than silently producing a
    # positive receipt after a late observed signal.
    watched = set(handlers)
    prior_mask = signal.pthread_sigmask(signal.SIG_BLOCK, watched)
    try:
        while True:
            pending = signal.sigpending() & watched
            while pending:
                number = signal.sigwait(pending)
                if not requested:
                    requested.append(number)
                pending = signal.sigpending() & watched
            if requested:
                summary["status"] = "INCOMPLETE"
                summary["qualification"] = "FAIL_OR_INCOMPLETE"
                if "observer interrupted" not in summary["errors"]:
                    summary["errors"].append("observer interrupted")
                result = 125
            summary["observer_signal"] = requested[0] if requested else None
            summary["guard_result_code"] = result
            summary["planned_wrapper_returncode"] = (
                -requested[0] if requested else result
            )
            summary["wrapper_returncode_scope"] = (
                "planned at receipt commit; caller must retain actual exit/signal"
            )
            payload = _artifacts(output, summary)
            if signal.sigpending() & watched:
                continue
            (output / "summary.json").write_bytes(payload)
            if not (signal.sigpending() & watched):
                break
    finally:
        for number, handler in handlers.items():
            signal.signal(number, handler)
        signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)
    return result, requested[0] if requested else None


def _run(args: Any, output: Path) -> tuple[int, int | None]:
    started = time.monotonic()
    summary: dict[str, Any] = {
        "scope": "sampled controlled-command interval only; external Actions resource intervals UNOBSERVED",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "argv": args.command,
        "cwd": str(Path.cwd()),
        "sample_interval_seconds": _INTERVAL,
        "limits": {
            "timeout_seconds": args.timeout_seconds,
            "rss_mib": args.max_rss_mib,
            "host_available_mib": args.min_available_mib,
            "disk_free_mib": args.min_disk_free_mib,
        },
        "status": "INCOMPLETE",
        "child_returncode": None,
        "errors": [],
        "sample_count": 0,
        "max_sample_gap_seconds": 0.0,
        "peak_owned_rss_bytes": 0,
        "peak_owned_rss_sample": None,
        "minimum_host_available_bytes": None,
        "disk_free_minima": {},
        "cgroup_before": _cgroup(),
        "cgroup_after": None,
    }
    if getattr(args, "locked312_standard_binding", None) is not None:
        summary["standard_execution_profile"] = args.locked312_standard_binding
        summary["limits"]["initial_host_available_mib"] = 12288
    tree = None
    leader = None
    prior_subreaper = None
    requested_signal: list[int] = []
    handlers = {}
    baseline = None
    last = None
    samples_bytes = 0
    result = 125
    sampler = None
    stream = None
    try:
        tree = _Owned()
        summary["observer_identity"] = {
            "pid": tree.owner.pid,
            "start": tree.owner.start,
        }
        if _children(tree.owner):
            raise ObservationError(
                "standalone observer has preexisting child processes"
            )
        prior_subreaper = _subreaper()
        if _subreaper(1) != 1:
            raise ObservationError("Linux subreaper admission failed")
        baseline = _kernel()
        summary["kernel_before"] = baseline
        if getattr(args, "locked312_standard_binding", None):
            summary["initial_host_available_bytes"] = (
                baseline["memory"]["MemAvailable"] * 1024
            )
            if summary["initial_host_available_bytes"] < 12288 * 1024**2:
                raise ObservationError("locked312 initial 12 GiB availability floor")

        def interrupted(signum: int, frame: Any) -> None:
            if not requested_signal:
                requested_signal.append(signum)

        for number in (signal.SIGINT, signal.SIGTERM):
            handlers[number] = signal.signal(number, interrupted)
        disks = list(
            dict.fromkeys([Path.cwd().resolve(), output.parent, *args.disk_path])
        )
        stream = (output / "samples.jsonl").open("x")

        def capture(stage: str) -> tuple[list[_Process], list[str]]:
            nonlocal last, samples_bytes
            now = time.monotonic()
            gap_breached = False
            if last is not None:
                gap = now - last
                summary["max_sample_gap_seconds"] = max(
                    summary["max_sample_gap_seconds"], gap
                )
                if gap > _MAX_GAP:
                    gap_breached = True
            last = now
            active = tree.sample()
            if leader is not None:
                # Discover direct adopted children AFTER leader exit. Never
                # decide completion from a pre-poll snapshot.
                if leader.poll() is not None:
                    active = tree.sample()
                tree.reap(leader.pid)
                active = tree.sample()
            current = _kernel()
            summary["kernel_after"] = current
            rss = sum(item.rss for item in active)
            available = current["memory"]["MemAvailable"] * 1024
            free = {
                str(path): os.statvfs(path).f_bavail * os.statvfs(path).f_frsize
                for path in disks
            }
            minimum = summary["minimum_host_available_bytes"]
            summary["minimum_host_available_bytes"] = (
                available if minimum is None else min(minimum, available)
            )
            for name, value in free.items():
                summary["disk_free_minima"][name] = min(
                    summary["disk_free_minima"].get(name, value), value
                )
            capture_seconds = time.monotonic() - now
            summary["max_sample_gap_seconds"] = max(
                summary["max_sample_gap_seconds"], capture_seconds
            )
            gap_breached = gap_breached or capture_seconds > _MAX_GAP
            sample = {
                "stage": stage,
                "elapsed_seconds": now - started,
                "owned_rss_bytes": rss,
                "owned_processes": len(active),
                "owned_process_vector": [item._asdict() for item in active],
                "host": current,
                "disk_free_bytes": free,
                "capture_seconds": capture_seconds,
            }
            if (
                summary["peak_owned_rss_sample"] is None
                or rss > summary["peak_owned_rss_bytes"]
            ):
                summary["peak_owned_rss_bytes"] = rss
                summary["peak_owned_rss_sample"] = {
                    name: sample[name]
                    for name in (
                        "stage",
                        "elapsed_seconds",
                        "owned_rss_bytes",
                        "owned_processes",
                        "owned_process_vector",
                    )
                }
            line = json.dumps(sample) + "\n"
            samples_bytes += len(line.encode())
            if samples_bytes > _MAX_SAMPLES_BYTES:
                raise ObservationError(
                    "aggregate sample evidence exceeds its 16 MiB bound"
                )
            stream.write(line)
            stream.flush()
            summary["sample_count"] += 1
            breaches = _growth(baseline, current)
            if gap_breached:
                breaches.append("required process sampling gap exceeded 2 seconds")
            if any(record["signal"] is not None for record in tree.wait_outcomes):
                breaches.append(
                    "adopted owned-process signal death observed; OOM cause UNVERIFIED"
                )
            if rss > args.max_rss_mib * 1024**2:
                breaches.append("sampled aggregate owned RSS limit")
            if available < args.min_available_mib * 1024**2:
                breaches.append("host available memory floor")
            if any(value < args.min_disk_free_mib * 1024**2 for value in free.values()):
                breaches.append("declared filesystem free-space floor")
            if now - started > args.timeout_seconds:
                breaches.append("controlled command interval deadline")
            if requested_signal:
                breaches.append("observer interrupted")
            return active, breaches

        sampler = capture
        while True:
            now = time.monotonic()
            active, breaches = capture("command")
            if breaches:
                summary["errors"].extend(breaches)
                break
            if leader is None:
                leader = subprocess.Popen(
                    args.command
                )  # Exact env/cwd/streams inherited.
                continue
            if leader.returncode is not None and not _remaining(tree):
                if _children(tree.owner):
                    continue
                summary["status"] = "COMPLETE"
                result = leader.returncode
                break
            time.sleep(max(0.0, _INTERVAL - (time.monotonic() - now)))
    except (OSError, ValueError, ObservationError) as error:
        summary["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        if tree is not None and leader is not None:
            summary["remaining_descendants"] = _cleanup(
                tree,
                leader,
                summary["errors"],
                sampler,
                summary.setdefault("signals_sent", []),
            )
            summary["child_returncode"] = leader.poll()
            if summary["remaining_descendants"]:
                summary["status"] = "INCOMPLETE"
                result = 125
        if tree is not None:
            summary["owned_pid_start_ledger"] = sorted(tree.identities)
        summary["cgroup_after"] = _cgroup()
        summary["cgroup_deltas"] = _cgroup_deltas(
            summary["cgroup_before"], summary["cgroup_after"]
        )
        try:
            summary["kernel_after"] = _kernel()
            if baseline is not None:
                summary["errors"].extend(_growth(baseline, summary["kernel_after"]))
                summary["host_vmstat_deltas"] = {
                    name: _delta(
                        baseline["vmstat"][name],
                        summary["kernel_after"]["vmstat"][name],
                    )
                    for name in baseline["vmstat"]
                }
        except ObservationError as error:
            summary["errors"].append(str(error))
        try:
            summary["cleanup_confirmed"] = (
                tree is not None
                and not _children(tree.owner)
                and not summary.get("remaining_descendants")
                and (leader is None or leader.returncode is not None)
            )
        except (ObservationError, OSError) as error:
            summary["cleanup_confirmed"] = False
            summary["errors"].append(str(error))
        if not summary["cleanup_confirmed"]:
            summary["errors"].append("owned cleanup could not be confirmed")
        if prior_subreaper is not None and summary["cleanup_confirmed"]:
            try:
                _subreaper(prior_subreaper)
            except OSError as error:
                summary["errors"].append(str(error))
                summary["status"] = "INCOMPLETE"
                result = 125
        if sampler is not None:
            try:
                _, breaches = sampler("terminal")
                summary["errors"].extend(
                    value for value in breaches if value not in summary["errors"]
                )
            except (ObservationError, OSError, ValueError) as error:
                summary["errors"].append(str(error))
        if stream is not None:
            stream.close()
        if summary["errors"]:
            summary["status"] = "INCOMPLETE"
            result = 125
        summary["elapsed_seconds"] = time.monotonic() - started
        summary["terminal_sample_gap_seconds"] = (
            None if last is None else time.monotonic() - last
        )
        summary["adopted_wait_outcomes"] = [] if tree is None else tree.wait_outcomes
        summary["wait_status_scope"] = (
            "observer-adopted children only; other descendant wait statuses UNAVAILABLE"
        )
        summary["command_status"] = (
            "PASS" if summary["child_returncode"] == 0 else "FAIL_OR_NOT_STARTED"
        )
        summary["qualification"] = (
            "PASS"
            if summary["status"] == "COMPLETE" and summary["child_returncode"] == 0
            else "FAIL_OR_INCOMPLETE"
        )
        result, observed_signal = _commit_summary(
            output, summary, requested_signal, result, handlers
        )
    return result, observed_signal


def _locked312_binding(profile: Path) -> dict[str, str]:
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location(
        "_locked312_execution", Path(__file__).with_name("_ci_standard_execution.py")
    )
    if spec is None or spec.loader is None:
        raise ValueError("locked312 execution policy is unavailable")
    policy = module_from_spec(spec)
    spec.loader.exec_module(policy)
    return policy.locked312_execution(profile)


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("--max-rss-mib", type=int, default=4096)
    parser.add_argument("--locked312-standard-profile", type=Path)
    parser.add_argument("--min-available-mib", type=int, default=2048)
    parser.add_argument("--min-disk-free-mib", type=int, default=4096)
    parser.add_argument("--disk-path", type=Path, action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    locked312 = args.locked312_standard_profile is not None
    if locked312:
        try:
            args.locked312_standard_binding = _locked312_binding(
                args.locked312_standard_profile
            )
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            parser.error(str(error))
    if (
        not args.command
        or not 0 < args.timeout_seconds <= (1100 if locked312 else 1200)
        or not (args.max_rss_mib == 5120 if locked312 else 0 < args.max_rss_mib <= 4096)
        or args.min_available_mib < (8192 if locked312 else 2048)
        or args.min_disk_free_mib < 4096
    ):
        parser.error(
            "command and bounded STANDARD 1200s/4GiB/2GiB-headroom/4GiB-disk "
            "or source-bound locked312 1100s/5GiB/8GiB-headroom/4GiB-disk limits required"
        )
    if os.path.lexists(args.output_dir):
        parser.error("resource evidence destination must be fresh")
    output = args.output_dir.resolve(strict=False)
    if output.is_relative_to(Path.cwd().resolve()):
        parser.error("resource evidence must be external to the command checkout")
    output.mkdir(parents=False, exist_ok=False)
    args.disk_path = [path.resolve(strict=True) for path in args.disk_path]
    code, interrupted = _run(args, output)
    number = interrupted if interrupted is not None else -code if code < 0 else None
    if number is not None:
        if number not in (signal.SIGKILL, signal.SIGSTOP):
            signal.signal(number, signal.SIG_DFL)
        os.kill(os.getpid(), number)
    return code


if __name__ == "__main__":
    raise SystemExit(_main())
