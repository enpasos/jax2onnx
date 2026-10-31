# tests/extra_tests/converter/source_original22_phase_observation.py

"""Opt-in, synchronous, value-free phase observations owned by one test module."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time


OUTPUT_ENV = "JAX2ONNX_ORIGINAL22_PHASE_OUTPUT"
_FILE: tuple[int, str, int | None] | None = None
_LABELS = frozenset(
    {
        "factory",
        "initial-state-checks",
        "preparation",
        "native-references",
        "prepared-native",
        "source-export",
        "artifact-save",
        "artifact-reload",
        "schema-check",
        "artifact-audit",
        "port-source-checks",
        "serialization",
        "cpu-session",
        "cpu-run",
        "runtime-comparisons",
        "runtime-source-checks",
    }
)


def _descriptor() -> int | None:
    global _FILE
    destination = os.environ.get(OUTPUT_ENV)
    if not destination:
        return None
    try:
        directory = Path(destination)
        if not directory.is_absolute():
            return None
        directory = directory.resolve(strict=True)
        repository = Path(__file__).resolve().parents[3]
        if not directory.is_dir() or directory.is_relative_to(repository):
            return None
        pid = os.getpid()
        key = (pid, str(directory))
        if _FILE is None or _FILE[:2] != key:
            previous, _FILE = _FILE, None
            if previous is not None and previous[2] is not None:
                try:
                    os.close(previous[2])
                except OSError:
                    pass
            _FILE = (*key, None)
            fd = os.open(
                directory / f"original22-phases-pid-{pid}.jsonl",
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_APPEND
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
            _FILE = (*key, fd)
        # One metadata-only descriptor per PID/output is retained until process
        # exit. No test/model/array/frame reference is stored in this registry.
        return _FILE[2]
    except Exception:
        return None


def _rss() -> dict[str, int]:
    result = {}
    with Path("/proc/self/status").open(encoding="ascii") as stream:
        for line in stream:
            name, _, value = line.partition(":")
            if name in {"VmRSS", "VmHWM", "RssAnon", "RssFile"}:
                parts = value.split()
                if len(parts) == 2 and parts[0].isdigit() and parts[1] == "kB":
                    result[name + "_bytes"] = int(parts[0]) * 1024
    return result


def _event(
    fd: int,
    event: str,
    label: str,
    fields: dict[str, int | bool],
    *,
    exception_type: str | None = None,
) -> None:
    try:
        record = {
            "event": event,
            "phase": label,
            "pid": os.getpid(),
            "monotonic_seconds": time.monotonic(),
            **fields,
            **_rss(),
        }
        if exception_type is not None:
            record["exception_type"] = exception_type
        os.write(fd, (json.dumps(record, sort_keys=True) + "\n").encode())
    except Exception:
        # A missing/failed observation never changes the ordinary test outcome.
        pass


@contextmanager
def phase(
    label: str,
    *,
    index: int,
    caller_x64: bool,
    batch: int | None = None,
    optimize: bool | None = None,
    replay: int | None = None,
    position: int | None = None,
) -> Iterator[None]:
    fd = None
    fields: dict[str, int | bool] = {}
    try:
        candidates = {
            "index": index,
            "caller_x64": caller_x64,
            "batch": batch,
            "optimize": optimize,
            "replay": replay,
            "position": position,
        }
        if (
            type(label) is str
            and label in _LABELS
            and all(
                value is None or type(value) in (int, bool)
                for value in candidates.values()
            )
        ):
            fields = {
                name: value for name, value in candidates.items() if value is not None
            }
            fd = _descriptor()
            if fd is not None:
                _event(fd, "enter", label, fields)
    except Exception:
        pass
    try:
        yield
    except BaseException as error:
        if fd is not None:
            try:
                _event(fd, "raised", label, fields, exception_type=type(error).__name__)
            except Exception:
                pass
        raise
    else:
        if fd is not None:
            _event(fd, "returned", label, fields)
