# scripts/build_corrected_jaxlib.py

"""Build or inspect the one pinned CPU jaxlib candidate; never install a wheel.

The remote build is a capacity/compilation trial. Byte provenance does not grant
private SOURCE numerical authority or qualify the original application tests.
Only Python's standard library is imported by this controller.
"""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import ctypes
from email.parser import BytesParser
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import selectors
import shutil
import signal
import stat
import struct
import subprocess
import sys
import tarfile
import time
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "build_support/jaxlib/cpu-bf16-build-recipe.json"
PATCH = ROOT / "build_support/jaxlib/xla-0.10.2-bf16-nested-conversion.patch"
JAX = "990e6a0b84138346e6a38785412f36356e0e5dc3"
XLA = "5a9e73cbd92530cac2ac36f4736a774b2412afe2"
LLVM = "35901313800ea6e6cbeb9226e51c7c4b29bfc40e"
MIB = 1024**2
GIB = 1024**3
OWNED_PROCESS_LIMIT = 4096
LIMITS = {
    "startup_memory": 12 * GIB,
    "startup_disk": 64 * GIB,
    "live_memory": 2 * GIB,
    "live_disk": 4 * GIB,
    "owned_rss": 12 * GIB,
    "timeout_seconds": 300 * 60,
    "log_bytes": 512 * MIB,
}
STAT_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_nlink",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)
PATCH_TARGETS = {
    "xla/service/cpu/ir_emitter.cc": {
        "before_sha256": "235ba806045eabc2848698e299f62dd5c8a1c6a8b64705363d3c46fef3da657f",
        "after_sha256": "09cce871b6bd7063d7a6654a5a0250a79e9ed697167d4ea3c9b50026610239d7",
    },
    "xla/service/cpu/tests/BUILD": {
        "before_sha256": "15c87fb74ee399d3bbc3d0c7781e41896d05759663c15d971f1ca31991835cf2",
        "after_sha256": "a591ea47865936565310a0dccd561bb9b6c2d4a91496f5e062dab7eb552afe55",
    },
    "xla/service/cpu/tests/nested_bf16_conversion_test.cc": {
        "before_sha256": None,
        "after_sha256": "8c912e987793833f415805b0c6ecc1732b0b980b3046d91f0ef12357a218dfe6",
    },
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def safe_parts(name):
    require(
        type(name) is str and 0 < len(name.encode()) <= 4096, "invalid relative name"
    )
    require(
        "\\" not in name and "\x00" not in name and ":" not in name,
        "unsafe relative name",
    )
    path = PurePosixPath(name)
    require(
        not path.is_absolute()
        and str(path) == name
        and all(p not in ("", ".", "..") for p in name.split("/")),
        "noncanonical relative name",
    )
    return path.parts


def file_record(path, limit=512 * MIB):
    before = path.lstat()
    require(
        stat.S_ISREG(before.st_mode) and before.st_size <= limit,
        "nonregular/oversized input",
    )
    digest = hashlib.sha256()
    count = 0
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        require(
            all(
                getattr(os.fstat(descriptor), k) == getattr(before, k)
                for k in STAT_FIELDS
            ),
            "input changed before read",
        )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            while chunk := stream.read(MIB):
                count += len(chunk)
                require(count <= limit, "input grew beyond bound")
                digest.update(chunk)
        after = path.lstat()
        require(
            count == before.st_size
            and all(getattr(after, k) == getattr(before, k) for k in STAT_FIELDS)
            and all(
                getattr(os.fstat(descriptor), k) == getattr(before, k)
                for k in STAT_FIELDS
            ),
            "input changed during read",
        )
    finally:
        os.close(descriptor)
    return {"bytes": count, "sha256": digest.hexdigest()}


def read_json(path, limit=8 * MIB):
    before = file_record(path, limit)
    raw = path.read_bytes()
    require(
        {"bytes": len(raw), "sha256": sha(raw)} == before, "JSON changed during read"
    )
    return json.loads(raw, object_pairs_hook=unique)


def write_json(path, value):
    raw = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
    require(len(raw) <= 8 * MIB, "JSON output exceeds bound")
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "wb") as stream:
        require(stream.write(raw) == len(raw), "short JSON write")
        stream.flush()
        os.fsync(stream.fileno())


def recipe_digest(recipe):
    return sha(canonical(recipe["inputs"]))


def validate_recipe(recipe, patch):
    require(
        set(recipe) == {"schema", "inputs", "derived"}
        and recipe["schema"] == "corrected-jaxlib-cpu-bf16-recipe-v1",
        "recipe schema mismatch",
    )
    inputs = recipe["inputs"]
    require(
        inputs["jax"]["commit"] == JAX
        and inputs["xla"]["commit"] == XLA
        and inputs["llvm_commit"] == LLVM,
        "wrong pinned revision",
    )
    require(
        inputs["jax"]["sha256"]
        == "df668a015f8b1cd5ecddcb45cdb7a798580573bcff0eb452422358629403488c"
        and inputs["jax"]["bytes"] == 20056129
        and inputs["xla"]["sha256"]
        == "08a52210a04cd68d38f6201d56273d04ac0f8b4e0da9f72677c74a48cc637422"
        and inputs["xla"]["bytes"] == 28172460,
        "wrong source archive",
    )
    require(
        inputs["jax"]["url"] == f"https://codeload.github.com/jax-ml/jax/tar.gz/{JAX}"
        and inputs["xla"]["url"]
        == f"https://codeload.github.com/openxla/xla/tar.gz/{XLA}",
        "wrong source locator",
    )
    require(
        inputs["bazel"]["version"] == "7.7.0"
        and inputs["bazel"]["sha256"]
        == "fe7e799cbc9140f986b063e06800a3d4c790525075c877d00a7112669824acbf",
        "wrong Bazel",
    )
    require(
        inputs["abis"] == {"3.11": "cp311", "3.12": "cp312"}
        and inputs["platform"] == "manylinux_2_27_x86_64",
        "wrong ABI/platform",
    )
    require(inputs["resources"] == LIMITS, "resource policy mismatch")
    require(
        type(inputs.get("build_jobs")) is int and inputs["build_jobs"] == 3,
        "build scheduling policy mismatch",
    )
    require(
        sha(patch)
        == inputs["patch"]["sha256"]
        == "0068ea6413ef79971b026d121838236b8622f093f2ba508ada06611fdb71aad0",
        "wrong compiler patch",
    )
    require(
        inputs["patch"]["targets"] == PATCH_TARGETS,
        "wrong compiler patch target contract",
    )
    require(
        inputs["hermetic_toolchain"]["rules_ml_toolchain_commit"]
        == "cad1047facbac4fb3c1124da68bf2cb36c7eb9ac"
        and inputs["hermetic_toolchain"]["required_archive_alternatives"]
        == [
            ["40963e4bc262dfa9a43146f610140af0068b023ace8f3c50f1705a7b50de0830"],
            [
                "54ec30358afcc9fb8aa74307db3046f5187f9fb89fb37064cdde906e062ebf36",
                "01b8e95e34e7d0117edd085577529b375ec422130ed212d2911727545314e6c2",
            ],
            [
                "9a06397f9dc4fe2237d126f964f1a3c942ce0036ba51976ea0d7fce0bc3e2fb7",
                "114a7d09078c6a2d29b506033e07442346099847ce6107c78ec3da86388ea4a5",
            ],
        ],
        "wrong hermetic toolchain contract",
    )
    require(
        inputs["build_config"] == "ci_linux_x86_64"
        and inputs["target_cpu_features"] == "release"
        and inputs["public_bazel_cache"]
        == "https://storage.googleapis.com/jax-bazel-cache/",
        "build policy mismatch",
    )
    digest = recipe_digest(recipe)
    require(
        recipe["derived"]
        == {"recipe_sha256": digest, "version": "0.10.2+jax2onnx.bf16." + digest[:16]},
        "derived recipe/version mismatch",
    )
    return digest


def cache_key(recipe, python_version):
    require(python_version in recipe["inputs"]["abis"], "unsupported Python ABI")
    return (
        "corrected-jaxlib-linux-x86_64-"
        + recipe["inputs"]["abis"][python_version]
        + "-"
        + recipe_digest(recipe)
    )


def resources():
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, _, value = line.partition(":")
        if name == "MemAvailable":
            mem[name] = int(value.split()[0]) * 1024
    require("MemAvailable" in mem, "available memory unobserved")
    return mem["MemAvailable"]


def capacity(path, startup=False):
    available, disk = resources(), shutil.disk_usage(path).free
    require(
        available >= LIMITS["startup_memory" if startup else "live_memory"],
        "insufficient available memory",
    )
    require(
        disk >= LIMITS["startup_disk" if startup else "live_disk"],
        "insufficient free disk",
    )
    return {"available_memory": available, "free_disk": disk}


class OwnedBuildProcesses:
    """One command's Linux descendant identities, including adopted orphans."""

    def __init__(self, journal_path):
        require(
            sys.platform == "linux"
            and hasattr(os, "pidfd_open")
            and hasattr(signal, "pidfd_send_signal"),
            "exact Linux process interfaces unavailable",
        )
        self.pid = os.getpid()
        self.known = {}
        self.leader = None
        self.closed = False
        self.retained_count = 0
        self.retired_count = 0
        self.journal_bytes = 0
        self.journal_digest = hashlib.sha256()
        self.journal_error = None
        self.unrecorded_identity_events = 0
        self.journal_path = journal_path
        self.journal = journal_path.open("xb")
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.libc.prctl.argtypes = [
            ctypes.c_int,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
        ]
        self.libc.prctl.restype = ctypes.c_int
        self.original_subreaper = self.subreaper()
        rows = self.snapshot()
        require(
            not any(r["ppid"] == self.pid for r in rows.values()),
            "preexisting direct children",
        )
        self.start = rows[self.pid]["start"]
        descriptor = os.pidfd_open(self.pid, 0)
        try:
            signal.pidfd_send_signal(descriptor, 0, None, 0)
        finally:
            os.close(descriptor)
        self.set_subreaper(1)

    def event(self, kind, row):
        if self.journal_error is not None:
            self.unrecorded_identity_events += 1
            return
        raw = (json.dumps({"event": kind, **row}, sort_keys=True) + "\n").encode()
        try:
            require(
                self.journal_bytes + len(raw) <= 64 * MIB,
                "complete owned-identity journal bound exhausted",
            )
            require(
                shutil.disk_usage(self.journal_path.parent).free
                >= LIMITS["live_disk"] + len(raw),
                "owned-identity journal disk floor",
            )
            require(
                self.journal.write(raw) == len(raw),
                "short owned-identity journal write",
            )
            self.journal_bytes += len(raw)
            self.journal_digest.update(raw)
        except BaseException as error:
            # A failed journal makes the trial unavailable, but must never stop
            # discovery or exact signalling during resource cleanup.
            self.journal_error = {
                "type": type(error).__name__,
                "message": str(error)[:512],
            }
            self.unrecorded_identity_events += 1

    def subreaper(self):
        value = ctypes.c_int()
        require(
            self.libc.prctl(37, ctypes.addressof(value), 0, 0, 0) == 0,
            "PR_GET_CHILD_SUBREAPER failed",
        )
        return value.value

    def set_subreaper(self, value):
        require(
            value in (0, 1)
            and self.libc.prctl(36, value, 0, 0, 0) == 0
            and self.subreaper() == value,
            "PR_SET_CHILD_SUBREAPER failed",
        )

    @staticmethod
    def process(pid):
        try:
            raw = (Path("/proc") / str(pid) / "stat").read_text()
        except (FileNotFoundError, ProcessLookupError):
            return None
        fields = raw[raw.rfind(")") + 2 :].split()
        require(len(fields) >= 22, "incomplete process identity")
        return {
            "pid": pid,
            "start": int(fields[19]),
            "ppid": int(fields[1]),
            "pgrp": int(fields[2]),
            "session": int(fields[3]),
            "state": fields[0],
            "rss": max(0, int(fields[21])) * os.sysconf("SC_PAGE_SIZE"),
        }

    @classmethod
    def snapshot(cls):
        rows = {}
        for path in Path("/proc").iterdir():
            if path.name.isdigit():
                value = cls.process(int(path.name))
                if value is not None:
                    rows[value["pid"]] = value
        return rows

    def retain(self, row):
        identity = (row["pid"], row["start"])
        if identity in self.known:
            return
        before = self.process(row["pid"])
        if before is None or before["start"] != row["start"]:
            return
        try:
            descriptor = os.pidfd_open(row["pid"], 0)
        except ProcessLookupError:
            return
        after = self.process(row["pid"])
        if after is None or after["start"] != row["start"]:
            os.close(descriptor)
            return
        self.known[identity] = descriptor
        self.retained_count += 1
        self.event("retained", row)

    def bind(self, pid):
        self.leader = pid
        row = self.process(pid)
        require(
            row is not None and row["ppid"] == self.pid and row["start"] >= self.start,
            "Popen identity unavailable",
        )
        self.retain(row)
        self.discover()

    def discover(self, cleanup=False):
        require(not self.closed, "closed ownership catalogue")
        rows = self.snapshot()
        require(rows[self.pid]["start"] == self.start, "controller identity changed")
        while True:
            count = len(self.known)
            parents = {
                pid: start
                for pid, start in self.known
                if pid in rows and rows[pid]["start"] == start
            }
            for row in rows.values():
                if (row["ppid"] == self.pid and row["start"] >= self.start) or (
                    row["ppid"] in parents and parents[row["ppid"]] <= row["start"]
                ):
                    self.retain(row)
            if len(self.known) == count:
                break
        for identity in list(self.known):
            row = rows.get(identity[0])
            if row is None or row["start"] != identity[1]:
                os.close(self.known.pop(identity))
                self.retired_count += 1
                self.event("retired", {"pid": identity[0], "start": identity[1]})
        if not cleanup:
            require(
                len(self.known) <= OWNED_PROCESS_LIMIT,
                "concurrent owned process identity bound exceeded",
            )
        return rows

    def live(self, rows):
        return [
            row
            for identity in self.known
            if (row := rows.get(identity[0])) is not None
            and row["start"] == identity[1]
            and row["state"] not in ("Z", "X")
        ]

    def signal_owned(self, sig, cleanup=False):
        rows = self.discover(cleanup=cleanup)
        for row in self.live(rows):
            try:
                signal.pidfd_send_signal(
                    self.known[(row["pid"], row["start"])], sig, None, 0
                )
            except ProcessLookupError:
                pass

    def reap_adopted(self, cleanup=False):
        rows = self.discover(cleanup=cleanup)
        for pid, start in self.known:
            row = rows.get(pid)
            if (
                pid != self.leader
                and row is not None
                and row["start"] == start
                and row["ppid"] == self.pid
            ):
                try:
                    os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    pass

    def close(self, process=None):
        require(not self.closed, "ownership already closed")
        for sig, duration in ((signal.SIGTERM, 2), (signal.SIGKILL, 5)):
            deadline = time.monotonic() + duration
            while True:
                self.signal_owned(sig, cleanup=True)
                if process is not None:
                    process.poll()  # Popen retains ownership of its own wait status.
                self.reap_adopted(cleanup=True)
                rows = self.discover(cleanup=True)
                if not self.known and not any(
                    r["ppid"] == self.pid for r in rows.values()
                ):
                    if process is not None:
                        process.wait(timeout=1)
                    require(
                        self.retired_count == self.retained_count,
                        "owned identity retirement incomplete",
                    )
                    self.set_subreaper(self.original_subreaper)
                    try:
                        self.journal.flush()
                        os.fsync(self.journal.fileno())
                    except BaseException as error:
                        self.journal_error = {
                            "type": type(error).__name__,
                            "message": str(error)[:512],
                        }
                    finally:
                        self.journal.close()
                    self.closed = True
                    return {
                        "closed_identity_count": self.retired_count,
                        "identity_journal_file": self.journal_path.name,
                        "identity_journal_sha256": self.journal_digest.hexdigest(),
                        "identity_journal_error": self.journal_error,
                        "unrecorded_identity_events": self.unrecorded_identity_events,
                        "remaining_live_owned_identities": [],
                        "remaining_adopted_children": [],
                        "subreaper_restored": True,
                    }
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
        # Do not restore the subreaper while descendants remain unclosed.
        raise ValueError("owned descendant cleanup incomplete")


class Trial:
    """One fresh owned descendant closure at a time; Bazel is always --batch."""

    def __init__(self, out):
        self.out = out
        self.started = time.monotonic()
        self.commands = []
        self.peak_rss = 0
        self.minimum_memory = 1 << 63
        self.minimum_disk = 1 << 63
        self.telemetry = (out / "resources.jsonl").open("x")
        self.samples = 0
        self.telemetry_bytes = 0
        self.last_sample = -2.0

    def sample(self, ownership=None):
        elapsed = time.monotonic() - self.started
        require(elapsed <= LIMITS["timeout_seconds"], "trial timeout")
        if elapsed - self.last_sample < 1:
            return
        self.last_sample = elapsed
        current = capacity(self.out)
        controller = OwnedBuildProcesses.process(os.getpid())
        require(controller is not None, "controller RSS unavailable")
        members = [controller]
        if ownership is not None:
            members += ownership.live(ownership.discover())
            require(
                ownership.journal_error is None,
                "complete owned-identity journal unavailable",
            )
        rss = sum(row["rss"] for row in members)
        require(len(members) <= 4096, "owned process count exceeds bound")
        self.peak_rss = max(self.peak_rss, rss)
        self.minimum_memory = min(self.minimum_memory, current["available_memory"])
        self.minimum_disk = min(self.minimum_disk, current["free_disk"])
        require(rss <= LIMITS["owned_rss"], "aggregate owned RSS exceeded")
        row = {
            "elapsed_seconds": round(time.monotonic() - self.started, 3),
            "owned_rss": rss,
            "owned_identities": [
                {
                    "pid": r["pid"],
                    "start": r["start"],
                    "pgrp": r["pgrp"],
                    "session": r["session"],
                }
                for r in members
            ],
            **current,
        }
        self.samples += 1
        require(self.samples <= 40000, "telemetry sample bound exceeded")
        raw = json.dumps(row) + "\n"
        self.telemetry_bytes += len(raw.encode())
        require(
            self.telemetry_bytes <= 64 * MIB,
            "complete resource telemetry bound exhausted",
        )
        self.telemetry.write(raw)
        self.telemetry.flush()

    def run(self, argv, cwd, label):
        require(
            all(type(a) is str and "\x00" not in a for a in argv),
            "invalid process argv",
        )
        self.sample()
        record = {"argv": argv, "cwd": str(cwd), "label": label, "returncode": None}
        self.commands.append(record)
        environment = dict(os.environ)
        # Build-only subprocess options; never application runtime flags.
        environment["PYTHONNOUSERSITE"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        ownership = OwnedBuildProcesses(self.out / (label + "-owned-identities.jsonl"))
        process = None
        selector = None
        primary_error = None
        count = 0
        try:
            process = subprocess.Popen(
                argv,
                cwd=cwd,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            ownership.bind(process.pid)
            require(
                ownership.journal_error is None,
                "complete owned-identity journal unavailable",
            )
            require(process.stdout is not None, "owned process pipe unavailable")
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            with (self.out / (label + ".log")).open("xb") as log:
                while selector.get_map() or process.poll() is None:
                    self.sample(ownership)
                    for key, _ in selector.select(timeout=1):
                        chunk = os.read(key.fileobj.fileno(), 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        count += len(chunk)
                        require(
                            count <= LIMITS["log_bytes"],
                            "complete build log bound exhausted",
                        )
                        require(
                            log.write(chunk) == len(chunk), "short process log write"
                        )
                    if process.poll() is not None and not selector.get_map():
                        break
                process.wait(timeout=5)
                log.flush()
                os.fsync(log.fileno())
            require(process.returncode == 0, "owned command failed: " + label)
        except BaseException as error:
            primary_error = error
            record["command_error"] = {
                "type": type(error).__name__,
                "message": str(error)[:1024],
            }
            raise
        finally:
            cleanup_error = None
            try:
                record["ownership_closure"] = ownership.close(process)
                require(
                    record["ownership_closure"]["identity_journal_error"] is None,
                    "complete owned-identity journal unavailable after cleanup",
                )
            except BaseException as error:
                cleanup_error = error
                record["cleanup_error"] = {
                    "type": type(error).__name__,
                    "message": str(error)[:1024],
                }
            finally:
                record["returncode"] = (
                    process.returncode if process is not None else None
                )
                record["log_bytes"] = count
                if selector is not None:
                    selector.close()
                if process is not None and process.stdout is not None:
                    process.stdout.close()
            if cleanup_error is not None and primary_error is None:
                raise cleanup_error

    def download(self, item, name):
        path = self.out / name
        digest = hashlib.sha256()
        count = 0
        parts = urllib.parse.urlsplit(item["url"])
        require(
            parts.scheme == "https"
            and parts.hostname in {"codeload.github.com", "releases.bazel.build"}
            and not parts.username
            and not parts.password,
            "nonprimary download",
        )
        with (
            urllib.request.urlopen(item["url"], timeout=30) as source,
            path.open("xb") as target,
        ):
            require(source.geturl() == item["url"], "unexpected download redirect")
            while chunk := source.read(MIB):
                self.sample()
                count += len(chunk)
                require(
                    count <= item.get("bytes", 128 * MIB),
                    "download size bound exceeded",
                )
                target.write(chunk)
                digest.update(chunk)
            target.flush()
            os.fsync(target.fileno())
        require(
            digest.hexdigest() == item["sha256"] and count == item.get("bytes", count),
            "download hash/size mismatch",
        )
        return path


def unpack(archive, root, prefix, trial):
    """Extract one hash-pinned source tar, then create confined symlinks last."""
    root.mkdir()
    names = set()
    links = []
    records = {}
    total = 0
    with tarfile.open(archive, "r:gz") as source:
        for entry in source:
            trial.sample()
            parts = safe_parts(entry.name.rstrip("/"))
            require(
                parts[0] == prefix and entry.name not in names,
                "source archive root/name mismatch",
            )
            names.add(entry.name)
            require(len(names) <= 100000, "source archive entry bound exceeded")
            if len(parts) == 1:
                require(entry.isdir(), "source archive root is not directory")
                continue
            relative = "/".join(parts[1:])
            target = root.joinpath(*parts[1:])
            target.parent.mkdir(parents=True, exist_ok=True)
            require(not target.is_symlink(), "source symlink collision")
            if entry.isdir():
                target.mkdir(exist_ok=True)
            elif entry.isfile():
                total += entry.size
                require(
                    entry.size <= 64 * MIB and total <= 2 * GIB,
                    "source tree bound exceeded",
                )
                stream = source.extractfile(entry)
                require(stream is not None, "source member unavailable")
                with stream, target.open("xb") as dest:
                    shutil.copyfileobj(stream, dest, MIB)
                target.chmod(0o755 if entry.mode & 0o111 else 0o644)
                records[relative] = {"kind": "file", **file_record(target, 64 * MIB)}
            elif entry.issym():
                require(
                    not PurePosixPath(entry.linkname).is_absolute()
                    and "\\" not in entry.linkname
                    and "\x00" not in entry.linkname,
                    "unsafe source symlink",
                )
                normalized = list(parts[1:-1])
                for part in entry.linkname.split("/"):
                    if part == "..":
                        require(normalized, "source symlink escapes owned root")
                        normalized.pop()
                    elif part not in ("", "."):
                        normalized.append(part)
                links.append((target, entry.linkname))
                records[relative] = {"kind": "symlink", "target": entry.linkname}
            else:
                raise ValueError("source hardlink/special member rejected")
    for target, text in links:
        require(
            not target.exists() and not target.is_symlink(), "source symlink collision"
        )
        target.symlink_to(text)
    return records


def apply_exact_patch(root, patch, targets):
    """Apply only the three reviewed XLA files, with exact hunk context/no fuzz."""
    lines = patch.decode().splitlines(keepends=True)
    index = 0
    seen = set()
    while index < len(lines):
        require(
            lines[index].startswith("--- ") and index + 1 < len(lines),
            "invalid exact patch header",
        )
        old_name = lines[index][4:].strip()
        new_name = lines[index + 1][4:].strip()
        require(
            lines[index + 1].startswith("+++ b/"), "invalid exact patch destination"
        )
        name = new_name[2:]
        safe_parts(name)
        require(
            name in targets and name not in seen, "unreviewed/duplicate patch target"
        )
        seen.add(name)
        row = targets[name]
        path = root.joinpath(*safe_parts(name))
        before = path.read_bytes() if path.exists() else b""
        require(
            (sha(before) if path.exists() else None) == row["before_sha256"],
            "source patch preimage mismatch/already applied",
        )
        require(
            old_name == ("a/" + name if row["before_sha256"] else "/dev/null"),
            "patch source mismatch",
        )
        old_lines = before.decode().splitlines(keepends=True)
        result = []
        position = 0
        index += 2
        while index < len(lines) and lines[index].startswith("@@ "):
            match = re.fullmatch(
                r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@\n", lines[index]
            )
            require(match is not None, "invalid exact hunk")
            start = max(0, int(match[1]) - 1)
            old_count = int(match[2] or 1)
            new_count = int(match[4] or 1)
            require(position <= start <= len(old_lines), "unordered exact hunk")
            result.extend(old_lines[position:start])
            position = start
            removed = added = 0
            index += 1
            while index < len(lines) and not lines[index].startswith(("@@ ", "--- ")):
                line = lines[index]
                require(line[:1] in (" ", "+", "-"), "unsupported exact patch line")
                if line[0] in " -":
                    require(
                        position < len(old_lines) and old_lines[position] == line[1:],
                        "exact hunk context mismatch",
                    )
                    position += 1
                    removed += 1
                if line[0] in " +":
                    result.append(line[1:])
                    added += 1
                index += 1
            require(
                removed == old_count and added == new_count,
                "exact hunk counts mismatch",
            )
        result.extend(old_lines[position:])
        after = "".join(result).encode()
        require(sha(after) == row["after_sha256"], "source patch postimage mismatch")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(after)
    require(seen == set(targets), "lost reviewed patch target")


def verify_wheel(path, recipe, python_version):
    """Check every wheel RECORD entry and ELF identity without importing code."""
    abi = recipe["inputs"]["abis"].get(python_version)
    require(abi is not None, "unsupported wheel ABI")
    version = recipe["derived"]["version"]
    tag = abi + "-" + abi + "-" + recipe["inputs"]["platform"]
    require(
        path.name == "jaxlib-" + version + "-" + tag + ".whl",
        "wheel filename/version/ABI mismatch",
    )
    before = file_record(path)
    # Bound central-directory allocation before ZipFile materializes its rows.
    with path.open("rb") as source:
        source.seek(max(0, before["bytes"] - 65557))
        tail = source.read(65557)
        candidates = []
        for offset in range(max(0, len(tail) - 21)):
            if tail[offset : offset + 4] == b"PK\x05\x06" and offset + 22 <= len(tail):
                end = struct.unpack_from("<4s4H2LH", tail, offset)
                if offset + 22 + end[-1] == len(tail):
                    candidates.append((before["bytes"] - len(tail) + offset, end))
        require(len(candidates) == 1, "wheel ZIP end record unavailable/ambiguous")
        end_offset, end = candidates[0]
        _, disk, directory_disk, disk_count, count, central_size, central_offset, _ = (
            end
        )
        require(
            disk == directory_disk == 0
            and disk_count == count <= 10000
            and count != 65535
            and central_size <= 16 * MIB
            and count * 46 <= central_size
            and central_offset + central_size == end_offset,
            "wheel ZIP directory exceeds finite schema",
        )
    members = {}
    native = {}
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        require(
            len(infos) <= 10000 and len({i.filename for i in infos}) == len(infos),
            "duplicate/oversized wheel directory",
        )
        require(
            sum(i.file_size for i in infos) <= 2 * GIB,
            "wheel expanded size exceeds bound",
        )
        for info in infos:
            safe_parts(info.filename)
            require(
                not info.is_dir()
                and not info.flag_bits & 1
                and stat.S_IFMT(info.external_attr >> 16) in (0, stat.S_IFREG)
                and info.file_size <= 512 * MIB,
                "unsafe wheel member",
            )
        prefix = "jaxlib-" + version + ".dist-info/"
        require(
            {prefix + n for n in ("METADATA", "WHEEL", "RECORD")}
            <= {i.filename for i in infos},
            "wheel metadata missing",
        )
        for name, maximum in (
            (prefix + "METADATA", MIB),
            (prefix + "WHEEL", 65536),
            (prefix + "RECORD", 8 * MIB),
            ("jaxlib/version.py", MIB),
        ):
            require(
                archive.getinfo(name).file_size <= maximum,
                "wheel metadata exceeds finite bound",
            )
        metadata = BytesParser().parsebytes(archive.read(prefix + "METADATA"))
        require(
            metadata.get_all("Name") == ["jaxlib"]
            and metadata.get_all("Version") == [version],
            "wheel METADATA mismatch",
        )
        wheel = BytesParser().parsebytes(archive.read(prefix + "WHEEL"))
        require(
            wheel.get_all("Tag") == [tag] and wheel.get("Root-Is-Purelib") == "false",
            "wheel WHEEL/ABI mismatch",
        )
        records = {}
        for row in csv.reader(io.StringIO(archive.read(prefix + "RECORD").decode())):
            require(
                len(row) == 3 and row[0] not in records,
                "wheel RECORD repeated/malformed",
            )
            records[row[0]] = row[1:]
        require(
            set(records) == {i.filename for i in infos}, "wheel RECORD is incomplete"
        )
        for info in infos:
            digest = hashlib.sha256()
            count = 0
            head = b""
            with archive.open(info) as source:
                while chunk := source.read(MIB):
                    if not head:
                        head = chunk[:64]
                    count += len(chunk)
                    require(count <= info.file_size, "wheel member grew")
                    digest.update(chunk)
            require(count == info.file_size, "short wheel member")
            encoded = "sha256=" + base64.urlsafe_b64encode(
                digest.digest()
            ).decode().rstrip("=")
            require(
                records[info.filename]
                == (
                    ["", ""]
                    if info.filename == prefix + "RECORD"
                    else [encoded, str(count)]
                ),
                "wheel RECORD hash/size mismatch",
            )
            members[info.filename] = {"bytes": count, "sha256": digest.hexdigest()}
            if info.filename.endswith(".so"):
                require(
                    len(head) >= 20
                    and head[:6] == b"\x7fELF\x02\x01"
                    and struct.unpack_from("<H", head, 18)[0] == 62,
                    "native member is not x86_64 ELF64",
                )
                native[info.filename] = members[info.filename]
        require(
            "jaxlib/libjax_common.so" in native
            and any(n.startswith("jaxlib/_jax.") for n in native),
            "jaxlib native providers missing",
        )
        raw = archive.read("jaxlib/version.py")
        require(len(raw) <= MIB, "version source exceeds bound")
        tree = ast.parse(raw)
        release = [
            n.value.value
            for n in tree.body
            if isinstance(n, (ast.Assign, ast.AnnAssign))
            and isinstance(n.value, ast.Constant)
            and (
                (
                    isinstance(n, ast.AnnAssign)
                    and isinstance(n.target, ast.Name)
                    and n.target.id == "_release_version"
                )
                or (
                    isinstance(n, ast.Assign)
                    and any(
                        isinstance(t, ast.Name) and t.id == "_release_version"
                        for t in n.targets
                    )
                )
            )
        ]
        require(release == [version], "wheel version.py mismatch")
    require(file_record(path) == before, "wheel changed during verification")
    return {
        "file": path.name,
        **before,
        "abi": abi,
        "platform": recipe["inputs"]["platform"],
        "version": version,
        "members": members,
        "native": native,
        "numerical_authority": False,
        "original_tests_qualified": False,
    }


def verify_completed(directory, recipe, python_version):
    manifest = read_json(directory / "wheel-provenance.json")
    require(
        set(manifest)
        == {
            "schema",
            "recipe_sha256",
            "cache_key",
            "python_version",
            "numerical_authority",
            "build_status",
            "source_commits",
            "patch_sha256",
            "compiler_tests_and_manylinux_status",
            "bit_reproducibility",
            "wheel",
            "final_success_receipt_sha256",
        }
        and manifest["schema"] == "corrected-jaxlib-wheel-provenance-v1"
        and manifest["recipe_sha256"] == recipe_digest(recipe)
        and manifest["cache_key"] == cache_key(recipe, python_version)
        and manifest["python_version"] == python_version
        and manifest["numerical_authority"] is False
        and manifest["source_commits"] == {"jax": JAX, "xla": XLA, "llvm": LLVM}
        and manifest["patch_sha256"] == recipe["inputs"]["patch"]["sha256"]
        and manifest["compiler_tests_and_manylinux_status"] == "PASSED"
        and manifest["build_status"]
        == "BUILT_AND_BYTE_VERIFIED_NOT_NUMERICALLY_QUALIFIED",
        "cache provenance mismatch",
    )
    wheels = list(directory.glob("*.whl"))
    require(
        len(wheels) == 1 and wheels[0].name == manifest["wheel"]["file"],
        "cache wheel set mismatch",
    )
    actual = verify_wheel(wheels[0], recipe, python_version)
    require(actual == manifest["wheel"], "cache wheel/native manifest mismatch")
    final_path = directory / "final-success.json"
    require(
        file_record(final_path, 8 * MIB)["sha256"]
        == manifest["final_success_receipt_sha256"],
        "final-success evidence hash mismatch",
    )
    final = read_json(final_path)
    require(
        final["schema"] == "corrected-jaxlib-final-success-v1"
        and final["status"] == "FINAL_BUILD_FENCES_HELD"
        and final["recipe_sha256"] == recipe_digest(recipe)
        and final["python_version"] == python_version
        and final["wheel"] == {k: actual[k] for k in ("file", "bytes", "sha256")}
        and final["source_and_repository_inputs_held"] is True
        and final["environment_held"] is True
        and final["telemetry_durable"] is True
        and final["numerical_authority"] is False,
        "final-success evidence contract mismatch",
    )
    require(
        [c["label"] for c in final["commands"]]
        == [
            "bazel-version",
            "jaxlib-build",
            "compiler-and-manylinux-tests",
            "resolved-module-graph",
        ],
        "final-success command evidence incomplete",
    )
    for command in final["commands"]:
        closed = command["ownership_closure"]
        require(
            command["returncode"] == 0
            and "command_error" not in command
            and "cleanup_error" not in command
            and closed["remaining_live_owned_identities"] == []
            and closed["remaining_adopted_children"] == []
            and closed["subreaper_restored"] is True
            and closed["identity_journal_error"] is None
            and closed["unrecorded_identity_events"] == 0,
            "final-success owned closure unavailable",
        )
    expected_files = {"resources.jsonl"}
    for command in final["commands"]:
        journal = command["ownership_closure"]["identity_journal_file"]
        safe_parts(journal)
        expected_files.update((command["label"] + ".log", journal))
        require(
            final["files"][journal]["sha256"]
            == command["ownership_closure"]["identity_journal_sha256"],
            "final-success identity journal binding mismatch",
        )
    require(
        expected_files
        <= set(final["files"])
        <= expected_files | {"jax/MODULE.bazel.lock", "xla/MODULE.bazel.lock"},
        "final-success log/dependency evidence incomplete",
    )
    for name, row in final["files"].items():
        maximum = LIMITS["log_bytes"] if name.endswith(".log") else 64 * MIB
        require(
            type(row["bytes"]) is int
            and 0 <= row["bytes"] <= maximum
            and type(row["sha256"]) is str
            and re.fullmatch("[0-9a-f]{64}", row["sha256"]),
            "final-success file evidence schema mismatch",
        )
    for name in ("original_source_catalog_sha256", "repository_input_catalog_sha256"):
        require(
            type(final[name]) is str and re.fullmatch("[0-9a-f]{64}", final[name]),
            "final-success original input binding unavailable",
        )
    dependencies = final["resolved_dependency_archives"]
    require(
        type(dependencies) is list
        and len(dependencies) <= 10000
        and all(
            type(r["sha256"]) is str
            and re.fullmatch("[0-9a-f]{64}", r["sha256"])
            and type(r["bytes"]) is int
            and 0 <= r["bytes"] <= 4 * GIB
            for r in dependencies
        ),
        "final-success dependency schema mismatch",
    )
    require(
        all(
            set(choices) & {r["sha256"] for r in dependencies}
            for choices in recipe["inputs"]["hermetic_toolchain"][
                "required_archive_alternatives"
            ]
        ),
        "final-success hermetic dependency evidence unavailable",
    )
    require(
        final["resources"]["peak_owned_rss_bytes"] <= LIMITS["owned_rss"]
        and final["resources"]["minimum_available_memory_bytes"]
        >= LIMITS["live_memory"]
        and final["resources"]["minimum_free_disk_bytes"] >= LIMITS["live_disk"],
        "final-success resource evidence violates budget",
    )
    require(
        {p.name for p in directory.iterdir()}
        == {wheels[0].name, "final-success.json", "wheel-provenance.json"},
        "completed cache contains unexpected files",
    )
    return manifest


def finish_trial(
    out,
    recipe,
    python_version,
    trial,
    report,
    source_records,
    pins,
    original_environment,
    candidate,
):
    """Close original-input/OS/log fences before minting any accepted completion."""
    errors = []
    error_count = 0

    def failed(row):
        nonlocal error_count
        error_count += 1
        if len(errors) < 128:
            errors.append(row)

    for tree, rows in source_records.items():
        for name, row in rows.items():
            path = out / tree / name
            try:
                if row["kind"] == "file":
                    require(
                        file_record(path, 64 * MIB)
                        == {k: row[k] for k in ("bytes", "sha256")},
                        "original build source changed",
                    )
                else:
                    require(
                        path.is_symlink() and os.readlink(path) == row["target"],
                        "original source link changed",
                    )
            except BaseException as error:
                failed(
                    {
                        "path": str(path),
                        "type": type(error).__name__,
                        "message": str(error)[:512],
                    }
                )
    try:
        require(
            {p: file_record(Path(p)) for p in pins} == pins,
            "repository/lock/controller inputs changed",
        )
        require(
            dict(os.environ) == original_environment, "controller environment changed"
        )
    except BaseException as error:
        failed({"type": type(error).__name__, "message": str(error)[:512]})
    try:
        trial.telemetry.flush()
        os.fsync(trial.telemetry.fileno())
    except BaseException as error:
        failed(
            {
                "type": type(error).__name__,
                "message": "telemetry close: " + str(error)[:512],
            }
        )
    finally:
        trial.telemetry.close()
    report.update(
        commands=trial.commands,
        peak_owned_rss_bytes=trial.peak_rss,
        minimum_available_memory_bytes=trial.minimum_memory,
        minimum_free_disk_bytes=trial.minimum_disk,
        source_and_repository_fence_errors=errors,
        source_and_repository_fence_error_count=error_count,
        original_source_and_repository_inputs_held=error_count == 0,
        environment_held=dict(os.environ) == original_environment,
        telemetry_durable=not errors,
        bazel_batch_fresh_owned_output_root=True,
    )
    require(error_count == 0, "post-build source/environment/telemetry fence failed")
    if candidate is None:
        return
    require(
        [c["label"] for c in trial.commands]
        == [
            "bazel-version",
            "jaxlib-build",
            "compiler-and-manylinux-tests",
            "resolved-module-graph",
        ],
        "build/test command evidence incomplete",
    )
    files = {}
    for command in trial.commands:
        closed = command["ownership_closure"]
        require(
            command["returncode"] == 0
            and "command_error" not in command
            and "cleanup_error" not in command
            and closed["remaining_live_owned_identities"] == []
            and closed["remaining_adopted_children"] == []
            and closed["subreaper_restored"] is True
            and closed["identity_journal_error"] is None
            and closed["unrecorded_identity_events"] == 0,
            "owned build closure not complete",
        )
        label = command["label"]
        files[label + ".log"] = file_record(out / (label + ".log"), LIMITS["log_bytes"])
        journal = closed["identity_journal_file"]
        safe_parts(journal)
        files[journal] = file_record(out / journal, 64 * MIB)
        require(
            files[journal]["sha256"] == closed["identity_journal_sha256"],
            "owned journal changed after closure",
        )
    files["resources.jsonl"] = file_record(out / "resources.jsonl", 64 * MIB)
    for tree in ("jax", "xla"):
        generated = out / tree / "MODULE.bazel.lock"
        if generated.exists():
            files[tree + "/MODULE.bazel.lock"] = file_record(generated, 16 * MIB)
    wheel = verify_wheel(candidate, recipe, python_version)
    final = {
        "schema": "corrected-jaxlib-final-success-v1",
        "status": "FINAL_BUILD_FENCES_HELD",
        "recipe_sha256": recipe_digest(recipe),
        "python_version": python_version,
        "wheel": {k: wheel[k] for k in ("file", "bytes", "sha256")},
        "source_and_repository_inputs_held": True,
        "environment_held": True,
        "telemetry_durable": True,
        "original_source_catalog_sha256": sha(canonical(source_records)),
        "repository_input_catalog_sha256": sha(canonical(pins)),
        "commands": trial.commands,
        "files": files,
        "resolved_dependency_archives": report["resolved_dependency_archives"],
        "resources": {
            k: report[k]
            for k in (
                "peak_owned_rss_bytes",
                "minimum_available_memory_bytes",
                "minimum_free_disk_bytes",
            )
        },
        "numerical_authority": False,
    }
    pending = out / "unverified" / "completion"
    completed = out / "completed"
    require(
        not pending.exists() and not completed.exists(),
        "completion destination already exists",
    )
    pending.mkdir()
    try:
        os.rename(candidate, pending / candidate.name)
        write_json(pending / "final-success.json", final)
        manifest = {
            "schema": "corrected-jaxlib-wheel-provenance-v1",
            "recipe_sha256": recipe_digest(recipe),
            "cache_key": cache_key(recipe, python_version),
            "python_version": python_version,
            "build_status": "BUILT_AND_BYTE_VERIFIED_NOT_NUMERICALLY_QUALIFIED",
            "numerical_authority": False,
            "wheel": wheel,
            "source_commits": {"jax": JAX, "xla": XLA, "llvm": LLVM},
            "patch_sha256": recipe["inputs"]["patch"]["sha256"],
            "compiler_tests_and_manylinux_status": "PASSED",
            "bit_reproducibility": "UNPROVEN",
            "final_success_receipt_sha256": file_record(
                pending / "final-success.json", 8 * MIB
            )["sha256"],
        }
        write_json(pending / "wheel-provenance.json", manifest)
        verify_completed(pending, recipe, python_version)
        os.rename(pending, completed)
        descriptor = os.open(out, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        for folder in (pending, completed):
            marker = folder / "wheel-provenance.json"
            if marker.exists():
                marker.unlink()  # Preserve every wheel/log byte, never an accepted failed marker.
        raise
    report["status"] = "BUILT_AND_BYTE_VERIFIED_NOT_NUMERICALLY_QUALIFIED"


def startup_options(out):
    return [
        "--batch",
        "--nohome_rc",
        "--nosystem_rc",
        "--host_jvm_args=-Xmx1024m",
        "--output_user_root=" + str(out / "bazel"),
    ]


def build_options(out, recipe, python_version):
    return [
        "--config=ci_linux_x86_64",
        "--repo_env=ML_WHEEL_TYPE=release",
        "--repo_env=ML_WHEEL_VERSION_SUFFIX=+"
        + recipe["derived"]["version"].split("+", 1)[1],
        "--repo_env=HERMETIC_PYTHON_VERSION=" + python_version,
        "--jobs=" + str(recipe["inputs"]["build_jobs"]),
        "--remote_executor=",
        "--remote_cache=" + recipe["inputs"]["public_bazel_cache"],
        "--remote_upload_local_results=false",
        "--bes_backend=",
        "--repository_cache=" + str(out / "repository-cache"),
        "--override_repository=xla=" + str(out / "xla"),
        "--override_module=xla=" + str(out / "xla"),
    ]


def remote_context_allowed(environment, machine):
    event = environment.get("GITHUB_EVENT_NAME")
    return (
        environment.get("GITHUB_ACTIONS") == "true"
        and environment.get("RUNNER_ENVIRONMENT") == "github-hosted"
        and (
            event == "workflow_dispatch"
            or (
                event == "push"
                and environment.get("GITHUB_REF") == "refs/heads/precision-migration"
            )
        )
        and environment.get("RUNNER_OS") == "Linux"
        and environment.get("RUNNER_ARCH") == "X64"
        and machine == "x86_64"
    )


def build_trial(out, recipe, python_version, report):
    require(
        remote_context_allowed(os.environ, platform.machine()),
        "remote standard-runner pilot only",
    )
    runner_temp = Path(os.environ["RUNNER_TEMP"]).resolve()
    require(
        out.is_relative_to(runner_temp), "output must be owned runner-temp directory"
    )
    require(
        not any(
            os.environ.get(n)
            for n in (
                "XLA_FLAGS",
                "BAZEL_OPTS",
                "JAXCI_XLA_GIT_DIR",
                "CLANG_PATH",
                "CC",
                "CXX",
            )
        ),
        "ambient compiler/runtime override",
    )
    report["startup_resources"] = capacity(out, startup=True)
    trial = Trial(out)
    original_environment = dict(os.environ)
    pins = {
        str(p): file_record(p)
        for p in [
            RECIPE,
            PATCH,
            Path(__file__).resolve(),
            ROOT / "poetry.lock",
            ROOT / "uv.lock",
            ROOT / "pyproject.toml",
            ROOT / ".github/workflows/ci.yml",
            ROOT / "tests/extra_tests/converter/test_source_native_top_k.py",
        ]
    }
    source_records = {}
    body_error = None
    candidate = None
    try:
        for name, commit in (("jax", JAX), ("xla", XLA)):
            archive = trial.download(recipe["inputs"][name], name + ".tar.gz")
            source_records[name] = unpack(
                archive, out / name, name + "-" + commit, trial
            )
            report[name + "_source_tree_sha256"] = sha(canonical(source_records[name]))
        for row in recipe["inputs"]["source_files"]:
            actual = file_record(out / row["tree"] / row["path"], 64 * MIB)
            require(actual["sha256"] == row["sha256"], "pinned build source mismatch")
        apply_exact_patch(
            out / "xla", PATCH.read_bytes(), recipe["inputs"]["patch"]["targets"]
        )
        for name, row in recipe["inputs"]["patch"]["targets"].items():
            source_records["xla"][name] = {
                "kind": "file",
                **file_record(out / "xla" / name, 64 * MIB),
            }
            require(
                source_records["xla"][name]["sha256"] == row["after_sha256"],
                "patched source mismatch",
            )
        bazel = trial.download(recipe["inputs"]["bazel"], "bazel-7.7.0")
        bazel.chmod(0o700)
        # Bazel 7.7's argc==2 metadata shortcut precedes startup parsing,
        # workspace access, JVM launch and installation.
        trial.run([str(bazel), "--version"], out, "bazel-version")
        require(
            (out / "bazel-version.log").read_text().strip() == "bazel 7.7.0",
            "actual Bazel version mismatch",
        )
        staging = out / "unverified"
        staging.mkdir()
        common = build_options(out, recipe, python_version)
        # build.py itself sets HERMETIC_PYTHON_VERSION and the local-XLA overrides.
        forwarded = [
            v
            for v in common
            if not v.startswith(
                (
                    "--repo_env=HERMETIC_PYTHON_VERSION=",
                    "--override_repository=",
                    "--override_module=",
                )
            )
        ]
        argv = [
            sys.executable,
            "build/build.py",
            "build",
            "--wheels=jaxlib",
            "--python_version=" + python_version,
            "--local_xla_path=" + str(out / "xla"),
            "--bazel_path=" + str(bazel),
            "--output_path=" + str(staging),
            "--verbose",
            "--detailed_timestamped_log",
        ]
        argv += ["--bazel_startup_options=" + option for option in startup_options(out)]
        argv += ["--bazel_options=" + option for option in forwarded]
        trial.run(argv, out / "jax", "jaxlib-build")
        base = [str(bazel), *startup_options(out)]
        trial.run(
            [
                *base,
                "test",
                *common,
                "@xla//xla/service/cpu/tests:nested_bf16_conversion_test_cpu",
                "//jaxlib/tools:jaxlib_manylinux_compliance_test",
                "--test_output=all",
            ],
            out / "jax",
            "compiler-and-manylinux-tests",
        )
        trial.run(
            [
                *base,
                "mod",
                "graph",
                "--output=json",
                "--override_module=xla=" + str(out / "xla"),
                "--repository_cache=" + str(out / "repository-cache"),
            ],
            out / "jax",
            "resolved-module-graph",
        )
        cache_records = []
        for path in sorted(
            (out / "repository-cache").glob("content_addressable/sha256/*/file")
        ):
            require(len(cache_records) < 10000, "resolved archive count exceeds bound")
            row = file_record(path, 4 * GIB)
            require(
                row["sha256"] == path.parent.name,
                "resolved dependency cache hash mismatch",
            )
            cache_records.append(row)
        required = recipe["inputs"]["hermetic_toolchain"][
            "required_archive_alternatives"
        ]
        present = {row["sha256"] for row in cache_records}
        require(
            all(set(choices) & present for choices in required),
            "resolved hermetic compiler/sysroot archive unavailable",
        )
        report["resolved_dependency_archives"] = cache_records
        wheels = list(staging.glob("*.whl"))
        require(len(wheels) == 1, "build did not yield exactly one wheel")
        verify_wheel(wheels[0], recipe, python_version)
        candidate = wheels[0]
    except BaseException as error:
        body_error = error
        raise
    finally:
        try:
            finish_trial(
                out,
                recipe,
                python_version,
                trial,
                report,
                source_records,
                pins,
                original_environment,
                candidate if body_error is None else None,
            )
        except BaseException as error:
            report["closing_error"] = {
                "type": type(error).__name__,
                "message": str(error)[:1024],
            }
            if body_error is None:
                raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("describe", "preflight", "build", "verify-cache")
    )
    parser.add_argument("--python-version", required=True, choices=("3.11", "3.12"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args(argv)
    recipe = read_json(RECIPE)
    validate_recipe(recipe, PATCH.read_bytes())
    require(
        file_record(Path(__file__).resolve())["sha256"]
        == recipe["inputs"]["controller_sha256"],
        "controller source identity mismatch",
    )
    if args.mode == "describe":
        print("cache_key=" + cache_key(recipe, args.python_version))
        print("recipe_sha256=" + recipe_digest(recipe))
        print("version=" + recipe["derived"]["version"])
        return 0
    require(
        args.out is not None
        and args.out.is_absolute()
        and not args.out.exists()
        and args.out == args.out.resolve()
        and not args.out.is_symlink(),
        "fresh canonical absolute output required",
    )
    args.out.mkdir(mode=0o700)
    report = {
        "schema": "corrected-jaxlib-remote-trial-v1",
        "mode": args.mode,
        "python_version": args.python_version,
        "recipe_sha256": recipe_digest(recipe),
        "status": "UNAVAILABLE",
        "numerical_authority": False,
        "original_tests_qualified": False,
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "cpus": os.cpu_count(),
        "source_and_binary_reproducibility": "NOT_ASSERTED",
        "errors": [],
    }
    try:
        if args.mode == "preflight":
            report["capacity"] = capacity(args.out, startup=True)
            report["status"] = "CAPACITY_OBSERVED_BUILD_NOT_RUN"
        elif args.mode == "verify-cache":
            require(args.cache_dir is not None, "exact cache directory required")
            capacity(args.out)
            report["cached_manifest"] = verify_completed(
                args.cache_dir, recipe, args.python_version
            )
            report["status"] = "EXACT_CACHE_BYTE_VERIFIED_NOT_NUMERICALLY_QUALIFIED"
        else:
            build_trial(args.out, recipe, args.python_version, report)
    except BaseException as error:
        report["status"] = "UNAVAILABLE"
        report["errors"].append(
            {"type": type(error).__name__, "message": str(error)[:2048]}
        )
        marker = args.out / "completed/wheel-provenance.json"
        if marker.exists():
            marker.unlink()
    finally:
        try:
            write_json(args.out / "trial.json", report)
        except BaseException:
            marker = args.out / "completed/wheel-provenance.json"
            if marker.exists():
                marker.unlink()
            raise
        print(
            json.dumps(
                {k: report[k] for k in ("status", "mode", "python_version", "errors")}
            )
        )
    return 1 if report["status"] == "UNAVAILABLE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
