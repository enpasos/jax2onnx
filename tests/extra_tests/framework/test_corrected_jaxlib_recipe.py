# tests/extra_tests/framework/test_corrected_jaxlib_recipe.py

"""Finite stdlib controls for build inputs and uninstalled wheel byte evidence."""

from __future__ import annotations

import base64
import copy
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock
import warnings
import zipfile


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "corrected_jaxlib_builder", ROOT / "scripts/build_corrected_jaxlib.py"
)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def recipe():
    return builder.read_json(builder.RECIPE)


def redigest(value):
    digest = builder.recipe_digest(value)
    value["derived"] = {
        "recipe_sha256": digest,
        "version": "0.10.2+jax2onnx.bf16." + digest[:16],
    }
    return value


def wheel_fixture(folder, value, python_version="3.11", mutation=None):
    abi = value["inputs"]["abis"][python_version]
    version = value["derived"]["version"]
    prefix = "jaxlib-" + version + ".dist-info/"
    native = bytearray(64)
    native[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<HHI", native, 16, 3, 62, 1)
    files = {
        prefix
        + "METADATA": (
            "Metadata-Version: 2.3\nName: jaxlib\nVersion: " + version + "\n"
        ).encode(),
        prefix
        + "WHEEL": (
            "Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: "
            + abi
            + "-"
            + abi
            + "-manylinux_2_27_x86_64\n"
        ).encode(),
        "jaxlib/version.py": (
            "_release_version: str = " + repr(version) + "\n"
        ).encode(),
        "jaxlib/_jax.so": bytes(native),
        "jaxlib/libjax_common.so": bytes(native) + b"format-fixture-not-executable",
    }
    if mutation in (
        "metadata-version",
        "wheel-abi",
        "version-source",
        "elf-architecture",
        "unsafe-path",
        "missing-native",
    ):
        if mutation == "metadata-version":
            files[prefix + "METADATA"] = b"Name: jaxlib\nVersion: 0.10.2\n"
        elif mutation == "wheel-abi":
            files[prefix + "WHEEL"] = files[prefix + "WHEEL"].replace(
                b"cp311-cp311", b"cp312-cp312"
            )
        elif mutation == "version-source":
            files["jaxlib/version.py"] = b"_release_version = '0.10.2'\n"
        elif mutation == "elf-architecture":
            wrong = bytearray(files["jaxlib/_jax.so"])
            struct.pack_into("<H", wrong, 18, 183)
            files["jaxlib/_jax.so"] = bytes(wrong)
        elif mutation == "unsafe-path":
            files["../outside"] = b"outside"
        else:
            del files["jaxlib/libjax_common.so"]
    rows = []
    for name, data in files.items():
        digest = (
            base64.urlsafe_b64encode(bytes.fromhex(builder.sha(data)))
            .decode()
            .rstrip("=")
        )
        rows.append([name, "sha256=" + digest, str(len(data))])
    if mutation == "record-omission":
        rows.pop()
    if mutation == "record-duplicate":
        rows.append(rows[0])
    rows.append([prefix + "RECORD", "", ""])
    text = io.StringIO()
    csv.writer(text).writerows(rows)
    files[prefix + "RECORD"] = text.getvalue().encode()
    if mutation == "record-hash":
        files["jaxlib/_jax.so"] += b"changed-after-record"
    path = folder / (
        "jaxlib-" + version + "-" + abi + "-" + abi + "-manylinux_2_27_x86_64.whl"
    )
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in files.items():
            info = zipfile.ZipInfo(name)
            info.external_attr = (
                (
                    stat.S_IFLNK
                    if mutation == "symlink" and name == "jaxlib/_jax.so"
                    else stat.S_IFREG
                )
                | 0o600
            ) << 16
            archive.writestr(info, data)
        if mutation == "duplicate-zip":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                archive.writestr("jaxlib/_jax.so", files["jaxlib/_jax.so"])
    return path


def bind_final_fixture(folder, value, manifest):
    """Synthetic uninstalled format evidence, never an actual compiler result."""
    commands = []
    files = {"resources.jsonl": {"bytes": 0, "sha256": builder.sha(b"")}}
    for label in (
        "bazel-version",
        "jaxlib-build",
        "compiler-and-manylinux-tests",
        "resolved-module-graph",
    ):
        journal = label + "-owned-identities.jsonl"
        files[label + ".log"] = {"bytes": 0, "sha256": builder.sha(b"")}
        files[journal] = {"bytes": 0, "sha256": builder.sha(b"")}
        commands.append(
            {
                "label": label,
                "returncode": 0,
                "ownership_closure": {
                    "closed_identity_count": 1,
                    "identity_journal_file": journal,
                    "identity_journal_sha256": builder.sha(b""),
                    "identity_journal_error": None,
                    "unrecorded_identity_events": 0,
                    "remaining_live_owned_identities": [],
                    "remaining_adopted_children": [],
                    "subreaper_restored": True,
                },
            }
        )
    final = {
        "schema": "corrected-jaxlib-final-success-v1",
        "status": "FINAL_BUILD_FENCES_HELD",
        "recipe_sha256": builder.recipe_digest(value),
        "python_version": manifest["python_version"],
        "wheel": {k: manifest["wheel"][k] for k in ("file", "bytes", "sha256")},
        "source_and_repository_inputs_held": True,
        "environment_held": True,
        "telemetry_durable": True,
        "original_source_catalog_sha256": builder.sha(b"format source catalogue"),
        "repository_input_catalog_sha256": builder.sha(b"format repository catalogue"),
        "commands": commands,
        "files": files,
        "resolved_dependency_archives": [
            {"bytes": 1, "sha256": choices[0]}
            for choices in value["inputs"]["hermetic_toolchain"][
                "required_archive_alternatives"
            ]
        ],
        "resources": {
            "peak_owned_rss_bytes": 1,
            "minimum_available_memory_bytes": builder.LIMITS["live_memory"],
            "minimum_free_disk_bytes": builder.LIMITS["live_disk"],
        },
        "numerical_authority": False,
    }
    target = folder / "final-success.json"
    target.write_text(json.dumps(final))
    manifest["final_success_receipt_sha256"] = builder.file_record(target)["sha256"]
    return final


SUPERVISOR = r'''
import ctypes, importlib.util, json, os, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("tiny_corrected_builder", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
out = Path(sys.argv[2])
mode = sys.argv[3]
libc = ctypes.CDLL(None)
def subreaper():
    value = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(value), 0, 0, 0) == 0
    return value.value
before = subreaper()
trial = module.Trial(out)
if mode == "timeout":
    module.LIMITS["timeout_seconds"] = 1  # This isolated test process only.
if mode == "rss":
    module.LIMITS["owned_rss"] = module.OwnedBuildProcesses.process(os.getpid())["rss"] + 24 * module.MIB
worker = r"""
import json, os, signal, subprocess, sys, time
from pathlib import Path
mode, directory = sys.argv[1], Path(sys.argv[2])
grandchild = 'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)'
if mode == 'rss':
    grandchild = 'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); data=bytearray(64*1024**2); data[::4096]=b"x"*16384; time.sleep(30)'
child = subprocess.Popen([sys.executable, '-B', '-c', grandchild], start_new_session=True,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
def start(pid):
    return int((Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[19])
(directory / 'worker.json').write_text(json.dumps({'worker': os.getpid(), 'grandchild': child.pid,
    'worker_start': start(os.getpid()), 'grandchild_start': start(child.pid),
    'worker_pgrp': os.getpgrp(), 'grandchild_pgrp': os.getpgid(child.pid)}))
if mode == 'waited':
    time.sleep(1.5)
    child.terminate()
    child.kill()
    child.wait()
elif mode in ('timeout', 'rss', 'count'):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(30)
elif mode == 'nonzero':
    raise SystemExit(7)
"""
if mode == "count":
    original_sample = trial.sample
    def lower_count_after_ready(ownership=None):
        try:
            json.loads((out / 'worker.json').read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        else:
            # Lower only this isolated control's admission limit after both
            # real identities and their complete PID/starttime record exist.
            module.OWNED_PROCESS_LIMIT = 1
        return original_sample(ownership)
    trial.sample = lower_count_after_ready
error = None
try:
    trial.run([sys.executable, '-B', '-c', worker, mode, str(out)], out, 'tiny')
except BaseException as cause:
    error = {'type': type(cause).__name__, 'message': str(cause)}
finally:
    trial.telemetry.flush()
    os.fsync(trial.telemetry.fileno())
    trial.telemetry.close()
    module.write_json(out / 'supervisor-result.json', {'error': error, 'commands': trial.commands,
        'before_subreaper': before, 'after_subreaper': subreaper(), 'peak_rss': trial.peak_rss})
'''


def real_supervision_case(test, mode):
    """Root executes this tiny OS control; the author only wrote its source."""
    with tempfile.TemporaryDirectory() as name:
        out = Path(name)
        sentinel = subprocess.Popen(
            [sys.executable, "-B", "-c", "import time; time.sleep(30)"],
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            # Sentinel is the supervisor's sibling, never its direct child.
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-c",
                    SUPERVISOR,
                    str(ROOT / "scripts/build_corrected_jaxlib.py"),
                    str(out),
                    mode,
                ],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=20,
            )
            test.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            test.assertIsNone(sentinel.poll(), "unrelated sibling was signalled")
            report = json.loads((out / "supervisor-result.json").read_text())
            test.assertEqual(report["before_subreaper"], report["after_subreaper"])
            test.assertEqual(len(report["commands"]), 1)
            closed = report["commands"][0]["ownership_closure"]
            test.assertEqual(closed["remaining_live_owned_identities"], [])
            test.assertEqual(closed["remaining_adopted_children"], [])
            test.assertTrue(closed["subreaper_restored"])
            test.assertIsNone(closed["identity_journal_error"])
            test.assertEqual(closed["unrecorded_identity_events"], 0)
            journal = out / closed["identity_journal_file"]
            test.assertEqual(
                builder.file_record(journal)["sha256"],
                closed["identity_journal_sha256"],
            )
            events = [json.loads(line) for line in journal.read_text().splitlines()]
            identities = {
                (r["pid"], r["start"]) for r in events if r["event"] == "retained"
            }
            worker = json.loads((out / "worker.json").read_text())
            test.assertNotEqual(worker["worker_pgrp"], worker["grandchild_pgrp"])
            test.assertIn(worker["grandchild"], {pid for pid, _ in identities})
            test.assertNotIn(sentinel.pid, {pid for pid, _ in identities})
            test.assertEqual(closed["closed_identity_count"], len(identities))
            for pid, start in identities:
                current = builder.OwnedBuildProcesses.process(pid)
                test.assertTrue(
                    current is None or current["start"] != start,
                    "an owned process identity survived cleanup",
                )
            if mode in ("timeout", "rss", "nonzero", "count"):
                test.assertIsNotNone(report["error"])
                expected = {
                    "timeout": "timeout",
                    "rss": "aggregate owned RSS",
                    "nonzero": "owned command failed",
                    "count": "concurrent owned process identity bound",
                }[mode]
                test.assertIn(expected, report["error"]["message"])
                test.assertIn(
                    expected, report["commands"][0]["command_error"]["message"]
                )
            else:
                test.assertIsNone(report["error"], report)
            if mode == "waited":
                samples = [
                    json.loads(line)
                    for line in (out / "resources.jsonl").read_text().splitlines()
                ]
                test.assertTrue(
                    any(
                        worker["grandchild"]
                        in {r["pid"] for r in row["owned_identities"]}
                        for row in samples
                    )
                )
        finally:
            # Independent failure cleanup uses exact retained worker identities,
            # never bare names/groups or a potentially reused numeric PID.
            if (out / "worker.json").exists():
                worker = json.loads((out / "worker.json").read_text())
                for role in ("worker", "grandchild"):
                    pid, start = worker[role], worker[role + "_start"]
                    current = builder.OwnedBuildProcesses.process(pid)
                    if current is not None and current["start"] == start:
                        try:
                            descriptor = os.pidfd_open(pid, 0)
                            try:
                                current = builder.OwnedBuildProcesses.process(pid)
                                if current is not None and current["start"] == start:
                                    builder.signal.pidfd_send_signal(
                                        descriptor, builder.signal.SIGKILL, None, 0
                                    )
                            finally:
                                os.close(descriptor)
                        except ProcessLookupError:
                            pass
            if sentinel.poll() is None:
                sentinel.terminate()
            sentinel.wait(timeout=5)


class CorrectedJaxlibRecipeTest(unittest.TestCase):
    def test_real_count_admission_failure_still_closes_owned_setsid_children(self):
        real_supervision_case(self, "count")

    def test_real_setsid_descendants_are_accounted_without_touching_sibling(self):
        real_supervision_case(self, "waited")

    def test_real_fast_parent_exit_adopts_and_reaps_setsid_descendant(self):
        real_supervision_case(self, "orphan")

    def test_real_resource_failures_preserve_failure_and_close_descendants(self):
        for mode in ("timeout", "rss", "nonzero"):
            with self.subTest(mode=mode):
                real_supervision_case(self, mode)

    def test_reused_pid_starttime_cannot_retain_or_signal_replacement(self):
        owned = object.__new__(builder.OwnedBuildProcesses)
        owned.known = {}
        old = {"pid": 12345, "start": 100, "ppid": 1, "state": "S"}
        replacement = {**old, "start": 200}
        with (
            mock.patch.object(
                builder.OwnedBuildProcesses, "process", side_effect=[old, replacement]
            ),
            mock.patch.object(builder.os, "pidfd_open", return_value=77),
            mock.patch.object(builder.os, "close") as close,
            mock.patch.object(builder.signal, "pidfd_send_signal") as send,
        ):
            owned.retain(old)
            self.assertEqual(owned.known, {})
            close.assert_called_once_with(77)
            owned.known = {(12345, 100): 77}
            self.assertEqual(owned.live({12345: replacement}), [])
            owned.closed = True
            with self.assertRaisesRegex(ValueError, "closed ownership"):
                owned.signal_owned(builder.signal.SIGTERM)
            send.assert_not_called()

    def test_late_real_source_fence_failure_keeps_staged_wheel_unverified(self):
        class ClosedFormatTrial:
            def __init__(self, out):
                self.telemetry = (out / "resources.jsonl").open("x")
                self.commands = []
                self.peak_rss = 1
                self.minimum_memory = builder.LIMITS["live_memory"]
                self.minimum_disk = builder.LIMITS["live_disk"]

        with tempfile.TemporaryDirectory() as name:
            out = Path(name)
            staged = out / "unverified"
            staged.mkdir()
            value = recipe()
            wheel = wheel_fixture(staged, value)
            before = builder.verify_wheel(wheel, value, "3.11")
            source = out / "authenticated-source.txt"
            source.write_bytes(b"before wheel staging")
            pins = {str(source): builder.file_record(source)}
            source.write_bytes(b"changed after valid wheel staging")
            report = {}
            with self.assertRaisesRegex(
                ValueError, "post-build source/environment/telemetry fence failed"
            ):
                builder.finish_trial(
                    out,
                    value,
                    "3.11",
                    ClosedFormatTrial(out),
                    report,
                    {},
                    pins,
                    dict(os.environ),
                    wheel,
                )
            self.assertEqual(builder.verify_wheel(wheel, value, "3.11"), before)
            self.assertTrue(report["source_and_repository_fence_errors"])
            self.assertFalse(report["original_source_and_repository_inputs_held"])
            self.assertEqual(list(out.rglob("wheel-provenance.json")), [])
            with self.assertRaises(FileNotFoundError):
                builder.verify_completed(out / "completed", value, "3.11")

    def test_cache_rejects_bound_failed_incomplete_or_missing_final_success(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            value = recipe()
            wheel = wheel_fixture(folder, value)
            manifest = {
                "schema": "corrected-jaxlib-wheel-provenance-v1",
                "recipe_sha256": builder.recipe_digest(value),
                "cache_key": builder.cache_key(value, "3.11"),
                "python_version": "3.11",
                "numerical_authority": False,
                "build_status": "BUILT_AND_BYTE_VERIFIED_NOT_NUMERICALLY_QUALIFIED",
                "source_commits": {
                    "jax": builder.JAX,
                    "xla": builder.XLA,
                    "llvm": builder.LLVM,
                },
                "patch_sha256": value["inputs"]["patch"]["sha256"],
                "compiler_tests_and_manylinux_status": "PASSED",
                "bit_reproducibility": "UNPROVEN",
                "wheel": builder.verify_wheel(wheel, value, "3.11"),
            }
            final = bind_final_fixture(folder, value, manifest)
            target = folder / "wheel-provenance.json"
            target.write_text(json.dumps(manifest))
            self.assertEqual(builder.verify_completed(folder, value, "3.11"), manifest)
            edits = [
                lambda v: v.update(status="UNVERIFIED"),
                lambda v: v.update(source_and_repository_inputs_held=False),
                lambda v: v.update(telemetry_durable=False),
                lambda v: v["commands"][1].update(returncode=1),
                lambda v: v["commands"][1]["ownership_closure"].update(
                    remaining_adopted_children=[12345]
                ),
                lambda v: v["resources"].update(minimum_free_disk_bytes=0),
                lambda v: v.update(original_source_catalog_sha256="unbound"),
                lambda v: v["files"].clear(),
                lambda v: v["resolved_dependency_archives"].clear(),
            ]
            for edit in edits:
                changed = copy.deepcopy(final)
                edit(changed)
                (folder / "final-success.json").write_text(json.dumps(changed))
                current = copy.deepcopy(manifest)
                current["final_success_receipt_sha256"] = builder.file_record(
                    folder / "final-success.json"
                )["sha256"]
                target.write_text(json.dumps(current))
                with self.assertRaises((ValueError, KeyError)):
                    builder.verify_completed(folder, value, "3.11")
            target.write_text(json.dumps(manifest))
            (folder / "final-success.json").unlink()
            with self.assertRaises(FileNotFoundError):
                builder.verify_completed(folder, value, "3.11")

    def test_remote_push_context_keeps_wrong_ref_and_local_build_denials(self):
        environment = {
            "GITHUB_ACTIONS": "true",
            "RUNNER_ENVIRONMENT": "github-hosted",
            "GITHUB_EVENT_NAME": "push",
            "GITHUB_REF": "refs/heads/precision-migration",
            "RUNNER_OS": "Linux",
            "RUNNER_ARCH": "X64",
        }
        self.assertTrue(builder.remote_context_allowed(environment, "x86_64"))
        manual = {**environment, "GITHUB_EVENT_NAME": "workflow_dispatch"}
        self.assertTrue(builder.remote_context_allowed(manual, "x86_64"))
        for field, value in (
            ("GITHUB_REF", "refs/heads/main"),
            ("GITHUB_REF", "refs/pull/1/merge"),
            ("GITHUB_EVENT_NAME", "pull_request"),
            ("GITHUB_EVENT_NAME", "local"),
            ("GITHUB_EVENT_NAME", None),
            ("GITHUB_ACTIONS", "false"),
            ("RUNNER_ENVIRONMENT", "self-hosted"),
            ("RUNNER_OS", "Windows"),
            ("RUNNER_ARCH", "ARM64"),
        ):
            with self.subTest(field=field, value=value):
                self.assertFalse(
                    builder.remote_context_allowed(
                        {**environment, field: value}, "x86_64"
                    )
                )
        self.assertFalse(builder.remote_context_allowed({}, "x86_64"))
        self.assertFalse(builder.remote_context_allowed(environment, "aarch64"))

    def test_authorized_branch_push_workflow_uses_exact_paths_and_cache(self):
        workflow = (ROOT / ".github/workflows/build-corrected-jaxlib.yml").read_text()
        expected_push = """  push:
    branches: [precision-migration]
    paths:
    - build_support/jaxlib/xla-0.10.2-bf16-nested-conversion.patch
    - build_support/jaxlib/cpu-bf16-build-recipe.json
    - scripts/build_corrected_jaxlib.py
    - .github/workflows/build-corrected-jaxlib.yml
    - tests/extra_tests/framework/test_corrected_jaxlib_recipe.py
    - docs/about/dependencies.md
"""
        events = workflow.split("\non:\n", 1)[1].split("\npermissions:\n", 1)[0]
        self.assertEqual(events.split("  workflow_dispatch:\n", 1)[0], expected_push)
        self.assertIn(
            "      if: github.event_name == 'push' || inputs.use_completed_cache\n",
            workflow,
        )
        self.assertIn(
            "      if: (github.event_name == 'push' || inputs.use_completed_cache) && steps.build.outcome == 'success'\n",
            workflow,
        )
        self.assertEqual(
            workflow.count(
                "        path: ${{ runner.temp }}/corrected-jaxlib-build-${{ matrix.python-version }}/completed\n"
            ),
            2,
        )
        self.assertNotIn("restore-keys", workflow)
        self.assertIn("python-version: ['3.11', '3.12']", workflow)
        self.assertIn("runs-on: ubuntu-24.04", workflow)
        self.assertIn("timeout-minutes: 350", workflow)

    def test_valid_recipe_has_distinct_exact_abi_keys_and_visible_version(self):
        value = recipe()
        digest = builder.validate_recipe(value, builder.PATCH.read_bytes())
        self.assertEqual(
            builder.file_record(ROOT / "scripts/build_corrected_jaxlib.py")["sha256"],
            value["inputs"]["controller_sha256"],
        )
        self.assertEqual(
            value["derived"]["version"], "0.10.2+jax2onnx.bf16." + digest[:16]
        )
        self.assertNotEqual(
            builder.cache_key(value, "3.11"), builder.cache_key(value, "3.12")
        )
        self.assertTrue(builder.cache_key(value, "3.11").endswith(digest))
        with self.assertRaisesRegex(ValueError, "unsupported Python ABI"):
            builder.cache_key(value, "3.13")

    def test_wrong_revision_archive_size_or_hash_rejects_even_after_redigest(self):
        for tree, field, replacement in (
            ("jax", "commit", "0" * 40),
            ("xla", "commit", "0" * 40),
            ("jax", "bytes", 20056128),
            ("xla", "bytes", 28172459),
            ("jax", "sha256", "0" * 64),
            ("xla", "sha256", "0" * 64),
        ):
            with self.subTest(tree=tree, field=field):
                value = recipe()
                value["inputs"][tree][field] = replacement
                with self.assertRaises(ValueError):
                    builder.validate_recipe(redigest(value), builder.PATCH.read_bytes())
        value = recipe()
        value["inputs"]["llvm_commit"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "revision"):
            builder.validate_recipe(redigest(value), builder.PATCH.read_bytes())

    def test_wrong_patch_targets_abi_tool_or_resource_contract_rejects(self):
        edits = [
            lambda v: v["inputs"]["patch"]["targets"].clear(),
            lambda v: v["inputs"]["abis"].update({"3.13": "cp313"}),
            lambda v: v["inputs"]["bazel"].update(version="7.4.1"),
            lambda v: v["inputs"]["hermetic_toolchain"][
                "required_archive_alternatives"
            ].clear(),
            lambda v: v["inputs"]["resources"].update(startup_disk=4 * builder.GIB),
            lambda v: v["inputs"].update(target_cpu_features="native"),
        ]
        for edit in edits:
            value = recipe()
            edit(value)
            with self.assertRaises(ValueError):
                builder.validate_recipe(redigest(value), builder.PATCH.read_bytes())
        value = recipe()
        wrong = builder.PATCH.read_bytes() + b"\n"
        value["inputs"]["patch"]["sha256"] = builder.sha(wrong)
        with self.assertRaisesRegex(ValueError, "compiler patch"):
            builder.validate_recipe(redigest(value), wrong)

    def test_both_synthetic_format_carriers_remain_without_numerical_authority(self):
        # These deliberately unexecutable ELF-header carriers test format/hash
        # parsing. Accepted format evidence never qualifies native behavior.
        with tempfile.TemporaryDirectory() as name:
            for py in ("3.11", "3.12"):
                path = wheel_fixture(Path(name), recipe(), py)
                result = builder.verify_wheel(path, recipe(), py)
                self.assertFalse(result["numerical_authority"])
                self.assertFalse(result["original_tests_qualified"])
                self.assertEqual(result["abi"], "cp" + py.replace(".", ""))

    def test_corrupt_metadata_record_names_native_identity_and_paths_reject(self):
        for mutation in (
            "metadata-version",
            "wheel-abi",
            "version-source",
            "elf-architecture",
            "unsafe-path",
            "missing-native",
            "record-omission",
            "record-duplicate",
            "record-hash",
            "symlink",
            "duplicate-zip",
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as name:
                path = wheel_fixture(Path(name), recipe(), mutation=mutation)
                with self.assertRaises(ValueError):
                    builder.verify_wheel(path, recipe(), "3.11")

    def test_foreign_filename_abi_and_incomplete_zip_fail(self):
        with tempfile.TemporaryDirectory() as name:
            path = wheel_fixture(Path(name), recipe())
            with self.assertRaisesRegex(ValueError, "ABI"):
                builder.verify_wheel(path, recipe(), "3.12")
            raw = path.read_bytes()
            path.write_bytes(raw[:-8])
            with self.assertRaisesRegex(ValueError, "ZIP"):
                builder.verify_wheel(path, recipe(), "3.11")

    def test_cache_requires_exact_recipe_abi_patch_and_every_wheel_hash(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            value = recipe()
            path = wheel_fixture(folder, value)
            manifest = {
                "schema": "corrected-jaxlib-wheel-provenance-v1",
                "recipe_sha256": builder.recipe_digest(value),
                "cache_key": builder.cache_key(value, "3.11"),
                "python_version": "3.11",
                "numerical_authority": False,
                "build_status": "BUILT_AND_BYTE_VERIFIED_NOT_NUMERICALLY_QUALIFIED",
                "source_commits": {
                    "jax": builder.JAX,
                    "xla": builder.XLA,
                    "llvm": builder.LLVM,
                },
                "patch_sha256": value["inputs"]["patch"]["sha256"],
                "compiler_tests_and_manylinux_status": "PASSED",
                "bit_reproducibility": "UNPROVEN",
                "wheel": builder.verify_wheel(path, value, "3.11"),
            }
            target = folder / "wheel-provenance.json"
            bind_final_fixture(folder, value, manifest)
            target.write_text(json.dumps(manifest))
            self.assertEqual(builder.verify_completed(folder, value, "3.11"), manifest)
            mutations = [
                lambda v: v.update(cache_key="corrected-jaxlib-linux-x86_64-cp311-"),
                lambda v: v.update(python_version="3.12"),
                lambda v: v.update(numerical_authority=True),
                lambda v: v.update(patch_sha256="0" * 64),
                lambda v: v["source_commits"].update(xla="0" * 40),
                lambda v: v["wheel"].update(sha256="0" * 64),
                lambda v: v["wheel"]["native"]["jaxlib/_jax.so"].update(
                    sha256="0" * 64
                ),
            ]
            for mutate in mutations:
                changed = copy.deepcopy(manifest)
                mutate(changed)
                target.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    builder.verify_completed(folder, value, "3.11")

    def test_exact_patch_has_no_fuzz_replay_or_unreviewed_target(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root / "file.cc"
            path.write_bytes(b"before\n")
            targets = {
                "file.cc": {
                    "before_sha256": builder.sha(b"before\n"),
                    "after_sha256": builder.sha(b"after\n"),
                }
            }
            patch = b"--- a/file.cc\n+++ b/file.cc\n@@ -1 +1 @@\n-before\n+after\n"
            builder.apply_exact_patch(root, patch, targets)
            self.assertEqual(path.read_bytes(), b"after\n")
            with self.assertRaisesRegex(ValueError, "preimage"):
                builder.apply_exact_patch(root, patch, targets)
            path.write_bytes(b"before\n")
            with self.assertRaises(ValueError):
                builder.apply_exact_patch(
                    root, patch.replace(b"-before", b"-BEFORE"), targets
                )
            with self.assertRaises(ValueError):
                builder.apply_exact_patch(
                    root, patch.replace(b"b/file.cc", b"b/../escape"), targets
                )

    def test_source_hardlink_or_traversal_never_extracts_outside_owned_root(self):
        class NoNativeTrial:
            def sample(self):
                pass

        for member in ("prefix/../../escape", "/absolute", "prefix/link"):
            with self.subTest(member=member), tempfile.TemporaryDirectory() as name:
                folder = Path(name)
                archive = folder / "source.tar.gz"
                with tarfile.open(archive, "w:gz") as target:
                    item = tarfile.TarInfo(member)
                    item.type = (
                        tarfile.LNKTYPE if member == "prefix/link" else tarfile.REGTYPE
                    )
                    item.linkname = "../../escape"
                    target.addfile(item, io.BytesIO(b""))
                with self.assertRaises(ValueError):
                    builder.unpack(archive, folder / "owned", "prefix", NoNativeTrial())
                self.assertFalse((folder / "escape").exists())


if __name__ == "__main__":
    unittest.main()
