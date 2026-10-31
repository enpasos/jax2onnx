# tests/extra_tests/framework/test_run_with_resources.py

"""Finite stdlib controls for sampled, command-scoped resource accounting."""

import json
from importlib.util import module_from_spec, spec_from_file_location
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest


_SCRIPT = Path(__file__).resolve().parents[3] / "scripts/run_with_resources.py"


def _observer():
    spec = spec_from_file_location("_resource_observer_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _command(tmp_path, code, *, options=(), arguments=(), environment=None):
    cwd = tmp_path / "cwd"
    cwd.mkdir(exist_ok=True)
    output = tmp_path / "resources"
    command = [
        sys.executable,
        "-B",
        str(_SCRIPT),
        "--output-dir",
        str(output),
        "--timeout-seconds",
        "10",
        *options,
        "--",
        sys.executable,
        "-B",
        "-c",
        code,
        *arguments,
    ]
    result = subprocess.run(
        command, cwd=cwd, env=environment, capture_output=True, text=True, timeout=18
    )
    summary = (
        json.loads((output / "summary.json").read_text())
        if (output / "summary.json").exists()
        else None
    )
    return result, summary, output


def _injected_command(tmp_path, patch, child_code, *, options=()):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    output = tmp_path / "resources"
    script = f"""import importlib.util,subprocess,sys,os,signal,time
spec=importlib.util.spec_from_file_location('resource_observer',{str(_SCRIPT)!r})
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
{patch}
sys.argv=['observer','--output-dir',{str(output)!r},'--timeout-seconds','10',*{list(options)!r},'--',sys.executable,'-B','-c',{child_code!r}]
raise SystemExit(module._main())
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return result, json.loads((output / "summary.json").read_text()), output


def test_ordinary_command_preserves_argv_environment_cwd_and_streams(tmp_path):
    environment = {
        **os.environ,
        "RESOURCE_TEST_SENTINEL": "unchanged",
        "JAX_ENABLE_X64": "original-literal",
    }
    code = "import os,sys,json;print(json.dumps([sys.argv[1:],os.getcwd(),os.environ['RESOURCE_TEST_SENTINEL'],os.environ['JAX_ENABLE_X64']]));print('child stderr',file=sys.stderr)"
    result, summary, output = _command(
        tmp_path, code, arguments=("a b", "--literal"), environment=environment
    )
    assert result.returncode == 0
    assert json.loads(result.stdout) == [
        ["a b", "--literal"],
        str(tmp_path / "cwd"),
        "unchanged",
        "original-literal",
    ]
    assert result.stderr == "child stderr\n"
    assert summary["qualification"] == "PASS" and summary["child_returncode"] == 0
    assert summary["limits"]["rss_mib"] == 4096
    assert "diagnostic_only" not in summary
    assert summary["cleanup_confirmed"] and summary["remaining_descendants"] == []
    rows = [
        json.loads(line) for line in (output / "samples.jsonl").read_text().splitlines()
    ]
    assert rows[-1]["stage"] == "terminal"
    assert summary["sample_count"] == len(rows)
    assert summary["terminal_sample_gap_seconds"] < 2
    assert (
        summary["peak_owned_rss_bytes"]
        >= max(row["owned_rss_bytes"] for row in rows)
        > 0
    )
    observer = summary["observer_identity"]
    ledger = {tuple(identity) for identity in summary["owned_pid_start_ledger"]}
    for row in rows:
        vector = row["owned_process_vector"]
        identities = {(item["pid"], item["start"]) for item in vector}
        assert len(vector) == len(identities) == row["owned_processes"]
        assert identities <= ledger
        assert (observer["pid"], observer["start"]) in identities
        assert sum(item["rss"] for item in vector) == row["owned_rss_bytes"]
        assert all(
            set(item) == {"pid", "parent", "start", "rss", "state"} for item in vector
        )
        assert all(item["parent"] >= 0 and len(item["state"]) == 1 for item in vector)
    peak = summary["peak_owned_rss_sample"]
    assert set(peak) == {
        "stage",
        "elapsed_seconds",
        "owned_rss_bytes",
        "owned_processes",
        "owned_process_vector",
    }
    peak_row = max(rows, key=lambda row: row["owned_rss_bytes"])
    assert peak == {name: peak_row[name] for name in peak}


@pytest.mark.parametrize("code", [7, 23])
def test_nonzero_child_status_is_preserved_separately_from_observation(tmp_path, code):
    result, summary, _ = _command(tmp_path, f"raise SystemExit({code})")
    assert result.returncode == code == summary["child_returncode"]
    assert summary["status"] == "COMPLETE"
    assert summary["qualification"] == "FAIL_OR_INCOMPLETE"


@pytest.mark.parametrize(
    "options",
    [
        ("--max-rss-mib", "5120", "--min-available-mib", "8192"),
        ("--diagnostic-only", "--max-rss-mib", "5121", "--min-available-mib", "8192"),
        (
            "--diagnostic-only",
            "--max-rss-mib",
            "5120",
            "--min-available-mib",
            "8192",
            "--timeout-seconds",
            "1101",
        ),
        ("--diagnostic-only", "--max-rss-mib", "5120", "--min-available-mib", "8191"),
        (
            "--diagnostic-only",
            "--max-rss-mib",
            "5120",
            "--min-available-mib",
            "8192",
            "--min-disk-free-mib",
            "4095",
        ),
        ("--diagnostic-only", "--max-rss-mib", "4096", "--min-available-mib", "8192"),
    ],
    ids=[
        "missing-opt-in",
        "rss-above5120",
        "work-above1100",
        "live-below8192",
        "disk-below4096",
        "diagnostic-cap-not5120",
    ],
)
def test_diagnostic_cli_rejects_invalid_limits_before_child(tmp_path, options):
    marker = tmp_path / "must-not-run"
    result, summary, output = _command(
        tmp_path,
        f"from pathlib import Path;Path({str(marker)!r}).touch()",
        options=options,
    )
    assert result.returncode == 2 and "bounded STANDARD" in result.stderr
    assert summary is None and not output.exists() and not marker.exists()


def _diagnostic_capacity_patch(available_kib):
    return f"""original_kernel=module._kernel
def kernel_with_mock_capacity():
    value=original_kernel()
    value['memory']['MemAvailable']={available_kib}
    return value
module._kernel=kernel_with_mock_capacity
"""


@pytest.mark.parametrize("code", [0, 7])
def test_diagnostic_cli_keeps_child_status_unqualified_with_mock_capacity(
    tmp_path, code
):
    result, summary, _ = _injected_command(
        tmp_path,
        _diagnostic_capacity_patch(12288 * 1024),
        f"print('diagnostic child');raise SystemExit({code})",
        options=(
            "--diagnostic-only",
            "--max-rss-mib",
            "5120",
            "--min-available-mib",
            "8192",
        ),
    )
    assert result.returncode == (125 if code == 0 else code)
    assert result.stdout == "diagnostic child\n"
    assert summary["status"] == "COMPLETE" and summary["child_returncode"] == code
    assert summary["qualification"] == summary["standard_acceptance"] == "UNQUALIFIED"
    assert summary["standard_acceptance_rss_mib"] == 4096
    assert summary["limits"]["rss_mib"] == 5120
    assert summary["limits"]["host_available_mib"] == 8192
    assert summary["limits"]["initial_host_available_mib"] == 12288
    assert summary["initial_host_available_bytes"] == 12288 * 1024**2
    assert summary["cleanup_confirmed"] and not summary["errors"]


def test_diagnostic_initial_capacity_rejection_precedes_child(tmp_path):
    marker = tmp_path / "must-not-run"
    result, summary, _ = _injected_command(
        tmp_path,
        _diagnostic_capacity_patch(12288 * 1024 - 1),
        f"from pathlib import Path;Path({str(marker)!r}).touch()",
        options=(
            "--diagnostic-only",
            "--max-rss-mib",
            "5120",
            "--min-available-mib",
            "8192",
        ),
    )
    assert result.returncode == 125 and summary["child_returncode"] is None
    assert not marker.exists()
    assert summary["qualification"] == "UNQUALIFIED"
    assert (
        "ObservationError: diagnostic initial 12 GiB availability floor"
        in summary["errors"]
    )
    assert summary["cleanup_confirmed"]


def test_process_vectors_do_not_bypass_the_sample_evidence_bound(tmp_path):
    result, summary, output = _injected_command(
        tmp_path, "module._MAX_SAMPLES_BYTES=1", "raise AssertionError('must not run')"
    )
    assert result.returncode == 125
    assert summary["child_returncode"] is None
    assert summary["qualification"] == "FAIL_OR_INCOMPLETE"
    assert any("sample evidence exceeds" in error for error in summary["errors"])
    assert (output / "samples.jsonl").read_bytes() == b""


def test_child_signal_is_preserved_as_an_actual_signal_exit(tmp_path):
    result, summary, _ = _command(
        tmp_path, "import os,signal;os.kill(os.getpid(),signal.SIGTERM)"
    )
    assert result.returncode == -signal.SIGTERM == summary["child_returncode"]
    assert summary["qualification"] == "FAIL_OR_INCOMPLETE"


def test_subreaper_discovers_setsid_grandchild_after_intermediate_exit(tmp_path):
    proof = tmp_path / "grandchild.json"
    code = f"""import os,time,json,pathlib
expected_parent=os.getppid()
if os.fork() == 0:
    if os.fork() == 0:
        os.setsid()
        deadline=time.monotonic()+2
        while os.getppid()!=expected_parent and time.monotonic()<deadline:time.sleep(0.005)
        pathlib.Path({str(proof)!r}).write_text(json.dumps({{'pid':os.getpid(),'parent':os.getppid()}}))
        time.sleep(0.55)
    os._exit(0)
os._exit(0)
"""
    result, summary, output = _command(tmp_path, code)
    assert result.returncode == 0 and summary["cleanup_confirmed"]
    child = json.loads(proof.read_text())
    assert child["parent"] == summary["observer_identity"]["pid"]
    identities = dict(summary["owned_pid_start_ledger"])
    assert child["pid"] in identities
    assert _observer()._live(child["pid"], identities[child["pid"]]) is None
    rows = [
        json.loads(line) for line in (output / "samples.jsonl").read_text().splitlines()
    ]
    assert max(row["owned_processes"] for row in rows) >= 2
    assert summary["elapsed_seconds"] >= 0.5
    assert any(
        item["pid"] == child["pid"]
        and item["start"] == identities[child["pid"]]
        and item["parent"] == child["parent"]
        for row in rows
        for item in row["owned_process_vector"]
    )


@pytest.mark.parametrize(
    "option,reason",
    [
        ("--max-rss-mib", "RSS"),
        ("--min-available-mib", "memory floor"),
        ("--min-disk-free-mib", "free-space floor"),
    ],
)
def test_observed_resource_breach_cannot_launch_or_pass(tmp_path, option, reason):
    value = "1" if option == "--max-rss-mib" else str(2**40)
    marker = tmp_path / "child-ran"
    result, summary, _ = _command(
        tmp_path,
        f"from pathlib import Path;Path({str(marker)!r}).touch()",
        options=(option, value),
    )
    assert result.returncode == 125
    assert summary["qualification"] == "FAIL_OR_INCOMPLETE"
    assert any(reason in error for error in summary["errors"])
    assert not marker.exists()


def test_deadline_stops_owned_work_and_samples_cleanup(tmp_path):
    result, summary, output = _command(
        tmp_path, "import time;time.sleep(30)", options=("--timeout-seconds", "0.35")
    )
    assert result.returncode == 125 and summary["child_returncode"] == -signal.SIGTERM
    assert summary["cleanup_confirmed"] and not summary["remaining_descendants"]
    assert any("deadline" in error for error in summary["errors"])
    rows = [
        json.loads(line) for line in (output / "samples.jsonl").read_text().splitlines()
    ]
    assert "cleanup" in [row["stage"] for row in rows]
    assert rows[-1]["stage"] == "terminal"
    assert summary["max_sample_gap_seconds"] >= 0


def test_observer_SIGTERM_uses_bounded_TERM_KILL_cleanup_and_preserves_signal(tmp_path):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    ready = tmp_path / "ready"
    output = tmp_path / "resources"
    code = f"import signal,time,pathlib,os;retained=[];signal.signal(signal.SIGTERM,lambda *args:retained.append(bytearray(16*1024**2)));pathlib.Path({str(ready)!r}).write_text(str(os.getpid()));time.sleep(30)"
    process = subprocess.Popen(
        [
            sys.executable,
            "-B",
            str(_SCRIPT),
            "--output-dir",
            str(output),
            "--timeout-seconds",
            "10",
            "--",
            sys.executable,
            "-B",
            "-c",
            code,
        ],
        cwd=cwd,
    )
    try:
        deadline = time.monotonic() + 5
        while (
            not ready.exists()
            and process.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        assert ready.exists()
        process.terminate()
        assert process.wait(timeout=12) == -signal.SIGTERM
        summary = json.loads((output / "summary.json").read_text())
        assert summary["child_returncode"] == -signal.SIGKILL
        assert summary["observer_signal"] == signal.SIGTERM
        assert summary["guard_result_code"] == 125
        assert (
            summary["planned_wrapper_returncode"]
            == process.returncode
            == -signal.SIGTERM
        )
        assert summary["cleanup_confirmed"] and not summary["remaining_descendants"]
        sent = [row["signal"] for row in summary["signals_sent"]]
        assert sent == [signal.SIGTERM, signal.SIGKILL]
        rows = [
            json.loads(line)
            for line in (output / "samples.jsonl").read_text().splitlines()
        ]
        command_peak = max(
            row["owned_rss_bytes"] for row in rows if row["stage"] == "command"
        )
        cleanup_peak = max(
            row["owned_rss_bytes"] for row in rows if row["stage"] == "cleanup"
        )
        assert cleanup_peak >= command_peak + 8 * 1024**2
        assert summary["peak_owned_rss_bytes"] == max(
            row["owned_rss_bytes"] for row in rows
        )
        peak = summary["peak_owned_rss_sample"]
        assert peak["stage"] == "cleanup"
        assert peak == {
            name: max(rows, key=lambda row: row["owned_rss_bytes"])[name]
            for name in peak
        }
        assert any(
            item["pid"] == int(ready.read_text()) and item["rss"] >= 16 * 1024**2
            for item in peak["owned_process_vector"]
        )
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


@pytest.mark.parametrize("gone", [False, True])
def test_unreadable_live_identity_is_not_silently_omitted(monkeypatch, gone):
    module = _observer()

    def unreadable(pid):
        raise PermissionError("controlled stat read failure")

    monkeypatch.setattr(module, "_process", unreadable)
    monkeypatch.setattr(module, "_gone", lambda pid: gone)
    if gone:
        assert module._live(101, 11) is None
    else:
        with pytest.raises(module.ObservationError, match="unreadable identity/RSS"):
            module._live(101, 11)


@pytest.mark.parametrize("when", ["enumeration", "admission"])
def test_reused_parent_identity_cannot_admit_an_unrelated_child(monkeypatch, when):
    module = _observer()
    parent = module._Process(101, 1, 11, 100, "S")
    state = {"reused": False, "child_reads": 0}

    class Task:
        def __truediv__(self, name):
            return self

        def read_text(self):
            if when == "enumeration":
                state["reused"] = True
            return "202"

    class Tasks:
        def iterdir(self):
            return iter([Task()])

    def live(pid, start=None):
        if pid == parent.pid:
            return parent._replace(start=99) if state["reused"] else parent
        state["child_reads"] += 1
        state["reused"] = True
        return module._Process(202, parent.pid, 22, 100, "S")

    monkeypatch.setattr(module, "Path", lambda value: Tasks())
    monkeypatch.setattr(module, "_live", live)
    with pytest.raises(module.ObservationError, match="parent PID was reused"):
        module._children(parent)
    assert state["child_reads"] == (0 if when == "enumeration" else 1)


def test_cleanup_never_signals_a_reused_pid(monkeypatch):
    module = _observer()
    item = module._Process(202, 101, 22, 100, "S")
    tree = SimpleNamespace(
        owner=module._Process(101, 1, 11, 100, "S"),
        identities={(101, 11), (202, 22)},
        reap=lambda pid: None,
    )
    rounds = iter([[item], [], []])
    monkeypatch.setattr(module, "_remaining", lambda tree: next(rounds, []))
    monkeypatch.setattr(module, "_live", lambda pid, start=None: None)
    sent = []
    monkeypatch.setattr(
        module.os, "kill", lambda pid, number: sent.append((pid, number))
    )
    monkeypatch.setattr(module.time, "sleep", lambda delay: None)
    assert module._cleanup(tree, SimpleNamespace(pid=303, poll=lambda: 0), []) == []
    assert sent == []


def test_reaping_never_steals_the_Popen_leader_or_foreign_child(monkeypatch):
    module = _observer()
    tree = module._Owned.__new__(module._Owned)
    tree.owner = module._Process(101, 1, 11, 100, "S")
    tree.identities = {(101, 11), (202, 22), (303, 33), (404, 44)}
    tree.wait_outcomes = []
    monkeypatch.setattr(
        module,
        "_live",
        lambda pid, start=None: module._Process(
            pid, 999 if pid == 404 else 101, start, 0, "Z"
        ),
    )
    reaped = []

    def wait(pid, flags):
        reaped.append(pid)
        return pid, 0

    monkeypatch.setattr(module.os, "waitpid", wait)
    tree.reap(202)
    assert reaped == [303]
    assert tree.wait_outcomes == [
        {"pid": 303, "start": 33, "returncode": 0, "signal": None}
    ]


@pytest.mark.parametrize("mapping", ["root", "nonroot", "empty"])
def test_cgroup_uses_actual_mount_mapping_or_reports_unavailable(
    tmp_path, monkeypatch, mapping
):
    module = _observer()
    mount = tmp_path / "mount space"
    (mount / "nested").mkdir(parents=True)
    if mapping != "empty":
        for name, value in {
            "memory.current": "12",
            "memory.max": "max",
            "memory.events.local": "oom 1\noom_kill 0\n",
            "memory.swap.current": "3",
            "memory.swap.max": "max",
        }.items():
            (mount / "nested" / name).write_text(value)
    original = Path.read_text

    def read(path, *args, **kwargs):
        if str(path) == "/proc/self/cgroup":
            return "0::/nested\n"
        if str(path) == "/proc/self/mountinfo":
            escaped = str(mount).replace(" ", "\\040")
            return f"30 20 0:99 {'/subtree' if mapping == 'nonroot' else '/'} {escaped} rw - cgroup2 cgroup rw\n"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    result = module._cgroup()
    if mapping == "root":
        assert result["status"] == "OBSERVED_SHARED"
        assert (
            result["path"] == str(mount / "nested") and result["memory.current"] == 12
        )
        assert result["memory.max"] == result["memory.swap.max"] == "max"
        assert result["memory.events.local"] == {"oom": 1, "oom_kill": 0}
        assert result["memory.peak"] is None
    else:
        assert result["status"] == "UNAVAILABLE"
        assert result.get("memory.current") is None


def test_optional_counter_unavailability_and_resets_are_never_zero_deltas():
    module = _observer()
    assert module._delta(None, 10) == {"status": "UNAVAILABLE", "value": None}
    assert module._delta(10, 2) == {"status": "RESET", "value": None}
    assert (
        module._cgroup_deltas({"status": "UNAVAILABLE"}, {})["status"]
        == "UNAVAILABLE_OR_MEMBERSHIP_CHANGED"
    )
    before = {
        "status": "OBSERVED_SHARED",
        "membership": "0::/x",
        "path": "/x",
        "memory.events": {"oom": 2},
    }
    after = {**before, "memory.events": {"oom": 1}}
    result = module._cgroup_deltas(before, after)
    assert result["memory.events"]["oom"]["value"] is None
    assert result["swap_growth"] is None
    unavailable = {
        "memory": {"MemAvailable": 100, "SwapFree": None},
        "vmstat": {"pswpin": None, "pswpout": None, "oom_kill": None},
        "cgroup": {"status": "UNAVAILABLE"},
    }
    assert module._growth(unavailable, unavailable) == []


def test_artifact_bytes_include_the_final_summary_without_copying_child_logs(tmp_path):
    module = _observer()
    (tmp_path / "samples.jsonl").write_text("{}\n")
    summary = {}
    payload = module._artifacts(tmp_path, summary)
    (tmp_path / "summary.json").write_bytes(payload)
    assert summary["observer_artifacts"]["bytes"] == sum(
        path.stat().st_size for path in tmp_path.iterdir()
    )
    assert summary["observer_artifacts"]["files"] == 2


def test_existing_output_is_rejected_without_overwriting_or_running_child(tmp_path):
    output = tmp_path / "resources"
    output.mkdir()
    (output / "sentinel").write_text("preserve")
    result, summary, _ = _command(tmp_path, "raise AssertionError('must not run')")
    assert result.returncode == 2 and summary is None
    assert (output / "sentinel").read_text() == "preserve"


def test_standalone_admission_does_not_signal_preexisting_children(tmp_path):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    output = tmp_path / "resources"
    marker = tmp_path / "must-not-run"
    code = f"""import importlib.util,subprocess,sys
spec=importlib.util.spec_from_file_location('resource_observer',{str(_SCRIPT)!r})
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
foreign=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
sys.argv=['observer','--output-dir',{str(output)!r},'--timeout-seconds','2','--',sys.executable,'-c',"from pathlib import Path;Path({str(marker)!r}).touch()"]
try:
    result=module._main()
    assert foreign.poll() is None
    raise SystemExit(result)
finally:
    foreign.terminate();foreign.wait()
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=8,
    )
    assert result.returncode == 125
    summary = json.loads((output / "summary.json").read_text())
    assert any("preexisting child" in value for value in summary["errors"])
    assert summary["child_returncode"] is None and not marker.exists()


def test_first_post_Popen_accounting_failure_still_closes_the_owned_leader(tmp_path):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    output = tmp_path / "resources"
    code = f"""import importlib.util,subprocess,sys
spec=importlib.util.spec_from_file_location('resource_observer',{str(_SCRIPT)!r})
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
original_sample=module._Owned.sample
def fail_before_admission(tree):
    if len(tree.identities)==1 and module._children(tree.owner):
        raise module.ObservationError('controlled first post-Popen accounting failure')
    return original_sample(tree)
module._Owned.sample=fail_before_admission
created=[];original_Popen=subprocess.Popen
def spawn(*args,**kwargs):
    child=original_Popen(*args,**kwargs);created.append(child);return child
module.subprocess.Popen=spawn
sys.argv=['observer','--output-dir',{str(output)!r},'--timeout-seconds','2','--',sys.executable,'-c','import time;time.sleep(30)']
try:
    result=module._main()
    assert all(child.poll() is not None for child in created)
    raise SystemExit(result)
finally:
    for child in created:
        if child.poll() is None:child.terminate();child.wait()
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 125
    summary = json.loads((output / "summary.json").read_text())
    assert summary["child_returncode"] == -signal.SIGTERM
    assert summary["cleanup_confirmed"] and not summary["remaining_descendants"]
    assert any("post-Popen accounting failure" in error for error in summary["errors"])
    assert any(
        row.get("ownership") == "unreaped-Popen-handle"
        for row in summary["signals_sent"]
    )


def test_missed_interval_fails_but_does_not_disable_fresh_cleanup_samples(tmp_path):
    patch = """original_kernel=module._kernel
calls=0
def late_once():
    global calls
    calls+=1
    if calls==3:time.sleep(2.05)
    return original_kernel()
module._kernel=late_once
"""
    result, summary, output = _injected_command(
        tmp_path, patch, "import time;time.sleep(30)"
    )
    assert result.returncode == 125 and summary["child_returncode"] == -signal.SIGTERM
    assert summary["cleanup_confirmed"]
    assert any("sampling gap" in value for value in summary["errors"])
    rows = [
        json.loads(line) for line in (output / "samples.jsonl").read_text().splitlines()
    ]
    assert summary["max_sample_gap_seconds"] > 2
    assert "cleanup" in [row["stage"] for row in rows]
    assert (
        rows[-1]["stage"] == "terminal" and summary["terminal_sample_gap_seconds"] < 2
    )


def test_signal_during_receipt_construction_is_recorded_as_non_PASS(tmp_path):
    patch = """original_artifacts=module._artifacts
injected=False
def interrupted_receipt(*args):
    global injected
    if not injected:
        injected=True
        os.kill(os.getpid(),signal.SIGTERM)
    return original_artifacts(*args)
module._artifacts=interrupted_receipt
"""
    result, summary, _ = _injected_command(tmp_path, patch, "pass")
    assert result.returncode == -signal.SIGTERM
    assert (
        summary["child_returncode"] == 0
        and summary["observer_signal"] == signal.SIGTERM
    )
    assert (
        summary["guard_result_code"] == 125
        and summary["planned_wrapper_returncode"] == -signal.SIGTERM
    )
    assert summary["qualification"] == "FAIL_OR_INCOMPLETE"


def test_observed_adopted_signal_death_fails_without_optional_OOM_counters(tmp_path):
    patch = """original_kernel=module._kernel
def no_optional():
    value=original_kernel()
    value['vmstat']={name:None for name in value['vmstat']}
    value['memory']['SwapFree']=None
    value['cgroup']={'status':'UNAVAILABLE'}
    return value
module._kernel=no_optional
module._cgroup=lambda:{'status':'UNAVAILABLE'}
"""
    child = """import os,time,signal
expected_parent=os.getppid()
if os.fork()==0:
    if os.fork()==0:
        os.setsid()
        deadline=time.monotonic()+2
        while os.getppid()!=expected_parent and time.monotonic()<deadline:time.sleep(0.005)
        os.kill(os.getpid(),signal.SIGKILL)
    os._exit(0)
os._exit(0)
"""
    result, summary, _ = _injected_command(tmp_path, patch, child)
    assert result.returncode == 125 and summary["child_returncode"] == 0
    assert any(
        row["signal"] == signal.SIGKILL for row in summary["adopted_wait_outcomes"]
    )
    assert summary["qualification"] == "FAIL_OR_INCOMPLETE"
    assert any("OOM cause UNVERIFIED" in value for value in summary["errors"])


def test_subreaper_restoration_failure_keeps_child0_but_writes_failed_receipt(tmp_path):
    patch = """original_subreaper=module._subreaper
initial=original_subreaper()
enabled=False
def restore_failure(value=None):
    global enabled
    if value==1:
        enabled=True
    elif enabled and value==initial:
        raise OSError('controlled subreaper restoration failure')
    return original_subreaper(value)
module._subreaper=restore_failure
"""
    result, summary, _ = _injected_command(tmp_path, patch, "pass")
    assert result.returncode == 125 and summary["child_returncode"] == 0
    assert summary["qualification"] == "FAIL_OR_INCOMPLETE"
    assert any("restoration failure" in value for value in summary["errors"])
