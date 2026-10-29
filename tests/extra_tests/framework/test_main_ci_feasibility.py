# tests/extra_tests/framework/test_main_ci_feasibility.py

"""Bounded controller/owned-process controls; no model or cohort acceptance."""

from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts/observe_main_ci_feasibility.py"
_GIB = 2**30


@pytest.fixture(scope="module")
def observer():
    spec = spec_from_file_location("_main_feasibility_test", _SCRIPT)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def records(observer):
    ids = [
        "tests/test_cases.py::test_pass",
        "tests/test_cases.py::test_setup",
        "tests/test_call.py::test_call",
    ]
    skips = [
        {"nodeid": ids[1], "phase": "setup", "reason": "setup reason", "line": 17},
        {"nodeid": ids[2], "phase": "call", "reason": "call reason", "line": 175},
    ]
    authority = {
        "collection": {
            "count": 3,
            "ordered_ids_sha256": observer._digest(ids),
            "phase_count": 8,
        },
        "skips": skips,
    }
    closures = {
        worker: {
            "worker_id": worker,
            "closed": True,
            "error": None,
            "exitstatus": 0,
            "testrun_uid": "uid",
            "shouldfail": False,
            "shouldstop": False,
        }
        for worker in ("gw0", "gw1")
    }
    events = []
    for index, node in enumerate(ids):
        worker = "gw1" if index == 2 else "gw0"
        allowance = next((item for item in skips if item["nodeid"] == node), None)
        for phase in (
            ("setup", "teardown") if index == 1 else ("setup", "call", "teardown")
        ):
            skipped = allowance is not None and phase == allowance["phase"]
            event = {
                "nodeid": node,
                "when": phase,
                "outcome": "skipped" if skipped else "passed",
                "duration": 0.1,
                "wasxfail_present": False,
                "worker_id": worker,
                "transport_worker_id": worker,
                "item_index": index,
                "testrun_uid": "uid",
            }
            if skipped:
                location = [
                    "/subject/" + node.split("::")[0],
                    allowance["line"],
                    "Skipped: " + allowance["reason"],
                ]
                event.update(skip_location=location, longrepr=str(tuple(location)))
            events.append(event)
    return authority, {worker: list(ids) for worker in closures}, events, closures


def test_complete_controller_stream_and_native_closures(observer, records):
    assert observer._validate(*records) == {
        "passed": 1,
        "skipped": 2,
        "count": 3,
        "phase_count": 8,
    }


@pytest.mark.parametrize("fault", ("order", "missing", "duplicate", "extra"))
def test_both_full_ordered_collections_are_mandatory(observer, records, fault):
    ids = records[1]["gw1"]
    if fault == "order":
        ids.reverse()
    elif fault == "missing":
        records[1].pop("gw1")
    elif fault == "duplicate":
        ids[-1] = ids[0]
    else:
        ids.append("tests/test_extra.py::test_extra")
    with pytest.raises(ValueError):
        observer._validate(*records)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("worker_id", "gw1"),
        ("transport_worker_id", "gw1"),
        ("testrun_uid", "forged"),
        ("item_index", False),
        ("item_index", 0.0),
        ("item_index", -1),
        ("item_index", 3),
        ("item_index", 1),
    ),
)
def test_native_transport_index_and_uid_cannot_be_forged(
    observer, records, field, value
):
    records[2][0][field] = value
    with pytest.raises(ValueError):
        observer._validate(*records)


@pytest.mark.parametrize(
    "field",
    (
        "worker_id",
        "transport_worker_id",
        "item_index",
        "testrun_uid",
        "wasxfail_present",
    ),
)
def test_missing_observations_are_not_proof(observer, records, field):
    records[2][0].pop(field)
    with pytest.raises(ValueError):
        observer._validate(*records)


@pytest.mark.parametrize(
    "fault", ("missing", "duplicate", "contradictory", "cross-worker")
)
def test_bad_phase_multiplicity_and_execution_ownership_reject(
    observer, records, fault
):
    events = records[2]
    if fault == "missing":
        events.pop(2)
    elif fault == "duplicate":
        events.insert(1, deepcopy(events[0]))
    elif fault == "contradictory":
        events[1]["outcome"] = "failed"
    else:
        events[1].update(worker_id="gw1", transport_worker_id="gw1")
    with pytest.raises(ValueError):
        observer._validate(*records)


@pytest.mark.parametrize("fault", ("missing", "open", "nonzero", "crash", "stop"))
def test_native_closure_is_required(observer, records, fault):
    closures = records[3]
    if fault == "missing":
        closures.pop("gw1")
    else:
        field, value = {
            "open": ("closed", False),
            "nonzero": ("exitstatus", 1),
            "crash": ("error", "worker crashed"),
            "stop": ("shouldstop", True),
        }[fault]
        closures["gw1"][field] = value
    with pytest.raises(ValueError):
        observer._validate(*records)


@pytest.mark.parametrize("value", (None, False, "xfail"))
def test_any_wasxfail_attribute_is_rejected(observer, records, value):
    records[2][0]["wasxfail"] = value
    with pytest.raises(ValueError):
        observer._validate(*records)


@pytest.mark.parametrize("fault", ("reason", "line", "owner", "stage"))
def test_skip_allowances_are_exact(observer, records, fault):
    event = records[2][3]
    if fault == "reason":
        event["skip_location"][2] += " extra"
    elif fault == "line":
        event["skip_location"][1] += 1
    elif fault == "owner":
        event["skip_location"][0] = "/wrong.py"
    else:
        event["when"] = "call"
    event["longrepr"] = str(tuple(event["skip_location"]))
    with pytest.raises(ValueError):
        observer._validate(*records)


def test_tracked_admission_precedes_generated_authority(observer):
    source = {
        "head": "frozen",
        "tracked_status": "",
        "files": {"tests/base.py": "sha"},
        "generated": {"base.py": "sha"},
    }
    authority = {
        "subject": {
            "commit": "frozen",
            "source_digest": observer._digest(source["files"]),
            "generated_digest": observer._digest(
                {**source["generated"], "new.py": "generated"}
            ),
        }
    }
    observer._check_source(source, authority)
    with pytest.raises(ValueError):
        observer._check_source(source, authority, generated=True)
    source["generated"]["new.py"] = "generated"
    observer._check_source(source, authority, generated=True)
    source["files"]["tests/base.py"] = "changed"
    with pytest.raises(ValueError):
        observer._check_source(source, authority)


def _sample() -> dict[str, int]:
    return {
        "effective_available": 6 * _GIB,
        "disk_free": 4 * _GIB,
        "SwapFree": 16 * _GIB,
        "pswpin": 0,
        "pswpout": 0,
        "host_oom_kill": 0,
        "cgroup_oom": 0,
        "cgroup_oom_kill": 0,
    }


def test_hosted_guard_accepts_exact_admission_and_live_boundaries(
    observer: Any,
) -> None:
    baseline = _sample()
    observer._admit(baseline)
    sample = {**baseline, "effective_available": 2 * _GIB}
    assert observer._stop_reason(sample, baseline, 4 * _GIB, 1200) is None


@pytest.mark.parametrize("name", tuple(_sample()))
def test_missing_safety_telemetry_fails_admission_and_returns_owned_stop_reason(
    observer: Any, name: str
) -> None:
    sample = _sample()
    sample.pop(name)
    with pytest.raises(ValueError):
        observer._admit(sample)
    assert observer._stop_reason(sample, _sample(), 0, 0)


@pytest.mark.parametrize("name", ("effective_available", "disk_free"))
def test_hosted_admission_reserves_are_not_the_local_twelve_eight_profile(
    observer: Any, name: str
) -> None:
    sample = _sample()
    sample[name] -= 1
    with pytest.raises(ValueError):
        observer._admit(sample)


@pytest.mark.parametrize(
    ("field", "value", "rss", "elapsed"),
    (
        ("effective_available", 2 * _GIB - 1, 0, 0),
        ("disk_free", 4 * _GIB - 1, 0, 0),
        ("SwapFree", 16 * _GIB - 1, 0, 0),
        ("pswpin", 1, 0, 0),
        ("pswpout", 1, 0, 0),
        ("host_oom_kill", 1, 0, 0),
        ("cgroup_oom", 1, 0, 0),
        ("cgroup_oom_kill", 1, 0, 0),
        ("effective_available", 6 * _GIB, 4 * _GIB + 1, 0),
        ("effective_available", 6 * _GIB, 0, 1200.001),
    ),
)
def test_every_resource_growth_or_budget_breach_produces_a_stop_reason(
    observer: Any, field: str, value: int, rss: int, elapsed: float
) -> None:
    sample = {**_sample(), field: value}
    assert observer._stop_reason(sample, _sample(), rss, elapsed)


def test_swap_counter_growth_still_stops_after_swapfree_recovers(observer: Any) -> None:
    sample = {**_sample(), "SwapFree": 17 * _GIB, "pswpout": 1}
    assert observer._stop_reason(sample, _sample(), 0, 0)


@pytest.mark.parametrize(
    "field", ("pswpin", "pswpout", "host_oom_kill", "cgroup_oom", "cgroup_oom_kill")
)
def test_counter_regression_is_missing_trustworthy_telemetry(
    observer: Any, field: str
) -> None:
    baseline = {**_sample(), field: 2}
    sample = {**baseline, field: 1}
    assert observer._stop_reason(sample, baseline, 0, 0)


@pytest.mark.parametrize("value", (None, False, -1, "0"))
def test_invalid_kernel_metric_cannot_bypass_cleanup(observer: Any, value: Any) -> None:
    sample = {**_sample(), "pswpout": value}
    with pytest.raises(ValueError):
        observer._admit(sample)
    assert observer._stop_reason(sample, _sample(), 0, 0)


def _stat(parent, start, state="S"):
    return {"parent": parent, "start": start, "rss": 4096, "state": state}


def test_live_owned_read_failure_cannot_be_counted_as_zero(observer, monkeypatch):
    monkeypatch.setattr(observer, "_read_stat", lambda pid: _stat(1, 11))
    with pytest.raises(ValueError, match="live/uncertain"):
        observer._after_read_error(100, 11, PermissionError("unreadable"))


def test_verified_exit_and_pid_reuse_are_not_signalled(observer, monkeypatch):
    monkeypatch.setattr(observer, "_read_stat", lambda pid: _stat(1, 99))
    assert observer._after_read_error(100, 11, FileNotFoundError()) is None
    calls = []
    monkeypatch.setattr(
        observer.os, "kill", lambda pid, number: calls.append((pid, number))
    )
    observer._signal_owned({100: 11}, {100: _stat(1, 11)}, signal.SIGTERM)
    assert calls == []


def test_disappearance_requires_a_liveness_recheck(observer, monkeypatch):
    def gone(pid):
        raise FileNotFoundError(pid)

    def not_live(pid, number):
        raise ProcessLookupError(pid)

    monkeypatch.setattr(observer, "_read_stat", gone)
    monkeypatch.setattr(observer.os, "kill", not_live)
    assert observer._after_read_error(100, 11, FileNotFoundError()) is None


def _guard_unit_profile(observer, monkeypatch):
    # These integration controls prove process/pluggy mechanics, not hosted safety telemetry.
    # The authorized local test command remains protected by its actual external resource guard.
    monkeypatch.setattr(observer, "_telemetry", lambda subject: _sample())
    monkeypatch.setattr(observer.os, "sched_getaffinity", lambda pid: {4, 5, 6, 7})


def test_normal_fast_leader_exit_is_resampled_without_false_descendants(
    observer, monkeypatch, tmp_path
):
    _guard_unit_profile(observer, monkeypatch)
    result = observer._guard(
        [sys.executable, "-c", "pass"], tmp_path, tmp_path, "fast", 5, os.environ.copy()
    )
    assert result["exit"] == 0 and result["stop"] is None and result["cleanup_complete"]
    assert result["subreaper_restored"]


def test_double_fork_setsid_before_first_poll_is_adopted_and_terminated(
    observer, monkeypatch, tmp_path
):
    _guard_unit_profile(observer, monkeypatch)
    marker = tmp_path / "grandchild.pid"
    code = (
        "import os,time,pathlib; p=os.fork(); os._exit(0) if p else None; os.setsid(); p=os.fork(); os._exit(0) if p else None; pathlib.Path("
        + repr(str(marker))
        + ").write_text(str(os.getpid())); time.sleep(3)"
    )
    original = observer._owned_tree
    first = True

    def delayed(owned, leader):
        nonlocal first
        if first:
            deadline = time.monotonic() + 2
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            first = False
        return original(owned, leader)

    monkeypatch.setattr(observer, "_owned_tree", delayed)
    with pytest.raises(ValueError):
        observer._guard(
            [sys.executable, "-c", code],
            tmp_path,
            tmp_path,
            "escape",
            5,
            os.environ.copy(),
        )
    record = json.loads((tmp_path / "escape-result.json").read_text())
    grandchild = int(marker.read_text())
    assert record["exit"] == 0  # The adopted reaper never steals Popen's leader status.
    assert record["cleanup_complete"] and record["subreaper_restored"]
    assert any(item["pid"] == grandchild for item in record["reaped_adopted"])
    with pytest.raises(ProcessLookupError):
        os.kill(grandchild, 0)


def test_actual_two_worker_nonnative_pytest_hooks_closure_and_junit(
    observer, monkeypatch, tmp_path
):
    _guard_unit_profile(observer, monkeypatch)
    subject = tmp_path / "subject"
    tests = subject / "tests"
    tests.mkdir(parents=True)
    setup = "import pytest,sys\nassert 'jax' not in sys.modules\n@pytest.fixture\ndef skip_setup():\n    pytest.skip('setup reason')\ndef test_setup(skip_setup): pass\ndef test_pass(): pass\n"
    call = "import pytest,sys\nassert 'jax' not in sys.modules\ndef test_call():\n    pytest.skip('call reason')\n"
    (tests / "test_cases.py").write_text(setup)
    (tests / "test_call.py").write_text(call)
    ids = [
        "tests/test_call.py::test_call",
        "tests/test_cases.py::test_setup",
        "tests/test_cases.py::test_pass",
    ]
    authority = {
        "collection": {
            "count": 3,
            "ordered_ids_sha256": observer._digest(ids),
            "phase_count": 8,
        },
        "skips": [
            {"nodeid": ids[1], "phase": "setup", "reason": "setup reason", "line": 6},
            {"nodeid": ids[0], "phase": "call", "reason": "call reason", "line": 4},
        ],
    }
    auth = tmp_path / "authority.json"
    auth.write_text(json.dumps(authority))
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    environment = {
        **os.environ,
        "PYTHONPATH": str(_SCRIPT.parent),
        "JAX2ONNX_MAIN_EVIDENCE": str(evidence),
        "JAX2ONNX_MAIN_AUTHORITY": str(auth),
    }
    junit = evidence / "junit.xml"
    observer._guard(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "observe_main_ci_feasibility",
            "-n",
            "2",
            "--dist=loadscope",
            "--max-worker-restart=0",
            "--junitxml=" + str(junit),
            "-o",
            "junit_family=xunit1",
        ],
        subject,
        evidence,
        "pytest-smoke",
        20,
        environment,
    )
    record = json.loads((evidence / "pytest/controller.json").read_text())
    assert record["native_controller_exitstatus"] == 0 and not record["errors"]
    assert (
        observer._validate(
            authority,
            record["collections"],
            observer._events(evidence / "pytest/phases.jsonl"),
            record["closures"],
        )["phase_count"]
        == 8
    )
    observer._junit(junit, authority)


def test_sigterm_uses_the_same_owned_cleanup_and_final_record(
    observer, monkeypatch, tmp_path
):
    _guard_unit_profile(observer, monkeypatch)
    previous = signal.getsignal(signal.SIGTERM)
    code = "import os,signal,time; os.kill(os.getppid(), signal.SIGTERM); time.sleep(3)"
    with pytest.raises(ValueError):
        observer._guard(
            [sys.executable, "-c", code],
            tmp_path,
            tmp_path,
            "term",
            5,
            os.environ.copy(),
        )
    record = json.loads((tmp_path / "term-result.json").read_text())
    assert "SIGTERM" in record["stop"] and record["cleanup_complete"]
    assert signal.getsignal(signal.SIGTERM) == previous


def test_unexpected_cpu_allocation_is_rejected_without_affinity_rewrite(
    observer, monkeypatch, tmp_path
):
    monkeypatch.setattr(observer.os, "sched_getaffinity", lambda pid: set(range(8)))
    with pytest.raises(ValueError, match="CPU allocation"):
        observer._guard([], tmp_path, tmp_path, "cpu", 5, {})


@pytest.mark.parametrize("race", ("enumeration", "admission"))
def test_reused_parent_cannot_authorize_unrelated_children(observer, monkeypatch, race):
    guard = os.getpid()
    state = {"reused": False}
    owned = {100: 11}

    def stat(pid):
        if pid == guard:
            return _stat(1, 1)
        if pid == 100:
            return _stat(guard, 99 if state["reused"] else 11)
        if race == "admission":
            state["reused"] = True
        return _stat(100, 22)

    def children(pid):
        if pid == guard:
            return {100}
        if pid == 100:
            if race == "enumeration":
                state["reused"] = True
            return {101}
        return set()

    monkeypatch.setattr(observer, "_read_stat", stat)
    monkeypatch.setattr(observer, "_children", children)
    with pytest.raises(ValueError):
        observer._owned_tree(owned, 100)
    assert 101 not in owned
    calls = []
    monkeypatch.setattr(
        observer.os, "kill", lambda pid, number: calls.append((pid, number))
    )
    observer._signal_owned(
        owned, {100: _stat(guard, 11), 101: _stat(100, 22)}, signal.SIGTERM
    )
    assert calls == []


def test_disappearing_parent_requires_independent_guard_adoption(observer, monkeypatch):
    guard = os.getpid()
    state = {"gone": False}

    def stat(pid):
        if pid == guard:
            return _stat(1, 1)
        if pid == 100:
            if state["gone"]:
                raise FileNotFoundError(pid)
            return _stat(guard, 11)
        state["gone"] = True
        return _stat(guard, 22)

    def children(pid):
        if pid == guard:
            return {101} if state["gone"] else {100}
        return {101} if pid == 100 else set()

    def liveness(pid, number):
        if pid == 100:
            raise ProcessLookupError(pid)
        raise AssertionError("unexpected liveness check")

    monkeypatch.setattr(observer, "_read_stat", stat)
    monkeypatch.setattr(observer, "_children", children)
    monkeypatch.setattr(observer.os, "kill", liveness)
    owned = {100: 11}
    table = observer._owned_tree(owned, 100)
    assert owned[101] == 22 and table[101]["parent"] == guard
