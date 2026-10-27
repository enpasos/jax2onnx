# tests/extra_tests/framework/test_pytest_batches.py

"""Exercise fresh-process suite accounting with tiny, real pytest suites."""

import base64
import builtins
import csv
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest


_RUNNER = Path(__file__).resolve().parents[3] / "scripts" / "run_pytest_batches.py"


def _runner() -> Any:
    spec = spec_from_file_location("_pytest_batches_under_test", _RUNNER)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_suite(
    tmp_path: Path, files: dict[str, str], *, batch_size: int = 2
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any], Path]:
    suite = tmp_path / "suite"
    suite.mkdir()
    for name, content in files.items():
        destination = suite / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content)
    receipts = tmp_path / "receipts"
    result = subprocess.run(
        [
            sys.executable,
            str(_RUNNER),
            "--output-dir",
            str(receipts),
            "--batch-size",
            str(batch_size),
            "--test-path",
            str(suite),
            "--",
            "-q",
        ],
        cwd=suite,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    run = json.loads((receipts / "run.json").read_text())
    return result, run, receipts


def test_real_batches_preserve_parameter_ids_skip_xfail_and_collection_skip(
    tmp_path: Path,
) -> None:
    result, run, folder = _run_suite(
        tmp_path,
        {
            "test_cases.py": """import pytest
@pytest.mark.parametrize('value', [1, 2], ids=['with spaces', 'x::[β]'])
def test_parameter(value):
    assert value in (1, 2)
@pytest.mark.skip(reason='original skip')
def test_skip():
    pass
@pytest.mark.xfail(strict=True, reason='original xfail')
def test_xfail():
    assert False
""",
            "test_collection_skip.py": """import pytest
pytest.skip('original collection skip', allow_module_level=True)
""",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert run["status"] == "PASS"
    assert len(run["nodeids"]) == len(run["completed"]) == 4
    assert run["completed"] == run["nodeids"]
    assert any("x::[" in nodeid for nodeid in run["nodeids"])
    assert len(run["batches"]) == 2
    assert len({process["pid"] for process in run["processes"]}) == 3
    baseline = json.loads((folder / "collection" / "receipt.json").read_text())
    assert len(baseline["collection_skips"]) == 1
    reports = []
    for process in run["processes"]:
        directory = folder / process["folder"]
        assert process["returncode"] == 0
        assert process["output_complete"] is True
        assert (directory / "output.log").is_file()
        assert (directory / "junit.xml").is_file()
        assert "host_memory" in process["before"]
        assert "memory.events" in process["child_started"]["cgroup"]
        if process["kind"] == "batch":
            receipt = json.loads((directory / "receipt.json").read_text())
            assert receipt["full_preselection_ids"] == run["nodeids"]
            assert receipt["assigned_ids"] == receipt["selected_ids"]
            assert receipt["full_collection_checked"] is True
            assert (directory / "events.jsonl").is_file()
            reports.extend(
                report for phases in receipt["phases"].values() for report in phases
            )
    assert any("wasxfail" in report for report in reports)
    assert any(
        report["when"] == "setup" and report["outcome"] == "skipped"
        for report in reports
    )


def test_each_real_batch_preserves_unassigned_module_import_effects(
    tmp_path: Path,
) -> None:
    result, run, folder = _run_suite(
        tmp_path,
        {
            "test_a_consumer.py": """import builtins
def test_requires_complete_collection():
    assert builtins._jax2onnx_batch_import_marker == 'registered'
""",
            "test_z_registration.py": """import builtins
builtins._jax2onnx_batch_import_marker = 'registered'
def test_registration():
    pass
""",
        },
        batch_size=1,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert run["status"] == "PASS" and len(run["batches"]) == 2
    first = json.loads((folder / "batch-0001" / "receipt.json").read_text())
    assert len(first["full_preselection_ids"]) == 2
    assert first["selected_ids"] == [run["nodeids"][0]]


@pytest.mark.parametrize("stage", ("call", "setup", "teardown"))
def test_real_batches_fail_on_every_failing_phase(stage: str, tmp_path: Path) -> None:
    content = {
        "call": "def test_case():\n    assert False, 'call failure'\n",
        "setup": """import pytest
@pytest.fixture
def resource():
    assert False, 'setup failure'
def test_case(resource):
    pass
""",
        "teardown": """import pytest
@pytest.fixture
def resource():
    yield
    assert False, 'teardown failure'
def test_case(resource):
    pass
""",
    }[stage]
    result, run, folder = _run_suite(tmp_path, {"test_cases.py": content})
    assert result.returncode != 0
    assert run["status"] == "FAIL" and run["completed"] == []
    assert len(run["processes"]) == 2
    receipt = json.loads((folder / "batch-0001" / "receipt.json").read_text())
    assert any(
        report["when"] == stage and report["outcome"] == "failed"
        for phases in receipt["phases"].values()
        for report in phases
    )
    assert (folder / "batch-0001" / "junit.xml").is_file()


@pytest.mark.parametrize("fault", ("drift", "duplicate", "missing-completion"))
def test_real_batches_fail_closed_on_collection_or_completion_corruption(
    fault: str, tmp_path: Path
) -> None:
    plugin = {
        "drift": """import sys
def pytest_collection_modifyitems(items):
    if any('/batch-' in arg for arg in sys.argv):
        items[0]._nodeid += '-unexpected'
""",
        "duplicate": """def pytest_collection_modifyitems(items):
    items.append(items[0])
""",
        "missing-completion": """def pytest_runtest_protocol(item, nextitem):
    return True
""",
    }[fault]
    result, run, folder = _run_suite(
        tmp_path,
        {"test_cases.py": "def test_case():\n    pass\n", "conftest.py": plugin},
    )
    assert result.returncode != 0
    assert run["status"] == "FAIL" and run["completed"] == []
    if fault == "duplicate":
        assert len(run["processes"]) == 1
        receipt = json.loads((folder / "collection" / "receipt.json").read_text())
        assert len(receipt["full_preselection_ids"]) == 2
    else:
        assert len(run["processes"]) == 2
        receipt = json.loads((folder / "batch-0001" / "receipt.json").read_text())
        if fault == "drift":
            assert receipt["full_preselection_ids"][0].endswith("-unexpected")
            assert receipt["phases"] == {}
            assert receipt["full_collection_checked"] is False
        else:
            assert receipt["exitstatus"] == 0 and receipt["completed"] == []
            assert "missing" in run["failure"]["message"]


@pytest.mark.parametrize("termination", ("signal", "abrupt"))
def test_real_batches_retain_interrupted_and_nonzero_child_evidence(
    termination: str, tmp_path: Path
) -> None:
    statement = (
        "os.kill(os.getpid(), signal.SIGTERM)"
        if termination == "signal"
        else "os._exit(7)"
    )
    result, run, folder = _run_suite(
        tmp_path,
        {
            "test_cases.py": (
                "import os, signal\ndef test_case():\n    " + statement + "\n"
            )
        },
    )
    assert result.returncode != 0
    assert run["status"] == "FAIL" and run["completed"] == []
    assert len(run["processes"]) == 2
    process = run["processes"][-1]
    assert process["returncode"] == (2 if termination == "signal" else 7)
    assert "after" in process and process["finished"]
    receipt = json.loads((folder / "batch-0001" / "receipt.json").read_text())
    assert receipt["completed"] == []
    if termination == "signal":
        assert receipt["interrupted_signal"] == 15
    assert (folder / "batch-0001" / "events.jsonl").is_file()
    assert (folder / "batch-0001" / "output.log").is_file()


def test_unassigned_collection_drift_stops_before_assigned_execution(
    tmp_path: Path,
) -> None:
    result, run, folder = _run_suite(
        tmp_path,
        {
            "test_a_assigned.py": """from pathlib import Path
def test_first():
    Path('unexpected-execution').write_text('ran')
""",
            "test_z_unassigned.py": "def test_later():\n    pass\n",
            "conftest.py": """import sys
def pytest_collection_modifyitems(items):
    if any('/batch-' in arg for arg in sys.argv):
        items[-1]._nodeid += '-unassigned-drift'
""",
        },
        batch_size=1,
    )
    assert result.returncode != 0 and run["status"] == "FAIL"
    assert not (tmp_path / "suite" / "unexpected-execution").exists()
    receipt = json.loads((folder / "batch-0001" / "receipt.json").read_text())
    assert receipt["full_preselection_ids"][-1].endswith("-unassigned-drift")
    assert receipt["assigned_ids"] == run["batches"][0]
    assert receipt["selected_ids"] == [] and receipt["phases"] == {}


def test_parent_signal_bounds_shutdown_when_child_ignores_termination(
    tmp_path: Path,
) -> None:
    result, run, folder = _run_suite(
        tmp_path,
        {
            "test_cases.py": """import os, signal, time
def test_ignores_termination():
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    os.kill(os.getppid(), signal.SIGTERM)
    time.sleep(60)
"""
        },
    )
    assert result.returncode != 0 and run["status"] == "FAIL"
    process = run["processes"][-1]
    assert process["interrupted_signal"] == 15
    assert process["returncode"] == -9
    assert process["elapsed_seconds"] < 30
    assert run["completed"] == []
    assert (folder / "batch-0001" / "events.jsonl").is_file()


def test_unavailable_passive_disk_telemetry_does_not_abort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _runner()

    def unavailable(path: Path) -> Any:
        raise OSError("telemetry unavailable")

    monkeypatch.setattr(module.shutil, "disk_usage", unavailable)
    assert module._resources(None)["disk"] == {"unavailable": "OSError"}


def test_batch_plan_keeps_scopes_and_isolates_each_native_model_owner_item() -> None:
    module = _runner()
    items = []
    for name, count, class_name in (
        ("test_small.py", 2, None),
        ("test_large.py", 4, "ClassA"),
        ("test_large.py", 2, "ClassB"),
        ("test_source_model_native_bindings_exports.py", 3, None),
        ("test_last.py", 2, None),
    ):
        for index in range(count):
            items.append(
                {
                    "nodeid": f"{name}::{class_name}::{index}",
                    "module": name,
                    "class": class_name,
                }
            )
    batches = module._plan(items, 4)
    assert [nodeid for batch in batches for nodeid in batch] == [
        item["nodeid"] for item in items
    ]
    assert all(1 <= len(batch) <= 4 for batch in batches)
    assert [len(batch) for batch in batches] == [2, 4, 2, 1, 1, 1, 2]


@pytest.mark.parametrize("fault", ("duplicate-phase", "missing-phase", "receipt-drift"))
def test_phase_accounting_rejects_duplicate_missing_or_inconsistent_completion(
    fault: str,
) -> None:
    module = _runner()
    reports = [
        {"when": phase, "outcome": "passed"} for phase in ("setup", "call", "teardown")
    ]
    receipt = {
        "items": [{"nodeid": "test.py::case"}],
        "selected_ids": ["test.py::case"],
        "assigned_ids": ["test.py::case"],
        "full_collection_checked": True,
        "full_preselection_ids": ["test.py::case"],
        "collection_skips": [],
        "collection_errors": [],
        "exitstatus": 0,
        "phases": {"test.py::case": reports},
        "completed": ["test.py::case"],
    }
    if fault == "duplicate-phase":
        reports.append(reports[-1])
    elif fault == "missing-phase":
        reports.pop()
    else:
        receipt["completed"] = []
    with pytest.raises(ValueError, match="completed|completion"):
        module._validate(receipt, ["test.py::case"], ["test.py::case"], [])


def _cast_observer_module() -> Any:
    path = _RUNNER.with_name("_source_cast_observer.py")
    spec = spec_from_file_location("_cast_observer_under_test", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _observer_off_request(module: Any, original: dict[str, Any]) -> dict[str, Any]:
    payload = (json.dumps(original, indent=2, ensure_ascii=False) + "\n").encode()
    return {
        **original,
        "diagnostic_control": {
            "schema": "source-cast-observer-off-v1",
            "nodeid": module._CAST_OBSERVER_NODE,
            "original_request_sha256": hashlib.sha256(payload).hexdigest(),
            "original_receipt_sha256": "1" * 64,
            "original_observer_sha256": "2" * 64,
            "observer_enabled": False,
        },
    }


@pytest.mark.parametrize(
    "fault",
    (
        "extra",
        "schema",
        "node",
        "enabled",
        "digest",
        "uppercase",
        "request-drift",
        "kind",
    ),
)
def test_observer_off_descriptor_rejects_unknown_or_unbound_control(fault: str) -> None:
    module = _runner()
    original = {
        "kind": "batch",
        "assigned": [module._CAST_OBSERVER_NODE],
        "full_preselection_ids": [module._CAST_OBSERVER_NODE],
        "collection_skips": [],
        "pytest_args": ["-x"],
    }
    request = _observer_off_request(module, original)
    assert module._diagnostic_control(request) == request["diagnostic_control"]
    if fault == "extra":
        request["diagnostic_control"]["extra"] = False
    elif fault == "schema":
        request["diagnostic_control"]["schema"] = "another-version"
    elif fault == "node":
        request["diagnostic_control"]["nodeid"] = "another-node"
    elif fault == "enabled":
        request["diagnostic_control"]["observer_enabled"] = 0
    elif fault == "digest":
        request["diagnostic_control"]["original_receipt_sha256"] = "bad"
    elif fault == "uppercase":
        request["diagnostic_control"]["original_observer_sha256"] = "A" * 64
    elif fault == "kind":
        request["kind"] = "collection"
    else:
        request["pytest_args"].append("changed")
    with pytest.raises(ValueError, match="observer-off"):
        module._diagnostic_control(request)


def test_worker_context_environment_is_bounded_without_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _runner()
    monkeypatch.setenv("JAX2ONNX_MAXTEXT_SRC", "/fixed/discovery")
    context = module._worker_context()
    assert context["environment"]["JAX2ONNX_MAXTEXT_SRC"] == "/fixed/discovery"
    assert "PYTHONHASHSEED" in context["environment"]
    monkeypatch.setenv("XLA_FLAGS", "x" * 8193)
    oversized = module._worker_context()
    assert oversized["status"] == "UNVERIFIED"
    assert oversized["environment"] == {}
    assert {"stage": "environment", "error_type": "ValueError"} in oversized["errors"]


def test_real_worker_separates_startup_environment_from_full_collection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("JAX_ENABLE_X64", raising=False)
    _, _, folder = _run_suite(
        tmp_path,
        {
            "conftest.py": "import os\nos.environ['JAX_PLATFORMS']='cpu'\nos.environ['JAX_ENABLE_X64']='True'\n",
            "test_case.py": "def test_case():\n    pass\n",
        },
    )
    receipt = json.loads((folder / "batch-0001/receipt.json").read_text())
    assert receipt["startup_context"]["environment"]["JAX_PLATFORMS"] is None
    assert receipt["startup_context"]["environment"]["JAX_ENABLE_X64"] is None
    assert receipt["startup_context"]["loaded_jax_config"]["status"] == "NOT_LOADED"
    assert receipt["collection_context"]["environment"]["JAX_PLATFORMS"] == "cpu"
    assert receipt["collection_context"]["environment"]["JAX_ENABLE_X64"] == "True"
    assert receipt["supplemental"] is False and receipt["cast_observer_enabled"] is True
    assert "diagnostic_control" not in json.loads(
        (folder / "batch-0001/request.json").read_text()
    )


@pytest.mark.parametrize("original_pass", (False, True))
def test_real_observer_off_control_never_imports_observer_or_validates_canonical(
    original_pass: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copied = tmp_path / "runner"
    copied.mkdir()
    runner = copied / "run_pytest_batches.py"
    runner.write_text(_RUNNER.read_text())
    marker = copied / "observer-imported"
    (copied / "_source_cast_observer.py").write_text(
        "from pathlib import Path\nPath(__file__).with_name('observer-imported').write_text('yes')\n"
        "def start_observation(*args):\n    raise RuntimeError('original observer sentinel')\n"
    )
    monkeypatch.setitem(globals(), "_RUNNER", runner)
    owner = "tests/extra_tests/converter/test_source_conversion_semantics.py"
    content = (
        "import pytest\nfrom pathlib import Path\n"
        "@pytest.mark.parametrize('source',['float64'])\n"
        "@pytest.mark.parametrize('target',['float16'])\n"
        "@pytest.mark.parametrize('group',['ties'])\n"
        "def test_floating_conversion_rounding_and_strict_special_value_gates(source,target,group):\n"
        "    p=Path('original-calls')\n    p.write_text(p.read_text()+'x' if p.exists() else 'x')\n"
        + (
            "    assert False, 'unchanged original failure'\n"
            if not original_pass
            else ""
        )
    )
    result, run, folder = _run_suite(tmp_path, {owner: content})
    assert marker.exists()
    marker.unlink()
    module = _runner()
    original = json.loads((folder / "batch-0001/request.json").read_text())
    request = _observer_off_request(module, original)
    control = tmp_path / "control"
    control.mkdir()
    (control / "request.json").write_text(
        json.dumps(request, indent=2, ensure_ascii=False) + "\n"
    )
    child = subprocess.run(
        [sys.executable, str(runner), "--worker", str(control / "request.json")],
        cwd=tmp_path / "suite",
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    receipt = json.loads((control / "receipt.json").read_text())
    assert not marker.exists()
    assert child.returncode == result.returncode
    assert (tmp_path / "suite/original-calls").read_text() == "xx"
    assert (
        receipt["supplemental"] is True and receipt["native_source_recomputed"] is True
    )
    assert receipt["cast_observer_enabled"] is False
    for name in ("target_context_before", "target_context_after"):
        assert receipt[name]["sys_profile_is_none"] is True
        assert receipt[name]["threading_profile_is_none"] is True
    with pytest.raises(ValueError, match="supplemental"):
        module._validate(
            receipt,
            original["assigned"],
            original["full_preselection_ids"],
            original["collection_skips"],
        )


def test_observer_off_control_never_clears_an_existing_profile(
    tmp_path: Path,
) -> None:
    module = _runner()
    suite = tmp_path / "suite"
    owner = suite / "tests/extra_tests/converter/test_source_conversion_semantics.py"
    owner.parent.mkdir(parents=True)
    owner.write_text(
        "import pytest\nfrom pathlib import Path\n"
        "@pytest.mark.parametrize('source',['float64'])\n"
        "@pytest.mark.parametrize('target',['float16'])\n"
        "@pytest.mark.parametrize('group',['ties'])\n"
        "def test_floating_conversion_rounding_and_strict_special_value_gates(source,target,group):\n"
        "    Path('unexpected-source-call').write_text('ran')\n"
    )
    (suite / "conftest.py").write_text(
        "import sys,pytest\nfrom pathlib import Path\n"
        "def PROFILE(*args):\n    return None\n"
        "@pytest.fixture(autouse=True)\n"
        "def active_profile():\n"
        "    before=sys.getprofile()\n    sys.setprofile(PROFILE)\n"
        "    try:\n        yield\n"
        "    finally:\n"
        "        assert sys.getprofile() is PROFILE\n"
        "        Path('profile-preserved').write_text('yes')\n"
        "        sys.setprofile(before)\n"
    )
    original = {
        "kind": "batch",
        "assigned": [module._CAST_OBSERVER_NODE],
        "full_preselection_ids": [module._CAST_OBSERVER_NODE],
        "collection_skips": [],
        "pytest_args": ["-q", str(suite)],
    }
    folder = tmp_path / "control"
    folder.mkdir()
    (folder / "request.json").write_text(
        json.dumps(
            _observer_off_request(module, original), indent=2, ensure_ascii=False
        )
        + "\n"
    )
    result = subprocess.run(
        [sys.executable, str(_RUNNER), "--worker", str(folder / "request.json")],
        cwd=suite,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode != 0
    assert not (suite / "unexpected-source-call").exists()
    assert (suite / "profile-preserved").read_text() == "yes"
    receipt = json.loads((folder / "receipt.json").read_text())
    assert receipt["native_source_recomputed"] is False
    assert receipt["target_context_before"]["sys_profile_is_none"] is False
    assert receipt["target_context_after"]["sys_profile_is_none"] is False


def test_child_explicit_environment_does_not_mutate_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    module = _runner()
    monkeypatch.setenv("JAX2ONNX_CONTROL_ENV_TEST", "parent")
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "test_case.py").write_text(
        "import os\nassert os.environ['JAX2ONNX_CONTROL_ENV_TEST']=='child'\n"
        "def test_case():\n    pass\n"
    )
    environment = dict(os.environ, JAX2ONNX_CONTROL_ENV_TEST="child")
    request = {
        "kind": "collection",
        "pytest_args": [str(suite), "--collect-only", "-q"],
    }
    run = {"processes": [], "status": "RUNNING"}
    assert (
        module._child(
            tmp_path / "child", request, run, tmp_path, environment=environment
        )
        == 0
    )
    assert os.environ["JAX2ONNX_CONTROL_ENV_TEST"] == "parent"
    receipt = json.loads((tmp_path / "child/receipt.json").read_text())
    assert (
        receipt["startup_context"]["environment"]["JAX2ONNX_CONTROL_ENV_TEST"]
        == "child"
    )


def test_child_diagnostic_timeout_is_terminal_without_retry(tmp_path: Path) -> None:
    module = _runner()
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "test_case.py").write_text(
        "import time\ntime.sleep(60)\ndef test_case():\n    pass\n"
    )
    request = {
        "kind": "collection",
        "pytest_args": [str(suite), "--collect-only", "-q"],
    }
    run = {"processes": [], "status": "RUNNING"}
    with pytest.raises(TimeoutError, match="fixed timeout"):
        module._child(tmp_path / "child", request, run, tmp_path, timeout_seconds=0.2)
    assert run["status"] == "FAIL" and len(run["processes"]) == 1
    assert run["processes"][0]["diagnostic_timeout_seconds"] == 0.2
    assert run["processes"][0]["returncode"] is not None


@pytest.mark.parametrize(
    "diagnostic,original_pass",
    (
        ("start-error", False),
        ("start-error", True),
        ("finish-error", False),
        ("finish-error", True),
        ("preexisting-profile", False),
        ("unmatched-node", True),
    ),
)
def test_cast_diagnostic_preserves_one_original_call_and_pytest_outcome(
    diagnostic: str,
    original_pass: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied = tmp_path / "runner"
    copied.mkdir()
    runner = copied / "run_pytest_batches.py"
    runner.write_text(_RUNNER.read_text())
    imported = copied / "observer-imported"
    observer = (
        "from pathlib import Path\n"
        "Path(__file__).with_name('observer-imported').write_text('imported')\n"
    )
    if diagnostic == "start-error":
        observer += (
            "def start_observation(*args):\n"
            "    raise RuntimeError('diagnostic setup sentinel')\n"
        )
    elif diagnostic == "finish-error":
        observer += (
            "class Observation:\n"
            "    def finish(self):\n"
            "        raise RuntimeError('diagnostic finish sentinel')\n"
            "def start_observation(*args):\n"
            "    return Observation()\n"
        )
    else:
        observer += "raise AssertionError('unselected diagnostic was imported')\n"
    (copied / "_source_cast_observer.py").write_text(observer)
    monkeypatch.setitem(globals(), "_RUNNER", runner)
    if diagnostic == "unmatched-node":
        owner = "test_cases.py"
        content = (
            "from pathlib import Path\n"
            "def test_once():\n"
            "    path = Path('original-calls')\n"
            "    path.write_text(path.read_text() + 'x' if path.exists() else 'x')\n"
        )
    else:
        owner = "tests/extra_tests/converter/test_source_conversion_semantics.py"
        content = (
            "from pathlib import Path\n"
            "import pytest\n"
            "@pytest.mark.parametrize('source', ['float64'])\n"
            "@pytest.mark.parametrize('target', ['float16'])\n"
            "@pytest.mark.parametrize('group', ['ties'])\n"
            "def test_floating_conversion_rounding_and_strict_special_value_gates(source, target, group):\n"
            "    path = Path('original-calls')\n"
            "    path.write_text(path.read_text() + 'x' if path.exists() else 'x')\n"
        )
        if diagnostic == "preexisting-profile":
            content += (
                "    import sys, conftest\n"
                "    assert sys.getprofile() is conftest.PROFILE\n"
            )
        if not original_pass:
            content += "    assert False, 'original failure sentinel'\n"
    files = {owner: content}
    if diagnostic == "preexisting-profile":
        files["conftest.py"] = (
            "import sys, pytest\n"
            "def PROFILE(*args):\n"
            "    return None\n"
            "@pytest.fixture(autouse=True)\n"
            "def existing_profile():\n"
            "    before = sys.getprofile()\n"
            "    sys.setprofile(PROFILE)\n"
            "    try:\n"
            "        yield\n"
            "    finally:\n"
            "        sys.setprofile(before)\n"
        )
    result, run, folder = _run_suite(tmp_path, files)
    assert (tmp_path / "suite" / "original-calls").read_text() == "x"
    receipt = json.loads((folder / "batch-0001" / "receipt.json").read_text())
    assert len(receipt["phases"]) == 1
    reports = next(iter(receipt["phases"].values()))
    assert [report["when"] for report in reports] == ["setup", "call", "teardown"]
    if original_pass:
        assert result.returncode == 0 and run["status"] == "PASS"
        assert all(report["outcome"] == "passed" for report in reports)
    else:
        assert result.returncode != 0 and run["status"] == "FAIL"
        assert reports[1]["outcome"] == "failed"
        assert "original failure sentinel" in reports[1]["longrepr"]
    if diagnostic == "unmatched-node":
        assert not imported.exists()
        assert not (
            folder / "batch-0001" / "source-f64-f16-cast-observation.json"
        ).exists()
    else:
        observation = json.loads(
            (folder / "batch-0001" / "source-f64-f16-cast-observation.json").read_text()
        )
        assert observation["status"] == "UNVERIFIED"
        expected = {
            "start-error": "observer-start-error",
            "finish-error": "observer-finish-error",
            "preexisting-profile": "preexisting-profile-preserved",
        }
        assert observation["reason"] == expected[diagnostic]
        assert imported.exists() is (diagnostic != "preexisting-profile")


@pytest.mark.parametrize("original_pass", (False, True))
@pytest.mark.parametrize("export_fault", ("none", "serialization", "durability"))
def test_executable_export_follows_durable_original_report_and_preserves_outcome(
    original_pass: bool,
    export_fault: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied = tmp_path / "runner"
    copied.mkdir()
    runner = copied / "run_pytest_batches.py"
    runner.write_text(_RUNNER.read_text())
    observer = (
        "import json, os, sys\nfrom pathlib import Path\n"
        "class Observation:\n"
        "    def __init__(self, folder):\n"
        "        self.folder = folder\n"
        "        self.fsync = os.fsync\n"
        "    def finish(self):\n"
        "        assert sys.getprofile() is None\n"
    )
    if export_fault == "durability":
        observer += (
            "        def fail(fd):\n"
            "            raise RuntimeError('durability sentinel')\n"
            "        os.fsync = fail\n"
        )
    observer += (
        "    def after_report(self, phase):\n"
        "        assert sys.getprofile() is None\n"
        "        receipt = json.loads((self.folder / 'receipt.json').read_text())\n"
        "        entries = receipt['phases'][phase['nodeid']]\n"
        "        assert [entry['when'] for entry in entries] == ['setup', 'call']\n"
        "        assert entries[-1]['outcome'] == phase['outcome']\n"
        "        events = [json.loads(line) for line in (self.folder / 'events.jsonl').read_text().splitlines()]\n"
        "        assert events[-1]['kind'] == 'phase' and events[-1]['when'] == 'call'\n"
        "        assert events[-1]['outcome'] == phase['outcome']\n"
        "        assert phase['durable_phase_recorded'] is True\n"
        "        Path('export-after-report').write_text(phase['outcome'])\n"
    )
    if export_fault == "serialization":
        observer += "        raise RuntimeError('serialization sentinel')\n"
    observer += (
        "    def close(self):\n"
        "        os.fsync = self.fsync\n"
        "        Path('capture-closed').write_text('yes')\n"
        "def start_observation(folder, *args):\n"
        "    return Observation(folder)\n"
    )
    (copied / "_source_cast_observer.py").write_text(observer)
    monkeypatch.setitem(globals(), "_RUNNER", runner)
    owner = "tests/extra_tests/converter/test_source_conversion_semantics.py"
    source = (
        "from pathlib import Path\nimport pytest\n"
        "@pytest.mark.parametrize('source', ['float64'])\n"
        "@pytest.mark.parametrize('target', ['float16'])\n"
        "@pytest.mark.parametrize('group', ['ties'])\n"
        "def test_floating_conversion_rounding_and_strict_special_value_gates(source, target, group):\n"
        "    path = Path('original-calls')\n"
        "    assert not path.exists()\n"
        "    path.write_text('once')\n"
    )
    if not original_pass:
        source += "    assert False, 'unchanged original failure sentinel'\n"
    result, run, folder = _run_suite(tmp_path, {owner: source})
    root = tmp_path / "suite"
    assert (root / "original-calls").read_text() == "once"
    assert (root / "capture-closed").read_text() == "yes"
    expected = "passed" if original_pass else "failed"
    if export_fault == "durability":
        assert not (root / "export-after-report").exists()
    else:
        assert (root / "export-after-report").read_text() == expected
    receipt = json.loads((folder / "batch-0001" / "receipt.json").read_text())
    phases = next(iter(receipt["phases"].values()))
    assert [phase["when"] for phase in phases] == ["setup", "call", "teardown"]
    assert [phase["outcome"] for phase in phases] == ["passed", expected, "passed"]
    assert (result.returncode == 0) is original_pass
    assert run["status"] == ("PASS" if original_pass else "FAIL")
    if not original_pass:
        assert "unchanged original failure sentinel" in phases[1]["longrepr"]
    if export_fault != "none":
        assert receipt["cast_executable_capture"]["status"] == "UNVERIFIED"


@pytest.mark.parametrize("fault", ("close", "write"))
def test_optional_executable_io_failure_preserves_complete_original_word_proof(
    fault: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _cast_observer_module()
    observation = module._Observation(tmp_path, module.NODE_ID, type(sys)("owner"))
    calls = []

    class Capture:
        def __init__(self) -> None:
            self.report = {"status": "UNVERIFIED", "native_owners_cleared": False}

        def after_report(self, phase: Any) -> None:
            calls.append("export")
            self.report["status"] = "ORIGINAL_EXECUTABLES_SERIALIZED"

        def close(self) -> None:
            self.report["native_owners_cleared"] = True
            calls.append("close")
            if fault == "close":
                raise OSError("optional close report I/O sentinel")

        def _write(self) -> None:
            calls.append("write")
            if fault == "write":
                raise OSError("optional final report I/O sentinel")

        def _fail(self, stage: str, error: BaseException) -> None:
            self.report["status"] = "UNVERIFIED"

    capture = Capture()
    observation.executable_capture = capture
    observation.report.update(
        status="ORIGINAL_CALL_WORDS_OBSERVED",
        executable_capture={"status": "UNVERIFIED"},
    )

    def fences() -> None:
        calls.append("original-fences")
        for name in (
            "previous_profile_restored",
            "native_context_restored",
            "source_settings_restored",
            "source_fence_held",
            "code_identity_fence_held",
        ):
            observation.report[name] = True

    monkeypatch.setattr(observation, "_end_fences", fences)
    observation.after_report({"original": "already recorded"})
    assert calls == ["original-fences", "export", "close", "original-fences", "write"]
    assert observation.executable_capture is None
    assert capture.report["native_owners_cleared"] is True
    assert capture.report["original_fences_after_serialization"] is True
    assert capture.report["status"] == "UNVERIFIED"
    assert observation.report["status"] == "ORIGINAL_CALL_WORDS_OBSERVED"
    assert observation.report["observer_errors"] == []
    assert observation.report["executable_capture"]["status"] == "UNVERIFIED"


def test_cast_observer_preexisting_profile_skips_before_native_or_source_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _cast_observer_module()
    existing = object()
    calls = []
    monkeypatch.setattr(module.sys, "getprofile", lambda: existing)
    monkeypatch.setattr(module.sys, "setprofile", lambda value: calls.append(value))

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("skipped diagnostic performed source/runtime work")

    monkeypatch.setattr(module, "_file", unexpected)
    monkeypatch.setattr(module.importlib.metadata, "distribution", unexpected)
    owner = type(sys)("unloaded-owner")
    observation = module.start_observation(tmp_path, module.NODE_ID, owner)
    observation.finish()
    report = json.loads((tmp_path / module.REPORT_NAME).read_text())
    assert report["status"] == "UNVERIFIED"
    assert report["reason"] == "preexisting-profile-preserved"
    assert report["artifact_events"] == report["runtime_events"] == []
    assert report["comparison_events"] == [] and calls == []


def test_cast_observer_preserves_profile_installed_by_original_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _cast_observer_module()
    observation = module._Observation(tmp_path, module.NODE_ID, type(sys)("owner"))
    installed = object()
    calls = []
    monkeypatch.setattr(module.sys, "getprofile", lambda: installed)
    monkeypatch.setattr(module.sys, "setprofile", lambda value: calls.append(value))
    monkeypatch.setattr(module, "_SOURCE_FILES", ())
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "test")
    monkeypatch.setattr(observation, "_native_modules", lambda: {})
    monkeypatch.setattr(observation, "_installed_native_files", lambda: None)
    observation.active = True
    observation.compat = type(sys)("compat")
    observation.compat.require_native_context_restored = lambda context: None
    monkeypatch.setattr(observation, "_settings", lambda: None)
    observation.report["artifact_events"] = [{"existing": "artifact"}]
    observation.report["runtime_events"] = [{"existing": "runtime"}]
    observation.report["comparison_events"] = [{"existing": "comparison"}]
    observation.finish()
    report = json.loads((tmp_path / module.REPORT_NAME).read_text())
    assert calls == [] and module.sys.getprofile() is installed
    assert report["status"] == "UNVERIFIED"
    assert report["reason"] == "original-call-changed-profile-preserved"
    assert report["previous_profile_restored"] is False


def _native_module_observation(
    module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, dict[str, Any], list[str]]:
    observation = module._Observation(tmp_path, module.NODE_ID, type(sys)("owner"))
    modules = {
        name: type(sys)(name) for name in ("jax", "jaxlib", "numpy", "onnxruntime")
    }
    observed_sys = type(sys)("observed-sys")
    observed_sys.modules = modules
    monkeypatch.setattr(module, "sys", observed_sys)
    checked = []

    def proof(name: str) -> dict[str, Any]:
        checked.append(name)
        return {"version": "test", "record_digest_verified": True}

    monkeypatch.setattr(observation, "_native_module", proof)
    return observation, modules, checked


@pytest.mark.parametrize("scipy_loaded", (False, True))
def test_cast_observer_records_optional_module_presence_without_importing(
    scipy_loaded: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _cast_observer_module()
    observation, modules, checked = _native_module_observation(
        module, tmp_path, monkeypatch
    )
    if scipy_loaded:
        modules["scipy"] = type(sys)("scipy")
    original_names = tuple(modules)
    result = observation._native_modules()
    snapshot = observation.report["runtime_module_snapshot"]
    assert tuple(modules) == original_names
    assert snapshot["scipy"]["sys_modules_contains_name"] is scipy_loaded
    assert snapshot["scipy"]["exact_module_type"] is scipy_loaded
    assert ("scipy" in checked) is scipy_loaded
    if scipy_loaded:
        assert result["scipy"]["status"] == "LOADED"
        assert result["scipy"]["record_digest_verified"] is True
    else:
        assert snapshot["scipy"]["binding_identity"] is None
        assert result["scipy"] == {
            "status": "NOT_LOADED",
            "record_digest_verified": False,
            "wheel_archive_hash_claimed": False,
        }
    for name in ("jax", "jaxlib", "numpy", "onnxruntime"):
        assert snapshot[name]["binding_identity"] == id(modules[name])
        assert result[name]["status"] == "LOADED"
        assert result[name]["record_digest_verified"] is True


@pytest.mark.parametrize("missing", ("jax", "jaxlib", "numpy", "onnxruntime"))
def test_cast_observer_requires_core_loaded_module_provenance(
    missing: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _cast_observer_module()
    observation, modules, checked = _native_module_observation(
        module, tmp_path, monkeypatch
    )
    del modules[missing]
    with pytest.raises(ValueError, match="runtime module was not loaded: " + missing):
        observation._native_modules()
    assert missing not in checked
    snapshot = observation.report["runtime_module_snapshot"]
    assert set(snapshot) == {"jax", "jaxlib", "numpy", "scipy", "onnxruntime"}
    assert snapshot[missing]["sys_modules_contains_name"] is False
    assert observation.report["loaded_native_modules"] is not None


@pytest.mark.parametrize("binding_kind", ("none", "foreign"))
def test_cast_observer_rejects_non_module_optional_binding_without_coercion(
    binding_kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _cast_observer_module()
    observation, modules, checked = _native_module_observation(
        module, tmp_path, monkeypatch
    )
    calls = []

    class Foreign:
        def __bool__(self) -> bool:
            calls.append("bool")
            raise AssertionError("diagnostic coerced a foreign module")

        def __str__(self) -> str:
            calls.append("str")
            raise AssertionError("diagnostic rendered a foreign module")

        def __eq__(self, other: Any) -> bool:
            calls.append("eq")
            raise AssertionError("diagnostic compared a foreign module")

    modules["scipy"] = None if binding_kind == "none" else Foreign()
    with pytest.raises(ValueError, match="runtime module was not loaded: scipy"):
        observation._native_modules()
    assert calls == [] and "scipy" not in checked
    snapshot = observation.report["runtime_module_snapshot"]["scipy"]
    assert snapshot["sys_modules_contains_name"] is True
    assert snapshot["exact_module_type"] is False


def _installed_manifest_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, Any, dict[str, Any], list[Any]]:
    module = _cast_observer_module()
    site = tmp_path / "site"
    site.mkdir()
    distributions = {}
    records = {}
    for name in module._NATIVE_DISTRIBUTIONS:
        package = name.replace("-", "_")
        info = site / f"{package}-1.0.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(f"Name: {name}\nVersion: 1.0\n\n")
        paths = [f"{package}/native.so"]
        if name == "jaxlib":
            paths.append(f"{package}/libkernel.so.2.0")
        if name == "numpy":
            paths.append("numpy.libs/libsupport.so.1")
        rows = []
        for relative in paths:
            path = site / relative
            path.parent.mkdir(exist_ok=True)
            payload = b"not executed: " + relative.encode()
            path.write_bytes(payload)
            digest = (
                base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
                .rstrip(b"=")
                .decode()
            )
            rows.append([relative, "sha256=" + digest, str(len(payload))])
        with (info / "RECORD").open("w", newline="") as stream:
            csv.writer(stream).writerows(rows)
        distributions[name] = module.importlib.metadata.PathDistribution(info)
        records[name] = {"info": info, "rows": rows, "site": site}
    monkeypatch.setattr(
        module.importlib.metadata, "distribution", distributions.__getitem__
    )
    observation = module._Observation(tmp_path, module.NODE_ID, type(sys)("owner"))
    observation.report["artifact_events"] = [{"original": "words"}]
    checkpoints = []
    original_write = observation.write

    def write() -> None:
        checkpoints.append(json.loads(json.dumps(observation.report)))
        original_write()

    monkeypatch.setattr(observation, "write", write)
    return module, observation, records, checkpoints


def test_installed_native_manifest_verifies_real_record_files_without_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, observation, records, checkpoints = _installed_manifest_fixture(
        tmp_path, monkeypatch
    )
    imported = []
    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in ("jax", "jaxlib", "numpy", "onnxruntime", "ml_dtypes"):
            imported.append(name)
            raise AssertionError("installed file proof imported a framework")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    observation._installed_native_files()
    manifest = observation.report["installed_native_files"]
    assert imported == [] and manifest["status"] == "VERIFIED"
    assert manifest["evidence_kind"] == "INSTALLED_RECORD_FILE"
    assert all(
        manifest[name] is False
        for name in (
            "native_modules_imported_by_observer",
            "loaded_native_binary_claimed",
            "executed_kernel_claimed",
            "wheel_archive_hash_claimed",
        )
    )
    assert manifest["files_started"] == 6
    assert manifest["declared_bytes"] == manifest["bytes_hashed"]
    assert observation.report["artifact_events"] == [{"original": "words"}]
    assert checkpoints and any(
        p["installed_native_files"]["status"] == "UNVERIFIED" for p in checkpoints
    )
    for name, item in manifest["distributions"].items():
        assert item["version"] == "1.0" and item["status"] == "VERIFIED"
        for key in ("metadata_file", "record_file"):
            assert (
                item[key]["stat_before"]
                == item[key]["stat_after"]
                == item[key]["stat_end"]
            )
        assert len(item["files"]) == len(records[name]["rows"])
        for proof in item["files"]:
            assert proof["status"] == "VERIFIED"
            assert (
                proof["record_digest_verified"] is proof["record_size_verified"] is True
            )
            assert proof["stat_before"] == proof["stat_after"]
            assert proof["sha256"] == proof["record_declared_sha256"]


@pytest.mark.parametrize(
    "fault",
    (
        "file-digest",
        "record-digest",
        "record-size",
        "duplicate",
        "escape",
        "symlink",
        "directory",
    ),
)
def test_installed_native_manifest_keeps_partial_proofs_on_corruption(
    fault: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, observation, records, checkpoints = _installed_manifest_fixture(
        tmp_path, monkeypatch
    )
    record = records["jaxlib"]
    rows = record["rows"]
    path = record["site"] / rows[1][0]
    if fault == "file-digest":
        path.write_bytes(b"x" * path.stat().st_size)
    elif fault == "record-digest":
        rows[1][1] = "sha256=" + "A" * 43
    elif fault == "record-size":
        rows[1][2] = str(path.stat().st_size + 1)
    elif fault == "duplicate":
        rows[1] = rows[0][:]
    elif fault == "escape":
        rows[1][0] = "../escape.so"
    else:
        path.unlink()
        if fault == "symlink":
            path.symlink_to(record["site"] / rows[0][0])
        else:
            path.mkdir()
    with (record["info"] / "RECORD").open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)
    with pytest.raises(ValueError):
        observation._installed_native_files()
    manifest = observation.report["installed_native_files"]
    assert manifest["status"] == "UNVERIFIED"
    assert manifest["distributions"]["jaxlib"]["files"][0]["status"] == "VERIFIED"
    assert manifest["distributions"]["jaxlib"]["error"]
    assert checkpoints[-1]["artifact_events"] == [{"original": "words"}]


@pytest.mark.parametrize("foreign_role", ("distribution", "path"))
def test_installed_native_manifest_rejects_foreign_protocols(
    foreign_role: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, observation, _, _ = _installed_manifest_fixture(tmp_path, monkeypatch)
    calls = []

    class Foreign:
        def __str__(self) -> str:
            calls.append("str")
            raise AssertionError("rendered foreign distribution metadata")

        def __fspath__(self) -> str:
            calls.append("fspath")
            raise AssertionError("coerced foreign distribution path")

    foreign = Foreign()
    dist = (
        foreign
        if foreign_role == "distribution"
        else module.importlib.metadata.PathDistribution(foreign)
    )
    monkeypatch.setattr(module.importlib.metadata, "distribution", lambda name: dist)
    with pytest.raises(ValueError, match="exact"):
        observation._installed_native_files()
    assert calls == []
    assert observation.report["installed_native_files"]["status"] == "UNVERIFIED"


@pytest.mark.parametrize("mutation", ("inode", "stat", "record"))
def test_installed_native_manifest_streams_and_fences_live_file_changes(
    mutation: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, observation, records, checkpoints = _installed_manifest_fixture(
        tmp_path, monkeypatch
    )
    monkeypatch.setattr(module, "_NATIVE_CHUNK_BYTES", 3)
    monkeypatch.setattr(module, "_NATIVE_PROGRESS_BYTES", 3)
    path = records["jaxlib"]["site"] / records["jaxlib"]["rows"][0][0]
    original_read = module.os.read
    changed = False

    def read(descriptor: int, count: int) -> bytes:
        nonlocal changed
        assert count <= 3
        chunk = original_read(descriptor, count)
        proof = observation.report["installed_native_files"]["distributions"]["jaxlib"]
        if chunk and proof["files"] and not changed:
            changed = True
            if mutation == "inode":
                replacement = path.with_name("replacement")
                replacement.write_bytes(path.read_bytes())
                replacement.replace(path)
            else:
                target = (
                    path if mutation == "stat" else records["jaxlib"]["info"] / "RECORD"
                )
                current = target.stat()
                module.os.utime(
                    target, ns=(current.st_atime_ns, current.st_mtime_ns + 1)
                )
        return chunk

    observed_os = type(sys)("observed-os")
    vars(observed_os).update(vars(module.os))
    observed_os.read = read
    monkeypatch.setattr(module, "os", observed_os)
    with pytest.raises(ValueError, match="changed"):
        observation._installed_native_files()
    assert changed
    assert any(
        0 < state["installed_native_files"]["bytes_hashed"] < path.stat().st_size
        for state in checkpoints
    )
    assert observation.report["installed_native_files"]["status"] == "UNVERIFIED"


@pytest.mark.parametrize(
    "bound,value",
    (
        ("_MAX_NATIVE_RECORD_BYTES", 1),
        ("_MAX_NATIVE_RECORD_ROWS", 1),
        ("_MAX_NATIVE_FILES", 1),
        ("_MAX_NATIVE_FILE_BYTES", 1),
        ("_MAX_NATIVE_TOTAL_BYTES", 1),
        ("_MAX_NATIVE_MANIFEST_BYTES", 4096),
    ),
)
def test_installed_native_manifest_enforces_finite_bounds(
    bound: str, value: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, observation, _, checkpoints = _installed_manifest_fixture(
        tmp_path, monkeypatch
    )
    monkeypatch.setattr(module, bound, value)
    with pytest.raises(ValueError, match="bound"):
        observation._installed_native_files()
    assert observation.report["installed_native_files"]["status"] == "UNVERIFIED"
    assert observation.report["artifact_events"] == [{"original": "words"}]
    assert checkpoints


@pytest.mark.parametrize(
    "change", ("unchanged", "source", "code", "settings", "context", "profile", "error")
)
def test_cast_observer_rechecks_original_fences_after_installed_audit(
    change: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _cast_observer_module()
    observation = module._Observation(tmp_path, module.NODE_ID, type(sys)("owner"))
    profile = {"value": observation.callback}
    monkeypatch.setattr(module.sys, "getprofile", lambda: profile["value"])
    monkeypatch.setattr(
        module.sys, "setprofile", lambda value: profile.update(value=value)
    )
    monkeypatch.setattr(module, "_ROOT", tmp_path)
    monkeypatch.setattr(module, "_SOURCE_FILES", ("original.py",))
    source = tmp_path / "original.py"
    source.write_text("before")
    observation.before_files = {"original.py": module._file(source)}
    observation.active = True
    observation.compat = type(sys)("compat")
    context_state = {"restored": True}

    def require_context(context: Any) -> None:
        if not context_state["restored"]:
            raise ValueError("original native context changed")

    observation.compat.require_native_context_restored = require_context
    state = {"value": "before"}
    observation.settings = "before"
    monkeypatch.setattr(observation, "_settings", lambda: state["value"])
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "test")
    monkeypatch.setattr(observation, "_native_modules", lambda: {})

    def original() -> None:
        return None

    observation.functions = {"original": original}
    observation.codes = {"original": original.__code__}
    for name in ("artifact_events", "runtime_events", "comparison_events"):
        observation.report[name] = [{"original": "words"}]

    def audit() -> None:
        if change in ("source", "error"):
            source.write_text("after")
        elif change == "settings":
            state["value"] = "after"
        elif change == "profile":
            profile["value"] = object()
        elif change == "context":
            context_state["restored"] = False
        elif change == "code":
            original.__code__ = (lambda: 1).__code__
        if change == "error":
            raise ValueError("installed audit sentinel")

    monkeypatch.setattr(observation, "_installed_native_files", audit)
    observation.finish()
    assert observation.report["status"] == (
        "ORIGINAL_CALL_WORDS_OBSERVED" if change == "unchanged" else "UNVERIFIED"
    )
    for name in ("artifact_events", "runtime_events", "comparison_events"):
        assert observation.report[name] == [{"original": "words"}]
    if change == "unchanged":
        return
    field = {
        "source": "source_fence_held",
        "error": "source_fence_held",
        "code": "code_identity_fence_held",
        "settings": "source_settings_restored",
        "context": "native_context_restored",
        "profile": "previous_profile_restored",
    }[change]
    assert observation.report[field] is False


def test_cast_observer_reads_existing_literal_jax_config() -> None:
    module = _cast_observer_module()
    config = type(sys)("existing-config")
    values = {
        "x64_enabled": False,
        "jax_default_matmul_precision": None,
        "jax_platforms": "cpu",
        "jax_numpy_dtype_promotion": "standard",
        "jax_numpy_rank_promotion": "allow",
    }
    vars(config).update(values)
    snapshot = module._config_values(config)
    assert snapshot == values
    config.x64_enabled = True
    assert snapshot["x64_enabled"] is False
    assert module._config_values(config)["x64_enabled"] is True


@pytest.mark.parametrize("promotion", ("standard", "strict"))
def test_cast_observer_records_actual_jax_dtype_promotion_scope(promotion: str) -> None:
    import jax

    module = _cast_observer_module()
    before = module._config_values(jax.config)
    with jax.numpy_dtype_promotion(promotion):
        native = jax.config.jax_numpy_dtype_promotion
        snapshot = module._config_values(jax.config)
        observed = snapshot["jax_numpy_dtype_promotion"]
        if type(native) is str:
            assert observed == promotion
        else:
            assert observed == {
                "kind": "native-jax-enum",
                "type_module": "jax._src.config",
                "type_name": "NumpyDtypePromotion",
                "type_identity": id(type(native)),
                "member": promotion.upper(),
                "member_identity": id(native),
                "value": promotion,
            }
        assert json.loads(json.dumps(snapshot)) == snapshot
    assert module._config_values(jax.config) == before
    assert snapshot["jax_numpy_dtype_promotion"] == observed


@pytest.mark.parametrize("foreign_kind", ("object", "str-subclass", "str-enum"))
@pytest.mark.parametrize(
    "field",
    (
        "x64_enabled",
        "jax_default_matmul_precision",
        "jax_platforms",
        "jax_numpy_dtype_promotion",
        "jax_numpy_rank_promotion",
    ),
)
def test_cast_observer_config_rejects_foreign_values_without_coercion(
    field: str, foreign_kind: str
) -> None:
    module = _cast_observer_module()
    calls = []

    class CoercionGuard:
        def __bool__(self) -> bool:
            calls.append("bool")
            raise AssertionError("diagnostic coerced a foreign config")

        def __str__(self) -> str:
            calls.append("str")
            raise AssertionError("diagnostic rendered a foreign config")

        def __eq__(self, other: Any) -> bool:
            calls.append("eq")
            raise AssertionError("diagnostic compared a foreign config")

    if foreign_kind == "str-enum":
        from enum import StrEnum

        class ForeignEnum(CoercionGuard, StrEnum):
            STANDARD = "standard"

        foreign = ForeignEnum.STANDARD
    elif foreign_kind == "str-subclass":

        class ForeignString(CoercionGuard, str):
            pass

        foreign = ForeignString("standard")
    else:
        foreign = CoercionGuard()

    config = type(sys)("existing-config")
    vars(config).update(
        x64_enabled=False,
        jax_default_matmul_precision=None,
        jax_platforms="cpu",
        jax_numpy_dtype_promotion="standard",
        jax_numpy_rank_promotion="allow",
    )
    vars(config)[field] = foreign
    with pytest.raises(ValueError, match="original .*config"):
        module._config_values(config)
    assert calls == []
