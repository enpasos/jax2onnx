# tests/extra_tests/framework/test_pytest_batches.py

"""Exercise fresh-process suite accounting with tiny, real pytest suites."""

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
        (suite / name).write_text(content)
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
