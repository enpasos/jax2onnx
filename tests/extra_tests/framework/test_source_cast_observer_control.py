# tests/extra_tests/framework/test_source_cast_observer_control.py

"""Exercise the controller with real tiny workers and synthetic word scaffolding.

These fixtures qualify orchestration/provenance, not any native cast result.
"""

from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

import pytest


_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _ROOT / "scripts/control_source_cast_observer.py"
_RUNNER = _ROOT / "scripts/run_pytest_batches.py"


def _load(path: Path) -> Any:
    spec = spec_from_file_location("_control_test_" + path.stem, path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _worker(folder: Path, request: dict[str, Any]) -> dict[str, Any]:
    folder.mkdir(parents=True)
    _write(folder / "request.json", request)
    completed = subprocess.run(
        [sys.executable, "-B", str(_RUNNER), "--worker", str(folder / "request.json")],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode in (0, 1), completed.stdout + completed.stderr
    return json.loads((folder / "receipt.json").read_text())


def _parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str = "fail"
) -> tuple[Any, Path, Path]:
    suite = tmp_path / "suite"
    owner = suite / "tests/extra_tests/converter/test_source_conversion_semantics.py"
    owner.parent.mkdir(parents=True)
    owner.write_text(
        """import json, os, sys, time
from pathlib import Path
import pytest

def controlled():
    return 'diagnostic_control' in json.loads(Path(sys.argv[-1]).read_text())

def test_predecessor():
    assert os.environ['JAX_ENABLE_X64'] == 'True'
    if controlled() and os.environ['JAX2ONNX_CONTROL_FIXTURE_MODE'] == 'earlier':
        assert False, 'unrelated predecessor failure'
    if controlled() and os.environ['JAX2ONNX_CONTROL_FIXTURE_MODE'] == 'profile':
        sys.setprofile(lambda *args: None)

@pytest.mark.parametrize('source', ['float64'])
@pytest.mark.parametrize('target', ['float16'])
@pytest.mark.parametrize('group', ['ties'])
def test_floating_conversion_rounding_and_strict_special_value_gates(source, target, group):
    mode = os.environ['JAX2ONNX_CONTROL_FIXTURE_MODE']
    if controlled() and mode == 'timeout':
        time.sleep(30)
    if controlled() and mode in ('pass', 'suffix'):
        return
    assert False, 'jax2onnx.precision.SampleAcceptanceError: values fail bitwise comparison'

def test_suffix():
    if controlled() and os.environ['JAX2ONNX_CONTROL_FIXTURE_MODE'] == 'suffix':
        assert False, 'unrelated suffix failure'
"""
    )
    (suite / "tests/test_z_import.py").write_text(
        """import os
os.environ.setdefault('JAX_ENABLE_X64', 'True')
def test_imported():
    pass
"""
    )
    monkeypatch.chdir(suite)
    monkeypatch.delenv("JAX_ENABLE_X64", raising=False)
    monkeypatch.delenv("PYTEST_PLUGINS", raising=False)
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.setenv("PYTEST_ADDOPTS", "-x")
    monkeypatch.setenv("JAX2ONNX_CONTROL_FIXTURE_MODE", mode)
    receipts = tmp_path / "original"
    options = ["-q", "--tb=short"]
    collection = _worker(
        receipts / "collection",
        {"kind": "collection", "pytest_args": [*options, "--collect-only"]},
    )
    full = collection["selected_ids"]
    request = {
        "kind": "batch",
        "assigned": full,
        "full_preselection_ids": full,
        "collection_skips": collection["collection_skips"],
        "pytest_args": options,
    }
    folder = receipts / "batch-0001"
    receipt = _worker(folder, request)
    # Supply clearly synthetic observed-word scaffolding for the controller's
    # strict parent proof. No fixture executes or claims native high-half words.
    scaffold = _load(
        _ROOT / "tests/extra_tests/framework/test_source_cast_failure_replay.py"
    )
    helper, _, proof_folder = scaffold._fixture(tmp_path / "synthetic-scaffold")
    observed = json.loads((proof_folder / helper.OBSERVATION).read_text())
    observed["scope"] = "SYNTHETIC WORD SCAFFOLD FOR TINY WORKER ORCHESTRATION ONLY"
    observed["runtime_environment"] = {
        name: receipt["target_context_before"]["environment"].get(name)
        for name in (
            "XLA_FLAGS",
            "JAX_PLATFORMS",
            "JAX_ENABLE_X64",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NPY_DISABLE_CPU_FEATURES",
        )
    }
    _write(folder / helper.OBSERVATION, observed)
    (folder / helper.MODEL).write_bytes((proof_folder / helper.MODEL).read_bytes())
    _write(
        receipts / "run.json",
        {
            "status": "FAIL",
            "nodeids": full,
            "batches": [full],
            "checked_out_commit": receipt["startup_context"]["checkout"]["head"],
            "interpreter": sys.executable,
            "pytest_addopts": receipt["startup_context"]["environment"][
                "PYTEST_ADDOPTS"
            ],
            "processes": [{"kind": "batch", "folder": folder.name, "returncode": 1}],
        },
    )
    return _load(_SCRIPT), receipts, folder


def _result(receipts: Path) -> dict[str, Any]:
    return json.loads(
        (receipts / "diagnostics/source-cast-observer-off-v1/run.json").read_text()
    )


@pytest.mark.parametrize(
    "mode, expected",
    (
        ("fail", "OBSERVER_NOT_NECESSARY"),
        ("pass", "TARGET_PASSED_ASSOCIATION_ONLY"),
        ("suffix", "TARGET_PASSED_ASSOCIATION_ONLY"),
    ),
)
def test_one_real_child_keeps_full_order_and_records_actual_target_and_suffix(
    mode: str, expected: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, receipts, folder = _parent(tmp_path, monkeypatch, mode)
    before = {path: path.read_bytes() for path in folder.iterdir() if path.is_file()}
    assert module.main(["--receipts-dir", str(receipts)]) == 0
    result = _result(receipts)
    assert result["status"] == expected
    assert result["child_invocations"] == len(result["processes"]) == 1
    assert result["assigned_count"] == result["full_collection_count"] == 4
    assert result["target_index"] == 1 and result["prefix_matched"] is True
    assert result["native_source_recomputed"] is True
    assert result["raw_native_words_captured"] is False
    assert result["original_evidence_fence_held"] is result["source_fence_held"] is True
    assert result["full_batch_success"] is (mode == "pass")
    if mode == "suffix":
        assert result["suffix_failures"][0]["nodeid"].endswith("::test_suffix")
        assert result["suffix_executed_count"] == 1
    assert {path: path.read_bytes() for path in before} == before


def test_startup_environment_is_restored_before_imports_without_mutating_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, receipts, _ = _parent(tmp_path, monkeypatch)
    monkeypatch.setenv("JAX_ENABLE_X64", "True")
    monkeypatch.setenv("JAX_CONTROL_LATE_ADDITION", "must-not-reach-child-startup")
    before = dict(os.environ)
    assert module.main(["--receipts-dir", str(receipts)]) == 0
    result = _result(receipts)
    child = json.loads(
        (
            receipts / "diagnostics/source-cast-observer-off-v1/batch-0001/receipt.json"
        ).read_text()
    )
    assert dict(os.environ) == before
    assert child["startup_context"]["environment"]["JAX_ENABLE_X64"] is None
    assert "JAX_CONTROL_LATE_ADDITION" not in child["startup_context"]["environment"]
    assert child["collection_context"]["environment"]["JAX_ENABLE_X64"] == "True"
    assert child["target_context_before"]["environment"]["JAX_ENABLE_X64"] == "True"
    assert result["observer_disabled_verified"] is True


@pytest.mark.parametrize("mode", ("earlier", "profile"))
def test_earlier_failure_or_preexisting_profile_is_unverified(
    mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, receipts, _ = _parent(tmp_path, monkeypatch, mode)
    assert module.main(["--receipts-dir", str(receipts)]) == 1
    result = _result(receipts)
    assert result["status"] == "UNVERIFIED" and result["child_invocations"] == 1
    child = json.loads(
        (
            receipts / "diagnostics/source-cast-observer-off-v1/batch-0001/receipt.json"
        ).read_text()
    )
    if mode == "profile":
        assert child["startup_context"]["sys_profile_is_none"] is True
        assert child["target_context_before"]["sys_profile_is_none"] is False
        assert child["native_source_recomputed"] is False
    else:
        assert "target_context_before" not in child


@pytest.mark.parametrize(
    "fault",
    (
        "startup-missing",
        "startup-forged",
        "head",
        "source",
        "target-environment",
        "request-order",
        "evidence",
    ),
)
def test_parent_proof_corruption_prevents_any_child(
    fault: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, receipts, folder = _parent(tmp_path, monkeypatch)
    if fault.startswith("startup") or fault in ("head", "source", "target-environment"):
        path = folder / "receipt.json"
        value = json.loads(path.read_text())
        if fault == "startup-missing":
            del value["startup_context"]
        elif fault == "startup-forged":
            value["startup_context"]["python"] = "different interpreter"
        elif fault == "head":
            value["startup_context"]["checkout"]["head"] = "0" * 40
        elif fault == "source":
            value["startup_context"]["source_files"]["scripts/run_pytest_batches.py"][
                "sha256"
            ] = ("0" * 64)
        else:
            value["target_context_before"]["environment"]["JAX_ENABLE_X64"] = "False"
        _write(path, value)
    elif fault == "request-order":
        path = folder / "request.json"
        value = json.loads(path.read_text())
        value["assigned"] = list(reversed(value["assigned"]))
        _write(path, value)
    else:
        (folder / "events.jsonl").write_text("")
    assert module.main(["--receipts-dir", str(receipts)]) == 1
    result = _result(receipts)
    assert result["status"] == "UNVERIFIED" and result["child_invocations"] == 0
    assert result["processes"] == []


def test_timeout_terminates_one_owned_child_and_preserves_original_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, receipts, folder = _parent(tmp_path, monkeypatch, "timeout")
    before = {path: path.read_bytes() for path in folder.iterdir() if path.is_file()}
    assert module.main(["--receipts-dir", str(receipts), "--timeout-seconds", "1"]) == 1
    result = _result(receipts)
    assert result["status"] == "UNVERIFIED"
    assert result["child_invocations"] == len(result["processes"]) == 1
    assert result["processes"][0]["returncode"] is not None
    assert result["processes"][0]["output_complete"] is True
    assert {path: path.read_bytes() for path in before} == before


def test_controller_interruption_cleans_up_its_child_and_records_unverified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, receipts, _ = _parent(tmp_path, monkeypatch, "timeout")
    child = subprocess.Popen(
        [sys.executable, "-B", str(_SCRIPT), "--receipts-dir", str(receipts)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    report = receipts / "diagnostics/source-cast-observer-off-v1/run.json"
    worker_receipt = report.parent / "batch-0001/receipt.json"
    worker_pid: int | None = None
    worker_start: str | None = None

    def remember_worker() -> None:
        nonlocal worker_pid, worker_start
        if not report.is_file():
            return
        value = json.loads(report.read_text())
        if value.get("processes") and value["processes"][0].get("pid"):
            worker_pid = value["processes"][0]["pid"]
            try:
                fields = (
                    Path(f"/proc/{worker_pid}/stat")
                    .read_text()
                    .rsplit(")", 1)[1]
                    .split()
                )
                if worker_start is None:
                    worker_start = fields[19]
            except FileNotFoundError:
                pass

    def owned_worker_exists() -> bool:
        if worker_pid is None:
            return False
        try:
            command = Path(f"/proc/{worker_pid}/cmdline").read_bytes().split(b"\0")
            fields = (
                Path(f"/proc/{worker_pid}/stat").read_text().rsplit(")", 1)[1].split()
            )
            return (
                worker_start == fields[19]
                and os.getpgid(worker_pid) == worker_pid
                and command[1:4]
                == [
                    os.fsencode(_RUNNER),
                    b"--worker",
                    os.fsencode(worker_receipt.with_name("request.json")),
                ]
            )
        except ProcessLookupError:
            return False
        except FileNotFoundError:
            return False

    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            remember_worker()
            if worker_pid is not None and worker_receipt.is_file():
                worker = json.loads(worker_receipt.read_text())
                # The existing worker checkpoints this boundary before yielding
                # to the target; its in-memory recomputed flag is saved later.
                if (
                    worker.get("target_context_before", {}).get("sys_profile_is_none")
                    is True
                ):
                    break
            time.sleep(0.02)
        else:
            pytest.fail("control child did not start")
        os.kill(child.pid, signal.SIGTERM)
        output, _ = child.communicate(timeout=20)
        assert child.returncode != 0, output
    finally:
        if child.poll() is None:
            try:
                os.kill(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.communicate(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.communicate(timeout=10)
        # _child puts its worker in a separate session. If the controller had
        # to be killed, terminate only the recorded request's verified group.
        remember_worker()
        if owned_worker_exists():
            assert worker_pid is not None
            try:
                os.killpg(worker_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 2
            while owned_worker_exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            if owned_worker_exists():
                try:
                    os.killpg(worker_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    result = _result(receipts)
    assert result["status"] == "UNVERIFIED" and result["child_invocations"] == 1
    assert result["processes"][0]["returncode"] is not None
    assert result["processes"][0]["output_complete"] is True


def test_output_collision_does_not_touch_original_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, receipts, folder = _parent(tmp_path, monkeypatch)
    before = {path: path.read_bytes() for path in folder.iterdir() if path.is_file()}
    assert (
        module.main(["--receipts-dir", str(receipts), "--output-dir", str(folder)]) == 1
    )
    assert {path: path.read_bytes() for path in before} == before
    assert not (folder / "run.json").exists()


def test_unrelated_missing_witness_runs_no_child_or_framework_import(
    tmp_path: Path,
) -> None:
    receipts = tmp_path / "missing"
    program = """import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'jax', 'jaxlib', 'jax2onnx', 'numpy', 'onnx', 'onnxruntime', 'flax', 'scipy'}:
        raise AssertionError('framework import on unrelated early exit: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = [sys.argv[1], '--receipts-dir', sys.argv[2]]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", program, str(_SCRIPT), str(receipts)],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = _result(receipts)
    assert result["status"] == "NOT_APPLICABLE"
    assert result["child_invocations"] == 0 and result["processes"] == []
