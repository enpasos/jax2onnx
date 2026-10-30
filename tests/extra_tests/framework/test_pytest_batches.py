# tests/extra_tests/framework/test_pytest_batches.py

"""Exercise fresh-process suite accounting with tiny, real pytest suites."""

import base64
import builtins
import csv
import hashlib
from importlib import metadata
from importlib.util import module_from_spec, spec_from_file_location
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
import xml.etree.ElementTree as ET

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
    suite = _write_suite(tmp_path, files)
    return _run_prepared_suite(tmp_path, suite, batch_size=batch_size)


def _write_suite(tmp_path: Path, files: dict[str, str]) -> Path:
    suite = tmp_path / "suite"
    suite.mkdir()
    for name, content in files.items():
        destination = suite / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content)
    return suite


def _run_prepared_suite(
    tmp_path: Path,
    suite: Path,
    *,
    batch_size: int = 2,
    runner_args: tuple[str, ...] = (),
    test_paths: tuple[str, ...] | None = None,
    environment: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any], Path]:
    receipts = tmp_path / "receipts"
    command = [
        sys.executable,
        str(_RUNNER),
        "--output-dir",
        str(receipts),
        "--batch-size",
        str(batch_size),
    ]
    for path in (str(suite),) if test_paths is None else test_paths:
        command.extend(("--test-path", path))
    command.extend((*runner_args, "--", "-q"))
    result = subprocess.run(
        command,
        cwd=suite,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    run = json.loads((receipts / "run.json").read_text())
    return result, run, receipts


def _prepare_lane_suite(
    tmp_path: Path,
    files: dict[str, str],
    baseline: list[str],
    *,
    collection_args: tuple[str, ...] = (),
) -> tuple[Path, Path, dict[str, str]]:
    suite = _write_suite(tmp_path, {"pytest.ini": "[pytest]\ntestpaths = .\n", **files})
    environment = os.environ.copy()
    environment.update(PYTEST_ADDOPTS="", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    environment.pop("JAX2ONNX_NATIVE_BINDINGS_EVIDENCE_DIR", None)
    child_environment = {
        **environment,
        "JAX2ONNX_NATIVE_BINDINGS_EVIDENCE_DIR": str(
            tmp_path / "receipts" / "native-model-artifacts"
        ),
    }
    manifest = {
        "schema": "jax2onnx-ci-baseline-v1",
        "baseline_commit": "1" * 40,
        "python_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "nodeids": baseline,
        "collection_args": list(collection_args),
        "source_files": {
            path.relative_to(suite)
            .as_posix(): hashlib.sha256(path.read_bytes())
            .hexdigest()
            for path in suite.rglob("*")
            if path.is_file()
        },
        "packages": {"pytest": metadata.version("pytest")},
        "environment": _runner()._ci_lane_environment(child_environment),
    }
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return suite, path, environment


def _run_lane_suite(
    tmp_path: Path,
    suite: Path,
    manifest: Path,
    environment: dict[str, str],
    *,
    lane: str = "standard",
    batch_size: int = 2,
    collection_args: tuple[str, ...] = (),
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any], Path]:
    return _run_prepared_suite(
        tmp_path,
        suite,
        batch_size=batch_size,
        runner_args=("--ci-lane", lane, "--ci-main-baseline", str(manifest)),
        test_paths=collection_args,
        environment=environment,
    )


_COLLECTION_TRANSITION_SOURCE = """import os
import jax
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
jax.config.update('jax_enable_x64', True)
"""


def _bind_collection_transition(
    tmp_path: Path,
    suite: Path,
    manifest: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    for name in ("JAX_PLATFORMS", "JAX_ENABLE_X64", "TF_CPP_MIN_LOG_LEVEL"):
        environment.pop(name, None)
    child_environment = {
        **environment,
        "JAX2ONNX_NATIVE_BINDINGS_EVIDENCE_DIR": str(
            tmp_path / "receipts/native-model-artifacts"
        ),
    }
    # Observe the official installed JAX literal metadata in a separate process.
    # The canonical runner must reproduce the real transition during collection.
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "from importlib.util import module_from_spec, spec_from_file_location\n"
            "import json\n"
            f"spec = spec_from_file_location('_transition_probe', {str(_RUNNER)!r})\n"
            "module = module_from_spec(spec)\nspec.loader.exec_module(module)\n"
            + _COLLECTION_TRANSITION_SOURCE
            + "print(json.dumps(module._worker_context()))\n",
        ],
        cwd=suite,
        env=child_environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr
    collected = json.loads(probe.stdout)
    payload = json.loads(manifest.read_text())
    payload["environment"] = _runner()._ci_lane_environment(child_environment)
    payload["collection_transition"] = {
        "startup_jax_config": {"status": "NOT_LOADED", "values": {}},
        "collected_environment": _runner()._ci_lane_environment(
            collected["environment"]
        ),
        "collected_jax_config": collected["loaded_jax_config"],
        "post_pytest_environment": _runner()._ci_lane_environment(
            collected["environment"]
        ),
        "post_pytest_jax_config": collected["loaded_jax_config"],
    }
    manifest.write_text(json.dumps(payload, indent=2) + "\n")
    return collected


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
            durations = receipt["phase_durations"]
            assert set(durations) == set(receipt["phases"])
            events = [
                json.loads(line)
                for line in (directory / "events.jsonl").read_text().splitlines()
            ]
            assert [event for event in events if event["kind"] == "phase-duration"] == [
                {"kind": "phase-duration", "nodeid": nodeid, **timing}
                for nodeid, timings in durations.items()
                for timing in timings
            ]
            testcase_times = {
                case.attrib["name"]: float(case.attrib["time"])
                for case in ET.parse(directory / "junit.xml").iter("testcase")
            }
            for nodeid, timings in durations.items():
                assert [timing["when"] for timing in timings] == [
                    phase["when"] for phase in receipt["phases"][nodeid]
                ]
                assert all(
                    type(timing["seconds"]) is float
                    and math.isfinite(timing["seconds"])
                    and timing["seconds"] >= 0
                    for timing in timings
                )
                assert all(
                    "seconds" not in phase for phase in receipt["phases"][nodeid]
                )
                # JUnit rounds each testcase's total setup/call/teardown time.
                assert (
                    abs(
                        sum(timing["seconds"] for timing in timings)
                        - testcase_times[nodeid.partition("::")[2]]
                    )
                    <= 0.0011
                )
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


@pytest.mark.parametrize("lane", ["standard", "extra"])
def test_canonical_lane_batches_keep_full_collection_and_one_request_protocol(
    lane: str, tmp_path: Path
) -> None:
    baseline = [
        "test_a_consumer.py::test_baseline",
        "test_a_consumer.py::test_skip",
        "test_a_consumer.py::test_xfail",
    ]
    extra = [
        "test_a_consumer.py::test_extra",
        "test_z_registration.py::test_registration",
    ]
    suite, manifest, environment = _prepare_lane_suite(
        tmp_path,
        {
            "test_a_consumer.py": """import builtins
from pathlib import Path
import pytest
def test_baseline():
    assert builtins._ci_batch_registered == 'registered'
    Path('ran-baseline').write_text('yes')
@pytest.mark.skip(reason='original skip')
def test_skip():
    pass
@pytest.mark.xfail(strict=True, reason='original xfail')
def test_xfail():
    assert False
def test_extra():
    assert builtins._ci_batch_registered == 'registered'
    Path('ran-extra').write_text('yes')
""",
            "test_z_registration.py": """import builtins
builtins._ci_batch_registered = 'registered'
def test_registration():
    pass
""",
            "test_collection_skip.py": """import pytest
pytest.skip('original collection skip', allow_module_level=True)
""",
        },
        baseline,
    )
    evidence_key = "JAX2ONNX_NATIVE_BINDINGS_EVIDENCE_DIR"
    original_environment = os.environ.get(evidence_key)
    result, run, folder = _run_lane_suite(
        tmp_path, suite, manifest, environment, lane=lane, batch_size=1
    )
    assert result.returncode == 0, result.stdout + result.stderr
    selected = baseline if lane == "standard" else extra
    full = baseline + extra
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert run["status"] == "PASS" and run["nodeids"] == full
    assert run["lane_nodeids"] == run["completed"] == selected
    assert run["batches"] == [[nodeid] for nodeid in selected]
    assert run["ci_lane"]["lane"] == lane
    assert run["ci_lane"]["manifest_sha256"] == digest
    assert len(run["processes"]) == 1 + len(selected)
    assert os.environ.get(evidence_key) == original_environment
    expected_environment = json.loads(manifest.read_text())["environment"]
    transport = json.loads(
        (folder / "native-model-artifact-transport.json").read_text()
    )
    assert transport["environment_mode"] == "RUN_OWNED"
    for process in run["processes"]:
        directory = folder / process["folder"]
        request_raw = (directory / "request.json").read_bytes()
        receipt_raw = (directory / "receipt.json").read_bytes()
        request, receipt = json.loads(request_raw), json.loads(receipt_raw)
        terminal = json.loads((directory / "lane-terminal.json").read_text())
        assert terminal["request_sha256"] == hashlib.sha256(request_raw).hexdigest()
        assert terminal["receipt_sha256"] == hashlib.sha256(receipt_raw).hexdigest()
        assert terminal["manifest_sha256"] == digest
        assert terminal["exitstatus"] == receipt["exitstatus"] == 0
        assert receipt["full_preselection_ids"] == full
        assert [item["nodeid"] for item in receipt["full_items"]] == full
        assert receipt["ci_lane"]["lane"] == lane
        assert receipt["ci_lane"]["manifest_sha256"] == digest
        assert receipt["ci_lane"]["errors"] == []
        assert receipt["ci_lane"]["environment"] == expected_environment
        assert receipt["startup_context"]["environment"][evidence_key] == str(
            folder / "native-model-artifacts"
        )
        assert len(receipt["collection_skips"]) == 1
        assert "--_j2o-ci-lane=" + lane in request["pytest_args"]
        assert "--_j2o-ci-main-baseline=" + str(manifest) in request["pytest_args"]
        assert "--_j2o-ci-main-sha256=" + digest in request["pytest_args"]
        assert not any(
            argument.startswith("--ci-lane-report")
            for argument in request["pytest_args"]
        )
        assert "-p" not in request["pytest_args"]
        if process["kind"] == "collection":
            assert set(request) == {"kind", "pytest_args"}
            assert receipt["phases"] == {}
        else:
            assert set(request) == {
                "kind",
                "assigned",
                "full_preselection_ids",
                "collection_skips",
                "pytest_args",
            }
            assert request["full_preselection_ids"] == full
            assert (
                request["assigned"]
                == receipt["assigned_ids"]
                == receipt["selected_ids"]
            )
            assert set(receipt["phases"]) == set(request["assigned"])
            assert len(ET.parse(directory / "junit.xml").findall(".//testcase")) == 2
    assert (suite / "ran-baseline").exists() is (lane == "standard")
    assert (suite / "ran-extra").exists() is (lane == "extra")


def test_canonical_empty_extra_lane_retains_collection_without_execution(
    tmp_path: Path,
) -> None:
    baseline = ["test_cases.py::test_baseline"]
    suite, manifest, environment = _prepare_lane_suite(
        tmp_path,
        {
            "test_cases.py": "from pathlib import Path\ndef test_baseline():\n    Path('unexpected-body').touch()\n"
        },
        baseline,
    )
    result, run, folder = _run_lane_suite(
        tmp_path, suite, manifest, environment, lane="extra"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert run["status"] == "PASS" and run["nodeids"] == baseline
    assert run["lane_nodeids"] == run["completed"] == run["batches"] == []
    assert len(run["processes"]) == 1
    receipt = json.loads((folder / "collection/receipt.json").read_text())
    assert receipt["full_preselection_ids"] == baseline and receipt["phases"] == {}
    assert not (suite / "unexpected-body").exists()


@pytest.mark.parametrize("lane", ["standard", "extra"])
def test_canonical_collection_transition_preserves_true_jax_startup(
    lane: str, tmp_path: Path
) -> None:
    baseline = ["test_a_consumer.py::test_baseline"]
    suite, manifest, environment = _prepare_lane_suite(
        tmp_path,
        {
            "test_a_consumer.py": """from pathlib import Path
import builtins
import jax
import jax.numpy as jnp
INITIAL_X64 = jax.config.x64_enabled
INITIAL_DTYPE = str(jnp.asarray(0.0).dtype)
def check_original_source():
    assert INITIAL_X64 is False
    assert INITIAL_DTYPE == 'float32'
    assert jax.config.x64_enabled is True
    assert builtins._ci_transition_registration == 'complete'
def test_baseline():
    check_original_source()
    Path('ran-baseline').touch()
def test_extra():
    check_original_source()
    Path('ran-extra').touch()
""",
            "test_z_registration.py": _COLLECTION_TRANSITION_SOURCE
            + "import builtins\nbuiltins._ci_transition_registration = 'complete'\n",
        },
        baseline,
    )
    collected = _bind_collection_transition(tmp_path, suite, manifest, environment)
    result, run, folder = _run_lane_suite(
        tmp_path, suite, manifest, environment, lane=lane, batch_size=1
    )
    assert result.returncode == 0, result.stdout + result.stderr
    selected = baseline if lane == "standard" else ["test_a_consumer.py::test_extra"]
    assert run["status"] == "PASS"
    assert run["completed"] == run["lane_nodeids"] == selected
    assert run["nodeids"] == baseline + ["test_a_consumer.py::test_extra"]
    for process in run["processes"]:
        directory = folder / process["folder"]
        request = json.loads((directory / "request.json").read_text())
        receipt = json.loads((directory / "receipt.json").read_text())
        assert len(request) == (2 if process["kind"] == "collection" else 5)
        assert receipt["startup_context"]["loaded_jax_config"] == {
            "status": "NOT_LOADED",
            "values": {},
        }
        for name in ("JAX_PLATFORMS", "JAX_ENABLE_X64", "TF_CPP_MIN_LOG_LEVEL"):
            assert receipt["startup_context"]["environment"].get(name) is None
        assert (
            receipt["collection_context"]["loaded_jax_config"]
            == collected["loaded_jax_config"]
        )
        assert (
            receipt["final_context"]["loaded_jax_config"]
            == collected["loaded_jax_config"]
        )
        assert receipt["collection_transition_completed"] is True
        assert receipt["post_pytest_completed"] is True
        assert receipt["full_preselection_ids"] == run["nodeids"]
        assert receipt["ci_lane"]["errors"] == []
    assert (suite / "ran-baseline").exists() is (lane == "standard")
    assert (suite / "ran-extra").exists() is (lane == "extra")


def test_canonical_staged_context_preserves_natural_pytest_environment_lifecycle(
    tmp_path: Path,
) -> None:
    suite, manifest, environment = _prepare_lane_suite(
        tmp_path,
        {
            "test_case.py": """import os
from pathlib import Path
from importlib.metadata import version
assert os.environ['PYTEST_VERSION'] == version('pytest')
def test_baseline():
    assert os.environ['PYTEST_VERSION'] == version('pytest')
    Path('body-ran').touch()
"""
        },
        ["test_case.py::test_baseline"],
    )
    environment.pop("PYTEST_VERSION", None)
    child_environment = {
        **environment,
        "JAX2ONNX_NATIVE_BINDINGS_EVIDENCE_DIR": str(
            tmp_path / "receipts/native-model-artifacts"
        ),
    }
    module = _runner()
    payload = json.loads(manifest.read_text())
    payload["environment"] = module._ci_lane_environment(child_environment)
    payload["collection_transition"] = {
        "startup_jax_config": {"status": "NOT_LOADED", "values": {}},
        "collected_environment": module._ci_lane_environment(
            {**child_environment, "PYTEST_VERSION": metadata.version("pytest")}
        ),
        "collected_jax_config": {"status": "NOT_LOADED", "values": {}},
        "post_pytest_environment": module._ci_lane_environment(child_environment),
        "post_pytest_jax_config": {"status": "NOT_LOADED", "values": {}},
    }
    manifest.write_text(json.dumps(payload, indent=2) + "\n")
    result, run, folder = _run_lane_suite(tmp_path, suite, manifest, environment)
    assert result.returncode == 0, result.stdout + result.stderr
    assert run["status"] == "PASS"
    assert run["completed"] == ["test_case.py::test_baseline"]
    assert (suite / "body-ran").exists()
    for process in run["processes"]:
        receipt = json.loads((folder / process["folder"] / "receipt.json").read_text())
        assert "PYTEST_VERSION" not in receipt["startup_context"]["environment"]
        assert receipt["collection_context"]["environment"]["PYTEST_VERSION"] == (
            metadata.version("pytest")
        )
        assert "PYTEST_VERSION" not in receipt["final_context"]["environment"]
        assert receipt["collection_transition_completed"] is True
        assert receipt["post_pytest_completed"] is True


def test_staged_unconfigure_mutation_is_rejected_even_when_pytest_restores_final_env(
    tmp_path: Path,
) -> None:
    suite, manifest, environment = _prepare_lane_suite(
        tmp_path,
        {
            "conftest.py": """import os
def pytest_unconfigure(config):
    if not config.option.collectonly:
        os.environ['PYTEST_VERSION'] = 'wrong-before-pytest-cleanup'
""",
            "test_case.py": "from pathlib import Path\ndef test_baseline():\n    Path('body-ran').touch()\n",
        },
        ["test_case.py::test_baseline"],
    )
    environment.pop("PYTEST_VERSION", None)
    child_environment = {
        **environment,
        "JAX2ONNX_NATIVE_BINDINGS_EVIDENCE_DIR": str(
            tmp_path / "receipts/native-model-artifacts"
        ),
    }
    module = _runner()
    payload = json.loads(manifest.read_text())
    payload["environment"] = module._ci_lane_environment(child_environment)
    payload["collection_transition"] = {
        "startup_jax_config": {"status": "NOT_LOADED", "values": {}},
        "collected_environment": module._ci_lane_environment(
            {**child_environment, "PYTEST_VERSION": metadata.version("pytest")}
        ),
        "collected_jax_config": {"status": "NOT_LOADED", "values": {}},
        "post_pytest_environment": module._ci_lane_environment(child_environment),
        "post_pytest_jax_config": {"status": "NOT_LOADED", "values": {}},
    }
    manifest.write_text(json.dumps(payload, indent=2) + "\n")
    result, run, folder = _run_lane_suite(tmp_path, suite, manifest, environment)
    assert result.returncode != 0 and run["status"] == "FAIL"
    assert run["completed"] == [] and len(run["processes"]) == 2
    assert (suite / "body-ran").exists()
    receipt = json.loads((folder / "batch-0001/receipt.json").read_text())
    assert receipt["ci_lane"]["native_exitstatus"] == 0
    assert receipt["ci_lane"]["errors"]
    assert receipt["completed"] == ["test_case.py::test_baseline"]
    assert all(
        phase["outcome"] == "passed"
        for phase in receipt["phases"]["test_case.py::test_baseline"]
    )
    assert "PYTEST_VERSION" not in receipt["final_context"]["environment"]
    assert receipt["post_pytest_completed"] is True


@pytest.mark.parametrize(
    "fault",
    [
        "static-context",
        "startup-environment",
        "startup-jax",
        "collected-environment",
        "collected-jax",
        "unknown-transition-field",
        "missing-transition-field",
        "unknown-jax-field",
        "missing-jax-field",
        "unverified-jax",
        "boolean-coercion",
        "enum-descriptor",
        "incomplete-environment",
        "unknown-environment-field",
        "unknown-snapshot-field",
        "invalid-not-loaded",
        "post-pytest-environment",
        "post-pytest-jax",
        "missing-post-pytest-environment",
        "missing-post-pytest-jax",
        "invalid-post-pytest-environment",
        "unverified-post-pytest-jax",
    ],
)
def test_canonical_collection_transition_rejects_unqualified_context_before_phases(
    fault: str, tmp_path: Path
) -> None:
    suite, manifest, environment = _prepare_lane_suite(
        tmp_path,
        {
            "test_case.py": _COLLECTION_TRANSITION_SOURCE
            + "from pathlib import Path\ndef test_baseline():\n    Path('body-ran').touch()\n"
        },
        ["test_case.py::test_baseline"],
    )
    _bind_collection_transition(tmp_path, suite, manifest, environment)
    payload = json.loads(manifest.read_text())
    transition = payload["collection_transition"]
    if fault == "static-context":
        del payload["collection_transition"]
    elif fault == "startup-environment":
        payload["environment"]["JAX_PLATFORMS"] = "cpu"
    elif fault == "startup-jax":
        transition["startup_jax_config"] = transition["collected_jax_config"]
    elif fault == "collected-environment":
        transition["collected_environment"].pop("TF_CPP_MIN_LOG_LEVEL")
    elif fault == "collected-jax":
        transition["collected_jax_config"]["values"]["x64_enabled"] = False
    elif fault == "unknown-transition-field":
        transition["ignored"] = None
    elif fault == "missing-transition-field":
        del transition["collected_jax_config"]
    elif fault == "unknown-jax-field":
        transition["collected_jax_config"]["values"]["ignored"] = None
    elif fault == "missing-jax-field":
        del transition["collected_jax_config"]["values"]["jax_disable_jit"]
    elif fault == "unverified-jax":
        transition["collected_jax_config"]["status"] = "UNVERIFIED"
    elif fault == "boolean-coercion":
        transition["collected_jax_config"]["values"]["x64_enabled"] = 1
    elif fault == "enum-descriptor":
        transition["collected_jax_config"]["values"]["jax_numpy_dtype_promotion"] = {
            "type": "jax._src.config.NumpyDtypePromotion",
            "member": "STANDARD",
            "value": "strict",
        }
    elif fault == "incomplete-environment":
        del transition["collected_environment"]["JAX_ENABLE_X64"]
    elif fault == "unknown-environment-field":
        transition["collected_environment"]["IGNORED"] = "unbound"
    elif fault == "unknown-snapshot-field":
        transition["startup_jax_config"]["ignored"] = None
    elif fault == "invalid-not-loaded":
        transition["startup_jax_config"]["values"]["x64_enabled"] = False
    elif fault == "post-pytest-environment":
        transition["post_pytest_environment"]["PYTEST_VERSION"] = "unqualified"
    elif fault == "post-pytest-jax":
        transition["post_pytest_jax_config"]["values"]["x64_enabled"] = False
    elif fault == "missing-post-pytest-environment":
        del transition["post_pytest_environment"]
    elif fault == "missing-post-pytest-jax":
        del transition["post_pytest_jax_config"]
    elif fault == "invalid-post-pytest-environment":
        transition["post_pytest_environment"] = []
    else:
        transition["post_pytest_jax_config"]["status"] = "UNVERIFIED"
    manifest.write_text(json.dumps(payload, indent=2) + "\n")
    result, run, folder = _run_lane_suite(tmp_path, suite, manifest, environment)
    assert result.returncode != 0 and run["status"] == "FAIL"
    assert run["completed"] == [] and not (suite / "body-ran").exists()
    for path in folder.glob("*/receipt.json"):
        receipt = json.loads(path.read_text())
        assert receipt["phases"] == receipt["phase_durations"] == {}
        assert receipt["completed"] == []


def test_canonical_transition_requires_completed_stage_before_test_protocol(
    tmp_path: Path,
) -> None:
    suite, manifest, environment = _prepare_lane_suite(
        tmp_path,
        {
            "conftest.py": """def pytest_collection_modifyitems(session, items):
    if not session.config.option.collectonly:
        session.config.hook.pytest_runtest_protocol(item=items[0], nextitem=None)
""",
            "test_case.py": _COLLECTION_TRANSITION_SOURCE
            + "from pathlib import Path\ndef test_baseline():\n    Path('body-ran').touch()\n",
        },
        ["test_case.py::test_baseline"],
    )
    _bind_collection_transition(tmp_path, suite, manifest, environment)
    result, run, folder = _run_lane_suite(tmp_path, suite, manifest, environment)
    assert result.returncode != 0 and run["status"] == "FAIL"
    assert run["completed"] == [] and len(run["processes"]) == 2
    assert not (suite / "body-ran").exists()
    receipt = json.loads((folder / "batch-0001/receipt.json").read_text())
    assert receipt["phases"] == receipt["phase_durations"] == {}
    assert any("stage is incomplete" in error for error in receipt["ci_lane"]["errors"])


@pytest.mark.parametrize("field", ["environment", "jax-config"])
@pytest.mark.parametrize("stage", ["postcollection", "protocol", "unconfigure"])
def test_canonical_transition_rechecks_protocol_and_final_context(
    field: str, stage: str, tmp_path: Path
) -> None:
    mutation = (
        "os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'"
        if field == "environment"
        else "jax.config.update('jax_enable_x64', False)"
    )
    hook = "import os\nimport jax\nimport pytest\n"
    if stage == "postcollection":
        hook += (
            "def pytest_runtestloop(session):\n"
            "    if not session.config.option.collectonly:\n"
            f"        {mutation}\n"
        )
    elif stage == "protocol":
        hook += (
            "@pytest.hookimpl(hookwrapper=True, tryfirst=True)\n"
            "def pytest_runtest_protocol(item):\n"
            f"    {mutation}\n"
            "    yield\n"
        )
    else:
        hook += (
            "def pytest_unconfigure(config):\n"
            "    if not config.option.collectonly:\n"
            f"        {mutation}\n"
        )
    suite, manifest, environment = _prepare_lane_suite(
        tmp_path,
        {
            "conftest.py": hook,
            "test_case.py": _COLLECTION_TRANSITION_SOURCE
            + "from pathlib import Path\ndef test_baseline():\n    Path('body-ran').touch()\n",
        },
        ["test_case.py::test_baseline"],
    )
    _bind_collection_transition(tmp_path, suite, manifest, environment)
    result, run, folder = _run_lane_suite(tmp_path, suite, manifest, environment)
    assert result.returncode != 0 and run["status"] == "FAIL"
    assert run["completed"] == [] and len(run["processes"]) == 2
    receipt = json.loads((folder / "batch-0001/receipt.json").read_text())
    assert receipt["ci_lane"]["errors"]
    assert (suite / "body-ran").exists() is (stage == "unconfigure")
    if stage != "unconfigure":
        assert receipt["phases"] == receipt["phase_durations"] == {}
    else:
        assert receipt["completed"] == ["test_case.py::test_baseline"]
        assert all(
            phase["outcome"] == "passed"
            for phase in receipt["phases"]["test_case.py::test_baseline"]
        )


def test_canonical_lane_rejects_collection_mode_change_before_any_test_phase(
    tmp_path: Path,
) -> None:
    suite, manifest, environment = _prepare_lane_suite(
        tmp_path,
        {
            "test_cases.py": """from pathlib import Path
def test_baseline():
    Path('baseline-body').touch()
def test_extra():
    Path('extra-body').touch()
""",
            "conftest.py": """import pytest
@pytest.hookimpl(tryfirst=True)
def pytest_runtestloop(session):
    session.config.option.collectonly = False
""",
        },
        ["test_cases.py::test_baseline"],
    )
    result, run, folder = _run_lane_suite(tmp_path, suite, manifest, environment)
    assert result.returncode != 0 and run["status"] == "FAIL"
    assert run["completed"] == [] and len(run["processes"]) == 1
    assert not (suite / "baseline-body").exists()
    assert not (suite / "extra-body").exists()
    receipt = json.loads((folder / "collection/receipt.json").read_text())
    assert receipt["phases"] == receipt["phase_durations"] == {}
    assert receipt["completed"] == [] and receipt["ci_lane"]["errors"]


def test_default_canonical_collection_rejects_execution_override_before_body(
    tmp_path: Path,
) -> None:
    result, run, folder = _run_suite(
        tmp_path,
        {
            "conftest.py": """import pytest
@pytest.hookimpl(tryfirst=True)
def pytest_runtestloop(session):
    session.config.option.collectonly = False
""",
            "test_case.py": "from pathlib import Path\ndef test_case():\n    Path('collection-body-ran').touch()\n",
        },
    )
    assert not (tmp_path / "suite/collection-body-ran").exists()
    assert result.returncode != 0 and run["status"] == "FAIL"
    assert run["completed"] == [] and len(run["processes"]) == 1
    receipt = json.loads((folder / "collection/receipt.json").read_text())
    assert receipt["phases"] == receipt["phase_durations"] == {}
    assert receipt["completed"] == []


def test_canonical_lane_preserves_seventeen_qualified_file_and_exact_case_selectors(
    tmp_path: Path,
) -> None:
    files = {
        f"test_owner_{index:02d}.py": "def test_baseline():\n    pass\n"
        for index in range(16)
    }
    files[
        "test_owner_00.py"
    ] += "def test_extra():\n    raise AssertionError('extra executed')\n"
    files[
        "test_exact.py"
    ] = """import pytest
@pytest.mark.parametrize('value', [1, 2], ids=['chosen', 'sibling'])
def test_parameter(value):
    assert value == 1
def test_unselected_sibling():
    raise AssertionError('sibling executed')
"""
    files["test_outside.py"] = "raise AssertionError('outside module imported')\n"
    selectors = tuple(f"test_owner_{index:02d}.py" for index in range(16)) + (
        "test_exact.py::test_parameter[chosen]",
    )
    baseline = [f"test_owner_{index:02d}.py::test_baseline" for index in range(16)] + [
        selectors[-1]
    ]
    suite, manifest, environment = _prepare_lane_suite(
        tmp_path, files, baseline, collection_args=selectors
    )
    result, run, folder = _run_lane_suite(
        tmp_path, suite, manifest, environment, collection_args=selectors, batch_size=50
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        run["status"] == "PASS" and run["lane_nodeids"] == run["completed"] == baseline
    )
    assert len(run["nodeids"]) == 18 and len(run["processes"]) == 2
    assert "test_owner_00.py::test_extra" in run["nodeids"]
    for process in run["processes"]:
        receipt = json.loads((folder / process["folder"] / "receipt.json").read_text())
        assert receipt["full_preselection_ids"] == run["nodeids"]
        assert receipt["ci_lane"]["errors"] == []
        request = json.loads((folder / process["folder"] / "request.json").read_text())
        actual_paths = [
            argument
            for argument in request["pytest_args"]
            if argument.startswith("test_")
        ]
        assert actual_paths == list(selectors)


@pytest.mark.parametrize("fault", ["missing", "reordered", "injected"])
def test_canonical_lane_rejects_changed_qualified_selectors_before_imports(
    fault: str, tmp_path: Path
) -> None:
    selectors = ("test_a.py", "test_b.py::test_second")
    suite, manifest, environment = _prepare_lane_suite(
        tmp_path,
        {
            "test_a.py": "from pathlib import Path\ndef test_first():\n    Path('unexpected-body').touch()\n",
            "test_b.py": "from pathlib import Path\ndef test_second():\n    Path('unexpected-body').touch()\n",
            "test_injected.py": "from pathlib import Path\nPath('unexpected-import').touch()\ndef test_injected():\n    Path('unexpected-body').touch()\n",
        },
        ["test_a.py::test_first", "test_b.py::test_second"],
        collection_args=selectors,
    )
    actual = {
        "missing": selectors[:1],
        "reordered": tuple(reversed(selectors)),
        "injected": (*selectors, "test_injected.py"),
    }[fault]
    result, run, _ = _run_lane_suite(
        tmp_path, suite, manifest, environment, collection_args=actual
    )
    assert result.returncode != 0 and run["status"] == "FAIL"
    assert run["completed"] == [] and len(run["processes"]) <= 1
    assert not (suite / "unexpected-body").exists()
    assert not (suite / "unexpected-import").exists()


def test_canonical_lane_rejects_foreign_deselection_of_only_extra_items(
    tmp_path: Path,
) -> None:
    suite, manifest, environment = _prepare_lane_suite(
        tmp_path,
        {
            "test_cases.py": "from pathlib import Path\ndef test_baseline():\n    Path('unexpected-body').touch()\ndef test_extra():\n    Path('unexpected-body').touch()\n",
            "conftest.py": """def pytest_collection_modifyitems(config, items):
    removed=[item for item in items if item.nodeid.endswith('::test_extra')]
    items[:]=[item for item in items if item not in removed]
    config.hook.pytest_deselected(items=removed)
""",
        },
        ["test_cases.py::test_baseline"],
    )
    result, run, folder = _run_lane_suite(tmp_path, suite, manifest, environment)
    assert result.returncode != 0 and run["status"] == "FAIL"
    assert run["completed"] == [] and not (suite / "unexpected-body").exists()
    assert len(run["processes"]) == 1
    receipt = json.loads((folder / "collection/receipt.json").read_text())
    assert receipt["phases"] == {} and receipt["ci_lane"]["errors"]


def test_canonical_lane_preserves_original_addopts_and_first_failfast_outcome(
    tmp_path: Path,
) -> None:
    baseline = ["test_cases.py::test_first_failure", "test_cases.py::test_later"]
    suite, manifest, environment = _prepare_lane_suite(
        tmp_path,
        {
            "test_cases.py": """from pathlib import Path
def test_first_failure():
    path=Path('first-calls')
    path.write_text(path.read_text()+'x' if path.exists() else 'x')
    assert False, 'original first failure'
def test_later():
    Path('unexpected-later-body').touch()
"""
        },
        baseline,
    )
    original_options = "-v --tb=short -x"
    environment["PYTEST_ADDOPTS"] = original_options
    payload = json.loads(manifest.read_text())
    payload["environment"] = _runner()._ci_lane_environment(
        {
            **environment,
            "JAX2ONNX_NATIVE_BINDINGS_EVIDENCE_DIR": str(
                tmp_path / "receipts/native-model-artifacts"
            ),
        }
    )
    manifest.write_text(json.dumps(payload, indent=2) + "\n")
    result, run, folder = _run_lane_suite(tmp_path, suite, manifest, environment)
    assert result.returncode != 0 and run["status"] == "FAIL"
    assert run["pytest_addopts"] == original_options and run["completed"] == []
    assert [process["kind"] for process in run["processes"]] == ["collection", "batch"]
    assert (suite / "first-calls").read_text() == "x"
    assert not (suite / "unexpected-later-body").exists()
    receipt = json.loads((folder / "batch-0001/receipt.json").read_text())
    assert (
        receipt["startup_context"]["environment"]["PYTEST_ADDOPTS"] == original_options
    )
    assert set(receipt["phases"]) == {baseline[0]}
    assert any(
        phase["when"] == "call" and phase["outcome"] == "failed"
        for phase in receipt["phases"][baseline[0]]
    )
    assert "original first failure" in (folder / "batch-0001/output.log").read_text()


@pytest.mark.parametrize(
    "fault",
    [
        "environment",
        "missing",
        "order",
        "source",
        "package",
        "manifest-after-collection",
    ],
)
def test_canonical_lane_rejects_unqualified_manifest_before_test_bodies(
    fault: str, tmp_path: Path
) -> None:
    baseline = ["test_cases.py::test_first", "test_cases.py::test_second"]
    files = {
        "test_cases.py": "from pathlib import Path\ndef test_first():\n    Path('unexpected-body').touch()\ndef test_second():\n    Path('unexpected-body').touch()\n"
    }
    if fault == "manifest-after-collection":
        files["conftest.py"] = (
            "from pathlib import Path\n"
            "def pytest_sessionfinish(session, exitstatus):\n"
            "    if session.config.option.collectonly:\n"
            f"        path=Path({str(tmp_path / 'baseline.json')!r})\n"
            "        path.write_bytes(path.read_bytes()+b' ')\n"
        )
    suite, manifest, environment = _prepare_lane_suite(tmp_path, files, baseline)
    payload = json.loads(manifest.read_text())
    if fault == "environment":
        payload["environment"] = _runner()._ci_lane_environment(environment)
    elif fault == "missing":
        payload["nodeids"].append("test_cases.py::test_missing")
    elif fault == "order":
        payload["nodeids"].reverse()
    elif fault == "package":
        payload["packages"]["pytest"] = "0.0.0"
    manifest.write_text(json.dumps(payload, indent=2) + "\n")
    if fault == "source":
        with (suite / "test_cases.py").open("a") as output:
            output.write("# source drift after qualification\n")
    result, run, folder = _run_lane_suite(tmp_path, suite, manifest, environment)
    assert result.returncode != 0 and run["status"] == "FAIL"
    assert run["completed"] == [] and not (suite / "unexpected-body").exists()
    assert len(run["processes"]) <= 1
    receipt_path = folder / "collection/receipt.json"
    if receipt_path.exists():
        assert json.loads(receipt_path.read_text())["phases"] == {}


def test_canonical_standard_batch_rechecks_unassigned_extra_inventory_before_body(
    tmp_path: Path,
) -> None:
    baseline = ["test_a_baseline.py::test_baseline"]
    suite, manifest, environment = _prepare_lane_suite(
        tmp_path,
        {
            "test_a_baseline.py": "from pathlib import Path\ndef test_baseline():\n    Path('unexpected-body').touch()\n",
            "test_z_extra.py": "def test_extra():\n    pass\n",
            "conftest.py": """import sys
def pytest_collection_modifyitems(items):
    if any('/batch-' in argument for argument in sys.argv):
        items[-1]._nodeid += '-unassigned-extra-drift'
""",
        },
        baseline,
    )
    result, run, folder = _run_lane_suite(tmp_path, suite, manifest, environment)
    assert result.returncode != 0 and run["status"] == "FAIL"
    assert run["completed"] == [] and len(run["processes"]) == 2
    assert not (suite / "unexpected-body").exists()
    receipt = json.loads((folder / "batch-0001/receipt.json").read_text())
    assert receipt["full_preselection_ids"][-1].endswith("-unassigned-extra-drift")
    assert receipt["phases"] == {} and receipt["full_collection_checked"] is False


@pytest.mark.parametrize(
    "fault", ["lazy-unbound-import", "late-unconfigure-source-drift"]
)
def test_canonical_lane_terminal_audit_fails_after_passed_native_reports(
    fault: str, tmp_path: Path
) -> None:
    files = {
        "test_cases.py": "from pathlib import Path\ndef test_baseline():\n    Path('body-ran').touch()\n",
        "helper.py": "VALUE = 1\n",
    }
    if fault == "lazy-unbound-import":
        files["test_cases.py"] += "    import helper\n    assert helper.VALUE == 1\n"
    else:
        files[
            "conftest.py"
        ] = """from pathlib import Path
def pytest_unconfigure(config):
    if not config.option.collectonly:
        path=Path(__file__).with_name('helper.py')
        path.write_text(path.read_text()+'# late source drift')
"""
    baseline = ["test_cases.py::test_baseline"]
    suite, manifest, environment = _prepare_lane_suite(tmp_path, files, baseline)
    if fault == "lazy-unbound-import":
        payload = json.loads(manifest.read_text())
        del payload["source_files"]["helper.py"]
        manifest.write_text(json.dumps(payload, indent=2) + "\n")
    result, run, folder = _run_lane_suite(tmp_path, suite, manifest, environment)
    assert result.returncode != 0 and run["status"] == "FAIL"
    assert len(run["processes"]) == 2 and run["completed"] == []
    assert (suite / "body-ran").exists()
    receipt = json.loads((folder / "batch-0001/receipt.json").read_text())
    assert receipt["completed"] == baseline and receipt["ci_lane"]["errors"]
    assert [phase["when"] for phase in receipt["phases"][baseline[0]]] == [
        "setup",
        "call",
        "teardown",
    ]
    assert all(phase["outcome"] == "passed" for phase in receipt["phases"][baseline[0]])


def test_canonical_worker_rejects_assignment_outside_its_qualified_lane(
    tmp_path: Path,
) -> None:
    full = ["test_cases.py::test_baseline", "test_cases.py::test_extra"]
    suite, manifest, environment = _prepare_lane_suite(
        tmp_path,
        {
            "test_cases.py": "from pathlib import Path\ndef test_baseline():\n    Path('unexpected-body').touch()\ndef test_extra():\n    Path('unexpected-body').touch()\n"
        },
        full[:1],
    )
    folder = tmp_path / "worker"
    folder.mkdir()
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    request = {
        "kind": "batch",
        "assigned": full[1:],
        "full_preselection_ids": full,
        "collection_skips": [],
        "pytest_args": [
            "-q",
            "--_j2o-ci-lane=standard",
            "--_j2o-ci-main-baseline=" + str(manifest),
            "--_j2o-ci-main-sha256=" + digest,
        ],
    }
    (folder / "request.json").write_text(json.dumps(request) + "\n")
    environment["JAX2ONNX_NATIVE_BINDINGS_EVIDENCE_DIR"] = str(
        tmp_path / "receipts/native-model-artifacts"
    )
    result = subprocess.run(
        [sys.executable, str(_RUNNER), "--worker", str(folder / "request.json")],
        cwd=suite,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode != 0 and not (suite / "unexpected-body").exists()
    receipt = json.loads((folder / "receipt.json").read_text())
    assert receipt["phases"] == {} and receipt["ci_lane"]["errors"]


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


def test_lane_batch_plan_keeps_reviewed_dino_owners_whole_without_regrouping() -> None:
    module = _runner()
    isolated = frozenset(
        {"tests/examples/test_eqx_dino.py", "tests/examples/test_nnx_dino.py"}
    )
    items = []
    for owner, count in (
        ("test_before.py", 1),
        ("tests/examples/test_eqx_dino.py", 2),
        ("test_between.py", 1),
        ("tests/examples/test_nnx_dino.py", 3),
        ("test_source_model_native_bindings_exports.py", 2),
        ("test_after.py", 1),
    ):
        items.extend(
            {"nodeid": f"{owner}::test_{index}", "module": owner, "class": None}
            for index in range(count)
        )
    default = module._plan(items, 5)
    batches = module._plan(items, 5, isolated_owners=isolated)
    ordered = [item["nodeid"] for item in items]
    assert [nodeid for batch in batches for nodeid in batch] == ordered
    assert [nodeid for batch in default for nodeid in batch] == ordered
    assert [len(batch) for batch in batches] == [1, 2, 1, 3, 1, 1, 1]
    assert [len(batch) for batch in default] == [4, 3, 1, 1, 1]
    for owner in isolated:
        expected = [item["nodeid"] for item in items if item["module"] == owner]
        assert expected in batches
    native = [
        item["nodeid"]
        for item in items
        if item["module"] == "test_source_model_native_bindings_exports.py"
    ]
    assert all([nodeid] in batches and [nodeid] in default for nodeid in native)


@pytest.mark.parametrize("fault", ["oversized", "noncontiguous"])
def test_lane_batch_plan_rejects_unpreservable_whole_owner(fault: str) -> None:
    module = _runner()
    owner = "tests/examples/test_eqx_dino.py"
    items = [
        {"nodeid": f"{owner}::test_{index}", "module": owner, "class": None}
        for index in range(3)
    ]
    limit = 2 if fault == "oversized" else 5
    if fault == "noncontiguous":
        items.insert(
            1,
            {
                "nodeid": "test_between.py::test_case",
                "module": "test_between.py",
                "class": None,
            },
        )
    with pytest.raises(ValueError):
        module._plan(items, limit, isolated_owners=frozenset({owner}))


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


def test_lane_enabled_observer_off_control_cannot_validate_canonical_completion(
    tmp_path: Path,
) -> None:
    module = _runner()
    owner = module._CAST_OBSERVER_NODE.split("::", 1)[0]
    suite, manifest, environment = _prepare_lane_suite(
        tmp_path,
        {
            owner: """import pytest
from pathlib import Path
@pytest.mark.parametrize('source', ['float64'])
@pytest.mark.parametrize('target', ['float16'])
@pytest.mark.parametrize('group', ['ties'])
def test_floating_conversion_rounding_and_strict_special_value_gates(source, target, group):
    path=Path('controlled-calls')
    path.write_text(path.read_text()+'x' if path.exists() else 'x')
"""
        },
        [module._CAST_OBSERVER_NODE],
    )
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    original = {
        "kind": "batch",
        "assigned": [module._CAST_OBSERVER_NODE],
        "full_preselection_ids": [module._CAST_OBSERVER_NODE],
        "collection_skips": [],
        "pytest_args": [
            "-q",
            "--_j2o-ci-lane=standard",
            "--_j2o-ci-main-baseline=" + str(manifest),
            "--_j2o-ci-main-sha256=" + digest,
        ],
    }
    request = _observer_off_request(module, original)
    folder = tmp_path / "control"
    folder.mkdir()
    (folder / "request.json").write_text(
        json.dumps(request, indent=2, ensure_ascii=False) + "\n"
    )
    environment["JAX2ONNX_NATIVE_BINDINGS_EVIDENCE_DIR"] = str(
        tmp_path / "receipts/native-model-artifacts"
    )
    result = subprocess.run(
        [sys.executable, str(_RUNNER), "--worker", str(folder / "request.json")],
        cwd=suite,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((folder / "receipt.json").read_text())
    assert (suite / "controlled-calls").read_text() == "x"
    assert receipt["supplemental"] is True and receipt["cast_observer_enabled"] is False
    assert receipt["native_source_recomputed"] is True
    assert (
        receipt["ci_lane"]["lane"] == "standard" and receipt["ci_lane"]["errors"] == []
    )
    assert receipt["completed"] == [module._CAST_OBSERVER_NODE]
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


def _prepare_campaign_suite(
    tmp_path, monkeypatch, mode, *, xml_controls=False, support=False
):
    reason = "original\x07skip\x7f" if xml_controls else "original skip"
    source = """from pathlib import Path
import pytest
def test_base():
    Path('base-ran').touch()
def test_original_skip():
    ORIGINAL_OUTCOME
def test_extra():
    Path('extra-ran').touch()
""".replace(
        "ORIGINAL_OUTCOME",
        {
            "allowed-skip": f"pytest.skip({reason!r})",
            "unexpected-skip": f"pytest.skip({reason!r})",
            "xfail": "pytest.xfail('historical xfail')",
            "pass": "pass",
        }[mode],
    )
    standard = ["test_cases.py::test_base", "test_cases.py::test_original_skip"]
    files = {
        "test_cases.py": source,
        "test_collection_skip.py": "import pytest\n"
        "pytest.skip('unselected collection skip', allow_module_level=True)\n",
    }
    selectors = ("test_cases.py", "test_collection_skip.py") if support else ()
    if support:
        files["test_outside.py"] = "raise AssertionError('outside command imported')\n"
    if xml_controls:
        case_id = "illegal\x07\x7f"
        files["test_cases.py"] = source.replace(
            "def test_original_skip():",
            f"@pytest.mark.parametrize('marker', [None], ids=[{case_id!r}])\n"
            "def test_original_skip(marker):",
        )
        standard[1] += f"[{case_id}]"
        files["pytest.ini"] = (
            "[pytest]\ntestpaths = .\n"
            "disable_test_id_escaping_and_forfeit_all_rights_to_community_support = True\n"
        )
        # A synthetic native collect report exercises pytest's real serializer;
        # the canonical observer and JUnit both receive the same raw text.
        files[
            "conftest.py"
        ] = """import pytest
@pytest.hookimpl(tryfirst=True)
def pytest_collectreport(report):
    if report.skipped:
        report.longrepr = 'collection\\x01skip\\x7f'
"""
    suite, manifest, environment = _prepare_lane_suite(
        tmp_path,
        files,
        standard,
        collection_args=selectors,
    )
    child_environment = {
        **environment,
        "JAX2ONNX_NATIVE_BINDINGS_EVIDENCE_DIR": str(
            tmp_path / "receipts/native-model-artifacts"
        ),
    }
    module = _runner()
    preparation = tmp_path / "preparation"
    preparation.mkdir()
    monkeypatch.chdir(suite)
    authority_options = (
        [
            "--_j2o-ci-lane=standard",
            f"--_j2o-ci-main-baseline={manifest}",
            "--_j2o-ci-main-sha256="
            + hashlib.sha256(manifest.read_bytes()).hexdigest(),
        ]
        if support
        else []
    )
    assert (
        module._child(
            preparation / "global-collection",
            {
                "kind": "collection",
                "pytest_args": ["-q", *authority_options, *selectors, "--collect-only"],
            },
            {"processes": []},
            preparation,
            environment=child_environment,
        )
        == 0
    )
    inventory = preparation / "global-collection/receipt.json"
    observed = json.loads(inventory.read_text())
    assert observed["phases"] == observed["phase_durations"] == {}
    assert observed["completed"] == []
    payload = json.loads(manifest.read_text())
    payload["collection_transition"] = {
        "startup_jax_config": observed["startup_context"]["loaded_jax_config"],
        "collected_environment": module._ci_lane_environment(
            observed["collection_context"]["environment"]
        ),
        "collected_jax_config": observed["collection_context"]["loaded_jax_config"],
        "post_pytest_environment": module._ci_lane_environment(
            observed["final_context"]["environment"]
        ),
        "post_pytest_jax_config": observed["final_context"]["loaded_jax_config"],
    }
    manifest.write_text(json.dumps(payload) + "\n")
    allowed_skips = []
    if mode in {"allowed-skip", "pass"}:
        allowed_skips.append(
            {
                "nodeid": standard[1],
                "phases": ["setup:passed", "call:skipped", "teardown:passed"],
                "source": "test_cases.py",
                "line": 7 if xml_controls else 6,
                "reason": reason,
            }
        )
    contract = tmp_path / "campaign.json"
    contract.write_text(
        json.dumps(
            {
                "schema": (
                    "jax2onnx-ci-support-campaign-v1"
                    if support
                    else "jax2onnx-ci-campaign-v1"
                ),
                "python_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
                "main_manifest": {
                    "path": manifest.name,
                    "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                },
                "command_inventory" if support else "global_inventory": {
                    "path": str(inventory.relative_to(tmp_path)),
                    "sha256": hashlib.sha256(inventory.read_bytes()).hexdigest(),
                },
                "allowed_skips": allowed_skips,
            }
        )
        + "\n"
    )
    return suite, manifest, environment, contract, observed


@pytest.mark.parametrize("mode", ["allowed-skip", "pass", "unexpected-skip", "xfail"])
def test_real_canonical_campaign_retains_full_context_and_strict_native_outcomes(
    tmp_path, monkeypatch, mode
):
    suite, manifest, environment, contract, inventory = _prepare_campaign_suite(
        tmp_path, monkeypatch, mode
    )
    result, run, receipts = _run_prepared_suite(
        tmp_path,
        suite,
        runner_args=(
            "--ci-lane",
            "standard",
            "--ci-main-baseline",
            str(manifest),
            "--ci-campaign-contract",
            str(contract),
        ),
        test_paths=(),
        environment=environment,
    )
    success = mode in {"allowed-skip", "pass"}
    assert (result.returncode == 0) is success, result.stdout + result.stderr
    assert (run["status"] == "PASS") is success
    assert run["nodeids"] == inventory["full_preselection_ids"]
    assert (suite / "base-ran").exists()
    assert not (suite / "extra-ran").exists()
    raw_collection = json.loads((receipts / "collection/receipt.json").read_text())
    assert raw_collection["full_items"] == inventory["full_items"]
    assert raw_collection["collection_skips"] == inventory["collection_skips"]
    assert raw_collection["collection_transition_completed"] is True
    assert raw_collection["post_pytest_completed"] is True
    if success:
        assert run["completed"] == json.loads(manifest.read_text())["nodeids"]
        assert len(run["processes"]) == 2
        batch = json.loads((receipts / "batch-0001/receipt.json").read_text())
        assert batch["full_preselection_ids"] == inventory["full_preselection_ids"]
        assert batch["collection_transition_completed"] is True
        assert batch["post_pytest_completed"] is True
    else:
        assert "campaign" in run["failure"]["message"]


def test_real_campaign_uses_pinned_pytest_binary_xml_serialization(
    tmp_path, monkeypatch
):
    suite, manifest, environment, contract, inventory = _prepare_campaign_suite(
        tmp_path, monkeypatch, "allowed-skip", xml_controls=True
    )
    result, run, receipts = _run_prepared_suite(
        tmp_path,
        suite,
        runner_args=(
            "--ci-lane",
            "standard",
            "--ci-main-baseline",
            str(manifest),
            "--ci-campaign-contract",
            str(contract),
        ),
        test_paths=(),
        environment=environment,
    )
    assert result.returncode == 0 and run["status"] == "PASS", (
        result.stdout + result.stderr
    )
    assert run["completed"] == json.loads(manifest.read_text())["nodeids"]
    assert inventory["collection_skips"][0]["longrepr"] == "collection\x01skip\x7f"
    folder = receipts / "batch-0001"
    tree = ET.parse(folder / "junit.xml")
    cases = tree.findall(".//testcase")
    runtime = next(
        case for case in cases if case.get("name", "").startswith("test_original_skip")
    )
    assert runtime.get("name") == "test_original_skip[illegal#x07#x7F]"
    assert runtime.find("skipped").get("message") == "original#x07skip#x7F"
    collection_skip = next(
        case.find("skipped")
        for case in cases
        if case.find("skipped") is not None
        and case.find("skipped").get("message") == "collection skipped"
    )
    assert collection_skip.text == "collection#x01skip#x7F"
    assert "type" not in collection_skip.attrib
    # A native collection skip cannot be re-labelled as an expected xfail.
    collection_skip.set("type", "pytest.xfail")
    tree.write(folder / "junit.xml")
    module = _runner()
    payload = json.loads(manifest.read_text())
    lane = module._CiLaneContext(manifest, suite, environment=payload["environment"])
    campaign = module._CiCampaignContext(contract, lane)
    receipt = json.loads((folder / "receipt.json").read_text())
    with pytest.raises(ValueError, match="JUnit skipped outcome differs"):
        campaign.batch(receipt, run["completed"], folder / "junit.xml")


def test_campaign_final_closure_rejects_semantically_unchanged_junit_mutation(
    tmp_path, monkeypatch
):
    suite, manifest, environment, contract, _ = _prepare_campaign_suite(
        tmp_path, monkeypatch, "pass"
    )
    module = _runner()
    original = module._CiCampaignContext.batch
    mutations = []

    def validate_then_mutate(self, receipt, assigned, junit_path):
        original(self, receipt, assigned, junit_path)
        mutations.append(junit_path)
        junit_path.write_bytes(junit_path.read_bytes() + b"\n")

    monkeypatch.setattr(module._CiCampaignContext, "batch", validate_then_mutate)
    monkeypatch.setattr(module.os, "environ", environment)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            str(_RUNNER),
            "--output-dir",
            str(tmp_path / "receipts"),
            "--ci-lane",
            "standard",
            "--ci-main-baseline",
            str(manifest),
            "--ci-campaign-contract",
            str(contract),
            "--",
            "-q",
        ],
    )
    assert module._main() == 1
    run = json.loads((tmp_path / "receipts/run.json").read_text())
    assert run["status"] == "FAIL"
    assert "closed canonical lane evidence changed" in run["failure"]["message"]
    assert run["completed"] == json.loads(manifest.read_text())["nodeids"]
    folder = tmp_path / "receipts/batch-0001"
    assert mutations == [folder / "junit.xml"]
    assert (
        run["ci_lane"]["closed_workers"][folder.name]["junit.xml"]
        != hashlib.sha256((folder / "junit.xml").read_bytes()).hexdigest()
    )
    native = json.loads((folder / "receipt.json").read_text())
    assert all(
        report["outcome"] == "passed"
        for reports in native["phases"].values()
        for report in reports
    )
    payload = json.loads(manifest.read_text())
    lane = module._CiLaneContext(manifest, suite, environment=payload["environment"])
    original(
        module._CiCampaignContext(contract, lane),
        native,
        run["completed"],
        folder / "junit.xml",
    )


def _prepare_provision_suite(tmp_path, monkeypatch, *, support=False):
    suite, manifest, environment, contract, inventory = _prepare_campaign_suite(
        tmp_path, monkeypatch, "allowed-skip", support=support
    )
    payload = json.loads(manifest.read_text())
    preparation_spec = spec_from_file_location(
        "_provision_fixture_preparation", _RUNNER.with_name("prepare_ci_standard.py")
    )
    preparation = module_from_spec(preparation_spec)
    preparation_spec.loader.exec_module(preparation)
    payload["packages"] = {
        preparation._name(distribution.metadata["Name"]): distribution.version
        for distribution in metadata.distributions()
    }
    plugins = [
        {
            "name": entry.name,
            "value": entry.value,
            "distribution": preparation._name(distribution.metadata["Name"]),
            "version": distribution.version,
        }
        for distribution in metadata.distributions()
        for entry in distribution.entry_points
        if entry.group == "pytest11"
    ]
    for name in ("poetry.lock", "pyproject.toml"):
        (suite / name).write_bytes((_RUNNER.parents[1] / name).read_bytes())
        payload["source_files"][name] = hashlib.sha256(
            (suite / name).read_bytes()
        ).hexdigest()
    transition = json.loads(json.dumps(payload["collection_transition"]))
    worker_environment = dict(payload["environment"])
    for mapping in (
        worker_environment,
        transition["collected_environment"],
        transition["post_pytest_environment"],
    ):
        mapping["JAX2ONNX_NATIVE_BINDINGS_EVIDENCE_DIR"] = (
            "${RUN}/native-model-artifacts"
        )
    data = {
        "schema": "jax2onnx-ci-standard-cohort-v1",
        "python_minor": payload["python_minor"],
        "baseline_commit": payload["baseline_commit"],
        "baseline": {"nodeids": payload["nodeids"], "generated": {}},
        "bindings": {
            name: payload["source_files"][name]
            for name in ("poetry.lock", "pyproject.toml")
        },
        "packages": payload["packages"],
        "bootstrap": {"pip": payload["packages"]["pip"]},
        "plugins": sorted(
            plugins, key=lambda value: (value["name"], value["distribution"])
        ),
        "environment": worker_environment,
        "collection_transition": transition,
        "allowed_skips": json.loads(contract.read_text())["allowed_skips"],
        "provenance": {},
    }
    if support:
        data.update(
            schema="jax2onnx-ci-support-cohort-v1",
            support_role="normalization",
            collection_args=payload["collection_args"],
        )
    cohort = suite / "cohort.json"
    cohort.write_text(json.dumps(data) + "\n")
    payload["source_files"]["cohort.json"] = hashlib.sha256(
        cohort.read_bytes()
    ).hexdigest()
    manifest.write_text(json.dumps(payload) + "\n")
    return suite, manifest, environment, cohort, inventory


@pytest.mark.parametrize(
    "fault",
    [None, "skip-mutation", "early-execution", "terminal", "subsequent-C", "read-race"],
)
def test_real_first_collection_provisions_one_unchanged_canonical_campaign(
    tmp_path, monkeypatch, fault
):
    module = _runner()
    suite, manifest, environment, cohort, inventory = _prepare_provision_suite(
        tmp_path, monkeypatch
    )
    monkeypatch.chdir(suite)
    monkeypatch.setattr(module.os, "environ", environment)
    original = module._child
    collections = []
    if fault == "read-race":
        original_read = module._context_json

        def transient_read(path, *args, **kwargs):
            data, raw = original_read(path, *args, **kwargs)
            if path == cohort:
                data["allowed_skips"][0]["reason"] = "transient unreviewed reason"
                raw = (json.dumps(data) + "\n").encode()
            return data, raw

        monkeypatch.setattr(module, "_context_json", transient_read)

    def child(folder, request, *args, **kwargs):
        result = original(folder, request, *args, **kwargs)
        if request["kind"] == "collection":
            collections.append(folder)
            if fault == "skip-mutation":
                data = json.loads(cohort.read_text())
                data["allowed_skips"][0]["reason"] = "forged"
                cohort.write_text(json.dumps(data) + "\n")
            elif fault == "terminal":
                terminal = folder / "lane-terminal.json"
                data = json.loads(terminal.read_text())
                data["exitstatus"] = 1
                terminal.write_text(json.dumps(data) + "\n")
            elif fault == "early-execution":
                receipt = folder / "receipt.json"
                data = json.loads(receipt.read_text())
                data["phases"] = {
                    inventory["full_preselection_ids"][0]: [
                        {"when": "call", "outcome": "passed"}
                    ]
                }
                receipt.write_text(json.dumps(data) + "\n")
                terminal = folder / "lane-terminal.json"
                data = json.loads(terminal.read_text())
                data["receipt_sha256"] = hashlib.sha256(
                    receipt.read_bytes()
                ).hexdigest()
                terminal.write_text(json.dumps(data) + "\n")
        elif fault == "subsequent-C":
            receipt = folder / "receipt.json"
            data = json.loads(receipt.read_text())
            data["full_preselection_ids"] = data["full_preselection_ids"][:-1]
            receipt.write_text(json.dumps(data) + "\n")
            terminal = folder / "lane-terminal.json"
            data = json.loads(terminal.read_text())
            data["receipt_sha256"] = hashlib.sha256(receipt.read_bytes()).hexdigest()
            terminal.write_text(json.dumps(data) + "\n")
        return result

    monkeypatch.setattr(module, "_child", child)
    output = tmp_path / "receipts"
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            str(_RUNNER),
            "--output-dir",
            str(output),
            "--ci-lane",
            "standard",
            "--ci-main-baseline",
            str(manifest),
            "--_ci-provision-campaign",
            str(cohort),
            "--",
            "-q",
        ],
    )
    assert module._main() == (0 if fault is None else 1)
    run = json.loads((output / "run.json").read_text())
    assert collections == ([] if fault == "read-race" else [output / "collection"])
    if fault is None:
        assert run["completed"] == json.loads(manifest.read_text())["nodeids"]
        assert len(run["processes"]) == 2
        contract = json.loads((output / "ci-provisioned-campaign.json").read_text())
        raw = (output / "collection/receipt.json").read_bytes()
        assert contract["global_inventory"]["sha256"] == hashlib.sha256(raw).hexdigest()
        assert (
            json.loads(raw)["full_preselection_ids"]
            == inventory["full_preselection_ids"]
        )
        assert run["ci_lane"]["extra_ids"] == ["test_cases.py::test_extra"]
        assert not (suite / "extra-ran").exists()
    else:
        assert run["status"] == "FAIL"
        if fault != "subsequent-C":
            assert len(run["processes"]) == (0 if fault == "read-race" else 1)


def test_provision_authority_conflicts_are_rejected_before_children(
    tmp_path, monkeypatch
):
    module = _runner()
    monkeypatch.setattr(
        module.sys,
        "argv",
        [str(_RUNNER), "--ci-campaign-contract=a", "--_ci-provision-campaign=b"],
    )
    with pytest.raises(SystemExit) as error:
        module._main()
    assert error.value.code == 2


@pytest.mark.parametrize(
    "provision",
    [False, True],
    ids=[
        "contract",
        (
            "first-collection"
            if sys.version_info[:2] in {(3, 12), (3, 13)}
            else "unsupported-normalization-role"
        ),
    ],
)
def test_real_support_campaign_preserves_original_command_universe(
    tmp_path, monkeypatch, provision
):
    module = _runner()
    if provision:
        suite, manifest, environment, authority, inventory = _prepare_provision_suite(
            tmp_path, monkeypatch, support=True
        )
        option = "--_ci-provision-campaign"
    else:
        suite, manifest, environment, authority, inventory = _prepare_campaign_suite(
            tmp_path, monkeypatch, "allowed-skip", support=True
        )
        option = "--ci-campaign-contract"
    selectors = json.loads(manifest.read_text())["collection_args"]
    monkeypatch.chdir(suite)
    monkeypatch.setattr(module.os, "environ", environment)
    output = tmp_path / "receipts"
    arguments = [
        str(_RUNNER),
        "--output-dir",
        str(output),
        "--ci-lane",
        "standard",
        "--ci-main-baseline",
        str(manifest),
        option,
        str(authority),
    ]
    for selector in selectors:
        arguments.extend(("--test-path", selector))
    monkeypatch.setattr(module.sys, "argv", [*arguments, "--", "-q"])
    if provision and sys.version_info[:2] not in {(3, 12), (3, 13)}:
        cohort = json.loads(authority.read_text())
        assert cohort["support_role"] == "normalization"
        assert (
            cohort["python_minor"]
            == f"{sys.version_info.major}.{sys.version_info.minor}"
        )
        children = []

        def forbidden_child(*args, **kwargs):
            children.append((args, kwargs))
            raise AssertionError("unsupported role launched a canonical child")

        monkeypatch.setattr(module, "_child", forbidden_child)
        assert module._main() == 1
        run = json.loads((output / "run.json").read_text())
        assert run["status"] == "FAIL"
        assert run["failure"] == {
            "type": "ValueError",
            "message": "unsupported support role/cohort or ordered selectors",
        }
        assert run["processes"] == run["completed"] == children == []
        assert not (output / "collection").exists()
        assert not list(output.glob("batch-*"))
        return
    assert module._main() == 0
    run = json.loads((output / "run.json").read_text())
    assert run["nodeids"] == inventory["full_preselection_ids"]
    assert run["completed"] == json.loads(manifest.read_text())["nodeids"]
    assert run["ci_lane"]["extra_ids"] == ["test_cases.py::test_extra"]
    assert len(run["processes"]) == 2
    assert not (suite / "extra-ran").exists()
    for folder in ("collection", "batch-0001"):
        receipt = json.loads((output / folder / "receipt.json").read_text())
        assert (
            receipt["collection_args"]
            == receipt["qualified_collection_args"]
            == selectors
        )
        assert receipt["full_preselection_ids"] == run["nodeids"]
        assert receipt["collection_skips"] == inventory["collection_skips"]
    if provision:
        contract = json.loads((output / "ci-provisioned-campaign.json").read_text())
        assert contract["schema"] == "jax2onnx-ci-support-campaign-v1"
        assert "global_inventory" not in contract
        assert (
            contract["command_inventory"]["sha256"]
            == hashlib.sha256(
                (output / "collection/receipt.json").read_bytes()
            ).hexdigest()
        )


@pytest.mark.parametrize(
    "fault", ["orphan", "extra-lane", "test-path", "positional", "keyword", "addopts"]
)
def test_campaign_admission_rejects_prefilters_before_any_child(
    tmp_path, monkeypatch, capsys, fault
):
    module = _runner()
    suite, manifest, _ = _prepare_lane_suite(
        tmp_path,
        {"test_cases.py": "def test_base(): pass\n"},
        ["test_cases.py::test_base"],
    )
    contract = tmp_path / "not-read-campaign.json"
    contract.write_text("{}\n")
    monkeypatch.chdir(suite)
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k base" if fault == "addopts" else "")
    arguments = [
        str(_RUNNER),
        "--output-dir",
        str(tmp_path / "campaign-output"),
        "--ci-campaign-contract",
        str(contract),
    ]
    if fault != "orphan":
        arguments.extend(
            (
                "--ci-lane",
                "extra" if fault == "extra-lane" else "standard",
                "--ci-main-baseline",
                str(manifest),
            )
        )
    if fault == "test-path":
        arguments.extend(("--test-path", "test_cases.py"))
    arguments.extend(("--", "-q"))
    if fault == "positional":
        arguments.append("test_cases.py::test_base")
    elif fault == "keyword":
        arguments.extend(("-k", "base"))
    monkeypatch.setattr(module.sys, "argv", arguments)
    children = []

    def forbidden_child(*args, **kwargs):
        children.append((args, kwargs))
        raise AssertionError("campaign admission launched a child")

    monkeypatch.setattr(module, "_child", forbidden_child)
    with pytest.raises(SystemExit) as error:
        module._main()
    assert error.value.code == 2
    assert "campaign" in capsys.readouterr().err
    assert children == []


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
