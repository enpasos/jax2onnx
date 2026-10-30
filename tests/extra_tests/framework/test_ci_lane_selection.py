# tests/extra_tests/framework/test_ci_lane_selection.py

"""Unused lane selection preserves full collection and native phase accounting."""

import copy
import hashlib
from importlib import metadata
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest


_ROOT = Path(__file__).resolve().parents[3]
_RUNNER = _ROOT / "scripts/run_pytest_batches.py"


def _runner() -> Any:
    spec = spec_from_file_location("_ci_lane_runner_under_test", _RUNNER)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _suite(
    tmp_path: Path,
    files: dict[str, str],
    baseline: list[str],
    *,
    collection_args: list[str] | None = None,
) -> tuple[Path, Path, dict[str, str]]:
    suite = tmp_path / "suite"
    suite.mkdir()
    sources = {
        "pytest.ini": "[pytest]\ntestpaths = .\nmarkers = extra: added coverage\n",
        **files,
    }
    for name, source in sources.items():
        path = suite / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(_ROOT), str(suite)))
    environment["PYTEST_ADDOPTS"] = ""
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    manifest = tmp_path / "baseline.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "jax2onnx-ci-baseline-v1",
                "baseline_commit": "a" * 40,
                "python_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
                "nodeids": baseline,
                "source_files": {
                    name: hashlib.sha256((suite / name).read_bytes()).hexdigest()
                    for name in sources
                },
                "packages": {"pytest": metadata.version("pytest")},
                "environment": _runner()._ci_lane_environment(environment),
                **(
                    {"collection_args": collection_args}
                    if collection_args is not None
                    else {}
                ),
            }
        )
    )
    return suite, manifest, environment


def _run(
    suite: Path,
    manifest: Path,
    environment: dict[str, str],
    report: Path,
    *,
    lane: str | None = "standard",
    options: tuple[str, ...] = (),
    invocation_dir: Path | None = None,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any] | None]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "scripts.run_pytest_batches",
    ]
    if lane is not None:
        command.extend(
            [
                f"--ci-lane={lane}",
                f"--ci-main-baseline={manifest}",
                f"--ci-lane-report={report}",
            ]
        )
    result = subprocess.run(
        [*command, *options],
        cwd=suite if invocation_dir is None else invocation_dir,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    receipt = report / "receipt.json"
    return result, json.loads(receipt.read_text()) if receipt.is_file() else None


def _assert_failed_before_body(result, receipt, suite):
    assert result.returncode != 0, result.stdout + result.stderr
    assert not (suite / "body-ran").exists()
    if receipt is not None:
        assert receipt["status"] != "PASS"
        assert not receipt["phases"]


_BODY = "from pathlib import Path\ndef test_base():\n    Path('body-ran').touch()\n"
_EXTRA = """import pytest
@pytest.mark.extra
def test_extra():
    from pathlib import Path
    Path('body-ran').touch()
"""


@pytest.mark.parametrize("collection_args", [None, []], ids=["absent", "empty"])
def test_standard_and_extra_project_same_complete_ordered_collection(
    tmp_path, collection_args
):
    baseline = ["test_a_consumer.py::test_base"]
    suite, manifest, environment = _suite(
        tmp_path,
        {
            "test_a_consumer.py": """import builtins
def test_base():
    assert builtins._ci_lane_import_effect == 'present'
""",
            "test_z_registration.py": """import builtins
builtins._ci_lane_import_effect = 'present'
def test_extra():
    pass
""",
        },
        baseline,
        collection_args=collection_args,
    )
    records = []
    for lane in ("standard", "extra"):
        result, receipt = _run(suite, manifest, environment, tmp_path / lane, lane=lane)
        assert result.returncode == 0, result.stdout + result.stderr
        assert receipt is not None and receipt["status"] == "PASS"
        assert receipt["qualified_collection_args"] == receipt["collection_args"] == []
        assert (
            receipt["manifest_sha256"]
            == hashlib.sha256(manifest.read_bytes()).hexdigest()
        )
        assert receipt["completed"] == receipt["selected_ids"]
        assert set(receipt["phases"]) == set(receipt["selected_ids"])
        assert not receipt["collection_errors"] and not receipt["errors"]
        records.append(receipt)
    standard, extra = records
    assert (
        standard["full_preselection_ids"]
        == extra["full_preselection_ids"]
        == [
            *baseline,
            "test_z_registration.py::test_extra",
        ]
    )
    assert standard["selected_ids"] == baseline
    assert extra["selected_ids"] == standard["deselected_ids"]
    assert extra["deselected_ids"] == baseline
    assert set(standard["selected_ids"]).isdisjoint(extra["selected_ids"])
    assert set(standard["selected_ids"] + extra["selected_ids"]) == set(
        standard["full_preselection_ids"]
    )


def test_ordinary_ordering_precedes_projection_and_preserves_baseline_order(tmp_path):
    baseline = ["test_cases.py::test_base_b", "test_cases.py::test_base_a"]
    suite, manifest, environment = _suite(
        tmp_path,
        {
            "test_cases.py": """def test_base_a(): pass
def test_extra(): pass
def test_base_b(): pass
""",
            "conftest.py": """def pytest_collection_modifyitems(items):
    items.sort(key=lambda item: item.nodeid, reverse=True)
""",
        },
        baseline,
    )
    result, receipt = _run(suite, manifest, environment, tmp_path / "report")
    assert result.returncode == 0, result.stdout + result.stderr
    assert receipt["selected_ids"] == baseline
    assert receipt["full_preselection_ids"] == ["test_cases.py::test_extra", *baseline]


@pytest.mark.parametrize(
    "fault", ["missing-baseline", "changed-source", "duplicate-id"]
)
def test_missing_changed_or_duplicate_contract_fails_before_body(tmp_path, fault):
    files = {"test_cases.py": _BODY}
    if fault == "duplicate-id":
        files[
            "conftest.py"
        ] = """def pytest_collection_modifyitems(items):
    items.append(items[0])
"""
    baseline = [
        (
            "test_cases.py::test_missing"
            if fault == "missing-baseline"
            else "test_cases.py::test_base"
        )
    ]
    suite, manifest, environment = _suite(tmp_path, files, baseline)
    if fault == "changed-source":
        (suite / "test_cases.py").write_text(_BODY + "# unqualified change\n")
    result, receipt = _run(suite, manifest, environment, tmp_path / "report")
    _assert_failed_before_body(result, receipt, suite)


@pytest.mark.parametrize(
    "options",
    [
        ("test_cases.py",),
        ("test_cases.py::test_base",),
        ("-k", "not extra"),
        ("-m", "not extra"),
        ("--ignore", "test_extra.py"),
        ("--ignore-glob", "*extra*"),
        ("--deselect", "test_extra.py::test_extra"),
        ("--lf",),
        ("--ff",),
        ("--nf",),
        ("--noconftest",),
        ("--confcutdir", "."),
        ("-o", "python_files=test_cases.py"),
        ("--override-ini", "addopts=-k not extra"),
    ],
)
def test_effective_prefilters_reject_even_when_only_extra_would_disappear(
    tmp_path, options
):
    suite, manifest, environment = _suite(
        tmp_path,
        {"test_cases.py": _BODY, "test_extra.py": _EXTRA},
        ["test_cases.py::test_base"],
    )
    result, receipt = _run(
        suite, manifest, environment, tmp_path / "report", options=options
    )
    _assert_failed_before_body(result, receipt, suite)


def test_unreported_conftest_extra_deselection_is_rejected(tmp_path):
    suite, manifest, environment = _suite(
        tmp_path,
        {
            "test_cases.py": _BODY,
            "test_extra.py": _EXTRA,
            "conftest.py": """def pytest_collection_modifyitems(items):
    items[:] = [item for item in items if 'extra' not in item.nodeid]
""",
        },
        ["test_cases.py::test_base"],
    )
    result, receipt = _run(suite, manifest, environment, tmp_path / "report")
    _assert_failed_before_body(result, receipt, suite)


def test_manifest_order_is_checked_instead_of_repaired(tmp_path):
    suite, manifest, environment = _suite(
        tmp_path,
        {"test_cases.py": _BODY + "def test_second():\n    pass\n"},
        ["test_cases.py::test_second", "test_cases.py::test_base"],
    )
    result, receipt = _run(suite, manifest, environment, tmp_path / "report")
    _assert_failed_before_body(result, receipt, suite)


def test_native_skip_xfail_rejection_and_collection_skip_outcomes_are_retained(
    tmp_path,
):
    baseline = [
        "test_cases.py::test_expected_rejection",
        "test_cases.py::test_skip",
        "test_cases.py::test_xfail",
    ]
    suite, manifest, environment = _suite(
        tmp_path,
        {
            "test_cases.py": """import pytest
def test_expected_rejection():
    with pytest.raises(ValueError):
        raise ValueError('expected rejection')
@pytest.mark.skip(reason='original runtime skip')
def test_skip():
    pass
@pytest.mark.xfail(strict=True, reason='original expected failure')
def test_xfail():
    assert False
""",
            "test_collection_skip.py": """import pytest
pytest.skip('original collection skip', allow_module_level=True)
""",
        },
        baseline,
    )
    result, receipt = _run(suite, manifest, environment, tmp_path / "report")
    assert result.returncode == 0, result.stdout + result.stderr
    assert receipt["status"] == "PASS" and receipt["completed"] == baseline
    assert len(receipt["collection_skips"]) == 1
    assert not receipt["collection_errors"]
    assert receipt["phases"][baseline[0]][1]["outcome"] == "passed"
    assert receipt["phases"][baseline[1]][0]["outcome"] == "skipped"
    assert receipt["phases"][baseline[2]][1]["outcome"] == "skipped"
    assert "wasxfail" in receipt["phases"][baseline[2]][1]
    assert set(receipt["phase_durations"]) == set(baseline)


@pytest.mark.parametrize("stage", ["setup", "call", "teardown"])
def test_each_native_failing_phase_prevents_lane_success(tmp_path, stage):
    files = {
        "setup": """import pytest
@pytest.fixture
def resource():
    assert False, 'setup failure'
def test_base(resource):
    pass
""",
        "call": "def test_base():\n    assert False, 'call failure'\n",
        "teardown": """import pytest
@pytest.fixture
def resource():
    yield
    assert False, 'teardown failure'
def test_base(resource):
    pass
""",
    }
    suite, manifest, environment = _suite(
        tmp_path, {"test_cases.py": files[stage]}, ["test_cases.py::test_base"]
    )
    result, receipt = _run(suite, manifest, environment, tmp_path / "report")
    assert result.returncode != 0
    assert receipt["status"] == "FAIL"
    assert any(
        report["when"] == stage and report["outcome"] == "failed"
        for reports in receipt["phases"].values()
        for report in reports
    )


def test_collection_error_prevents_every_test_body(tmp_path):
    suite, manifest, environment = _suite(
        tmp_path,
        {"test_cases.py": _BODY, "test_broken.py": "def syntax error\n"},
        ["test_cases.py::test_base"],
    )
    result, receipt = _run(suite, manifest, environment, tmp_path / "report")
    _assert_failed_before_body(result, receipt, suite)
    assert receipt["collection_errors"]


@pytest.mark.parametrize("fault", ["protocol-without-reports", "process-crash"])
def test_missing_execution_phases_or_process_crash_cannot_become_pass(tmp_path, fault):
    files = {"test_cases.py": "def test_base():\n    pass\n"}
    if fault == "protocol-without-reports":
        files[
            "conftest.py"
        ] = """def pytest_runtest_protocol(item, nextitem):
    return True
"""
    else:
        files["test_cases.py"] = "import os\ndef test_base():\n    os._exit(17)\n"
    suite, manifest, environment = _suite(tmp_path, files, ["test_cases.py::test_base"])
    result, receipt = _run(suite, manifest, environment, tmp_path / "report")
    assert result.returncode != 0
    assert receipt["status"] != "PASS"
    assert "test_cases.py::test_base" not in receipt["completed"]


def test_late_collection_finish_mutation_is_rejected_before_body(tmp_path):
    suite, manifest, environment = _suite(
        tmp_path,
        {
            "test_cases.py": _BODY,
            "conftest.py": """def pytest_collection_finish(session):
    session.items[:] = []
""",
        },
        ["test_cases.py::test_base"],
    )
    result, receipt = _run(suite, manifest, environment, tmp_path / "report")
    _assert_failed_before_body(result, receipt, suite)


def test_lane_report_directory_cannot_be_reused_or_overwritten(tmp_path):
    suite, manifest, environment = _suite(
        tmp_path,
        {"test_cases.py": "def test_base(): pass\n"},
        ["test_cases.py::test_base"],
    )
    folder = tmp_path / "report"
    result, receipt = _run(suite, manifest, environment, folder)
    assert result.returncode == 0, result.stdout + result.stderr
    assert receipt["status"] == "PASS"
    before = (folder / "receipt.json").read_bytes()
    again, _ = _run(suite, manifest, environment, folder)
    assert again.returncode != 0
    assert (folder / "receipt.json").read_bytes() == before


def test_loading_plugin_without_lane_options_preserves_default_collection(tmp_path):
    suite, manifest, environment = _suite(
        tmp_path,
        {"test_cases.py": "def test_base(): pass\ndef test_extra(): pass\n"},
        ["test_cases.py::test_base"],
    )
    result, receipt = _run(
        suite, manifest, environment, tmp_path / "unused-report", lane=None
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout
    assert receipt is None and not (tmp_path / "unused-report").exists()


@pytest.mark.skipif(
    os.environ.get("JAX2ONNX_TEST_CI_LANE_XDIST") != "1",
    reason="assigned to the opt-in synthetic xdist qualification lane",
)
def test_opt_in_xdist_projects_before_worker_ids_are_published(tmp_path):
    baseline = ["test_cases.py::test_base_a", "test_cases.py::test_base_b"]
    suite, manifest, environment = _suite(
        tmp_path,
        {
            "test_cases.py": """def test_base_a(): pass
def test_base_b(): pass
def test_extra():
    raise AssertionError('extra must never reach the xdist scheduler')
"""
        },
        baseline,
        collection_args=["test_cases.py"],
    )
    result, receipt = _run(
        suite,
        manifest,
        environment,
        tmp_path / "report",
        options=("-p", "xdist.plugin", "-n", "2", "test_cases.py"),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert receipt["status"] == "PASS"
    assert receipt["selected_ids"] == baseline
    assert set(receipt["phases"]) == set(baseline)
    assert set(receipt["completed"]) == set(baseline)
    assert all(
        [phase["when"] for phase in receipt["phases"][nodeid]]
        == ["setup", "call", "teardown"]
        for nodeid in baseline
    )

    workers = list((tmp_path / "report" / "workers").glob("*/receipt.json"))
    assert len(workers) == 2
    for path in workers:
        worker = json.loads(path.read_text())
        assert worker["qualified_collection_args"] == ["test_cases.py"]
        assert worker["collection_args"] == ["test_cases.py"]


@pytest.mark.parametrize("fault", ["package-version", "python-cohort", "environment"])
def test_unqualified_runtime_context_is_rejected_before_body(tmp_path, fault):
    suite, manifest, environment = _suite(
        tmp_path, {"test_cases.py": _BODY}, ["test_cases.py::test_base"]
    )
    contract = json.loads(manifest.read_text())
    if fault == "package-version":
        contract["packages"]["pytest"] = "0.unqualified"
    elif fault == "python-cohort":
        contract["python_minor"] = "3.99"
    else:
        contract["environment"]["PYTHONPATH"] = "unqualified-context"
    manifest.write_text(json.dumps(contract))
    result, receipt = _run(suite, manifest, environment, tmp_path / "report")
    _assert_failed_before_body(result, receipt, suite)


@pytest.mark.parametrize(
    "unbound", ["test_cases.py", "conftest.py", "helper.py", "pytest.ini"]
)
def test_actual_local_contract_sources_cannot_be_omitted(tmp_path, unbound):
    suite, manifest, environment = _suite(
        tmp_path,
        {
            "test_cases.py": "import helper\n" + _BODY,
            "conftest.py": "def pytest_configure(config):\n    pass\n",
            "helper.py": "MARKER = 'contract dependency'\n",
        },
        ["test_cases.py::test_base"],
    )
    contract = json.loads(manifest.read_text())
    del contract["source_files"][unbound]
    manifest.write_text(json.dumps(contract))
    result, receipt = _run(suite, manifest, environment, tmp_path / "report")
    _assert_failed_before_body(result, receipt, suite)


def _source_audit_context(tmp_path, monkeypatch):
    module = _runner()
    suite, manifest, _ = _suite(
        tmp_path, {"test_cases.py": _BODY}, ["test_cases.py::test_base"]
    )
    payload = json.loads(manifest.read_text())
    payload["environment"] = module._ci_lane_environment(os.environ)
    manifest.write_text(json.dumps(payload))
    context = module._CiLaneContext(manifest, suite)
    observed_sys = type(sys)("audit-sys")
    observed_sys.version_info = sys.version_info
    observed_sys.modules = {}
    monkeypatch.setattr(module, "sys", observed_sys)
    config = SimpleNamespace(
        rootpath=suite,
        option=SimpleNamespace(),
        inipath=suite / "pytest.ini",
        getini=lambda name: [],
    )
    return module, context, config


def test_source_audit_deduplicates_exact_raw_paths_only_within_one_audit(
    tmp_path, monkeypatch
):
    module, context, config = _source_audit_context(tmp_path, monkeypatch)
    owner = context.root / "test_cases.py"
    alternate = str(context.root) + "/./test_cases.py"
    module.sys.modules.update(
        ordinary=SimpleNamespace(__file__=str(owner)),
        lexical_alias=SimpleNamespace(__file__=alternate),
    )
    module.sys.modules["tests.owner"] = SimpleNamespace(__file__=str(owner))
    calls = []
    original = context.check_source

    def checked(path):
        calls.append(os.fspath(path))
        return original(path)

    monkeypatch.setattr(context, "check_source", checked)
    items = [SimpleNamespace(path=owner) for _ in range(8)]
    # Mandatory manifest/hash resolutions remain independent of this count.
    for _ in range(2):
        calls.clear()
        assert context.check(config, items) == context.sources
        assert calls == [str(owner), alternate]
    assert original(None) is None
    assert original(owner) == owner.resolve()


@pytest.mark.parametrize("name", ["tests.alias", "jax2onnx", "jax2onnx.alias"])
def test_source_audit_preserves_protected_namespace_checks_on_memo_hits(
    tmp_path, monkeypatch, name
):
    module, context, config = _source_audit_context(tmp_path, monkeypatch)
    outside = tmp_path / "external.py"
    outside.write_text("VALUE = 1\n")
    module.sys.modules["ordinary"] = SimpleNamespace(__file__=str(outside))
    assert context.check(config) == context.sources
    module.sys.modules[name] = SimpleNamespace(__file__=str(outside))
    calls = []
    original = context.check_source

    def checked(path):
        calls.append(os.fspath(path))
        return original(path)

    monkeypatch.setattr(context, "check_source", checked)
    with pytest.raises(ValueError, match="test/converter module resolves outside"):
        context.check(config)
    assert calls == [str(outside)]


@pytest.mark.parametrize("target", ["outside", "unbound"])
def test_source_audit_resolves_retargeted_symlink_in_next_audit(
    tmp_path, monkeypatch, target
):
    module, context, config = _source_audit_context(tmp_path, monkeypatch)
    alias = context.root / "alias.py"
    alias.symlink_to(context.root / "test_cases.py")
    module.sys.modules["tests.alias"] = SimpleNamespace(__file__=str(alias))
    assert context.check(config) == context.sources
    replacement = (
        tmp_path / "outside.py" if target == "outside" else context.root / "unbound.py"
    )
    replacement.write_text("VALUE = 1\n")
    alias.unlink()
    alias.symlink_to(replacement)
    message = (
        "test/converter module resolves outside"
        if target == "outside"
        else "unbound collected owner or dependency source"
    )
    with pytest.raises(ValueError, match=message):
        context.check(config)


def test_collect_only_inventory_does_not_claim_execution_success(tmp_path):
    suite, manifest, environment = _suite(
        tmp_path, {"test_cases.py": _BODY}, ["test_cases.py::test_base"]
    )
    result, receipt = _run(
        suite, manifest, environment, tmp_path / "report", options=("--collect-only",)
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert receipt["status"] == "COLLECTED"
    assert receipt["selected_ids"] == ["test_cases.py::test_base"]
    assert not receipt["completed"] and not receipt["phases"]
    assert not (suite / "body-ran").exists()


@pytest.mark.parametrize(
    "distributed",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                os.environ.get("JAX2ONNX_TEST_CI_LANE_XDIST") != "1",
                reason="assigned to the opt-in synthetic xdist qualification lane",
            ),
        ),
    ],
    ids=["serial", "opt-in-xdist"],
)
def test_lazy_body_import_final_audit_fails_after_passed_native_reports(
    tmp_path, distributed
):
    baseline = ["test_cases.py::test_base"]
    suite, manifest, environment = _suite(
        tmp_path,
        {
            "test_cases.py": """from pathlib import Path
def test_base():
    import helper
    assert helper.MARKER == 'late contract dependency'
    Path('body-ran').touch()
""",
            "helper.py": "MARKER = 'late contract dependency'\n",
        },
        baseline,
    )
    contract = json.loads(manifest.read_text())
    del contract["source_files"]["helper.py"]
    manifest.write_text(json.dumps(contract))
    options = ("-p", "xdist.plugin", "-n", "2") if distributed else ()
    result, receipt = _run(
        suite, manifest, environment, tmp_path / "report", options=options
    )
    assert (suite / "body-ran").exists()
    assert result.returncode != 0, result.stdout + result.stderr
    assert receipt["status"] == "FAIL"
    assert receipt["completed"] == baseline
    assert [phase["when"] for phase in receipt["phases"][baseline[0]]] == [
        "setup",
        "call",
        "teardown",
    ]
    assert all(phase["outcome"] == "passed" for phase in receipt["phases"][baseline[0]])
    assert "helper.py" in json.dumps(receipt)


def test_collect_only_late_import_audit_failure_is_nonzero_and_not_collected(tmp_path):
    suite, manifest, environment = _suite(
        tmp_path,
        {
            "test_cases.py": _BODY,
            "helper.py": "MARKER = 'late collection dependency'\n",
            "conftest.py": """def pytest_collection_finish(session):
    import helper
    assert helper.MARKER == 'late collection dependency'
""",
        },
        ["test_cases.py::test_base"],
    )
    contract = json.loads(manifest.read_text())
    del contract["source_files"]["helper.py"]
    manifest.write_text(json.dumps(contract))
    result, receipt = _run(
        suite, manifest, environment, tmp_path / "report", options=("--collect-only",)
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert receipt["status"] == "FAIL"
    assert not receipt["phases"] and not receipt["completed"]
    assert not (suite / "body-ran").exists()
    assert "helper.py" in json.dumps(receipt["errors"])


_TERMINAL_WORKER_MUTATION = """import json
from pathlib import Path
import time
import pytest

@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_sessionfinish(session, exitstatus):
    yield
    worker = getattr(session.config, 'workerinput', None)
    if worker is None:
        return
    report_root = Path(worker['ci_lane']['report_root'])
    expected = {'test_cases.py::test_base_a', 'test_cases.py::test_base_b'}
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            events = [json.loads(line) for line in
                      (report_root / 'events.jsonl').read_text().splitlines()]
        except (FileNotFoundError, ValueError):
            events = []
        if expected <= {event.get('nodeid') for event in events
                        if event.get('when') == 'teardown'}:
            break
        time.sleep(0.01)
    else:
        raise AssertionError('controller did not retain all native phase reports')
    path = report_root / 'workers' / worker['workerid'] / 'receipt.json'
    terminal = json.loads(path.read_text())
    assert terminal['finished'] and not terminal['errors']
    fault = '__FAULT__'
    if fault == 'missing':
        path.unlink()
    else:
        if fault == 'failed':
            terminal['status'] = 'FAIL'
            terminal['errors'].append('terminal worker witness')
        else:
            terminal['manifest_sha256'] = 'f' * 64
        path.write_text(json.dumps(terminal))
    session.exitstatus = 0
"""


@pytest.mark.skipif(
    os.environ.get("JAX2ONNX_TEST_CI_LANE_XDIST") != "1",
    reason="assigned to the opt-in synthetic xdist qualification lane",
)
@pytest.mark.parametrize("fault", ["failed", "missing", "manifest"])
def test_opt_in_xdist_ingests_actual_terminal_worker_receipt_after_passed_reports(
    tmp_path, fault
):
    baseline = ["test_cases.py::test_base_a", "test_cases.py::test_base_b"]
    suite, manifest, environment = _suite(
        tmp_path,
        {
            "test_cases.py": "def test_base_a(): pass\ndef test_base_b(): pass\n",
            "conftest.py": _TERMINAL_WORKER_MUTATION.replace("__FAULT__", fault),
        },
        baseline,
    )
    result, receipt = _run(
        suite,
        manifest,
        environment,
        tmp_path / "report",
        options=("-p", "xdist.plugin", "-n", "2"),
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert receipt["status"] == "FAIL"
    assert not receipt["crashes"]
    assert receipt["errors"]
    assert (
        receipt["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    )
    assert set(receipt["completed"]) == set(baseline)
    for nodeid in baseline:
        assert [phase["when"] for phase in receipt["phases"][nodeid]] == [
            "setup",
            "call",
            "teardown",
        ]
        assert all(phase["outcome"] == "passed" for phase in receipt["phases"][nodeid])
    if fault == "failed":
        assert "terminal worker witness" in json.dumps(receipt)


def test_qualified_mixed_file_and_exact_case_context_projects_both_lanes(tmp_path):
    qualified = ["test_selected.py", "test_cases.py::test_exact[chosen]"]
    baseline = ["test_selected.py::test_base", qualified[1]]
    suite, manifest, environment = _suite(
        tmp_path,
        {
            "test_selected.py": """def test_base(): pass
def test_extra(): pass
""",
            "test_cases.py": """import pytest
@pytest.mark.parametrize('value', [1, 2], ids=['chosen', 'sibling'])
def test_exact(value):
    assert value == 1, 'unselected parameter sibling executed'
def test_unselected_sibling():
    raise AssertionError('unselected sibling executed')
""",
            "test_outside.py": "raise AssertionError('outside context imported')\n",
        },
        baseline,
        collection_args=qualified,
    )
    records = []
    for lane in ("standard", "extra"):
        result, receipt = _run(
            suite,
            manifest,
            environment,
            tmp_path / lane,
            lane=lane,
            options=tuple(qualified),
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert receipt["status"] == "PASS"
        assert receipt["qualified_collection_args"] == qualified
        assert receipt["collection_args"] == qualified
        records.append(receipt)
    standard, extra = records
    assert (
        standard["full_preselection_ids"]
        == extra["full_preselection_ids"]
        == [baseline[0], "test_selected.py::test_extra", baseline[1]]
    )
    assert standard["selected_ids"] == baseline
    assert extra["selected_ids"] == ["test_selected.py::test_extra"]
    assert standard["deselected_ids"] == extra["selected_ids"]
    assert extra["deselected_ids"] == baseline


@pytest.mark.parametrize("fault", ["missing", "reordered", "injected"])
def test_collection_arguments_must_equal_qualified_ordered_context(tmp_path, fault):
    qualified = ["test_cases.py", "test_other.py::test_exact"]
    suite, manifest, environment = _suite(
        tmp_path,
        {
            "test_cases.py": _BODY,
            "test_other.py": "def test_exact():\n    pass\n",
            "test_extra.py": """from pathlib import Path
Path('unexpected-import').touch()
def test_extra(): pass
""",
        },
        ["test_cases.py::test_base", qualified[1]],
        collection_args=qualified,
    )
    actual = {
        "missing": qualified[:1],
        "reordered": list(reversed(qualified)),
        "injected": [*qualified, "test_extra.py"],
    }[fault]
    result, receipt = _run(
        suite, manifest, environment, tmp_path / "report", options=tuple(actual)
    )
    _assert_failed_before_body(result, receipt, suite)
    assert not (suite / "unexpected-import").exists()
    assert receipt["qualified_collection_args"] == qualified
    assert receipt["collection_args"] == actual


def test_qualified_file_selector_rejects_different_invocation_directory(tmp_path):
    suite, manifest, environment = _suite(
        tmp_path,
        {"test_cases.py": _BODY, "nested/test_cases.py": _BODY},
        ["nested/test_cases.py::test_base"],
        collection_args=["test_cases.py"],
    )
    result, receipt = _run(
        suite,
        manifest,
        environment,
        tmp_path / "report",
        options=("test_cases.py",),
        invocation_dir=suite / "nested",
    )
    _assert_failed_before_body(result, receipt, suite)
    assert not (suite / "nested" / "body-ran").exists()


def test_qualified_arguments_rechecked_after_collection_hooks(tmp_path):
    suite, manifest, environment = _suite(
        tmp_path,
        {
            "test_cases.py": _BODY,
            "conftest.py": """def pytest_collection_modifyitems(config, items):
    config.option.file_or_dir = ['test_other.py']
""",
        },
        ["test_cases.py::test_base"],
        collection_args=["test_cases.py"],
    )
    result, receipt = _run(
        suite, manifest, environment, tmp_path / "report", options=("test_cases.py",)
    )
    _assert_failed_before_body(result, receipt, suite)
    assert "changed after qualification" in json.dumps(receipt["errors"])


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "not-list",
        "non-string",
        "wrong-schema",
        "outside",
        "absolute",
        "parent",
        "glob",
        "directory",
        "helper",
        "unbound",
        "missing",
        "option",
    ],
)
def test_invalid_or_unbound_qualified_collection_selector_rejects(tmp_path, fault):
    suite, manifest, environment = _suite(
        tmp_path,
        {
            "test_cases.py": _BODY,
            "helper.py": _BODY,
            "directory/test_inside.py": _BODY,
        },
        ["test_cases.py::test_base"],
    )
    outside = tmp_path / "test_cases.py"
    outside.write_text(_BODY)
    if fault == "outside":
        (suite / "test_alias.py").symlink_to(outside)
    invalid = {
        "none": None,
        "not-list": "test_cases.py",
        "non-string": [1],
        "wrong-schema": ["test_cases.py"],
        "outside": ["test_alias.py"],
        "absolute": [str(suite / "test_cases.py")],
        "parent": ["../test_cases.py"],
        "glob": ["test_*.py"],
        "directory": ["directory"],
        "helper": ["helper.py"],
        "unbound": ["test_cases.py"],
        "missing": ["test_missing.py"],
        "option": ["--maxfail=1"],
    }[fault]
    contract = json.loads(manifest.read_text())
    contract["collection_args"] = invalid
    if fault == "wrong-schema":
        contract["schema"] = "unqualified-collection-schema"
    if fault == "outside":
        contract["nodeids"] = ["test_alias.py::test_base"]
        contract["source_files"]["test_alias.py"] = hashlib.sha256(
            outside.read_bytes()
        ).hexdigest()
    if fault == "helper":
        contract["nodeids"] = ["helper.py::test_base"]
    if fault == "directory":
        contract["nodeids"] = ["directory/test_inside.py::test_base"]
    if fault == "unbound":
        del contract["source_files"]["test_cases.py"]
    manifest.write_text(json.dumps(contract))
    actual = (
        tuple(invalid)
        if type(invalid) is list and all(type(item) is str for item in invalid)
        else ()
    )
    result, receipt = _run(
        suite, manifest, environment, tmp_path / "report", options=actual
    )
    _assert_failed_before_body(result, receipt, suite)


@pytest.mark.parametrize(
    ("selector", "concrete"),
    [
        ("test_cases.py::test_case", "test_cases.py::test_case[chosen]"),
        ("test_cases.py::TestGroup", "test_cases.py::TestGroup::test_member"),
    ],
)
def test_qualified_node_selector_must_be_one_concrete_case(
    tmp_path, selector, concrete
):
    suite, manifest, environment = _suite(
        tmp_path,
        {
            "test_cases.py": """from pathlib import Path
import pytest
@pytest.mark.parametrize('value', [1, 2], ids=['chosen', 'sibling'])
def test_case(value):
    Path('body-ran').touch()
class TestGroup:
    def test_member(self):
        Path('body-ran').touch()
"""
        },
        [concrete],
        collection_args=[selector],
    )
    result, receipt = _run(
        suite, manifest, environment, tmp_path / "report", options=(selector,)
    )
    _assert_failed_before_body(result, receipt, suite)


@pytest.mark.parametrize(
    "fault",
    [
        "receipt",
        "request",
        "source-proof",
        "lane",
        "collection-phase",
        "collection-completed",
        "collection-timing",
        "selector-missing",
        "selector-changed",
    ],
)
def test_canonical_closed_lane_receipt_binds_terminal_authority_and_raw_files(
    tmp_path, fault
):
    module = _runner()
    suite, manifest, environment = _suite(
        tmp_path,
        {"test_cases.py": _BODY},
        ["test_cases.py::test_base"],
        collection_args=["test_cases.py"] if fault.startswith("selector-") else None,
    )
    context = module._CiLaneContext(manifest, suite, environment=environment)
    folder = tmp_path / "closed-worker"
    folder.mkdir()
    request = {
        "kind": "collection",
        "pytest_args": [
            "--_j2o-ci-lane=standard",
            "--_j2o-ci-main-baseline=" + str(manifest),
            "--_j2o-ci-main-sha256=" + context.sha256,
            *context.manifest.get("collection_args", []),
            "--collect-only",
        ],
    }
    receipt = {
        "kind": "collection",
        "pytest_args": request["pytest_args"],
        "exitstatus": 0,
        "finished": "closed",
        "phases": {},
        "completed": [],
        "phase_durations": {},
        "ci_lane": {
            "lane": "standard",
            "manifest_path": str(context.path),
            "manifest_sha256": context.sha256,
            "environment": context.manifest["environment"],
            "source_files": context.sources,
            "source_files_after": context.sources,
            "errors": [],
        },
    }
    if fault.startswith("selector-"):
        receipt.update(
            collection_args=["test_cases.py"],
            qualified_collection_args=["test_cases.py"],
        )
    module._write(folder / "request.json", request)
    module._write(folder / "receipt.json", receipt)
    terminal = {
        "receipt_sha256": hashlib.sha256(
            (folder / "receipt.json").read_bytes()
        ).hexdigest(),
        "request_sha256": hashlib.sha256(
            (folder / "request.json").read_bytes()
        ).hexdigest(),
        "manifest_sha256": context.sha256,
        "exitstatus": 0,
    }
    module._write(folder / "lane-terminal.json", terminal)
    assert module._ci_lane_closed(folder, request, context) == receipt
    if fault in {"receipt", "request"}:
        path = folder / (fault + ".json")
        path.write_bytes(path.read_bytes() + b" ")
    else:
        if fault == "source-proof":
            receipt["ci_lane"]["source_files_after"] = {}
        elif fault == "collection-phase":
            receipt["phases"] = {
                "test_cases.py::test_base": [{"when": "setup", "outcome": "passed"}]
            }
        elif fault == "collection-completed":
            receipt["completed"] = ["test_cases.py::test_base"]
        elif fault == "collection-timing":
            receipt["phase_durations"] = {
                "test_cases.py::test_base": [{"when": "setup", "seconds": 0.0}]
            }
        elif fault == "selector-missing":
            del receipt["qualified_collection_args"]
        elif fault == "selector-changed":
            receipt["collection_args"] = ["test_foreign.py"]
        else:
            receipt["ci_lane"]["lane"] = "extra"
        module._write(folder / "receipt.json", receipt)
        terminal["receipt_sha256"] = hashlib.sha256(
            (folder / "receipt.json").read_bytes()
        ).hexdigest()
        module._write(folder / "lane-terminal.json", terminal)
    with pytest.raises(ValueError, match="canonical lane"):
        module._ci_lane_closed(folder, request, context)


@pytest.mark.parametrize(
    "snapshot", ["startup_context", "collection_context", "final_context"]
)
@pytest.mark.parametrize(
    "fault",
    [
        "environment",
        "jax-config",
        "missing",
        "unverified",
        "incomplete-stage",
        "missing-collection-stage",
        "missing-post-pytest-stage",
        "incomplete-post-pytest-stage",
    ],
)
def test_staged_closed_receipt_rejects_semantic_snapshot_forgery_with_valid_hashes(
    tmp_path, snapshot, fault
):
    module = _runner()
    suite, manifest, environment = _suite(
        tmp_path, {"test_cases.py": _BODY}, ["test_cases.py::test_base"]
    )
    payload = json.loads(manifest.read_text())
    payload["collection_transition"] = {
        "startup_jax_config": {"status": "NOT_LOADED", "values": {}},
        "collected_environment": {
            **payload["environment"],
            "TF_CPP_MIN_LOG_LEVEL": "1",
        },
        "collected_jax_config": {"status": "NOT_LOADED", "values": {}},
        "post_pytest_environment": dict(payload["environment"]),
        "post_pytest_jax_config": {"status": "NOT_LOADED", "values": {}},
    }
    manifest.write_text(json.dumps(payload) + "\n")
    context = module._CiLaneContext(manifest, suite, environment=environment)
    folder = tmp_path / "closed-worker"
    folder.mkdir()
    request = {
        "kind": "collection",
        "pytest_args": [
            "--_j2o-ci-lane=standard",
            "--_j2o-ci-main-baseline=" + str(manifest),
            "--_j2o-ci-main-sha256=" + context.sha256,
            "--collect-only",
        ],
    }
    receipt = {
        "kind": "collection",
        "pytest_args": request["pytest_args"],
        "exitstatus": 0,
        "finished": "closed",
        "phases": {},
        "completed": [],
        "phase_durations": {},
        "collection_transition_completed": True,
        "post_pytest_completed": True,
        "ci_lane": {
            "lane": "standard",
            "manifest_path": str(context.path),
            "manifest_sha256": context.sha256,
            "environment": context.manifest["environment"],
            "source_files": context.sources,
            "source_files_after": context.sources,
            "errors": [],
        },
    }
    for name in ("startup_context", "collection_context", "final_context"):
        receipt[name] = {
            "status": "OBSERVED",
            "errors": [],
            "environment": dict(
                payload["environment"]
                if name == "startup_context"
                else payload["collection_transition"][
                    (
                        "post_pytest_environment"
                        if name == "final_context"
                        else "collected_environment"
                    )
                ]
            ),
            "loaded_jax_config": {"status": "NOT_LOADED", "values": {}},
        }
    module._write(folder / "request.json", request)

    def close():
        module._write(folder / "receipt.json", receipt)
        module._write(
            folder / "lane-terminal.json",
            {
                "receipt_sha256": hashlib.sha256(
                    (folder / "receipt.json").read_bytes()
                ).hexdigest(),
                "request_sha256": hashlib.sha256(
                    (folder / "request.json").read_bytes()
                ).hexdigest(),
                "manifest_sha256": context.sha256,
                "exitstatus": 0,
            },
        )

    close()
    assert module._ci_lane_closed(folder, request, context) == receipt
    if fault == "environment":
        receipt[snapshot]["environment"]["JAX_ENABLE_X64"] = "forged"
    elif fault == "jax-config":
        receipt[snapshot]["loaded_jax_config"] = {
            "status": "OBSERVED",
            "values": {
                "x64_enabled": True,
                "jax_disable_jit": False,
                "jax_default_matmul_precision": None,
                "jax_platforms": None,
                "jax_numpy_dtype_promotion": "standard",
                "jax_numpy_rank_promotion": "allow",
            },
        }
    elif fault == "unverified":
        receipt[snapshot]["loaded_jax_config"]["status"] = "UNVERIFIED"
    elif fault == "incomplete-stage":
        receipt["collection_transition_completed"] = False
    elif fault == "missing-collection-stage":
        del receipt["collection_transition_completed"]
    elif fault == "missing-post-pytest-stage":
        del receipt["post_pytest_completed"]
    elif fault == "incomplete-post-pytest-stage":
        receipt["post_pytest_completed"] = False
    else:
        del receipt[snapshot]
    close()
    with pytest.raises(ValueError, match="collection-transition"):
        module._ci_lane_closed(folder, request, context)


@pytest.mark.parametrize(
    "options",
    [
        (),
        ("--collect-only",),
        pytest.param(
            ("-p", "xdist.plugin", "-n", "2"),
            marks=pytest.mark.skipif(
                os.environ.get("JAX2ONNX_TEST_CI_LANE_XDIST") != "1",
                reason="assigned to the opt-in synthetic xdist qualification lane",
            ),
        ),
    ],
    ids=["serial", "collection", "xdist"],
)
def test_direct_staged_context_rejects_before_owner_import(tmp_path, options):
    suite, manifest, environment = _suite(
        tmp_path,
        {
            "test_cases.py": "from pathlib import Path\nPath('owner-imported').touch()\n"
            + _BODY
        },
        ["test_cases.py::test_base"],
    )
    payload = json.loads(manifest.read_text())
    payload["collection_transition"] = {
        "startup_jax_config": {"status": "NOT_LOADED", "values": {}},
        "collected_environment": payload["environment"],
        "collected_jax_config": {"status": "NOT_LOADED", "values": {}},
        "post_pytest_environment": dict(payload["environment"]),
        "post_pytest_jax_config": {"status": "NOT_LOADED", "values": {}},
    }
    manifest.write_text(json.dumps(payload) + "\n")
    result, receipt = _run(
        suite, manifest, environment, tmp_path / "report", options=options
    )
    _assert_failed_before_body(result, receipt, suite)
    assert not (suite / "owner-imported").exists()
    assert "canonical pre-pytest startup evidence" in result.stdout + result.stderr


@pytest.mark.parametrize("fault", ["duplicate", "missing", "invalid-sha"])
def test_private_canonical_lane_authority_rejects_ambiguous_or_unbound_options(fault):
    module = _runner()
    options = [
        "--_j2o-ci-lane=standard",
        "--_j2o-ci-main-baseline=/tmp/frozen-manifest.json",
        "--_j2o-ci-main-sha256=" + "a" * 64,
    ]
    assert module._ci_lane_worker_authority(options)["--_j2o-ci-lane"] == "standard"
    if fault == "duplicate":
        options.append(options[0])
    elif fault == "missing":
        options.pop()
    else:
        options[-1] = "--_j2o-ci-main-sha256=not-a-sha"
    with pytest.raises(ValueError, match="CI|SHA"):
        module._ci_lane_worker_authority(options)


def test_requested_direct_collection_rejects_execution_mode_override(tmp_path):
    suite, manifest, environment = _suite(
        tmp_path,
        {
            "test_cases.py": _BODY + "def test_extra():\n    pass\n",
            "conftest.py": """from pathlib import Path
import pytest

@pytest.hookimpl(tryfirst=True)
def pytest_runtestloop(session):
    Path('override-attempted').touch()
    session.config.option.collectonly = False
""",
        },
        ["test_cases.py::test_base"],
    )
    result, receipt = _run(
        suite, manifest, environment, tmp_path / "report", options=("--collect-only",)
    )
    assert (suite / "override-attempted").exists()
    assert result.returncode != 0, result.stdout + result.stderr
    assert receipt["status"] == "FAIL"
    assert receipt["phases"] == receipt["phase_durations"] == {}
    assert receipt["completed"] == []
    assert not (suite / "body-ran").exists()


def _campaign_fixture(tmp_path, *, support=False):
    module = _runner()
    standard = ["test_cases.py::test_base", "test_cases.py::test_other"]
    extra = ["test_cases.py::test_extra"]
    suite, manifest, environment = _suite(
        tmp_path,
        {
            "test_cases.py": "def test_base(): pass\n"
            "def test_other(): pass\ndef test_extra(): pass\n"
        },
        standard,
        collection_args=["test_cases.py"] if support else None,
    )
    manifest_payload = json.loads(manifest.read_text())
    environment_map = manifest_payload["environment"]
    jax_config = {"status": "NOT_LOADED", "values": {}}
    manifest_payload["collection_transition"] = {
        "startup_jax_config": jax_config,
        "collected_environment": environment_map,
        "collected_jax_config": jax_config,
        "post_pytest_environment": environment_map,
        "post_pytest_jax_config": jax_config,
    }
    manifest.write_text(json.dumps(manifest_payload) + "\n")
    full = standard + extra
    items = [
        {"nodeid": nodeid, "module": "test_cases.py", "class": None} for nodeid in full
    ]
    inventory = {
        "kind": "collection",
        "pytest_args": ["--collect-only"],
        "items": items,
        "full_items": copy.deepcopy(items),
        "full_preselection_ids": full,
        "selected_ids": full,
        "assigned_ids": [],
        "full_collection_checked": True,
        "collection_errors": [],
        "collection_skips": [],
        "exitstatus": 0,
        "finished": "closed",
        "phases": {},
        "phase_durations": {},
        "completed": [],
        "ci_lane": {"standard_ids": standard, "extra_ids": extra},
        **(
            {
                "collection_args": ["test_cases.py"],
                "qualified_collection_args": ["test_cases.py"],
            }
            if support
            else {}
        ),
    }
    for stage in ("startup_context", "collection_context", "final_context"):
        inventory[stage] = {
            "status": "OBSERVED",
            "errors": [],
            "environment": copy.deepcopy(environment_map),
            "loaded_jax_config": copy.deepcopy(jax_config),
        }
    inventory_path = tmp_path / "global-collection.json"
    inventory_path.write_text(json.dumps(inventory) + "\n")
    contract = {
        "schema": (
            "jax2onnx-ci-support-campaign-v1" if support else "jax2onnx-ci-campaign-v1"
        ),
        "python_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "main_manifest": {
            "path": manifest.name,
            "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        },
        "command_inventory" if support else "global_inventory": {
            "path": inventory_path.name,
            "sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        },
        "allowed_skips": [],
    }
    contract_path = tmp_path / "campaign.json"
    contract_path.write_text(json.dumps(contract) + "\n")
    context = module._CiLaneContext(manifest, suite, environment=environment)
    return module, context, contract_path, inventory_path, inventory


def _campaign_rewrite_inventory(contract_path, inventory_path, inventory):
    inventory_path.write_text(json.dumps(inventory) + "\n")
    contract = json.loads(contract_path.read_text())
    inventory_name = (
        "command_inventory" if "command_inventory" in contract else "global_inventory"
    )
    contract[inventory_name]["sha256"] = hashlib.sha256(
        inventory_path.read_bytes()
    ).hexdigest()
    contract_path.write_text(json.dumps(contract) + "\n")


def _campaign_batch(tmp_path, inventory):
    assigned = inventory["ci_lane"]["standard_ids"]
    receipt = copy.deepcopy(inventory)
    receipt.update(
        kind="batch",
        items=inventory["items"][: len(assigned)],
        assigned_ids=assigned,
        selected_ids=assigned,
        completed=assigned,
        phases={
            nodeid: [
                {"when": phase, "outcome": "passed"}
                for phase in ("setup", "call", "teardown")
            ]
            for nodeid in assigned
        },
    )
    junit_path = tmp_path / "batch-junit.xml"
    _campaign_junit(junit_path, receipt)
    return receipt, assigned, junit_path


def _campaign_junit(path, receipt):
    from xml.etree import ElementTree as ET

    suite = ET.Element("testsuite")
    for nodeid in receipt["assigned_ids"]:
        owner, _, name = nodeid.partition("::")
        case = ET.SubElement(
            suite,
            "testcase",
            classname=owner.removesuffix(".py").replace("/", "."),
            name=name,
            file=owner,
        )
        if any(report["outcome"] == "skipped" for report in receipt["phases"][nodeid]):
            ET.SubElement(case, "skipped", type="pytest.skip", message="original skip")
    ET.ElementTree(suite).write(path)


@pytest.mark.parametrize("support", [False, True], ids=["global", "support"])
def test_campaign_contract_binds_relative_inputs_and_exact_projection(
    tmp_path, support
):
    module, lane, contract_path, _, inventory = _campaign_fixture(
        tmp_path, support=support
    )
    campaign = module._CiCampaignContext(contract_path, lane)
    standard, extra = campaign.collection(inventory)
    assert standard == lane.manifest["nodeids"]
    assert extra == ["test_cases.py::test_extra"]
    assert set(standard).isdisjoint(extra)
    batch, assigned, junit_path = _campaign_batch(tmp_path, inventory)
    campaign.batch(batch, assigned, junit_path)
    campaign.recheck()


@pytest.mark.parametrize(
    "fault",
    [
        "unknown-root",
        "missing-root",
        "duplicate-root",
        "duplicate-reference",
        "unknown-reference",
        "bad-sha",
        "wrong-inventory-sha",
        "wrong-cohort",
        "crossed-main",
        "missing-inventory",
        "prefiltered-main",
        "unbound-skip-source",
    ],
)
def test_campaign_contract_rejects_unreviewed_admission(tmp_path, fault):
    module, lane, path, inventory_path, _ = _campaign_fixture(tmp_path)
    payload = json.loads(path.read_text())
    if fault == "unknown-root":
        payload["accepted"] = True
    elif fault == "missing-root":
        del payload["global_inventory"]
    elif fault == "duplicate-root":
        path.write_text(
            path.read_text().replace('"schema":', '"schema": "duplicate", "schema":', 1)
        )
    elif fault == "duplicate-reference":
        path.write_text(
            path.read_text().replace('"path":', '"path": "duplicate", "path":', 1)
        )
    elif fault == "unknown-reference":
        payload["main_manifest"]["accepted"] = True
    elif fault == "bad-sha":
        payload["global_inventory"]["sha256"] = "g" * 64
    elif fault == "wrong-inventory-sha":
        payload["global_inventory"]["sha256"] = "0" * 64
    elif fault == "wrong-cohort":
        payload["python_minor"] = "3.12" if sys.version_info.minor != 12 else "3.11"
    elif fault == "crossed-main":
        other = path.with_name("other-baseline.json")
        other.write_bytes(lane.path.read_bytes())
        payload["main_manifest"]["path"] = other.name
    elif fault == "missing-inventory":
        inventory_path.unlink()
    elif fault == "prefiltered-main":
        manifest = json.loads(lane.path.read_text())
        manifest["collection_args"] = ["test_cases.py"]
        lane.path.write_text(json.dumps(manifest) + "\n")
        payload["main_manifest"]["sha256"] = hashlib.sha256(
            lane.path.read_bytes()
        ).hexdigest()
        lane = module._CiLaneContext(
            lane.path, lane.root, environment=manifest["environment"]
        )
    else:
        payload["allowed_skips"] = [
            {
                "nodeid": lane.manifest["nodeids"][0],
                "phases": ["setup:skipped", "teardown:passed"],
                "source": "unreviewed.py",
                "line": 1,
                "reason": "original skip",
            }
        ]
    if not fault.startswith("duplicate-"):
        path.write_text(json.dumps(payload) + "\n")
    with pytest.raises((OSError, ValueError)):
        module._CiCampaignContext(path, lane)


@pytest.mark.parametrize(
    "fault",
    [
        "missing-C",
        "missing-baseline-C",
        "reordered-C",
        "items",
        "executed",
        "incomplete",
        "missing-S",
        "extra-S",
        "extra-A",
    ],
)
def test_campaign_contract_rejects_incomplete_or_crossed_global_inventory(
    tmp_path, fault
):
    module, lane, path, inventory_path, inventory = _campaign_fixture(tmp_path)
    if fault == "missing-C":
        inventory["full_preselection_ids"].pop()
    elif fault == "missing-baseline-C":
        for key in ("full_preselection_ids", "selected_ids", "items", "full_items"):
            inventory[key] = inventory[key][1:]
    elif fault == "reordered-C":
        inventory["full_preselection_ids"] = list(
            reversed(inventory["full_preselection_ids"])
        )
    elif fault == "items":
        inventory["full_items"][0]["module"] = "other.py"
    elif fault == "executed":
        inventory["phases"] = {
            lane.manifest["nodeids"][0]: [{"when": "setup", "outcome": "passed"}]
        }
    elif fault == "incomplete":
        inventory["full_collection_checked"] = False
    elif fault == "missing-S":
        inventory["ci_lane"]["standard_ids"] = lane.manifest["nodeids"][:-1]
    elif fault == "extra-S":
        inventory["ci_lane"]["standard_ids"] = inventory["full_preselection_ids"]
    else:
        inventory["ci_lane"]["extra_ids"] = [lane.manifest["nodeids"][0]]
    _campaign_rewrite_inventory(path, inventory_path, inventory)
    with pytest.raises(ValueError):
        module._CiCampaignContext(path, lane)


@pytest.mark.parametrize("fault", ["order", "items", "collection-skip"])
def test_campaign_collection_matches_every_fresh_global_observation(tmp_path, fault):
    module, lane, path, _, inventory = _campaign_fixture(tmp_path)
    campaign = module._CiCampaignContext(path, lane)
    assert campaign.collection(inventory)[0] == lane.manifest["nodeids"]
    fresh = copy.deepcopy(inventory)
    if fault == "order":
        for key in ("items", "full_items", "full_preselection_ids", "selected_ids"):
            fresh[key] = list(reversed(fresh[key]))
    elif fault == "items":
        fresh["items"][0]["class"] = "test_cases.py::DifferentScope"
        fresh["full_items"] = copy.deepcopy(fresh["items"])
    else:
        fresh["collection_skips"] = [
            {"nodeid": "unselected.py", "longrepr": "unreviewed"}
        ]
    with pytest.raises(ValueError):
        campaign.collection(fresh)


@pytest.mark.parametrize(
    "stage", ["startup_context", "collection_context", "final_context"]
)
def test_campaign_keeps_every_independently_bound_context_stage(tmp_path, stage):
    module, lane, path, _, inventory = _campaign_fixture(tmp_path)
    campaign = module._CiCampaignContext(path, lane)
    campaign.collection(inventory)
    fresh = copy.deepcopy(inventory)
    fresh[stage]["loaded_jax_config"]["status"] = "UNVERIFIED"
    with pytest.raises(ValueError, match="collection-transition"):
        campaign.collection(fresh)


@pytest.mark.parametrize("phase", ["setup", "call"])
def test_campaign_allows_only_the_exact_complete_original_skip(tmp_path, phase):
    module, lane, path, _, inventory = _campaign_fixture(tmp_path)
    payload = json.loads(path.read_text())
    names = ["setup", "teardown"] if phase == "setup" else ["setup", "call", "teardown"]
    expected = [
        {"when": name, "outcome": "skipped" if name == phase else "passed"}
        for name in names
    ]
    payload["allowed_skips"] = [
        {
            "nodeid": lane.manifest["nodeids"][0],
            "phases": [report["when"] + ":" + report["outcome"] for report in expected],
            "source": "test_cases.py",
            "line": 1,
            "reason": "original skip",
        }
    ]
    path.write_text(json.dumps(payload) + "\n")
    campaign = module._CiCampaignContext(path, lane)
    batch, assigned, junit_path = _campaign_batch(tmp_path, inventory)
    # An allowance does not require a skip when the original case now passes.
    campaign.batch(batch, assigned, junit_path)
    expected[0 if phase == "setup" else 1]["longrepr"] = repr(
        (str(lane.root / "test_cases.py"), 1, "Skipped: original skip")
    )
    batch["phases"][assigned[0]] = expected
    _campaign_junit(junit_path, batch)
    campaign.batch(batch, assigned, junit_path)
    expected[0 if phase == "setup" else 1]["longrepr"] = repr(
        ("test_cases.py", 1, "Skipped: original skip")
    )
    campaign.batch(batch, assigned, junit_path)
    expected[0 if phase == "setup" else 1]["longrepr"] = repr(
        (str(lane.root / "test_cases.py"), 2, "Skipped: original skip")
    )
    with pytest.raises(ValueError):
        campaign.batch(batch, assigned, junit_path)


@pytest.mark.parametrize("fault", ["phase", "reason", "source", "wasxfail", "junit"])
def test_campaign_skip_allowance_does_not_authorize_other_outcomes(tmp_path, fault):
    module, lane, path, _, inventory = _campaign_fixture(tmp_path)
    payload = json.loads(path.read_text())
    payload["allowed_skips"] = [
        {
            "nodeid": lane.manifest["nodeids"][0],
            "phases": ["setup:passed", "call:skipped", "teardown:passed"],
            "source": "test_cases.py",
            "line": 1,
            "reason": "original skip",
        }
    ]
    path.write_text(json.dumps(payload) + "\n")
    campaign = module._CiCampaignContext(path, lane)
    batch, assigned, junit_path = _campaign_batch(tmp_path, inventory)
    reports = batch["phases"][assigned[0]]
    reports[1].update(
        outcome="skipped",
        longrepr=repr((str(lane.root / "test_cases.py"), 1, "Skipped: original skip")),
    )
    _campaign_junit(junit_path, batch)
    campaign.batch(batch, assigned, junit_path)
    if fault == "phase":
        reports.pop(1)
        reports[0].update(
            outcome="skipped",
            longrepr=repr(
                (str(lane.root / "test_cases.py"), 1, "Skipped: original skip")
            ),
        )
    elif fault == "reason":
        reports[1]["longrepr"] = repr(
            (str(lane.root / "test_cases.py"), 1, "Skipped: different reason")
        )
    elif fault == "source":
        reports[1]["longrepr"] = repr(
            ("/unrelated/test_cases.py", 1, "Skipped: original skip")
        )
    elif fault == "wasxfail":
        reports[1]["wasxfail"] = ""
    else:
        junit_path.write_text(
            junit_path.read_text().replace(
                'message="original skip"', 'message="different reason"'
            )
        )
    with pytest.raises(ValueError):
        campaign.batch(batch, assigned, junit_path)


@pytest.mark.parametrize(
    "fault",
    [
        "skip",
        "wasxfail-empty",
        "wasxfail-passed",
        "missing-phase",
        "duplicate-phase",
        "phase-order",
        "completion-order",
        "foreign-source",
        "junit-missing",
        "junit-identity",
        "junit-failure",
    ],
)
@pytest.mark.parametrize("support", [False, True], ids=["global", "support"])
def test_campaign_batch_rejects_unqualified_native_or_junit_outcomes(
    tmp_path, fault, support
):
    module, lane, path, _, inventory = _campaign_fixture(tmp_path, support=support)
    campaign = module._CiCampaignContext(path, lane)
    batch, assigned, junit_path = _campaign_batch(tmp_path, inventory)
    campaign.batch(batch, assigned, junit_path)
    reports = batch["phases"][assigned[0]]
    if fault == "skip":
        reports[1].update(
            outcome="skipped",
            longrepr=repr(
                (str(lane.root / "test_cases.py"), 1, "Skipped: original skip")
            ),
        )
    elif fault.startswith("wasxfail"):
        reports[1]["wasxfail"] = "" if fault.endswith("empty") else "original xfail"
    elif fault == "missing-phase":
        reports.pop()
    elif fault == "duplicate-phase":
        reports.append(reports[-1].copy())
    elif fault == "phase-order":
        reports.reverse()
    elif fault == "completion-order":
        batch["completed"] = list(reversed(assigned))
    elif fault == "foreign-source":
        reports[1].update(
            outcome="skipped",
            longrepr=repr(("/foreign/test_cases.py", 1, "Skipped: original skip")),
        )
    elif fault == "junit-missing":
        junit_path.write_text("<testsuite/>")
    elif fault == "junit-identity":
        junit_path.write_text(
            junit_path.read_text().replace('name="test_base"', 'name="different_case"')
        )
    else:
        junit_path.write_text(
            '<testsuite><testcase classname="test_cases" name="test_base"><failure/></testcase><testcase classname="test_cases" name="test_other"/></testsuite>'
        )
    with pytest.raises(ValueError):
        campaign.batch(batch, assigned, junit_path)


@pytest.mark.parametrize("target", ["contract", "inventory", "main", "source"])
def test_campaign_rechecks_reviewed_bytes_before_final_success(tmp_path, target):
    module, lane, path, inventory_path, inventory = _campaign_fixture(tmp_path)
    campaign = module._CiCampaignContext(path, lane)
    campaign.collection(inventory)
    campaign.recheck()
    changed = {
        "contract": path,
        "inventory": inventory_path,
        "main": lane.path,
        "source": lane.root / "test_cases.py",
    }[target]
    changed.write_bytes(changed.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        campaign.recheck()


def test_campaign_requires_transition_but_generic_static_manifest_stays_valid(tmp_path):
    module, lane, path, _, _ = _campaign_fixture(tmp_path)
    manifest = json.loads(lane.path.read_text())
    del manifest["collection_transition"]
    lane.path.write_text(json.dumps(manifest) + "\n")
    contract = json.loads(path.read_text())
    contract["main_manifest"]["sha256"] = hashlib.sha256(
        lane.path.read_bytes()
    ).hexdigest()
    path.write_text(json.dumps(contract) + "\n")
    static = module._CiLaneContext(
        lane.path, lane.root, environment=manifest["environment"]
    )
    assert static.transition is None
    static.recheck(environment=manifest["environment"])
    with pytest.raises(
        ValueError, match="requires independently observed collection_transition"
    ):
        module._CiCampaignContext(path, static)


@pytest.mark.parametrize(
    "stage", ["startup_context", "collection_context", "final_context"]
)
def test_campaign_rejects_missing_independently_observed_snapshot(tmp_path, stage):
    module, lane, path, inventory_path, inventory = _campaign_fixture(tmp_path)
    del inventory[stage]
    _campaign_rewrite_inventory(path, inventory_path, inventory)
    with pytest.raises(ValueError, match="context snapshot is unverified"):
        module._CiCampaignContext(path, lane)


def test_campaign_rejects_unqualified_pytest_xml_serializer(tmp_path, monkeypatch):
    module, lane, path, _, _ = _campaign_fixture(tmp_path)
    original = metadata.version
    monkeypatch.setattr(
        metadata,
        "version",
        lambda name: "9.1.2" if name == "pytest" else original(name),
    )
    with pytest.raises(ValueError, match="qualified pytest 9.1.1"):
        module._CiCampaignContext(path, lane)


@pytest.mark.parametrize("field", ["collection_args", "qualified_collection_args"])
@pytest.mark.parametrize("fault", ["missing", "global", "foreign", "injected"])
@pytest.mark.parametrize("stage", ["initial", "collection", "batch"])
def test_support_campaign_binds_both_receipt_selectors_at_every_stage(
    tmp_path, field, fault, stage
):
    module, lane, path, inventory_path, inventory = _campaign_fixture(
        tmp_path, support=True
    )
    campaign = module._CiCampaignContext(path, lane)
    if stage == "batch":
        receipt, assigned, junit_path = _campaign_batch(tmp_path, inventory)
    else:
        receipt = copy.deepcopy(inventory)
    if fault == "missing":
        del receipt[field]
    else:
        receipt[field] = {
            "global": [],
            "foreign": ["test_foreign.py"],
            "injected": ["test_cases.py", "test_foreign.py"],
        }[fault]
    with pytest.raises(ValueError, match="selectors"):
        if stage == "initial":
            _campaign_rewrite_inventory(path, inventory_path, receipt)
            module._CiCampaignContext(path, lane)
        elif stage == "collection":
            campaign.collection(receipt)
        else:
            campaign.batch(receipt, assigned, junit_path)


@pytest.mark.parametrize("fault", ["global-key", "global-schema", "empty-authority"])
def test_support_campaign_rejects_global_inventory_substitution(tmp_path, fault):
    module, lane, path, _, _ = _campaign_fixture(tmp_path, support=True)
    contract = json.loads(path.read_text())
    if fault == "global-key":
        contract["global_inventory"] = contract.pop("command_inventory")
    elif fault == "global-schema":
        contract["schema"] = "jax2onnx-ci-campaign-v1"
    else:
        manifest = json.loads(lane.path.read_text())
        manifest["collection_args"] = []
        lane.path.write_text(json.dumps(manifest) + "\n")
        contract["main_manifest"]["sha256"] = hashlib.sha256(
            lane.path.read_bytes()
        ).hexdigest()
        lane = module._CiLaneContext(
            lane.path, lane.root, environment=manifest["environment"]
        )
    path.write_text(json.dumps(contract) + "\n")
    with pytest.raises(ValueError):
        module._CiCampaignContext(path, lane)


@pytest.mark.parametrize(
    "fault", ["empty", "reordered", "injected", "keyword", "marker", "positional"]
)
def test_support_campaign_accepts_only_the_original_ordered_command(monkeypatch, fault):
    module = _runner()
    monkeypatch.setenv("PYTEST_ADDOPTS", "")
    selectors = ["test_cases.py", "test_exact.py::TestOwner::test_exact"]
    module._ci_campaign_arguments(selectors, ["-q", "--durations=50"], selectors)
    actual = {
        "empty": [],
        "reordered": list(reversed(selectors)),
        "injected": [*selectors, "test_other.py"],
    }.get(fault, selectors)
    options = {
        "keyword": ["-k", "base"],
        "marker": ["-m", "slow"],
        "positional": ["test_cases.py"],
    }.get(fault, [])
    with pytest.raises(ValueError):
        module._ci_campaign_arguments(actual, options, selectors)
