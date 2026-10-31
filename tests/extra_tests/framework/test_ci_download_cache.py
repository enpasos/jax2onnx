# tests/extra_tests/framework/test_ci_download_cache.py

"""Hosted download-cache safety and the one-shot diagnostic exit policy."""

from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]


def _helper():
    spec = spec_from_file_location(
        "_ci_download_cache", ROOT / "scripts/prepare_ci_download_cache.py"
    )
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _layout(tmp_path):
    home = tmp_path / "home"
    pip = home / ".cache/pip"
    poetry = home / ".cache/pypoetry"
    names = [
        pip / "http-v2/body",
        pip / "wheels/package.whl",
        poetry / "artifacts/package.whl",
        poetry / "cache/repositories/pypi/entry",
    ]
    protected = [
        poetry / "virtualenvs/env",
        tmp_path / "workspace",
        tmp_path / "receipts",
        tmp_path / "owned-tmp",
    ]
    for path in names + [path / "preserve" for path in protected]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"unchanged runtime or task data")
    return home, pip, poetry, names, tuple(protected)


def test_release_only_downloads_preserves_actual_runtime_and_task_data(tmp_path):
    module = _helper()
    home, pip, poetry, downloads, protected = _layout(tmp_path)
    runtime = module._runtime_identity()
    report = tmp_path / "cache-report.json"
    result = module.release_download_caches(
        home=home,
        pip_cache=pip,
        poetry_cache=poetry,
        protected=protected,
        disk_paths=(tmp_path,),
        report=report,
    )
    assert (
        result["status"] == "PASS"
        and result["runtime_before"] == runtime == result["runtime_after"]
    )
    assert all(not path.exists() for path in downloads)
    assert all(
        (path / "preserve").read_bytes() == b"unchanged runtime or task data"
        for path in protected
    )
    assert sum(item["files"] for item in result["caches"].values()) == 4
    assert json.loads(report.read_text())["runtime_identity_held"] is True
    assert (
        str(tmp_path) in result["disk_before"] and str(tmp_path) in result["disk_after"]
    )


@pytest.mark.parametrize(
    "fault",
    ["outside", "root-symlink", "parent-symlink", "overlap", "home-cache-symlink"],
)
def test_cache_scope_rejection_precedes_every_deletion(tmp_path, fault):
    module = _helper()
    home, pip, poetry, downloads, protected = _layout(tmp_path)
    if fault == "outside":
        pip = tmp_path / "outside"
        pip.mkdir()
    elif fault == "root-symlink":
        real = pip.with_name("pip-real")
        pip.rename(real)
        pip.symlink_to(real, target_is_directory=True)
    elif fault == "parent-symlink":
        real = tmp_path / "poetry-outside"
        poetry.rename(real)
        poetry.symlink_to(real, target_is_directory=True)
    elif fault == "overlap":
        pip = poetry / "virtualenvs"
    else:
        real = tmp_path / "user-cache-outside"
        (home / ".cache").rename(real)
        (home / ".cache").symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError):
        module.release_download_caches(
            home=home,
            pip_cache=pip,
            poetry_cache=poetry,
            protected=protected,
            disk_paths=(tmp_path,),
            report=tmp_path / "report.json",
        )
    assert all(
        path.read_bytes() == b"unchanged runtime or task data" for path in downloads
    )
    assert all(
        (path / "preserve").read_bytes() == b"unchanged runtime or task data"
        for path in protected
    )
    assert not (tmp_path / "report.json").exists()


def test_runtime_identity_change_is_reported_as_failure(tmp_path, monkeypatch):
    module = _helper()
    home, pip, poetry, _, protected = _layout(tmp_path)
    values = iter([{"runtime": "before"}, {"runtime": "changed"}])
    monkeypatch.setattr(module, "_runtime_identity", lambda: next(values))
    report = tmp_path / "report.json"
    with pytest.raises(ValueError, match="runtime identity changed"):
        module.release_download_caches(
            home=home,
            pip_cache=pip,
            poetry_cache=poetry,
            protected=protected,
            disk_paths=(tmp_path,),
            report=report,
        )
    assert json.loads(report.read_text())["status"] == "FAIL"
    assert all((path / "preserve").exists() for path in protected)


def test_local_entry_never_resolves_or_deletes_user_caches(tmp_path, monkeypatch):
    module = _helper()
    monkeypatch.setenv("GITHUB_ACTIONS", "false")
    monkeypatch.setattr(
        sys, "argv", ["prepare", "--report", str(tmp_path / "report.json")]
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("local invocation accessed package-cache APIs")

    monkeypatch.setattr(module.subprocess, "check_output", forbidden)
    assert module.main() == 0 and not (tmp_path / "report.json").exists()


def _workflow():
    return yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())


def test_cache_report_preserves_existing_receipts_archive_root():
    steps = _workflow()["jobs"]["test"]["steps"]
    receipts = next(
        step
        for step in steps
        if step.get("name") == "Upload full-suite pytest receipts"
    )
    cache = next(
        step
        for step in steps
        if step.get("name") == "Upload hosted download-cache preparation report"
    )
    assert (
        receipts["with"]["name"]
        == "pytest-receipts-python-${{ matrix.python-version }}"
    )
    assert receipts["with"]["path"] == "${{ runner.temp }}/jax2onnx-pytest-batches"
    assert (
        cache["with"]["path"] == "${{ runner.temp }}/ci-download-cache-preparation.json"
    )
    assert cache["with"]["name"] != receipts["with"]["name"]
    assert cache["uses"] == receipts["uses"]


@pytest.mark.parametrize(
    "candidate",
    [
        "locked311",
        "locked312",
        "locked313",
        "locked314",
        "minimum312",
        "normalization312",
        "normalization313",
        "node312",
    ],
)
def test_execution_limits_change_only_for_locked312(candidate):
    job = _workflow()["jobs"]["candidate-standard"]
    work = next(step["run"] for step in job["steps"] if step.get("id") == "work")
    prelude = work.split("python -B scripts/run_with_resources.py", 1)[0]
    result = subprocess.run(
        [
            "bash",
            "-c",
            'python() { printf "capacity probe stub\\n"; }\n'
            + prelude
            + '\nprintf "%s %s %s\\n" "$execution_rss_mib" "$live_available_mib" "${diagnostic_args[*]}"',
        ],
        env={**os.environ, "CANDIDATE_ID": candidate},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1].split() == (
        ["5120", "8192", "--diagnostic-only"]
        if candidate == "locked312"
        else ["4096", "2048"]
    )
    assert result.stdout.count("capacity probe stub") == (candidate == "locked312")
    assert (
        '"${diagnostic_args[@]}"'
        in work.split("python -B scripts/run_with_resources.py", 1)[1].split(
            "--output-dir", 1
        )[0]
    )


@pytest.mark.parametrize("available_kib", [12288 * 1024 - 1, 12288 * 1024])
def test_initial_capacity_probe_enforces_floor_with_python_optimization(available_kib):
    job = _workflow()["jobs"]["candidate-standard"]
    work = next(step["run"] for step in job["steps"] if step.get("id") == "work")
    line = next(
        line.strip()
        for line in work.splitlines()
        if line.strip().startswith("python -B -c ")
    )
    arguments = shlex.split(line)
    probe = arguments[arguments.index("-c") + 1]
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-O",
            "-c",
            "from pathlib import Path\n"
            + f"Path.read_text = lambda self: 'MemAvailable: {available_kib} kB\\n'\n"
            + probe,
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == (0 if available_kib == 12288 * 1024 else 1)
    assert (
        f"diagnostic initial available bytes: {available_kib * 1024}" in result.stdout
    )
    assert ("one-shot diagnostic needs 12 GiB" in result.stderr) == bool(
        result.returncode
    )


@pytest.mark.parametrize("work_exit", [0, 7])
def test_locked312_diagnostic_never_becomes_standard_success(tmp_path, work_exit):
    workflow = _workflow()
    job = workflow["jobs"]["candidate-standard"]
    step = next(step for step in job["steps"] if step.get("id") == "work")
    script = step["run"]
    assert (
        subprocess.run(
            ["bash", "-n"], input=script, text=True, capture_output=True
        ).returncode
        == 0
    )
    finalizer = script.rsplit("\nBASH\n", 1)[1]
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    resources = tmp_path / "evidence-work-resources"
    resources.mkdir()
    summary = resources / "summary.json"
    original = (
        b'{"status":"COMPLETE","qualification":"PASS","limits":{"rss_mib":5120}}\n'
    )
    summary.write_bytes(original)
    result = subprocess.run(
        ["bash", "-c", f"set +e\n(exit {work_exit})\n" + finalizer],
        env={**os.environ, "CANDIDATE_ID": "locked312", "CANDIDATE_DIR": str(evidence)},
        text=True,
        capture_output=True,
    )
    assert result.returncode == (work_exit or 125)
    policy = json.loads((evidence / "diagnostic-execution-policy.json").read_text())
    assert policy["actual_work_exit"] == work_exit
    assert (
        policy["diagnostic_only"] is True
        and policy["standard_acceptance"] == "UNQUALIFIED"
    )
    assert (
        policy["execution_rss_mib"] == 5120
        and policy["standard_acceptance_rss_mib"] == 4096
    )
    assert summary.read_bytes() == original


def test_diagnostic_is_scoped_and_full_original_workload_stays_intact():
    job = _workflow()["jobs"]["candidate-standard"]
    matrix = job["strategy"]["matrix"]["include"]
    assert len(matrix) == 8
    assert [
        entry["id"] for entry in matrix if "DIAGNOSTIC" in entry.get("label", "")
    ] == ["locked312"]
    work = next(step["run"] for step in job["steps"] if step.get("id") == "work")
    assert "execution_rss_mib=4096" in work and "execution_rss_mib=5120" in work
    assert "live_available_mib=2048" in work and "live_available_mib=8192" in work
    assert (
        '--min-available-mib "$live_available_mib"' in work
        and "--timeout-seconds 1100" in work
    )
    assert "required = 12288 * 1024 * 1024" in work and job["timeout-minutes"] == 20
    assert '--_ci-provision-campaign "$CANDIDATE_COHORT" "${selection_args[@]}"' in work
    checks = (ROOT / "scripts/run_all_checks.sh").read_text()
    assert (
        checks.index("poetry run python scripts/generate_tests.py")
        < checks.index("scripts/prepare_ci_download_cache.py")
        < checks.index("python scripts/run_pytest_batches.py")
    )
    assert (
        '"${GITHUB_ACTIONS:-}" == true' in checks
        and '"${RUNNER_ENVIRONMENT:-}" == github-hosted' in checks
    )
