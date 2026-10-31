# tests/extra_tests/framework/test_ci_download_cache.py

"""Hosted cache safety and fixed serial STANDARD part transport."""

from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile

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
    "candidate,part",
    [
        ("locked311", ""),
        ("locked312-a", "a"),
        ("locked312-b", "b"),
        ("locked313", ""),
        ("locked314", ""),
        ("minimum312", ""),
        ("normalization312", ""),
        ("normalization313", ""),
        ("node312", ""),
        ("locked312", ""),
    ],
)
def test_normal_five_gib_limits_apply_only_to_fixed_locked312_parts(candidate, part):
    job = _workflow()["jobs"]["candidate-standard"]
    work = next(step["run"] for step in job["steps"] if step.get("id") == "work")
    prelude = work.split("python -B scripts/run_with_resources.py", 1)[0]
    profile = "internal_docs/precision/ci_standard/py312-owners.json"
    result = subprocess.run(
        [
            "bash",
            "-c",
            'python() { printf "capacity probe stub\\n"; }\n'
            + prelude
            + '\nprintf "%s\\n" "$execution_rss_mib" "$live_available_mib" '
            '"${execution_profile_args[@]}"',
        ],
        env={
            **os.environ,
            "CANDIDATE_ID": candidate,
            "CANDIDATE_PART": part,
            "CANDIDATE_COHORT": profile,
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    observed = [
        line for line in result.stdout.splitlines() if line != "capacity probe stub"
    ]
    assert observed == (
        ["5120", "8192", "--locked312-standard-profile", profile]
        if part
        else ["4096", "2048"]
    )
    assert result.stdout.count("capacity probe stub") == bool(part)
    assert (
        '"${execution_profile_args[@]}"'
        in work.split("python -B scripts/run_with_resources.py", 1)[1].split(
            "--output-dir", 1
        )[0]
    )


@pytest.mark.parametrize("part", ["", "b", "unreviewed"])
def test_locked312_part_identity_mismatch_fails_before_capacity_probe(part):
    job = _workflow()["jobs"]["candidate-standard"]
    work = next(step["run"] for step in job["steps"] if step.get("id") == "work")
    prelude = work.split("python -B scripts/run_with_resources.py", 1)[0]
    result = subprocess.run(
        ["bash", "-c", "python() { echo unexpected-probe; }\n" + prelude],
        env={
            **os.environ,
            "CANDIDATE_ID": "locked312-a",
            "CANDIDATE_PART": part,
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0 and "unexpected-probe" not in result.stdout


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
    assert f"locked312 initial available bytes: {available_kib * 1024}" in result.stdout
    assert ("locked312 needs 12 GiB" in result.stderr) == bool(result.returncode)


@pytest.mark.parametrize("work_exit", [0, 7])
def test_normal_part_preserves_the_resource_observer_exit(tmp_path, work_exit):
    job = _workflow()["jobs"]["candidate-standard"]
    work = next(step["run"] for step in job["steps"] if step.get("id") == "work")
    launcher = work.split("bash -euo pipefail - <<'BASH'\n", 1)[0]
    assert work.rstrip().endswith("BASH")
    stub = (
        "python() {\n"
        '  if [[ "${2:-}" == -c ]]; then return 0; fi\n'
        '  return "$TEST_WORK_EXIT"\n'
        "}\n"
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            stub + launcher + "bash -euo pipefail - <<'BASH'\nexit 0\nBASH\n",
        ],
        env={
            **os.environ,
            "CANDIDATE_ID": "locked312-a",
            "CANDIDATE_PART": "a",
            "CANDIDATE_COHORT": "internal_docs/precision/ci_standard/py312-owners.json",
            "CANDIDATE_DIR": str(tmp_path / "evidence"),
            "RUNNER_TEMP": str(tmp_path),
            "GITHUB_WORKSPACE": str(tmp_path),
            "TEST_WORK_EXIT": str(work_exit),
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode == work_exit, result.stderr


def test_fixed_parts_keep_the_full_profile_serial_and_reject_parallel_policy():
    job = _workflow()["jobs"]["candidate-standard"]
    matrix = job["strategy"]["matrix"]["include"]
    assert len(matrix) == 9
    assert {entry["id"] for entry in matrix} == {
        "locked311",
        "locked312-a",
        "locked312-b",
        "locked313",
        "locked314",
        "minimum312",
        "normalization312",
        "normalization313",
        "node312",
    }
    parts = [entry for entry in matrix if entry.get("part")]
    assert [(entry["id"], entry["part"]) for entry in parts] == [
        ("locked312-a", "a"),
        ("locked312-b", "b"),
    ]
    assert all(
        entry["python"] == "3.12"
        and entry["cohort"] == "py312-owners"
        and entry["role"] == "locked"
        for entry in parts
    )
    work = next(step["run"] for step in job["steps"] if step.get("id") == "work")
    assert "execution_rss_mib=4096" in work and "execution_rss_mib=5120" in work
    assert "live_available_mib=2048" in work and "live_available_mib=8192" in work
    assert (
        '--min-available-mib "$live_available_mib"' in work
        and "--timeout-seconds 1100" in work
    )
    assert "required = 12288 * 1024 * 1024" in work and job["timeout-minutes"] == 20
    assert '--_ci-provision-campaign "$CANDIDATE_COHORT" "${selection_args[@]}"' in work
    assert 'render_args+=(--ci-standard-part "$CANDIDATE_PART")' in work
    assert 'part_args=(--ci-standard-part "$CANDIDATE_PART")' in work
    assert 'scripts/run_pytest_batches.py "${part_args[@]}"' in work
    assert '--batch-size 500 --output-dir "$CANDIDATE_DIR/run"' in work
    assert "--min-disk-free-mib 4096" in work
    for forbidden in (
        "--diagnostic-only",
        "diagnostic-execution-policy",
        "--max-workers",
        "--ci-waves",
        "--batch-size 250",
        "execution_rss_mib=8192",
        "--max-rss-mib 8192",
    ):
        assert forbidden not in work


def test_hosted_cache_preparation_stays_between_generation_and_full_checks():
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


def test_primary_results_precede_bulk_receipts_without_changing_archive_layout():
    steps = _workflow()["jobs"]["candidate-standard"]["steps"]
    primary = next(step for step in steps if step.get("id") == "primary")
    pack = next(step for step in steps if step.get("id") == "pack")
    receipts = next(step for step in steps if step.get("id") == "receipts")
    assert steps.index(primary) < steps.index(pack) < steps.index(receipts)
    assert primary["if"] == "always()"
    assert primary["with"]["name"] == (
        "candidate-primary-${{ matrix.id }}-${{ job.status }}"
    )
    assert primary["with"]["path"].splitlines() == [
        "${{ runner.temp }}/ci-candidate-${{ matrix.id }}-work-resources/summary.json",
        "${{ runner.temp }}/ci-candidate-${{ matrix.id }}/standard-start-capacity.json",
        "${{ runner.temp }}/ci-candidate-${{ matrix.id }}/run/run.json",
        "${{ runner.temp }}/ci-candidate-${{ matrix.id }}/run/primary/",
    ]
    assert receipts["with"]["name"] == (
        "candidate-standard-${{ matrix.id }}-${{ job.status }}"
    )
    assert receipts["with"]["path"].splitlines() == [
        "${{ runner.temp }}/ci-candidate-${{ matrix.id }}.tar.gz",
        "${{ runner.temp }}/ci-candidate-${{ matrix.id }}-accounting.json",
        "${{ runner.temp }}/ci-candidate-${{ matrix.id }}-pack-resources/",
    ]


@pytest.mark.parametrize("work_exit", [0, 7])
def test_cleanup_retains_optional_source_originals_for_later_failures(
    tmp_path, work_exit
):
    steps = _workflow()["jobs"]["candidate-standard"]["steps"]
    work = next(step["run"] for step in steps if step.get("id") == "work")
    cleanup = work.split("cleanup() {", 1)[1].split("trap cleanup EXIT", 1)[0]
    evidence = tmp_path / "evidence"
    (evidence / "setup-diagnostics").mkdir(parents=True)
    original = evidence / "optional-sources/maxtext/original.py"
    original.parent.mkdir(parents=True)
    original.write_bytes(b"complete pinned source bytes")
    result = subprocess.run(
        [
            "bash",
            "-c",
            "set +e\ncleanup() {" + cleanup + f"\n(exit {work_exit})\ncleanup\n",
        ],
        env={
            **os.environ,
            "CANDIDATE_DIR": str(evidence),
            "CANDIDATE_TOOLS": str(tmp_path / "tools"),
            "CANDIDATE_VENV": str(tmp_path / "venv"),
            "CANDIDATE_NODE_PARENT": str(tmp_path / "node"),
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode == work_exit, result.stderr
    assert original.read_bytes() == b"complete pinned source bytes"


@pytest.mark.parametrize(
    "prior_status,primary_outcome,has_failure",
    [
        ("success", "success", False),
        ("success", "failure", True),
        ("failure", "success", True),
    ],
)
def test_bulk_pack_separates_routine_sources_and_preserves_full_failure_payload(
    tmp_path, prior_status, primary_outcome, has_failure
):
    steps = _workflow()["jobs"]["candidate-standard"]["steps"]
    pack = next(step for step in steps if step.get("id") == "pack")
    archive_script = (
        "paths=()"
        + pack["run"].split("paths=()", 1)[1].split("python -B - <<'PY'", 1)[0]
    )
    evidence = tmp_path / "ci-candidate-locked312-a"
    originals = {
        evidence / "run/run.json": b"original canonical receipt",
        evidence / "optional-sources/maxtext/original.py": b"complete pinned source",
        tmp_path / "workspace/onnx/model.onnx": b"complete original model",
        tmp_path / "owned-tmp/data.bin": b"complete temporary output",
    }
    for path, payload in originals.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", archive_script],
        env={
            **os.environ,
            "CANDIDATE_DIR": str(evidence),
            "CANDIDATE_TMP": str(tmp_path / "owned-tmp"),
            "CANDIDATE_NODE_PARENT": str(tmp_path / "node"),
            "RUNNER_TEMP": str(tmp_path),
            "GITHUB_WORKSPACE": str(tmp_path / "workspace"),
            "CANDIDATE_PRIOR_STATUS": prior_status,
            "CANDIDATE_PRIMARY_OUTCOME": primary_outcome,
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    with tarfile.open(str(evidence) + ".tar.gz") as archive:
        assert "ci-candidate-locked312-a/run/run.json" in archive.getnames()
        assert not any("optional-sources" in name for name in archive.getnames())
    failure = Path(str(evidence) + "-failure-payloads.tar.gz")
    assert failure.exists() == has_failure
    if has_failure:
        with tarfile.open(failure) as archive:
            for name, payload in (
                ("optional-sources/maxtext/original.py", b"complete pinned source"),
                ("onnx/model.onnx", b"complete original model"),
                ("owned-tmp/data.bin", b"complete temporary output"),
            ):
                stream = archive.extractfile(name)
                assert stream is not None and stream.read() == payload
    assert all(path.read_bytes() == payload for path, payload in originals.items())
    failure_upload = next(
        step for step in steps if step.get("id") == "failure_payloads"
    )
    raw_upload = next(
        step for step in steps if "raw failure evidence" in step.get("name", "")
    )
    assert "steps.primary.outcome != 'success'" in failure_upload["if"]
    assert "steps.primary.outcome != 'success'" in raw_upload["if"]


def test_fixed_join_requires_both_complete_parts_and_original_authority_paths():
    job = _workflow()["jobs"]["candidate-standard-locked312-join"]
    assert job["needs"] == "candidate-standard" and "always()" in job["if"]
    assert job["timeout-minutes"] == 20
    downloads = [
        step for step in job["steps"] if "download-artifact@" in step.get("uses", "")
    ]
    assert len(downloads) == 2
    for part, step in zip(("a", "b"), downloads):
        assert step["uses"] == (
            "actions/download-artifact@37930b1c2abaa49bbe596cd826c3c89aef350131"
        )
        assert step["if"] == "always()"
        assert step["with"]["name"] == f"candidate-standard-locked312-{part}-success"
        assert step["with"]["path"] == (
            "${{ runner.temp }}/ci-standard-part-downloads/" + part
        )
    work = next(step["run"] for step in job["steps"] if step.get("id") == "join")
    assert (
        "--max-rss-mib 4096 --min-available-mib 2048 --min-disk-free-mib 4096" in work
    )
    assert "--timeout-seconds 1100" in work
    assert work.count("python scripts/generate_tests.py") == 1
    assert work.index("python scripts/generate_tests.py") < work.index(
        "--ci-join-standard-parts"
    )
    assert '"$RUNNER_TEMP/ci-candidate-locked312-a/run"' in work
    assert '"$RUNNER_TEMP/ci-candidate-locked312-b/run"' in work
    assert '--ci-standard-profile "$CANDIDATE_COHORT"' in work
    assert "--collect-only" not in work and "--max-workers" not in work
    restore = work.split("<<'PY_RESTORE_PARTS'\n", 1)[1].split("\nPY_RESTORE_PARTS", 1)[
        0
    ]
    assert "path.is_absolute()" in restore and '".." in path.parts' in restore
    assert "member.isdir() or member.isfile()" in restore
    assert "50000" in restore and "512 * 1024**2" in restore
    assert "original part destination already exists" in restore


def test_join_observes_current_attempt_jobs_before_acceptance_without_token_logs():
    job = _workflow()["jobs"]["candidate-standard-locked312-join"]
    assert job["permissions"] == {"contents": "read", "actions": "read"}
    step = next(step for step in job["steps"] if step.get("id") == "join")
    assert step["env"]["CANDIDATE_JOBS_API_TOKEN"] == "${{ github.token }}"
    work = step["run"]
    assert work.index("unset CANDIDATE_JOBS_API_TOKEN") < work.index("poetry install")
    assert work.index("PY_JOB_OBSERVATIONS\n") < work.index("--ci-join-standard-parts")
    assert '--ci-job-observations "$CANDIDATE_DIR/job-observations.json"' in work
    snapshot = work.split("<<'PY_JOB_OBSERVATIONS'\n", 1)[1].split(
        "\nPY_JOB_OBSERVATIONS", 1
    )[0]
    assert 'os.environ["GITHUB_RUN_ID"]' in snapshot
    assert 'os.environ["GITHUB_RUN_ATTEMPT"]' in snapshot
    assert '"/attempts/"' in snapshot and '"/jobs?per_page=100"' in snapshot
    assert "RejectRedirects" in snapshot
    assert 'response.headers.get("Link")' in snapshot
    assert "2 * 1024**2 + 1" in snapshot and "timeout=20" in snapshot
    assert "print(" not in snapshot
    assert "redirect_request" in snapshot


def _assert_early_env_has_no_runner_context(workflow):
    # Actions job/workflow env excludes runner, unlike step env/with. YAML and
    # bash parsers cannot detect this expression-context admission boundary.
    scopes = [("workflow", workflow.get("env", {}))]
    scopes.extend((name, job.get("env", {})) for name, job in workflow["jobs"].items())
    for scope, environment in scopes:
        for key, value in environment.items():
            assert not (
                isinstance(value, str)
                and re.search(
                    r"\$\{\{(?:(?!\}\}).)*\brunner\s*(?:\.|\[)", value, re.DOTALL
                )
            ), f"runner context is unavailable in {scope} env {key}"


def test_workflow_and_job_env_use_only_available_actions_contexts():
    _assert_early_env_has_no_runner_context(_workflow())


@pytest.mark.parametrize("scope", ["workflow", "job"])
@pytest.mark.parametrize(
    "expression",
    ["${{ runner.temp }}", "${{ format('{0}/receipts', runner['temp']) }}"],
)
def test_unavailable_runner_context_is_rejected_before_workflow_admission(
    scope, expression
):
    workflow = deepcopy(_workflow())
    if scope == "workflow":
        environment = workflow.setdefault("env", {})
    else:
        environment = workflow["jobs"]["candidate-standard-locked312-join"].setdefault(
            "env", {}
        )
    environment["FORBIDDEN_PATH"] = expression
    with pytest.raises(AssertionError, match="runner context is unavailable"):
        _assert_early_env_has_no_runner_context(workflow)


def test_join_paths_are_initialized_from_real_runner_temp_before_any_consumer(tmp_path):
    workflow = _workflow()
    job = workflow["jobs"]["candidate-standard-locked312-join"]
    step = job["steps"][0]
    assert step["id"] == "join-paths" and step["shell"] == "bash"
    names = {
        "CANDIDATE_DIR": "ci-candidate-locked312-join",
        "CANDIDATE_TOOLS": "ci-candidate-locked312-join-tools",
        "CANDIDATE_VENV": "ci-candidate-locked312-join-venv",
        "CANDIDATE_TMP": "ci-candidate-locked312-join-tmp",
        "CANDIDATE_DOWNLOADS": "ci-standard-part-downloads",
    }
    assert set(names).isdisjoint(job["env"])
    actual_temporary = tmp_path / "actual runner temporary with spaces"
    actual_temporary.mkdir()
    environment_file = tmp_path / "GITHUB_ENV"
    environment_file.touch()
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", step["run"]],
        env={
            **os.environ,
            "RUNNER_TEMP": str(actual_temporary),
            "GITHUB_ENV": str(environment_file),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    written = dict(
        line.split("=", 1) for line in environment_file.read_text().splitlines()
    )
    assert written == {
        key: str(actual_temporary / suffix) for key, suffix in names.items()
    }
    assert not list(actual_temporary.iterdir())
    assert "${{ runner." not in step["run"]
    _assert_early_env_has_no_runner_context(workflow)
