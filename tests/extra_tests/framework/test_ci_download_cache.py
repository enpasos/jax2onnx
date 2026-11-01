# tests/extra_tests/framework/test_ci_download_cache.py

"""Hosted cache safety, exact acceptance routing and bounded CI transport."""

from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
import hashlib
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


def test_source_metadata_anchor_is_provisioned_in_shallow_jobs_before_pytest(tmp_path):
    workflow = _workflow()
    steps = workflow["jobs"]["test"]["steps"]
    history = next(
        step
        for step in steps
        if step.get("name") == "Fetch frozen original SOURCE history"
    )
    position = steps.index(history)
    assert steps[position - 1]["name"] == "Verify checked-out commit"
    assert steps[position + 1]["uses"].startswith("actions/setup-python@")
    assert history["env"]["EXPECTED_CHECKOUT_SHA"] == (
        "${{ needs.acceptance-context.outputs.head_sha }}"
    )
    assert workflow["jobs"]["test"]["strategy"]["matrix"]["python-version"] == [
        "3.11",
        "3.12",
        "3.13",
        "3.14",
    ]
    assert "continue-on-error" not in history
    script = history["run"]
    for name in ("onnxruntime-web-full", "onnxruntime-web-chrome-full"):
        browser_steps = _nightly_workflow()["jobs"][name]["steps"]
        browser_history = next(
            step
            for step in browser_steps
            if step.get("name") == "Fetch frozen original SOURCE history"
        )
        index = browser_steps.index(browser_history)
        assert browser_steps[index - 1]["name"] == "Verify selected acceptance commit"
        assert browser_steps[index + 1]["uses"].startswith("actions/setup-python@")
        assert (
            browser_history["env"]["EXPECTED_CHECKOUT_SHA"]
            == "${{ needs.check-changes.outputs.head_sha }}"
        )
        assert (
            browser_history["run"] == script
            and "continue-on-error" not in browser_history
        )
    anchor = "ccc861b0dff5042958a267addc43b622a4b61d49"
    fetch = 'git fetch --no-tags --depth=1 origin "$source_anchor"'
    assert script.splitlines()[0] == "set -euo pipefail"
    assert script.count("source_anchor=" + anchor) == script.count(fetch) == 1
    assert (
        'test "$(git rev-parse --verify \'ccc861b0^{commit}\')" = "$source_anchor"'
        in script
    )
    assert 'git archive "$source_anchor" tests > /dev/null' in script
    assert (
        'test "$(git rev-parse --verify \'HEAD^{commit}\')" = "$source_head"' in script
    )
    assert 'test -z "$(git status --porcelain --untracked-files=no)"' in script
    candidate = workflow["jobs"]["candidate-standard"]
    work = next(step["run"] for step in candidate["steps"] if step.get("id") == "work")
    body = "\n".join("  " + line for line in script.splitlines()[1:])
    guarded = (
        'if [[ "$CANDIDATE_LANE" == extra && "$CANDIDATE_ROLE" == locked ]]; then\n'
        + body
        + "\nfi"
    )
    assert work.count(guarded) == 1
    assert (
        work.index("bash -euo pipefail - <<'BASH'")
        < work.index('test "$actual_commit" = "$expected_commit"')
        < work.index(guarded)
        < work.index('source "$CANDIDATE_DIR/profile.env"')
        < work.index("python scripts/generate_tests.py")
        < work.index("python -B scripts/run_pytest_batches.py")
    )
    for name, job in workflow["jobs"].items():
        if name not in {"test", "candidate-standard"}:
            assert all(
                "source_anchor=" not in step.get("run", "")
                for step in job.get("steps", [])
            )

    environment = {
        name: value
        for name, value in os.environ.items()
        if name
        not in {
            "GIT_DIR",
            "GIT_WORK_TREE",
            "GIT_COMMON_DIR",
            "GIT_INDEX_FILE",
            "GIT_OBJECT_DIRECTORY",
            "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        }
    }
    environment.update(GIT_OPTIONAL_LOCKS="0", GIT_ALLOW_PROTOCOL="file")

    def git(directory, *arguments, check=True):
        result = subprocess.run(
            ["git", "-c", "core.filemode=false", "-C", str(directory), *arguments],
            env=environment,
            capture_output=True,
            timeout=30,
        )
        if check:
            assert result.returncode == 0, result.stderr.decode(errors="replace")
        return result

    head = git(ROOT, "rev-parse", "--verify", "HEAD^{commit}").stdout.decode().strip()
    git(ROOT, "cat-file", "-e", anchor + "^{commit}")
    original_archive = hashlib.sha256(
        git(ROOT, "archive", anchor, "tests").stdout
    ).digest()
    objects = ROOT / (
        git(ROOT, "rev-parse", "--git-path", "objects").stdout.decode().strip()
    )

    def bare_origin(name, object_directory, refs):
        origin = tmp_path / name
        origin.mkdir()
        git(origin, "init", "--bare", "--quiet")
        # Only temporary repositories receive alternates/refs. ROOT objects are
        # served read-only; no clone, copied pack or ROOT configuration is needed.
        (origin / "objects/info/alternates").write_text(str(object_directory) + "\n")
        for ref, commit in refs.items():
            git(origin, "update-ref", "refs/heads/" + ref, commit)
        return origin

    origin = bare_origin(
        "origin", objects.resolve(), {"current": head, "anchor": anchor}
    )
    checkout = tmp_path / "shallow"
    checkout.mkdir()
    git(checkout, "init", "--quiet")
    git(checkout, "remote", "add", "origin", origin.as_uri())
    git(checkout, "fetch", "--no-tags", "--depth=1", "origin", head)
    # History is the real current HEAD, with one checked-out file to bound the
    # fixture's retained bytes; sparse checkout does not make the anchor present.
    git(checkout, "config", "core.sparseCheckout", "true")
    (checkout / ".git/info/sparse-checkout").write_text("/README.md\n")
    git(checkout, "checkout", "--quiet", "--detach", "FETCH_HEAD")
    assert git(checkout, "rev-parse", "--is-shallow-repository").stdout == b"true\n"
    assert git(checkout, "cat-file", "-e", anchor + "^{commit}", check=False).returncode
    untracked = checkout / "preserved-untracked.txt"
    untracked.write_bytes(b"original untracked bytes\x00")

    def snapshot():
        return (
            git(checkout, "rev-parse", "--verify", "HEAD^{commit}").stdout,
            git(checkout, "ls-files", "--stage", "-z").stdout,
            git(checkout, "status", "--porcelain", "--untracked-files=all").stdout,
            (checkout / "README.md").read_bytes(),
            untracked.read_bytes(),
        )

    before = snapshot()
    missing = bare_origin("missing", checkout / ".git/objects", {"current": head})
    (missing / "shallow").write_text(head + "\n")
    assert git(missing, "cat-file", "-e", anchor + "^{commit}", check=False).returncode
    wrong_fetch = fetch.replace('"$source_anchor"', '"$source_head"')
    for name, remote, commands in (
        ("missing", missing, script),
        ("wrong-object", origin, script.replace(fetch, wrong_fetch)),
        ("valid", origin, script),
    ):
        git(checkout, "remote", "set-url", "origin", remote.as_uri())
        marker = tmp_path / (name + "-pytest-body")
        result = subprocess.run(
            ["bash", "-c", commands + '\nprintf reached > "$PROVISION_MARKER"\n'],
            cwd=checkout,
            env={
                **environment,
                "EXPECTED_CHECKOUT_SHA": head,
                "PROVISION_MARKER": str(marker),
            },
            capture_output=True,
            timeout=30,
        )
        assert snapshot() == before
        if name == "valid":
            assert result.returncode == 0, result.stderr.decode(errors="replace")
            assert marker.read_text() == "reached"
        else:
            assert result.returncode != 0 and not marker.exists()
            assert git(
                checkout, "cat-file", "-e", anchor + "^{commit}", check=False
            ).returncode
    assert (
        git(checkout, "rev-parse", "--verify", "ccc861b0^{commit}").stdout
        == (anchor + "\n").encode()
    )
    fetched_archive = hashlib.sha256(
        git(checkout, "archive", anchor, "tests").stdout
    ).digest()
    assert fetched_archive == original_archive
    assert hashlib.sha256(git(ROOT, "archive", anchor, "tests").stdout).digest() == (
        original_archive
    )
    assert snapshot() == before


@pytest.mark.parametrize(
    "candidate,part",
    [
        ("locked311", ""),
        ("locked311-a", "a"),
        ("locked311-b", "b"),
        ("locked313-a", "a"),
        ("locked313-b", "b"),
        ("locked312-a", "a"),
        ("locked312-b", "b"),
        ("locked313", ""),
        ("locked314", ""),
        ("locked314-a", "a"),
        ("locked314-b", "b"),
        ("minimum312", ""),
        ("normalization312", ""),
        ("normalization313", ""),
        ("node312", ""),
        ("locked312", ""),
        ("focused311", ""),
        ("focused312", ""),
        ("focused313", ""),
        ("focused314", ""),
    ],
)
def test_normal_five_gib_limits_apply_only_to_fixed_locked312_parts(candidate, part):
    job = _workflow()["jobs"]["candidate-standard"]
    work = next(step["run"] for step in job["steps"] if step.get("id") == "work")
    prelude = work.split("python -B scripts/run_with_resources.py", 1)[0]
    focused = candidate.startswith("focused")
    profile = "internal_docs/precision/ci_standard/" + (
        f"py{candidate[-3:]}-extra.json" if focused else "py312-owners.json"
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            'python() { printf "capacity probe stub\\n"; }\n'
            + prelude
            + '\nprintf "%s\\n" "$execution_rss_mib" "$live_available_mib" '
            '"$execution_work_seconds" "${execution_profile_args[@]}"',
        ],
        env={
            **os.environ,
            "CANDIDATE_ID": candidate,
            "CANDIDATE_PART": part,
            "CANDIDATE_COHORT": profile,
            "CANDIDATE_ROLE": "locked",
            "CANDIDATE_LANE": "extra" if focused else "standard",
            "CANDIDATE_SCOPE": "focused" if focused else "",
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    observed = [
        line for line in result.stdout.splitlines() if line != "capacity probe stub"
    ]
    expected = ["4096", "2048", "1100"]
    if candidate in {"locked312-a", "locked312-b"}:
        expected = ["5120", "8192", "1100", "--locked312-standard-profile", profile]
    elif focused:
        expected = ["4096", "2048", "2000", "--focused-extra-profile", profile]
    assert observed == expected
    assert result.stdout.count("capacity probe stub") == (
        candidate in {"locked312-a", "locked312-b"}
    )
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
    matrix = [
        entry
        for entry in job["strategy"]["matrix"]["include"]
        if entry.get("lane", "standard") == "standard"
    ]
    assert len(matrix) == 12
    assert {entry["id"] for entry in matrix} == {
        "locked311-a",
        "locked311-b",
        "locked312-a",
        "locked312-b",
        "locked313-a",
        "locked313-b",
        "locked314-a",
        "locked314-b",
        "minimum312",
        "normalization312",
        "normalization313",
        "node312",
    }
    parts = [entry for entry in matrix if entry.get("part")]
    assert [(entry["id"], entry["part"]) for entry in parts] == [
        ("locked" + minor + "-" + part, part)
        for minor in ("311", "312", "313", "314")
        for part in ("a", "b")
    ]
    assert all(
        entry["python"] == "3." + entry["id"][7:9]
        and entry["cohort"] == "py" + entry["id"][6:9] + "-owners"
        and entry["role"] == "locked"
        for entry in parts
    )
    work = next(step["run"] for step in job["steps"] if step.get("id") == "work")
    assert "execution_rss_mib=4096" in work and "execution_rss_mib=5120" in work
    assert "live_available_mib=2048" in work and "live_available_mib=8192" in work
    assert (
        '--min-available-mib "$live_available_mib"' in work
        and '--timeout-seconds "$execution_work_seconds"' in work
        and "execution_work_seconds=1100" in work
    )
    assert "required = 12288 * 1024 * 1024" in work
    assert job["timeout-minutes"] == "${{ matrix.scope == 'focused' && 35 || 20 }}"
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


def test_focused_extra_rows_add_four_global_cohorts_with_separate_job_limits():
    job = _workflow()["jobs"]["candidate-standard"]
    matrix = job["strategy"]["matrix"]["include"]
    assert len(matrix) == 17 and job["strategy"]["fail-fast"] is False
    focused = [entry for entry in matrix if entry.get("scope") == "focused"]
    assert focused == [
        {
            "id": "focused" + minor,
            "python": "3." + minor[1:],
            "cohort": "py" + minor + "-extra",
            "role": "locked",
            "lane": "extra",
            "scope": "focused",
        }
        for minor in ("311", "312", "313", "314")
    ]
    for entry in focused:
        profile_path = (
            ROOT / "internal_docs/precision/ci_standard" / (entry["cohort"] + ".json")
        )
        profile = json.loads(profile_path.read_text())
        assert profile.get("collection_args", []) == []
        assert "standard_parts" not in profile
    assert job["env"]["CANDIDATE_LANE"] == "${{ matrix.lane || 'standard' }}"
    assert job["env"]["CANDIDATE_SCOPE"] == "${{ matrix.scope || '' }}"
    assert job["timeout-minutes"] == "${{ matrix.scope == 'focused' && 35 || 20 }}"
    assert job["name"] == (
        "candidate ${{ matrix.scope == 'all' && 'support EXTRA all' || "
        "matrix.lane == 'extra' && 'focused EXTRA' || 'STANDARD' }} "
        "(${{ matrix.label || matrix.id }})"
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("CANDIDATE_ROLE", "minimum"),
        ("CANDIDATE_LANE", "standard"),
        ("CANDIDATE_SCOPE", "all"),
        ("CANDIDATE_PART", "a"),
    ],
)
def test_focused_resource_extension_requires_its_exact_lane_role_scope_and_no_part(
    key, value
):
    job = _workflow()["jobs"]["candidate-standard"]
    work = next(step["run"] for step in job["steps"] if step.get("id") == "work")
    prelude = work.split("python -B scripts/run_with_resources.py", 1)[0]
    environment = {
        **os.environ,
        "CANDIDATE_ID": "focused312",
        "CANDIDATE_ROLE": "locked",
        "CANDIDATE_LANE": "extra",
        "CANDIDATE_SCOPE": "focused",
        "CANDIDATE_PART": "",
        "CANDIDATE_COHORT": "internal_docs/precision/ci_standard/py312-extra.json",
    }
    environment[key] = value
    result = subprocess.run(
        ["bash", "-c", prelude + '\nprintf "unexpected-admission\\n"'],
        env=environment,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0 and "unexpected-admission" not in result.stdout


@pytest.mark.parametrize(
    "candidate",
    [
        "locked311",
        "locked311-a",
        "locked311-b",
        "locked313-a",
        "locked313-b",
        "locked312-a",
        "locked312-b",
        "locked313",
        "locked314",
        "locked314-a",
        "locked314-b",
        "minimum312",
        "minimum312-extra",
        "normalization312",
        "normalization313",
        "node312",
        "focused311",
        "focused312",
        "focused313",
        "focused314",
    ],
)
def test_canonical_dispatch_preserves_standard_parts_and_explicit_focused_scope(
    candidate,
):
    job = _workflow()["jobs"]["candidate-standard"]
    entry = (
        next(
            item
            for item in job["strategy"]["matrix"]["include"]
            if item["id"] == candidate
        )
        if candidate not in {"locked311", "locked313", "locked314"}
        else {"role": "locked"}
    )
    work = next(step["run"] for step in job["steps"] if step.get("id") == "work")
    dispatch = (
        "part_args=()"
        + work.split("part_args=()", 1)[1].split(
            "python -B scripts/run_pytest_batches.py", 1
        )[0]
    )
    result = subprocess.run(
        [
            "bash",
            "-euo",
            "pipefail",
            "-c",
            dispatch + '\nprintf "%s\\n" "${part_args[@]}" "${lane_args[@]}"',
        ],
        env={
            **os.environ,
            "CANDIDATE_ID": candidate,
            "CANDIDATE_PART": entry.get("part", ""),
            "CANDIDATE_LANE": entry.get("lane", "standard"),
            "CANDIDATE_SCOPE": entry.get("scope", ""),
            "CANDIDATE_ROLE": entry["role"],
            "CANDIDATE_COHORT": "internal_docs/precision/ci_standard/"
            + entry.get("cohort", "unused")
            + ".json",
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    expected = ["--ci-lane", entry.get("lane", "standard")]
    if entry.get("part"):
        expected = ["--ci-standard-part", entry["part"], *expected]
    if entry.get("lane") == "extra":
        expected += ["--ci-extra-scope", entry["scope"]]
    assert result.stdout.splitlines() == expected
    assert '"${lane_args[@]}" --ci-main-baseline "$CANDIDATE_DIR/main.json"' in work
    assert '--_ci-provision-campaign "$CANDIDATE_COHORT"' in work


@pytest.mark.parametrize(
    "candidate,needs_optional",
    [
        ("locked311", False),
        ("locked312-a", True),
        ("locked312-b", True),
        ("focused311", False),
        ("focused312", True),
        ("focused313", False),
        ("focused314", False),
    ],
)
def test_global_focused312_retains_both_optional_installs_and_renderer_sources(
    tmp_path, candidate, needs_optional
):
    job = _workflow()["jobs"]["candidate-standard"]
    work = next(step["run"] for step in job["steps"] if step.get("id") == "work")
    marker = work.index('mkdir "$CANDIDATE_DIR/optional-sources"')
    start = work.rindex("if [[", 0, marker)
    setup = work[start : work.index('if [[ "$CANDIDATE_ROLE" == minimum', marker)]
    render = (
        "render_args=()"
        + work.split("render_args=()", 1)[1].split(
            'if [[ "$CANDIDATE_ROLE" == node-smoke', 1
        )[0]
    )
    evidence = tmp_path / candidate
    evidence.mkdir()
    result = subprocess.run(
        [
            "bash",
            "-euo",
            "pipefail",
            "-c",
            'git() { if [[ "${3:-}" == rev-parse ]]; then '
            'printf "reviewed-ref\\n"; fi; return 0; }\n'
            'poetry() { printf "%s\\n" "$*"; }\n'
            + setup
            + render
            + "\nif (( ${#render_args[@]} )); then "
            'printf "%s\\n" "${render_args[@]}"; fi',
        ],
        env={
            **os.environ,
            "CANDIDATE_ID": candidate,
            "CANDIDATE_DIR": str(evidence),
            "JAX2ONNX_MAXTEXT_SRC": str(evidence / "optional-sources/maxtext"),
            "JAX2ONNX_MAXDIFFUSION_SRC": str(
                evidence / "optional-sources/maxdiffusion"
            ),
            "JAX2ONNX_MAXTEXT_REF": "reviewed-ref",
            "JAX2ONNX_MAXDIFFUSION_REF": "reviewed-ref",
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == (
        [
            "install --no-interaction --with maxtext",
            "install --no-interaction --with maxdiffusion",
            "--maxtext-source",
            str(evidence / "optional-sources/maxtext"),
            "--maxdiffusion-source",
            str(evidence / "optional-sources/maxdiffusion"),
        ]
        if needs_optional
        else []
    )
    assert (evidence / "optional-sources").exists() == needs_optional


def _candidate_quality_commands():
    job = _workflow()["jobs"]["candidate-standard"]
    work = next(step["run"] for step in job["steps"] if step.get("id") == "work")
    start = work.index("# Preserve Main's pinned all-files quality check")
    end = work.index('if [[ "$CANDIDATE_ID" == locked312-a', start)
    return job, work, work[start:end]


def test_candidate_quality_uses_original_pins_before_optional_installs_and_generation():
    job, work, commands = _candidate_quality_commands()
    owner_cohorts = {
        entry["id"]: entry["python"]
        for entry in job["strategy"]["matrix"]["include"]
        if entry.get("lane", "standard") == "standard"
        and entry["role"] == "locked"
        and entry.get("part") != "b"
    }
    assert owner_cohorts == {
        "locked311-a": "3.11",
        "locked312-a": "3.12",
        "locked313-a": "3.13",
        "locked314-a": "3.14",
    }
    original = (ROOT / "scripts/run_all_checks.sh").read_text()
    for invocation in (
        "poetry run python scripts/ensure_path_markers.py --fix",
        "poetry run pre-commit run --all-files",
    ):
        assert original.count(invocation) == commands.count(invocation) == 1
    assert (
        work.index("bash -euo pipefail - <<'BASH'")
        < work.index('source "$CANDIDATE_DIR/profile.env"')
        < work.index("poetry install --no-interaction")
        < work.index(commands)
        < work.index('poetry install --no-interaction --with "$group"')
        < work.index("python scripts/generate_tests.py")
        < work.index("python -B scripts/prepare_ci_standard.py")
    )
    assert "execution_work_seconds=1100" in work
    assert job["timeout-minutes"] == "${{ matrix.scope == 'focused' && 35 || 20 }}"
    assert "--all-files" in commands and "--hook-stage" not in commands
    assert "continue-on-error" not in next(
        step for step in job["steps"] if step.get("id") == "work"
    )
    assert not re.search(r"\b(?:SKIP|PRE_COMMIT_ALLOW_NO_CONFIG)\s*=", work)
    pins = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())["repos"]
    assert [(repo["repo"], repo.get("rev")) for repo in pins[:2]] == [
        ("https://github.com/psf/black", "26.10.1"),
        ("https://github.com/astral-sh/ruff-pre-commit", "v0.17.3"),
    ]


@pytest.mark.parametrize(
    "candidate,owns_quality",
    [
        ("locked311", True),
        ("locked311-a", True),
        ("locked311-b", False),
        ("locked313-a", True),
        ("locked313-b", False),
        ("locked312-a", True),
        ("locked312-b", False),
        ("locked313", True),
        ("locked314", True),
        ("locked314-a", True),
        ("locked314-b", False),
        ("minimum312", False),
        ("normalization312", False),
        ("normalization313", False),
        ("node312", False),
        ("focused311", False),
        ("focused312", False),
        ("focused313", False),
        ("focused314", False),
    ],
)
def test_candidate_quality_runs_once_per_locked_cohort_without_support_hooks(
    candidate, owns_quality
):
    job, _, commands = _candidate_quality_commands()
    entry = (
        next(
            value
            for value in job["strategy"]["matrix"]["include"]
            if value["id"] == candidate
        )
        if candidate not in {"locked311", "locked313", "locked314"}
        else {"role": "locked"}
    )
    result = subprocess.run(
        [
            "bash",
            "-euo",
            "pipefail",
            "-c",
            'poetry() { printf "%s\\n" "$*"; }\n'
            + commands
            + '\nprintf "%s\\n" after-quality\n',
        ],
        env={
            **os.environ,
            "CANDIDATE_ROLE": entry["role"],
            "CANDIDATE_PART": entry.get("part", ""),
            "CANDIDATE_LANE": entry.get("lane", "standard"),
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == (
        [
            "run python scripts/ensure_path_markers.py --fix",
            "run pre-commit run --all-files",
            "after-quality",
        ]
        if owns_quality
        else ["after-quality"]
    )


@pytest.mark.parametrize(
    "candidate",
    [
        pytest.param("locked311-a", id="locked311"),
        "locked312-a",
        pytest.param("locked313-a", id="locked313"),
        pytest.param("locked314-a", id="locked314"),
    ],
)
@pytest.mark.parametrize("failure", ["markers", "hooks"])
def test_candidate_quality_failure_stops_work_before_optional_setup(candidate, failure):
    job, _, commands = _candidate_quality_commands()
    entry = next(
        value
        for value in job["strategy"]["matrix"]["include"]
        if value["id"] == candidate
    )
    assert entry["role"] == "locked"
    assert entry.get("lane", "standard") == "standard"
    assert entry.get("part") == "a"
    result = subprocess.run(
        [
            "bash",
            "-euo",
            "pipefail",
            "-c",
            'poetry() {\n  printf "%s\\n" "$*"\n'
            '  if [[ "$*" == "run python scripts/ensure_path_markers.py --fix" '
            '&& "$QUALITY_FAILURE" == markers ]]; then return 73; fi\n'
            '  if [[ "$*" == "run pre-commit run --all-files" ]]; then return 74; fi\n'
            "}\n" + commands + '\nprintf "%s\\n" unexpected-optional-setup\n',
        ],
        env={
            **os.environ,
            "CANDIDATE_ROLE": entry["role"],
            "CANDIDATE_PART": entry.get("part", ""),
            "CANDIDATE_LANE": entry.get("lane", "standard"),
            "QUALITY_FAILURE": failure,
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode == (73 if failure == "markers" else 74), result.stderr
    assert "unexpected-optional-setup" not in result.stdout
    assert result.stdout.count("run pre-commit run --all-files") == (failure == "hooks")


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
    active = (
        "matrix.scope != 'focused' || "
        "needs.acceptance-context.outputs.full_acceptance != 'true' || "
        "needs.acceptance-context.outputs.routine_activated != 'true'"
    )
    assert primary["if"] == "(" + active + ") && (always())"
    for scope in ("", "all", "focused"):
        for activated, full in (
            ("false", "false"),
            ("false", "true"),
            ("true", "false"),
            ("true", "true"),
        ):

            def condition(expression):
                return (
                    expression.replace("matrix.scope", repr(scope))
                    .replace(
                        "needs.acceptance-context.outputs.full_acceptance",
                        repr(full),
                    )
                    .replace(
                        "needs.acceptance-context.outputs.routine_activated",
                        repr(activated),
                    )
                    .replace("always()", "True")
                    .replace("||", " or ")
                    .replace("&&", " and ")
                )

            # always() remains true even after failed work. Qualification never
            # drops its primary before packing; later inactive full-mode F has
            # no executing step and cannot emit a manufactured acceptance file.
            required = scope != "focused" or full != "true" or activated != "true"
            assert eval(condition(primary["if"]), {"__builtins__": {}}, {}) is required
            if not required:
                assert all(
                    eval(condition(step["if"]), {"__builtins__": {}}, {}) is False
                    for step in steps
                )
    assert primary["with"]["name"] == (
        "${{ matrix.lane == 'extra' && 'candidate-extra-primary' "
        "|| 'candidate-primary' }}"
        "-${{ matrix.id }}-${{ job.status }}"
    )
    assert primary["with"]["path"].splitlines() == [
        "${{ runner.temp }}/ci-candidate-${{ matrix.id }}-work-resources/summary.json",
        "${{ runner.temp }}/ci-candidate-${{ matrix.id }}/standard-start-capacity.json",
        "${{ runner.temp }}/ci-candidate-${{ matrix.id }}/run/run.json",
        "${{ runner.temp }}/ci-candidate-${{ matrix.id }}/run/primary/",
    ]
    assert receipts["with"]["name"] == (
        "${{ matrix.lane == 'extra' && 'candidate-extra' || 'candidate-standard' }}"
        "-${{ matrix.id }}-${{ job.status }}"
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
@pytest.mark.parametrize("candidate", ["locked312-a", "focused312"])
def test_bulk_pack_separates_routine_sources_and_preserves_full_failure_payload(
    tmp_path, prior_status, primary_outcome, has_failure, candidate
):
    steps = _workflow()["jobs"]["candidate-standard"]["steps"]
    pack = next(step for step in steps if step.get("id") == "pack")
    archive_script = (
        "paths=()"
        + pack["run"].split("paths=()", 1)[1].split("python -B - <<'PY'", 1)[0]
    )
    evidence = tmp_path / ("ci-candidate-" + candidate)
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
        assert evidence.name + "/run/run.json" in archive.getnames()
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


@pytest.mark.parametrize(
    "candidate",
    [
        "locked311-a",
        "locked312-a",
        "locked313-a",
        "locked314-a",
        "focused312",
        "minimum312-extra",
    ],
)
def test_accounting_keeps_focused_extra_budgets_separate_from_standard(
    tmp_path, candidate
):
    job = _workflow()["jobs"]["candidate-standard"]
    pack = next(step for step in job["steps"] if step.get("id") == "pack")
    code = pack["run"].rsplit("python -B - <<'PY'\n", 1)[1].split("\nPY", 1)[0]
    evidence = tmp_path / ("ci-candidate-" + candidate)
    Path(str(evidence) + ".tar.gz").write_bytes(b"bounded receipt archive")
    focused = candidate == "focused312"
    extra = focused or candidate == "minimum312-extra"
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        env={
            **os.environ,
            "CANDIDATE_DIR": str(evidence),
            "CANDIDATE_ID": candidate,
            "CANDIDATE_LANE": "extra" if extra else "standard",
            "CANDIDATE_SCOPE": "focused" if focused else "all" if extra else "",
            "CANDIDATE_PART": "" if extra else "a",
            "EXPECTED_CHECKOUT_SHA": "a" * 40,
            "CANDIDATE_PRIOR_STATUS": "success",
            "CANDIDATE_PRIMARY_OUTCOME": "success",
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    accounting = json.loads(Path(str(evidence) + "-accounting.json").read_text())
    assert accounting["campaign_family"] == ("EXTRA" if extra else "STANDARD")
    assert accounting["ci_extra_scope"] == (
        "focused" if focused else "all" if extra else None
    )
    assert accounting["work_timeout_seconds"] == (2000 if focused else 1100)
    assert accounting["whole_job_timeout_seconds"] == (2100 if focused else 1200)
    expected_rss = 5120 if candidate == "locked312-a" else 4096
    assert accounting["execution_rss_mib"] == expected_rss
    assert accounting["standard_part"] == (None if extra else "a")
    assert "<=9600s (160 job-minutes)" in accounting["campaign_budgets"]
    assert "<=104857600 bytes" in accounting["campaign_budgets"]
    if focused:
        assert "Four focused EXTRA spans <=2100s each; separate aggregate" in (
            accounting["campaign_budgets"]
        )
        assert accounting["extra_acceptance_rss_mib"] == 4096
        assert "standard_acceptance_rss_mib" not in accounting
    elif extra:
        assert (
            "Complete minimum312 support EXTRA span <=1200s"
            in accounting["campaign_budgets"]
        )
        assert accounting["extra_acceptance_rss_mib"] == 4096
        assert "standard_acceptance_rss_mib" not in accounting
    else:
        assert (
            "twelve STANDARD compute spans, minimum311, shared acceptance resolver and four fixed joins <=1200s each"
            in accounting["campaign_budgets"]
        )
        assert accounting["standard_acceptance_rss_mib"] == expected_rss
        assert "extra_acceptance_rss_mib" not in accounting


def test_fixed_join_requires_both_complete_parts_and_original_authority_paths():
    job = _workflow()["jobs"]["candidate-standard-locked312-join"]
    assert (
        job["needs"] == ["acceptance-context", "candidate-standard"]
        and "always()" in job["if"]
    )
    assert job["timeout-minutes"] == 20
    assert job["strategy"]["fail-fast"] is False
    assert job["strategy"]["matrix"]["include"] == [
        {
            "id": "locked" + minor,
            "python": "3." + minor[1:],
            "cohort": "py" + minor + "-owners",
        }
        for minor in ("311", "312", "313", "314")
    ]
    assert job["name"] == "candidate STANDARD ${{ matrix.id }} complete union"
    assert job["env"]["CANDIDATE_STANDARD_ID"] == "${{ matrix.id }}"
    downloads = [
        step for step in job["steps"] if "download-artifact@" in step.get("uses", "")
    ]
    assert len(downloads) == 2
    for part, step in zip(("a", "b"), downloads):
        assert step["uses"] == (
            "actions/download-artifact@37930b1c2abaa49bbe596cd826c3c89aef350131"
        )
        assert step["if"] == "always()"
        assert step["with"]["name"] == (
            "candidate-standard-${{ matrix.id }}-" + part + "-success"
        )
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
    assert '"$RUNNER_TEMP/ci-candidate-${CANDIDATE_STANDARD_ID}-a/run"' in work
    assert '"$RUNNER_TEMP/ci-candidate-${CANDIDATE_STANDARD_ID}-b/run"' in work
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
            "CANDIDATE_STANDARD_ID": "locked312",
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


@pytest.mark.parametrize("minor", ["311", "313", "314"])
def test_ordinary_fixed_union_paths_use_real_runner_temp_and_exact_cohort(
    tmp_path, minor
):
    job = _workflow()["jobs"]["candidate-standard-locked312-join"]
    step = job["steps"][0]
    actual_temporary = tmp_path / "actual ordinary runner temporary with spaces"
    actual_temporary.mkdir()
    environment_file = tmp_path / "GITHUB_ENV"
    environment_file.touch()
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", step["run"]],
        env={
            **os.environ,
            "RUNNER_TEMP": str(actual_temporary),
            "GITHUB_ENV": str(environment_file),
            "CANDIDATE_STANDARD_ID": "locked" + minor,
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    written = dict(
        line.split("=", 1) for line in environment_file.read_text().splitlines()
    )
    assert written == {
        "CANDIDATE_DIR": str(
            actual_temporary / ("ci-candidate-locked" + minor + "-join")
        ),
        "CANDIDATE_TOOLS": str(
            actual_temporary / ("ci-candidate-locked" + minor + "-join-tools")
        ),
        "CANDIDATE_VENV": str(
            actual_temporary / ("ci-candidate-locked" + minor + "-join-venv")
        ),
        "CANDIDATE_TMP": str(
            actual_temporary / ("ci-candidate-locked" + minor + "-join-tmp")
        ),
        "CANDIDATE_DOWNLOADS": str(actual_temporary / "ci-standard-part-downloads"),
    }
    assert not list(actual_temporary.iterdir())
    work = next(s["run"] for s in job["steps"] if s.get("id") == "join")
    assert 'if [[ "$CANDIDATE_STANDARD_ID" == locked312 ]]' in work
    assert 'if os.environ["CANDIDATE_STANDARD_ID"] == "locked312"' in work
    assert (
        'name = "ci-candidate-" + os.environ["CANDIDATE_STANDARD_ID"] + "-" + part'
        in work
    )
    _assert_early_env_has_no_runner_context(_workflow())


def _workflow_python(body, marker):
    opening = "python -B - <<'" + marker + "'\n"
    assert body.count(opening) == 1
    return body.split(opening, 1)[1].split("\n" + marker, 1)[0]


def _acceptance_repository(tmp_path, changed, message, *, old=False):
    repo = tmp_path / "checkout"
    repo.mkdir()

    environment = dict(os.environ)
    if old:
        environment.update(
            GIT_AUTHOR_DATE="2000-01-01T00:00:00+00:00",
            GIT_COMMITTER_DATE="2000-01-01T00:00:00+00:00",
        )

    def git(*arguments):
        return (
            subprocess.check_output(
                ["git", "-C", str(repo), *arguments], env=environment
            )
            .decode()
            .strip()
        )

    git("init", "-q")
    (repo / "README.md").write_text("original\n")
    git("add", ".")
    identity = (
        "-c",
        "commit.gpgSign=false",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "user.name=CI",
        "-c",
        "user.email=ci@example.invalid",
    )
    git(*identity, "commit", "-q", "-m", "original")
    before = git("rev-parse", "HEAD")
    target = repo / changed
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("changed\n")
    git("add", ".")
    git(*identity, "commit", "-q", "-m", message)
    return repo, before, git("rev-parse", "HEAD"), git


@pytest.mark.parametrize(
    "case,changed,event_name,action,requested,full",
    [
        (
            "routine-ci",
            ".github/workflows/ci.yml",
            "pull_request",
            "synchronize",
            "",
            False,
        ),
        (
            "routine-control",
            "tests/extra_tests/framework/test_ci_download_cache.py",
            "pull_request",
            "synchronize",
            "",
            False,
        ),
        ("routine-docs", "docs/guide.md", "pull_request", "synchronize", "", False),
        (
            "production",
            "jax2onnx/plugins/primitive.py",
            "pull_request",
            "synchronize",
            "",
            True,
        ),
        (
            "science",
            "tests/extra_tests/converter/test_reference.py",
            "pull_request",
            "synchronize",
            "",
            True,
        ),
        ("lock", "poetry.lock", "pull_request", "synchronize", "", True),
        (
            "cohort",
            "internal_docs/precision/ci_standard/py311-extra.json",
            "pull_request",
            "synchronize",
            "",
            True,
        ),
        (
            "unknown",
            "new_acceptance_input.bin",
            "pull_request",
            "synchronize",
            "",
            True,
        ),
        ("marker", "docs/guide.md", "pull_request", "synchronize", "", True),
        ("opened", "docs/guide.md", "pull_request", "opened", "", True),
        (
            "unavailable-before",
            "docs/guide.md",
            "pull_request",
            "synchronize",
            "",
            True,
        ),
        ("wrong-after", "docs/guide.md", "pull_request", "synchronize", "", True),
        ("push", "docs/guide.md", "push", "", "", True),
        ("manual-full", "docs/guide.md", "workflow_dispatch", "", "full", True),
        ("manual-routine", "docs/guide.md", "workflow_dispatch", "", "routine", False),
    ],
)
def test_acceptance_mode_uses_real_commit_transition_and_fail_closed_full_fallback(
    tmp_path, case, changed, event_name, action, requested, full
):
    message = (
        "reviewed change\n\nPrecision-Acceptance: full"
        if case == "marker"
        else "reviewed change"
    )
    repo, before, head, git = _acceptance_repository(tmp_path, changed, message)
    event = tmp_path / "event.json"
    event.write_text(
        json.dumps(
            {
                "action": action,
                "before": "f" * 40 if case == "unavailable-before" else before,
                "after": "0" * 40 if case == "wrong-after" else head,
            }
        )
    )
    output = tmp_path / "outputs"
    step = _workflow()["jobs"]["acceptance-context"]["steps"][1]
    result = subprocess.run(
        [sys.executable, "-B", "-c", _workflow_python(step["run"], "PY_MODE")],
        cwd=repo,
        env={
            **os.environ,
            "EXPECTED_CHECKOUT_SHA": head,
            "GITHUB_EVENT_PATH": str(event),
            "GITHUB_EVENT_NAME": event_name,
            "REQUESTED_ACCEPTANCE_MODE": requested,
            "GITHUB_OUTPUT": str(output),
            "PRECISION_ROUTINE_ACTIVATED": "false",
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert output.read_text().splitlines() == [
        "head_sha=" + head,
        "full_acceptance=" + str(full).lower(),
        "routine_activated=false",
    ]
    assert git("rev-parse", "HEAD") == head and git("status", "--porcelain") == ""


@pytest.mark.parametrize(
    "fault", ["checkout", "malformed-sha", "manual-mode", "activation-mode"]
)
def test_acceptance_mode_rejects_wrong_authority_before_output(tmp_path, fault):
    repo, before, head, _ = _acceptance_repository(tmp_path, "docs/guide.md", "change")
    event = tmp_path / "event.json"
    event.write_text(
        json.dumps({"action": "synchronize", "before": before, "after": head})
    )
    output = tmp_path / "outputs"
    step = _workflow()["jobs"]["acceptance-context"]["steps"][1]
    selected = (
        "not-a-commit"
        if fault == "malformed-sha"
        else "0" * 40 if fault == "checkout" else head
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", _workflow_python(step["run"], "PY_MODE")],
        cwd=repo,
        env={
            **os.environ,
            "EXPECTED_CHECKOUT_SHA": selected,
            "GITHUB_EVENT_PATH": str(event),
            "GITHUB_EVENT_NAME": "workflow_dispatch",
            "REQUESTED_ACCEPTANCE_MODE": (
                "unreviewed" if fault == "manual-mode" else "full"
            ),
            "GITHUB_OUTPUT": str(output),
            "PRECISION_ROUTINE_ACTIVATED": (
                "unknown" if fault == "activation-mode" else "false"
            ),
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0 and not output.exists()


def test_exhaustive_and_routine_owners_share_exact_head_without_focused_replay():
    workflow = _workflow()
    jobs = workflow["jobs"]
    resolver = jobs["acceptance-context"]
    resolver_work = resolver["steps"][1]["run"]
    assert (
        "--timeout-seconds 240 --max-rss-mib 4096 --min-available-mib 2048"
        in resolver_work
    )
    assert "--min-disk-free-mib 4096" in resolver_work
    upload = resolver["steps"][2]
    assert upload["if"] == "always()" and upload["with"]["name"].startswith(
        "candidate-standard-acceptance-context-"
    )
    assert resolver["outputs"] == {
        "head_sha": "${{ steps.mode.outputs.head_sha }}",
        "full_acceptance": "${{ steps.mode.outputs.full_acceptance }}",
        "routine_activated": "${{ steps.mode.outputs.routine_activated }}",
    }
    full = "needs.acceptance-context.outputs.full_acceptance == 'true' || needs.acceptance-context.outputs.routine_activated != 'true'"
    assert workflow["env"]["PRECISION_ROUTINE_ACTIVATED"] == "false"
    for name in ("test", "normalization-accuracy", "onnxruntime-web-smoke"):
        assert jobs[name]["needs"] == "acceptance-context" and jobs[name]["if"] == full
    assert jobs["test"]["strategy"]["matrix"]["python-version"] == [
        "3.11",
        "3.12",
        "3.13",
        "3.14",
    ]
    assert jobs["test"]["env"]["JAX2ONNX_TEST_CI_LANE_XDIST"] == "1"
    assert jobs["test"]["env"]["JAX2ONNX_RUN_FIXED_PARTS_SMOKE"] == "1"
    original_full = [
        step
        for step in jobs["test"]["steps"]
        if step.get("name", "").startswith("Run full checks")
    ]
    assert len(original_full) == 2 and all(
        step["run"] == "./scripts/run_all_checks.sh" for step in original_full
    )
    assert original_full[0]["env"] == {
        "JAX2ONNX_RUN_MAXTEXT": "1",
        "JAX2ONNX_RUN_MAXDIFFUSION": "1",
    }
    for name, job in jobs.items():
        if name == "acceptance-context":
            continue
        for step in job.get("steps", []):
            if step.get("uses", "").startswith("actions/checkout@"):
                assert (
                    step["with"]["ref"]
                    == "${{ needs.acceptance-context.outputs.head_sha }}"
                )
    active = "matrix.scope != 'focused' || needs.acceptance-context.outputs.full_acceptance != 'true' || needs.acceptance-context.outputs.routine_activated != 'true'"
    assert all(active in step["if"] for step in jobs["candidate-standard"]["steps"])
    assert jobs["exhaustive-web"]["uses"] == "./.github/workflows/nightly-qs.yml"
    assert jobs["exhaustive-web"]["if"] == full
    assert jobs["exhaustive-web"]["with"] == {
        "head_sha": "${{ needs.acceptance-context.outputs.head_sha }}",
        "full_acceptance": True,
    }
    assert "concurrency" not in workflow


def test_minimum312_extra_uses_complete_support_authority_and_ordinary_limits():
    job = _workflow()["jobs"]["candidate-standard"]
    entry = next(
        row
        for row in job["strategy"]["matrix"]["include"]
        if row["id"] == "minimum312-extra"
    )
    assert entry == {
        "id": "minimum312-extra",
        "python": "3.12",
        "cohort": "minimum312",
        "role": "minimum",
        "lane": "extra",
        "scope": "all",
    }
    profile = json.loads(
        (ROOT / "internal_docs/precision/ci_standard/minimum312.json").read_text()
    )
    assert profile["support_role"] == "minimum" and profile["allowed_skips"] == []
    assert len(profile["collection_args"]) == 17 and "focused_extra_ids" not in profile
    work = next(step["run"] for step in job["steps"] if step.get("id") == "work")
    prelude = work.split("python -B scripts/run_with_resources.py", 1)[0]
    result = subprocess.run(
        [
            "bash",
            "-euo",
            "pipefail",
            "-c",
            prelude
            + '\nprintf "%s\\n" "$execution_work_seconds" "$execution_rss_mib" "$live_available_mib" "${#execution_profile_args[@]}" "${execution_profile_args[@]}"',
        ],
        env={
            **os.environ,
            "CANDIDATE_ID": entry["id"],
            "CANDIDATE_ROLE": entry["role"],
            "CANDIDATE_LANE": "extra",
            "CANDIDATE_SCOPE": "all",
            "CANDIDATE_PART": "",
            "CANDIDATE_COHORT": "internal_docs/precision/ci_standard/minimum312.json",
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0 and result.stdout.splitlines() == [
        "1100",
        "4096",
        "2048",
        "0",
    ]
    assert (
        "--focused-extra-profile" not in result.stdout
        and "--locked312-standard-profile" not in result.stdout
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("CANDIDATE_ID", "foreign-extra"),
        ("CANDIDATE_LANE", "standard"),
        ("CANDIDATE_ROLE", "locked"),
        ("CANDIDATE_SCOPE", "focused"),
        ("CANDIDATE_PART", "a"),
        ("CANDIDATE_COHORT", "internal_docs/precision/ci_standard/py312-extra.json"),
    ],
)
def test_complete_support_extra_rejects_foreign_dispatch_before_runner(key, value):
    work = next(
        step["run"]
        for step in _workflow()["jobs"]["candidate-standard"]["steps"]
        if step.get("id") == "work"
    )
    dispatch = (
        "part_args=()"
        + work.split("part_args=()", 1)[1].split(
            "python -B scripts/run_pytest_batches.py", 1
        )[0]
    )
    environment = {
        **os.environ,
        "CANDIDATE_ID": "minimum312-extra",
        "CANDIDATE_ROLE": "minimum",
        "CANDIDATE_LANE": "extra",
        "CANDIDATE_SCOPE": "all",
        "CANDIDATE_PART": "",
        "CANDIDATE_COHORT": "internal_docs/precision/ci_standard/minimum312.json",
    }
    environment[key] = value
    result = subprocess.run(
        [
            "bash",
            "-euo",
            "pipefail",
            "-c",
            dispatch + '\nprintf "unexpected-runner\\n"',
        ],
        env=environment,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0 and "unexpected-runner" not in result.stdout


@pytest.mark.parametrize("failure", ["none", "minimum-install", "pip-check", "pytest"])
def test_minimum311_monitors_original_seventeen_selectors_and_stops_on_failure(
    tmp_path, failure
):
    job = _workflow()["jobs"]["minimum-dependencies"]
    assert job["timeout-minutes"] == 20 and job["needs"] == "acceptance-context"
    step = next(
        step
        for step in job["steps"]
        if step.get("name") == "Observe minimum-version setup and compatibility checks"
    )
    commands = r"""
    python() {
      printf 'observer'; printf ' %q' "$@"; printf '\n'
      while [[ "$1" != -- ]]; do shift; done
      shift
      "$@"
    }
    pip() { printf 'pip'; printf ' %q' "$@"; printf '\n'; }
    poetry() {
      printf 'poetry' >> "$EVENTS"; printf ' %q' "$@" >> "$EVENTS"; printf '\n' >> "$EVENTS"
      case "$*" in
        'run python -') cat > /dev/null ;;
        'run python scripts/install_minimum_dependencies.py') [[ "$FAIL_STAGE" != minimum-install ]] || return 71 ;;
        'run pip check') [[ "$FAIL_STAGE" != pip-check ]] || return 72 ;;
        'run pytest '*) [[ "$JAX_PLATFORMS" == cpu && "$JAX_ENABLE_X64" == True ]]; [[ "$FAIL_STAGE" != pytest ]] || return 73 ;;
      esac
    }
    export -f python pip poetry
    """
    events = tmp_path / "events"
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", commands + "\n" + step["run"]],
        env={
            **os.environ,
            "RUNNER_TEMP": str(tmp_path),
            "GITHUB_WORKSPACE": str(tmp_path),
            "EXPECTED_CHECKOUT_SHA": "a" * 40,
            "EVENTS": str(events),
            "FAIL_STAGE": failure,
        },
        text=True,
        capture_output=True,
    )
    lines = [shlex.split(line) for line in events.read_text().splitlines()]
    assert lines[:2] == [
        ["poetry", "install", "--no-interaction"],
        ["poetry", "run", "python", "scripts/install_minimum_dependencies.py"],
    ]
    assert (
        "--timeout-seconds 1100 --max-rss-mib 4096 --min-available-mib 2048"
        in result.stdout
    )
    assert "--min-disk-free-mib 4096" in result.stdout
    expected = {"none": 0, "minimum-install": 71, "pip-check": 72, "pytest": 73}[
        failure
    ]
    assert result.returncode == expected, result.stdout + result.stderr
    generation = [
        line
        for line in lines
        if line == ["poetry", "run", "python", "scripts/generate_tests.py"]
    ]
    selected = [line for line in lines if line[:3] == ["poetry", "run", "pytest"]]
    if failure in {"minimum-install", "pip-check"}:
        assert generation == selected == []
    else:
        assert len(generation) == len(selected) == 1
        profile = json.loads(
            (ROOT / "internal_docs/precision/ci_standard/minimum312.json").read_text()
        )
        assert selected[0] == [
            "poetry",
            "run",
            "pytest",
            "-q",
            "--junitxml",
            str(tmp_path / "minimum311-junit.xml"),
            *profile["collection_args"],
        ]
        assert lines.index(generation[0]) < lines.index(selected[0])
    receipts = next(
        step
        for step in job["steps"]
        if step.get("name") == "Upload minimum311 STANDARD receipts"
    )
    assert receipts["if"] == "always()" and receipts["with"]["name"].startswith(
        "candidate-standard-minimum311-"
    )
    failure_payload = next(
        step
        for step in job["steps"]
        if step.get("name", "").startswith("Preserve complete minimum311")
    )
    assert (
        failure_payload["if"] == "failure()"
        and "minimum311-tmp/" in failure_payload["with"]["path"]
    )


def _nightly_workflow():
    return yaml.safe_load((ROOT / ".github/workflows/nightly-qs.yml").read_text())


@pytest.mark.parametrize(
    "case", ["open", "closed", "explicit", "foreign", "malformed", "unavailable"]
)
def test_browser_head_resolver_uses_one_exact_active_commit_without_token_output(
    tmp_path, monkeypatch, capsys, case
):
    import io
    import urllib.request

    workflow = _nightly_workflow()
    step = workflow["jobs"]["check-changes"]["steps"][0]
    script = _workflow_python(step["run"], "PY_HEAD")
    calls = []
    head, main = "a" * 40, "b" * 40
    output = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("GITHUB_REPOSITORY", "enpasos/jax2onnx")
    monkeypatch.setenv("GH_TOKEN", "synthetic-secret-not-logged")
    monkeypatch.setenv("REQUESTED_HEAD_SHA", head if case == "explicit" else "")

    def response(request, timeout):
        assert (
            timeout == 30
            and request.get_header("Authorization")
            == "Bearer synthetic-secret-not-logged"
        )
        calls.append(request.full_url)
        if case == "unavailable":
            raise OSError("synthetic API failure")
        if request.full_url.endswith("/pulls/277"):
            value = {
                "state": "closed" if case == "closed" else "open",
                "base": {"ref": "main"},
                "head": {
                    "sha": "not-an-exact-sha" if case == "malformed" else head,
                    "repo": {
                        "full_name": (
                            "foreign/fork" if case == "foreign" else "enpasos/jax2onnx"
                        )
                    },
                },
            }
        else:
            assert request.full_url.endswith("/commits/main")
            value = {"sha": main}
        return io.BytesIO(json.dumps(value).encode())

    monkeypatch.setattr(urllib.request, "urlopen", response)
    if case in {"foreign", "malformed", "unavailable"}:
        with pytest.raises((SystemExit, OSError)):
            exec(compile(script, "actual-nightly-head-resolver", "exec"), {})
        assert not output.exists()
    else:
        exec(compile(script, "actual-nightly-head-resolver", "exec"), {})
        assert output.read_text().splitlines() == [
            "head_sha=" + (main if case == "closed" else head)
        ]
    expected = (
        []
        if case == "explicit"
        else ["https://api.github.com/repos/enpasos/jax2onnx/pulls/277"]
    )
    if case == "closed":
        expected.append("https://api.github.com/repos/enpasos/jax2onnx/commits/main")
    assert calls == expected
    assert "synthetic-secret-not-logged" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "full,wrong_head", [(True, False), (False, False), (True, True)]
)
def test_browser_full_acceptance_cannot_age_skip_or_drift_from_selected_head(
    tmp_path, full, wrong_head
):
    repo, _, head, _ = _acceptance_repository(
        tmp_path, "docs/guide.md", "old acceptance input", old=True
    )
    # Both actual commits predate the age window; no fake git-log output.
    output = tmp_path / "outputs"
    workflow = _nightly_workflow()
    check = next(
        step
        for step in workflow["jobs"]["check-changes"]["steps"]
        if step.get("id") == "check"
    )
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", check["run"]],
        cwd=repo,
        env={
            **os.environ,
            "EXPECTED_CHECKOUT_SHA": "0" * 40 if wrong_head else head,
            "FULL_ACCEPTANCE": str(full).lower(),
            "GITHUB_OUTPUT": str(output),
        },
        text=True,
        capture_output=True,
    )
    if wrong_head:
        assert result.returncode != 0 and not output.exists()
    else:
        assert result.returncode == 0, result.stderr
        assert output.read_text().splitlines() == ["should_run=" + str(full).lower()]
    for job, timeout, command in [
        ("onnxruntime-web-full", 180, "scripts/run_onnxruntime_web_full.sh"),
        (
            "onnxruntime-web-chrome-full",
            240,
            "scripts/run_onnxruntime_web_chrome_full.sh",
        ),
    ]:
        value = workflow["jobs"][job]
        assert value["timeout-minutes"] == timeout
        checkout = next(
            step
            for step in value["steps"]
            if step.get("uses", "").startswith("actions/checkout@")
        )
        assert checkout["with"]["ref"] == "${{ needs.check-changes.outputs.head_sha }}"
        assert any(
            step.get("run") == command + " -n auto --dist=loadscope -q --ff -x"
            for step in value["steps"]
        )
    assert workflow["concurrency"]["cancel-in-progress"] is False


@pytest.mark.parametrize(
    "activated,full", [(False, False), (False, True), (True, False), (True, True)]
)
def test_tracked_qualification_switch_preserves_every_owner_until_later_activation(
    tmp_path, activated, full
):
    workflow = _workflow()
    events = workflow.get("on", workflow.get(True))  # PyYAML's YAML1.1 spelling of on.
    assert set(events["workflow_dispatch"]["inputs"]) == {"acceptance_mode"}
    assert workflow["env"]["PRECISION_ROUTINE_ACTIVATED"] == "false"
    repo, before, head, _ = _acceptance_repository(
        tmp_path, "docs/guide.md", "reviewed transition"
    )
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"before": before, "after": head}))
    output = tmp_path / "outputs"
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            _workflow_python(
                workflow["jobs"]["acceptance-context"]["steps"][1]["run"], "PY_MODE"
            ),
        ],
        cwd=repo,
        env={
            **os.environ,
            "EXPECTED_CHECKOUT_SHA": head,
            "GITHUB_EVENT_PATH": str(event),
            "GITHUB_EVENT_NAME": "workflow_dispatch",
            "REQUESTED_ACCEPTANCE_MODE": "full" if full else "routine",
            "PRECISION_ROUTINE_ACTIVATED": str(activated).lower(),
            "GITHUB_OUTPUT": str(output),
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    actual = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert actual == {
        "head_sha": head,
        "full_acceptance": str(full).lower(),
        "routine_activated": str(activated).lower(),
    }
    # These are the actual fixed Actions predicates. Only the synthetic future
    # source value varies here; no manual workflow input can activate retirement.
    legacy = "needs.acceptance-context.outputs.full_acceptance == 'true' || needs.acceptance-context.outputs.routine_activated != 'true'"
    focused = "matrix.scope != 'focused' || needs.acceptance-context.outputs.full_acceptance != 'true' || needs.acceptance-context.outputs.routine_activated != 'true'"
    replaced = (
        legacy.replace(
            "needs.acceptance-context.outputs.full_acceptance",
            repr(actual["full_acceptance"]),
        )
        .replace(
            "needs.acceptance-context.outputs.routine_activated",
            repr(actual["routine_activated"]),
        )
        .replace("||", " or ")
    )
    legacy_runs = eval(replaced, {"__builtins__": {}}, {})
    assert legacy_runs is (full or not activated)
    for name in [
        "test",
        "normalization-accuracy",
        "onnxruntime-web-smoke",
        "exhaustive-web",
    ]:
        assert workflow["jobs"][name]["if"] == legacy
    assert "if" not in workflow["jobs"]["minimum-dependencies"]
    steps = workflow["jobs"]["candidate-standard"]["steps"]
    assert all(focused in step["if"] for step in steps)
    for scope in ["focused", "all", ""]:
        replaced = (
            focused.replace("matrix.scope", repr(scope))
            .replace(
                "needs.acceptance-context.outputs.full_acceptance",
                repr(actual["full_acceptance"]),
            )
            .replace(
                "needs.acceptance-context.outputs.routine_activated",
                repr(actual["routine_activated"]),
            )
            .replace("||", " or ")
        )
        assert eval(replaced, {"__builtins__": {}}, {}) is (
            scope != "focused" or not full or not activated
        )
