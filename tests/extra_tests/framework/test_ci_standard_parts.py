# tests/extra_tests/framework/test_ci_standard_parts.py

"""Fixed full-S parts reuse real canonical workers and the strict native join."""

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests.extra_tests.framework.test_pytest_batches import (
    _RUNNER,
    _prepare_main_owner_provision_suite,
    _runner,
)


def test_fixed_profile_preserves_original_full_s_and_cut():
    module = _runner()
    profile = json.loads(
        (
            _RUNNER.parents[1] / "internal_docs/precision/ci_standard/py312-owners.json"
        ).read_bytes()
    )
    full = profile["baseline"]["nodeids"]
    parts = module._ci_standard_parts(full, profile["standard_parts"])
    assert len(full) == 4621
    assert len(parts["a"]) == 216 and len(parts["b"]) == 4405
    assert (
        parts["a"][-1]
        == "tests/examples/test_nnx_dino.py::Test_NnxDinoPatchEmbed::test_nnx_patch_embed"
    )
    assert (
        parts["b"][0]
        == "tests/examples/test_nnx_gpt_oss.py::Test_FlaxAttentionBlock::test_gpt_oss_attention_flax"
    )
    assert parts["a"] + parts["b"] == full
    assert set(parts["a"]).isdisjoint(parts["b"])
    assert sum("test_maxtext.py::" in node for node in parts["a"]) == 45
    assert sum("test_maxdiffusion.py::" in node for node in parts["a"]) == 5


@pytest.mark.parametrize(
    "declaration",
    [
        None,
        {},
        {"split_after": "missing"},
        {"split_after": "b"},
        {"split_after": "a", "dynamic": True},
    ],
)
def test_fixed_parts_reject_undeclared_empty_or_dynamic_boundary(declaration):
    with pytest.raises(ValueError, match="STANDARD parts"):
        _runner()._ci_standard_parts(["a", "b"], declaration)


def _observations(head, run_id="123", attempt="1"):
    now = datetime.now(timezone.utc)
    return {
        "total_count": 2,
        "jobs": [
            {
                "id": index,
                "run_id": int(run_id),
                "run_attempt": int(attempt),
                "head_sha": head,
                "name": f"candidate STANDARD (locked312 serial part {part} (5GiB))",
                "status": "completed",
                "conclusion": "success",
                "started_at": (now - timedelta(seconds=60)).isoformat(),
                "completed_at": now.isoformat(),
            }
            for index, part in enumerate(("a", "b"), 1)
        ],
    }


@pytest.mark.parametrize(
    "fault",
    [
        "head",
        "attempt",
        "missing",
        "duplicate",
        "cancelled",
        "whole-deadline",
        "zero-id",
        "same-id",
        "truncated",
    ],
)
def test_join_requires_complete_same_head_attempt_successful_whole_jobs(fault):
    head = "a" * 40
    observations = _observations(head)
    job = observations["jobs"][0]
    if fault == "head":
        job["head_sha"] = "b" * 40
    elif fault == "attempt":
        job["run_attempt"] = 2
    elif fault == "missing":
        observations["jobs"].pop()
        observations["total_count"] = len(observations["jobs"])
    elif fault == "truncated":
        observations["jobs"].pop()
    elif fault == "duplicate":
        observations["jobs"][1] = dict(job)
    elif fault == "cancelled":
        job["conclusion"] = "cancelled"
    elif fault == "zero-id":
        job["id"] = 0
    elif fault == "same-id":
        observations["jobs"][1]["id"] = job["id"]
    else:
        job["started_at"] = (
            datetime.fromisoformat(job["completed_at"]) - timedelta(seconds=1201)
        ).isoformat()
    message = (
        "Actions part job is missing or duplicated"
        if fault == "missing"
        else "complete current-attempt" if fault == "truncated" else "Actions"
    )
    with pytest.raises(ValueError, match=message):
        _runner()._ci_standard_job_times(
            observations, {"checkout": head, "run_id": "123", "run_attempt": "1"}
        )


def test_join_retains_actual_whole_job_times():
    head = "a" * 40
    times = _runner()._ci_standard_job_times(
        _observations(head), {"checkout": head, "run_id": "123", "run_attempt": "1"}
    )
    assert list(times) == ["a", "b"]
    assert times["a"]["elapsed_seconds"] == times["b"]["elapsed_seconds"] == 60


def _tiny_parts(tmp_path, monkeypatch):
    module = _runner()
    suite, template, environment, cohort, selectors = (
        _prepare_main_owner_provision_suite(tmp_path, monkeypatch)
    )
    # A real clone supplies the actual existing parent identity without inventing
    # a commit or pretending that these harmless fixture files are hosted Main.
    clone = tmp_path / "git-context"
    subprocess.run(
        [
            "git",
            "clone",
            "--quiet",
            "--shared",
            "--no-checkout",
            str(_RUNNER.parents[1]),
            str(clone),
        ],
        check=True,
    )
    head = subprocess.check_output(
        ["git", "--no-optional-locks", "rev-parse", "HEAD"],
        cwd=_RUNNER.parents[1],
        text=True,
    ).strip()
    subprocess.run(
        ["git", "update-ref", "--no-deref", "HEAD", head], cwd=clone, check=True
    )
    (clone / ".git").rename(suite / ".git")
    case = suite / "test_cases.py"
    with case.open("a") as stream:
        stream.write(
            "\ndef test_failure_witness():\n    import os\n    assert os.environ.get('FIXED_PART_FAIL') != '1'\n\ndef test_later_witness():\n    Path('later-witness-ran').touch()\n"
        )
    payload = json.loads(template.read_bytes())
    payload["nodeids"].extend(
        ["test_cases.py::test_failure_witness", "test_cases.py::test_later_witness"]
    )
    payload["source_files"][case.name] = hashlib.sha256(case.read_bytes()).hexdigest()
    data = json.loads(cohort.read_bytes())
    data["baseline"]["nodeids"] = payload["nodeids"]
    data["standard_parts"] = {"split_after": payload["nodeids"][0]}
    module._write(cohort, data)
    payload["source_files"][cohort.name] = hashlib.sha256(
        cohort.read_bytes()
    ).hexdigest()
    environment.update(
        GITHUB_RUN_ID="123", GITHUB_RUN_ATTEMPT="1", GIT_OPTIONAL_LOCKS="0"
    )
    base = tmp_path / "parts"
    base.mkdir()
    return module, suite, payload, environment, cohort, selectors, head, base


def _run_tiny_part(fixture, part, *, fail=False):
    module, suite, template, environment, cohort, selectors, head, base = fixture
    parent = base / (part + ("-incomplete" if fail else ""))
    parent.mkdir()
    run = parent / "run"
    manifest = parent / "baseline.json"
    payload = json.loads(json.dumps(template))
    native = str(run / "native-model-artifacts")
    for mapping in (
        payload["environment"],
        payload["collection_transition"]["collected_environment"],
        payload["collection_transition"]["post_pytest_environment"],
    ):
        mapping["JAX2ONNX_NATIVE_BINDINGS_EVIDENCE_DIR"] = native
    profile_sha = hashlib.sha256(cohort.read_bytes()).hexdigest()
    assignment = module._ci_standard_parts(
        payload["nodeids"], {"split_after": payload["nodeids"][0]}
    )[part]
    payload["standard_part"] = {
        "id": part,
        "profile": cohort.name,
        "profile_sha256": profile_sha,
        "split_after": payload["nodeids"][0],
        "full_s_sha256": module._ci_standard_ids_digest(payload["nodeids"]),
        "assigned_sha256": module._ci_standard_ids_digest(assignment),
        "checkout": head,
        "run_id": "123",
        "run_attempt": "1",
    }
    module._write(manifest, payload)
    arguments = [
        sys.executable,
        "-B",
        str(_RUNNER.with_name("run_with_resources.py")),
        "--output-dir",
        str(parent) + "-work-resources",
        "--timeout-seconds",
        "90",
        "--max-rss-mib",
        "4096",
        "--min-available-mib",
        "8192",
        "--disk-path",
        str(suite),
        "--disk-path",
        str(base),
        "--",
        sys.executable,
        "-B",
        str(_RUNNER),
        "--ci-standard-part",
        part,
        "--ci-lane",
        "standard",
        "--ci-main-baseline",
        str(manifest),
        "--_ci-provision-campaign",
        str(cohort),
        "--output-dir",
        str(run),
        "--batch-size",
        "1",
    ]
    for selector in selectors:
        arguments.extend(("--test-path", selector))
    arguments.extend(("--", "-q"))
    actual = subprocess.run(
        arguments,
        cwd=suite,
        env={**environment, "FIXED_PART_FAIL": "1" if fail else "0"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    (parent / "actual-cli-output.log").write_text(actual.stdout + actual.stderr)
    return actual, run


@pytest.mark.skipif(
    sys.version_info[:2] != (3, 12)
    or os.environ.get("JAX2ONNX_RUN_FIXED_PARTS_SMOKE") != "1",
    reason="explicit official312 real serial-parts integration control",
)
def test_real_312_two_serial_main_owner_parts_join_and_fail_closed(
    tmp_path, monkeypatch
):
    fixture = _tiny_parts(tmp_path, monkeypatch)
    module, suite, payload, environment, cohort, selectors, head, base = fixture
    a, root_a = _run_tiny_part(fixture, "a")
    assert a.returncode == 0, a.stdout + a.stderr
    b, root_b = _run_tiny_part(fixture, "b")
    assert b.returncode == 0, b.stdout + b.stderr
    monkeypatch.chdir(suite)
    monkeypatch.setattr(module.os, "environ", environment)
    observations = base / "unit-control-job-observations.json"
    # These are explicitly synthetic Actions observations around REAL resource
    # intervals, testing association only; they are never hosted timing proof.
    unit_jobs = _observations(head)
    for job, root in zip(unit_jobs["jobs"], (root_a, root_b), strict=True):
        resource = json.loads(
            (Path(str(root.parent) + "-work-resources") / "summary.json").read_bytes()
        )
        start = datetime.fromisoformat(resource["started_utc"])
        job["started_at"] = start.isoformat()
        job["completed_at"] = (
            start + timedelta(seconds=resource["elapsed_seconds"])
        ).isoformat()
    module._write(observations, unit_jobs)

    def join(roots, name):
        output = base / name
        code = module._ci_join_standard_parts(
            roots, cohort, output, job_observations=observations
        )
        return code, json.loads((output / "run.json").read_bytes())

    code, result = join([root_a, root_b], "join")
    assert code == 0, result
    assert result["completed_ids"] == payload["nodeids"]
    assert result["numerical_acceptance"] == "NOT_ASSERTED"
    assert result["full_B_to_F_acceptance"] == "UNVERIFIED"
    for root, part in ((root_a, "a"), (root_b, "b")):
        run = json.loads((root / "run.json").read_bytes())
        assert run["standard_part_acceptance"] == "PART_ONLY_JOIN_REQUIRED"
        assert run["standard_part"]["id"] == part
        owner = json.loads((root / "collection/receipt.json").read_bytes())
        for folder in root.glob("batch-*"):
            receipt = json.loads((folder / "receipt.json").read_bytes())
            assert receipt["full_preselection_ids"] == owner["full_preselection_ids"]
            assert receipt["full_collection_checked"] is True
            assert receipt["collection_transition_completed"] is True
            assert receipt["post_pytest_completed"] is True
    assert (
        not (suite / "extra-ran").exists()
        and not (suite / "outside-extra-ran").exists()
    )
    for roots, name in (
        ([root_a, root_a], "duplicate"),
        ([root_b, root_a], "relabel"),
        ([root_a, base / "missing"], "missing"),
    ):
        assert join(roots, name)[0] == 1
    original = (root_b / "run.json").read_bytes()
    for key, value in (("checkout", "f" * 40), ("run_attempt", "2")):
        run = json.loads(original)
        run["standard_part"][key] = value
        module._write(root_b / "run.json", run)
        assert join([root_a, root_b], "wrong-" + key)[0] == 1
        (root_b / "run.json").write_bytes(original)
    receipt = root_b / "batch-0003/receipt.json"
    original_receipt = receipt.read_bytes()
    receipt.write_bytes(original_receipt + b"\n")
    assert join([root_a, root_b], "tamper")[0] == 1
    receipt.write_bytes(original_receipt)
    # Semantic rejection must survive coherent receipt/terminal/run-hash
    # rebinding: a digest check alone is insufficient native qualification.
    terminal = receipt.with_name("lane-terminal.json")
    originals = {
        path: path.read_bytes() for path in (receipt, terminal, root_b / "run.json")
    }
    for fault, message in (
        ("wasxfail", "wasxfail"),
        ("worker-source", "controller source snapshots"),
    ):
        document = json.loads(originals[receipt])
        if fault == "wasxfail":
            document["phases"]["test_cases.py::test_failure_witness"][1][
                "wasxfail"
            ] = "foreign"
        else:
            document["startup_context"]["source_files"][
                "scripts/run_pytest_batches.py"
            ]["sha256"] = ("0" * 64)
        module._write(receipt, document)
        closed = json.loads(originals[terminal])
        closed["receipt_sha256"] = hashlib.sha256(receipt.read_bytes()).hexdigest()
        module._write(terminal, closed)
        run = json.loads(originals[root_b / "run.json"])
        for path in (receipt, terminal):
            run["ci_lane"]["closed_workers"]["batch-0003"][path.name] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        module._write(root_b / "run.json", run)
        rejected, report = join([root_a, root_b], fault)
        assert rejected == 1 and message in report["failure"]["message"], report
        for path, raw in originals.items():
            path.write_bytes(raw)
    resource_path = Path(str(root_b.parent) + "-work-resources") / "summary.json"
    raw_resource = resource_path.read_bytes()
    resource = json.loads(raw_resource)
    resource["cleanup_confirmed"] = False
    module._write(resource_path, resource)
    rejected, report = join([root_a, root_b], "resource-cleanup-incomplete")
    assert (
        rejected == 1
        and report["failure"]["message"]
        == "STANDARD part resource/cleanup/capacity closure is incomplete"
    ), report
    resource_path.write_bytes(raw_resource)
    resource = json.loads(raw_resource)
    worker_pid = json.loads((root_b / "run.json").read_bytes())["processes"][-1]["pid"]
    resource["owned_pid_start_ledger"] = [
        entry for entry in resource["owned_pid_start_ledger"] if entry[0] != worker_pid
    ]
    module._write(resource_path, resource)
    rejected, report = join([root_a, root_b], "missing-worker-pid")
    assert rejected == 1 and "RUN/workers" in report["failure"]["message"], report
    resource_path.write_bytes(raw_resource)
    shifted = json.loads(json.dumps(unit_jobs))
    for field in ("started_at", "completed_at"):
        shifted["jobs"][1][field] = (
            datetime.fromisoformat(shifted["jobs"][1][field]) + timedelta(hours=1)
        ).isoformat()
    module._write(observations, shifted)
    rejected, report = join([root_a, root_b], "foreign-job-resource-interval")
    assert (
        rejected == 1 and "actual Actions part job" in report["failure"]["message"]
    ), report
    module._write(observations, unit_jobs)
    manifest = root_b.parent / "baseline.json"
    original_manifest = manifest.read_bytes()
    document = json.loads(original_manifest)
    document["collection_transition"]["collected_environment"][
        "JAX_PLATFORMS"
    ] = "different"
    module._write(manifest, document)
    run = json.loads(original)
    run["ci_lane"]["manifest_sha256"] = hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()
    module._write(root_b / "run.json", run)
    rejected, report = join([root_a, root_b], "profile-transition-divergence")
    assert (
        rejected == 1
        and report["failure"]["message"]
        == "provisioned context differs from reviewed template"
    ), report
    manifest.write_bytes(original_manifest)
    (root_b / "run.json").write_bytes(original)
    (suite / "later-witness-ran").unlink()
    failure, incomplete = _run_tiny_part(fixture, "b", fail=True)
    assert failure.returncode != 0
    failed = json.loads((incomplete / "run.json").read_bytes())
    assert failed["status"] == "FAIL"
    assert failed["part_batch_indexes"] == [2, 3, 4]
    assert (incomplete / "batch-0003/receipt.json").is_file()
    assert not (incomplete / "batch-0004").exists()
    assert not (suite / "later-witness-ran").exists()
    assert join([root_a, incomplete], "incomplete")[0] == 1


@pytest.mark.parametrize("batch_size", [1, 499])
def test_actual_cli_rejects_hosted_part_repacking_before_any_run(tmp_path, batch_size):
    manifest = tmp_path / "baseline.json"
    manifest.write_text(
        json.dumps(
            {
                "standard_part": {
                    "profile": "internal_docs/precision/ci_standard/py312-owners.json"
                }
            }
        )
    )
    output = tmp_path / "never-started"
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(_RUNNER),
            "--ci-lane",
            "standard",
            "--ci-main-baseline",
            str(manifest),
            "--_ci-provision-campaign",
            str(tmp_path / "unused-profile.json"),
            "--ci-standard-part",
            "a",
            "--batch-size",
            str(batch_size),
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 2
    assert "original 500-case batching" in result.stderr
    assert not output.exists()


def test_worker_source_snapshot_uses_actual_script_root_and_one_read(
    tmp_path, monkeypatch
):
    module = _runner()
    empty = tmp_path / "unrelated-cwd"
    empty.mkdir()
    monkeypatch.chdir(empty)
    snapshot = module._worker_context()
    original = module._context_read
    calls = []

    def read(path, limit=2 * 1024**2):
        calls.append(path)
        return original(path, limit)

    monkeypatch.setattr(module, "_context_read", read)
    root, bindings = module._ci_worker_source_files()
    assert root == _RUNNER.parents[1].resolve()
    assert calls == [
        (root / name).resolve(strict=True) for name in module._WORKER_SOURCE_FILES
    ]
    assert bindings == snapshot["source_files"]
    for name, proof in bindings.items():
        payload = (root / name).read_bytes()
        assert proof == {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
