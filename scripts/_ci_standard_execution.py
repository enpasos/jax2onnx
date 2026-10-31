# scripts/_ci_standard_execution.py

"""Admission for the one reviewed hosted STANDARD execution envelope."""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


LOCKED312_PROFILE = "internal_docs/precision/ci_standard/py312-owners.json"


def locked312_execution(profile: Path) -> dict[str, str]:
    """Bind the 5 GiB envelope to current tracked hosted Python 3.12 sources."""
    root = Path.cwd().resolve()
    expected = os.environ.get("EXPECTED_CHECKOUT_SHA", "")
    if (
        os.environ.get("GITHUB_ACTIONS") != "true"
        or os.environ.get("RUNNER_OS") != "Linux"
        or os.environ.get("CANDIDATE_ID") not in {"locked312-a", "locked312-b"}
        or os.environ.get("CANDIDATE_PART") not in {"a", "b"}
        or os.environ.get("CANDIDATE_ID")
        != "locked312-" + os.environ.get("CANDIDATE_PART", "")
        or os.environ.get("CANDIDATE_ROLE") != "locked"
        or os.environ.get("CANDIDATE_COHORT") != LOCKED312_PROFILE
        or Path(os.environ.get("GITHUB_WORKSPACE", "/")).resolve() != root
        or sys.version_info[:2] != (3, 12)
        or re.fullmatch("[0-9a-f]{40}", expected) is None
        or profile.resolve(strict=True) != root / LOCKED312_PROFILE
    ):
        raise ValueError(
            "5 GiB STANDARD execution requires the hosted locked312 profile"
        )

    def git(*arguments: str) -> bytes:
        return subprocess.check_output(
            ["git", "--no-optional-locks", *arguments], cwd=root, stderr=subprocess.PIPE
        )

    if git("rev-parse", "HEAD").decode().strip() != expected:
        raise ValueError("locked312 checkout differs from the current event head")
    bindings = {}
    for relative in (
        LOCKED312_PROFILE,
        "scripts/_ci_standard_execution.py",
        "scripts/run_with_resources.py",
        "scripts/run_pytest_batches.py",
        ".github/workflows/ci.yml",
    ):
        actual = (root / relative).read_bytes()
        if actual != git("show", f"HEAD:{relative}"):
            raise ValueError("locked312 execution sources differ from the tracked head")
        bindings[relative] = hashlib.sha256(actual).hexdigest()
    data = json.loads(profile.read_bytes())
    if (
        data.get("schema") != "jax2onnx-ci-main-owner-cohort-v1"
        or data.get("python_minor") != "3.12"
        or data.get("baseline_commit") != "81ad6ca2592635c8d2ced6dbd77292c487f0351b"
    ):
        raise ValueError("locked312 execution profile has an unsupported authority")
    return {"checkout": expected, "profile": LOCKED312_PROFILE, **bindings}
