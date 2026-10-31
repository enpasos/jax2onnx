# scripts/prepare_ci_download_cache.py

"""Release hosted-job package downloads without touching installed runtimes."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from typing import Any


def _runtime_identity() -> dict[str, Any]:
    packages = []
    for distribution in distributions():
        metadata = distribution.read_text("METADATA") or ""
        packages.append(
            (
                distribution.metadata["Name"],
                distribution.version,
                str(Path(distribution.locate_file("")).resolve()),
                hashlib.sha256(metadata.encode()).hexdigest(),
            )
        )
    return {
        "executable": str(Path(sys.executable).resolve()),
        "prefix": str(Path(sys.prefix).resolve()),
        "python": sys.version,
        "packages": sorted(packages),
    }


def _usage(path: Path) -> dict[str, int]:
    logical = allocated = files = 0
    if path.exists():
        for entry in path.rglob("*"):
            info = entry.lstat()
            if stat.S_ISREG(info.st_mode):
                logical += info.st_size
                allocated += info.st_blocks * 512
                files += 1
    return {"files": files, "logical_bytes": logical, "allocated_bytes": allocated}


def _space(paths: tuple[Path, ...]) -> dict[str, dict[str, int]]:
    return {
        str(path): {
            "device": path.stat().st_dev,
            "free_bytes": shutil.disk_usage(path).free,
        }
        for path in paths
    }


def release_download_caches(
    *,
    home: Path,
    pip_cache: Path,
    poetry_cache: Path,
    protected: tuple[Path, ...],
    disk_paths: tuple[Path, ...],
    report: Path,
) -> dict[str, Any]:
    """Delete only the three validated download-cache roots, after full admission."""
    if (home / ".cache").is_symlink():
        raise ValueError("hosted user cache root is a symlink")
    owner_cache = (home / ".cache").resolve()
    keep = tuple(
        path.resolve()
        for path in (*protected, Path(sys.prefix), poetry_cache / "virtualenvs")
    )
    targets = (
        pip_cache,
        poetry_cache / "artifacts",
        poetry_cache / "cache" / "repositories",
    )
    resolved = []
    for path in targets:
        if path.is_symlink():
            raise ValueError("download cache root is a symlink")
        actual = path.resolve()
        if actual == owner_cache or not actual.is_relative_to(owner_cache):
            raise ValueError("download cache escapes the hosted user's cache")
        if any(
            actual == item or actual.is_relative_to(item) or item.is_relative_to(actual)
            for item in keep
        ):
            raise ValueError("download cache overlaps a protected runtime/output root")
        if path.exists() and (not path.is_dir() or path.stat().st_uid != os.getuid()):
            raise ValueError("download cache root is not an owned directory")
        resolved.append(actual)
    before = _runtime_identity()
    record: dict[str, Any] = {
        "status": "STARTED",
        "scope": "hosted-job package downloads only; no virtualenv/model/receipt/temp deletion",
        "runtime_before": before,
        "disk_before": _space(disk_paths),
        "caches": {str(path): _usage(path) for path in resolved},
    }
    report.write_text(json.dumps(record, indent=2) + "\n")
    for path in resolved:
        if path.exists():
            shutil.rmtree(path)
    after = _runtime_identity()
    record.update(runtime_after=after, disk_after=_space(disk_paths))
    record["observed_free_delta_bytes"] = {
        path: values["free_bytes"] - record["disk_before"][path]["free_bytes"]
        for path, values in record["disk_after"].items()
    }
    record["runtime_identity_held"] = before == after
    record["status"] = "PASS" if before == after else "FAIL"
    report.write_text(json.dumps(record, indent=2) + "\n")
    if before != after:
        raise ValueError(
            "installed runtime identity changed during download-cache release"
        )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if (
        os.environ.get("GITHUB_ACTIONS") != "true"
        or os.environ.get("RUNNER_ENVIRONMENT") != "github-hosted"
    ):
        print("Download-cache preparation is disabled outside a GitHub-hosted job.")
        return 0
    workspace = Path(os.environ["GITHUB_WORKSPACE"]).resolve()
    temporary = Path(os.environ["RUNNER_TEMP"]).resolve()
    if workspace != Path(__file__).resolve().parents[1]:
        raise ValueError("hosted workspace differs from this source checkout")
    if not args.report.resolve().is_relative_to(temporary):
        raise ValueError("cache report must stay in the owned runner temporary tree")
    pip_cache = Path(
        subprocess.check_output(
            [sys.executable, "-m", "pip", "cache", "dir"], text=True
        ).strip()
    )
    poetry_cache = Path(
        subprocess.check_output(["poetry", "config", "cache-dir"], text=True).strip()
    )
    record = release_download_caches(
        home=Path.home(),
        pip_cache=pip_cache,
        poetry_cache=poetry_cache,
        protected=(workspace, temporary),
        disk_paths=(workspace, temporary),
        report=args.report,
    )
    print(
        json.dumps(
            {
                "status": record["status"],
                "cache_paths": list(record["caches"]),
                "disk_before": record["disk_before"],
                "disk_after": record["disk_after"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
