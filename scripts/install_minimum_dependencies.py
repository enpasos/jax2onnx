#!/usr/bin/env python3
# scripts/install_minimum_dependencies.py

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Sequence


_NAME_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_.\-,]+\])?)")
_BOUND_RE = re.compile(r"(==|>=)\s*([^,;\s]+)")


def minimum_requirement(dependency: str) -> str:
    """Return an exact lower-bound requirement for a PEP 508 dependency string."""

    unmarked = dependency.split(";", 1)[0].strip()
    name_match = _NAME_RE.match(unmarked)
    if name_match is None:
        raise ValueError(f"Cannot parse dependency name from {dependency!r}")
    name = name_match.group(1)

    for operator, version in _BOUND_RE.findall(unmarked):
        if operator in {"==", ">="}:
            return f"{name}=={version}"
    raise ValueError(f"Dependency {dependency!r} has no exact or lower bound")


def minimum_requirements(pyproject_path: Path) -> list[str]:
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    dependencies = pyproject.get("project", {}).get("dependencies", [])
    if not isinstance(dependencies, list):
        raise TypeError("[project].dependencies must be a list")
    return [minimum_requirement(str(dependency)) for dependency in dependencies]


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Install direct runtime dependencies at the minimum versions declared "
            "in pyproject.toml."
        )
    )
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=Path("pyproject.toml"),
        help="Path to pyproject.toml.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the pip command without installing anything.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    requirements = minimum_requirements(args.pyproject)
    command = [sys.executable, "-m", "pip", "install", *requirements]

    if args.dry_run:
        print(" ".join(command))
        return 0

    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
