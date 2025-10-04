#!/usr/bin/env python3
# scripts/ensure_path_markers.py

"""Ensure Python files start with a path marker comment followed by a blank line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

# Directories that should never be touched by the marker enforcement.
IGNORED_TOP_LEVEL = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    "build",
    "dist",
    "python-venv",
    "debugvenv",
    "venv",
}

# Subtrees (relative to repo root) that are regenerated and should be skipped.
IGNORED_PREFIXES = (
    ("tests", "examples"),
    ("tests", "primitives"),
)


def iter_python_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        try:
            relative_parts = path.relative_to(root).parts
        except ValueError:
            # File lies outside the declared root; skip for safety.
            continue

        if any(part in IGNORED_TOP_LEVEL for part in path.parts):
            continue

        if any(relative_parts[: len(prefix)] == prefix for prefix in IGNORED_PREFIXES):
            continue

        if path.is_symlink():
            continue
        yield path


def expected_marker(root: Path, file_path: Path) -> str:
    relative = file_path.relative_to(root)
    return f"# {relative.as_posix()}"


def normalize_newlines(lines: list[str]) -> list[str]:
    # Drop Windows carriage returns if present.
    return [line.rstrip("\r") for line in lines]


def ensure_marker(file_path: Path, marker: str, fix: bool) -> bool:
    raw_text = file_path.read_text(encoding="utf-8")
    lines = normalize_newlines(raw_text.split("\n"))

    # Guarantee the list has at least one element so indexing is safe.
    if not lines:
        lines = [""]

    insert_at = 0
    if lines[0].startswith("#!"):
        insert_at = 1

    current_marker = lines[insert_at] if insert_at < len(lines) else None
    modified = False

    if current_marker != marker:
        if not fix:
            return False

        if insert_at < len(lines):
            if lines[insert_at].startswith("# "):
                lines[insert_at] = marker
            else:
                lines.insert(insert_at, marker)
        else:
            lines.append(marker)
        modified = True

    marker_index = insert_at if insert_at < len(lines) else len(lines) - 1

    # Ensure there is a blank line immediately after the marker.
    next_index = marker_index + 1
    next_line = lines[next_index] if next_index < len(lines) else None
    has_blank_line = next_line == ""

    if not has_blank_line:
        if next_line is not None and next_line.strip() == "":
            if not fix:
                return False
            lines[next_index] = ""
            modified = True
        else:
            if not fix:
                return False
            lines.insert(next_index, "")
            modified = True

    # Remove stray empty lines above the marker (except keeping shebang gap).
    if insert_at == 0:
        while len(lines) > 1 and lines[0] == "" and not lines[1].startswith("# "):
            lines.pop(0)
            modified = True

    if fix and modified:
        # Ensure trailing newline when writing back.
        file_path.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")

    return True


def process(root: Path, fix: bool) -> int:
    failures: list[Path] = []
    for file_path in iter_python_files(root):
        marker = expected_marker(root, file_path)
        has_marker = ensure_marker(file_path, marker, fix)
        if not has_marker:
            failures.append(file_path)

    if failures and not fix:
        for path in failures:
            rel = path.relative_to(root)
            print(f"Marker header check failed: {rel.as_posix()}", file=sys.stderr)
        print(
            "Run `poetry run python scripts/ensure_path_markers.py --fix` to update markers.",
            file=sys.stderr,
        )
        return 1

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (defaults to project root).",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Rewrite files in place instead of just checking.",
    )

    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not root.exists():
        parser.error(f"Root path does not exist: {root}")

    return process(root, fix=args.fix)


if __name__ == "__main__":
    sys.exit(main())
