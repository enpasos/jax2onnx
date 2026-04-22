# scripts/_coverage_generation.py

"""Shared helpers for generated coverage documentation."""

from __future__ import annotations

import difflib
from pathlib import Path


def _with_trailing_newline(content: str) -> str:
    return content if content.endswith("\n") else content + "\n"


def _diff_preview(actual: str, expected: str, *, max_lines: int = 80) -> str:
    diff = list(
        difflib.unified_diff(
            actual.splitlines(keepends=True),
            expected.splitlines(keepends=True),
            fromfile="current",
            tofile="generated",
        )
    )
    if len(diff) <= max_lines:
        return "".join(diff)
    preview = "".join(diff[:max_lines])
    return preview + f"... diff truncated after {max_lines} lines ...\n"


def write_or_check_generated(
    path: Path,
    content: str,
    *,
    check: bool,
    label: str,
) -> None:
    """Write generated content or check whether the target is current."""

    expected = _with_trailing_newline(content)
    if not check:
        path.write_text(expected, encoding="utf-8")
        print(f"Wrote {label} to {path}")
        return

    try:
        actual = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SystemExit(f"{label} missing: {path}") from exc

    if actual != expected:
        diff = _diff_preview(actual, expected)
        raise SystemExit(f"{label} is stale: {path}\n{diff}")

    print(f"{label} is up to date: {path}")
