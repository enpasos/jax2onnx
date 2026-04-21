# tests/extra_tests/framework/test_coverage_docs.py

"""Guard generated component coverage docs against internal drift."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
from typing import Final

REPO_ROOT = Path(__file__).resolve().parents[3]

SNAPSHOT_RE: Final = re.compile(r"^- (?P<label>.+?): `(?P<count>\d+)`$")
ROW_RE: Final = re.compile(
    r"^\| \[(?P<check>[ x])\] \| `(?P<name>[^`]+)` \| "
    r"`(?P<status>[^`]+)` \| `(?P<modules>[^`]*)` \| (?P<note>.*) \|$"
)


def _read_doc(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def _snapshot_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in text.splitlines():
        match = SNAPSHOT_RE.match(line)
        if match is not None:
            counts[match.group("label")] = int(match.group("count"))
    return counts


def _table_rows(text: str) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for line in text.splitlines():
        match = ROW_RE.match(line)
        if match is not None:
            rows.append(
                (
                    match.group("check"),
                    match.group("name"),
                    match.group("status"),
                )
            )
    return rows


def _assert_doc_counts(
    *,
    relative_path: str,
    status_labels: dict[str, str],
    checked_statuses: set[str],
    unchecked_statuses: set[str],
    total_label: str,
    derived_counts: dict[str, int] | None = None,
) -> None:
    text = _read_doc(relative_path)
    assert "## Snapshot" in text
    assert "## Full Checklist" in text

    snapshot = _snapshot_counts(text)
    rows = _table_rows(text)
    assert rows, f"{relative_path} has no coverage table rows"

    statuses = Counter(status for _, _, status in rows)
    expected_statuses = (
        set(status_labels.values()) | checked_statuses | unchecked_statuses
    )
    assert set(statuses) <= expected_statuses
    assert snapshot[total_label] == len(rows)

    for label, status in status_labels.items():
        assert snapshot[label] == statuses[status]

    for label, expected in (derived_counts or {}).items():
        assert snapshot[label] == expected

    for checkbox, name, status in rows:
        if status in checked_statuses:
            assert checkbox == "x", f"{relative_path}: {name} should be checked"
        elif status in unchecked_statuses:
            assert checkbox == " ", f"{relative_path}: {name} should be unchecked"
        else:
            raise AssertionError(
                f"{relative_path}: unknown status {status!r} for {name}"
            )


def test_jax_lax_coverage_doc_snapshot_matches_table() -> None:
    _assert_doc_counts(
        relative_path="docs/user_guide/jax_lax_coverage.md",
        total_label="Total docs operators",
        status_labels={
            "Covered (direct plugin)": "covered",
            "Covered (via alias primitive)": "covered_indirect",
            "Composite/helper (no standalone plugin expected)": "composite",
            "Out of scope (distributed/token/host)": "out_of_scope",
            "Missing primitive plugins": "missing",
            "Missing `lax.linalg` plugins": "missing_linalg",
        },
        checked_statuses={
            "covered",
            "covered_indirect",
            "composite",
            "out_of_scope",
        },
        unchecked_statuses={"missing", "missing_linalg"},
    )


def test_jax_numpy_coverage_doc_snapshot_matches_table() -> None:
    _assert_doc_counts(
        relative_path="docs/user_guide/jax_numpy_coverage.md",
        total_label="Total docs entries",
        status_labels={
            "Covered (direct plugin)": "covered",
            "Covered (via alias/indirect signal)": "covered_indirect",
            "Composite/helper entries": "composite",
            "Non-functional entries (dtype/type/constants)": "non_functional",
            "Missing dedicated plugin coverage": "missing",
        },
        checked_statuses={
            "covered",
            "covered_indirect",
            "composite",
            "non_functional",
        },
        unchecked_statuses={"missing"},
    )


def test_flax_api_coverage_doc_snapshot_matches_table() -> None:
    text = _read_doc("docs/user_guide/flax_api_coverage.md")
    statuses = Counter(status for _, _, status in _table_rows(text))
    in_scope_count = (
        statuses["covered"]
        + statuses["covered_indirect"]
        + statuses["composite"]
        + statuses["missing"]
    )

    _assert_doc_counts(
        relative_path="docs/user_guide/flax_api_coverage.md",
        total_label="Total API entries discovered",
        status_labels={
            "Covered (direct Flax plugin)": "covered",
            "Covered (indirect via lower-level plugins)": "covered_indirect",
            "Composite/helper entries": "composite",
            "Out-of-scope entries": "out_of_scope",
            "Missing dedicated Flax coverage": "missing",
        },
        derived_counts={
            "In-scope neural entries (`flax.linen/*`, `flax.nnx/nn/*`)": in_scope_count,
        },
        checked_statuses={
            "covered",
            "covered_indirect",
            "composite",
            "out_of_scope",
        },
        unchecked_statuses={"missing"},
    )
