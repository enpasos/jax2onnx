#!/usr/bin/env python3
# scripts/generate_jnp_operator_coverage.py

"""Generate a checklist of jax.numpy coverage vs jax2onnx plugins."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_ROOT = REPO_ROOT / "jax2onnx" / "plugins"
DEFAULT_OUTPUT = REPO_ROOT / "work_notes_coverage_jnp.md"
DEFAULT_MKDOCS_OUTPUT = REPO_ROOT / "docs" / "user_guide" / "jax_numpy_coverage.md"
DEFAULT_DOC_URL = "https://docs.jax.dev/en/latest/jax.numpy.html"

AUTOSUMMARY_RE = re.compile(r"_autosummary/jax\.numpy\.([A-Za-z0-9_.]+)\.html")
JAX_DOC_URL_RE = re.compile(r'jax_doc\s*=\s*"([^"]+)"')
JAX_DOC_NAME_RE = re.compile(r"jax\.numpy\.([A-Za-z0-9_.]+?)(?:\.html|$)")
JAXPR_LITERAL_RE = re.compile(r'jaxpr_primitive\s*=\s*"([A-Za-z0-9_.-]+)"')
JAXPR_NAME_RE = re.compile(r"jaxpr_primitive\s*=\s*([A-Za-z0-9_.]+)_p\.name")

ALIAS_TO_OP = {
    "absolute": "abs",
    "acos": "arccos",
    "acosh": "arccosh",
    "asin": "arcsin",
    "asinh": "arcsinh",
    "atan": "arctan",
    "atan2": "arctan2",
    "atanh": "arctanh",
    "bitwise_invert": "bitwise_not",
    "concat": "concatenate",
    "conjugate": "conj",
    "cumulative_prod": "cumprod",
    "cumulative_sum": "cumsum",
    "pow": "power",
    "round_": "round",
    "tril": "triu",
}

COMPOSITE_HELPERS = {
    "append",
    "apply_along_axis",
    "apply_over_axes",
    "array",
    "array_equal",
    "array_equiv",
    "array_repr",
    "array_split",
    "array_str",
    "asarray",
    "astype",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "broadcast_arrays",
    "broadcast_shapes",
    "broadcast_to",
    "c_",
    "column_stack",
    "diag_indices",
    "diag_indices_from",
    "diagflat",
    "dsplit",
    "dstack",
    "ediff1d",
    "expand_dims",
    "flatnonzero",
    "fliplr",
    "flipud",
    "frombuffer",
    "fromfunction",
    "fromiter",
    "fromstring",
    "hsplit",
    "hstack",
    "indices",
    "insert",
    "isclose",
    "isneginf",
    "isposinf",
    "ix_",
    "kaiser",
    "meshgrid",
    "mgrid",
    "msort",
    "ndenumerate",
    "newaxis",
    "ogrid",
    "piecewise",
    "r_",
    "repeat",
    "resize",
    "rollaxis",
    "s_",
    "setdiff1d",
    "setxor1d",
    "stack",
    "take_along_axis",
    "trim_zeros",
    "union1d",
    "unwrap",
    "vander",
    "vsplit",
    "vstack",
}

NON_FUNCTIONAL_ENTRIES = {
    "ComplexWarning",
    "bool_",
    "byte",
    "bytes_",
    "cdouble",
    "character",
    "clongdouble",
    "complex64",
    "complex128",
    "complex_",
    "complexfloating",
    "csingle",
    "double",
    "dtype",
    "finfo",
    "flatiter",
    "floating",
    "generic",
    "half",
    "inexact",
    "int_",
    "int8",
    "int16",
    "int32",
    "int64",
    "integer",
    "issubsctype",
    "issubdtype",
    "longdouble",
    "ndarray",
    "number",
    "promote_types",
    "result_type",
    "signedinteger",
    "single",
    "timedelta64",
    "uint",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "unsignedinteger",
    "void",
}

QUICK_WIN_CANDIDATES = {
    "abs",
    "all",
    "any",
    "argmax",
    "argmin",
    "ceil",
    "cos",
    "cosh",
    "diag",
    "dot",
    "equal",
    "exp",
    "floor",
    "isfinite",
    "log",
    "max",
    "min",
    "ones",
    "pad",
    "sign",
    "sin",
    "sinh",
    "sqrt",
    "sum",
    "tan",
    "tanh",
    "zeros",
}


@dataclass(frozen=True)
class CoverageRow:
    name: str
    checkbox: str
    status: str
    modules: str
    note: str


def _module_id(path: Path) -> str:
    rel = path.relative_to(PLUGIN_ROOT).as_posix()
    return rel[:-3] if rel.endswith(".py") else rel


def fetch_doc_ops(doc_url: str) -> list[str]:
    request = Request(  # noqa: S310 (trusted docs URL)
        doc_url,
        headers={"User-Agent": "jax2onnx-jnp-coverage-script/1.0"},
    )
    with urlopen(request, timeout=30) as response:
        html = response.read().decode("utf-8", errors="ignore")
    ops = sorted(set(AUTOSUMMARY_RE.findall(html)))
    if not ops:
        raise RuntimeError(f"No jax.numpy autosummary entries found in '{doc_url}'.")
    return ops


def collect_plugin_signals(
    plugin_root: Path,
) -> tuple[DefaultDict[str, set[str]], DefaultDict[str, set[str]]]:
    doc_usage: DefaultDict[str, set[str]] = defaultdict(set)
    prim_usage: DefaultDict[str, set[str]] = defaultdict(set)

    for path in sorted(plugin_root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        module = _module_id(path)

        for jax_doc in JAX_DOC_URL_RE.findall(text):
            for op in JAX_DOC_NAME_RE.findall(jax_doc):
                doc_usage[op].add(module)
                doc_usage[op.split(".")[-1]].add(module)

        for lit in JAXPR_LITERAL_RE.findall(text):
            prim_usage[lit].add(module)
            prim_usage[lit.replace("-", "_")].add(module)
            prim_usage[lit.replace("_", "-")].add(module)

        for name_expr in JAXPR_NAME_RE.findall(text):
            prim = name_expr.split(".")[-1]
            prim_usage[prim].add(module)
            prim_usage[prim.replace("-", "_")].add(module)
            prim_usage[prim.replace("_", "-")].add(module)

    return doc_usage, prim_usage


def _status_for_op(
    op: str,
    *,
    doc_usage: dict[str, set[str]],
    prim_usage: dict[str, set[str]],
) -> tuple[str, str, str]:
    base = op.split(".")[-1]
    modules = set(doc_usage.get(op, set())) | set(doc_usage.get(base, set()))
    modules_text = ", ".join(sorted(modules)[:3]) if modules else "-"

    if modules:
        return "covered", modules_text, "Direct plugin coverage via `jax_doc` metadata."

    alias = ALIAS_TO_OP.get(base)
    if alias and (
        doc_usage.get(alias)
        or doc_usage.get(alias.split(".")[-1])
        or prim_usage.get(alias)
        or prim_usage.get(alias.replace("-", "_"))
        or prim_usage.get(alias.replace("_", "-"))
    ):
        alias_modules = (
            set(doc_usage.get(alias, set()))
            | set(doc_usage.get(alias.split(".")[-1], set()))
            | set(prim_usage.get(alias, set()))
            | set(prim_usage.get(alias.replace("-", "_"), set()))
            | set(prim_usage.get(alias.replace("_", "-"), set()))
        )
        alias_modules_text = (
            ", ".join(sorted(alias_modules)[:3]) if alias_modules else "-"
        )
        return "covered_indirect", alias_modules_text, f"Covered via alias `{alias}`."

    if base in NON_FUNCTIONAL_ENTRIES:
        return (
            "non_functional",
            modules_text,
            "Constant/dtype/type helper entry; no standalone plugin expected.",
        )

    if base in COMPOSITE_HELPERS:
        return (
            "composite",
            modules_text,
            "Composite/helper API; typically lowered through other primitives.",
        )

    return "missing", modules_text, "Missing dedicated `jax.numpy` plugin coverage."


def build_rows(
    ops: list[str],
    *,
    doc_usage: dict[str, set[str]],
    prim_usage: dict[str, set[str]],
) -> list[CoverageRow]:
    rows: list[CoverageRow] = []
    for op in ops:
        status, modules_text, note = _status_for_op(
            op, doc_usage=doc_usage, prim_usage=prim_usage
        )
        checkbox = (
            "x"
            if status in {"covered", "covered_indirect", "composite", "non_functional"}
            else " "
        )
        rows.append(
            CoverageRow(
                name=op,
                checkbox=checkbox,
                status=status,
                modules=modules_text,
                note=note,
            )
        )
    return rows


def render_markdown(rows: list[CoverageRow], *, doc_url: str, title: str) -> str:
    total = len(rows)
    covered = sum(1 for r in rows if r.status == "covered")
    covered_indirect = sum(1 for r in rows if r.status == "covered_indirect")
    composite = sum(1 for r in rows if r.status == "composite")
    non_functional = sum(1 for r in rows if r.status == "non_functional")
    missing = sum(1 for r in rows if r.status == "missing")

    quick_win = [
        r.name
        for r in rows
        if r.status == "missing" and r.name.split(".")[-1] in QUICK_WIN_CANDIDATES
    ]

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## Scope")
    lines.append(
        "- Source list: all autosummary entries linked on the official `jax.numpy` page:"
        f" `{doc_url}`"
    )
    lines.append(
        "- Coverage signal: `jax_doc` metadata + `jaxpr_primitive` registrations in `jax2onnx/plugins/**/*.py`."
    )
    lines.append("")
    lines.append("Regenerate with:")
    lines.append("")
    lines.append("```bash")
    lines.append("poetry run python scripts/generate_jnp_operator_coverage.py")
    lines.append("```")
    lines.append("")
    lines.append("## Snapshot")
    lines.append(f"- Total docs entries: `{total}`")
    lines.append(f"- Covered (direct plugin): `{covered}`")
    lines.append(f"- Covered (via alias/indirect signal): `{covered_indirect}`")
    lines.append(f"- Composite/helper entries: `{composite}`")
    lines.append(f"- Non-functional entries (dtype/type/constants): `{non_functional}`")
    lines.append(f"- Missing dedicated plugin coverage: `{missing}`")
    lines.append("")
    lines.append("## Priority Gap Queue")
    if quick_win:
        for name in quick_win:
            lines.append(f"- [ ] `{name}`")
    else:
        lines.append(
            "- No quick-win candidates currently marked missing by this heuristic."
        )
    lines.append("")
    lines.append("## Full Checklist")
    lines.append(
        "Legend: `covered`, `covered_indirect`, `composite`, `non_functional`, `missing`."
    )
    lines.append("")
    lines.append("| Checklist | jax.numpy Entry | Status | Modules (signals) | Notes |")
    lines.append("|:--|:--|:--|:--|:--|")
    for row in rows:
        lines.append(
            f"| [{row.checkbox}] | `{row.name}` | `{row.status}` | `{row.modules}` | {row.note} |"
        )
    lines.append("")
    lines.append("## Next Steps")
    lines.append(
        "1. Implement missing quick-win `jax.numpy` plugins from the queue above."
    )
    lines.append(
        "2. Add metadata testcases for each new plugin and regenerate tests (`scripts/generate_tests.py`)."
    )
    lines.append(
        "3. Re-run this script after each batch to keep coverage docs in sync."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate jax.numpy coverage checklist from docs + plugin signals."
    )
    parser.add_argument("--doc-url", default=DEFAULT_DOC_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mkdocs-output", type=Path, default=DEFAULT_MKDOCS_OUTPUT)
    args = parser.parse_args()

    ops = fetch_doc_ops(args.doc_url)
    doc_usage, prim_usage = collect_plugin_signals(PLUGIN_ROOT)
    rows = build_rows(ops, doc_usage=doc_usage, prim_usage=prim_usage)

    work_notes_content = render_markdown(
        rows,
        doc_url=args.doc_url,
        title="Work Notes: JAX NumPy Coverage Checklist",
    )
    args.output.write_text(work_notes_content + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {args.output}")

    mkdocs_content = render_markdown(
        rows,
        doc_url=args.doc_url,
        title="JAX NumPy Coverage Checklist",
    )
    if args.mkdocs_output != args.output:
        args.mkdocs_output.write_text(mkdocs_content + "\n", encoding="utf-8")
        print(f"Wrote {len(rows)} rows to {args.mkdocs_output}")


if __name__ == "__main__":
    main()
