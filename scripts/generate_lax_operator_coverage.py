#!/usr/bin/env python3
# scripts/generate_lax_operator_coverage.py

"""Generate a checklist of jax.lax operator coverage vs jax2onnx plugins."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict
from urllib.request import Request, urlopen

from scripts._coverage_generation import write_or_check_generated

REPO_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_ROOT = REPO_ROOT / "jax2onnx" / "plugins"
DEFAULT_OUTPUT = REPO_ROOT / "work_notes_coverage3.md"
DEFAULT_MKDOCS_OUTPUT = REPO_ROOT / "docs" / "user_guide" / "jax_lax_coverage.md"
DEFAULT_DOC_URL = "https://docs.jax.dev/en/latest/jax.lax.html"

AUTOSUMMARY_RE = re.compile(r"_autosummary/jax\.lax\.([A-Za-z0-9_.]+)\.html")
JAX_DOC_URL_RE = re.compile(r'jax_doc\s*=\s*"([^"]+)"')
JAX_DOC_NAME_RE = re.compile(r"jax\.lax\.([A-Za-z0-9_.]+)")
JAXPR_LITERAL_RE = re.compile(r'jaxpr_primitive\s*=\s*"([A-Za-z0-9_.-]+)"')
JAXPR_NAME_RE = re.compile(r"jaxpr_primitive\s*=\s*([A-Za-z0-9_.]+)_p\.name")

ALIAS_TO_PRIMITIVE = {
    "approx_max_k": "approx_top_k",
    "approx_min_k": "approx_top_k",
    "batch_matmul": "dot_general",
    "betainc": "regularized_incomplete_beta",
    "bitwise_and": "and",
    "bitwise_not": "not",
    "bitwise_or": "or",
    "bitwise_xor": "xor",
    "conv": "conv_general_dilated",
    "dot": "dot_general",
    "fori_loop": "lax.fori_loop",
    "ragged_dot": "ragged_dot_general",
    "reciprocal": "integer_pow",
    "triangular_solve": "triangular_solve",
    "while_loop": "while",
    "with_sharding_constraint": "sharding_constraint",
}

COMPOSITE_HELPERS = {
    "associative_scan",
    "broadcast",
    "broadcasted_iota",
    "broadcast_shapes",
    "broadcast_to_rank",
    "collapse",
    "composite",
    "conv_dimension_numbers",
    "conv_general_dilated_local",
    "conv_general_dilated_patches",
    "conv_transpose",
    "conv_with_general_padding",
    "custom_root",
    "dynamic_index_in_dim",
    "dynamic_slice_in_dim",
    "dynamic_update_index_in_dim",
    "dynamic_update_slice_in_dim",
    "empty",
    "expand_dims",
    "full",
    "full_like",
    "index_in_dim",
    "index_take",
    "map",
    "SvdAlgorithm",
    "qdwh",
    "random_gamma_grad",
    "scaled_dot",
    "scatter_apply",
    "slice_in_dim",
    "sort_key_val",
    "switch",
    "tile",
}

DISTRIBUTED_TOKEN_OR_HOST = {
    "after_all",
    "all_gather",
    "all_to_all",
    "axis_index",
    "axis_size",
    "platform_dependent",
    "pmax",
    "pmean",
    "pmin",
    "ppermute",
    "precv",
    "psend",
    "pshuffle",
    "psum",
    "psum_scatter",
    "pswapaxes",
    "ragged_all_to_all",
}

QUICK_WIN_CANDIDATES = {
    "cumlogsumexp",
    "cummax",
    "cummin",
    "erf_inv",
    "reduce_window",
    "reduce_window_max",
    "reduce_window_min",
    "scatter_sub",
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
        headers={"User-Agent": "jax2onnx-coverage-script/1.0"},
    )
    with urlopen(request, timeout=20) as response:
        html = response.read().decode("utf-8", errors="ignore")
    ops = sorted(set(AUTOSUMMARY_RE.findall(html)))
    if not ops:
        raise RuntimeError(f"No autosummary operators found in '{doc_url}'.")
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


def _modules_for_op(
    op: str,
    *,
    doc_usage: dict[str, set[str]],
    prim_usage: dict[str, set[str]],
) -> set[str]:
    base = op.split(".")[-1]
    modules: set[str] = set()
    modules |= doc_usage.get(op, set())
    modules |= doc_usage.get(base, set())

    prim_names = {base, base.replace("_", "-"), base.replace("-", "_")}
    alias = ALIAS_TO_PRIMITIVE.get(base)
    if alias is not None:
        prim_names |= {alias, alias.replace("_", "-"), alias.replace("-", "_")}

    for prim in prim_names:
        modules |= prim_usage.get(prim, set())
    return modules


def _status_for_op(
    op: str,
    *,
    doc_usage: dict[str, set[str]],
    prim_usage: dict[str, set[str]],
) -> tuple[str, str, str]:
    base = op.split(".")[-1]
    modules = _modules_for_op(op, doc_usage=doc_usage, prim_usage=prim_usage)
    modules_text = ", ".join(sorted(modules)[:3]) if modules else "-"

    direct_doc = bool(doc_usage.get(op) or doc_usage.get(base))
    direct_prim = bool(
        prim_usage.get(base)
        or prim_usage.get(base.replace("_", "-"))
        or prim_usage.get(base.replace("-", "_"))
    )

    if direct_doc or direct_prim:
        return "covered", modules_text, "Direct plugin coverage."

    alias = ALIAS_TO_PRIMITIVE.get(base)
    if alias and (
        prim_usage.get(alias)
        or prim_usage.get(alias.replace("_", "-"))
        or prim_usage.get(alias.replace("-", "_"))
    ):
        return "covered_indirect", modules_text, f"Covered via `{alias}` primitive."

    if base in COMPOSITE_HELPERS:
        return (
            "composite",
            modules_text,
            "Composite/helper API; no standalone primitive plugin.",
        )

    if base in DISTRIBUTED_TOKEN_OR_HOST:
        return (
            "out_of_scope",
            modules_text,
            "Distributed/token/host path; currently out of converter scope.",
        )

    if op.startswith("linalg.") and base != "triangular_solve":
        return (
            "missing_linalg",
            modules_text,
            "Missing `jax.lax.linalg` primitive plugin.",
        )

    return "missing", modules_text, "Missing primitive plugin."


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
        if status in {"covered", "covered_indirect", "composite", "out_of_scope"}:
            checkbox = "x"
        else:
            checkbox = " "
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
    covered_direct = sum(1 for r in rows if r.status == "covered")
    covered_indirect = sum(1 for r in rows if r.status == "covered_indirect")
    composite = sum(1 for r in rows if r.status == "composite")
    out_of_scope = sum(1 for r in rows if r.status == "out_of_scope")
    missing = sum(1 for r in rows if r.status == "missing")
    missing_linalg = sum(1 for r in rows if r.status == "missing_linalg")

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## Scope")
    lines.append(
        "- Source list: all operators linked on the official `jax.lax` docs page:"
        f" `{doc_url}`"
    )
    lines.append(
        "- Coverage signal: `jax_doc` metadata + `jaxpr_primitive` registrations in `jax2onnx/plugins/**/*.py`."
    )
    lines.append("")
    lines.append("## Snapshot")
    lines.append(f"- Total docs operators: `{total}`")
    lines.append(f"- Covered (direct plugin): `{covered_direct}`")
    lines.append(f"- Covered (via alias primitive): `{covered_indirect}`")
    lines.append(f"- Composite/helper (no standalone plugin expected): `{composite}`")
    lines.append(f"- Out of scope (distributed/token/host): `{out_of_scope}`")
    lines.append(f"- Missing primitive plugins: `{missing}`")
    lines.append(f"- Missing `lax.linalg` plugins: `{missing_linalg}`")
    lines.append("")
    lines.append("## Full Checklist")
    lines.append(
        "Legend: `covered`, `covered_indirect`, `composite`, `out_of_scope`, `missing`, `missing_linalg`."
    )
    lines.append("")
    lines.append(
        "| Checklist | jax.lax Operator | Status | Modules (signals) | Notes |"
    )
    lines.append("|:--|:--|:--|:--|:--|")
    for row in rows:
        lines.append(
            f"| [{row.checkbox}] | `{row.name}` | `{row.status}` | `{row.modules}` | {row.note} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate work_notes_coverage3.md from JAX docs + plugin signals."
    )
    parser.add_argument("--doc-url", default=DEFAULT_DOC_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mkdocs-output", type=Path, default=DEFAULT_MKDOCS_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check whether the MkDocs coverage page is current without writing files.",
    )
    args = parser.parse_args()

    ops = fetch_doc_ops(args.doc_url)
    doc_usage, prim_usage = collect_plugin_signals(PLUGIN_ROOT)
    rows = build_rows(ops, doc_usage=doc_usage, prim_usage=prim_usage)
    work_notes_content = render_markdown(
        rows,
        doc_url=args.doc_url,
        title="Work Notes: JAX LAX Coverage Checklist (v3)",
    )

    mkdocs_content = render_markdown(
        rows,
        doc_url=args.doc_url,
        title="JAX LAX Coverage Checklist",
    )
    if args.check:
        write_or_check_generated(
            args.mkdocs_output,
            mkdocs_content + "\n",
            check=True,
            label="JAX LAX coverage page",
        )
        return

    write_or_check_generated(
        args.output,
        work_notes_content + "\n",
        check=False,
        label=f"JAX LAX work notes coverage page ({len(rows)} rows)",
    )
    if args.mkdocs_output != args.output:
        write_or_check_generated(
            args.mkdocs_output,
            mkdocs_content + "\n",
            check=False,
            label=f"JAX LAX MkDocs coverage page ({len(rows)} rows)",
        )


if __name__ == "__main__":
    main()
