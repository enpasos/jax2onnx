#!/usr/bin/env python3
# scripts/generate_eqx_nn_coverage.py

"""Generate an Equinox nn API coverage checklist vs jax2onnx plugins."""

from __future__ import annotations

import argparse
import html
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from scripts._coverage_generation import write_or_check_generated

REPO_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_ROOT = REPO_ROOT / "jax2onnx" / "plugins"
EQX_PLUGIN_ROOT = PLUGIN_ROOT / "equinox" / "eqx" / "nn"
DEFAULT_DOC_ROOT = "https://docs.kidger.site/equinox/api/nn/"
DEFAULT_OUTPUT = REPO_ROOT / "work_notes_coverage_eqx_nn.md"
DEFAULT_MKDOCS_OUTPUT = REPO_ROOT / "docs" / "user_guide" / "equinox_nn_coverage.md"

SUBPAGE_LINK_RE = re.compile(r'href="(?P<href>/equinox/api/nn/[^"#?]+/)"')
ENTRY_ANCHOR_RE = re.compile(r'href="#(?P<name>equinox\.nn\.[A-Za-z0-9_.]+)"')
JAX_DOC_URL_RE = re.compile(r'jax_doc\s*=\s*"([^"]+)"')
JAXPR_LITERAL_RE = re.compile(r'jaxpr_primitive\s*=\s*"([A-Za-z0-9_.-]+)"')
JAXPR_NAME_RE = re.compile(r"jaxpr_primitive\s*=\s*([A-Za-z0-9_.]+)_p\.name")
COMPONENT_RE = re.compile(r'component\s*=\s*"([A-Za-z0-9_.-]+)"')

OUT_OF_SCOPE_EQX = {
    "equinox.nn.State",
    "equinox.nn.StateIndex",
    "equinox.nn.StatefulLayer",
    "equinox.nn.Shared",
    "equinox.nn.inference_mode",
    "equinox.nn.make_with_state",
}

ALIAS_KEY_MAP = {
    "multiheadattention": "multihead_attention",
    "rotarypositionalembedding": "rotary_positional_embedding",
    "layernorm": "layer_norm",
    "rmsnorm": "rms_norm",
    "groupnorm": "group_norm",
    "batchnorm": "batch_norm",
    "conv1d": "conv",
    "conv2d": "conv",
    "conv3d": "conv",
    "convtranspose": "conv",
    "convtranspose1d": "conv",
    "convtranspose2d": "conv",
    "convtranspose3d": "conv",
    "avgpool1d": "avg_pool",
    "avgpool2d": "avg_pool",
    "avgpool3d": "avg_pool",
    "maxpool1d": "max_pool",
    "maxpool2d": "max_pool",
    "maxpool3d": "max_pool",
    "adaptivepool": "adaptive_pool",
    "adaptiveavgpool1d": "adaptive_pool",
    "adaptiveavgpool2d": "adaptive_pool",
    "adaptiveavgpool3d": "adaptive_pool",
    "adaptivemaxpool1d": "adaptive_pool",
    "adaptivemaxpool2d": "adaptive_pool",
    "adaptivemaxpool3d": "adaptive_pool",
    "identity": "identity",
    "dropout": "dropout",
    "linear": "linear",
    "mlp": "linear",
}


@dataclass(frozen=True)
class DocEntry:
    name: str
    page_url: str


@dataclass(frozen=True)
class CoverageRow:
    entry: DocEntry
    checkbox: str
    status: str
    modules: str
    note: str


@dataclass(frozen=True)
class PluginSignals:
    eqx_exact: dict[str, set[str]]
    eqx_tokens: dict[str, set[str]]
    global_tokens: dict[str, set[str]]


def _module_id(path: Path) -> str:
    rel = path.relative_to(PLUGIN_ROOT).as_posix()
    return rel[:-3] if rel.endswith(".py") else rel


def _token_keys(name: str) -> set[str]:
    leaf = name.strip()
    if leaf.endswith("()"):
        leaf = leaf[:-2]
    leaf = leaf.split(".")[-1]
    leaf = leaf.replace("-", "_")
    if not leaf:
        return set()

    snake = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", leaf)
    snake = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", snake).strip("_")
    raw = leaf.lower().strip("_")
    snake_l = snake.lower().strip("_")

    keys = {
        raw,
        raw.replace("_", ""),
        snake_l,
        snake_l.replace("_", ""),
    }
    return {k for k in keys if k}


def _doc_name_from_url(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.fragment:
        frag = parsed.fragment.strip()
        if frag:
            return frag
    if not parsed.path:
        return None
    tail = parsed.path.rstrip("/").split("/")[-1]
    if not tail:
        return None
    if tail.endswith(".html"):
        tail = tail[:-5]
    return tail or None


def _fetch_html(url: str) -> str:
    request = Request(  # noqa: S310 (trusted docs URL)
        url,
        headers={"User-Agent": "jax2onnx-eqx-coverage-script/1.0"},
    )
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="ignore")


def fetch_doc_entries(doc_root: str) -> list[DocEntry]:
    root_html = _fetch_html(doc_root)
    subpages = sorted(
        {
            urljoin(doc_root, html.unescape(m.group("href")))
            for m in SUBPAGE_LINK_RE.finditer(root_html)
        }
    )
    if not subpages:
        raise RuntimeError(f"No Equinox nn subpages discovered in '{doc_root}'.")

    by_name: dict[str, DocEntry] = {}
    for page_url in subpages:
        page_html = _fetch_html(page_url)
        names = sorted(set(ENTRY_ANCHOR_RE.findall(page_html)))
        for name in names:
            if name not in by_name:
                by_name[name] = DocEntry(name=name, page_url=page_url)

    entries = sorted(by_name.values(), key=lambda e: e.name)
    if not entries:
        raise RuntimeError(f"No `equinox.nn.*` API entries found under '{doc_root}'.")
    return entries


def collect_plugin_signals(plugin_root: Path) -> PluginSignals:
    eqx_exact: DefaultDict[str, set[str]] = defaultdict(set)
    eqx_tokens: DefaultDict[str, set[str]] = defaultdict(set)
    global_tokens: DefaultDict[str, set[str]] = defaultdict(set)

    for path in sorted(plugin_root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        module = _module_id(path)
        is_eqx = path.is_relative_to(EQX_PLUGIN_ROOT)

        for key in _token_keys(path.stem):
            global_tokens[key].add(module)
            if is_eqx:
                eqx_tokens[key].add(module)

        for jax_doc in JAX_DOC_URL_RE.findall(text):
            doc_name = _doc_name_from_url(jax_doc)
            if doc_name is None:
                continue
            for key in _token_keys(doc_name):
                global_tokens[key].add(module)
                if is_eqx:
                    eqx_tokens[key].add(module)
            if is_eqx and doc_name.startswith("equinox.nn."):
                eqx_exact[doc_name].add(module)

        for lit in JAXPR_LITERAL_RE.findall(text):
            prim_leaf = lit.split(".")[-1].replace("-", "_")
            for key in _token_keys(prim_leaf):
                global_tokens[key].add(module)
                if is_eqx:
                    eqx_tokens[key].add(module)

        for name_expr in JAXPR_NAME_RE.findall(text):
            prim_leaf = name_expr.split(".")[-1]
            for key in _token_keys(prim_leaf):
                global_tokens[key].add(module)
                if is_eqx:
                    eqx_tokens[key].add(module)

        for component in COMPONENT_RE.findall(text):
            for key in _token_keys(component):
                global_tokens[key].add(module)
                if is_eqx:
                    eqx_tokens[key].add(module)

    return PluginSignals(
        eqx_exact=dict(eqx_exact),
        eqx_tokens=dict(eqx_tokens),
        global_tokens=dict(global_tokens),
    )


def _is_out_of_scope(entry: DocEntry) -> bool:
    if entry.name in OUT_OF_SCOPE_EQX:
        return True
    return any(entry.name.startswith(f"{prefix}.") for prefix in OUT_OF_SCOPE_EQX)


def _is_helper_entry(entry: DocEntry) -> bool:
    leaf = entry.name.split(".")[-1]
    if leaf.startswith("__") and leaf.endswith("__"):
        return True
    if leaf in {"get", "set", "substate", "update", "is_stateful"}:
        return True
    return False


def _join_modules(modules: set[str]) -> str:
    return ", ".join(sorted(modules)[:3]) if modules else "-"


def _coverage_for_entry(entry: DocEntry, *, signals: PluginSignals) -> CoverageRow:
    token_keys = set(_token_keys(entry.name))
    alias_keys = {ALIAS_KEY_MAP[k] for k in token_keys if k in ALIAS_KEY_MAP}

    if _is_out_of_scope(entry):
        return CoverageRow(
            entry=entry,
            checkbox="x",
            status="out_of_scope",
            modules="-",
            note="State/inference helper surface; no standalone converter plugin expected.",
        )

    if _is_helper_entry(entry):
        return CoverageRow(
            entry=entry,
            checkbox="x",
            status="composite",
            modules="-",
            note="Class helper/dunder API; no standalone plugin expected.",
        )

    direct_modules = set(signals.eqx_exact.get(entry.name, set()))
    for key in token_keys:
        direct_modules |= set(signals.eqx_tokens.get(key, set()))
    for key in alias_keys:
        direct_modules |= set(signals.eqx_tokens.get(key, set()))

    if direct_modules:
        return CoverageRow(
            entry=entry,
            checkbox="x",
            status="covered",
            modules=_join_modules(direct_modules),
            note="Direct Equinox plugin signal (jax_doc/jaxpr/component).",
        )

    indirect_modules: set[str] = set()
    for key in token_keys:
        indirect_modules |= set(signals.global_tokens.get(key, set()))
    for key in alias_keys:
        indirect_modules |= set(signals.global_tokens.get(key, set()))

    if indirect_modules:
        return CoverageRow(
            entry=entry,
            checkbox="x",
            status="covered_indirect",
            modules=_join_modules(indirect_modules),
            note="Covered indirectly via lower-level JAX/other plugin signals.",
        )

    return CoverageRow(
        entry=entry,
        checkbox=" ",
        status="missing",
        modules="-",
        note="Missing dedicated Equinox plugin coverage.",
    )


def build_rows(entries: list[DocEntry], *, signals: PluginSignals) -> list[CoverageRow]:
    return [_coverage_for_entry(entry, signals=signals) for entry in entries]


def render_markdown(rows: list[CoverageRow], *, doc_root: str, title: str) -> str:
    total = len(rows)
    covered = sum(1 for r in rows if r.status == "covered")
    covered_indirect = sum(1 for r in rows if r.status == "covered_indirect")
    composite = sum(1 for r in rows if r.status == "composite")
    out_of_scope = sum(1 for r in rows if r.status == "out_of_scope")
    missing = sum(1 for r in rows if r.status == "missing")

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## Scope")
    lines.append(
        f"- Source list: all `equinox.nn.*` API anchors discovered from: `{doc_root}`"
    )
    lines.append(
        "- Coverage signal: `jax_doc`, `jaxpr_primitive`, and `component` metadata in `jax2onnx/plugins/**/*.py`."
    )
    lines.append("")
    lines.append("## Snapshot")
    lines.append(f"- Total Equinox nn API entries: `{total}`")
    lines.append(f"- Covered (direct Equinox plugin): `{covered}`")
    lines.append(f"- Covered (indirect signal): `{covered_indirect}`")
    lines.append(f"- Composite/helper entries: `{composite}`")
    lines.append(f"- Out-of-scope state/inference entries: `{out_of_scope}`")
    lines.append(f"- Missing dedicated Equinox coverage: `{missing}`")
    lines.append("")
    lines.append("## Full Checklist")
    lines.append(
        "Legend: `covered`, `covered_indirect`, `composite`, `out_of_scope`, `missing`."
    )
    lines.append("")
    lines.append(
        "| Checklist | Equinox API Entry | Status | Modules (signals) | Notes |"
    )
    lines.append("|:--|:--|:--|:--|:--|")
    for row in rows:
        lines.append(
            f"| [{row.checkbox}] | `{row.entry.name}` | `{row.status}` | "
            f"`{row.modules}` | {row.note} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Equinox nn coverage checklist from docs + plugin signals."
    )
    parser.add_argument("--doc-root", default=DEFAULT_DOC_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mkdocs-output", type=Path, default=DEFAULT_MKDOCS_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check whether the MkDocs coverage page is current without writing files.",
    )
    args = parser.parse_args()

    entries = fetch_doc_entries(args.doc_root)
    signals = collect_plugin_signals(PLUGIN_ROOT)
    rows = build_rows(entries, signals=signals)

    work_notes = render_markdown(
        rows,
        doc_root=args.doc_root,
        title="Work Notes: Equinox NN Coverage Checklist",
    )

    mkdocs_page = render_markdown(
        rows,
        doc_root=args.doc_root,
        title="Equinox NN Coverage Checklist",
    )
    if args.check:
        write_or_check_generated(
            args.mkdocs_output,
            mkdocs_page + "\n",
            check=True,
            label="Equinox NN coverage page",
        )
        return

    write_or_check_generated(
        args.output,
        work_notes + "\n",
        check=False,
        label=f"Equinox NN work notes coverage page ({len(rows)} rows)",
    )
    if args.mkdocs_output != args.output:
        write_or_check_generated(
            args.mkdocs_output,
            mkdocs_page + "\n",
            check=False,
            label=f"Equinox NN MkDocs coverage page ({len(rows)} rows)",
        )


if __name__ == "__main__":
    main()
