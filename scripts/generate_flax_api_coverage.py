#!/usr/bin/env python3
# scripts/generate_flax_api_coverage.py

"""Generate a checklist of Flax API coverage vs jax2onnx plugins."""

from __future__ import annotations

import argparse
import html
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict
from urllib.parse import urlparse
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_ROOT = REPO_ROOT / "jax2onnx" / "plugins"
FLAX_PLUGIN_ROOT = PLUGIN_ROOT / "flax"
DEFAULT_OUTPUT = REPO_ROOT / "work_notes_coverage_flax.md"
DEFAULT_MKDOCS_OUTPUT = REPO_ROOT / "docs" / "user_guide" / "flax_api_coverage.md"
DEFAULT_DOC_URL = "https://flax.readthedocs.io/en/latest/api_reference/index.html"

DOC_ENTRY_RE = re.compile(
    r'href="(?P<href>[^"]+#(?P<anchor>flax\.[^"]+))"[^>]*><code class="docutils '
    r'literal notranslate"><span class="pre">(?P<label>[^<]+)</span></code>'
)
JAX_DOC_URL_RE = re.compile(r'jax_doc\s*=\s*"([^"]+)"')
JAXPR_LITERAL_RE = re.compile(r'jaxpr_primitive\s*=\s*"([A-Za-z0-9_.-]+)"')
JAXPR_NAME_RE = re.compile(r"jaxpr_primitive\s*=\s*([A-Za-z0-9_.]+)_p\.name")
COMPONENT_RE = re.compile(r'component\s*=\s*"([A-Za-z0-9_.-]+)"')

IN_SCOPE_PATH_PREFIXES = (
    "flax.linen/",
    "flax.nnx/nn/",
)
COMPOSITE_PATH_SUFFIXES = (
    "flax.nnx/nn/dtypes.html",
    "flax.nnx/nn/initializers.html",
)
ALIAS_KEY_MAP = {
    "swish": "silu",
    "hardsilu": "hardswish",
    "softsign": "soft_sign",
}


@dataclass(frozen=True)
class DocEntry:
    name: str
    doc_path: str
    label: str


@dataclass(frozen=True)
class CoverageRow:
    entry: DocEntry
    checkbox: str
    status: str
    modules: str
    note: str


@dataclass(frozen=True)
class PluginSignals:
    flax_exact: dict[str, set[str]]
    flax_tokens: dict[str, set[str]]
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


def fetch_doc_entries(doc_url: str) -> list[DocEntry]:
    request = Request(  # noqa: S310 (trusted docs URL)
        doc_url,
        headers={"User-Agent": "jax2onnx-flax-coverage-script/1.0"},
    )
    with urlopen(request, timeout=30) as response:
        html_text = response.read().decode("utf-8", errors="ignore")

    seen: dict[str, DocEntry] = {}
    for match in DOC_ENTRY_RE.finditer(html_text):
        href = html.unescape(match.group("href"))
        anchor = html.unescape(match.group("anchor"))
        label = html.unescape(match.group("label")).strip()

        path = href.split("#", 1)[0]
        if not anchor.startswith("flax."):
            continue
        if anchor in seen:
            continue
        seen[anchor] = DocEntry(name=anchor, doc_path=path, label=label)

    entries = sorted(seen.values(), key=lambda e: e.name)
    if not entries:
        raise RuntimeError(f"No Flax API entries found in '{doc_url}'.")
    return entries


def collect_plugin_signals(plugin_root: Path) -> PluginSignals:
    flax_exact: DefaultDict[str, set[str]] = defaultdict(set)
    flax_tokens: DefaultDict[str, set[str]] = defaultdict(set)
    global_tokens: DefaultDict[str, set[str]] = defaultdict(set)

    for path in sorted(plugin_root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        module = _module_id(path)
        is_flax = path.is_relative_to(FLAX_PLUGIN_ROOT)

        for key in _token_keys(path.stem):
            global_tokens[key].add(module)
            if is_flax:
                flax_tokens[key].add(module)

        for jax_doc in JAX_DOC_URL_RE.findall(text):
            doc_name = _doc_name_from_url(jax_doc)
            if doc_name is None:
                continue
            for key in _token_keys(doc_name):
                global_tokens[key].add(module)
                if is_flax:
                    flax_tokens[key].add(module)
            if is_flax and doc_name.startswith("flax."):
                flax_exact[doc_name].add(module)

        for lit in JAXPR_LITERAL_RE.findall(text):
            prim_leaf = lit.split(".")[-1].replace("-", "_")
            for key in _token_keys(prim_leaf):
                global_tokens[key].add(module)
                if is_flax:
                    flax_tokens[key].add(module)

        for name_expr in JAXPR_NAME_RE.findall(text):
            prim_leaf = name_expr.split(".")[-1]
            for key in _token_keys(prim_leaf):
                global_tokens[key].add(module)
                if is_flax:
                    flax_tokens[key].add(module)

        for component in COMPONENT_RE.findall(text):
            for key in _token_keys(component):
                global_tokens[key].add(module)
                if is_flax:
                    flax_tokens[key].add(module)

    return PluginSignals(
        flax_exact=dict(flax_exact),
        flax_tokens=dict(flax_tokens),
        global_tokens=dict(global_tokens),
    )


def _is_in_scope(entry: DocEntry) -> bool:
    return entry.doc_path.startswith(IN_SCOPE_PATH_PREFIXES)


def _is_composite_helper(entry: DocEntry) -> bool:
    if entry.doc_path.endswith(COMPOSITE_PATH_SUFFIXES):
        return True
    if ".initializers." in entry.name:
        return True
    if ".nn.dtypes." in entry.name:
        return True
    parts = entry.name.split(".")
    if len(parts) >= 2 and parts[-1] in {"create", "apply_gradients"}:
        return True
    if len(parts) >= 3 and parts[-2][:1].isupper() and parts[-1][:1].islower():
        return True
    return False


def _join_modules(modules: set[str]) -> str:
    return ", ".join(sorted(modules)[:3]) if modules else "-"


def _coverage_for_entry(entry: DocEntry, *, signals: PluginSignals) -> CoverageRow:
    token_keys = set(_token_keys(entry.name))
    alias_keys = {ALIAS_KEY_MAP[k] for k in token_keys if k in ALIAS_KEY_MAP}

    if not _is_in_scope(entry):
        return CoverageRow(
            entry=entry,
            checkbox="x",
            status="out_of_scope",
            modules="-",
            note="Outside neural-module surface; no dedicated converter plugin expected.",
        )

    if _is_composite_helper(entry):
        return CoverageRow(
            entry=entry,
            checkbox="x",
            status="composite",
            modules="-",
            note="Helper/config API; no standalone ONNX plugin expected.",
        )

    direct_modules = set(signals.flax_exact.get(entry.name, set()))
    for key in token_keys:
        direct_modules |= set(signals.flax_tokens.get(key, set()))
    for key in alias_keys:
        direct_modules |= set(signals.flax_tokens.get(key, set()))

    if direct_modules:
        return CoverageRow(
            entry=entry,
            checkbox="x",
            status="covered",
            modules=_join_modules(direct_modules),
            note="Direct Flax plugin signal (jax_doc/jaxpr/component).",
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
            note="Covered indirectly via lower-level JAX plugin signals.",
        )

    return CoverageRow(
        entry=entry,
        checkbox=" ",
        status="missing",
        modules="-",
        note="Missing dedicated Flax plugin coverage.",
    )


def build_rows(entries: list[DocEntry], *, signals: PluginSignals) -> list[CoverageRow]:
    return [_coverage_for_entry(entry, signals=signals) for entry in entries]


def render_markdown(rows: list[CoverageRow], *, doc_url: str, title: str) -> str:
    total = len(rows)
    in_scope = sum(1 for r in rows if _is_in_scope(r.entry))
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
        "- Source list: code-annotated API entries linked from Flax API index:"
        f" `{doc_url}`"
    )
    lines.append(
        "- Coverage signal: `jax_doc`, `jaxpr_primitive`, and `component` metadata in `jax2onnx/plugins/**/*.py`."
    )
    lines.append("")
    lines.append("## Snapshot")
    lines.append(f"- Total API entries discovered: `{total}`")
    lines.append(
        f"- In-scope neural entries (`flax.linen/*`, `flax.nnx/nn/*`): `{in_scope}`"
    )
    lines.append(f"- Covered (direct Flax plugin): `{covered}`")
    lines.append(f"- Covered (indirect via lower-level plugins): `{covered_indirect}`")
    lines.append(f"- Composite/helper entries: `{composite}`")
    lines.append(f"- Out-of-scope entries: `{out_of_scope}`")
    lines.append(f"- Missing dedicated Flax coverage: `{missing}`")
    lines.append("")
    lines.append("## Full Checklist")
    lines.append(
        "Legend: `covered`, `covered_indirect`, `composite`, `out_of_scope`, `missing`."
    )
    lines.append("")
    lines.append("| Checklist | Flax API Entry | Status | Modules (signals) | Notes |")
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
        description="Generate Flax API coverage checklist from docs + plugin signals."
    )
    parser.add_argument("--doc-url", default=DEFAULT_DOC_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mkdocs-output", type=Path, default=DEFAULT_MKDOCS_OUTPUT)
    args = parser.parse_args()

    entries = fetch_doc_entries(args.doc_url)
    signals = collect_plugin_signals(PLUGIN_ROOT)
    rows = build_rows(entries, signals=signals)

    work_notes_content = render_markdown(
        rows,
        doc_url=args.doc_url,
        title="Work Notes: Flax API Coverage Checklist",
    )
    args.output.write_text(work_notes_content + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {args.output}")

    mkdocs_content = render_markdown(
        rows,
        doc_url=args.doc_url,
        title="Flax API Coverage Checklist",
    )
    if args.mkdocs_output != args.output:
        args.mkdocs_output.write_text(mkdocs_content + "\n", encoding="utf-8")
        print(f"Wrote {len(rows)} rows to {args.mkdocs_output}")


if __name__ == "__main__":
    main()
