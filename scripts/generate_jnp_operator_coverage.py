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

from scripts._coverage_generation import write_or_check_generated

REPO_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_ROOT = REPO_ROOT / "jax2onnx" / "plugins"
JNP_PLUGIN_ROOT = PLUGIN_ROOT / "jax" / "numpy"
DEFAULT_OUTPUT = REPO_ROOT / "work_notes_coverage_jnp.md"
DEFAULT_MKDOCS_OUTPUT = REPO_ROOT / "docs" / "user_guide" / "jax_numpy_coverage.md"
DEFAULT_DOC_URL = "https://docs.jax.dev/en/latest/jax.numpy.html"

AUTOSUMMARY_RE = re.compile(r"_autosummary/jax\.numpy\.([A-Za-z0-9_.]+)\.html")
JAX_DOC_URL_RE = re.compile(r'jax_doc\s*=\s*"([^"]+)"')
JAX_DOC_NAME_RE = re.compile(r"jax\.numpy\.([A-Za-z0-9_.]+?)(?:\.html|$)")
JAXPR_LITERAL_RE = re.compile(r'jaxpr_primitive\s*=\s*"([A-Za-z0-9_.-]+)"')
JAXPR_NAME_RE = re.compile(r"jaxpr_primitive\s*=\s*([A-Za-z0-9_.]+)_p\.name")
COMPONENT_LITERAL_RE = re.compile(r'component\s*=\s*"([A-Za-z0-9_.-]+)"')

ALIAS_TO_OP = {
    "absolute": "abs",
    "arccos": "acos",
    "arccosh": "acosh",
    "arcsin": "asin",
    "arcsinh": "asinh",
    "arctan": "atan",
    "arctan2": "atan2",
    "arctanh": "atanh",
    "angle": "atan2",
    "argsort": "sort",
    "around": "round",
    "bitwise_count": "population_count",
    "bitwise_invert": "bitwise_not",
    "cbrt": "cbrt",
    "concat": "concatenate",
    "conjugate": "conj",
    "cumulative_prod": "cumprod",
    "cumulative_sum": "cumsum",
    "copy": "copy",
    "deg2rad": "mul",
    "degrees": "mul",
    "empty": "broadcast_in_dim",
    "empty_like": "broadcast_in_dim",
    "fft2": "fft",
    "fftn": "fft",
    "flip": "rev",
    "float_power": "pow",
    "full_like": "broadcast_in_dim",
    "hfft": "irfft",
    "hypot": "sqrt",
    "ifft2": "ifft",
    "ifftn": "ifft",
    "ihfft": "rfft",
    "identity": "iota",
    "imag": "imag",
    "inner": "dot_general",
    "irfft2": "irfft",
    "irfftn": "irfft",
    "iscomplex": "imag",
    "isinf": "eq",
    "isnan": "ne",
    "isreal": "imag",
    "log1p": "log1p",
    "log2": "log",
    "log10": "log",
    "logical_and": "and",
    "logical_not": "not",
    "logical_or": "or",
    "logical_xor": "xor",
    "kron": "mul",
    "matrix_transpose": "transpose",
    "mod": "rem",
    "multiply": "mul",
    "negative": "neg",
    "nextafter": "nextafter",
    "not_equal": "ne",
    "ones_like": "broadcast_in_dim",
    "permute_dims": "transpose",
    "pow": "power",
    "ptp": "reduce_max",
    "rad2deg": "mul",
    "radians": "mul",
    "ravel": "reshape",
    "real": "real",
    "reciprocal": "integer_pow",
    "remainder": "rem",
    "rint": "round",
    "round": "round",
    "round_": "round",
    "rfft2": "rfft",
    "rfftn": "rfft",
    "signbit": "shift_right_arithmetic",
    "sort_complex": "sort",
    "square": "square",
    "subtract": "sub",
    "swapaxes": "transpose",
    "tril": "triu",
    "true_divide": "div",
    "zeros_like": "broadcast_in_dim",
}

COMPOSITE_HELPERS = {
    "allclose",
    "append",
    "apply_along_axis",
    "apply_over_axes",
    "argwhere",
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
    "average",
    "bincount",
    "count_nonzero",
    "broadcast_arrays",
    "broadcast_shapes",
    "broadcast_to",
    "c_",
    "column_stack",
    "diag_indices",
    "diag_indices_from",
    "diagflat",
    "diff",
    "divmod",
    "dsplit",
    "dstack",
    "ediff1d",
    "expand_dims",
    "extract",
    "fmax",
    "fmin",
    "flatnonzero",
    "fliplr",
    "flipud",
    "frombuffer",
    "fromfunction",
    "fromiter",
    "fromstring",
    "heaviside",
    "hsplit",
    "hstack",
    "indices",
    "insert",
    "isclose",
    "isin",
    "isneginf",
    "isposinf",
    "ix_",
    "kaiser",
    "logaddexp",
    "logaddexp2",
    "meshgrid",
    "mgrid",
    "modf",
    "msort",
    "nan_to_num",
    "ndenumerate",
    "newaxis",
    "nonzero",
    "ogrid",
    "piecewise",
    "positive",
    "r_",
    "repeat",
    "resize",
    "rollaxis",
    "s_",
    "setdiff1d",
    "setxor1d",
    "slogdet",
    "stack",
    "take_along_axis",
    "trunc",
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
    "can_cast",
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
    "einsum_path",
    "finfo",
    "flatiter",
    "flexible",
    "float16",
    "float32",
    "float64",
    "float_",
    "floating",
    "from_dlpack",
    "fromfile",
    "frompyfunc",
    "generic",
    "get_printoptions",
    "half",
    "iinfo",
    "index_exp",
    "inexact",
    "int_",
    "int8",
    "int16",
    "int32",
    "int64",
    "integer",
    "iscomplexobj",
    "isdtype",
    "isrealobj",
    "isscalar",
    "issubsctype",
    "issubdtype",
    "iterable",
    "load",
    "longdouble",
    "ndim",
    "ndarray",
    "number",
    "object_",
    "printoptions",
    "promote_types",
    "result_type",
    "save",
    "savez",
    "set_printoptions",
    "signedinteger",
    "single",
    "timedelta64",
    "uint",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "unsignedinteger",
    "ufunc",
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
) -> tuple[
    DefaultDict[str, set[str]],
    DefaultDict[str, set[str]],
    DefaultDict[str, set[str]],
]:
    doc_usage: DefaultDict[str, set[str]] = defaultdict(set)
    prim_usage: DefaultDict[str, set[str]] = defaultdict(set)
    component_usage: DefaultDict[str, set[str]] = defaultdict(set)

    for path in sorted(plugin_root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        module = _module_id(path)
        is_jnp_plugin = path.is_relative_to(JNP_PLUGIN_ROOT)

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

        if is_jnp_plugin:
            for component in COMPONENT_LITERAL_RE.findall(text):
                component_usage[component].add(module)
                component_usage[component.replace("-", "_")].add(module)
                component_usage[component.replace("_", "-")].add(module)

    return doc_usage, prim_usage, component_usage


def _usage_for_key(key: str, usage: dict[str, set[str]]) -> set[str]:
    return (
        set(usage.get(key, set()))
        | set(usage.get(key.split(".")[-1], set()))
        | set(usage.get(key.replace("-", "_"), set()))
        | set(usage.get(key.replace("_", "-"), set()))
    )


def _status_for_op(
    op: str,
    *,
    doc_usage: dict[str, set[str]],
    prim_usage: dict[str, set[str]],
    component_usage: dict[str, set[str]],
) -> tuple[str, str, str]:
    base = op.split(".")[-1]
    modules = (
        _usage_for_key(op, doc_usage)
        | _usage_for_key(base, doc_usage)
        | _usage_for_key(op, component_usage)
        | _usage_for_key(base, component_usage)
    )
    modules_text = ", ".join(sorted(modules)[:3]) if modules else "-"

    if modules:
        return (
            "covered",
            modules_text,
            "Direct plugin coverage via `jax_doc` or `component` metadata.",
        )

    alias = ALIAS_TO_OP.get(base)
    if alias:
        alias_modules = (
            _usage_for_key(alias, doc_usage)
            | _usage_for_key(alias, component_usage)
            | _usage_for_key(alias, prim_usage)
        )
    else:
        alias_modules = set()

    if alias and alias_modules:
        alias_modules_text = (
            ", ".join(sorted(alias_modules)[:3]) if alias_modules else "-"
        )
        return (
            "covered_indirect",
            alias_modules_text,
            f"Covered via alias or lower-level primitive `{alias}`.",
        )

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
    component_usage: dict[str, set[str]],
) -> list[CoverageRow]:
    rows: list[CoverageRow] = []
    for op in ops:
        status, modules_text, note = _status_for_op(
            op,
            doc_usage=doc_usage,
            prim_usage=prim_usage,
            component_usage=component_usage,
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

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## Scope")
    lines.append(
        "- Source list: all autosummary entries linked on the official `jax.numpy` page:"
        f" `{doc_url}`"
    )
    lines.append(
        "- Coverage signal: `jax_doc`/`component` metadata + `jaxpr_primitive` registrations in `jax2onnx/plugins/**/*.py`."
    )
    lines.append("")
    lines.append("## Snapshot")
    lines.append(f"- Total docs entries: `{total}`")
    lines.append(f"- Covered (direct plugin): `{covered}`")
    lines.append(f"- Covered (via alias/indirect signal): `{covered_indirect}`")
    lines.append(f"- Composite/helper entries: `{composite}`")
    lines.append(f"- Non-functional entries (dtype/type/constants): `{non_functional}`")
    lines.append(f"- Missing dedicated plugin coverage: `{missing}`")
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
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate jax.numpy coverage checklist from docs + plugin signals."
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
    doc_usage, prim_usage, component_usage = collect_plugin_signals(PLUGIN_ROOT)
    rows = build_rows(
        ops,
        doc_usage=doc_usage,
        prim_usage=prim_usage,
        component_usage=component_usage,
    )

    work_notes_content = render_markdown(
        rows,
        doc_url=args.doc_url,
        title="Work Notes: JAX NumPy Coverage Checklist",
    )

    mkdocs_content = render_markdown(
        rows,
        doc_url=args.doc_url,
        title="JAX NumPy Coverage Checklist",
    )
    if args.check:
        write_or_check_generated(
            args.mkdocs_output,
            mkdocs_content + "\n",
            check=True,
            label="JAX NumPy coverage page",
        )
        return

    write_or_check_generated(
        args.output,
        work_notes_content + "\n",
        check=False,
        label=f"JAX NumPy work notes coverage page ({len(rows)} rows)",
    )
    if args.mkdocs_output != args.output:
        write_or_check_generated(
            args.mkdocs_output,
            mkdocs_content + "\n",
            check=False,
            label=f"JAX NumPy MkDocs coverage page ({len(rows)} rows)",
        )


if __name__ == "__main__":
    main()
