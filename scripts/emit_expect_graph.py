#!/usr/bin/env python3
# scripts/emit_expect_graph.py

"""Generate an expect_graph(...) snippet for a plugin testcase."""

from __future__ import annotations

import argparse
import pprint
import sys
import tempfile
from pathlib import Path
from typing import Sequence, Any, Optional

import jax
import numpy as np
import onnx

from jax2onnx.plugins._post_check_onnx_graph import (
    auto_expect_graph_spec,
    _GraphView,
    DEFAULT_PASSTHROUGH_OPS,
    _match_path_on_graph,
    _parse_shape,
    _normalize_graph_filter,
    _nodes,
    _inputs_of,
    _extract_constant_array,
    _find_producer_node,
)
from jax2onnx.user_interface import to_onnx
from tests import t_generator as tgen


def _regenerate_spec(tp: dict, keep_dir: Path | None) -> dict:
    """Export the generated test ``tp`` as that test does and describe the graph."""
    model_name = tp["testcase"]

    if keep_dir is not None:
        keep_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = keep_dir / f"{model_name}.onnx"
        cleanup_path: Path | None = None
    else:
        tmp_file = tempfile.NamedTemporaryFile(
            suffix=".onnx", prefix=f"{model_name}_", delete=False
        )
        tmp_file.close()
        onnx_path = Path(tmp_file.name)
        cleanup_path = onnx_path

    model_path: str | None = None
    try:
        # The test generator owns the callable, the input specs and the export
        # options, so this exports exactly what the generated test checks.
        with jax.enable_x64(bool(tp.get("_jax_x64_test_setting", False))):
            fn = tp["callable"]
            if hasattr(fn, "instantiate"):
                fn = fn.instantiate()
            model_path = to_onnx(
                fn=fn,
                inputs=tgen.input_specs_for_test_param(tp, fn),
                return_mode="file",
                output_path=str(onnx_path),
                **tgen.to_onnx_options_for_test_param(tp),
            )
        model = onnx.load_model(model_path)
        spec = auto_expect_graph_spec(model)
        return _annotate_inputs_with_constants(model, spec)
    finally:
        if keep_dir is None and cleanup_path is not None and cleanup_path.exists():
            cleanup_path.unlink()
        if keep_dir is None and model_path and model_path != str(onnx_path):
            extra = Path(model_path)
            if extra.exists():
                extra.unlink()


def _format_positional(value) -> str:
    rep = pprint.pformat(value, width=80)
    rep = rep.replace("\n", "\n    ")
    return f"    {rep},"


def _format_keyword(name: str, value) -> str:
    rep = pprint.pformat(value, width=80)
    indent = " " * (len(name) + 6)
    rep = rep.replace("\n", "\n" + indent)
    return f"    {name}={rep},"


def _format_expect_graph_call(spec: dict) -> str:
    lines = ["expect_graph("]
    lines.append(_format_positional(spec["specs"]))

    if spec.get("symbols"):
        lines.append(_format_keyword("symbols", spec["symbols"]))
    if spec.get("mode", "all") != "all":
        lines.append(_format_keyword("mode", spec["mode"]))
    if spec.get("must_absent"):
        lines.append(_format_keyword("must_absent", spec["must_absent"]))
    if spec.get("counts"):
        lines.append(_format_keyword("counts", spec["counts"]))
    if spec.get("no_unused_inputs"):
        lines.append(_format_keyword("no_unused_inputs", spec["no_unused_inputs"]))
    if spec.get("no_unused_function_inputs"):
        lines.append(
            _format_keyword(
                "no_unused_function_inputs", spec["no_unused_function_inputs"]
            )
        )
    if spec.get("search_functions"):
        lines.append(_format_keyword("search_functions", spec["search_functions"]))
    if spec.get("attrs"):
        lines.append(_format_keyword("attrs", spec["attrs"]))

    lines.append(")")
    return "\n".join(lines)


def _annotate_inputs_with_constants(model, spec: dict) -> dict:
    gv = _GraphView(
        model,
        search_functions=spec.get("search_functions", False),
        passthrough_ops=DEFAULT_PASSTHROUGH_OPS,
    )

    annotated_specs: list[Any] = []

    for item in spec.get("specs", []):
        entry_dict = _normalize_spec_entry(item)
        inputs_map = _infer_constant_inputs_for_entry(gv, entry_dict, spec)
        if inputs_map:
            existing = entry_dict.get("inputs")
            if isinstance(existing, dict):
                merged = dict(existing)
                merged.update(inputs_map)
                entry_dict["inputs"] = merged
            else:
                entry_dict["inputs"] = inputs_map

        if set(entry_dict.keys()) == {"path"}:
            annotated_specs.append(entry_dict["path"])
        else:
            annotated_specs.append(entry_dict)

    new_spec = dict(spec)
    new_spec["specs"] = annotated_specs
    if new_spec.get("mode") == "all":
        new_spec.pop("mode", None)
    if new_spec.get("search_functions") is False:
        new_spec.pop("search_functions", None)
    return new_spec


def _normalize_spec_entry(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return dict(item)
    if isinstance(item, tuple) and len(item) == 2:
        path, options = item
        entry = dict(options)
        entry["path"] = path
        return entry
    return {"path": item}


def _infer_constant_inputs_for_entry(
    gv: _GraphView, entry: dict[str, Any], spec: dict[str, Any]
) -> dict[int, dict[str, Any]]:
    path = entry.get("path")
    if not isinstance(path, str):
        return {}

    tokens = [token.strip() for token in path.strip("^$ ").split("->") if token.strip()]
    steps: list[tuple[str, Optional[tuple]]]
    steps = []
    for tok in tokens:
        if ":" in tok:
            op, sh = tok.split(":", 1)
            steps.append((op.strip(), _parse_shape(sh)))
        else:
            steps.append((tok, None))

    symbol_env: dict[str, Any] = dict(spec.get("symbols", {}))
    symbol_env.update(entry.get("symbols", {}))

    graph_filter = entry.get("graph")
    allowed = _normalize_graph_filter(graph_filter)

    for gname, graph in gv.graphs:
        if allowed is not None and gname not in allowed:
            continue
        ok, _, matched = _match_path_on_graph(
            graph,
            steps,
            dict(symbol_env),
            gv.passthrough_ops,
            gname,
            gv._shape_index.get(gname, {}),
        )
        if not ok or not matched:
            continue
        nodes_seq = _nodes(graph)
        target_idx = matched[-1]
        if target_idx >= len(nodes_seq):
            continue
        node_obj = nodes_seq[target_idx]
        inputs = _inputs_of(node_obj)

        # Detect preceding Not for boolean edge
        if len(inputs) >= 3:
            not_node = _find_producer_node(nodes_seq, inputs[2])
            if not_node is not None and getattr(not_node, "op_type", "") == "Not":
                if not path.strip().startswith("Not"):
                    path = f"Not -> {path}"
                    entry["path"] = path

        inferred: dict[int, dict[str, Any]] = {}
        for idx, value in enumerate(inputs):
            arr = _extract_constant_array(value, nodes_seq, graph)
            if arr is None:
                continue
            arr_np = np.asarray(arr)
            if arr_np.size != 1:
                continue
            scalar = arr_np.reshape(())
            if arr_np.dtype.kind == "b":
                inferred[idx] = {"const_bool": bool(scalar)}
            else:
                rounded = float(np.round(np.asarray(scalar, dtype=np.float64), 6))
                inferred[idx] = {"const": rounded}
        if inferred:
            return inferred
    return {}


def _emit_for_testcase(name: str, meta: dict, keep_dir: Path | None) -> str:
    spec = _regenerate_spec(meta, keep_dir)
    call = _format_expect_graph_call(spec)
    return f"# {name}\n{call}\n"


def _generated_test_params() -> dict[str, dict]:
    """Parameters of every generated test, by test name.

    A testcase can expand to several tests, for example ``<name>_dynamic`` and
    ``<name>_f64``. Each of them is selectable. The authored name selects the
    test of that name, or the first variant when no test carries it.
    """
    params: dict[str, dict] = {}
    for entry in tgen.load_plugin_metadata():
        variants = tgen.generate_test_params(entry)
        for variant in variants:
            params.setdefault(variant["testcase"], variant)
        if variants:
            params.setdefault(entry["testcase"], variants[0])
    return params


def _select_test_params(names: Sequence[str]) -> dict[str, dict]:
    params = _generated_test_params()
    unknown = [name for name in names if name not in params]
    if unknown:
        raise SystemExit(f"Unknown testcase(s): {', '.join(unknown)}")
    return {name: params[name] for name in names}


def generate_spec_for_testcase(name: str, keep_dir: Path | None = None) -> dict:
    return _regenerate_spec(_select_test_params([name])[name], keep_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("testcase", nargs="+", help="Testcase name(s) to materialize")
    parser.add_argument(
        "--keep-onnx",
        metavar="DIR",
        type=Path,
        help="Optional directory to keep generated ONNX models for inspection.",
    )
    args = parser.parse_args(argv)

    for name, tp in _select_test_params(args.testcase).items():
        sys.stdout.write(_emit_for_testcase(name, tp, args.keep_onnx))

    return 0


if __name__ == "__main__":
    sys.exit(main())
