#!/usr/bin/env python3
# scripts/emit_expect_graph.py

"""Generate an expect_graph(...) snippet for a plugin testcase."""

from __future__ import annotations

import argparse
import pprint
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Sequence, Any, Callable, Optional

import jax
import jax.numpy as jnp
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


def _materialize_callable(entry: dict, *, dtype: Any | None) -> Callable[..., Any]:
    call = entry.get("callable")
    factory = entry.get("callable_factory")
    if call is not None:
        if dtype is not None and hasattr(call, "with_dtype"):
            return call.with_dtype(dtype)
        return call
    if factory is not None:
        requested_dtype = dtype if dtype is not None else jnp.float32
        return factory(requested_dtype)
    raise ValueError(f"Entry '{entry.get('testcase')}' has no callable or factory")


def _shape_inputs(entry: dict) -> list:
    shapes = entry.get("input_shapes")
    if not shapes:
        return []
    dtypes = entry.get("input_dtypes")
    if dtypes:
        if len(dtypes) != len(shapes):
            raise ValueError(
                f"Input dtype/count mismatch for testcase '{entry.get('testcase')}'"
            )
        return [
            jax.ShapeDtypeStruct(tuple(shape), dt) for shape, dt in zip(shapes, dtypes)
        ]
    return [
        tuple(shape) if isinstance(shape, (list, tuple)) else shape for shape in shapes
    ]


def _value_inputs(entry: dict) -> list:
    values = entry.get("input_values")
    if not values:
        return []
    return [np.asarray(val) for val in values]


def _build_inputs(entry: dict) -> list:
    values = _value_inputs(entry)
    if values:
        return values
    return _shape_inputs(entry)


def _regenerate_spec(entry: dict, keep_dir: Path | None) -> dict:
    inputs = _build_inputs(entry)
    input_params = entry.get("input_params", {})
    opset = entry.get("opset_version", 21)
    enable_double_precision = bool(entry.get("enable_double_precision", False))
    requested_dtype = jnp.float64 if enable_double_precision else jnp.float32
    fn = _materialize_callable(entry, dtype=requested_dtype)

    model_name = entry.get("testcase", "spec")

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
        model_path = to_onnx(
            fn=fn,
            inputs=list(inputs),
            input_params=input_params,
            model_name=model_name,
            opset=opset,
            enable_double_precision=enable_double_precision,
            return_mode="file",
            output_path=str(onnx_path),
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


def _select_metadata(names: Sequence[str], entries: Iterable[dict]) -> dict[str, dict]:
    mapping = {entry["testcase"]: entry for entry in entries}
    unknown = [nm for nm in names if nm not in mapping]
    if unknown:
        raise SystemExit(f"Unknown testcase(s): {', '.join(unknown)}")
    return {nm: mapping[nm] for nm in names}


def generate_spec_for_testcase(name: str, keep_dir: Path | None = None) -> dict:
    entries = tgen.load_plugin_metadata()
    selected = _select_metadata([name], entries)
    meta = selected[name]
    return _regenerate_spec(meta, keep_dir)


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

    entries = tgen.load_plugin_metadata()
    selected = _select_metadata(args.testcase, entries)

    for name, meta in selected.items():
        snippet = _emit_for_testcase(name, meta, args.keep_onnx)
        sys.stdout.write(snippet)

    return 0


if __name__ == "__main__":
    sys.exit(main())
