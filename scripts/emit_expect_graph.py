#!/usr/bin/env python3
# scripts/emit_expect_graph.py

"""Generate an expect_graph(...) snippet for a plugin testcase."""

from __future__ import annotations

import argparse
import pprint
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Sequence, Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import onnx

from jax2onnx.plugins._post_check_onnx_graph import auto_expect_graph_spec
from jax2onnx.user_interface import to_onnx
from tests import t_generator as tgen


def _materialize_callable(entry: dict) -> Callable[..., Any]:
    call = entry.get("callable")
    factory = entry.get("callable_factory")
    if call is not None:
        return call
    if factory is not None:
        return factory(jnp.float32)
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
    fn = _materialize_callable(entry)
    inputs = _build_inputs(entry)
    input_params = entry.get("input_params", {})
    opset = entry.get("opset_version", 21)
    enable_double_precision = bool(entry.get("enable_double_precision", False))

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
        return auto_expect_graph_spec(model)
    finally:
        if keep_dir is None and cleanup_path is not None and cleanup_path.exists():
            cleanup_path.unlink()
        if keep_dir is None and model_path != str(onnx_path):
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
