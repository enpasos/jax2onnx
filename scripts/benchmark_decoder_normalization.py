#!/usr/bin/env python3
# scripts/benchmark_decoder_normalization.py

"""Compare CPU decoder latency and JAX parity for normalization export modes.

Run from the repository root, for example::

    poetry run python scripts/benchmark_decoder_normalization.py \
        --iterations 1000 --repeats 7 --output /tmp/decoder-normalization.json

The benchmark reuses one deterministic tiny decoder and identical float32
inputs for every export. Each timing sample is the mean of ``--iterations``
synchronous ``InferenceSession.run`` calls, including Python call overhead.
The reported latency is the median of those samples. Export, session creation,
JAX execution and warmup are excluded. Timing is informational, with no pass/fail
performance threshold; run on an otherwise idle machine for useful comparisons.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile
import time
import tomllib
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort

from jax2onnx import to_onnx
from jax2onnx.plugins.examples.nnx.transformer_decoder_with_sequential import (
    TransformerDecoder,
)
from jax2onnx.plugins.plugin_system import construct_and_call, with_rng_seed


@dataclass
class BenchmarkCase:
    session: ort.InferenceSession
    feeds: dict[str, np.ndarray]
    report: dict[str, Any]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or platform.machine()


def _source_details() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True
    )
    return {
        "project_version": project["version"],
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "git_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def _node_counts(model: onnx.ModelProto) -> dict[str, int]:
    """Count operators in the graph, subgraphs and local function definitions."""
    counts: Counter[str] = Counter()

    def visit(nodes: Any) -> None:
        for node in nodes:
            domain = f"{node.domain}::" if node.domain else ""
            counts[f"{domain}{node.op_type}"] += 1
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    visit(attribute.g.node)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for graph in attribute.graphs:
                        visit(graph.node)

    visit(model.graph.node)
    for function in model.functions:
        visit(function.node)
    return dict(sorted(counts.items()))


def _errors(output: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    # Compute errors in float64, without implying the JAX reference is float64.
    difference = np.abs(output.astype(np.float64) - reference.astype(np.float64))
    max_abs_reference = float(np.max(np.abs(reference)))
    floor = float(np.finfo(np.float32).eps * max(1.0, max_abs_reference))
    return {
        "max_absolute_error": float(np.max(difference)),
        "relative_linf_error": float(np.max(difference))
        / max(max_abs_reference, floor),
        "max_pointwise_relative_error": float(
            np.max(difference / np.maximum(np.abs(reference), floor))
        ),
        "pointwise_relative_denominator_floor": floor,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--iterations", type=_positive_int, default=1000)
    parser.add_argument("--warmup", type=_positive_int, default=50)
    parser.add_argument("--repeats", type=_positive_int, default=7)
    parser.add_argument("--batch-size", type=_positive_int, default=2)
    parser.add_argument("--decoder-length", type=_positive_int, default=8)
    parser.add_argument("--encoder-length", type=_positive_int, default=4)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()

    jax.config.update("jax_platforms", "cpu")
    jax.config.update("jax_enable_x64", False)
    decoder = (
        construct_and_call(
            TransformerDecoder,
            num_layers=1,
            embed_dim=16,
            num_heads=4,
            ff_dim=32,
            attention_dropout=0.5,
            encoder_attention_dropout=0.5,
            rngs=with_rng_seed(0),
        )
        .with_dtype(jnp.float32)
        .instantiate()
    )
    rng = np.random.default_rng(1)
    inputs = [
        rng.standard_normal((args.batch_size, args.decoder_length, 16)).astype(
            np.float32
        ),
        rng.standard_normal((args.batch_size, args.encoder_length, 16)).astype(
            np.float32
        ),
    ]
    # The decoder's deterministic=True default disables dropout for every run.
    reference = np.asarray(decoder(*(jnp.asarray(value) for value in inputs)))
    if not np.all(np.isfinite(reference)):
        raise ValueError("The JAX reference contains non-finite values")
    feeds = dict(zip(("decoder_input", "encoder_output"), inputs))
    report: dict[str, Any] = {
        "environment": {
            "cpu": _cpu_model(),
            "source": _source_details(),
            "platform": platform.platform(),
            "cpu_affinity": (
                sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None
            ),
            "python": platform.python_version(),
            "versions": {
                package: version(package)
                for package in (
                    "jax",
                    "jaxlib",
                    "flax",
                    "numpy",
                    "onnx",
                    "onnx-ir",
                    "onnxruntime",
                )
            },
            "execution_provider": "CPUExecutionProvider",
            "graph_optimization_level": "ORT_ENABLE_ALL",
            "execution_mode": "ORT_SEQUENTIAL",
            "intra_op_num_threads": 1,
            "inter_op_num_threads": 1,
        },
        "model": {
            "example": "tiny_decoder_with_sequential",
            "model_seed": 0,
            "input_seed": 1,
            "input_shapes": [list(value.shape) for value in inputs],
            "dtype": "float32",
            "opset": 23,
            "deterministic": True,
            "num_layers": 1,
            "embed_dim": 16,
            "num_heads": 4,
            "ff_dim": 32,
        },
        "measurement": {
            "iterations_per_repeat": args.iterations,
            "warmup_calls_per_case": args.warmup,
            "repeats": args.repeats,
            "latency_scope": "synchronous session.run including Python overhead",
            "aggregation": "median of per-repeat mean milliseconds per call",
            "case_order": "rotate the first case each repeat",
            "reference": "JAX CPU float32 with the same weights and inputs",
            "parity_rtol": 1e-4,
            "parity_atol": 1e-4,
        },
        "cases": [],
    }
    cases: list[BenchmarkCase] = []
    # Optimized graphs are only needed to inspect fusion; never persist models.
    with tempfile.TemporaryDirectory(prefix="jax2onnx-decoder-benchmark-") as tmp:
        for shape_mode in ("static", "symbolic"):
            specs = (
                [value.shape for value in inputs]
                if shape_mode == "static"
                else [("B", "H", 16), ("B", "X", 16)]
            )
            for normalization_mode in ("auto", "prefer_native"):
                model = to_onnx(
                    decoder,
                    specs,
                    opset=23,
                    input_names=list(feeds),
                    normalization_mode=normalization_mode,
                )
                options = ort.SessionOptions()
                options.intra_op_num_threads = 1
                options.inter_op_num_threads = 1
                options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
                optimized_path = Path(tmp) / f"{shape_mode}-{normalization_mode}.onnx"
                options.optimized_model_filepath = str(optimized_path)
                session = ort.InferenceSession(
                    model.SerializeToString(),
                    sess_options=options,
                    providers=["CPUExecutionProvider"],
                )
                output = session.run(None, feeds)[0]
                if not np.all(np.isfinite(output)):
                    raise ValueError(
                        f"Non-finite output for {shape_mode}/{normalization_mode}"
                    )
                np.testing.assert_allclose(output, reference, rtol=1e-4, atol=1e-4)
                case_report = {
                    "shape_mode": shape_mode,
                    "normalization_mode": normalization_mode,
                    "exported_node_counts": _node_counts(model),
                    "optimized_node_counts": _node_counts(onnx.load(optimized_path)),
                    "errors_against_jax": _errors(output, reference),
                    "repeat_mean_ms": [],
                }
                cases.append(BenchmarkCase(session, feeds, case_report))

        for case in cases:
            for _ in range(args.warmup):
                case.session.run(None, case.feeds)
        for repeat in range(args.repeats):
            offset = repeat % len(cases)
            for case in cases[offset:] + cases[:offset]:
                started = time.perf_counter_ns()
                for _ in range(args.iterations):
                    case.session.run(None, case.feeds)
                elapsed_ms = (time.perf_counter_ns() - started) / 1e6 / args.iterations
                case.report["repeat_mean_ms"].append(elapsed_ms)

    print("shape      normalization    median ms    max abs error    relative L-inf")
    for case in cases:
        case.report["median_ms"] = statistics.median(case.report["repeat_mean_ms"])
        report["cases"].append(case.report)
        errors = case.report["errors_against_jax"]
        print(
            f"{case.report['shape_mode']:<10} "
            f"{case.report['normalization_mode']:<16} "
            f"{case.report['median_ms']:>10.6f} "
            f"{errors['max_absolute_error']:>16.6g} "
            f"{errors['relative_linf_error']:>17.6g}"
        )
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
