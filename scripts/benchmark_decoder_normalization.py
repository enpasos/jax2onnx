#!/usr/bin/env python3
# scripts/benchmark_decoder_normalization.py

"""Compare CPU decoder latency and JAX parity for normalization export modes.

Run from the repository root, for example::

    poetry run python scripts/benchmark_decoder_normalization.py \
        --iterations 1000 --repeats 7 --output /tmp/decoder-normalization.json

The benchmark reuses one deterministic decoder (or standalone RMSNorm) and
identical inputs for every export of the selected dtype. Each timing sample
is the mean of ``--iterations``
synchronous ``InferenceSession.run`` calls, including Python call overhead.
The reported warmed latency is the median of those samples. Session creation
and the first inference are timed separately; export, optimized-graph inspection,
JAX execution and warmup are excluded from warmed timing. Timing is informational,
with no pass/fail performance threshold; run on an otherwise idle machine.
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
from jax import export as jax_export
import jax.numpy as jnp
from flax import nnx
import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]

from jax2onnx import to_onnx
from jax2onnx.plugins.examples.nnx.transformer_decoder_with_sequential import (
    TransformerDecoder,
)
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    with_requested_dtype,
    with_rng_seed,
)


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


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
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
    floor = float(np.finfo(reference.dtype).eps * max(1.0, max_abs_reference))
    return {
        "max_absolute_error": float(np.max(difference)),
        "relative_linf_error": float(np.max(difference))
        / max(max_abs_reference, floor),
        "max_pointwise_relative_error": float(
            np.max(difference / np.maximum(np.abs(reference), floor))
        ),
        "pointwise_relative_denominator_floor": floor,
    }


def _input_specs(
    inputs: list[np.ndarray], shape_mode: str
) -> list[jax.ShapeDtypeStruct]:
    if shape_mode == "static":
        shapes = [value.shape for value in inputs]
    else:
        # Share B between decoder and encoder, but keep their lengths independent.
        scope = jax_export.SymbolicScope()
        shapes = [
            jax_export.symbolic_shape(
                f"B, {'H' if index == 0 else 'X'}, {value.shape[-1]}", scope=scope
            )
            for index, value in enumerate(inputs)
        ]
    return [
        jax.ShapeDtypeStruct(shape, value.dtype) for shape, value in zip(shapes, inputs)
    ]


def _session_options(intra_op_threads: int) -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.intra_op_num_threads = intra_op_threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return options


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--iterations", type=_positive_int, default=1000)
    parser.add_argument("--warmup", type=_positive_int, default=50)
    parser.add_argument("--repeats", type=_positive_int, default=7)
    parser.add_argument("--batch-size", type=_positive_int, default=2)
    parser.add_argument("--decoder-length", type=_positive_int, default=8)
    parser.add_argument("--encoder-length", type=_positive_int, default=4)
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--workload", choices=("decoder", "rmsnorm"), default="decoder")
    parser.add_argument(
        "--embed-dim",
        type=_positive_int,
        default=16,
        help="Decoder embedding width or standalone RMSNorm width",
    )
    parser.add_argument("--num-heads", type=_positive_int, default=4)
    parser.add_argument(
        "--ff-dim",
        type=_positive_int,
        default=None,
        help="Decoder feed-forward width (default: twice --embed-dim)",
    )
    parser.add_argument(
        "--intra-op-threads",
        type=_nonnegative_int,
        default=1,
        help="ORT intra-op threads; 0 selects ORT's default",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args(argv)
    if args.workload == "decoder" and args.embed_dim % args.num_heads:
        parser.error("--embed-dim must be divisible by --num-heads for the decoder")
    ff_dim = args.ff_dim or 2 * args.embed_dim
    dtype = np.dtype(args.dtype)
    parity_rtol = parity_atol = 2e-3 if dtype == np.float16 else 1e-4

    jax.config.update("jax_platforms", "cpu")
    jax.config.update("jax_enable_x64", False)
    input_names: tuple[str, ...]
    lengths: tuple[int, ...]
    if args.workload == "decoder":
        factory = construct_and_call(
            TransformerDecoder,
            num_layers=1,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            ff_dim=ff_dim,
            attention_dropout=0.5,
            encoder_attention_dropout=0.5,
            dtype=with_requested_dtype(),
            param_dtype=with_requested_dtype(),
            rngs=with_rng_seed(0),
        )
        input_names = ("decoder_input", "encoder_output")
        lengths = (args.decoder_length, args.encoder_length)
    else:
        factory = construct_and_call(
            nnx.RMSNorm,
            args.embed_dim,
            epsilon=1e-5,
            dtype=with_requested_dtype(),
            param_dtype=with_requested_dtype(),
            rngs=with_rng_seed(0),
        )
        input_names = ("input",)
        lengths = (args.decoder_length,)
    fn = factory.with_dtype(jnp.dtype(dtype)).instantiate()
    rng = np.random.default_rng(1)
    inputs = [
        rng.standard_normal((args.batch_size, length, args.embed_dim)).astype(dtype)
        for length in lengths
    ]
    # The decoder's deterministic=True default disables dropout for every run.
    reference = np.asarray(fn(*(jnp.asarray(value) for value in inputs)))
    if reference.dtype != dtype:
        raise ValueError(f"JAX output dtype {reference.dtype} does not match {dtype}")
    if not np.all(np.isfinite(reference)):
        raise ValueError("The JAX reference contains non-finite values")
    feeds = dict(zip(input_names, inputs))
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
            "intra_op_num_threads": args.intra_op_threads,
            "inter_op_num_threads": 1,
        },
        "model": {
            "example": (
                "tiny_decoder_with_sequential"
                if args.workload == "decoder"
                else "nnx_rmsnorm"
            ),
            "workload": args.workload,
            "model_seed": 0,
            "input_seed": 1,
            "input_shapes": [list(value.shape) for value in inputs],
            "dtype": args.dtype,
            "parameter_dtype": args.dtype,
            "reference_output_dtype": str(reference.dtype),
            "opset": 23,
            "deterministic": True,
            "num_layers": 1 if args.workload == "decoder" else None,
            "embed_dim": args.embed_dim,
            "num_heads": args.num_heads if args.workload == "decoder" else None,
            "ff_dim": ff_dim if args.workload == "decoder" else None,
        },
        "measurement": {
            "iterations_per_repeat": args.iterations,
            "warmup_calls_per_case": args.warmup,
            "repeats": args.repeats,
            "latency_scope": "synchronous session.run including Python overhead",
            "aggregation": "median of per-repeat mean milliseconds per call",
            "case_order": "rotate the first case each repeat",
            "reference": f"JAX CPU {args.dtype} with the same weights and inputs",
            "parity_rtol": parity_rtol,
            "parity_atol": parity_atol,
            "error_calculation_dtype": "float64",
            "relative_error_floor_dtype": args.dtype,
            "startup_samples_per_case": 1,
            "session_initialization_scope": (
                "InferenceSession creation from serialized bytes, including graph "
                "optimization and prepacking; excludes optimized-model file writes"
            ),
            "first_run_scope": "first synchronous session.run before warmup",
        },
        "cases": [],
    }
    cases: list[BenchmarkCase] = []
    # Optimized graphs are only needed to inspect fusion; never persist models.
    with tempfile.TemporaryDirectory(prefix="jax2onnx-decoder-benchmark-") as tmp:
        for shape_mode in ("static", "symbolic"):
            specs = _input_specs(inputs, shape_mode)
            for normalization_mode in ("auto", "prefer_native"):
                model = to_onnx(
                    fn,
                    specs,
                    opset=23,
                    input_names=list(feeds),
                    normalization_mode=normalization_mode,
                )
                serialized = model.SerializeToString()
                options = _session_options(args.intra_op_threads)
                started = time.perf_counter_ns()
                session = ort.InferenceSession(
                    serialized,
                    sess_options=options,
                    providers=["CPUExecutionProvider"],
                )
                session_initialization_ms = (time.perf_counter_ns() - started) / 1e6
                started = time.perf_counter_ns()
                output = session.run(None, feeds)[0]
                first_run_ms = (time.perf_counter_ns() - started) / 1e6
                if output.dtype != dtype:
                    raise ValueError(
                        f"ORT output dtype {output.dtype} does not match {dtype}"
                    )
                if not np.all(np.isfinite(output)):
                    raise ValueError(
                        f"Non-finite output for {shape_mode}/{normalization_mode}"
                    )
                np.testing.assert_allclose(
                    output, reference, rtol=parity_rtol, atol=parity_atol
                )
                # Inspect a separate session so file writes do not inflate startup.
                optimized_path = Path(tmp) / f"{shape_mode}-{normalization_mode}.onnx"
                inspection_options = _session_options(args.intra_op_threads)
                inspection_options.optimized_model_filepath = str(optimized_path)
                inspection_session = ort.InferenceSession(
                    serialized,
                    sess_options=inspection_options,
                    providers=["CPUExecutionProvider"],
                )
                del inspection_session
                case_report: dict[str, Any] = {
                    "shape_mode": shape_mode,
                    "normalization_mode": normalization_mode,
                    "exported_node_counts": _node_counts(model),
                    "optimized_node_counts": _node_counts(onnx.load(optimized_path)),
                    "errors_against_jax": _errors(output, reference),
                    "output_dtype": str(output.dtype),
                    "session_initialization_ms": session_initialization_ms,
                    "first_run_ms": first_run_ms,
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

    print(
        "shape      normalization     session ms    first ms    median ms    "
        "max abs error    relative L-inf"
    )
    for case in cases:
        case.report["median_ms"] = statistics.median(case.report["repeat_mean_ms"])
        report["cases"].append(case.report)
        errors = case.report["errors_against_jax"]
        print(
            f"{case.report['shape_mode']:<10} "
            f"{case.report['normalization_mode']:<16} "
            f"{case.report['session_initialization_ms']:>11.3f} "
            f"{case.report['first_run_ms']:>11.6f} "
            f"{case.report['median_ms']:>10.6f} "
            f"{errors['max_absolute_error']:>16.6g} "
            f"{errors['relative_linf_error']:>17.6g}"
        )
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
