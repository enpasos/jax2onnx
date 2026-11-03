#!/usr/bin/env python3
# scripts/run_quantized_mnist.py

"""Evaluate one frozen MNIST ONNX artifact in a NumPy/ORT-only environment.

Run each model and optimization level in a separate fresh process. For example:
  env -u PYTHONPATH /tmp/mnist-runtime/bin/python -I run_quantized_mnist.py \
      --model /tmp/mnist/fp32.onnx --variant fp32 \
      --data /tmp/mnist/mnist.npz --optimization off --output /tmp/fp32-off.json

Only the official, checksummed 10,000-image test set is accepted. This script
does not train, export or download data. After four separate runtime processes,
use --compare-results DIRECTORY --output summary.json to compare their evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any


# https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
MNIST_SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
RUNTIME_PINS = {"numpy": "2.4.6", "onnxruntime": "1.31.1"}
TRAINING_PACKAGES = ("jax", "jaxlib", "flax", "qwix", "jax2onnx")
NATIVE_MODES = ("eager", "fresh_jit", "warm_jit")
WARMUP_REPETITIONS = 100
TIMED_REPETITIONS = 1000
RTOL = 1e-6
ATOL = 1e-6


def _current_rss_bytes() -> int | None:
    """Read current resident memory, rather than the process high-water mark."""
    status = Path("/proc/self/status")
    if not status.exists():
        return None
    for line in status.read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return None


def _peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _check_runtime_environment() -> dict[str, Any]:
    if os.environ.get("PYTHONPATH"):
        raise RuntimeError("Unset PYTHONPATH before running the independent runtime.")
    absent = {}
    for name in TRAINING_PACKAGES:
        try:
            importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            distribution_absent = True
        else:
            distribution_absent = False
        module_absent = importlib.util.find_spec(name) is None
        absent[name] = {
            "distribution_absent": distribution_absent,
            "module_absent": module_absent,
        }
        if not distribution_absent or not module_absent:
            raise RuntimeError(
                f"{name} is installed or importable; use the clean runtime environment."
            )
    versions = {name: importlib.metadata.version(name) for name in sorted(RUNTIME_PINS)}
    if versions != RUNTIME_PINS:
        raise RuntimeError(
            f"Runtime versions {versions} do not match pinned versions {RUNTIME_PINS}."
        )
    return {
        "training_packages": absent,
        "pythonpath_unset_or_empty": True,
        "python_isolated_mode": bool(sys.flags.isolated),
        "versions": versions,
        "installed_distributions": {
            distribution.metadata["Name"]: distribution.version
            for distribution in importlib.metadata.distributions()
        },
    }


def _validate_logits(logits: Any, shape: tuple[int, int], description: str) -> None:
    import numpy as np

    if logits.shape != shape or logits.dtype != np.dtype("float32"):
        raise ValueError(
            f"{description}: expected float32 {shape}, got {logits.dtype} {logits.shape}."
        )
    if not np.isfinite(logits).all():
        raise ValueError(f"{description}: logits contain NaN or infinity.")


def _error_count(logits: Any, labels: Any) -> int:
    import numpy as np

    return int(np.count_nonzero(logits.argmax(axis=1) != labels))


def _compare_logits(actual: Any, expected: Any) -> dict[str, Any]:
    import numpy as np

    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return {
        "max_abs_error": float(difference.max()),
        "within_w8a8_tolerance": bool(
            np.allclose(actual, expected, rtol=RTOL, atol=ATOL)
        ),
        "identical_predictions": bool(
            np.array_equal(actual.argmax(axis=1), expected.argmax(axis=1))
        ),
    }


def _native_validation(
    path: Path, variant: str, actual: Any, labels: Any
) -> dict[str, Any]:
    import numpy as np

    references = {}
    with np.load(path, allow_pickle=False) as archive:
        if not np.array_equal(archive["labels"], labels):
            raise ValueError(
                "Native reference labels differ from the canonical test set."
            )
        for mode in NATIVE_MODES:
            for name in (f"fp32_{mode}", f"{variant}_{mode}"):
                references[name] = archive[name]
                _validate_logits(references[name], (10000, 10), name)
    comparisons = {}
    errors = {}
    passed = True
    for mode in NATIVE_MODES:
        name = f"{variant}_{mode}"
        comparisons[name] = _compare_logits(actual, references[name])
        errors[name] = _error_count(references[name], labels)
        fp32_name = f"fp32_{mode}"
        errors[fp32_name] = _error_count(references[fp32_name], labels)
        passed = passed and errors[fp32_name] <= 300
        if variant == "w8a8":
            comparison = comparisons[name]
            passed = (
                passed
                and comparison["within_w8a8_tolerance"]
                and comparison["identical_predictions"]
                and errors[name] - errors[fp32_name] <= 50
            )
    native_pairwise = {}
    if variant == "w8a8":
        for index, left in enumerate(NATIVE_MODES):
            for right in NATIVE_MODES[index + 1 :]:
                key = f"{left}_vs_{right}"
                comparison = _compare_logits(
                    references[f"w8a8_{left}"], references[f"w8a8_{right}"]
                )
                native_pairwise[key] = comparison
                passed = (
                    passed
                    and comparison["within_w8a8_tolerance"]
                    and comparison["identical_predictions"]
                )
    return {
        "status": "checked",
        "archive_sha256": _sha256(path),
        "errors": errors,
        "ort_vs_native": comparisons,
        "native_pairwise": native_pairwise,
        "rtol": RTOL,
        "atol": ATOL,
        "fp32_logit_tolerance_is_acceptance_gate": False,
        "checked_gates_passed": bool(passed),
    }


def _hardware() -> dict[str, Any]:
    cpu_model = None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.partition(":")[2].strip()
                break
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count(),
    }


def _compare_results(directory: Path, output: Path) -> int:
    """Check four serial run records against the frozen inline model files."""
    import numpy as np

    frozen = json.loads((directory / "export.json").read_text())
    native_sha = _sha256(directory / "native_outputs.npz")
    with np.load(directory / "native_outputs.npz", allow_pickle=False) as archive:
        labels = archive["labels"]
    if labels.shape != (10000,) or labels.dtype != np.dtype("uint8"):
        raise ValueError("Expected all 10000 uint8 native reference labels.")
    logits, errors, checks, sizes = {}, {}, {}, {}
    absent = {
        name: {"distribution_absent": True, "module_absent": True}
        for name in TRAINING_PACKAGES
    }
    for variant in ("fp32", "w8a8"):
        artifact = frozen["artifacts"][variant]
        model = directory / artifact["file"]
        sizes[variant] = model.stat().st_size
        actual_artifact = (sizes[variant], _sha256(model))
        for mode in ("off", "on"):
            key = f"{variant}-{mode}"
            report = json.loads((directory / f"{key}.json").read_text())
            with np.load(directory / f"{key}.npz", allow_pickle=False) as archive:
                logits[key] = archive["logits"]
                np.testing.assert_array_equal(archive["labels"], labels)
            _validate_logits(logits[key], (10000, 10), key)
            errors[key] = _error_count(logits[key], labels)
            native = _native_validation(
                directory / "native_outputs.npz", variant, logits[key], labels
            )
            checks[f"{key}_native_fidelity"] = native["checked_gates_passed"]
            checks[f"{key}_frozen_artifact"] = (
                actual_artifact
                == (artifact["bytes"], artifact["sha256"])
                == (report["artifact"]["bytes"], report["artifact"]["sha256"])
            )
            checks[f"{key}_runtime_evidence"] = (
                report["variant"] == variant
                and report["checked_gates_passed"] is True
                and report["native_validation"]["status"] == "checked"
                and report["native_validation"]["checked_gates_passed"] is True
                and report["native_validation"]["archive_sha256"] == native_sha
                and report["data"] == {"sha256": MNIST_SHA256, "test_examples": 10000}
                and report["batch_size"] == 1
                and report["environment"]["versions"] == RUNTIME_PINS
                and report["environment"]["training_packages"] == absent
                and report["environment"]["pythonpath_unset_or_empty"] is True
                and report["runtime"]["providers"] == ["CPUExecutionProvider"]
                and report["runtime"]["intra_op_threads"] == 1
                and report["runtime"]["inter_op_threads"] == 1
                and report["runtime"]["graph_optimization_level"]
                == ("ORT_DISABLE_ALL" if mode == "off" else "ORT_ENABLE_ALL")
                and report["accuracy"]["errors"] == errors[key]
            )
    extra_errors = {
        mode: errors[f"w8a8-{mode}"] - errors[f"fp32-{mode}"] for mode in ("off", "on")
    }
    checks["fp32_ort_accuracy"] = all(
        errors[f"fp32-{mode}"] <= 300 for mode in extra_errors
    )
    checks["w8a8_ort_accuracy_loss"] = all(
        value <= 50 for value in extra_errors.values()
    )
    comparison = _compare_logits(logits["w8a8-off"], logits["w8a8-on"])
    checks["w8a8_optimization_fidelity"] = (
        comparison["within_w8a8_tolerance"] and comparison["identical_predictions"]
    )
    ratio = sizes["w8a8"] / sizes["fp32"]
    checks["artifact_ratio"] = ratio <= 0.4
    summary = {
        "checks": checks,
        "errors": errors,
        "extra_ort_errors": extra_errors,
        "artifact_bytes": sizes,
        "artifact_ratio": ratio,
        "w8a8_optimization_comparison": comparison,
        "native_outputs_sha256": native_sha,
        "export_record_sha256": _sha256(directory / "export.json"),
        "checked_gates_passed": all(checks.values()),
        "scope": "runtime comparison; preparation/graph checks come from the frozen export record",
        "all_acceptance_gates_checked": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary))
    return 0 if summary["checked_gates_passed"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compare-results", type=Path, help="Compare four frozen run records."
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--variant", choices=("fp32", "w8a8"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--native-outputs", type=Path)
    parser.add_argument("--optimization", choices=("off", "on"))
    parser.add_argument("--output", required=True, type=Path, help="Metrics JSON path.")
    parser.add_argument("--logits", type=Path, help="Optional logits/labels NPZ path.")
    args = parser.parse_args()
    run_args = (args.model, args.variant, args.data, args.optimization)
    if args.compare_results is None and any(value is None for value in run_args):
        parser.error(
            "Inference requires --model, --variant, --data and --optimization."
        )
    if args.compare_results is not None and any(
        value is not None for value in (*run_args, args.native_outputs, args.logits)
    ):
        parser.error("Use --compare-results separately from inference options.")
    return args


def main() -> int:
    args = parse_args()
    before_imports_rss = _current_rss_bytes()
    environment = _check_runtime_environment()
    thread_environment = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    for name in thread_environment:
        os.environ[name] = "1"
    if args.compare_results is not None:
        return _compare_results(args.compare_results, args.output)

    import_start = time.perf_counter_ns()
    import numpy as np
    import onnxruntime as ort

    import_ms = (time.perf_counter_ns() - import_start) / 1e6
    memory = {
        "before_numpy_ort_imports": before_imports_rss,
        "after_numpy_ort_imports": _current_rss_bytes(),
    }
    if "CPUExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("ONNX Runtime CPUExecutionProvider is unavailable.")
    data_sha256 = _sha256(args.data)
    if data_sha256 != MNIST_SHA256:
        raise ValueError("--data is not the checksummed canonical MNIST NPZ archive.")
    model_bytes = args.model.read_bytes()
    artifact = {
        "path": str(args.model.resolve()),
        "bytes": len(model_bytes),
        "sha256": hashlib.sha256(model_bytes).hexdigest(),
        "format": "self-contained ONNX; loaded from bytes, no external data files",
    }
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    optimization = {
        "off": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "on": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }[args.optimization]
    options.graph_optimization_level = optimization
    memory["before_session_creation"] = _current_rss_bytes()
    session_start = time.perf_counter_ns()
    session = ort.InferenceSession(
        model_bytes, sess_options=options, providers=["CPUExecutionProvider"]
    )
    session_ms = (time.perf_counter_ns() - session_start) / 1e6
    del model_bytes
    memory["after_session_creation"] = _current_rss_bytes()
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError(f"Unexpected active providers: {session.get_providers()}")
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if (
        len(inputs) != 1
        or inputs[0].shape != [1, 784]
        or inputs[0].type != "tensor(float)"
        or len(outputs) != 1
        or outputs[0].shape != [1, 10]
        or outputs[0].type != "tensor(float)"
    ):
        raise ValueError(
            "Expected exactly one float32 (1,784) input and (1,10) output."
        )
    with np.load(args.data, allow_pickle=False) as archive:
        pixels = archive["x_test"]
        labels = archive["y_test"]
    if pixels.shape != (10000, 28, 28) or pixels.dtype != np.dtype("uint8"):
        raise ValueError("Expected the complete 10000x28x28 uint8 test images.")
    if labels.shape != (10000,) or labels.dtype != np.dtype("uint8"):
        raise ValueError("Expected the complete 10000 uint8 test labels.")

    def prepare(image: Any) -> Any:
        value = image.reshape(1, 784).astype(np.float32)
        value /= np.float32(255)
        return value

    first_feed = {inputs[0].name: prepare(pixels[0])}
    output_names = [outputs[0].name]
    memory["before_first_inference_with_uint8_data"] = _current_rss_bytes()
    first_start = time.perf_counter_ns()
    first_logits = session.run(output_names, first_feed)[0]
    first_ms = (time.perf_counter_ns() - first_start) / 1e6
    _validate_logits(first_logits, (1, 10), "First ORT inference")
    memory["after_first_inference"] = _current_rss_bytes()
    for _ in range(WARMUP_REPETITIONS):
        session.run(output_names, first_feed)
    durations_ns = np.empty(TIMED_REPETITIONS, dtype=np.int64)
    for index in range(TIMED_REPETITIONS):
        start = time.perf_counter_ns()
        session.run(output_names, first_feed)
        durations_ns[index] = time.perf_counter_ns() - start
    memory["after_warmed_inference"] = _current_rss_bytes()
    del first_feed

    logits = np.empty((10000, 10), dtype=np.float32)
    logits[0] = first_logits[0]
    evaluation_start = time.perf_counter_ns()
    for index in range(1, 10000):
        result = session.run(output_names, {inputs[0].name: prepare(pixels[index])})[0]
        _validate_logits(result, (1, 10), f"ORT inference {index}")
        logits[index] = result[0]
    evaluation_ms = (time.perf_counter_ns() - evaluation_start) / 1e6
    memory["after_complete_heldout_inference"] = _current_rss_bytes()
    memory["peak_through_heldout_inference"] = _peak_rss_bytes()
    errors = _error_count(logits, labels)
    native = (
        _native_validation(args.native_outputs, args.variant, logits, labels)
        if args.native_outputs is not None
        else {"status": "not_requested", "checked_gates_passed": None}
    )
    passed = (args.variant != "fp32" or errors <= 300) and native[
        "checked_gates_passed"
    ] is not False
    report = {
        "variant": args.variant,
        "artifact": artifact,
        "data": {"sha256": data_sha256, "test_examples": 10000},
        "preprocessing": "uint8 pixels -> float32 / float32(255), row-major (1,784)",
        "batch_size": 1,
        "accuracy": {
            "errors": errors,
            "correct": 10000 - errors,
            "fraction": (10000 - errors) / 10000,
            "percent": (10000 - errors) / 100,
        },
        "environment": {
            **environment,
            "python": platform.python_version(),
            "executable": sys.executable,
            "hardware": _hardware(),
        },
        "runtime": {
            "providers": session.get_providers(),
            "available_providers": ort.get_available_providers(),
            "intra_op_threads": 1,
            "inter_op_threads": 1,
            "execution_mode": "ORT_SEQUENTIAL",
            "graph_optimization_level": (
                "ORT_DISABLE_ALL" if args.optimization == "off" else "ORT_ENABLE_ALL"
            ),
            "thread_environment": {
                name: os.environ[name] for name in thread_environment
            },
            "cpu_memory_arena": options.enable_cpu_mem_arena,
            "memory_pattern": options.enable_mem_pattern,
        },
        "timing_ms": {
            "numpy_ort_imports": import_ms,
            "session_creation": session_ms,
            "first_inference": first_ms,
            "warmed_mean": float(durations_ns.mean() / 1e6),
            "warmed_median": float(np.median(durations_ns) / 1e6),
            "warmed_p95": float(np.percentile(durations_ns, 95) / 1e6),
            "remaining_9999_examples_with_preprocessing_and_validation": evaluation_ms,
        },
        "timing_protocol": {
            "warmup_repetitions": WARMUP_REPETITIONS,
            "timed_repetitions": TIMED_REPETITIONS,
            "sample": "canonical test image 0",
            "timed_call": "session.run only; excludes input preparation and validation",
            "session_creation_excludes": "model/data reads, hashing and imports",
        },
        "memory_bytes": memory,
        "memory_protocol": {
            "current_rss": "/proc/self/status VmRSS (KiB converted to bytes)",
            "peak_rss": "resource.ru_maxrss (Linux KiB/macOS bytes converted to bytes)",
            "scope": "one model; excludes JAX/export and native reference loading",
            "retained_inputs": "raw uint8 test images; one float32 image at a time",
            "unavailable_measurements": None,
        },
        "native_validation": native,
        "checked_gates_passed": bool(passed),
        "all_acceptance_gates_checked": False,
        "requires_separate_comparison": [
            "W8A8 ORT errors minus matching FP32 ORT errors <= 50",
            "ORT optimization off/on W8A8 logits and predictions",
            "W8A8 artifact bytes <= 40% of matching FP32 artifact bytes",
            "export graph/storage and full ONNX checking",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.logits is not None:
        args.logits.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.logits, logits=logits, labels=labels)
    print(
        json.dumps(
            {
                "metrics": str(args.output),
                "errors": errors,
                "checked_gates_passed": passed,
                "all_acceptance_gates_checked": False,
            }
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
