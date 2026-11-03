# tests/extra_tests/test_quantized_mnist_runtime.py

"""Offline acceptance-record checks; no ONNX, ORT, JAX, training or downloads."""

import json

import numpy as np
import pytest

from scripts import run_quantized_mnist as runtime


@pytest.fixture
def runtime_records(tmp_path):
    labels = np.zeros(10000, dtype=np.uint8)
    logits = np.zeros((10000, 10), dtype=np.float32)
    references = {
        f"{variant}_{mode}": logits
        for variant in ("fp32", "w8a8")
        for mode in runtime.NATIVE_MODES
    }
    np.savez(tmp_path / "native_outputs.npz", **references, labels=labels)
    native_sha = runtime._sha256(tmp_path / "native_outputs.npz")
    environment = {
        "versions": runtime.RUNTIME_PINS,
        "pythonpath_unset_or_empty": True,
        "training_packages": {
            name: {"distribution_absent": True, "module_absent": True}
            for name in runtime.TRAINING_PACKAGES
        },
    }
    artifacts = {}
    for variant, size in (("fp32", 1000), ("w8a8", 200)):
        # The comparison checks actual file bytes/hashes; it does not parse ONNX.
        model = tmp_path / f"{variant}.onnx"
        model.write_bytes(b"x" * size)
        artifacts[variant] = {
            "file": model.name,
            "bytes": size,
            "sha256": runtime._sha256(model),
        }
        for mode in ("off", "on"):
            report = {
                "variant": variant,
                "artifact": artifacts[variant],
                "checked_gates_passed": True,
                "native_validation": {
                    "status": "checked",
                    "checked_gates_passed": True,
                    "archive_sha256": native_sha,
                },
                "data": {"sha256": runtime.MNIST_SHA256, "test_examples": 10000},
                "batch_size": 1,
                "environment": environment,
                "runtime": {
                    "providers": ["CPUExecutionProvider"],
                    "intra_op_threads": 1,
                    "inter_op_threads": 1,
                    "graph_optimization_level": (
                        "ORT_DISABLE_ALL" if mode == "off" else "ORT_ENABLE_ALL"
                    ),
                },
                "accuracy": {"errors": 0},
            }
            (tmp_path / f"{variant}-{mode}.json").write_text(json.dumps(report))
            np.savez(tmp_path / f"{variant}-{mode}.npz", logits=logits, labels=labels)
    (tmp_path / "export.json").write_text(json.dumps({"artifacts": artifacts}))
    return tmp_path, logits, labels


def test_matching_records_accept_legitimate_zero_logits(runtime_records):
    directory, _, _ = runtime_records
    output = directory / "summary.json"
    assert runtime._compare_results(directory, output) == 0
    summary = json.loads(output.read_text())
    assert summary["checked_gates_passed"] is True
    assert summary["all_acceptance_gates_checked"] is False
    assert summary["artifact_ratio"] == 0.2


def test_replaced_logits_fail_despite_same_predictions_and_ort_agreement(
    runtime_records,
):
    directory, original, labels = runtime_records
    changed = original.copy()
    changed[0, 0] = np.float32(0.01)
    np.testing.assert_array_equal(changed.argmax(axis=1), original.argmax(axis=1))
    assert runtime._error_count(changed, labels) == 0
    for mode in ("off", "on"):
        np.savez(directory / f"w8a8-{mode}.npz", logits=changed, labels=labels)
    output = directory / "summary.json"
    assert runtime._compare_results(directory, output) == 1
    checks = json.loads(output.read_text())["checks"]
    assert checks["w8a8_optimization_fidelity"] is True
    assert checks["w8a8-off_native_fidelity"] is False
    assert checks["w8a8-on_native_fidelity"] is False


@pytest.mark.parametrize("mutation", ("model_hash", "optimization"))
def test_report_binds_actual_model_and_optimization(runtime_records, mutation):
    directory, _, _ = runtime_records
    path = directory / "w8a8-off.json"
    report = json.loads(path.read_text())
    if mutation == "model_hash":
        report["artifact"]["sha256"] = "0" * 64
        expected_check = "w8a8-off_frozen_artifact"
    else:
        report["runtime"]["graph_optimization_level"] = "ORT_ENABLE_ALL"
        expected_check = "w8a8-off_runtime_evidence"
    path.write_text(json.dumps(report))
    output = directory / "summary.json"
    assert runtime._compare_results(directory, output) == 1
    assert json.loads(output.read_text())["checks"][expected_check] is False
