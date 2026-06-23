# tests/extra_tests/framework/test_deployment_readiness_report.py

from pathlib import Path
import json

import jax.numpy as jnp
import onnx

from jax2onnx import (
    deployment_readiness_report,
    to_onnx,
    write_deployment_readiness_report,
)


def _simple_model(x):
    return jnp.tanh(x + 1.0)


def test_deployment_readiness_report_summarizes_exported_model() -> None:
    model = to_onnx(
        _simple_model,
        inputs=[("B", 4)],
        input_names=["features"],
        output_names=["scores"],
    )

    report = deployment_readiness_report(model)

    assert report.is_ready
    assert report.checker.ok
    assert report.shape_inference.ok
    assert report.inputs[0].name == "features"
    assert report.inputs[0].dtype == "FLOAT"
    assert report.inputs[0].shape == ("B", 4)
    assert report.outputs[0].name == "scores"
    assert any(operator.op_type == "Add" for operator in report.operators)
    assert any(operator.op_type == "Tanh" for operator in report.operators)
    assert any("symbolic dimension 'B'" in warning for warning in report.warnings)


def test_deployment_readiness_report_writes_json_artifact(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    report_path = tmp_path / "report.json"
    to_onnx(
        _simple_model,
        inputs=[(2, 4)],
        return_mode="file",
        output_path=model_path,
    )

    report = write_deployment_readiness_report(model_path, report_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert report.is_ready
    assert payload["is_ready"] is True
    assert payload["checker"]["ok"] is True
    assert payload["shape_inference"]["ok"] is True
    assert payload["operators"]


def test_deployment_readiness_report_captures_invalid_model() -> None:
    report = deployment_readiness_report(onnx.ModelProto())

    assert not report.is_ready
    assert not report.checker.ok
    assert report.checker.message
