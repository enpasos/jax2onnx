# tests/extra_tests/converter/test_conversion_primitive_recording.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import pytest

from jax2onnx import onnx_function, user_interface
from jax2onnx.converter import conversion_api, lowering_dispatch
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.source_profile import SourcePrecisionError
from jax2onnx.precision import CheckStatus, get_precision_report
from tests.extra_tests.converter.source_precision_diagnostics import DiagnosticValidator


def _add_one(x):
    return x + 1


def test_record_primitive_calls_file_writes_lowering_records(tmp_path) -> None:
    log_path = tmp_path / "primitive_calls.json"

    conversion_api.to_onnx(
        fn=_add_one,
        inputs=[jax.ShapeDtypeStruct((2,), jnp.float32)],
        input_params=None,
        model_name="primitive_calls",
        opset=21,
        enable_double_precision=False,
        record_primitive_calls_file=str(log_path),
    )

    records = json.loads(log_path.read_text(encoding="utf-8"))
    assert records
    add_record = next(record for record in records if record["primitive_name"] == "add")
    assert add_record["sequence_id"] >= 0
    assert add_record["plugin_file_hint"]
    assert add_record["inputs_aval"]
    assert add_record["outputs_aval"]
    assert add_record["inputs_jax_vars"]
    assert add_record["outputs_jax_vars"]
    assert add_record["outputs_onnx_names"]


@onnx_function
def _source_recorded_inner(x):
    return x + np.float64(1)


@onnx_function
def _source_recorded_outer(x):
    return _source_recorded_inner(x) + np.float64(2)


def _source_recorded_function(x):
    return _source_recorded_outer(x)


def _source_recorded_if(x):
    return jax.lax.cond(x > np.float64(0), lambda y: y + y, lambda y: -y, x)


def _source_recorded_loop(x):
    return jax.lax.while_loop(
        lambda state: state[0] < np.int32(2),
        lambda state: (state[0] + np.int32(1), state[1] + state[1]),
        (np.int32(0), x),
    )[1]


def _source_recorded_combined(x):
    def step(state):
        value = jax.lax.cond(
            state[1] > np.float64(0), _source_recorded_outer, lambda y: -y, state[1]
        )
        return state[0] + np.int32(1), value

    return jax.lax.while_loop(
        lambda state: state[0] < np.int32(2), step, (np.int32(0), x)
    )[1]


_SOURCE_MODELS = {
    "root": (_add_one, False),
    "function": (_source_recorded_function, False),
    "if": (_source_recorded_if, False),
    "loop": (_source_recorded_loop, True),
    "combined": (_source_recorded_combined, True),
}


def _source_export(fn, *, diagnostic=False, **options):
    with jax.enable_x64(True), jax.default_matmul_precision(None):
        return user_interface._to_onnx(
            fn,
            [jax.ShapeDtypeStruct((), np.float64)],
            model_name="source_recording",
            input_names=["feature"],
            output_names=["result"],
            precision_policy=PrecisionPolicy.SOURCE,
            diagnostic_jaxpr_validator=DiagnosticValidator() if diagnostic else None,
            **options,
        )


def _observe_writes(monkeypatch):
    observed = []
    write = lowering_dispatch.save_primitive_calls_log

    def record(records, path):
        observed.append((tuple(records), path))
        write(records, path)

    monkeypatch.setattr(lowering_dispatch, "save_primitive_calls_log", record)
    return observed


@pytest.fixture(autouse=True)
def _forbid_runtime_sessions(monkeypatch):
    import onnxruntime

    def forbidden(*args, **kwargs):
        pytest.fail("Primitive recording tests must not create runtime sessions")

    monkeypatch.setattr(onnxruntime, "InferenceSession", forbidden)


@pytest.mark.parametrize("mode", ["ir", "proto"])
@pytest.mark.parametrize("scope", _SOURCE_MODELS)
def test_source_recording_preserves_artifacts_and_flushes_nested_scopes_once(
    scope: str, mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fn, diagnostic = _SOURCE_MODELS[scope]
    log_path = tmp_path / "source_calls.json"
    writes = _observe_writes(monkeypatch)
    protos, reports = [], []
    for recording in (False, True, False):
        artifact = _source_export(
            fn,
            diagnostic=diagnostic,
            return_mode=mode,
            record_primitive_calls_file=str(log_path) if recording else None,
        )
        proto = ir.to_proto(artifact) if isinstance(artifact, ir.Model) else artifact
        onnx.checker.check_model(proto, full_check=True)
        protos.append(proto.SerializeToString(deterministic=True))
        reports.append(get_precision_report(artifact))
        assert len(writes) == (0 if len(protos) == 1 else 1)
    assert protos[0] == protos[1] == protos[2]
    assert reports[0] == reports[1] == reports[2]
    report = reports[0]
    assert report.profile == (
        "source-diagnostic-v1" if diagnostic else "source-static-v1"
    )
    statuses = {check.category: check.status for check in report.checks}
    assert statuses["lowering"] is (
        CheckStatus.UNVERIFIED if diagnostic else CheckStatus.VALIDATED
    )
    assert statuses["runtime-execution"] is CheckStatus.UNVERIFIED
    assert statuses["numerical-acceptance"] is CheckStatus.UNVERIFIED

    records = json.loads(log_path.read_text())
    assert len(records) == len(writes[0][0]) and writes[0][1] == str(log_path)
    names = [record["primitive_name"] for record in records]
    if scope == "root":
        assert names == ["add"]
    if scope in {"function", "combined"}:
        for name in ("_source_recorded_inner", "_source_recorded_outer"):
            assert sum(primitive.endswith(name) for primitive in names) == 1
    if scope in {"if", "combined"}:
        assert "cond" in names and "neg" in names
    if diagnostic:
        assert "while" in names and "lt" in names
    for record in records:
        assert record["sequence_id"] >= 0 and record["plugin_file_hint"]
        for role in ("inputs", "outputs"):
            assert len(record[f"{role}_aval"]) == len(record[f"{role}_jax_vars"])
            assert len(record[f"{role}_aval"]) == len(record[f"{role}_onnx_names"])
            assert all(
                dtype in {"float64", "int32", "bool"}
                for _, dtype, _ in record[f"{role}_aval"]
            )
        assert all(record["outputs_onnx_names"])


@pytest.mark.parametrize("target", ["missing-parent", "directory"])
def test_source_log_write_failures_preserve_successful_artifact(
    target: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = (
        tmp_path / "missing" / "calls.json" if target == "missing-parent" else tmp_path
    )
    writes = _observe_writes(monkeypatch)
    artifacts = [
        _source_export(_source_recorded_outer, record_primitive_calls_file=log_path)
        for log_path in (None, str(path))
    ]
    assert len(writes) == 1 and writes[0][0]
    assert artifacts[0].SerializeToString(deterministic=True) == artifacts[
        1
    ].SerializeToString(deterministic=True)
    assert get_precision_report(artifacts[0]) == get_precision_report(artifacts[1])
    onnx.checker.check_model(artifacts[1], full_check=True)
    if target == "missing-parent":
        assert not path.parent.exists()


@pytest.mark.parametrize("stage", ["lowering", "finalization"])
@pytest.mark.parametrize("existing", [False, True])
def test_failed_log_write_cannot_mask_source_lowering_or_finalization_errors(
    stage: str, existing: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    failure = RuntimeError(f"recording {stage} failure witness")
    writes = _observe_writes(monkeypatch)
    if stage == "lowering":
        lower = lowering_dispatch._lower_registered_equation

        def fail_after_recording(*args: Any, **kwargs: Any):
            lower(*args, **kwargs)
            raise failure

        monkeypatch.setattr(
            lowering_dispatch, "_lower_registered_equation", fail_after_recording
        )
    else:

        def fail_finalization(model):
            raise failure

        monkeypatch.setattr(
            conversion_api, "_finalize_model_value_shapes", fail_finalization
        )
    path = tmp_path / "model.onnx"
    if existing:
        path.write_bytes(b"preexisting artifact")
    with pytest.raises(
        SourcePrecisionError if stage == "finalization" else RuntimeError
    ) as caught:
        _source_export(
            _add_one,
            record_primitive_calls_file=str(tmp_path),
            return_mode="file",
            output_path=path,
        )
    assert len(writes) == 1 and writes[0][0]
    if stage == "lowering":
        assert caught.value is failure
    else:
        assert caught.value.__cause__ is failure
        assert caught.value.phase == "finalization"
    assert (
        path.read_bytes() == b"preexisting artifact" if existing else not path.exists()
    )
