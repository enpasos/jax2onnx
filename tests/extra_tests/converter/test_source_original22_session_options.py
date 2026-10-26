# tests/extra_tests/converter/test_source_original22_session_options.py

"""Small controls for the original22 memory-pattern diagnostic target."""

import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import onnx
from onnx import helper
import pytest

from tests.extra_tests.converter import (
    test_source_original22_state_preparation as owner,
)


def _expected_target(optimize: bool) -> dict[str, Any]:
    defaults = owner.ort.SessionOptions()
    return dict(
        optimize=optimize,
        providers=["CPUExecutionProvider"],
        enable_mem_pattern=False,
        enable_cpu_mem_arena=defaults.enable_cpu_mem_arena,
        intra_op_num_threads=1,
        inter_op_num_threads=1,
        execution_mode=str(owner.ort.ExecutionMode.ORT_SEQUENTIAL),
        graph_optimization_level=str(
            owner.ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            if optimize
            else owner.ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        ),
    )


@pytest.mark.parametrize("optimize", (False, True))
def test_session_preserves_blob_cpu_and_both_graph_modes(
    monkeypatch: pytest.MonkeyPatch, optimize: bool
) -> None:
    blob = bytes(bytearray(b"exact original serialized bytes"))
    calls: list[dict[str, Any]] = []
    session = SimpleNamespace(
        get_providers=lambda: ["CPUExecutionProvider"],
        get_session_options=lambda: calls[0]["options"],
    )

    def create(received_blob: bytes, options: Any, providers: list[str]) -> Any:
        calls.append(dict(blob=received_blob, options=options, providers=providers))
        return session

    monkeypatch.setattr(owner.ort, "InferenceSession", create)
    assert owner._session(blob, optimize) is session
    assert len(calls) == 1
    assert calls[0]["blob"] is blob
    assert calls[0]["providers"] == ["CPUExecutionProvider"]
    assert owner._target_session_options(session, optimize) == _expected_target(
        optimize
    )
    # Scalar metadata is JSON serializable and contains no session/options/blob.
    json.dumps(owner._target_session_options(session, optimize), sort_keys=True)


@pytest.mark.parametrize("optimize", (False, True))
def test_session_rejects_an_actual_memory_pattern_flag_reported_true(
    monkeypatch: pytest.MonkeyPatch, optimize: bool
) -> None:
    calls: list[tuple[bytes, Any, list[str]]] = []

    def create(blob: bytes, options: Any, providers: list[str]) -> Any:
        calls.append((blob, options, providers))
        assert options.enable_mem_pattern is False
        observed = SimpleNamespace(
            enable_mem_pattern=True,
            intra_op_num_threads=options.intra_op_num_threads,
            inter_op_num_threads=options.inter_op_num_threads,
            execution_mode=options.execution_mode,
            graph_optimization_level=options.graph_optimization_level,
        )
        return SimpleNamespace(
            get_providers=lambda: ["CPUExecutionProvider"],
            get_session_options=lambda: observed,
        )

    monkeypatch.setattr(owner.ort, "InferenceSession", create)
    blob = b"returned actual flag must be checked"
    with pytest.raises(AssertionError):
        owner._session(blob, optimize)
    assert len(calls) == 1 and calls[0][0] is blob


@pytest.mark.parametrize("optimize", (False, True))
def test_actual_dynamic_cpu_session_runs_b2_then_b5_with_reported_flag_false(
    optimize: bool, record_property: Any
) -> None:
    graph = helper.make_graph(
        [helper.make_node("Identity", ["features"], ["output"])],
        "original22-session-options-control",
        [helper.make_tensor_value_info("features", onnx.TensorProto.FLOAT, ["B", 3])],
        [helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, ["B", 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 23)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    session = owner._session(model.SerializeToString(), optimize)
    target = owner._target_session_options(session, optimize)
    assert target == _expected_target(optimize)
    record_property("original22-session-options", json.dumps(target, sort_keys=True))
    assert [value.name for value in session.get_inputs()] == ["features"]
    assert [value.name for value in session.get_outputs()] == ["output"]
    for batch in (2, 5):
        features = np.arange(batch * 3, dtype=np.float32).reshape(batch, 3)
        outputs = session.run(None, {"features": features})
        assert len(outputs) == 1
        assert outputs[0].dtype == features.dtype and outputs[0].shape == features.shape
        np.testing.assert_array_equal(outputs[0], features)
