from __future__ import annotations

import numpy as np
import onnx
from onnx import numpy_helper
import pytest

from jax2onnx.sandbox.issue52_scatter_payload_repro import (
    _feed_forward_fn,
    _load_payload,
)
from jax2onnx.user_interface import to_onnx


def _export_issue52_model(fn, prim0, initial_time, time_step):
    kwargs = dict(
        inputs=[prim0, initial_time, time_step],
        model_name="issue52_scatter_payload",
        enable_double_precision=True,
        opset=21,
    )
    try:
        return to_onnx(fn, **kwargs)
    except TypeError:
        kwargs.pop("enable_double_precision", None)
        return to_onnx(fn, **kwargs)


@pytest.mark.filterwarnings("ignore:.*Removing initializer.*:UserWarning")
def test_issue52_scatter_payload_roundtrip(tmp_path):
    ort = pytest.importorskip(
        "onnxruntime", reason="onnxruntime is required to reproduce issue #52"
    )

    closed, prim0, initial_time, time_step = _load_payload()
    feed_forward = _feed_forward_fn(closed)

    expected_outputs = [
        np.asarray(arr) for arr in feed_forward(prim0, initial_time, time_step)
    ]

    model = _export_issue52_model(feed_forward, prim0, initial_time, time_step)
    model_path = tmp_path / "issue52_scatter_payload.onnx"
    model_path.write_bytes(model.SerializeToString())

    try:
        session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        ort_inputs = {
            name: np.asarray(value)
            for name, value in zip(
                [tensor.name for tensor in session.get_inputs()],
                (prim0, initial_time, time_step),
            )
        }
        ort_outputs = session.run(None, ort_inputs)
    except Exception as exc:  # pragma: no cover - regression to be fixed
        pytest.xfail(
            "Pending fix for issue #52: Scatter window lowering emits mismatched"
            f" shapes inside nested Loop bodies (onnxruntime error: {exc})"
        )

    assert len(ort_outputs) == len(expected_outputs)
    for index, (observed, expected) in enumerate(zip(ort_outputs, expected_outputs)):
        np.testing.assert_allclose(
            observed,
            expected,
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"Mismatch detected in output {index}",
        )


def _iter_graphs(model_proto: "onnx.ModelProto"):
    queue = [model_proto.graph]
    while queue:
        graph = queue.pop()
        yield graph
        for node in graph.node:
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    queue.append(attr.g)


def _get_constant_tensor(
    model_proto: "onnx.ModelProto", name: str
) -> np.ndarray | None:
    for graph in _iter_graphs(model_proto):
        for node in graph.node:
            if node.op_type != "Constant" or not node.output:
                continue
            if node.output[0] != name:
                continue
            for attr in node.attribute:
                if attr.name == "value" and attr.HasField("t"):
                    return numpy_helper.to_array(attr.t)
    return None


@pytest.mark.xfail(
    reason="Scatter window dimension drops the 6-element extent (issue #52)"
)
def test_issue52_scatter_window_keeps_update_axis(tmp_path):
    ort = pytest.importorskip(
        "onnxruntime", reason="onnxruntime is required to reproduce issue #52"
    )

    closed, prim0, initial_time, time_step = _load_payload()
    feed_forward = _feed_forward_fn(closed)

    model = _export_issue52_model(feed_forward, prim0, initial_time, time_step)
    model_path = tmp_path / "issue52_scatter_payload.onnx"
    model_path.write_bytes(model.SerializeToString())

    with pytest.raises(Exception):
        session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        feeds = {
            tensor.name: np.asarray(value)
            for tensor, value in zip(
                session.get_inputs(), (prim0, initial_time, time_step)
            )
        }
        session.run(None, feeds)

    model_proto = onnx.load(str(model_path))
    const_name = "scan_loop_0/scan_loop_0/bcast_dim_c_24"
    const_value = _get_constant_tensor(model_proto, const_name)
    assert const_value is not None, f"Missing Constant node for {const_name}"

    expected = np.asarray([6], dtype=np.int64)
    assert np.array_equal(
        const_value,
        expected,
    ), "Scatter window dimension should match the 6-element updates axis"
