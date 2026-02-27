# tests/extra_tests/loop/test_loop_scatter_payload_regression.py

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import onnx
from onnx import numpy_helper
import pytest

from jax2onnx.user_interface import to_onnx
from ..helpers.issue52_scatter_payload_fixture import (
    _feed_forward_fn,
    _load_payload,
)


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

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    ort_inputs = {
        name: np.asarray(value)
        for name, value in zip(
            [tensor.name for tensor in session.get_inputs()],
            (prim0, initial_time, time_step),
        )
    }
    ort_outputs = session.run(None, ort_inputs)

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


def _iter_initializers(model_proto: "onnx.ModelProto"):
    for graph in _iter_graphs(model_proto):
        for init in graph.initializer:
            yield init.name, np.asarray(numpy_helper.to_array(init))


def test_issue52_scatter_window_keeps_update_axis(tmp_path):
    ort = pytest.importorskip(
        "onnxruntime", reason="onnxruntime is required to reproduce issue #52"
    )

    closed, prim0, initial_time, time_step = _load_payload()
    feed_forward = _feed_forward_fn(closed)

    model = _export_issue52_model(feed_forward, prim0, initial_time, time_step)
    model_path = tmp_path / "issue52_scatter_payload.onnx"
    model_path.write_bytes(model.SerializeToString())

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    feeds = {
        tensor.name: np.asarray(value)
        for tensor, value in zip(session.get_inputs(), (prim0, initial_time, time_step))
    }
    session.run(None, feeds)

    model_proto = onnx.load(str(model_path))
    bcast_dim_values = []
    for name, value in _iter_initializers(model_proto):
        if "bcast_dim_c_" not in name:
            continue
        if value.size != 1:
            continue
        bcast_dim_values.append(int(value.reshape(-1)[0]))

    assert bcast_dim_values, "Expected broadcast-dimension initializers for issue #52"
    assert 6 in bcast_dim_values, (
        "Scatter window dimension should include extent 6; "
        f"got values {sorted(set(bcast_dim_values))}"
    )


def test_issue52_const_slice_broadcast(tmp_path):
    ort = pytest.importorskip(
        "onnxruntime", reason="onnxruntime is required to reproduce issue #52"
    )

    @jax.jit
    def _const_slice_add():
        const = jnp.arange(1.0, 7.0, dtype=jnp.float32).reshape(1, 6, 1, 1)
        loop_values = jnp.arange(5.0, dtype=jnp.float32).reshape(5, 1, 1, 1)
        window = const[:, 1:4, :, :]
        return loop_values + window

    model = to_onnx(
        _const_slice_add,
        inputs=[],
        model_name="issue52_const_slice",
        opset=21,
    )
    model_path = tmp_path / "issue52_const_slice.onnx"
    model_path.write_bytes(model.SerializeToString())

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    observed = session.run(None, {})
    expected = np.asarray(_const_slice_add())
    assert len(observed) == 1
    np.testing.assert_allclose(observed[0], expected, rtol=1e-6, atol=1e-6)
