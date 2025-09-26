"""IR regression: broadcast_in_dim + reshape share shape helpers without SSA clashes."""

from __future__ import annotations

import collections
import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax import lax
from onnx import AttributeProto

from jax2onnx.user_interface import to_onnx

try:  # optional runtime smoke test
    import onnxruntime as ort

    HAS_ORT = True
except Exception:  # pragma: no cover - ORT dependency optional
    HAS_ORT = False


def _collect_node_outputs_recursive(graph):
    outputs: list[str] = []
    for node in graph.node:
        outputs.extend([name for name in node.output if name])
        for attr in node.attribute:
            if attr.type == AttributeProto.GRAPH and attr.g is not None:
                outputs.extend(_collect_node_outputs_recursive(attr.g))
            elif attr.type == AttributeProto.GRAPHS and attr.graphs:
                for subgraph in attr.graphs:
                    outputs.extend(_collect_node_outputs_recursive(subgraph))
    return outputs


def _find_first_loop_body(graph):
    for node in graph.node:
        if node.op_type == "Loop":
            for attr in node.attribute:
                if attr.type == AttributeProto.GRAPH and attr.g is not None:
                    return attr.g
    for node in graph.node:
        for attr in node.attribute:
            if attr.type == AttributeProto.GRAPH and attr.g is not None:
                body = _find_first_loop_body(attr.g)
                if body is not None:
                    return body
            elif attr.type == AttributeProto.GRAPHS and attr.graphs:
                for subgraph in attr.graphs:
                    body = _find_first_loop_body(subgraph)
                    if body is not None:
                        return body
    return None


def _body_broadcast_and_reshape_on_same_h(y):
    dt = y.dtype
    one = jnp.array(1.0, dt)

    def step(carry, _):
        h = carry + jnp.broadcast_to(one, carry.shape)
        broadcast_shape = jnp.broadcast_to(one, h.shape)
        h2 = h + jnp.array(0.0, dt)
        reshaped = jnp.reshape(h2, h.shape)
        out = h + broadcast_shape + reshaped
        return carry + jnp.array(0.0, dt), out

    _, ys = lax.scan(step, y, xs=None, length=2)
    return ys


@pytest.mark.parametrize("dtype", [jnp.float64])
def test_loop_body_shared_shape_helpers_are_ssa_unique(tmp_path, dtype):
    spec = jax.ShapeDtypeStruct(("B", 4), dtype)
    model = to_onnx(
        _body_broadcast_and_reshape_on_same_h,
        inputs=[spec],
        enable_double_precision=True,
        loosen_internal_shapes=True,
        opset=21,
        model_name="broadcast_and_reshape_shared_shape_ssa",
        use_onnx_ir=True,
    )

    body = _find_first_loop_body(model.graph)
    assert body is not None, "Expected a Loop body subgraph"
    outs = _collect_node_outputs_recursive(body)
    dupes = [name for name, count in collections.Counter(outs).items() if count > 1]
    assert not dupes, f"Duplicate outputs in Loop body (SSA violation): {dupes}"

    if HAS_ORT:
        out_path = tmp_path / "broadcast_and_reshape_shared_shape_ssa.onnx"
        out_path.write_bytes(model.SerializeToString())
        try:
            ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        except Exception as exc:  # pragma: no cover - regression guard
            pytest.fail(f"onnxruntime failed to load IR model: {exc}")


@pytest.mark.parametrize("dtype", [jnp.float64])
def test_numeric_executes(tmp_path, dtype):
    if not HAS_ORT:
        pytest.skip("onnxruntime not available")
    spec = jax.ShapeDtypeStruct((7, 4), dtype)
    model = to_onnx(
        _body_broadcast_and_reshape_on_same_h,
        inputs=[spec],
        enable_double_precision=True,
        loosen_internal_shapes=True,
        opset=21,
        model_name="broadcast_and_reshape_shared_shape_numeric",
        use_onnx_ir=True,
    )
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    x = np.zeros((7, 4), np.float64)
    feeds = {sess.get_inputs()[0].name: x}
    (y,) = sess.run(None, feeds)
    assert y.shape == (2, 7, 4)
