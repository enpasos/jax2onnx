"""Ensure converter keeps Shape helpers SSA-unique when multiple Adds appear."""

from __future__ import annotations

import collections
import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax import lax
from onnx import AttributeProto

from jax2onnx.user_interface import to_onnx

try:
    import onnxruntime as ort

    HAS_ORT = True
except Exception:  # pragma: no cover
    HAS_ORT = False


def _collect_node_outputs_recursive(graph):
    outs: list[str] = []
    for node in graph.node:
        outs.extend([name for name in node.output if name])
        for attr in node.attribute:
            if attr.type == AttributeProto.GRAPH and attr.g is not None:
                outs.extend(_collect_node_outputs_recursive(attr.g))
            elif attr.type == AttributeProto.GRAPHS and attr.graphs:
                for subgraph in attr.graphs:
                    outs.extend(_collect_node_outputs_recursive(subgraph))
    return outs


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


def _body_two_adds_each_needs_shape(y):
    dt = y.dtype
    one = jnp.array(1.0, dt)
    two = jnp.array(2.0, dt)
    three = jnp.array(3.0, dt)
    four = jnp.array(4.0, dt)

    def step(carry, _):
        h1 = carry + jnp.broadcast_to(one, carry.shape)
        b1 = jnp.broadcast_to(three, h1.shape)

        c_scaled = carry * jnp.array(1.0, dt)
        h2 = c_scaled + jnp.broadcast_to(two, carry.shape)
        b2 = jnp.broadcast_to(four, h2.shape)

        out = b1 + b2 + (h1 * 0) + (h2 * 0)
        return carry + jnp.array(0.0, dt), out

    _, ys = lax.scan(step, y, xs=None, length=2)
    return ys


@pytest.mark.parametrize("dtype", [jnp.float64])
def test_two_adds_shape_helpers_are_ssa_unique(tmp_path, dtype):
    spec = jax.ShapeDtypeStruct(("B", 4), dtype)
    model = to_onnx(
        _body_two_adds_each_needs_shape,
        inputs=[spec],
        enable_double_precision=True,
        opset=21,
        model_name="two_adds_shape_helpers_ssa",
    )

    body = _find_first_loop_body(model.graph)
    assert body is not None, "Expected Loop body subgraph"
    outs = _collect_node_outputs_recursive(body)
    dupes = [name for name, cnt in collections.Counter(outs).items() if cnt > 1]
    assert not dupes, f"Duplicate outputs in Loop body (SSA violation): {dupes}"

    if HAS_ORT:
        out_path = tmp_path / "two_adds_shape_helpers_ssa.onnx"
        out_path.write_bytes(model.SerializeToString())
        try:
            ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        except Exception as exc:  # pragma: no cover
            pytest.fail(f"onnxruntime failed to load IR model: {exc}")


@pytest.mark.parametrize("dtype", [jnp.float64])
def test_numeric_executes(dtype):
    if not HAS_ORT:
        pytest.skip("onnxruntime not available")
    spec = jax.ShapeDtypeStruct((7, 4), dtype)
    model = to_onnx(
        _body_two_adds_each_needs_shape,
        inputs=[spec],
        enable_double_precision=True,
        opset=21,
        model_name="two_adds_shape_helpers_numeric",
    )
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    x = np.zeros((7, 4), np.float64)
    feeds = {sess.get_inputs()[0].name: x}
    (y,) = sess.run(None, feeds)
    assert y.shape == (2, 7, 4)
