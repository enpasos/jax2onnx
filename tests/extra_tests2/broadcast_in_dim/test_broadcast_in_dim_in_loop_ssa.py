"""Converter2 SSA guard when broadcast_in_dim appears within Loop bodies."""

from __future__ import annotations

import collections
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


def _fn_loop_body_has_two_broadcasts(y):
    def body(carry, _):
        one = jnp.broadcast_to(jnp.array(1.0, dtype=carry.dtype), carry.shape)
        zero = jnp.broadcast_to(jnp.array(0.0, dtype=carry.dtype), carry.shape)
        e = carry + one
        g = e + zero
        reshaped = jnp.reshape(e + jnp.array(0.0, dtype=carry.dtype), e.shape)
        out = g + reshaped
        return carry + jnp.array(0.0, dtype=carry.dtype), out

    _, ys = lax.scan(body, y, xs=None, length=2)
    return ys


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


def test_broadcast_in_dim_inside_loop_ssa_unique(tmp_path):
    spec = jax.ShapeDtypeStruct(("B", 4), jnp.float32)
    model = to_onnx(
        _fn_loop_body_has_two_broadcasts,
        inputs=[spec],
        opset=21,
        model_name="broadcast_in_dim_loop_ssa_unique",
        use_onnx_ir=True,
    )

    all_outs = _collect_node_outputs_recursive(model.graph)
    dupes = [name for name, count in collections.Counter(all_outs).items() if count > 1]
    assert not dupes, f"Duplicate node outputs detected (SSA violation): {dupes}"

    loop_body = _find_first_loop_body(model.graph)
    assert loop_body is not None, "Expected Loop subgraph"
    body_outs = _collect_node_outputs_recursive(loop_body)
    body_dupes = [
        name for name, count in collections.Counter(body_outs).items() if count > 1
    ]
    assert not body_dupes, f"Duplicate outputs in Loop body: {body_dupes}"

    shape_helpers = [name for name in body_outs if name.endswith("__shape")]
    shape_dupes = [
        name for name, count in collections.Counter(shape_helpers).items() if count > 1
    ]
    assert not shape_dupes, f"Duplicate '__shape' helpers in Loop body: {shape_dupes}"

    if HAS_ORT:
        out_path = tmp_path / "broadcast_in_dim_loop_ssa_unique.onnx"
        out_path.write_bytes(model.SerializeToString())
        try:
            ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        except Exception as exc:  # pragma: no cover
            pytest.fail(f"onnxruntime failed to load IR model: {exc}")


def test_broadcast_in_dim_loop_emits_shape_nodes_against_body_inputs():
    spec = jax.ShapeDtypeStruct(("B", 4), jnp.float32)
    model = to_onnx(
        _fn_loop_body_has_two_broadcasts,
        inputs=[spec],
        opset=21,
        model_name="broadcast_in_dim_loop_shape_inputs",
        use_onnx_ir=True,
    )
    loop_body = _find_first_loop_body(model.graph)
    assert loop_body is not None
    body_inputs = {vi.name for vi in loop_body.input}
    produced: set[str] = set()
    for node in loop_body.node:
        produced.update(out for out in node.output if out)
    known = body_inputs | produced

    shape_nodes = [node for node in loop_body.node if node.op_type == "Shape"]
    assert shape_nodes, "Expected runtime Shape helpers inside Loop body"
    for node in shape_nodes:
        assert node.input, "Shape node without inputs"
        assert (
            node.input[0] in known
        ), f"Shape input {node.input[0]!r} not bound to body inputs/defs"
