"""Tests to ensure converter2 keeps SSA names unique for broadcast_in_dim patterns."""

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


def _fn_scan_two_broadcasts(y):
    def body(carry, _):
        one = jnp.broadcast_to(jnp.array(1.0, dtype=carry.dtype), carry.shape)
        zero = jnp.broadcast_to(jnp.array(0.0, dtype=carry.dtype), carry.shape)
        out = carry + one + zero + jnp.reshape(carry, carry.shape)
        return carry + jnp.array(0.0, dtype=carry.dtype), out

    _, ys = lax.scan(body, y, xs=None, length=2)
    return ys


def _collect_node_output_names(graph):
    names: list[str] = []
    for node in graph.node:
        names.extend([out for out in node.output if out])
        for attr in node.attribute:
            if attr.type == AttributeProto.GRAPH and attr.g is not None:
                names.extend(_collect_node_output_names(attr.g))
            elif attr.type == AttributeProto.GRAPHS and attr.graphs:
                for subgraph in attr.graphs:
                    names.extend(_collect_node_output_names(subgraph))
    return names


def test_broadcast_in_dim_helpers_unique_ssa(tmp_path):
    spec = jax.ShapeDtypeStruct(("B", 4), jnp.float32)
    model = to_onnx(
        _fn_scan_two_broadcasts,
        inputs=[spec],
        opset=21,
        model_name="broadcast_in_dim_ssa_unique",
        use_onnx_ir=True,
    )

    all_outs = _collect_node_output_names(model.graph)
    dupes = [name for name, count in collections.Counter(all_outs).items() if count > 1]
    assert not dupes, f"Found duplicate node outputs: {dupes}"

    if HAS_ORT:
        out_path = tmp_path / "broadcast_in_dim_ssa_unique.onnx"
        out_path.write_bytes(model.SerializeToString())
        try:
            ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        except Exception as exc:  # pragma: no cover
            pytest.fail(f"onnxruntime failed to load IR model: {exc}")
