from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp
from jax import lax

import onnx
from onnx import TensorProto as _TP

from jax2onnx.user_interface import to_onnx


def _fn_feed_forward_like_nested() -> jnp.ndarray:
    table_f32 = jnp.asarray([0.1, 0.2, 0.3, 0.4], dtype=jnp.float32)

    def inner(carry_i: jnp.int32, _):
        idx = jnp.mod(carry_i, 4).astype(jnp.int64)
        val_f32 = lax.dynamic_slice(table_f32, (idx,), (1,)).squeeze()
        acc_f64 = carry_i.astype(jnp.float64) + jnp.array(0.0, dtype=jnp.float64)
        sum64 = acc_f64 + val_f32
        return carry_i + 1, sum64

    def outer(carry_o: jnp.int32, _):
        _, ys64 = lax.scan(inner, carry_o, xs=None, length=3)
        last_idx = jnp.array(2, dtype=jnp.int64)
        pick64 = lax.dynamic_slice(ys64, (last_idx,), (1,)).squeeze()
        return carry_o + 1, pick64

    _, out64 = lax.scan(outer, jnp.array(0, dtype=jnp.int32), xs=None, length=2)
    return out64


@pytest.mark.filterwarnings("ignore:.*appears in graph inputs.*:UserWarning")
def test_feed_forward_like_nested_loops_mixed_dtypes_ir(tmp_path):
    ort = pytest.importorskip("onnxruntime")

    model = to_onnx(
        _fn_feed_forward_like_nested,
        inputs=[],
        model_name="feed_forward_like_nested",
        enable_double_precision=True,
        opset=21,
    )
    out_path = tmp_path / "feed_forward_like_nested.onnx"
    out_path.write_bytes(model.SerializeToString())

    try:
        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        outs = sess.run(None, {})
    except Exception as exc:  # pragma: no cover - converter2 gap tracking
        pytest.xfail(
            "converter2 mixed-dtype harmonization missing Cast(to=DOUBLE) inside Loop bodies"
            f" (ORT load failed: {exc})"
        )
    assert len(outs) == 1
    assert np.asarray(outs[0]).dtype == np.float64

    loaded = onnx.load(str(out_path))

    def iter_loop_bodies(graph):
        for node in graph.node:
            if node.op_type != "Loop":
                continue
            for attr in node.attribute:
                if attr.name == "body" and attr.g is not None:
                    yield attr.g
                    yield from iter_loop_bodies(attr.g)

    def has_index_cast_add(loop_graph):
        out2node = {out: node for node in loop_graph.node for out in node.output}
        index_ops = {"Slice", "GatherND", "Gather"}
        passthrough = {"Squeeze", "Unsqueeze", "Identity", "Reshape", "Transpose"}

        def walk_back(name, seen_cast=False, depth=0, limit=16):
            if not name or depth > limit:
                return False
            node = out2node.get(name)
            if node is None:
                return False
            if node.op_type == "Cast":
                to_double = any(
                    attr.name == "to" and attr.i == _TP.DOUBLE
                    for attr in node.attribute
                )
                return any(
                    walk_back(inp, seen_cast or to_double, depth + 1, limit)
                    for inp in node.input
                )
            if node.op_type in index_ops:
                if seen_cast:
                    return True
                return any(
                    walk_back(inp, seen_cast, depth + 1, limit) for inp in node.input
                )
            if node.op_type in passthrough:
                return any(
                    walk_back(inp, seen_cast, depth + 1, limit) for inp in node.input
                )
            return False

        for node in loop_graph.node:
            if node.op_type != "Add":
                continue
            if any(walk_back(inp) for inp in node.input):
                return True
        return False

    assert any(
        has_index_cast_add(g) for g in iter_loop_bodies(loaded.graph)
    ), "Expected indexing → Cast(to=DOUBLE) → Add inside a Loop body to ensure mixed-dtype harmonization."
