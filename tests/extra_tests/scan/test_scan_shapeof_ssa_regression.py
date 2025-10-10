# tests/extra_tests/scan/test_scan_shapeof_ssa_regression.py

from __future__ import annotations

import collections
import numpy as np
import jax
import jax.numpy as jnp
import pytest
import onnx
from jax import lax

from jax2onnx.user_interface import to_onnx


def _all_node_outputs_recursive(g):
    outs = []
    for n in g.node:
        outs.extend([o for o in n.output if o])
        for a in n.attribute:
            if a.type == onnx.AttributeProto.GRAPH and a.g is not None:
                outs.extend(_all_node_outputs_recursive(a.g))
            elif a.type == onnx.AttributeProto.GRAPHS and a.graphs:
                for sg in a.graphs:
                    outs.extend(_all_node_outputs_recursive(sg))
    return outs


def _scan_body_multi_shape(carry, _):
    add = carry + 1.0
    z0 = add + jnp.broadcast_to(0.0, (add.shape[0], add.shape[1]))
    z1 = add + jnp.broadcast_to(1.0, (add.shape[0], add.shape[1]))
    z2 = jnp.reshape(add, (add.shape[0], add.shape[1]))
    return add, (z0 + z1 + z2)


def _scan_wrapper(x):
    carry_out, ys = lax.scan(_scan_body_multi_shape, x, xs=None, length=2)
    return carry_out + ys[-1]


def _nested_fori_scan_wrapper(x):
    def body(_, state):
        return _scan_wrapper(state)

    return lax.fori_loop(0, 2, body, x)


class TestScanShapeOfSSARegression:
    def test_scan_shapeof_ssa_regression(self, tmp_path, monkeypatch):
        ort = pytest.importorskip("onnxruntime")

        spec = jax.ShapeDtypeStruct(("B", 4), jnp.float32)
        model = to_onnx(
            _scan_wrapper,
            inputs=[spec],
            opset=21,
            model_name="scan_shapeof_ssa_regression",
        )

        outs = _all_node_outputs_recursive(model.graph)
        dupes = [
            n
            for n, c in collections.Counter(outs).items()
            if c > 1 and n.endswith("__shape")
        ]
        assert not dupes, f"Duplicate '*__shape' helpers detected: {dupes}"

        path = tmp_path / "scan_ssa.onnx"
        path.write_bytes(model.SerializeToString())

        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        B = 3
        x = np.random.randn(B, 4).astype(np.float32)

        ref = np.asarray(_scan_wrapper(x))
        got = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        np.testing.assert_allclose(ref, got, rtol=1e-5, atol=1e-6)

    def test_nested_fori_scan_shapeof_ssa_regression(self, tmp_path):
        ort = pytest.importorskip("onnxruntime")

        spec = jax.ShapeDtypeStruct(("B", 4), jnp.float32)
        model = to_onnx(
            _nested_fori_scan_wrapper,
            inputs=[spec],
            opset=21,
            model_name="nested_fori_scan_shapeof_ssa_regression",
        )

        path = tmp_path / "nested_scan_ssa.onnx"
        path.write_bytes(model.SerializeToString())

        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        B = 3
        x = np.random.randn(B, 4).astype(np.float32)

        ref = np.asarray(_nested_fori_scan_wrapper(x))
        got = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        np.testing.assert_allclose(ref, got, rtol=1e-5, atol=1e-6)
