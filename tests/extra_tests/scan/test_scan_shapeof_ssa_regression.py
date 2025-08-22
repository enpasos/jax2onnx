# tests/extra_tests/scan/test_scan_shapeof_ssa_regression.py

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from jax import lax
from jax2onnx import to_onnx

# We’ll also add a local SSA guard by monkeypatching the builder,
# so we fail at export time if duplicate output names appear.
from jax2onnx.converter.onnx_builder import OnnxBuilder


def _ssa_guard_builder(monkeypatch):
    orig_add_node = OnnxBuilder.add_node

    def add_node_ssa(self, node):
        # check dup outputs in this graph _before_ adding
        existing = {o for n in self.nodes for o in n.output}
        dups = [o for o in node.output if o in existing]
        if dups:
            raise AssertionError(
                f"[SSA regression] Duplicate output symbol(s): {dups} in node {node.name or node.op_type}"
            )
        return orig_add_node(self, node)

    monkeypatch.setattr(OnnxBuilder, "add_node", add_node_ssa, raising=True)


# Body that creates a single 'add' and queries its shape multiple times
def _scan_body_multi_shape(carry, _):
    a = carry
    add = a + 1.0  # the producer whose shape will be requested repeatedly

    # two broadcasts that each need Shape(add)
    z0 = add + jnp.broadcast_to(0.0, (add.shape[0], add.shape[1]))
    z1 = add + jnp.broadcast_to(1.0, (add.shape[0], add.shape[1]))

    # a reshape that consults Shape(add) again
    z2 = jnp.reshape(add, (add.shape[0], add.shape[1]))

    # return a scan output too, to mimic real scan lowering
    return add, (z0 + z1 + z2)


def _scan_wrapper(x):
    # 2 steps, no xs to keep things small; produces Loop with scan outputs
    carry_out, ys = lax.scan(_scan_body_multi_shape, x, xs=None, length=2)
    # Return both a carried state and a scan output to stress the body graph
    return carry_out + ys[-1]


def _nested_fori_scan_wrapper(x):
    # Put the scan in a fori_loop body to mirror feed-forward style composition
    def body(_, state):
        return _scan_wrapper(state)

    return lax.fori_loop(0, 2, body, x)


class TestScanShapeOfSSARegression:
    def test_scan_shapeof_ssa_regression(self, tmp_path, monkeypatch):
        _ssa_guard_builder(monkeypatch)

        spec = jax.ShapeDtypeStruct(("B", 4), jnp.float32)

        model = to_onnx(
            _scan_wrapper,
            inputs=[spec],
            loosen_internal_shapes=True,  # increase Shape() usage
            opset=21,
            model_name="scan_shapeof_ssa_regression",
        )

        # Try to load and run in ORT – should succeed (will fail on the bug)
        try:
            import onnxruntime as ort
        except Exception:
            pytest.skip("onnxruntime not installed")

        path = tmp_path / "m.onnx"
        path.write_bytes(model.SerializeToString())

        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        B = 3
        x = np.random.randn(B, 4).astype(np.float32)

        ref = np.asarray(_scan_wrapper(x))
        got = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        np.testing.assert_allclose(ref, got, rtol=1e-5, atol=1e-6)

    def test_nested_fori_scan_shapeof_ssa_regression(self, tmp_path, monkeypatch):
        _ssa_guard_builder(monkeypatch)

        spec = jax.ShapeDtypeStruct(("B", 4), jnp.float32)

        model = to_onnx(
            _nested_fori_scan_wrapper,
            inputs=[spec],
            loosen_internal_shapes=True,
            opset=21,
            model_name="nested_fori_scan_shapeof_ssa_regression",
        )

        try:
            import onnxruntime as ort
        except Exception:
            pytest.skip("onnxruntime not installed")

        path = tmp_path / "m2.onnx"
        path.write_bytes(model.SerializeToString())

        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        B = 3
        x = np.random.randn(B, 4).astype(np.float32)

        ref = np.asarray(_nested_fori_scan_wrapper(x))
        got = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        np.testing.assert_allclose(ref, got, rtol=1e-5, atol=1e-6)
