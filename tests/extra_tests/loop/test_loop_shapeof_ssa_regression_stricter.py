# tests/extra_tests/loop/test_loop_shapeof_ssa_regression_stricter.py

import numpy as np
import jax
import jax.numpy as jnp
import pytest
from jax import lax
from jax2onnx.user_interface import to_onnx


# Body forces multiple Shape-of on the SAME intermediate 'add'
def _body_with_multi_shape(i, a):
    add = a + 1.0
    # two broadcasts via add.shape
    b1 = jnp.broadcast_to(0.0, (add.shape[0], add.shape[1]))
    z1 = add + b1
    b2 = jnp.broadcast_to(1.0, (add.shape[0], add.shape[1]))
    z2 = add + b2
    # and a reshape that consults the same shape again
    z3 = jnp.reshape(add, (add.shape[0], add.shape[1]))
    return z1 + z2 + z3


def _model(x):
    return lax.fori_loop(0, 2, _body_with_multi_shape, x)


class TestLoopShapeOfSSARegressionStricter:
    def test_loop_shapeof_ssa_and_ort(self, tmp_path, monkeypatch):
        # turn on early SSA assertion in builder so we fail at export-time if broken
        monkeypatch.setenv("JAX2ONNX_ASSERT_SSA", "1")

        spec = jax.ShapeDtypeStruct(("B", 4), jnp.float32)
        model = to_onnx(
            _model,
            inputs=[spec],
            opset=21,
            model_name="loop_shapeof_ssa_stricter",
        )
        onnx_path = tmp_path / "m.onnx"
        onnx_path.write_bytes(model.SerializeToString())

        # ORT should load and match numerics
        try:
            import onnxruntime as ort
        except Exception:
            pytest.skip("onnxruntime not installed")

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        B = 3
        x = np.random.randn(B, 4).astype(np.float32)
        ref = np.asarray(_model(x))
        got = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        np.testing.assert_allclose(ref, got, rtol=1e-5, atol=1e-6)
