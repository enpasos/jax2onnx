# tests/extra_tests/loop/test_loop_shapeof_ssa_regression.py

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from jax2onnx.user_interface import to_onnx


def loop_body_ssa_bug(x0):
    """
    Minimal fori_loop body that triggers multiple Shape-of requests for
    the SAME tensor ('Add' output) via two different consumers:

      1) reshape back to z.shape  -> needs Shape(z)
      2) zeros(z.shape)           -> needs Shape(z)

    On buggy converters, both paths materialize a Shape node with the same
    output name like 'Add__shape', violating SSA inside the Loop body.
    """

    def body_fun(i, carry):
        z = carry + 1.0  # (B, 4)  -- creates an 'Add' node
        a = jnp.reshape(z, (z.shape[0], -1))  # (B, 4)  -- shape-of(z) used here
        b = jnp.reshape(a, z.shape)  # (B, 4)  -- shape-of(z) used again
        dummy = jnp.zeros(z.shape, z.dtype)  # (B, 4)  -- shape-of(z) used again
        out = b + dummy  # keep both paths alive
        return out

    # 2 iterations to ensure we hit the Loop path in ONNX
    return jax.lax.fori_loop(0, 2, body_fun, x0)


class TestLoopShapeOfSSARegression:
    def test_loop_shapeof_ssa_regression(self):
        # Dynamic first dim to force runtime Shape() use
        input_spec = jax.ShapeDtypeStruct(("B", 4), jnp.float32)

        # Build ONNX with loosened internal shapes to maximize Shape() usage
        model = to_onnx(
            loop_body_ssa_bug,
            inputs=[input_spec],
            model_name="loop_shapeof_ssa_regression",
            opset=21,
        )

        # Try loading with ORT: buggy builds throw INVALID_GRAPH with "...__shape"
        ort = pytest.importorskip("onnxruntime")

        # Save & load
        serialized = model.SerializeToString()
        sess = ort.InferenceSession(serialized)

        # Quick numeric check (sanity): ORT == JAX
        x = np.random.randn(3, 4).astype(np.float32)  # B=3 at runtime
        y_jax = np.asarray(loop_body_ssa_bug(x))
        y_onnx = np.asarray(sess.run(None, {sess.get_inputs()[0].name: x})[0])
        np.testing.assert_allclose(y_jax, y_onnx, rtol=1e-6, atol=1e-6)
