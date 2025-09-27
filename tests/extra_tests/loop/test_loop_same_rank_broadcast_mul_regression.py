import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax import lax
from jax2onnx import to_onnx

try:
    import onnxruntime as ort

    HAS_ORT = True
except Exception:
    HAS_ORT = False


def _inner_scan_body(carry, _):
    # same-rank broadcast: carry is (B, 4, 5); weight is (1, 4, 5)
    w = jnp.ones((1, 4, 5), carry.dtype) * 2.0
    out = carry * w
    return out, out


def _outer_fori_body(i, state):
    _, ys = lax.scan(_inner_scan_body, state, xs=None, length=2)
    return ys[-1]


def ff_like(y0):
    return lax.fori_loop(0, 1, _outer_fori_body, y0)


@pytest.mark.skipif(not HAS_ORT, reason="onnxruntime not installed")
@pytest.mark.parametrize("dtype", [jnp.float64])  # mirrors integration (double)
def test_loop_same_rank_broadcast_mul_regression(dtype):
    spec = jax.ShapeDtypeStruct(("B", 4, 5), dtype)

    model = to_onnx(
        ff_like,
        inputs=[spec],
        enable_double_precision=True,
        opset=21,
        model_name="same_rank_broadcast_mul_regression",
    )

    # Build ORT session (this used to fail with TypeInferenceError before the fix)
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )

    # Numeric check against JAX
    x = np.random.rand(2, 4, 5).astype(np.float64)
    y_ref = np.asarray(ff_like(jnp.asarray(x, dtype=dtype)))

    feeds = {sess.get_inputs()[0].name: x}
    (y_onnx,) = sess.run(None, feeds)
    np.testing.assert_allclose(y_onnx, y_ref, rtol=1e-7, atol=1e-12)
