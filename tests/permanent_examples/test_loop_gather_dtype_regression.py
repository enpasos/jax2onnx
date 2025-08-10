import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import onnxruntime as ort

from jax2onnx.converter.user_interface import to_onnx


def _fn_loop_gather_const_f32():
    # Captured constant with explicit f32 (important!)
    table_f32 = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
    # Use a 2-D index so backends tend to lower to GatherND
    idx = jnp.array([[1, 2]], dtype=jnp.int64)  # points at scalar table_f32[1, 2]

    def body(carry, _):
        g = table_f32[idx[0, 0], idx[0, 1]]  # scalar from f32 table (g.dtype == f32)
        # also produce a double-valued path so double mode is exercised
        z = carry + jnp.asarray(0.0, dtype=jnp.float64)
        return z, (g, z)  # y0: f32 stacked, y1: f64 stacked

    def fn():
        c0 = jnp.array(0.0, dtype=jnp.float64)
        ys = lax.scan(body, c0, xs=None, length=3)[1]
        return ys[0], ys[1]  # (f32[3], f64[3])

    return fn


def test_loop_gather_const_f32_loads_and_runs(tmp_path):
    fn = _fn_loop_gather_const_f32()

    # Export with double mode enabled (this used to provoke a float/double clash in Loop body)
    model = to_onnx(
        fn, inputs=[], enable_double_precision=True, opset=21, model_name="loop_gather_dtype"
    )
    p = tmp_path / "loop_gather_dtype.onnx"
    p.write_bytes(model.SerializeToString())

    # Historically failed here with:
    # [TypeInferenceError] ... gathernd_0 ... Type (tensor(float)) ... expected (tensor(double))
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])

    # JAX reference
    y0_ref, y1_ref = fn()
    y0_ref = np.asarray(jax.device_get(y0_ref))  # f32[3]
    y1_ref = np.asarray(jax.device_get(y1_ref))  # f64[3]

    # ORT run (no inputs; model is pure function)
    y0_onnx, y1_onnx = sess.run(None, {})

    assert y0_onnx.dtype == np.float32
    assert y1_onnx.dtype == np.float64
    np.testing.assert_allclose(y0_onnx, y0_ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(y1_onnx, y1_ref, rtol=1e-12, atol=1e-12)
