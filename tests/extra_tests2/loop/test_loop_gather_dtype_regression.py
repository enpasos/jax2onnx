# file: tests/permanent_examples/test_loop_gather_dtype_regression.py

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import onnxruntime as ort

from jax2onnx.user_interface import to_onnx


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
        fn,
        inputs=[],
        enable_double_precision=True,
        opset=21,
        model_name="loop_gather_dtype",
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


def _fn_nested_loop_shared_table_f32():
    table = jnp.asarray([0.3, 0.6, 0.9, 1.2], dtype=jnp.float32)

    def inner(carry, _):
        idx = (carry % 4).astype(jnp.int64)
        v = table[idx]  # f32 Gather/GatherND from captured const
        c64 = jnp.asarray(carry, jnp.float64)
        y = c64 + jnp.asarray(
            0.0, jnp.float64
        )  # keep inner in f64 mode; don't force-cast v
        return carry + 1, (y, v)

    def outer(carry, _):
        # inner scan has xs=None -> lowered to Loop; `table` is captured const → Loop state
        _, ys = lax.scan(inner, carry, xs=None, length=3)
        _ = table[0] + jnp.float32(
            0.0
        )  # touch table so it's also captured by the OUTER scan
        return carry, (ys[0][-1], ys[1][-1])

    def main():
        # outer scan also has xs=None -> nested Loop; `table` captured again → outer Loop state too
        c0 = jnp.asarray(0, dtype=jnp.int32)
        _, (y64, y32) = lax.scan(outer, c0, xs=None, length=2)
        return y64, y32  # (f64[], f32[])

    return main


def test_loop_gather_const_f32_nested_shared_table(tmp_path):
    fn = _fn_nested_loop_shared_table_f32()
    model = to_onnx(
        fn,
        inputs=[],
        enable_double_precision=True,
        opset=21,
        model_name="loop_gather_dtype_nested_shared",
    )
    p = tmp_path / "loop_gather_dtype_nested_shared.onnx"
    p.write_bytes(model.SerializeToString())

    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    ref_y64, ref_y32 = map(np.asarray, fn())
    got_y64, got_y32 = [np.asarray(x) for x in sess.run(None, {})]
    assert got_y64.dtype == np.float64 and got_y32.dtype == np.float32
    np.testing.assert_allclose(got_y64, ref_y64, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(got_y32, ref_y32, rtol=1e-6, atol=1e-6)
