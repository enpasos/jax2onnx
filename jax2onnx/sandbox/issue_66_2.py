import jax
import jax.numpy as jnp
import numpy as np
import onnxruntime as ort
from jax2onnx import to_onnx

jax.config.update("jax_enable_x64", True)


def broken():
    float_arr = jnp.array([1.0, 2.0], dtype=jnp.float32)
    int_arr = jnp.array([3, 4], dtype=jnp.int32)
    concat_result = jnp.concatenate([float_arr, int_arr])  # -> f32
    lookup = jnp.array([100, 200, 300, 400, 500], dtype=jnp.int32)
    indices = jnp.clip(concat_result.astype(jnp.int32), 0, len(lookup) - 1)
    indexed_vals = jnp.take(lookup, indices)  # -> i32
    float_vals = concat_result * 1.5  # -> f32
    return concat_result, indexed_vals, float_vals


def _rtol_atol_for(dtype: np.dtype):
    if np.issubdtype(dtype, np.floating):
        return (1e-9, 1e-12) if dtype == np.float64 else (3e-5, 1e-6)
    return (0.0, 0.0)


def _assert_close(jax_outs, ort_outs):
    assert len(jax_outs) == len(
        ort_outs
    ), f"Output arity mismatch: JAX={len(jax_outs)} ORT={len(ort_outs)}"
    for i, (j, o) in enumerate(zip(jax_outs, ort_outs)):
        j_np = np.asarray(j)
        o_np = np.asarray(o)

        if j_np.shape != o_np.shape:
            raise AssertionError(
                f"[out {i}] shape mismatch: JAX={j_np.shape} ORT={o_np.shape}"
            )

        # Choose tolerance based on the higher-precision float involved (if any)
        if np.issubdtype(j_np.dtype, np.floating) or np.issubdtype(
            o_np.dtype, np.floating
        ):
            dtype_for_tol = np.result_type(j_np.dtype, o_np.dtype)
            rtol, atol = _rtol_atol_for(dtype_for_tol)
            np.testing.assert_allclose(
                j_np.astype(dtype_for_tol, copy=False),
                o_np.astype(dtype_for_tol, copy=False),
                rtol=rtol,
                atol=atol,
                err_msg=f"[out {i}] float mismatch (rtol={rtol}, atol={atol})",
            )
        else:
            # ints/bools must match exactly
            np.testing.assert_array_equal(
                j_np, o_np, err_msg=f"[out {i}] exact mismatch"
            )


if __name__ == "__main__":
    try:
        # JAX reference
        jax_outs = broken()

        # Export & save
        onnx_model = to_onnx(broken, inputs=[], enable_double_precision=True)
        with open("broken.onnx", "wb") as f:
            f.write(onnx_model.SerializeToString())

        # ORT run (no inputs)
        sess = ort.InferenceSession("broken.onnx")
        ort_outs = sess.run(None, {})

        # Numeric equivalence
        _assert_close(jax_outs, ort_outs)
        print("OK: ONNX outputs numerically match JAX.")

    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
