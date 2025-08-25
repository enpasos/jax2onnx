# file: tests/extra_tests/test_concat_mixed_int_float_and_take.py

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax2onnx import to_onnx

# Match the repro: run with double precision enabled (to stress dtype handling)
jax.config.update("jax_enable_x64", True)


def broken():
    """
    Zero-arg pipeline exercising:
      - mixed dtypes in concatenate (f32 + i32 -> f32)
      - integer indexing via take
      - float arithmetic on the concatenated result
    Returns three tensors:
      0) concat_result: f32
      1) indexed_vals:  i32
      2) float_vals:    f32
    """
    float_arr = jnp.array([1.0, 2.0], dtype=jnp.float32)
    int_arr = jnp.array([3, 4], dtype=jnp.int32)
    concat_result = jnp.concatenate([float_arr, int_arr])  # -> f32
    lookup = jnp.array([100, 200, 300, 400, 500], dtype=jnp.int32)
    indices = jnp.clip(concat_result.astype(jnp.int32), 0, len(lookup) - 1)
    indexed_vals = jnp.take(lookup, indices)  # -> i32
    float_vals = concat_result * 1.5  # weak-typed scalar -> stays f32
    return concat_result, indexed_vals, float_vals


def _rtol_atol_for(dtype: np.dtype):
    # Tight for f64 (if it ever appears), reasonable for f32; exact for ints/bools.
    if np.issubdtype(dtype, np.floating):
        return (1e-9, 1e-12) if dtype == np.float64 else (3e-5, 1e-6)
    return (0.0, 0.0)


class TestConcatMixedTakeZeroArg:
    def test_zeroarg_concat_gather_and_arith_loads_and_matches(self):
        # 1) JAX reference and expected arity
        jax_outputs = jax.tree_util.tree_leaves(broken())
        assert len(jax_outputs) == 3, "Expected 3 outputs from `broken()`."

        # Helpful debug: jaxpr for the zero-arg function
        print("--- DEBUGGING JAXPR (broken) ---")
        print(jax.make_jaxpr(lambda: broken())())
        print("--- END DEBUGGING ---")

        # 2) Convert to ONNX (no inputs) with double precision enabled
        try:
            onnx_model = to_onnx(
                broken,
                inputs=[],
                model_name="concat_mixed_int_float_and_take_zeroarg",
                enable_double_precision=True,
            )
        except Exception as e:
            pytest.fail(f"ONNX conversion failed unexpectedly: {e}")

        # 3) Output-count check
        num_onnx_outputs = len(onnx_model.graph.output)
        print(f"ONNX model has {num_onnx_outputs} outputs.")
        assert num_onnx_outputs == len(
            jax_outputs
        ), f"Output count mismatch! JAX: {len(jax_outputs)}, ONNX: {num_onnx_outputs}."

        # 4) ONNX Runtime numeric check
        print("Output count is correct. Proceeding to validate ONNX Runtime.")
        try:
            import onnxruntime as ort

            sess = ort.InferenceSession(onnx_model.SerializeToString())

            # zero-arg model → no inputs to feed
            onnx_results = sess.run(None, {})
            assert len(onnx_results) == len(jax_outputs)

            for i, (j, o) in enumerate(zip(jax_outputs, onnx_results)):
                j_np = np.asarray(j)
                o_np = np.asarray(o)

                # Shapes must match
                assert (
                    j_np.shape == o_np.shape
                ), f"[out {i}] shape mismatch: JAX={j_np.shape} ORT={o_np.shape}"

                # Dtypes: compare numerically; ints/bools must be exact
                if np.issubdtype(j_np.dtype, np.floating) or np.issubdtype(
                    o_np.dtype, np.floating
                ):
                    # Compare at the higher precision between the two
                    tol_dtype = np.result_type(j_np.dtype, o_np.dtype)
                    rtol, atol = _rtol_atol_for(tol_dtype)
                    np.testing.assert_allclose(
                        j_np.astype(tol_dtype, copy=False),
                        o_np.astype(tol_dtype, copy=False),
                        rtol=rtol,
                        atol=atol,
                        err_msg=f"[out {i}] float mismatch (rtol={rtol}, atol={atol})",
                    )
                else:
                    np.testing.assert_array_equal(
                        j_np, o_np, err_msg=f"[out {i}] exact (int/bool) mismatch"
                    )

        except ImportError:
            print("Skipping ONNX Runtime validation because it is not installed.")
        except Exception as e:
            pytest.fail(f"ONNX Runtime validation failed: {e}")
