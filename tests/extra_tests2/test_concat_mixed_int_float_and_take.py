from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from jax2onnx.user_interface import to_onnx


def _broken_zero_arg() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Minimal repro for mixed-dtype concat feeding integer indexing."""
    float_arr = jnp.array([1.0, 2.0], dtype=jnp.float32)
    int_arr = jnp.array([3, 4], dtype=jnp.int32)
    concat_result = jnp.concatenate([float_arr, int_arr])  # -> float32
    lookup = jnp.array([100, 200, 300, 400, 500], dtype=jnp.int32)
    indices = jnp.clip(concat_result.astype(jnp.int32), 0, len(lookup) - 1)
    indexed_vals = jnp.take(lookup, indices)  # -> int32
    float_vals = concat_result * 1.5  # scalar promotes to float32
    return concat_result, indexed_vals, float_vals


def _rtol_atol_for(dtype: np.dtype) -> tuple[float, float]:
    if np.issubdtype(dtype, np.floating):
        return (1e-9, 1e-12) if dtype == np.float64 else (3e-5, 1e-6)
    return (0.0, 0.0)


def test_zeroarg_concat_gather_and_arith_matches_onnx_ir():
    original_x64 = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        jax_outputs = list(map(np.asarray, _broken_zero_arg()))
        assert len(jax_outputs) == 3

        onnx_model = to_onnx(
            _broken_zero_arg,
            inputs=[],
            model_name="concat_mixed_int_float_and_take_zeroarg",
            enable_double_precision=True,
            use_onnx_ir=True,
        )

        assert len(onnx_model.graph.output) == len(jax_outputs)

        ort = pytest.importorskip("onnxruntime")
        session = ort.InferenceSession(onnx_model.SerializeToString())
        onnx_results = [np.asarray(arr) for arr in session.run(None, {})]
        assert len(onnx_results) == len(jax_outputs)

        for idx, (expected, got) in enumerate(zip(jax_outputs, onnx_results)):
            assert expected.shape == got.shape, (
                f"[out {idx}] shape mismatch: JAX={expected.shape} ORT={got.shape}"
            )
            if np.issubdtype(expected.dtype, np.floating) or np.issubdtype(
                got.dtype, np.floating
            ):
                tol_dtype = np.result_type(expected.dtype, got.dtype)
                rtol, atol = _rtol_atol_for(tol_dtype)
                np.testing.assert_allclose(
                    expected.astype(tol_dtype, copy=False),
                    got.astype(tol_dtype, copy=False),
                    rtol=rtol,
                    atol=atol,
                    err_msg=f"[out {idx}] float mismatch (rtol={rtol}, atol={atol})",
                )
            else:
                np.testing.assert_array_equal(
                    expected, got, err_msg=f"[out {idx}] exact (int/bool) mismatch"
                )
    finally:
        jax.config.update("jax_enable_x64", original_x64)
