# tests/extra_tests/test_issue_203_linear_transpose_transpose.py

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax2onnx.user_interface import to_onnx


@jax.jit
def _issue_203_linear_transpose_of_transpose(
    x: jnp.ndarray, cotangent: jnp.ndarray
) -> jnp.ndarray:
    def fn(y: jnp.ndarray) -> jnp.ndarray:
        return jnp.transpose(y, (1, 0))

    (x_bar,) = jax.linear_transpose(fn, x)(cotangent)
    return x_bar


def test_issue_203_linear_transpose_transpose_exports_with_ir_pipeline():
    to_onnx(
        _issue_203_linear_transpose_of_transpose,
        inputs=[(2, 3), (3, 2)],
        model_name="issue_203_linear_transpose_transpose",
    )
