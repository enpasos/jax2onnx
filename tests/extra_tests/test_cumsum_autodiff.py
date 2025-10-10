# tests/extra_tests/test_cumsum_autodiff.py

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax2onnx.user_interface import to_onnx


@jax.jit
def _cumsum_vjp(x: jnp.ndarray, y: jnp.ndarray):
    return jax.vjp(jnp.cumsum, x)[1](y)


@jax.jit
@jax.grad
def _cumsum_last_term_grad(x: jnp.ndarray):
    return jnp.cumsum(x)[-1]


def test_cumsum_vjp_exports_with_ir_pipeline():
    to_onnx(
        _cumsum_vjp,
        inputs=[(10,), (10,)],
        model_name="cumsum_vjp_ir",
    )


def test_cumsum_last_term_grad_exports_with_ir_pipeline():
    to_onnx(
        _cumsum_last_term_grad,
        inputs=[(10,)],
        model_name="cumsum_last_term_grad_ir",
    )
