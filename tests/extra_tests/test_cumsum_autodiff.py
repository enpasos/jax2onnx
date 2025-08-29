# tests/regression/test_cumsum_autodiff.py
import jax
import jax.numpy as jnp
from jax2onnx import to_onnx

@jax.jit
def cumsum_vjp(x, y):
    return jax.vjp(jnp.cumsum, x)[1](y)

@jax.jit
@jax.grad
def cumsum_last_term_grad(x):
    return jnp.cumsum(x)[-1]

def test_cumsum_vjp_exports():
    to_onnx(cumsum_vjp, [(10,), (10,)])  # no exception

def test_cumsum_last_term_grad_exports():
    to_onnx(cumsum_last_term_grad, [(10,)])  # no exception
