import jax
from jax import numpy as jnp
from jax2onnx import to_onnx


@jax.jit
def cumsum_vjp(x, y):
    return jax.vjp(jnp.cumsum, x)[1](y)


@jax.jit
@jax.grad
def cumsum_last_term_grad(x):
    return jnp.cumsum(x)[-1]


to_onnx(cumsum_vjp, [(10,), (10,)])  # fails
to_onnx(cumsum_last_term_grad, [(10,)])  # fails
