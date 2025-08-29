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


jax.make_jaxpr(cumsum_vjp)(jnp.ones((10,)), jnp.ones((10,)))  # works
to_onnx(cumsum_vjp, [(10,), (10,)])  # works
jax.make_jaxpr(cumsum_last_term_grad)(jnp.ones((10,)))  # works
to_onnx(cumsum_last_term_grad, [(10,)])  # fails
