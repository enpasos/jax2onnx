# check_irfft.py

import jax
import jax.numpy as jnp


def f(x):
    return jnp.fft.irfft(x)


x = jnp.zeros((5,), dtype=jnp.complex64)
print(jax.make_jaxpr(f)(x))
