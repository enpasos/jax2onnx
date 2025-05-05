from jax import numpy as jnp


def _my_tile(t):
    repeats_tensor = jnp.array([3, 1, 1], dtype=jnp.int32)
    return jnp.tile(t, repeats_tensor)


# t with shape (1, 1, 8)
t = jnp.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=jnp.int32)
t = t.reshape((1, 1, 8))
y = _my_tile(t)
print("t.shape:", t.shape)

# prints: t.shape: (1, 1, 8)
