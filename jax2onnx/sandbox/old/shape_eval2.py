import jax.numpy as jnp
from jax import eval_shape
from jax.core import ShapedArray

# ShapedArray inputs
a = ShapedArray((2, 3), jnp.float32)
b = ShapedArray((4, 3), jnp.float32)


# Use a lambda to fix axis=0
def concat_fn(arrays):
    return jnp.concatenate(arrays, axis=0)


result = eval_shape(concat_fn, [a, b])

print(result.shape)  # Expected output: (6, 3)
print(result.dtype)  # Expected output: float32
