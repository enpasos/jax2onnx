from jax import eval_shape
from jax.core import ShapedArray
import jax.numpy as jnp
from functools import partial

# ShapedArray inputs
a = ShapedArray((2, 3), jnp.float32)
b = ShapedArray((4, 3), jnp.float32)

# Bind the static argument 'axis' using partial
concat_fn = partial(jnp.concatenate, axis=0)

# Now pass the sequence of arrays as the single argument
result = eval_shape(concat_fn, [a, b])

print(result.shape)  # Expected output: (6, 3)
print(result.dtype)  # Expected output: float32
