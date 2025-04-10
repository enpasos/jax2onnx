import jax.numpy as jnp
from jax import eval_shape
from jax.core import ShapedArray

# ShapedArray inputs (same as ShapeDtypeStruct for eval_shape)
a = ShapedArray((2, 3), jnp.float32)
b = ShapedArray((4, 3), jnp.float32)

# Wrap them in a list and pass to jnp.concatenate
abstract_args = ([a, b],)  # Single positional argument: sequence of arrays
kwargs = {"axis": 0}

result = eval_shape(jnp.concatenate, *abstract_args, **kwargs)

print(result.shape)  # (6, 3)
print(result.dtype)  # float32
