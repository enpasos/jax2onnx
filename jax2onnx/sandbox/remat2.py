import jax
import jax.numpy as jnp
from jax2onnx import to_onnx


@jax.checkpoint
def g(x):
    y = jnp.sin(x)
    z = jnp.sin(y)
    return z


if __name__ == "__main__":
    dummy_scalar = jnp.array(2.0, dtype=jnp.float32)
    onnx_model = to_onnx(g, inputs=[dummy_scalar])
