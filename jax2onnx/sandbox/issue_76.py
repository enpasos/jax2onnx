import jax
import jax.numpy as jnp

from jax2onnx import to_onnx

jax.config.update("jax_enable_x64", True)

def func():
    a = jnp.array([1], dtype=jnp.int32)  # Keep as int32
    b = jnp.array([2], dtype=jnp.int32)  # Keep as int32
    return jax.lax.concatenate([a,b], dimension=0).astype(jnp.float32)

if __name__ == "__main__":
    model = to_onnx(
        fn=func,
        inputs=[],
        enable_double_precision=True,
    )
    with open("model.onnx", "wb") as f:
        f.write(model.SerializeToString())