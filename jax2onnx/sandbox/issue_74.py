import jax.numpy as jnp
import jax
from jax2onnx import to_onnx

jax.config.update("jax_enable_x64", True)


def concatenate(data_expanded, identity_values):
    return jnp.concatenate([data_expanded, identity_values], axis=1, dtype=jnp.float32)


# Example usage:
if __name__ == "__main__":

    data_reshaped = jnp.array([5], dtype=jnp.int32).reshape(1)
    data_expanded = jnp.broadcast_to(data_reshaped, (5, 1))
    identity_values = jnp.arange(5, dtype=jnp.int32).reshape(5, 1)

    model = to_onnx(
        fn=concatenate,
        inputs=[data_expanded, identity_values],
        enable_double_precision=True,
    )

    with open("model.onnx", "wb") as f:
        f.write(model.SerializeToString())
