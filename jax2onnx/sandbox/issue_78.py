import jax
import jax.numpy as jnp
from jax import jit
from jax2onnx import to_onnx

jax.config.update("jax_enable_x64", True)


@jit
def func(data, indices):
    data = jnp.asarray(data, dtype=jnp.float64)
    gathered = data[indices]

    result = gathered * 2.0
    result = jnp.sin(result) + jnp.cos(result)

    mask = result > 0.5
    filtered_result = jnp.where(mask, result, 0.0)

    return filtered_result


if __name__ == "__main__":
    data = jnp.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
    )

    indices = jnp.array([0, 2])

    result = func(data, indices)

    model = to_onnx(
        fn=func,
        inputs=[data, indices],
        enable_double_precision=True,
    )

    with open("model.onnx", "wb") as f:
        f.write(model.SerializeToString())
