import jax.numpy as jnp
from jax2onnx import to_onnx
import os


def select_test(x, k, phi_0, select):
    base = jnp.sin(x * k + phi_0)
    return jnp.select(
        [select == i for i in range(3)],
        [base ** (i + 1) for i in range(3)],
        jnp.zeros_like(x),
    )


if __name__ == "__main__":
    os.makedirs("onnx_models", exist_ok=True)

    # Create dummy inputs
    dummy_1d = jnp.ones((3,), dtype=jnp.float32)
    dummy_scalar = jnp.array(0.0, dtype=jnp.float32)

    # Export model
    model_name = "select_test"
    print(f"Exporting {model_name}.onnx...")
    onnx_model = to_onnx(
        select_test, inputs=[dummy_1d, dummy_scalar, dummy_scalar, dummy_scalar]
    )

    with open(f"./{model_name}.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
