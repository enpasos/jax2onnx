import jax.numpy as jnp
from jax2onnx import to_onnx


def create_scatter_function():
    def scatter_model(operand, indices, updates):
        result = operand.at[:, 5:261, 5:261, :].set(updates)
        return result

    return scatter_model


def main():
    scatter_fn = create_scatter_function()

    operand_shape = (5, 266, 266, 1)
    updates_shape = (5, 256, 256, 1)

    operand = jnp.zeros(operand_shape, dtype=jnp.float64)
    indices = jnp.array([[5, 5]], dtype=jnp.int32)
    updates = jnp.ones(updates_shape, dtype=jnp.float64)

    try:
        scatter_fn(operand, indices, updates)
    except Exception as e:
        print(f"JAX function failed: {e}")
        return

    try:
        to_onnx(
            scatter_fn,
            [operand, indices, updates],
            model_name="scatter_error_example",
            enable_double_precision=True,
        )
    except Exception as e:
        print("ONNX conversion failed with error:")
        print(f"  {type(e).__name__}: {e}")

        # Print the specific shape mismatch details
        if "Updates element count mismatch" in str(e):
            print("\nShape mismatch details:")
            print(f"  JAX updates shape: {updates_shape}")
            print("  Expected ONNX shape: (5, 266, 266, 1, 1)")
            print(f"  Operand shape: {operand_shape}")


if __name__ == "__main__":
    main()
