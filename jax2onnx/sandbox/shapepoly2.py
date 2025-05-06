"""
Demo for working with JAX symbolic shapes in export and concat operations.
Demonstrates concatenating two tensors with dynamic batch dimension.
"""

import numpy as np
import jax
from jax import export
from jax import numpy as jnp


def f(a, b):  # a: f32[B, 1, 8], b: f32[B, 10, 8]
    """Concatenate two tensors along axis 1."""
    return jnp.concat((a, b), axis=1)


def demo_symbolic_shapes():
    """Demonstrate symbolic shapes with JAX export for concatenation."""
    # Construct symbolic dimension variables
    B_tuple = export.symbolic_shape("B")
    B = B_tuple[0]  # Extract the single dimension from the tuple
    print(f"Created symbolic dimension: B={B}")

    # Use symbolic dimensions to construct shapes
    a_shape = (B, 1, 8)
    b_shape = (B, 10, 8)
    print(f"Symbolic input shapes: a={a_shape}, b={b_shape}")

    # Export with symbolic shapes
    exported = export.export(jax.jit(f))(
        jax.ShapeDtypeStruct(a_shape, jnp.float32),
        jax.ShapeDtypeStruct(b_shape, jnp.float32),
    )

    print("Export complete!")
    print(f"Input shapes: {exported.in_avals}")
    print(f"Output shapes: {exported.out_avals}")

    # Call with concrete shapes (with B=3), without re-tracing
    batch_size = 3
    a_input = np.ones((batch_size, 1, 8), dtype=np.float32)
    b_input = np.ones((batch_size, 10, 8), dtype=np.float32) * 2

    result = exported.call(a_input, b_input)

    print(f"\nConcrete input shapes: a={a_input.shape}, b={b_input.shape}")
    print(f"Concrete output shape: {result.shape}")
    print(f"Output data (first sample, first few values): {result[0, 0:5, 0]}")

    # Try another batch size (B=5)
    batch_size = 5
    a_input = np.ones((batch_size, 1, 8), dtype=np.float32) * 3
    b_input = np.ones((batch_size, 10, 8), dtype=np.float32) * 4

    another_result = exported.call(a_input, b_input)

    print(f"\nAnother batch size (B={batch_size}):")
    print(f"Output shape: {another_result.shape}")
    print(f"Output data (first sample, first few values): {another_result[0, 0:5, 0]}")


if __name__ == "__main__":
    print("JAX Symbolic Shapes with Dynamic Batch Concatenation Demo")
    print("--------------------------------------------------------")
    demo_symbolic_shapes()
