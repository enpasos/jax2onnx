"""
Demo for working with JAX symbolic shapes in export and concat operations.
Demonstrates concatenating two tensors with dynamic batch dimension.
"""

import jax
from jax import export
from jax import numpy as jnp


def f(a, b):  # a: f32[B, 1, 8], b: f32[B, 10, 8]
    """Concatenate two tensors along axis 1."""
    return jnp.concatenate((a, b), axis=1)


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


if __name__ == "__main__":
    print("JAX Symbolic Shapes with Dynamic Batch Concatenation Demo")
    print("--------------------------------------------------------")
    demo_symbolic_shapes()
