"""
Demo for working with JAX symbolic shapes in export and concat operations.
"""

import numpy as np
import jax
from jax import export
from jax import numpy as jnp


def f(x):  # f: f32[a, b]
    """Concatenate input with itself along axis 1."""
    return jnp.concat([x, x], axis=1)


def demo_symbolic_shapes():
    """Demonstrate symbolic shapes with JAX export."""
    # Construct symbolic dimension variables
    a, b = export.symbolic_shape("a, b")
    print(f"Created symbolic dimensions: a={a}, b={b}")

    # Use symbolic dimensions to construct shapes
    x_shape = (a, b)
    print(f"Symbolic input shape: {x_shape}")

    # Export with symbolic shapes
    exported = export.export(jax.jit(f))(jax.ShapeDtypeStruct(x_shape, jnp.int32))

    print("Export complete!")
    print(f"Input shapes: {exported.in_avals}")
    print(f"Output shapes: {exported.out_avals}")

    # Call with concrete shapes (with a=3 and b=4), without re-tracing
    concrete_input = np.ones((3, 4), dtype=np.int32)
    result = exported.call(concrete_input)

    print(f"\nConcrete input shape: {concrete_input.shape}")
    print(f"Concrete output shape: {result.shape}")
    print(f"Output data (first row): {result[0]}")

    # Try another shape (a=5, b=2)
    another_input = np.ones((5, 2), dtype=np.int32) * 2
    another_result = exported.call(another_input)

    print(f"\nAnother input shape: {another_input.shape}")
    print(f"Another output shape: {another_result.shape}")
    print(f"Output data (first row): {another_result[0]}")


if __name__ == "__main__":
    print("JAX Symbolic Shapes with Concatenation Demo")
    print("-------------------------------------------")
    demo_symbolic_shapes()
