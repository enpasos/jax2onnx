# Symbolic Dimensions in jax2onnx

Symbolic dimensions allow JAX models to handle inputs with flexible shapes at runtime, essential for dynamic batching and variable-length sequences. The most common use case is representing dynamic batch sizes symbolically (e.g., "B").

This document explains how symbolic dimensions, particularly batch dimensions, are implemented in the jax2onnx converter, highlighting key ideas and their relevance to plugin developers.

---

## Core Idea and Implementation

The core idea behind symbolic dimensions in `jax2onnx` is aligning JAX's symbolic shape polymorphism concept with ONNX's symbolic dimension support.

- **JAX Shape Polymorphism**: JAX enables specifying symbolic dimensions using shape polymorphism. See the official JAX documentation [Shape Polymorphism](https://docs.jax.dev/en/latest/export/shape_poly.html).

- **ONNX Symbolic Dimensions**: ONNX supports symbolic dimensions natively by assigning symbolic names instead of fixed numeric values.

**jax2onnx bridges these concepts by:**

- Mapping JAX symbolic dimension names directly to ONNX symbolic dimension parameters (`dim_param`).
- Ensuring symbolic dimension names defined in JAX are preserved exactly in the exported ONNX model.

## Main Components in jax2onnx

### 1. Symbolic Dimension Tracking

- `jax2onnx.converter.onnx_builder.OnnxBuilder`
  - Maintains a map (`var_to_symbol_map`) linking JAX symbolic dimensions to ONNX symbolic dimension parameters.
  - Resolves and preserves symbolic dimension names during conversion.

### 2. Conversion Process (`to_onnx`)

- Uses JAX's shape polymorphism for input shapes.
- Abstracts dimensions using symbolic identifiers, mapping them to ONNX symbolic dimensions.

### Example Conversion Call:

```python
model = to_onnx(fn=lambda x: jnp.squeeze(x, axis=(-1, -3)), inputs=[(1, "B", 1)])
```

This example defines a symbolic batch dimension named "B". jax2onnx ensures this symbolic dimension is directly preserved in the resulting ONNX model.

---

## Testing Symbolic Dimension Preservation

A typical test to ensure symbolic dimensions are preserved looks like this:

```python
import onnx
import jax.numpy as jnp
from jax2onnx import to_onnx


def test_symbolic_batch_dim_is_preserved():
    # Use abstracted axes with a symbolic name "B"
    model = to_onnx(fn=lambda x: jnp.squeeze(x, axis=(-1, -3)), inputs=[(1, "B", 1)])

    # Extract the input tensor from the ONNX model
    input_tensor = model.graph.input[0]
    input_shape = input_tensor.type.tensor_type.shape
    dim_param = input_shape.dim[1].dim_param

    # Assert that the symbolic dimension is preserved
    assert dim_param == "B", f"Expected symbolic dim 'B', got: {dim_param}"
```

---

## Information for Plugin Developers

Plugin developers extending `jax2onnx` should keep the following in mind regarding symbolic dimensions:

- When implementing custom primitive handlers (`PrimitivePlugin`), always propagate symbolic dimension metadata from inputs to outputs.
- Use provided utilities from `OnnxBuilder` to register value metadata, ensuring correct symbolic dimension tracking.
- Always test custom handlers with symbolic dimensions to confirm dimension names are preserved accurately.

---

## Related Resources

For a deeper understanding of symbolic shapes in JAX:

- [JAX Shape Polymorphism](https://docs.jax.dev/en/latest/export/shape_poly.html)

This ensures alignment between JAX and ONNX symbolic dimensions, facilitating flexible and dynamic model deployment.

