# Handling Symbolic Dimensions in jax2onnx

This document outlines a robust, maintainable approach to handling symbolic dimensions during the conversion of JAX models to ONNX, particularly when extracting symbolic dimension values as runtime tensors.

---

## Problem Statement

Currently, when the primitive `dim_as_value` is encountered, the ONNX conversion method (`to_onnx`) lacks the necessary context (specifically, it receives empty inputs). Thus, the method cannot directly determine which tensor and dimension a symbolic dimension (e.g., `B`) originates from, leading to incorrect or arbitrary fallback behavior.

---

## Proposed Solution Overview

The key idea is to systematically record symbolic dimension metadata **at tracing time** in `jaxpr_converter.py`. This metadata maps symbolic dimensions to their original source tensors and axes.

When `dim_as_value.to_onnx` executes, it consults this pre-recorded metadata to accurately construct the required ONNX operators to extract the symbolic dimension at runtime.

---

## Detailed Implementation

### Step 1: Extending `Jaxpr2OnnxConverter`

Modify the `Jaxpr2OnnxConverter` class to include a new attribute:

```python
from jax._src.export.shape_poly import _DimExpr

class Jaxpr2OnnxConverter:
    # Map symbolic dimensions to their origin tensor names and axes
    symbolic_dim_to_origin: dict[_DimExpr, tuple[str, int]]

    def __init__(self, ...):
        self.symbolic_dim_to_origin = {}
```

### Step 2: Populating Symbolic Dimension Metadata

At the beginning of `trace_jaxpr()`, loop over all input variables and symbolic shapes, recording symbolic dimensions:

```python
def trace_jaxpr(self, ...):
    ...
    for input_var, input_spec in zip(self.jaxpr.jaxpr.invars, symbolic_avals):
        shape = input_spec.shape
        tensor_name = self.get_name(input_var)
        for axis, dim in enumerate(shape):
            if isinstance(dim, _DimExpr):
                self.symbolic_dim_to_origin[dim] = (tensor_name, axis)
```

Example:

If your input is defined as:
```python
a:f32[B, 64, 14, 14]
```
The mapping will store:
```python
symbolic_dim_to_origin = { B: ('a', 0) }
```

---

### Step 3: Adjusting `dim_as_value.to_onnx`

In the ONNX export for `dim_as_value`, use the recorded metadata to construct the runtime extraction:

```python
def to_onnx(s: Jaxpr2OnnxConverter, node_inputs, node_outputs, params):
    out_var = node_outputs[0]
    out_name = s.get_name(out_var)
    dim_expr = params["dim"]

    if dim_expr not in s.symbolic_dim_to_origin:
        raise ValueError(f"No origin tensor found for symbolic dimension: {dim_expr}")

    source_name, axis = s.symbolic_dim_to_origin[dim_expr]

    # Shape extraction subgraph
    shape_node = s.get_unique_name("shape_of_tensor")
    s.add_node(helper.make_node("Shape", inputs=[source_name], outputs=[shape_node]))
    s.add_shape_info(shape_node, (len(s.builder.value_info_metadata[source_name]["shape"]),), np.int64)

    axis_const = s.get_constant_name(np.array([axis], np.int64))
    gather_node = s.get_unique_name("gather_dim")
    s.add_node(helper.make_node("Gather", inputs=[shape_node, axis_const], outputs=[gather_node], axis=0))
    s.add_shape_info(gather_node, (1,), np.int64)

    squeeze_node = s.get_unique_name("squeezed_dim")
    s.add_node(helper.make_node("Squeeze", inputs=[gather_node], outputs=[squeeze_node], axes=[0]))
    s.add_shape_info(squeeze_node, (), np.int64)

    s.add_node(helper.make_node("Cast", inputs=[squeeze_node], outputs=[out_name], to=int(TensorProto.INT32)))
    s.add_shape_info(out_name, (), np.int32)
```

---

## Benefits of This Approach

- **Robustness:** Eliminates arbitrary constants and fallbacks.
- **Maintainability:** Centralized dimension metadata management.
- **Traceability:** Clear and reproducible mapping from symbolic dimensions to tensors.
- **Extensibility:** Easily accommodates new scenarios requiring symbolic dimension handling.

---

## Error Handling

- If a symbolic dimension is unresolved (missing from `symbolic_dim_to_origin`), an explicit exception is raised:

```python
raise ValueError(f"Symbolic dimension '{dim_expr}' has no registered input origin.")
```

This avoids silent failures and makes debugging straightforward.

---

## Recommendations for Developers

- Regularly validate symbolic dimension handling through comprehensive test coverage.
- Ensure new primitives or plugins that require symbolic dimension information correctly reference the central metadata.
- Clearly document any changes to symbolic dimension logic in your developer guide.

---

Following this approach will ensure accurate, efficient, and maintainable handling of symbolic dimensions within your `jax2onnx` conversions.

