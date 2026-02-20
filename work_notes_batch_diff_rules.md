# Work Notes: JAX Differentiation and Batching Rules in jax2onnx

## Overview
`jax2onnx` intercepts certain high-level JAX functions (e.g., `jax.nn.dot_product_attention`, `jnp.concatenate`) and replaces them with custom primitives (e.g., `_DPA_PRIM`, `_CONCAT_PRIM`). This is done to preserve high-level macro-operations in the resulting ONNX graph, enabling better optimization by runtimes like ONNXRuntime.

However, when users apply JAX transformations like `jax.grad` (autodiff) or `jax.vmap` (batching) to code containing these custom primitives, JAX fails with a `NotImplementedError` because it doesn't know how to trace these custom primitives.

To fix this, we must implement differentiation rules (`ad.defjvp` or `defcustom_jvp`) and batching rules (`batching.primitive_batchers`) for our custom primitives.

## Supported Approaches

There are two primary approaches for implementing these rules, and we should use a mixture of both depending on the nature of the operation.

### Approach A: Fallback to Native JAX
In this approach, the transformation rule simply delegates the operation back to the original JAX implementation. JAX then traces the original implementation down to its core lower-level primitives.

*   **How it works:** In the rule definition, we call the original JAX function (e.g., `jnp.concatenate`) instead of binding our custom primitive.
*   **When to use:** Use this for functions that decompose into native basic primitives that `jax2onnx` **already translates cleanly and optimally** (e.g., `concatenate`, `reshape`, simple math).
*   **Why use it:** It leverages JAX's robust internal logic for complex calculus, edge cases, broadcasting, and zero-tangents, saving us from writing error-prone custom math. Since `jax2onnx` can handle the resulting low-level primitives efficiently, there is no performance penalty in the final ONNX graph.
*   **Example Usage:** `jnp.concatenate` (#191). The batching rule for `_CONCAT_PRIM` already uses this fallback approach. We should implement the differentiation rule (`ad.defjvp`) using the same fallback method.

### Approach B: Explicit Primitive Binding
In this approach, we write an explicit mathematical/structural rule that defines how our custom primitive transforms, and we bind our custom primitive again within the rule.

*   **How it works:** We manually compute the transformed shapes, axes, or gradients, and then emit a new instance of our custom primitive (e.g., `_DPA_PRIM.bind(...)`).
*   **When to use:** Use this for high-level composite functions (e.g., `dot_product_attention`, `layer_norm`) where falling back to native JAX would result in a massive, fragmented graph of microscopic operations (`lax.mul`, `lax.reduce_sum`, `lax.exp`).
*   **Why use it:** Falling back would destroy the clean macro-operator abstraction that we desperately want to preserve in the ONNX graph for hardware acceleration (like FlashAttention). We *must* explicitly tell JAX how to transform our custom primitive so that the tracing engine spits out another instance of our custom primitive, preserving the abstraction.
*   **Example Usage:** `jax.nn.dot_product_attention` (#190). We need to write an explicit batching rule (`batching.primitive_batchers`) that adjusts the batch shapes and binds `_DPA_PRIM` dynamically over the expanded dimensions.

## Implementation Plan

### 1. Identify Target Custom Primitives
Identify all custom primitives defined in `jax2onnx/plugins/`. Key examples currently causing issues:
*   `_CONCAT_PRIM` in `jax2onnx/plugins/jax/numpy/concatenate.py`
*   `_DPA_PRIM` in `jax2onnx/plugins/jax/nn/dot_product_attention.py`

### 2. Implement Missing Rules

#### `jnp.concatenate` (Issue #191)
*   **Rule Needed:** Differentiation (JVP) rule.
*   **Approach:** Approach A (Fallback).
*   **Implementation Location:** `\\wsl.localhost\Ubuntu\home\enpasos\projects\jax2onnx\jax2onnx\plugins\jax\numpy\concatenate.py`
*   **Logic:** Use `jax.interpreters.ad.defjvp` to define a rule that falls back to the original `jnp.concatenate` implementation for computing the tangents.

#### `jax.nn.dot_product_attention` (Issue #190)
*   **Rule Needed:** Batching rule.
*   **Approach:** Approach B (Explicit Binding).
*   **Implementation Location:** `\\wsl.localhost\Ubuntu\home\enpasos\projects\jax2onnx\jax2onnx\plugins\jax\nn\dot_product_attention.py`
*   **Logic:** Register an entry in `jax.interpreters.batching.primitive_batchers`. Use `jax.interpreters.batching.bdim_at_front` to manage the mapped dimensions, adjust the input shapes, and then explicitly call `_DPA_PRIM.bind(...)` to maintain the operation as a single node.

### 3. Future Rule Coverage
For any newly added custom primitives in `jax2onnx` that replace high-level JAX functions:
1.  **Assess:** Does this primitive represent a high-level composite operation (like attention) or a fundamental operation (like concat)?
2.  **Decide:** Choose Approach A or B based on whether the fallback expansion produces a desirable or undesirable ONNX graph.
3.  **Implement:** Provide at least the batching rule (`vmap` support) and, if applicable, the JVP rule (`grad` support).
