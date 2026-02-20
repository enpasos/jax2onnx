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

## Review of This Note (2026-02-20)

I agree with the core direction and with both approaches.

Two clarifications are important:

1.  Not every primitive needs a gradient rule.
    - Random and indexing/selection style ops can be intentionally non-differentiable.
    - For these, we should document "no grad support by design" and avoid accidental regressions.
2.  "grad support" can require more than one mechanism in complex cases.
    - `ad.defjvp` fallback is usually the fastest unblocker.
    - For high-level macro primitives (for example attention), we may later need a stricter explicit rule strategy to preserve macro structure in exported graphs.

## Rule Coverage Snapshot (Current)

Quick scan scope:
- `jax2onnx/plugins/jax/numpy/*.py`
- `jax2onnx/plugins/jax/nn/*.py`
- `jax2onnx/plugins/jax/random/*.py`
- Files defining custom primitives (`make_jnp_primitive(...)` or `Primitive(...)`).

Current status:
- Total custom-primitive modules scanned: `60`
- With batching rule (`primitive_batchers` or unary helper): `51` (`85.0%`)
- With explicit autodiff rule (`ad.defjvp` / `primitive_jvps` / transpose/linear rules): `0` (`0.0%`)

By area:
- `jax/numpy`: `35` total, `30` batching, `0` autodiff
- `jax/nn`: `22` total, `21` batching, `0` autodiff
- `jax/random`: `3` total, `0` batching, `0` autodiff

## Verified Conversion Gaps

Direct `to_onnx(...)` checks confirm the rule gap:
- `grad(jnp.concatenate)` -> `NotImplementedError` (missing differentiation rule)
- `grad(jax.nn.dot_product_attention)` -> `NotImplementedError`
- `vmap(jax.nn.dot_product_attention)` -> `NotImplementedError` (missing batching rule)
- `grad(jnp.reshape)` -> `NotImplementedError`
- `grad(jax.nn.relu)` -> `NotImplementedError`

## Coverage List (Prioritized)

| Primitive / Group | Current vmap | Current grad | Recommended Approach | Priority | Action |
|---|---|---|---|---|---|
| `jax.numpy.concatenate` (`_CONCAT_PRIM`) | Yes | No | A (fallback) | P0 | Add `ad.defjvp` fallback rule to original `jnp.concatenate`. |
| `jax.nn.dot_product_attention` (`_DPA_PRIM`) | No | No | B for vmap, then staged grad plan | P0 | Add explicit batching rule with `_DPA_PRIM.bind(...)`; add grad unblocker path after vmap is stable. |
| Shape-only numpy wrappers (`reshape`, `squeeze`, `transpose`, `stack`, `split`, `tile`) | Mostly Yes | No | A (fallback) | P0 | Add shared autodiff helper for shape ops to reduce repeated code. |
| Core nn activations (`relu`, `sigmoid`, `silu`, `gelu`, `elu`, `leaky_relu`, `softplus`, `softsign`, `selu`, `celu`, `mish`) | Yes | No | A (fallback) | P1 | Add a common unary autodiff registration helper and apply to all differentiable activations. |
| Reduction-like wrappers (`mean`, `prod`, `logsumexp`, `log_softmax`, `softmax`, `standardize`) | Mostly Yes | No | A first, B only if graph quality degrades | P1 | Add autodiff fallback rules; verify ONNX graph quality with focused tests. |
| Non-differentiable by nature (`hardmax`, `one_hot`, random samplers) | Mixed | No | Explicit non-goal | P1 | Document as "no grad support by design" and add policy tests for clear error behavior. |
| Missing batching in deterministic numpy ops (`arange`, `compress`, `eye`, `linalg_det`, `windows`) | No | No | A (fallback) | P2 | Add batching rules only where real model demand exists. |
| Random samplers (`bernoulli`, `categorical`, `normal`) | No | No | Usually non-goal for grad, selective for vmap | P2 | Keep grad unsupported; decide whether per-op vmap support is needed for target workloads. |

## Suggested Execution Order

1.  Land P0 for `concatenate` and `dot_product_attention`.
2.  Introduce a shared fallback-autodiff utility for common unary and shape ops.
3.  Mark explicit non-goals (non-differentiable primitives) in docs/tests.
4.  Expand batching only for demand-driven primitives with missing rules.

## Implementation Status for #190 and #191

Implemented in this branch:

- `#191` (`jax.numpy.concatenate` under `jax.grad`)
  - Added primitive autodiff/JVP registration for `_CONCAT_PRIM` in
    `jax2onnx/plugins/jax/numpy/concatenate.py`.
  - Rule uses fallback-to-native behavior (original `jnp.concatenate`) for
    primal and tangent computation.
  - Verified with conversion regression:
    `to_onnx(jax.grad(lambda x: jnp.sum(jnp.concatenate((x, x), axis=0))))`.

- `#190` (`jax.nn.dot_product_attention` under `jax.vmap`)
  - Added explicit batching rule for `_DPA_PRIM` in
    `jax2onnx/plugins/jax/nn/dot_product_attention.py` via
    `batching.primitive_batchers`.
  - Rule keeps `_DPA_PRIM.bind(...)` in the batched path and handles both:
    - direct mapped rank-4 input path,
    - vmap-over-TNH path by flattening/unflattening the first two batch axes.
  - Verified with conversion regression for per-sample TNH kernel:
    `to_onnx(vmap(lambda q,k,v: jax.nn.dot_product_attention(q,k,v)))`.

Regression coverage added via plugin metadata testcases:
- `primitives.jnp / concatenate / concatenate_grad_issue191`
- `primitives.nn / dot_product_attention / dpa_vmap_tnh_issue190`
