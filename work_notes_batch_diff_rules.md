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
- Total custom-primitive modules scanned: `61`
- With batching rule (`primitive_batchers` or unary helper): `52` (`85.2%`)
- With explicit autodiff rule (`ad.defjvp` / `primitive_jvps` / transpose/linear rules): `8` (`13.1%`)

By area:
- `jax/numpy`: `36` total, `30` batching, `7` autodiff
- `jax/nn`: `22` total, `22` batching, `1` autodiff
- `jax/random`: `3` total, `0` batching, `0` autodiff

## Verified Conversion Gaps

Direct `to_onnx(...)` checks confirm the rule gap:
- `grad(jax.nn.dot_product_attention)` -> `NotImplementedError`

## Coverage List (Prioritized)

| Primitive / Group | Current vmap | Current grad | Recommended Approach | Priority | Action |
|---|---|---|---|---|---|
| `jax.numpy.concatenate` (`_CONCAT_PRIM`) | Yes | Yes | A (fallback) | Done | Implemented via fallback JVP + plugin testcase (`concatenate_grad_issue191`). |
| `jax.nn.dot_product_attention` (`_DPA_PRIM`) | Yes | No | B for vmap, then staged grad plan | P0 | vmap implemented; next is gradient rule strategy without losing macro structure. |
| Shape-only numpy wrappers (`reshape`, `squeeze`, `transpose`, `stack`, `split`, `tile`) | Mostly Yes | Yes | A (fallback) | P0 | Implemented via shared fallback helper; `tile` uses `lax`-decomposition in JVP to avoid raw `tile` conversion gaps. |
| Core nn activations (`relu`, `sigmoid`, `silu`, `gelu`, `elu`, `leaky_relu`, `softplus`, `softsign`, `selu`, `celu`, `mish`) | Yes | Partial | A (fallback/manual) | P1 | `relu` grad rule implemented; continue with remaining differentiable activations. |
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

## Continued Progress (Post #190/#191)

Implemented in this branch:

- Added shared fallback autodiff helper:
  - `jax2onnx/plugins/jax/_autodiff_utils.py`
  - provides `register_fallback_jvp_rule(...)` for primitive JVP registration.

- Extended shape-wrapper gradient support:
  - `jax2onnx/plugins/jax/numpy/reshape.py`
    - registered fallback JVP for `_RESHAPE_PRIM`.
    - added plugin testcase:
      `reshape_grad_issue_batch_diff_rules`.
  - `jax2onnx/plugins/jax/numpy/squeeze.py`
    - registered fallback JVP + testcase:
      `squeeze_grad_issue_batch_diff_rules`.
  - `jax2onnx/plugins/jax/numpy/transpose.py`
    - registered fallback JVP + testcase:
      `transpose_grad_issue_batch_diff_rules`.
  - `jax2onnx/plugins/jax/numpy/stack.py`
    - registered fallback JVP + testcase:
      `stack_grad_issue_batch_diff_rules`.
  - `jax2onnx/plugins/jax/numpy/split.py`
    - registered fallback JVP + testcase:
      `split_grad_issue_batch_diff_rules`.
  - `jax2onnx/plugins/jax/numpy/tile.py`
    - registered fallback JVP + testcase:
      `tile_grad_issue_batch_diff_rules`.
    - JVP fallback uses `lax.reshape` + `lax.broadcast_in_dim` to keep
      conversion on supported primitives in grad traces.

- Refactored concatenate gradient registration:
  - `jax2onnx/plugins/jax/numpy/concatenate.py`
    - now uses the shared helper through a variadic adapter
      (`_concatenate_fallback_jvp_impl`).

- Added relu gradient support:
  - `jax2onnx/plugins/jax/nn/relu.py`
    - registered explicit primitive JVP rule for `_RELU_PRIM`.
    - added plugin testcase:
      `relu_grad_issue_batch_diff_rules`.

Validation:
- `tests/primitives/test_jnp.py -k "concatenate_grad_issue191 or reshape_grad_issue_batch_diff_rules or squeeze_grad_issue_batch_diff_rules or transpose_grad_issue_batch_diff_rules or stack_grad_issue_batch_diff_rules or split_grad_issue_batch_diff_rules or tile_grad_issue_batch_diff_rules"` passes.
- `tests/primitives/test_nn.py -k "relu_grad_issue_batch_diff_rules"` passes.
