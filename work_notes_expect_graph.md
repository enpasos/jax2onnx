# Expect Graph Coverage Notes

## Goal
Attach auto-generated `expect_graph` structural assertions to every plugin and example testcase so each converter path is validated against the ONNX IR it produces.

## Coverage Snapshot
### Completed
- Full expect_graph coverage for flax/nnx activations and linear-family modules (e.g., `linear_general.py`, `conv.py`).
- JAX lax primitives refreshed: arithmetic, reductions, broadcast/reshape, gather/scatter family, control-flow utilities (`cond`, `scan`, `gather`, `scatter_add/max/mul`, etc.).
- Example suites seeded: `examples.jaxfluids`, `examples.jnp` (fori_loop, issue18 suite, select, sort), `examples.lax` (cond_scatter*, remat2, scatter_window), core `examples.nnx` stacks (autoencoder, fori_loop, MHA, sequential/transformer variants, ViT), and initial `examples.onnx_functions` fixtures.
- Primitive coverage expanded: `primitives.lax.broadcast_in_dim`, `primitives.core.custom_jvp_generic` & `dim_as_value`, plus the `primitives.nn` activation/attention set (`celu`, `dot_product_attention`, `elu`, `gelu`, `identity`, `truncated_normal`, `leaky_relu`) now carry expect_graph assertions across all testcases.
- Latest sweep covered the next ten plugins in queue: `primitives.nn` (`mish`, `relu`, `selu`, `sigmoid`, `soft_sign`, `softmax`, `softplus`) and `primitives.jnp` (`add`, `arange`, `clip`) now emit expect_graph checks for every testcase.
- Follow-on sweep landed expect_graph coverage for `primitives.jnp` (`concatenate`, `cumsum`, `einsum`, `linspace`, `matmul`, `pow`, `power`, `prod`, `reshape`, `select`), including dynamic/constant fallbacks and symbolic dimension handling.
- Most recent pass knocked out another ten: `primitives.jnp` (`shape`, `sort`, `split`, `squeeze`, `stack`, `take`, `tile`, `transpose`, `unstack`, `where`) now assert their lowering graphs across static, broadcasted, and symbolic scenarios.
- Attention/GPU fixtures revalidated: `examples.nnx.multi_head_attention` and `examples.eqx_dino` variants now have fresh snippets (no drift; DINOv3 expectations trimmed to `VisionTransformer:*` with `no_unused_inputs=True`).

### Remaining
- Sweep any newly added examples (e.g., future attention/GPU fixtures) as they land and confirm snippets stay current.
- Track future rare lax primitives (e.g., `lax.bitwise_xor`) when tests appear.
- Refresh `docs/dev_guides/expect_graph_reference.md` with the latest expect_graph snippets.

## Active Focus
- `docs/dev_guides/expect_graph_reference.md`
  - ✅ 2025-10-28: Folded the scatter refresh, Issue 18 loop snippets, and GRU cell update into the reference section with regenerated expect_graph code blocks.
  - Regenerated snippets with `JAX_ENABLE_X64=1 poetry run python scripts/emit_expect_graph.py …` and cross-checked output before editing the doc.
  - Highlighted the AGENTS 2025-10-02 reminders (rng seeding + attention mask normalisation) in the refreshed section.

## Session TODO
- [x] Land expect_graph coverage for the next ten plugins in the queue (broadcast_in_dim through leaky_relu) and verify their targeted pytest suites.
- [x] Regenerate expect_graph specs for the scatter refresh, GRU cell, and Issue 18 loop fixtures via `poetry run python scripts/emit_expect_graph.py <testcase>` and merge them into `docs/dev_guides/expect_graph_reference.md` with context notes.
  - scatter coverage (`scatter_add_*`, `scatter_mul_*`, `scatter_max_*`, `scatter_min_*`, `scatter_set_*`, `scatter_window_update_*`, `cond_scatter_*`) ✅ captured 2025-10-14; doc synced 2025-10-28.
  - nnx coverage: `gru_cell_basic` (float32) ✅ snippet refreshed; evaluate need for a float64 variant separately.
  - issue 18 loops: `fori_loop_fn`, `while_loop_fn`, `scan_fn`, `where_fn` ✅ regenerated and documented.
  - Emit large batches in chunks (`poetry run python scripts/emit_expect_graph.py scatter_add_vector scatter_add_scalar`) so review diffs stay readable.
- [x] Preview the Markdown (or run the doc build) after editing to verify formatting.
- [x] Re-read the AGENTS guardrails (2025-10-02 update covers RNG helpers and attention mask normalisation) before touching plugin metadata to keep IR-only boundaries and key handling compliant.

## Latest Captures (2025-10-14)
- Regenerated scatter expect_graph snippets via `JAX_ENABLE_X64=1 poetry run python scripts/emit_expect_graph.py` for:
  - `scatter_add_vector` → `['ScatterND:4']`
  - `scatter_add_scalar` → `ScatterND:6` with input 2 fixed to const `5.0`
  - `scatter_add_simple_1d` / `scatter_add_batch_updates_1d_operand` → `['ScatterND:5']`
  - `scatter_add_window_2d_operand_1d_indices` → `['ScatterND:2x3']`
  - `scatter_add_mismatched_window_dims_from_user_report` → `['ScatterND:5x208x1x1']` (stdout spammed array dumps; consider redirecting logs before rerunning the rest of the batch).
- Captured fresh Issue 18 loop specs:
  - `fori_loop_fn` keeps const loop bound + `Loop` path.
  - `while_loop_fn` still emits `Less -> Loop` with large int guard.
  - `scan_fn` now shows `['Loop:B']` with `symbols={'B': None}` — update docs to mention the symbol.
- `gru_cell_basic` emit shows two chained `Add` segments (no explicit `Tanh`); kept ONNX at `tmp/expect/gru_cell_basic.onnx` for inspection before refreshing the metadata entry / docs.
- Reference doc now captures the fused-Add path so metadata/doc parity is restored.
- Attention fixture spot-checks (2025-10-28):
  - `multihead_attention_nn`, `multihead_attention_nnx`, `multihead_attention_2_nnx` regenerated; snippets unchanged (still pass with `search_functions=True` in metadata).
  - `eqx_dinov3_vit_*` variants regenerated; metadata now matches emitted `VisionTransformer:*` paths and asserts `no_unused_inputs=True`.
- Plugin sweep (2025-10-14):
  - `primitives.core.jit_inline` → `jit_identity` now asserts inline `Add`.
  - `primitives.nnx.layer_norm` → `layer_norm_bias_scale`, `layer_norm_multiaxis`.
  - `primitives.lax.scatter_add` → scalar/simple/depth regressions + `scatter_depth2_fp64_type_mismatch`.
  - `primitives.lax.select_n` → bool/int selector permutations.
  - `primitives.lax.squeeze` → unit-dim, axis-specific, and edge regressions.
  - `primitives.lax.transpose` → square, 4d, reverse, and auto-axis cases.
  - `primitives.lax.while_loop` → 17 outstanding loop fixtures (tuple state, counter, vector, closure, mixed-rank, CNN/NNX regressions).
  - `examples.gpt` sweep: token/position embeddings, transformer block/stack, embeddings/head, and broadcast add now assert their ONNX function calls.
  - `examples.onnx_functions` sweep: 002–004 nested function samples carry expect_graph checks for their function call nodes.
  - Added expect_graph coverage for `examples.onnx_functions` 005–016 (including the final nested-function variant).
  - `examples.onnx_functions` follow-up: 005–010 transformer variants, ViT-based 012–013, and function-default cases 014–015 now capture expect_graph snippets (VisionTransformer outputs noted at `Bx10`).
  - Focused pytest runs: `tests/primitives/test_core.py -k jit_inline`, `tests/primitives/test_nnx.py -k layer_norm`, `tests/primitives/test_lax.py` (full), plus targeted reruns for adjusted specs (all green).
  - Full suite verified via `poetry run pytest -q` (1554 passed, 2 xfailed).
  - Latest batch validated with `poetry run pytest -q tests/examples/test_onnx_functions.py`.

## Plugin Coverage Queue (Next Ten)
- **primitives.core.jit_inline** ✅
- **primitives.nnx.layer_norm** ✅
- **primitives.lax.scatter_add** ✅
- **primitives.lax.select_n** ✅
- **primitives.lax.squeeze** ✅
- **primitives.lax.transpose** ✅
- **primitives.lax.while_loop** ✅
- **docs/dev_guides/expect_graph_reference.md**
  - Regenerate tables after the above batches land and sync guidance.


Pending follow-up once these land: sweep random+control-flow fixtures and update the dev guides with any new structural patterns uncovered.

## Standard Workflow
1. Pick the next component from the TODO portion of the checklist.
2. For each testcase in that component:
   - Run `poetry run python scripts/emit_expect_graph.py <testcase_name>` to capture the current graph snippet.
   - Add (or merge into existing) `post_check_onnx_graph` entries using the emitted snippet. When merging with existing logic, wrap both checks in a helper to preserve prior expectations.
3. Re-run the targeted pytest class, e.g. `poetry run pytest -q tests/primitives/test_nnx.py::Test_<Component>`.
4. Update this checklist before moving on.

## Context for New Chats
When starting a new session, run through the **Standard Workflow** above, using this file as the single source of truth for which components still need coverage.

## Guardrails & References (AGENTS.md)
- Keep plugin + converter code ONNX-IR only; never import protobufs (`tests/extra_tests/framework/test_no_onnx_in_converter_plugins.py` enforces this).
- Ensure deterministic module construction during fixtures: wrap calls with `construct_and_call(...).with_requested_dtype()` and `with_rng_seed(...)`, splitting PRNG keys before reuse.
- Regenerate structural expectations via `scripts/emit_expect_graph.py` whenever metadata changes so parity stays intact.
- Tooling defaults (per AGENTS.md): Python 3.11+ with Poetry; run `poetry run ruff check .`, `poetry run ruff format .`, `poetry run mypy src`, and `poetry run pytest -q` before wrapping.
- Primary docs to revisit while filling gaps: `docs/dev_guides/expect_graph_reference.md` (structural assertions), `docs/dev_guides/onnx_ir_builder.md` (builder rules), and `docs/dev_guides/subgraph_input_handling.md` (control-flow cases).
- Before wrapping a session, line up with repo routine: run focused pytest first, expand to the suite, then `ruff` + `mypy` as final checks.
- NNX fixtures must seed RNGs through `with_rng_seed(...)` and avoid inline lambdas so callables stay hashable under JAX ≥0.7.
- Attention plugins rely on masked-weight normalisation for numerical stability; expect_graph assertions should preserve that behaviour when adding coverage.
