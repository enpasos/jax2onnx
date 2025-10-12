# Expect Graph Coverage Notes

## Goal
Attach auto-generated `expect_graph` structural assertions to every plugin and example testcase so each converter path is validated against the ONNX IR it produces.

## Coverage Snapshot
### Completed
- Full expect_graph coverage for flax/nnx activations and linear-family modules (e.g., `linear_general.py`, `conv.py`).
- JAX lax primitives refreshed: arithmetic, reductions, broadcast/reshape, gather/scatter family, control-flow utilities (`cond`, `scan`, `gather`, `scatter_add/max/mul`, etc.).
- Example suites seeded: `examples.jaxfluids`, `examples.jnp` (fori_loop, issue18 suite, select, sort), `examples.lax` (cond_scatter*, remat2, scatter_window), core `examples.nnx` stacks (autoencoder, fori_loop, MHA, sequential/transformer variants, ViT), and initial `examples.onnx_functions` fixtures.

### Remaining
- Sweep any straggling examples (e.g., attention/GPU fixtures) and confirm snippets stay current as tests evolve.
- Double-check rarely used lax primitives (`lax.or`, `lax.bitwise_or`) once tests exist.
- Refresh `docs/dev_guides/expect_graph_reference.md` with the latest expect_graph snippets.

## Active Focus
- `docs.dev_guides/expect_graph_reference.md`
  - Fold the latest expect_graph additions (scatter updates, GRU cell, Issue 18 loops) into the docs tables.
  - Regenerate the Markdown via the documented script to keep guidance in sync.
  - Spot-check the rendered output once refreshed.

## Session TODO
- [ ] Update `docs/dev_guides/expect_graph_reference.md` with the new snippet IDs and explanatory notes.
- [ ] Preview the Markdown (or run the doc build) after editing to verify formatting.

## Plugin Coverage Queue (Next Ten)
- **docs.dev_guides/expect_graph_reference.md**
  - Regenerate the expect_graph reference tables once code updates land.
  - Ensure examples/lax sections point at the latest snippet IDs.
- **primitives.lax.or / bitwise_or**
  - Add metadata + expect_graph coverage when lightweight tests become available.
- **examples.nnx attention/transformer variants**
  - Periodically re-emit snippets to catch masked-softmax or residual topology changes.
- **examples.jnp control-flow**
  - Re-run emit/verify if Issue 18 fixtures or new loop examples land in the suite.


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
- Primary docs to revisit while filling gaps: `docs/dev_guides/expect_graph_reference.md` (structural assertions), `docs/dev_guides/onnx_ir_builder.md` (builder rules), and `docs/dev_guides/subgraph_input_handling.md` (control-flow cases).
- Before wrapping a session, line up with repo routine: run focused pytest first, expand to the suite, then `ruff` + `mypy` as final checks.
- NNX fixtures must seed RNGs through `with_rng_seed(...)` and avoid inline lambdas so callables stay hashable under JAX ≥0.7.
- Attention plugins rely on masked-weight normalisation for numerical stability; expect_graph assertions should preserve that behaviour when adding coverage.
