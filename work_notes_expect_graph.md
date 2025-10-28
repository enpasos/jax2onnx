# Expect Graph Coverage Notes

## Goal
Keep every plugin and example wired with an up-to-date `expect_graph(...)` snippet so structural regressions surface immediately.

## Coverage Snapshot
- **Plugins:** activations, linear family, lax arithmetic/reductions/scatter, control-flow, and the `primitives.nn` / `primitives.jnp` sweeps all ship with expect_graph checks.
- **Examples:** nnx stacks (autoencoder, transformer decoders, multi-head attention), Equinox DINO, GPT, and ViT (including flattened variants) have regenerated snippets recorded in `docs/dev_guides/expect_graph_reference.md`.
- **Docs:** `docs/dev_guides/expect_graph_reference.md` mirrors current snippets; update it whenever behaviour changes.

## Remaining Work
- Monitor new examples or primitives as they land and document their expect_graph snippets promptly (no outstanding components as of 2025-10-28).
- Keep an eye on rare primitives (e.g., prospective `lax.bitwise_xor`) and add coverage once the lowering/tests exist.
- Re-run `python scripts/emit_expect_graph.py <testcase>` whenever metadata changes and sync the docs afterwards.

## Standard Workflow
1. Pick the next uncovered component from the remaining-work list (or run a quick scanner for missing `register_example` / `register_primitive` entries).
2. `poetry run python scripts/emit_expect_graph.py <testcase>` to capture the current snippet.
3. Update metadata/tests with the snippet, rerun the focused pytest target, then expand to the relevant suite if needed.
4. Refresh `docs/dev_guides/expect_graph_reference.md` and, if applicable, the coverage tables.
5. Before wrapping, ensure `expect_graph` docs match the code and leave a short note here if priorities shift.

## Guardrails
- Plugins and converters stay ONNX-IR only (no protobuf imports).
- Use `construct_and_call(...).with_requested_dtype()` and `with_rng_seed(...)` helpers so fixtures rebuild deterministically; split PRNG keys before reuse.
- Attention plugins must retain masked-weight normalisation; expect_graph snippets should reflect the normalised path.
- Run the core tooling (`poetry run pytest -q`, `poetry run ruff check .`, `poetry run mypy src`) before shipping larger sweeps.
