# DINO Equinox Export Work Notes

## Repo Guardrails (AGENTS.md)
- IR-only converter: keep ONNX protobuf imports out of `converter/` and `plugins/`.
- Deterministic module construction: use `construct_and_call(...).with_requested_dtype()` and `with_rng_seed(...)`; never seed at import.
- Single-use PRNG keys: split before distributing keys; enable `jax_debug_key_reuse` when debugging.
- Tooling: Python 3.11+, Poetry, Ruff (`check` + `format`), mypy, pytest; supported runtime stack JAX ≥0.7.2
- Workflow checklist: install with `poetry install -E all`, run focused pytest during development, full suite + lint + mypy before merging.
- Metadata parity: keep `expect_graph` specs aligned with lowering and regenerate via `scripts/emit_expect_graph.py` after behaviour changes.

## Background Thread
- Kickoff from @clementpoiret: greenlight to use Equinox DINOv3 as first bigger ONNX-IR export example for jax2onnx 0.9.0, replacing protobuf path.
- DINOv3 includes RoPE positional embeddings; to be thorough, also cover a standard learned positional embedding variant (see Equimo `posemb.py` at commit `ca0dae7`).
- Learned posemb across multiple image sizes needs `jax.image.resize` (with/without antialiasing) support, aligning with ONNX `Resize`.

## Focus
- `jax2onnx/plugins/examples/eqx/dino.py`: ensure the example runs under the IR-only pipeline and adheres to the above guardrails.
- Track blockers, test coverage, and export parity updates directly in this document as work progresses.

## Progress Log
- Ran `Test_PatchEmbed::test_patch_embed`; initial failure because `eqx.nn.Conv2d` expects unbatched inputs and our patched `jnp.squeeze` lacked a batching rule when vmapped.
- Updated `PatchEmbed.__call__` to apply `eqx.filter_vmap` over the batch dimension.
- Implemented a batching rule for the custom `jnp.squeeze` primitive (`jax2onnx/plugins/jax/numpy/squeeze.py`) by delegating to JAX’s native `_squeeze_batch_rule`.
- Re-ran the focused test; `Test_PatchEmbed::test_patch_embed` now passes.
- Float64 variant surfaced ONNX Runtime gaps: align convolution parameters with the input dtype inside `PatchEmbed.__call__`, and mark the example to skip numeric validation because ORT (CPU) lacks a Conv kernel for `float64`.
