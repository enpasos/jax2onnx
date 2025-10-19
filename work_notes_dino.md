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
- Long-term alignment: keep the example as close as possible to Equimo’s DINOv3 implementation and source trained parameters directly from upstream Equimo or the Meta/Facebook DINOv3 releases once format compatibility is clear.

## Focus
- `jax2onnx/plugins/examples/eqx/dino.py`: ensure the example runs under the IR-only pipeline and adheres to the above guardrails.
- Track blockers, test coverage, and export parity updates directly in this document as work progresses.
- **Strict Directive:** Keep the Equinox example code as close as reasonably possible to the upstream Equimo implementation; prefer enhancing `jax2onnx` over diverging from the source unless a minimal shim is absolutely required.

## Progress Log (Completed)
- Pretrained export: CLI script `scripts/export_dinov3_pretrained.py` now produces ONNX directly via IR. Added runtime shims (RoPE cache freezing, deterministic dropout paths, GELU activation) so Equimo’s `dinov3_vits16_pretrain_lvd1689m` checkpoint exports successfully and deterministically.
- Added `tests/examples/test_eqx_dino_pretrained_runtime.py` – optional ONNX Runtime smoke test comparing the exported graph against the patched JAX model when `DINO_EQX_ONNX` (and optionally `DINO_EQX_WEIGHTS`) are provided.
- Added `scripts/map_equimo_dino_weights.py` to lift Equimo checkpoints into the simplified `examples.eqx_dino` VisionTransformer (`.eqx` serialisation output). The mapper currently bails out when register tokens are present because the plain example architecture does not model them yet.
- PatchEmbed: introduced `eqx.filter_vmap` wrappers and a batching rule for the custom `jnp.squeeze` primitive so `Test_PatchEmbed::test_patch_embed` passes for both static/dynamic batches.
- Vision blocks: LayerNorm/MLP now run under `eqx.filter_vmap`, keeping Equimo semantics while satisfying ONNX tracing (fixes the transformer + ViT paths).
- Attention + RoPE:
  - Restored the upstream two-argument rotary API and threaded token length explicitly so dynamic batches export cleanly.
  - Refactored `Attention` to reuse an in-module `AttentionCore` and a shared `RotaryProcessHeads` helper; the plugin lowers RoPE alongside the attention primitive, including dynamic batch support.
- EQX primitive registration: hooked `eqx.nn.Conv2d`, `eqx.nn.MultiheadAttention`, and `eqx.nn.RotaryPositionalEmbedding` into the plugin registry with focused expect-graph coverage.
- IR optimizer: added `remove_identity_reshapes_ir` to strip redundant reshape corridors, simplifying the generated attention graphs.
- Imaging utilities: implemented a `jax.image.resize` lowering (nearest, linear, cubic) so posemb grids can be resized when we add learned positional embeddings.
- Examples & expect_graph updates:
  - Simplified `AttentionCore` usage (no `@onnx_function` indirection) and refreshed tests to assert operator counts rather than fragile reshape chains.
  - Adjusted EQX multihead attention expect-graphs to reflect the optimized operator layout after the reshape cleanup.

## Notes & Attempts
- Export script applies Equimo-specific shims (freeze RoPE caches, bypass dropout randomness, replace exact GELU) to keep the IR pipeline deterministic. These patches deliberately stay inside the CLI so core example modules remain untouched.
- Float64 runtime gaps still push the examples toward `run_only_f32_variant`; revisit once ONNX Runtime catches up.
- Earlier attempt to reshape RoPE caches by changing `jax.numpy.pow` abstract evaluation was rolled back due to recursion failures—keep in mind if dynamic-dim support resurfaces.

## Structuring Plan (Attention + RoPE)
1. **Module Layers**
   - Confirm any additional RoPE variants (e.g., learned cache reuse) still compose cleanly via the new `RotaryProcessHeads`.
   - Evaluate whether other Equinox helpers (relative position shifts, etc.) can be expressed as lightweight process-head adapters.
2. **Plugin Enhancements**
   - Generalise detection so closely related closures (e.g., custom modules wrapping `RotaryProcessHeads`) can be white-listed without re-implementing the lowering.
   - Explore surfacing the rotary caches as reusable nodes when multiple attention layers share the same sequence length, to reduce constant duplication.
3. **Testing & Docs**
   - Extend coverage with a regression that toggles between rotary/no-rotary `process_heads` to ensure the plugin continues to dispatch correctly.
   - Document the supported pattern in the example docstring (and plugin README) so contributors know to reuse `RotaryProcessHeads` instead of hand-written head rewrites.
- Stand up a learned positional embedding example mirroring Equimo’s `posemb.py`; confirm interpolation paths exercise the new `jax.image.resize` lowering (both antialias off/on).
- Audit parameter parity against upstream Equimo/DINOv3 checkpoints—identify a reproducible source for pretrained weights or add guidance on importing Equimo’s parameters (priority: match Equimo repo first, fall back to Meta’s DINOv3 release).
- **Weights ingestion**
  1. Request Meta’s official DINOv3 weights via <https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/> (required for the `.pth` files referenced by Equimo).
  2. Drop the downloaded checkpoints into `~/.cache/torch/hub/dinov3/weights/` with the exact filenames expected by Equimo’s `models/dinov3.py`.
  3. Use the helper `scripts/convert_dinov3_from_equimo.py --variant <id>` (wraps Equimo’s `convert_torch_to_equinox`) to serialize Equinox checkpoints into `~/.cache/equimo/dinov3/{variant}.tar.lz4`.
  4. Load checkpoints inside examples/tests via `load_pretrained_dinov3(...)`. A guarded integration test (`tests/examples/test_eqx_dino_pretrained.py`) compares the model’s features against a reference dump when the following env vars are set: `DINO_EQX_WEIGHTS`, `DINO_EQX_IMAGE`, `DINO_EQX_EXPECTED`, `DINO_EQX_VARIANT` (optional).
  5. Use `scripts/generate_dinov3_reference.py` to generate the reference activation file from an input tensor once weights are available.
- Pretrained flow:
  - Reused the Equimo conversion script (mirroring `models/dinov3.py`) so Meta’s `.pth` checkpoints can be converted—or downloaded directly from the Equimo HF hub—and consumed via `load_pretrained_dinov3`.
  - Added `scripts/generate_dinov3_reference.py` plus `tests/examples/test_eqx_dino_pretrained.py` so real weights can be smoke-tested against a reference activation when env vars point to cached inputs/outputs.
