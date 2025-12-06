# DINOv3 Flax/NNX Parity

- Goal: add a Flax/NNX DINOv3 example + exporter stack to mirror the existing Equinox path (`jax2onnx/plugins/examples/eqx/dino.py`), matching the dual coverage we already have for GPT-OSS (Equinox + Flax/NNX).

## Plan
- Port the Equinox model surfaces (PatchEmbed, Attention, Block, VisionTransformer, rotary helper) to nnx modules, respecting the NNX container rules (array fields in `nnx.List`/`nnx.data`, hashable callables, no import-time PRNG).
- Wire deterministic construction helpers (`construct_and_call`, `with_rng_seed`, `with_requested_dtype`) so rng/dtype variants retrace cleanly per AGENTS guardrails.
- Emit ONNX/expect_graph fixtures for key pieces (rotary cache, attention core, block, full ViT) and add parity tests alongside the Equinox ones.
- Extend docs/readme entries (coverage table, dinov3 how-to) and exporter scripts if the nnx path needs its own CLI shim.
- Validate with focused pytest + ruff/mypy before adding to any release notes.

## Implementation status
- Equinox example + scripts/docs exist; Flax/NNX variant scaffolded in `jax2onnx/plugins/examples/nnx/dino.py` (DinoRoPE, rotary process_heads, PatchEmbed, Attention/Block, VisionTransformer with nnx.List/nnx.Param usage).
- README vision section now references the NNX DINO example; still need to add any Netron artifacts/expect_graph snippets once exports are validated.
- GPT-OSS shows the desired pattern (`plugins/examples/eqx/gpt_oss.py` vs `plugins/examples/nnx/gpt_oss_flax.py`); reuse its rng/layout patterns where possible.
- Guardrails to keep in mind: IR-only in converter/plugins, deterministic module construction (no import-time seeds), single-use PRNG splits, and keep structural expectations in sync with metadata/tests.
