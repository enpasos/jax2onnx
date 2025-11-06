# GPT-OSS Equinox Example

- Goal (see issue #127): implement GPT-OSS in Equinox/Flax/NNX, import official weights, export to ONNX, and prove numerical parity against the PyTorch reference. Work here tracks the Equinox slice of https://github.com/enpasos/jax2onnx/issues/127.
- Context: follow `AGENTS.md` guardrails (IR-only in converter/plugins, deterministic module construction, single-use PRNG keys, expect_graph parity). Use `docs/dev_guides/onnx_ir_builder.md` and `docs/readme/dinov3` as structural references.
- Initial plan:
  1. Audit `plugins/examples/eqx/dino.py` and associated tests/expect_graph snippets to replicate the pattern for GPT-OSS.
  2. Extract GPT-OSS architecture + weight layout from https://github.com/openai/gpt-oss (identify config, rotary embeddings, attention blocks, LM head).
  3. Implement the Equinox GPT-OSS module tree with `@onnx_function` wrappers, ensuring ONNX-friendly shapes/types and plugin compatibility.
  4. Add expect_graph specs, regression/policy tests, and an equivalence check that loads GPT-OSS weights and compares ONNX vs. original outputs.
- Implementation status: core Equinox modules implemented (`RMSNorm`, `RotaryEmbedding`, attention/MoE/Transformer) with helpers for batched linear application; metadata + generated tests registered via `examples.eqx_gpt_oss`.
- Recent fixes:
  - Replaced `lax.rsqrt` in `RMSNorm` with `jnp.sqrt`/`jnp.reciprocal` so the ONNX pipeline has matching primitive coverage.
  - Sequenced rotary/mask logic now enforces concrete sequence lengths via `jax.core.concrete_or_error`, avoiding dynamic-dim sentinels when tracing the examples.
  - Added torch-style BF16 numerics helpers (`_torch_linear_bf16`, `_torch_rms_norm`, `_torch_top_k`) and plumbed `use_torch_gate` through transformer construction so checkpoint loads can flip on PyTorch-compatible rounding without impacting JAX training paths.
  - RMSNorm scales remain `float32` when mapping checkpoints, matching the torch reference modules.
  - MLP gating/experts now quantise intermediate matmuls back to bf16 before applying activation/bias to cut parity drift (remaining max δ≈2.5 on logits, float32 pass still off by ~21).
- Outstanding:
  - Stage diff now dominated by MoE combine (`proj2` / weighted sum). Torch-aligned bf16 rounding still needs refinement to bring logits <2 diff and to recover float32 parity.
  - Once RMSNorm is corrected, re-run `/tmp/compare_attention_stage_eqx.py` to confirm QKV/rotary align before analyzing the sink/softmax path.
  - After attention parity is achieved, re-run `poetry run pytest tests/extra_tests/test_eqx_gpt_oss_parity.py::test_eqx_matches_torch_with_bfloat16_weights -q`, then expand to the float32-parameter variant and revisit MoE helper coverage.

## GPT-OSS Architecture Notes

- Reference implementation lives under `gpt_oss/torch/model.py` with `ModelConfig`, `RMSNorm`, `RotaryEmbedding`, attention/MLP MoE blocks, and a `Transformer` composed of embedding → repeated `TransformerBlock` → RMSNorm → unembedding.
- Rotary embedding uses YaRN-style concentration via `scaling_factor`, `ntk_alpha`, `ntk_beta`, and applies to reshaped (KV grouped) queries/keys.
- Attention: projects to Q/K/V with per-layer sinks tensor appended as extra attention slot; uses sliding window mask on alternating layers and merges sinks via concatenating to attention logits.
- MLP: top-k mixture-of-experts (num_experts=128, experts_per_token=4) implemented with tensor gather and einsums; weights stored per-expert with optional distributed shard.
- Weights loader (`gpt_oss/torch/weights.py`) maps parameters, including MXFP4 compressed blocks for MoE weights, and handles dist world-size slicing.
- Checkpoint layout: directory containing `config.json` + `.safetensors` shards; `Transformer.from_checkpoint` builds `ModelConfig` from JSON then loads weights via `Checkpoint`.

## Equinox Port Sketch

- Target modules mirror PyTorch structure: `RMSNorm`, `RotaryEmbedding`, `AttentionBlock`, `MLPBlock`, `TransformerBlock`, `Transformer`.
- `RMSNorm`: use Equinox field for learned scale, implement epsilon + dtype cast identical to torch version.
- `RotaryEmbedding`: compute YaRN concentration and inv_freq via JAX, materialise sin/cos caches once per forward via host-side helper similar to DINO.
- `AttentionBlock`: maintain sinks parameter, use `eqx.nn.Linear` for qkv/out projections, reshape heads into `(seq, num_kv, q_mult, head_dim)` before RoPE, apply sliding window mask + appended sink channel, project back via matmul.
- `MLPBlock`: rely on `jax.lax.top_k` for gate selection, gather expert weights via `jax.vmap`/`jnp.take_along_axis`, run swiglu (clamped) + second projection, then weighted sum using softmax weights.
- `Transformer`: embed tokens, iterate blocks, final RMSNorm + linear head; provide helper to split logits vs. hidden for debugging akin to DINO’s `block_outputs`.
- Expect to expose `construct_and_call(...)` metadata for `RMSNorm`, `AttentionBlock`, `MLPBlock`, `TransformerBlock`, and top-level `Transformer`.
