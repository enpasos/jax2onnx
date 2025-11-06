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
- `scripts/probe_eqx_gpt_oss_parity.py` now prints stage-by-stage diffs; with bf16 params attention is down to |Δ|max≈1.0, but MoE combine still carries ~34 of error (float32 path remains ≫10²).
- Outstanding:
  - Restore bf16 precision by matching torch’s round-trips inside block0 MLP; script shows attention already within tolerance (|Δ|≈1).
  - Float32 parity still misses badly (block0 MLP ≈3e2). Need to decide whether to keep torch weights in bf16 when running the float32 Equinox path, or add explicit bf16 casts around matmuls so we mimic torch’s compute story.
  - Once RMSNorm is corrected, re-run `/tmp/compare_attention_stage_eqx.py` to confirm QKV/rotary align before analyzing the sink/softmax path.
  - After attention parity is achieved, re-run `poetry run pytest tests/extra_tests/test_eqx_gpt_oss_parity.py::test_eqx_matches_torch_with_bfloat16_weights -q`, then expand to the float32-parameter variant and revisit MoE helper coverage.

## 20B Real-Weights Parity Plan

Goal: run parity against the released `openai/gpt-oss-20b` checkpoint, using real prompts and the official harmony tokenizer/format, then bring the Equinox port into alignment.

1. **Tokenizer & Harmony Setup**
   - Vendor in (or add optional dependency on) the GPT-OSS `openai-harmony` package.
   - Load the harmony-compatible tokenizer (`tokenizer.model`) and chat template.
   - Encode a README prompt (e.g. “Explain quantum mechanics clearly and concisely.”) into token IDs + attention mask.
   - Store fixture prompt(s) plus expected harmony conversation metadata so tests stay deterministic.

2. **Torch 20B Reference Loader**
   - Download `openai/gpt-oss-20b` locally (≈16 GB) and expose a helper that loads the official checkpoint into the Torch reference model.
   - Confirm the prompt round-trips through the README inference pipeline and yields a coherent harmony response (no random weights).

3. **Equinox Mapping for 20B**
   - Extend `_config_from_torch_transformer` / `_populate_eqx_from_torch` to ingest the real 20B weights (watch MoE MXFP4 specifics).
   - Ensure parameter dtype handling copies the released bf16 tensors without mutating them.

4. **Parity Harness Upgrade**
   - Replace the tiny random probe with a harness that:
     1. Tokenizes the real prompt.
     2. Runs Torch 20B forward to produce logits / harmony output.
     3. Mirrors the same prompt through the Equinox model.
     4. Compares logits and key intermediate activations (attention, MoE combine).
   - Keep the tiny random probe around as a fast unit test, but gate the real-weight parity behind an integration flag.

5. **Conformance Testing**
   - Add pytest cases that call the real-weight parity harness (mark as slow / requires model).
   - Record acceptable tolerances (target |Δ|max ≲ 1e-3 on logits).
   - Document hardware expectations (GPU/CPU RAM) and instructions for obtaining the weights.

6. **Before Baseline**
   - Once parity passes on the real prompt(s), rerun the MoE/attention investigations and update expect_graph metadata.
   - Capture the verified prompt + output pair in docs so future regressions have a reference.

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
