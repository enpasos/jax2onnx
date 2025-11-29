# GPT-OSS Work Notes (Flax/NNX Path)

## Goal

Build a reproducible pipeline that:

1. Loads GPT-OSS checkpoints into the Flax/NNX reference modules (`jax2onnx/plugins/examples/nnx/gpt_oss_flax.py`).
2. Exports numerically faithful ONNX graphs (with optional debug taps) via `scripts/export_flax_gpt_oss_to_onnx.py`.
3. Proves JAX ↔ ONNX parity using `scripts/run_flax_gpt_oss_onnx.py` (logits, hidden states, and MoE internals).
4. Promotes the validated ONNX to `docs/onnx/examples/nnx_gpt_oss/` and documents the workflow for future checkpoints.

## Current Status (Baseline5 – 2025-11-12)

- **Instrumentation:** The exporter accepts `--emit-hidden-states` and `--emit-block-debug`. The harness understands `--compare-hidden-states` / `--compare-block-debug` and prints max/mean diffs per block tensor. Debug taps cover attention input/output plus the full MoE pipeline (normed tokens, gate logits/indices/weights, dense weights, prelinear/activated/expert outputs, fused result).
- **Parity:** With the 2-layer GPT-OSS checkpoint (`~/.cache/gpt_oss/gpt-oss-20b/flax_params_2layers.msgpack`, seq_len=32) the debug ONNX export matches JAX:
  - Logits `max |diff| ≈ 1.9e-05`
  - Hidden states `max |diff| ≤ 1.5e-04`
  - All MoE debug tensors `max |diff| ≤ 4.5e-04`
- **Routing evidence:** `scripts/gpt_oss_routing_parity.py` now has captured both the 2-layer slice (perfect match) and the full 24-layer run (22/24 layers match, remaining layers differ only by ≤4e-03 gate deltas); see `docs/onnx/examples/nnx_gpt_oss/baseline5_parity.md`.
- **Artifacts:** Debug export lives at `/tmp/gpt_oss_transformer_flax_debug.onnx` (paired `.data`). No canonical artifact committed yet—once verified, it will be copied into `docs/onnx/examples/nnx_gpt_oss/` as “baseline5”. The committed Baseline5 ONNX has been re-saved so its external data file is `gpt_oss_transformer_flax_baseline5.onnx.data`, meaning no legacy filename shims are required.



## Reproducing Baseline5

```bash
JAX_PLATFORM_NAME=cpu ORT_LOG_SEVERITY_LEVEL=4 poetry run python scripts/export_flax_gpt_oss_to_onnx.py \
  --params ~/.cache/gpt_oss/gpt-oss-20b/flax_params_2layers.msgpack \
  --config ~/.cache/gpt_oss/gpt-oss-20b/flax_params_2layers.config.json \
  --output /tmp/gpt_oss_transformer_flax_debug.onnx \
  --sequence-length 16 \
  --emit-hidden-states \
  --emit-block-debug \
  --skip-validation

JAX_PLATFORM_NAME=cpu ORT_LOG_SEVERITY_LEVEL=4 poetry run python scripts/run_flax_gpt_oss_onnx.py \
  --prompt "What is the capital of France?" \
  --params ~/.cache/gpt_oss/gpt-oss-20b/flax_params_2layers.msgpack \
  --config ~/.cache/gpt_oss/gpt-oss-20b/flax_params_2layers.config.json \
  --onnx /tmp/gpt_oss_transformer_flax_debug.onnx \
  --sequence-length 16 \
  --compare-hidden-states \
  --compare-block-debug
```

## Outstanding Tasks

1. **Promote canonical artifact:** Re-export without debug outputs, run `run_flax_gpt_oss_onnx.py` (no compare flags) to capture final logits diff, then place `gpt_oss_transformer_flax.onnx(.data)` under `docs/onnx/examples/nnx_gpt_oss/` as “baseline5”.
2. **Documentation:** Add a workflow write-up (`docs/readme/gpt_oss/` or similar) describing the checkpoint download → Flax load → ONNX export → parity verification process. Reference the new debug flags.
 3. **Regression coverage:** Create a lightweight pytest (toy config, short seq len) that runs the harness with `--compare-hidden-states`. This protects the instrumentation and keeps future MoE tweaks honest.
 4. **Scaling plan:** Extend beyond the 2-layer checkpoint (full GPT-OSS stack, BF16/FP32 variants) once GPU-backed parity runs are available; reuse the block-debug taps to localize any discrepancies.

## Next: Equinox path (baseline 13 target)

- Use the Flax/NNX export (baseline 13) as the behavioral reference: same checkpoint/config/seq_len and prompt as the working ONNX-only run.
- Stage an Equinox parameter bundle mirroring the Flax config (start with a 2-layer slice for fast iteration).
- Port the model wiring to Equinox modules; keep weight naming/layout aligned for easy mapping.
- Build a parity harness against the Flax/NNX baseline: logits + key intermediates (attn projections, gate logits/indices/weights, MLP outputs).
- Export Equinox → ONNX with the same function structuring (blocks, shared ops); emit a small seq_len (16/32) artifact first.
- Validate with onnxruntime vs Equinox outputs; only then consider larger layer counts/sequence lengths.

## ONNX-only smoke test (tokenizer + generation)

- Exported the full 20B model with a reduced window to dodge WSL memory limits:

  ```bash
  JAX_PLATFORM_NAME=cpu \
  poetry run python scripts/export_flax_gpt_oss_to_onnx.py \
    --params ~/.cache/gpt_oss/gpt-oss-20b/flax_params.msgpack \
    --config ~/.cache/gpt_oss/gpt-oss-20b/flax_params.config.json \
    --output /tmp/gpt_oss_transformer_flax_seq16.onnx \
    --sequence-length 16 \
    --skip-validation

  mkdir -p artifacts/gpt_oss_full_seq16
  mv /tmp/gpt_oss_transformer_flax_seq16.onnx artifacts/gpt_oss_full_seq16/
  mv /tmp/gpt_oss_transformer_flax_seq16.onnx.data artifacts/gpt_oss_full_seq16/
  ```

- Ran `scripts/run_onnx_only.py` (now using the official GPT-OSS tokenizer) on a short prompt to keep within the 16-token window:

  ```bash
  poetry run pip install onnxruntime  # once, if not already present

  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \
  poetry run python scripts/run_onnx_only.py \
    --onnx artifacts/gpt_oss_full_seq16/gpt_oss_transformer_flax_seq16.onnx \
    --config ~/.cache/gpt_oss/gpt-oss-20b/flax_params.config.json \
    --prompt "France capital? Answer:" \
    --sequence-length 16 \
    --generate-steps 8 \
    --expand-functions \
    --runtime ort
  ```

- Decoded output includes a clean mention of Paris (e.g., `Decoded tokens:  Paris. So answer: Paris. So ...`). The script now also tries to extract `"text": "..."` segments if the model replies in its usual JSON-wrapped format.

- Takeaway: with the real tokenizer + a short prompt, the ONNX-only path produces human-readable tokens; for longer prompts/responses re-export with a larger `--sequence-length` or add KV cache support to avoid re-running the full window each step.

## Original (Torch) ↔ JAX Parity Checklist

Before trusting ONNX exports we prove that the Flax model reproduces the original GPT-OSS torch logits for at least one seeded prompt/sequence length. Run the dedicated parity script and capture the diff:

```bash
# 1) Ensure tmp/gpt-oss-jax-vs-torch-numerical-comparison contains the gpt_oss repo.
# 2) Use the staged Flax params/config emitted by export_flax_gpt_oss_params.py.
JAX_PLATFORM_NAME=cpu \
poetry run python scripts/probe_flax_gpt_oss_parity.py \
  --prompt "France capital? Answer:" \
  --params ~/.cache/gpt_oss/gpt-oss-20b/flax_params_2layers.msgpack \
  --config ~/.cache/gpt_oss/gpt-oss-20b/flax_params_2layers.config.json \
  --torch-checkpoint ~/.cache/gpt_oss/gpt-oss-20b/original \
  --sequence-length 16 \
  --gpt-oss-path tmp/gpt-oss-jax-vs-torch-numerical-comparison \
  --torch-device cpu \
  --torch-max-layers 2
```

The script tokenizes the prompt (tiktoken if available, otherwise a byte fallback), pads/truncates to `--sequence-length`, promotes the Torch reference to float32, and runs both frameworks while capturing per-token logits plus per-block debug tensors (normed inputs, q/k/v, gate logits, expert weights, fused outputs, etc.). Baseline5 (2-layer bundle, prompt above) now lands at logits max `|Δ| ≈ 3e-5` with stage stats ≤`3e-4`. Commit (or at least stash) the console output alongside the exported artifact so reviewers can see which checkpoint/prompt proved parity and which tensors were inspected.

Mirror the Equinox parity workflow when recording evidence:

- Match model depth and dtype. When using the 2-layer staged bundle pass `--torch-max-layers 2` so both frameworks compare the same blocks, and ensure bf16 weights stay bf16 everywhere (the script guards conversions for you).
- Seed everything. Reuse the fixed prompt above plus `--seed 0` (default) so reruns regenerate identical logits/stage tensors.
- Capture the full report. Reviewers expect the command line, logits diff summary, the top stage-diff table, and an empty `[issues]` section (meaning every tensor matched shape-for-shape). This mirrors the Equinox checklist and keeps artifacts auditable.
- Store the parity transcript next to the promoted ONNX (e.g., `docs/onnx/examples/nnx_gpt_oss/baseline5_parity.md`) just like we do for the Equinox examples.

Once Original ↔ JAX parity is proven for the checkpoint, the JAX ↔ ONNX harness closes the chain and we can ship the ONNX artifact with confidence.

# [2025-11-27] Systematic Parity Restoration: RoPE & Equinox

**Status:** RoPE fixed (q_rot/k_rot 0.0); remaining drift in attn_core/norm under probe.
**Symptoms (latest):**
- `q` input diff: 0.000000 (bfloat16)
- `q_rot` output diff: 0.000000 (bfloat16) — fixed
- `norm` output diff: ~0.016746 (float32) — epsilon/dtype
- `attn_core` diff: ~2.06 (bf16) — likely sinks/SDPA

### Step 1: RoPE Standalone Verification (High Priority)
Complete.
- [x] `scripts/debug_rope_parity.py` to feed identical inputs to Flax/NNX vs Equinox RoPE.
- [x] Rotation mode aligned (halves/split); q_rot/k_rot diff 0.0.
- [x] RoPE tables generated in dtype-consistent way; respects bf16 inputs.

### Step 2: RMSNorm Precision
- [x] Epsilon parity set to 1e-5 across attention/MLP/transformer norms.
- [x] Accumulation in float32 for bf16 inputs.
- [ ] Verify probe shows norm diff ≈ 0.

### Step 3: Attention Sinks & Core Stability
- [x] Sinks init scaled to std 0.02 to match Flax.
- [ ] Run `scripts/debug_attn_parity.py` with high-variance sinks; confirm stability and magnitude.
- [ ] Rerun full probe to check attn_core diff drops.

### Notes
- Attn parity drift now likely in SDPA with sinks; RoPE is fixed.
- If drift persists after sinks check, consider Layer 0 injection test to isolate SDPA.


## [2025-11-27] Systematic Parity Plan: Equinox vs Flax/NNX

**Current Blocker:** Divergence detected at `block0.attn.q_rot`.
**Symptoms:**
- `q` input diff: 0.000000 (bfloat16)
- `q_rot` output diff: 0.250000 (bfloat16) -> **CRITICAL FAILURE**
- `norm` output diff: 0.016746 (float32) -> **PRECISION ISSUE**

### Step 1: RoPE Standalone Verification
We are 99% confident the Rotary Embedding logic differs.
- [ ] Create `scripts/debug_rope_parity.py`
- [ ] **Check Interleaving:** Verify if complex conversion assumes `x[..., ::2], x[..., 1::2]` (interleaved) or `x[..., :d//2], x[..., d//2:]` (halved).
    - *Note:* LLaMA/GPT-NeoX usually uses "pairs" (interleaved). Standard ViT/RoPE implementations sometimes vary.
- [ ] **Check Frequencies:** Dump `inv_freq` from both implementations. Verify `base=10000.0` and float precision consistency.

### Step 2: RMSNorm Precision
- [ ] Audit `RMSNorm` epsilon values in config.
- [ ] Verify that Equinox `RMSNorm` performs accumulation in `float32` even if input is `bfloat16`.

### Step 3: Layer 0 Injection Test
Instead of full model probing:
- [x] Dump `block0` intermediates from Flax/NNX using `orbax` or `numpy.save`.
- [x] Write a test `tests/test_eqx_layer0.py` that loads weights + input and asserts output match against the dump.

## [2025-11-29] Equinox Parity Resolution

**Status:** Parity Achieved.
**Root Cause:** The probe script initialized weights with `std=1.0` (Torch default), causing massive activations (~15k) and softmax overflow.
**Fix:** Scaled initialization to `std=0.02` (GPT-OSS default).

**Final Diffs (bfloat16):**
- `logits`: 0.000061
- `attn_core`: 0.000244
- `norm`: 0.000977

**Final Diffs (float32):**
- `logits`: 0.000072
- `block0.attn.norm`: 0.000236

The pipeline is now ready for full model porting and export verification.
