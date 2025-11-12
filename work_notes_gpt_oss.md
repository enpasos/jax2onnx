# GPT-OSS Work Notes (Flax/NNX Path)

## Goal

Build a reproducible pipeline that:

1. Loads GPT-OSS checkpoints into the Flax/NNX reference modules (`jax2onnx/plugins/examples/nnx/gpt_oss_flax.py`).
2. Exports numerically faithful ONNX graphs (with optional debug taps) via `scripts/export_flax_gpt_oss_to_onnx.py`.
3. Proves JAX ↔ ONNX parity using `scripts/run_flax_gpt_oss_onnx.py` (logits, hidden states, and MoE internals).
4. Promotes the validated ONNX to `docs/onnx/examples/nnx_gpt_oss/` and documents the workflow for future checkpoints.

## Current Status (Baseline2 – 2025-11-12)

- **Instrumentation:** The exporter accepts `--emit-hidden-states` and `--emit-block-debug`. The harness understands `--compare-hidden-states` / `--compare-block-debug` and prints max/mean diffs per block tensor. Debug taps cover attention input/output plus the full MoE pipeline (normed tokens, gate logits/indices/weights, dense weights, prelinear/activated/expert outputs, fused result).
- **Parity:** With the 2-layer GPT-OSS checkpoint (`~/.cache/gpt_oss/gpt-oss-20b/flax_params_2layers.msgpack`, seq_len=32) the debug ONNX export matches JAX:
  - Logits `max |diff| ≈ 1.9e-05`
  - Hidden states `max |diff| ≤ 1.5e-04`
  - All MoE debug tensors `max |diff| ≤ 4.5e-04`
- **Artifacts:** Debug export lives at `/tmp/gpt_oss_transformer_flax_debug.onnx` (paired `.data`). No canonical artifact committed yet—once verified, it will be copied into `docs/onnx/examples/nnx_gpt_oss/` as “baseline2”.

## Reproducing Baseline2

```bash
JAX_PLATFORM_NAME=cpu ORT_LOG_SEVERITY_LEVEL=4 poetry run python scripts/export_flax_gpt_oss_to_onnx.py \
  --params ~/.cache/gpt_oss/gpt-oss-20b/flax_params_2layers.msgpack \
  --config ~/.cache/gpt_oss/gpt-oss-20b/flax_params_2layers.config.json \
  --output /tmp/gpt_oss_transformer_flax_debug.onnx \
  --sequence-length 32 \
  --emit-hidden-states \
  --emit-block-debug \
  --skip-validation

JAX_PLATFORM_NAME=cpu ORT_LOG_SEVERITY_LEVEL=4 poetry run python scripts/run_flax_gpt_oss_onnx.py \
  --prompt "What is the capital of France?" \
  --params ~/.cache/gpt_oss/gpt-oss-20b/flax_params_2layers.msgpack \
  --config ~/.cache/gpt_oss/gpt-oss-20b/flax_params_2layers.config.json \
  --onnx /tmp/gpt_oss_transformer_flax_debug.onnx \
  --sequence-length 32 \
  --compare-hidden-states \
  --compare-block-debug
```

## Outstanding Tasks

1. **Promote canonical artifact:** Re-export without debug outputs, run `run_flax_gpt_oss_onnx.py` (no compare flags) to capture final logits diff, then place `gpt_oss_transformer_flax.onnx(.data)` under `docs/onnx/examples/nnx_gpt_oss/` as “baseline2”.
2. **Documentation:** Add a workflow write-up (`docs/readme/gpt_oss/` or similar) describing the checkpoint download → Flax load → ONNX export → parity verification process. Reference the new debug flags.
3. **Regression coverage:** Create a lightweight pytest (toy config, short seq len) that runs the harness with `--compare-hidden-states`. This protects the instrumentation and keeps future MoE tweaks honest.
4. **Scaling plan:** Extend beyond the 2-layer checkpoint (full GPT-OSS stack, BF16/FP32 variants) once GPU-backed parity runs are available; reuse the block-debug taps to localize any discrepancies.

## Original (Torch) ↔ JAX Parity Checklist

Before trusting ONNX exports we prove that the Flax model reproduces the original GPT-OSS torch logits for at least one seeded prompt/sequence length. Run the dedicated parity script and capture the diff:

```bash
# 1) Ensure tmp/gpt-oss-jax-vs-torch-numerical-comparison contains the gpt_oss repo.
# 2) Use the staged Flax params/config emitted by export_flax_gpt_oss_params.py.
JAX_PLATFORM_NAME=cpu \
poetry run python scripts/probe_flax_gpt_oss_parity.py \
  --prompt "What is the capital of France?" \
  --params ~/.cache/gpt_oss/gpt-oss-20b/flax_params_2layers.msgpack \
  --config ~/.cache/gpt_oss/gpt-oss-20b/flax_params_2layers.config.json \
  --torch-checkpoint ~/.cache/gpt_oss/gpt-oss-20b/original \
  --sequence-length 32 \
  --gpt-oss-path tmp/gpt-oss-jax-vs-torch-numerical-comparison \
  --torch-device cpu
```

The script tokenizes the prompt (tiktoken if available, otherwise a byte fallback), pads/truncates to `--sequence-length`, runs both frameworks, and then reports per-token logits plus per-block debug diffs (normed inputs, gate logits, expert weights, fused outputs, etc.). Baseline2 sticks to the seeded prompt above and expects logits max `|Δ|` ≲ `2e-4` with stage stats in the `1e-4–1e-3` range. Commit (or at least stash) the console output alongside the exported artifact so reviewers can see which checkpoint/prompt proved parity and which tensors were inspected.

Mirror the Equinox parity workflow when recording evidence:

- Match model depth and dtype. When using the 2-layer staged bundle pass `--torch-max-layers 2` so both frameworks compare the same blocks, and ensure bf16 weights stay bf16 everywhere (the script guards conversions for you).
- Seed everything. Reuse the fixed prompt above plus `--seed 0` (default) so reruns regenerate identical logits/stage tensors.
- Capture the full report. Reviewers expect the command line, logits diff summary, the top stage-diff table, and an empty `[issues]` section (meaning every tensor matched shape-for-shape). This mirrors the Equinox checklist and keeps artifacts auditable.
- Store the parity transcript next to the promoted ONNX (e.g., `docs/onnx/examples/nnx_gpt_oss/baseline2_parity.md`) just like we do for the Equinox examples.

Once Original ↔ JAX parity is proven for the checkpoint, the JAX ↔ ONNX harness closes the chain and we can ship the ONNX artifact with confidence.
