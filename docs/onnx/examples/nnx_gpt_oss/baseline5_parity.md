# Baseline5 Parity (JAX ↔ ONNX)

```
JAX_PLATFORM_NAME=cpu ORT_LOG_SEVERITY_LEVEL=1 \
poetry run python scripts/run_flax_gpt_oss_onnx.py \
  --prompt "What is the capital of France?" \
  --params ~/.cache/gpt_oss/gpt-oss-20b/flax_params_2layers.msgpack \
  --config ~/.cache/gpt_oss/gpt-oss-20b/flax_params_2layers.config.json \
  --onnx docs/onnx/examples/nnx_gpt_oss/gpt_oss_transformer_flax_baseline5.onnx \
  --sequence-length 32
```

- ONNX / JAX logits shapes: `(32, 201088)`
- Prompt tokens: `['What', ' is', ' the', ' capital', ' of', ' France', '?']`
- Final-token logits max |Δ|: **1.33514404296875e-05**
- ONNX Runtime stripped a handful of unused initializers automatically (warnings in stdout).

## Original (Torch) ↔ JAX Parity (Baseline5)

```
JAX_PLATFORM_NAME=cpu \
poetry run python scripts/probe_flax_gpt_oss_parity.py \
  --prompt "What is the capital of France?" \
  --params ~/.cache/gpt_oss/gpt-oss-20b/flax_params_2layers.msgpack \
  --config ~/.cache/gpt_oss/gpt-oss-20b/flax_params_2layers.config.json \
  --torch-checkpoint ~/.cache/gpt_oss/gpt-oss-20b/original \
  --sequence-length 32 \
  --gpt-oss-path tmp/gpt-oss-jax-vs-torch-numerical-comparison \
  --torch-device cpu \
  --torch-max-layers 2
```

- Torch/JAX logits (7 valid tokens) `max |Δ| = 3.3e-05`, `mean |Δ| = 2.5e-06`, `median |Δ| = 2.0e-06`.
- Prompt tokens: `['What', ' is', ' the', ' capital', ' of', ' France', '?']`.
- Stage diffs (top 10 by max |Δ|):

  | tensor | max | mean |
  | --- | --- | --- |
  | `block1.mlp_expert_outputs` | 3.05e-04 | 2.33e-06 |
  | `block1.mlp_fused` | 2.37e-04 | 1.22e-06 |
  | `block1.output` | 1.98e-04 | 1.31e-06 |
  | `block0.mlp_expert_outputs` | 1.37e-04 | 9.04e-07 |
  | `block1.attn_k` | 6.9e-05 | 6.73e-07 |
  | `block1.post_attention` | 6.9e-05 | 6.73e-07 |
  | `block0.mlp_activated_outputs` | 6.5e-05 | 3.55e-07 |
  | `block1.mlp_activated_outputs` | 6.1e-05 | 3.36e-07 |
  | `block0.mlp_fused` | 5.7e-05 | 6.62e-07 |
  | `block0.output` | 5.0e-05 | 5.96e-07 |

- No `[issues]` were reported; remaining per-stage entries stay ≤2.3e-5 and match shapes exactly.

## Expert Routing Parity Snapshot

```
JAX_PLATFORM_NAME=cpu \
poetry run python scripts/gpt_oss_routing_parity.py \
  --prompt "What is the capital of France?" \
  --gpt-oss-path tmp/gpt-oss-jax-vs-torch-numerical-comparison \
  --jax-checkpoint ~/.cache/gpt_oss/gpt-oss-20b/orbax \
  --torch-checkpoint ~/.cache/gpt_oss/gpt-oss-20b/original \
  --torch-device cpu \
  --max-layers 2 \
  --max-tokens 1
```

- Tokenized prompt to a single token (`[3923]`) to match the public routing study.
- Layer parity (2-layer slice): **100% expert-ID agreement**, gate weight mean diff ≤9.77e-04, max diff 1.95e-03.
- Script dropped a markdown report at `artifacts/gpt_oss_routing/20251113-075427_summary.md` detailing per-layer stats; keep alongside the ONNX artifact for auditors.
