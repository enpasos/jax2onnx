# GPT-OSS Weights Workflow

The GPT-OSS examples in this repo come in two flavors:

- **Flax/NNX (`jax2onnx/plugins/examples/nnx/gpt_oss_flax.py`)** – this is the
  path backed by the routing parity harness, staged checkpoint exporter, and the
  `FlaxTransformer` expect-graph tests.
- **Equinox (`jax2onnx/plugins/examples/eqx/gpt_oss.py`)** – kept for historical
  comparison and still covered by the Equinox parity tests.

Unless you specifically need the Equinox version, follow Sections 2–4 below for
Flax/NNX. The Equinox workflow now lives in Sections 5–6.

All commands assume you are at the project root with the Poetry environment
available. The workflow targets CPU-only tools; feel free to switch the device
flags to `cuda:X` if you have GPU support.

## Related Examples

For detailed component exports (like `GPT-OSS Attention`, `MLP`, `RMSNorm`), see the `GPT-OSS` entries in the [Examples](../../examples.md) table.

## 1. Download a GPT-OSS checkpoint

OpenAI publishes GPT-OSS on Hugging Face under both 20B and 120B variants. The
commands below fetch the original shard layout expected by
`gpt_oss.torch.model.Transformer.from_checkpoint`.

```bash
mkdir -p ~/.cache/gpt_oss/gpt-oss-20b
poetry run huggingface-cli download openai/gpt-oss-20b original \
  --repo-type model \
  --local-dir ~/.cache/gpt_oss/gpt-oss-20b \
  --local-dir-use-symlinks False
```

> `huggingface_hub` ≥1.0 no longer installs the `huggingface-cli` entry point by
> default, and recent releases ignore `--local-dir-use-symlinks`. If the command
> above fails with “command not found” or you prefer to stay within Python,
> download the checkpoint folder directly via `snapshot_download`:

```bash
poetry run python - <<'PY'
from pathlib import Path

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="openai/gpt-oss-20b",
    repo_type="model",
    allow_patterns=["original/*"],
    local_dir=Path("~/.cache/gpt_oss/gpt-oss-20b").expanduser(),
)
PY
```

This grabs the `original/` shard set expected by
`Transformer.from_checkpoint`. Omit `allow_patterns` if you want the full repo
contents (tokenizer, chat template, etc.).

After the download finishes you should have a directory containing `config.json`
and a set of `*.safetensors` shards.

## 2. Stage GPT-OSS weights for Flax/NNX

Run the staging helper to materialize a Flax `.msgpack` bundle plus a matching
`config.json`. The exporter and expect-graph tests consume this format directly.

```bash
poetry run python scripts/export_flax_gpt_oss_params.py \
  --checkpoint ~/.cache/gpt_oss/gpt-oss-20b/original \
  --output ~/.cache/gpt_oss/gpt-oss-20b/flax_params.msgpack
```

Use `--gpt-oss-path` if the helper repo lives somewhere other than the default
`tmp/gpt-oss-jax-vs-torch-numerical-comparison`. The script automatically
detects Orbax vs. SafeTensors checkpoints and writes
`flax_params.msgpack.config.json` beside the serialized parameters.

## 3. Export the Flax/NNX transformer to ONNX

With staged params in place, call the ONNX exporter. It instantiates the
`examples.nnx_gpt_oss.FlaxTransformer` module, loads the staged parameters via
`nnx.Param` assignments, and traces the full embedding → blocks → norm → head
pipeline.

```bash
poetry run python scripts/export_flax_gpt_oss_to_onnx.py \
  --params ~/.cache/gpt_oss/gpt-oss-20b/flax_params.msgpack \
  --output artifacts/gpt_oss_flax.onnx \
  --sequence-length 256
```

Notes:

- `--sequence-length` controls both the tracing inputs and the rotary/mask
  tables. Start small (e.g. 128) while verifying the workflow, then bump the
  length to your deployment target.
- Pass `--config /path/to/config.json` if the staging script’s JSON lives
  elsewhere.
- The exporter mirrors the exact callable covered by
  `tests/examples/test_nnx_gpt_oss.py::Test_FlaxTransformer`. Run that test (or
  the whole file) to sanity-check ONNX numeric validation locally:

  ```bash
  poetry run pytest tests/examples/test_nnx_gpt_oss.py::Test_FlaxTransformer -q
  ```

> **Tip:** When iterating on the exporter it can be helpful to trim the staged
> checkpoint down to a couple of layers. The snippet below keeps only the first
> two Transformer blocks while preserving the original hidden size/head layout,
> producing a much smaller bundle that exports quickly:
>
> ```bash
> poetry run python - <<'PY'
> from pathlib import Path
> import json
> import flax.serialization as serialization
>
> root = Path("~/.cache/gpt_oss/gpt-oss-20b").expanduser()
> params = serialization.msgpack_restore((root / "flax_params.msgpack").read_bytes())
> keep = {k: params[k] for k in ["embedding", "norm", "unembedding"]}
> for idx in range(2):
>     keep[f"block_{idx}"] = params[f"block_{idx}"]
> (root / "flax_params_2layers.msgpack").write_bytes(serialization.to_bytes(keep))
>
> config = json.loads((root / "flax_params.config.json").read_text())
> config["num_hidden_layers"] = 2
> (root / "flax_params_2layers.config.json").write_text(json.dumps(config, indent=2))
> print("Wrote 2-layer bundle under", root)
> PY
> ```
>
> Export with `--params .../flax_params_2layers.msgpack --config .../flax_params_2layers.config.json`
> to keep traces under a few minutes.

## 4. Flax/NNX routing parity harness

The parity harness from PR #217 verifies that the staged Flax/NNX model makes
identical expert choices to the PyTorch reference. There is an optional slow
smoke test in `tests/extra_tests/test_flax_routing_parity.py` that runs the
harness with `--max-layers 4 --max-tokens 2` on CPU whenever checkpoints are
present.

To run the harness manually (e.g. with longer prompts or more layers):

```bash
JAX_PLATFORM_NAME=cpu poetry run python scripts/gpt_oss_routing_parity.py \
  --gpt-oss-path tmp/gpt-oss-jax-vs-torch-numerical-comparison \
  --jax-checkpoint ~/.cache/gpt_oss/gpt-oss-20b/original \
  --torch-checkpoint ~/.cache/gpt_oss/gpt-oss-20b/original \
  --prompt "What is the capital of France?" \
  --max-layers 24 \
  --max-tokens 4 \
  --torch-device cpu \
  --output-dir artifacts/gpt_oss_routing/flax
```

The harness writes `artifacts/gpt_oss_routing/flax/<timestamp>_summary.md`
containing per-layer match rates and gate diffs. Adjust `--max-layers` and
`--max-tokens` to keep runs developer-friendly, and prefer `--torch-device cpu`
to avoid CUDA OOMs during PyTorch checkpoint loading.

## Baseline parity snapshot (Baseline5, 2-layer slice)

- Instrumentation: `export_flax_gpt_oss_to_onnx.py` supports `--emit-hidden-states` and `--emit-block-debug`; `run_flax_gpt_oss_onnx.py` compares hidden states and block-debug tensors (attention I/O, MoE norms/gates/experts, fused outputs).
- Parity (2-layer checkpoint, seq_len=32, debug export): logits max |Δ| ≈ 1.9e-05; hidden states ≤ 1.5e-04; MoE debug tensors ≤ 4.5e-04.
- Routing evidence: `scripts/gpt_oss_routing_parity.py` captures both a 2-layer slice (exact match) and a full 24-layer run (22/24 layers match; remaining layers differ by ≤ 4e-03 gate deltas). See `docs/onnx/examples/nnx_gpt_oss/baseline5_parity.md`.
- Artifacts: debug export at `/tmp/gpt_oss_transformer_flax_debug.onnx` (with `.data`). The committed Baseline5 artifact lives under `docs/onnx/examples/nnx_gpt_oss/` with external data named `gpt_oss_transformer_flax_baseline5.onnx.data`.

### Reproduce the Baseline5 debug export and parity check

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

### Original (Torch) ↔ Flax parity checklist

Prove the staged Flax bundle matches the Torch checkpoint before shipping ONNX:

```bash
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

The script tokenizes the prompt, runs both frameworks, and reports logits/stage tensor deltas. Store the transcript next to the promoted ONNX (e.g., `docs/onnx/examples/nnx_gpt_oss/baseline5_parity.md`).

### ONNX-only smoke test (tokenizer + generation)

For a short-window ONNX-only run (no JAX/Torch in memory):

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

For longer prompts/responses, re-export with a larger `--sequence-length` or add KV cache support to avoid re-running the full window.

## 5. (Legacy) Export the Equinox example to ONNX

Use the helper script to load the checkpoint, mirror it into the IR-only Equinox
modules, and emit an ONNX graph. The script preserves the exact callable used by
our tests so structural expectations continue to hold. On memory constrained
systems it helps to run the export in two stages:

1. **Stage the Equinox weights** (reads SafeTensors → writes `.eqx`, no ONNX yet):

   ```bash
   poetry run python scripts/export_eqx_gpt_oss_example_with_mapped_weights.py \
     --checkpoint ~/.cache/gpt_oss/gpt-oss-20b/original \
     --save-eqx ~/.cache/gpt_oss/gpt-oss-20b/eqx_gpt_oss_transformer.eqx \
     --seq-len 256 \
     --dynamic-b \
     --skip-onnx
   ```

2. **Convert the cached Equinox model to ONNX** (no PyTorch in memory):

```bash
poetry run python scripts/export_eqx_gpt_oss_example_with_mapped_weights.py \
  --eqx ~/.cache/gpt_oss/gpt-oss-20b/eqx_gpt_oss_transformer.eqx \
  --output ~/.cache/gpt_oss/gpt-oss-20b/eqx_gpt_oss_transformer.onnx \
  --seq-len 256 \
  --dynamic-b 
```

- `--dynamic-b` emits a symbolic batch axis (`B`) that matches the example tests.
- Omit `--dynamic-b` and/or add `--dynamic-seq` to tailor the exported shapes.
- `--save-eqx` keeps the mapped Equinox parameters around for future exports.
- Pass a higher `--seq-len` (e.g. 512) once the 256-token run succeeds; longer
  sequences raise memory pressure while tracing the attention blocks.

## 6. (Legacy) Validate Equinox parity (optional)

Numerical comparisons between the PyTorch and ONNX/JAX paths are covered by the
regression tests in `tests/extra_tests/test_eqx_gpt_oss_parity.py`. When the
optional dependencies above are installed, this test asserts the Equinox model
tracks the PyTorch reference to within a small tolerance (absolute differences
stay below `~1e0` when working in bfloat16).

Run a focused check with:

```bash
poetry run pytest -q tests/extra_tests/test_eqx_gpt_oss_parity.py
```
