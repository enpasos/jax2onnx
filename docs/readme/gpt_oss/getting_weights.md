# GPT-OSS Weights Workflow

The GPT-OSS example under `jax2onnx/plugins/examples/eqx/gpt_oss.py` mirrors
OpenAI’s reference Transformer architecture. This guide walks through loading an
official checkpoint, mapping it into the Equinox example, and exporting an ONNX
graph that preserves the IR-only layout used by the test-suite.

All commands assume you are at the project root with the Poetry environment
available. The workflow targets CPU-only tools; feel free to switch the device
flags to `cuda:X` if you have GPU support.

## Prerequisites

Install the optional dependencies needed to read GPT-OSS checkpoints:

```bash
poetry install --with test
poetry run pip install gpt-oss safetensors
poetry run pip install torch --index-url https://download.pytorch.org/whl/cpu
poetry run pip install huggingface_hub
```

> The `gpt-oss` wheel already depends on `torch` and `safetensors`, but
> installing them explicitly (`cpu` wheel shown above) makes it clear which
> variants the parity scripts expect.

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

## 2. Export the Equinox example to ONNX

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

## 3. Validate parity (optional)

Numerical comparisons between the PyTorch and ONNX/JAX paths are covered by the
new regression tests in `tests/extra_tests/test_eqx_gpt_oss_parity.py`. When the
optional dependencies above are installed, this test asserts the Equinox model
tracks the PyTorch reference to within a small tolerance (absolute differences
stay below `~1e0` when working in bfloat16).

Run a focused check with:

```bash
poetry run pytest -q tests/extra_tests/test_eqx_gpt_oss_parity.py
```
