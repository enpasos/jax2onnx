# SotA Example Maintenance

This page is maintainer-facing. User-facing SotA pages should explain stable
export and validation workflows; generated tests, pinned upstream refs, parity
baselines, and artifact promotion details belong here.

## Reference Table Refresh

Use `scripts/generate_readme.sh` when the public component/example tables need
to include MaxText or MaxDiffusion status. It installs both optional dependency
groups, prepares pinned upstream checkouts, preserves dirty primary checkouts by
using clean fallback directories, then runs `scripts/generate_readme.py`.
Run this workflow from Python 3.12 or newer, matching the optional SotA stack
used by `scripts/run_all_checks.sh`.

```bash
./scripts/generate_readme.sh
```

Override `JAX2ONNX_MAXTEXT_MODELS` or `JAX2ONNX_MAXDIFFUSION_MODELS` before
running the script when a review should cover only a subset.

## MaxText

Use a compatible MaxText checkout when refreshing generated examples:

```bash
mkdir -p tmp
if [ ! -d tmp/maxtext/.git ]; then
  git clone https://github.com/AI-Hypercomputer/maxtext.git tmp/maxtext
fi
git -C tmp/maxtext fetch --depth 1 origin 17d805e3488104b5de96bd19be09491ff73c57c1
git -C tmp/maxtext checkout --detach FETCH_HEAD

export JAX2ONNX_MAXTEXT_SRC="$PWD/tmp/maxtext"
export JAX2ONNX_MAXTEXT_MODELS=all  # or "gemma-2b,llama2-7b"
export JAX2ONNX_MAXTEXT_REF=17d805e3488104b5de96bd19be09491ff73c57c1
poetry install --with maxtext
poetry run python scripts/generate_tests.py
poetry run pytest -q tests/examples/test_maxtext.py
```

The standard repository runner can include the same checks:

```bash
JAX2ONNX_RUN_MAXTEXT=1 ./scripts/run_all_checks.sh
```

By default, `run_all_checks.sh` does not run MaxText checks. With
`JAX2ONNX_RUN_MAXTEXT=1`, it prepares `JAX2ONNX_MAXTEXT_SRC`, installs
`--with maxtext`, regenerates tests, runs the MaxText example tests, then
continues with the regular suite. On Python versions below 3.12, the MaxText
block is skipped.

## MaxDiffusion

Use a compatible MaxDiffusion checkout when refreshing generated examples:

```bash
mkdir -p tmp
if [ ! -d tmp/maxdiffusion/.git ]; then
  git clone https://github.com/google/maxdiffusion.git tmp/maxdiffusion
fi
git -C tmp/maxdiffusion fetch --depth 1 origin b4f95730bf4f00c4fd9e3dd2fdda1b50484afda2
git -C tmp/maxdiffusion checkout --detach FETCH_HEAD

export JAX2ONNX_MAXDIFFUSION_SRC="$PWD/tmp/maxdiffusion"
export JAX2ONNX_MAXDIFFUSION_MODELS=all
export JAX2ONNX_MAXDIFFUSION_REF=b4f95730bf4f00c4fd9e3dd2fdda1b50484afda2
poetry install --with maxdiffusion
poetry run python scripts/generate_tests.py
poetry run pytest -q tests/examples/test_maxdiffusion.py
```

The standard repository runner can include the same checks:

```bash
JAX2ONNX_RUN_MAXDIFFUSION=1 ./scripts/run_all_checks.sh
```

By default, `run_all_checks.sh` does not run MaxDiffusion checks. With
`JAX2ONNX_RUN_MAXDIFFUSION=1`, it prepares `JAX2ONNX_MAXDIFFUSION_SRC`, checks
out the pinned ref, installs `--with maxdiffusion`, regenerates tests, runs the
MaxDiffusion example tests, then continues with the regular suite. On Python
versions below 3.12, the MaxDiffusion block is skipped.

## GPT-OSS Parity

Before promoting a GPT-OSS ONNX export, prove the staged Flax bundle still
matches the reference checkpoint. Keep parity transcripts in review artifacts or
durable maintainer notes when they support a public example update; do not put
one-off run logs in user-facing docs.

Routing parity:

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

Two-layer debug export and hidden-state comparison:

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

Torch-to-Flax parity:

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

Do not commit generated `.onnx` or `.onnx.data` files; publish sample models
according to [Artifact Publishing](artifact_publishing.md).

## DINOv3 NNX

The NNX DINO example registry uses explicit component names such as
`NnxDinoPatchEmbed`, `NnxDinoAttentionCore`, `NnxDinoAttention`,
`NnxDinoBlock`, and `FlaxDINOv3VisionTransformer` so generated test files and
published model paths stay unambiguous.

To capture submodules alongside the full ViT during maintenance:

```bash
poetry run pytest -q tests/examples/test_nnx_dino.py
```

When the NNX stack or the Equinox sibling implementation changes, also run the
forward-parity regression that copies Equinox weights into the NNX model:

```bash
poetry run pytest -q tests/extra_tests/examples/test_nnx_dino_parity.py
```

Keep generated artifacts local while validating, then publish sample models via
the artifact workflow when the public example table needs updated links.
