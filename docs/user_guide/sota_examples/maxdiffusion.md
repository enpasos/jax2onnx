# MaxDiffusion Support 🚀

[MaxDiffusion](https://github.com/AI-Hypercomputer/maxdiffusion) is a collection of reference implementations of various Latent Diffusion Models written in pure Python/JAX that runs on TPUs and GPUs. `jax2onnx` provides a self-contained example stack to export these models to ONNX.

## Related Examples

All supported MaxDiffusion model configurations are listed with their test status in the [Examples](../examples.md) table.

## Supported Families

We currently support exporting the **UNet** (`FlaxUNet2DConditionModel`) from
SDXL-family configurations.  Wan, LTX-Video, and Flux transformer architectures
are not yet wired up and are skipped during registration.

*   **SDXL** (`base_xl.yml`, `base_xl_lightning.yml`, `base_2_base.yml`)

## Usage

### Dependencies

```bash
poetry install --with maxdiffusion
```

> **Note:** This installs `omegaconf`, `transformers`, `tensorflow-cpu`,
> `tensorboardX`, `absl-py`, `datasets`, and other utilities.
> It does **not** install the MaxDiffusion source tree itself; you must
> point `JAX2ONNX_MAXDIFFUSION_SRC` at a local clone.

### Environment Configuration

*   **`JAX2ONNX_MAXDIFFUSION_SRC`** (**required**): Path to a local clone of the MaxDiffusion repository.
    Without this variable the plugin is silently skipped.
*   **`JAX2ONNX_MAXDIFFUSION_MODELS`** (optional): Comma-separated list of
    config file names (e.g. `base_xl.yml`).  Set to `all` to test every
    config.  Defaults to `base_xl.yml`.
*   **`JAX2ONNX_MAXDIFFUSION_REF`** (optional): Git ref used by
    `run_all_checks.sh`.  Defaults to a pinned commit hash for
    reproducibility.

## Testing

```bash
cd tmp
git clone https://github.com/AI-Hypercomputer/maxdiffusion.git
cd ..
export JAX2ONNX_MAXDIFFUSION_SRC=tmp/maxdiffusion
poetry install --with maxdiffusion
poetry run python scripts/generate_tests.py
poetry run pytest -q tests/examples/test_maxdiffusion.py
```

ONNX outputs land in `docs/onnx/examples/maxdiffusion`.

You can also include the MaxDiffusion SotA checks in the standard repository runner:

```bash
JAX2ONNX_RUN_MAXDIFFUSION=1 ./scripts/run_all_checks.sh
```

By default, `run_all_checks.sh` does **not** run MaxDiffusion checks.
With `JAX2ONNX_RUN_MAXDIFFUSION=1`, it clones or reuses
`JAX2ONNX_MAXDIFFUSION_SRC` (default: `tmp/maxdiffusion`), checks out
the pinned ref, installs `--with maxdiffusion`, regenerates tests, runs
`tests/examples/test_maxdiffusion.py`, then executes the full pytest suite.
On Python < 3.12 the script automatically skips the MaxDiffusion block.
