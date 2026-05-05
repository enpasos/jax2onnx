# MaxDiffusion Support

`jax2onnx` includes optional
[MaxDiffusion](https://github.com/google/maxdiffusion) examples for exporting
selected diffusion-model components to ONNX. The current example stack focuses
on the Flax UNet used by SDXL-family configurations.

## Related Examples

All supported MaxDiffusion model configurations are listed with their validation
status in the [Examples](../examples.md) reference table.

## Supported Families

The supported MaxDiffusion example surface is the **UNet**
(`FlaxUNet2DConditionModel`) from SDXL-family configurations. The published
examples currently include:

- `base14.yml`
- `base21.yml`
- `base_2_base.yml`
- `base_xl.yml`
- `base_xl_lightning.yml`

Wan, LTX-Video, and Flux transformer architectures are outside the current
example surface.

## Usage

### Dependencies

Use Python 3.12 or newer for local MaxDiffusion validation.

```bash
poetry install --with maxdiffusion
```

> **Note:** This installs `omegaconf`, `transformers`, `tensorflow-cpu`,
> `tensorboardX`, `absl-py`, `datasets`, and other utilities.
> It does **not** install the MaxDiffusion source tree itself. You must
> point `JAX2ONNX_MAXDIFFUSION_SRC` at a local clone.

### Environment Configuration

- **`JAX2ONNX_MAXDIFFUSION_SRC`**: Path to a local MaxDiffusion checkout. Without
  this variable, MaxDiffusion examples are skipped.
- **`JAX2ONNX_MAXDIFFUSION_MODELS`**: Optional comma-separated list of config file
  names, for example `base_xl.yml,base_2_base.yml`. Set it to `all` to validate
  every discoverable non-skipped config. The default is `base_xl.yml`.

## Validation

Use the [Examples](../examples.md) reference table to inspect the currently
published MaxDiffusion ONNX exports and their validation status. Testcase links
open the model directly in Netron.

For local validation, use a compatible MaxDiffusion checkout:

```bash
git clone https://github.com/google/maxdiffusion.git
export JAX2ONNX_MAXDIFFUSION_SRC="$PWD/maxdiffusion"
export JAX2ONNX_MAXDIFFUSION_MODELS=base_xl.yml
poetry install --with maxdiffusion

poetry run python scripts/generate_tests.py
poetry run pytest -q tests/examples/test_maxdiffusion.py
```

Set `JAX2ONNX_MAXDIFFUSION_MODELS=all` only when you want to validate every
supported config; the generated test file is created after `scripts/generate_tests.py`
sees the configured MaxDiffusion checkout.
