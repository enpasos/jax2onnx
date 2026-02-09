# MaxText Support 🚀

[MaxText](https://github.com/AI-Hypercomputer/maxtext) is a high-performance, arbitrary-scale, open-source LLM framework written in pure Python/JAX. `jax2onnx` provides a self-contained example stack to export these models to ONNX.

- MaxText (DeepSeek, Gemma, GPT-3, Kimi, Llama, Mistral, Qwen) - https://github.com/AI-Hypercomputer/maxtext

## Related Examples

All supported MaxText model families (`DeepSeek`, `Gemma`, `Llama`, `Mistral`, `Qwen`, etc.) are listed with their test status in the [Examples](../examples.md) table.

## Supported Families

We support exporting the following model families from the MaxText model zoo:

*   **DeepSeek** (v2 / v3)
*   **Gemma** (2 / 3)
*   **GPT-3**
*   **Kimi** (K2)
*   **Llama** (2 / 3 / 3.1 / 4)
*   **Mistral**
*   **Qwen** (3 / 3-Next / Omni)

## Usage

### Dependencies

To run the MaxText examples, you need to install the following additional dependencies:

```bash
poetry install --with maxtext
```

> **Note:** This installs `omegaconf`, `transformers`, `sentencepiece`, `tensorflow-cpu`, and `tensorboardX`. `tensorflow-cpu` is required because MaxText uses `tensorboard` and some TF utilities.
> It does **not** install the MaxText source tree itself; use `JAX2ONNX_MAXTEXT_SRC` (recommended) or install a `MaxText` package separately.

### Environment Configuration

*   **`JAX2ONNX_MAXTEXT_SRC`** (Optional): Path to a local clone of the MaxText repository. If not set, the system attempts to resolve it from an installed `MaxText` package.
*   **`JAX2ONNX_MAXTEXT_MODELS`** (Optional): A comma-separated list of model config names to test (e.g., `llama2-7b.yml`). If unset, it defaults to a standard set of representative models.

## Testing

To run all the latest MaxText examples (use `poetry run` to stay in the project venv):

```bash
cd tmp
git clone https://github.com/AI-Hypercomputer/maxtext.git
cd ..
export JAX2ONNX_MAXTEXT_SRC=tmp/maxtext
export JAX2ONNX_MAXTEXT_MODELS=all  # or "gemma-2b,llama2-7b"
poetry install --with maxtext
poetry run python scripts/generate_tests.py
poetry run pytest -q tests/examples/test_maxtext.py
```

ONNX outputs land in `docs/onnx/examples/maxtext`.

This will:
1.  Dynamically discover MaxText configs.
2.  Instantiate the models with minimal inference settings (batch_size=1, seq_len=32).
3.  Export them to ONNX and verify the graph structure.
