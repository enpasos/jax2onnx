# MaxText Support 🚀

[MaxText](https://github.com/AI-Hypercomputer/maxtext) is a high-performance, arbitrary-scale, open-source LLM framework written in pure Python/JAX. `jax2onnx` provides a self-contained example stack to export these models to ONNX.

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
pip install flax omegaconf orbax-checkpoint transformers sentencepiece tensorflow-cpu tensorboardX onnx-ir
```

> **Note:** `tensorflow-cpu` is required because MaxText uses `tensorboard` and some TF utilities for data loading and logging.

### Environment Configuration

*   **`JAX2ONNX_MAXTEXT_SRC`** (Optional): Path to a local clone of the MaxText repository. If not set, the system attempts to resolve it from an installed `MaxText` package.
*   **`JAX2ONNX_MAXTEXT_MODELS`** (Optional): A comma-separated list of model config names to test (e.g., `llama2-7b.yml`). If unset, it defaults to a standard set of representative models.

## Testing

You can run the verification tests to ensure models verify and export correctly using `pytest`:

```bash
pytest -v -k maxtext
```

This will:
1.  Dynamically discover MaxText configs.
2.  Instantiate the models with minimal inference settings (batch_size=1, seq_len=32).
3.  Export them to ONNX and verify the graph structure.
