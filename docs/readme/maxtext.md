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

## 0.11.1 Updates

Version **0.11.1** introduced:
*   A self-contained MaxText example stack.
*   Tests covering all the families listed above.
*   MaxText dependency stubs and new primitive support needed for these exports.
*   Tightened subgraph cleanup for cleaner ONNX graphs.

*(More detailed guides and equivalence checks coming soon)*
