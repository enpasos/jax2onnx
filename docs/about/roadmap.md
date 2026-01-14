# Roadmap

## Planned

* Broaden coverage of JAX, Flax NNX/Linen, and Equinox components.
* Expand SotA example support for vision and language models.
* Improve support for **physics-based simulations**.
* Support for **[MaxDiffusion](https://github.com/AI-Hypercomputer/maxdiffusion)**.

## Current Version

### **jax2onnx 0.11.1** – MaxText model family coverage & cleaner exported graphs

* **Comprehensive MaxText example stack:**
  Added a fully **comprehensive MaxText example + test suite** covering exports for **DeepSeek, Gemma, GPT-3, Kimi, Llama, Mistral, and Qwen** model families.

* **MaxText stubs & new primitive coverage:**
  Introduced **MaxText dependency stubs** and implemented **new primitive support** required to enable those exports end-to-end.

* **Cleaner ONNX graphs via stricter subgraph cleanup:**
  Tightened **subgraph cleanup** to produce **cleaner, more minimal ONNX graphs** (less leftover/unused substructure after export).



## Past Versions

See [Past Versions](past_versions.md) for the full release archive.
