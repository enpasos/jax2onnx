# jax2onnx 🌟

[![CI](https://github.com/enpasos/jax2onnx/actions/workflows/ci.yml/badge.svg)](https://github.com/enpasos/jax2onnx/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/jax2onnx.svg)](https://pypi.org/project/jax2onnx/)

`jax2onnx` converts your [JAX](https://docs.jax.dev/),  [Flax NNX](https://flax.readthedocs.io/en/latest/), [Flax Linen](https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html), [Equinox](https://docs.kidger.site/equinox/) functions directly into the ONNX format.


![jax2onnx.svg](https://enpasos.github.io/jax2onnx/readme/images/jax2onnx.svg)

## ✨ Key Features

- **simple API**  
  Easily convert JAX callables—including Flax NNX, Flax Linen and Equinox models—into ONNX format using `to_onnx(...)`.

- **model structure preserved**  
  With `@onnx_function`, submodules appear as named functions in the ONNX graph (e.g. in Netron). Useful for readability and reuse.

- **dynamic input support**  
  Use abstract dimensions like `'B'` or pass scalars as runtime inputs. Models stay flexible without retracing.

- **plugin-based extensibility**  
  Add support for new primitives by writing small, local plugins.

- **onnx-ir native pipeline**  
  Conversion, optimization, and post-processing all run on the typed `onnx_ir` toolkit—no protobuf juggling—and stay memory-lean before the final ONNX serialization.

- **Netron-friendly outputs**  
  Generated graphs carry shape/type annotations and a clean hierarchy, so tools like Netron stay easy to read.


## 🚀 Getting Started

Check out the [Getting Started guide](user_guide/getting_started.md) to install and export your first model in minutes.

---

## 🤝 How to Contribute

We warmly welcome contributions!

**How you can help:**

- **Add a plugin:** Extend `jax2onnx` by writing a simple Python file in [`jax2onnx/plugins`](https://github.com/enpasos/jax2onnx/tree/main/jax2onnx/plugins):
  a primitive or an example. The [Plugin Quickstart](dev_guides/plugin_quickstart.md) walks through the process step-by-step.
- **Bug fixes & improvements:** PRs and issues are always welcome.
 

---



---

## 📅 Roadmap and Releases

### **Planned**

  * Broaden coverage of JAX, Flax NNX/Linen, and Equinox components.
  * Expand SotA example support for vision and language models.
  * Improve support for **physics-based simulations**

### **Current Productive Version**

* **0.11.0**:
  * Initial Flax Linen support: core layers (Dense/DenseGeneral, Conv/ConvTranspose/ConvLocal, pooling, BatchNorm/LayerNorm/GroupNorm/RMSNorm/InstanceNorm), Dropout, Einsum/Embed, spectral/weight norm wrappers, activation coverage (GELU plus glu/hard_*/log_*/relu6/silu-swish/tanh/normalize/one_hot), attention stack (dot_product_attention, dot_product_attention_weights, make_attention_mask/make_causal_mask, SelfAttention, MultiHeadDotProductAttention, MultiHeadAttention), recurrent stack (SimpleCell, GRUCell, MGUCell, LSTMCell, OptimizedLSTMCell, ConvLSTMCell, RNN, Bidirectional), and Linen examples (MLP/CNN/Sequential).
  * Modernized IR optimization pipeline: standard onnx_ir CSE pass adoption, removed legacy helpers/getattr patterns, and simplified tests with direct graph iteration.

### **Past Versions**

See [`past_versions`](readme/past_versions.md) for the full release archive.


---

## 📌 Dependencies

**Latest supported version of major dependencies:**

| Library       | Versions |  
|:--------------|:---------| 
| `JAX`         | 0.8.2    | 
| `Flax`        | 0.12.2   | 
| `Equinox`     | 0.13.2   | 
| `onnx-ir`     | 0.1.13   | 
| `onnx`        | 1.20.0   |  
| `onnxruntime` | 1.23.2   |  

*For exact pins and extras, see `pyproject.toml`.*


---

## 📜 License

This project is licensed under the Apache License, Version 2.0. See [`LICENSE`](https://github.com/enpasos/jax2onnx/blob/main/LICENSE) for details.

---

## 🌟 Special Thanks

✨ Special thanks to [@clementpoiret](https://github.com/clementpoiret) for initiating Equinox support and for [Equimo](https://github.com/clementpoiret/equimo), which brings modern vision models—such as [DINOv3](https://ai.meta.com/dinov3/)—to JAX/Equinox.

✨ Special thanks to [@justinchuby](https://github.com/justinchuby) for introducing **onnx-ir** as a scalable and more efficient way to handle ONNX model construction.  

✨ Special- Language: [MaxText](readme/maxtext.md) model zoointroducing us to [gpt-oss-jax-vs-torch-numerical-comparison](https://github.com/atveit/gpt-oss-jax-vs-torch-numerical-comparison).

✨ Special thanks for example contributions to [@burakssen](https://github.com/burakssen), [@Cadynum](https://github.com/Cadynum), [@clementpoiret](https://github.com/clementpoiret) and [@PVirie](https://github.com/PVirie)

✨ Special thanks for plugin contributions to [@burakssen](https://github.com/burakssen), [@clementpoiret](https://github.com/clementpoiret), [@Clouder0](https://github.com/Clouder0), [@rakadam](https://github.com/rakadam) and [benmacadam64](https://github.com/benmacadam64)

✨ Special thanks to [@benmacadam64](https://github.com/benmacadam64) for championing the complex-number handling initiative.

✨ Special thanks to [tumaer/JAXFLUIDS](https://github.com/tumaer/JAXFLUIDS) for contributing valuable insights rooted in physics simulation use cases.

✨ Special thanks to [@lutzroeder](https://github.com/lutzroeder) for making shapes internal to ONNX function visible in his great Netron viewer.

- [ONNX: Function value_info support #1447](https://github.com/lutzroeder/netron/issues/1447)


✨ Special thanks to the community members involved in:

- [Flax Feature Request #4430](https://github.com/google/flax/issues/4430)
- [JAX Feature Request #26430](https://github.com/jax-ml/jax/issues/26430)

✨ Special thanks to [@limarta](https://github.com/limarta), whose elegant [jaxpr-to-ONNX demonstration](https://gist.github.com/limarta/855a88cc1c0163487a9dc369891147ab) significantly inspired this project.

---

**Happy converting! 🎉**
