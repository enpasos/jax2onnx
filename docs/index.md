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

**Happy converting! 🎉**
