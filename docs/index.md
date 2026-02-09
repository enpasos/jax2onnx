# jax2onnx 🌟

[![CI](https://github.com/enpasos/jax2onnx/actions/workflows/ci.yml/badge.svg)](https://github.com/enpasos/jax2onnx/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/jax2onnx.svg)](https://pypi.org/project/jax2onnx/)

`jax2onnx` converts your [JAX](https://docs.jax.dev/), [Flax NNX](https://flax.readthedocs.io/en/latest/), [Flax Linen](https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html), [Equinox](https://docs.kidger.site/equinox/) functions directly into the ONNX format.

![jax2onnx.svg](https://enpasos.github.io/jax2onnx/images/jax2onnx.svg)

## ✨ Key Features

- **Simple API** – Convert JAX callables using `to_onnx(...)`.
- **Model structure preserved** – With `@onnx_function`, submodules appear as named functions in the ONNX graph.
- **Dynamic input support** – Use abstract dimensions like `'B'` or pass scalars as runtime inputs.
- **Plugin-based extensibility** – Add support for new primitives by writing small, local plugins.
- **onnx-ir native pipeline** – Conversion, optimization, and post-processing all run on the typed [`onnx_ir`](https://github.com/onnx/ir-py) toolkit.
- **Netron-friendly outputs** – Generated graphs carry shape/type annotations and a clean hierarchy.

## 🚀 Getting Started

Check out the [Getting Started guide](user_guide/getting_started.md) to install and export your first model in minutes.

See [Roadmap](about/roadmap.md) for planned features and [Past Versions](about/past_versions.md) for the full release archive.

---

**Happy converting! 🎉**
