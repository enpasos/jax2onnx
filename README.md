# jax2onnx 🌟

[![CI](https://github.com/enpasos/jax2onnx/actions/workflows/ci.yml/badge.svg)](https://github.com/enpasos/jax2onnx/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/jax2onnx.svg)](https://pypi.org/project/jax2onnx/) 
[![Documentation](https://img.shields.io/badge/docs-live-blue)](https://enpasos.github.io/jax2onnx/)

`jax2onnx` converts your [JAX](https://docs.jax.dev/),  [Flax NNX](https://flax.readthedocs.io/en/latest/), [Flax Linen](https://flax-linen.readthedocs.io/en/latest/), [Equinox](https://docs.kidger.site/equinox/) functions directly into the ONNX format.


![jax2onnx.svg](https://enpasos.github.io/jax2onnx/images/jax2onnx.svg)

## 📚 Documentation

**[Read the full documentation here](https://enpasos.github.io/jax2onnx/)**

## 🚀 Quick Install

```bash
pip install jax2onnx
```

## ⚡ Quick Usage

```python
from jax2onnx import to_onnx
from flax import nnx

model = MyFlaxModel(...)
to_onnx(model, [("B", 32)], return_mode="file", output_path="model.onnx")
```

## 🤝 Contributing

We warmly welcome contributions! Please check our [Developer Guide](https://enpasos.github.io/jax2onnx/developer_guide/plugin_system/) for plugin tutorials and architecture details.

## 📜 License

Apache License, Version 2.0. See [`LICENSE`](./LICENSE).


## 🌟 Special Thanks

A huge thank you to all [our contributors and the community](https://enpasos.github.io/jax2onnx/about/acknowledgements/) for their help and inspiration!

---

**Happy converting! 🎉**
