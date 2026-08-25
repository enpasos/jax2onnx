# Dependencies

**Latest supported versions of major dependencies:**

| Library       | Version |
|:--------------|:--------|
| [`JAX`](https://github.com/jax-ml/jax) | 0.11.0 |
| [`Flax`](https://github.com/google/flax) | 0.12.8 |
| [`Equinox`](https://github.com/patrick-kidger/equinox) | 0.13.8 |
| [`onnx-ir`](https://github.com/onnx/ir-py) | 1.0.0 |
| [`onnx`](https://github.com/onnx/onnx) | 1.22.0 |
| [`onnxruntime`](https://github.com/microsoft/onnxruntime) | 1.29.0 |
| [`onnxruntime-web`](https://www.npmjs.com/package/onnxruntime-web) | 1.29.0 |

`onnxruntime-web` tracks the latest stable npm release and is validated in both
Node.js/WASM and Chromium smoke flows.

*For minimum supported versions and optional extras, see [`pyproject.toml`](https://github.com/enpasos/jax2onnx/blob/main/pyproject.toml). For the fully resolved Poetry environment, see `poetry.lock`.*
