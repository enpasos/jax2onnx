# Dependencies

**Latest stable releases of major dependencies:**

| Library       | Version |
|:--------------|:--------|
| [`JAX`](https://github.com/jax-ml/jax) | 0.11.2 |
| [`Flax`](https://github.com/google/flax) | 0.12.10 |
| [`Equinox`](https://github.com/patrick-kidger/equinox) | 0.13.8 |
| [`onnx-ir`](https://github.com/onnx/ir-py) | 1.0.0 |
| [`onnx`](https://github.com/onnx/onnx) | 1.23.0 |
| [`onnxruntime`](https://github.com/microsoft/onnxruntime) | 1.30.0 |
| [`onnxruntime-web`](https://www.npmjs.com/package/onnxruntime-web) | 1.30.0 |

The latest releases are not necessarily selected for every supported Python
version; Poetry resolves compatible versions for each environment.
`onnxruntime-web` 1.30.0 is locked in `package-lock.json` for the
Node.js/WASM and Chromium/WASM smoke flows. The supported Python runtime
minimum remains ONNX Runtime 1.24.1; exports do not select operators based on
the runtime installed on the export machine.

*For minimum supported versions and optional extras, see [`pyproject.toml`](https://github.com/enpasos/jax2onnx/blob/main/pyproject.toml). For the fully resolved Poetry environment, see `poetry.lock`.*
