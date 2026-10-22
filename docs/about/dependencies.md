# Dependencies

**Selected major dependency versions:**

| Library       | Version |
|:--------------|:--------|
| [`JAX`](https://github.com/jax-ml/jax) | 0.12.0.1 |
| [`Flax`](https://github.com/google/flax) | 0.12.10 |
| [`Equinox`](https://github.com/patrick-kidger/equinox) | 0.13.8 |
| [`onnx-ir`](https://github.com/onnx/ir-py) | 1.1.0 |
| [`onnx`](https://github.com/onnx/onnx) | 1.23.2 |
| [`onnxruntime`](https://github.com/microsoft/onnxruntime) | 1.31.0 |
| [`onnxruntime-web`](https://www.npmjs.com/package/onnxruntime-web) | 1.30.0 |

The latest releases are not necessarily selected for every supported Python
version; Poetry resolves compatible versions for each environment.
The modern Poetry stack uses JAX and jaxlib **0.12.0.1** and ONNX IR
**1.1.0** (updated 2026-10-22).
The supported minimums are **Python 3.11**, **JAX/jaxlib 0.10.2** and
**Flax 0.12.8**. `poetry.lock` retains JAX/jaxlib **0.10.2** with Flax **0.12.8**
for Python 3.11/3.12, and the modern stack above for Python 3.13/3.14.
`uv.lock` selects JAX/jaxlib **0.10.2** on Python 3.11 and **0.12.0** on
Python 3.12+, with Flax **0.12.8** and ONNX IR **1.0.0**; this separate lock
retains its earlier resolved versions.
The minimum-dependency CI job installs every direct runtime dependency at its
declared lower bound on Python 3.11; a locked environment with the minimum
framework versions can still use newer versions of other dependencies.
The supported ONNX IR minimum is **1.0.0** (`onnx-ir>=1.0.0`).
`onnxruntime-web` 1.30.0 is locked in `package-lock.json` for the
Node.js and Chromium WASM validation flows. The supported Python runtime
minimum remains ONNX Runtime 1.24.1; exports do not select operators based on
the runtime installed on the export machine.

*For dependency requirements and optional dependency groups, see [`pyproject.toml`](https://github.com/enpasos/jax2onnx/blob/main/pyproject.toml). For resolved Python dependency versions and their markers, see `poetry.lock` and `uv.lock`.*
