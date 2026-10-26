# Dependencies

**Latest stable releases of major dependencies (as of 2026-10-13):**

| Library       | Version |
|:--------------|:--------|
| [`JAX`](https://github.com/jax-ml/jax) | 0.11.2 |
| [`Flax`](https://github.com/google/flax) | 0.12.10 |
| [`Equinox`](https://github.com/patrick-kidger/equinox) | 0.13.8 |
| [`onnx-ir`](https://github.com/onnx/ir-py) | 1.0.0 |
| [`onnx`](https://github.com/onnx/onnx) | 1.23.2 |
| [`onnxruntime`](https://github.com/microsoft/onnxruntime) | 1.31.0 |
| [`onnxruntime-web`](https://www.npmjs.com/package/onnxruntime-web) | 1.30.0 |

The latest releases are not necessarily selected for every supported Python
version; Poetry resolves compatible versions for each environment.
The supported ONNX IR minimum is **1.0.0** (`onnx-ir>=1.0.0`).
`onnxruntime-web` 1.30.0 is locked in `package-lock.json` for the
Node.js and Chromium WASM validation flows. The supported Python runtime
minimum remains ONNX Runtime 1.24.1; exports do not select operators based on
the runtime installed on the export machine.

## ONNX Runtime kernel gaps

Some lowerings depend on kernels or behavior that ONNX Runtime lacks. Because
exports must run on every supported version, a workaround can be removed only
once the supported minimum includes the fix; a fix in the locked release alone
is not enough. Re-check this table whenever the locked ONNX Runtime or the
supported minimum changes, and update the versions it was checked against. The
`minimum-dependencies` CI job runs the related precision tests at the supported
minimum.

| Gap | Checked | Affected exports | Current handling |
|:--|:--|:--|:--|
| No bfloat16 `Log` on CPU or CUDA; no bfloat16 `Exp` on CPU | CPU 1.24.1, 1.31.0; CUDA 1.31.0 | bfloat16 `log1p`, `expm1`, `sinh`, `jax.nn.log1mexp`, `jax.nn.log_sigmoid` | Computed in float32 and rounded back. Float32 is also the more accurate choice (within 1 ulp), so a new kernel alone is no reason to switch. |
| No float64 or bfloat16 `Sinh` on CPU; no `Sinh` on CUDA | CPU 1.24.1, 1.31.0; CUDA 1.31.0 | `lax.sinh` and `jnp.sinh` other than float32 | expm1-based form |
| No float64 `Cos` on CPU | Missing in 1.24.1, present in 1.31.0 | float64 `lax.cos`, `jnp.cos` | Half-angle form `1 - 2 * Sin(x / 2)²` |
| No float64 `Erf` on CPU | CPU 1.24.1, 1.31.0 | float64 exact `jax.nn.gelu`, `nnx.gelu` | None: the export uses `Erf` (or `Gelu`, which expands to it) and fails to load; float64 test variants are skipped. |
| CPU `Where` returns `+0.0` when it selects `-0.0` | CPU 1.24.1, 1.31.0 | `log1p`, `expm1`, `sinh` | `log1p` and `sinh` end in `Mul` to keep the sign; `expm1(-0.0)` exports as `+0.0`. |

To check a kernel, create a session for a one-node model with that operator and
dtype. For providers other than CPU, set the session option
`session.disable_cpu_ep_fallback` to `1` so the node cannot fall back to CPU. A
`NOT_IMPLEMENTED` error means the kernel is missing.

*For dependency requirements and optional dependency groups, see [`pyproject.toml`](https://github.com/enpasos/jax2onnx/blob/main/pyproject.toml). For resolved Python dependency versions, see `poetry.lock`.*
