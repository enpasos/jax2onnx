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

## Observed native JAX CPU BF16 TopK failure

CI549 and CI551 captured a failure in the official JAX/jaxlib 0.10.2 CPU
reference before ONNX conversion. On the affected hosted CPU, the first B2
native call selected the distinct positive BF16 subnormal words `0001` and
`007f` as indices `[0, 1]` instead of `[1, 0]`; selected-input words were
retained. Passing local runs do not erase this hardware-dependent
counterexample. The exact compiler instruction responsible is unestablished.

The original reproduction remains in
[`test_source_bfloat16_top_k_portable_words_and_native_equivalence[finite-inline]`](https://github.com/enpasos/jax2onnx/blob/0a9638da26585bfafc0485ee9766135cd0a991a6/tests/extra_tests/converter/test_source_native_top_k.py#L1855-L1940),
using the declared [`_bf16_word_input(False)` payload](https://github.com/enpasos/jax2onnx/blob/0a9638da26585bfafc0485ee9766135cd0a991a6/tests/extra_tests/converter/test_source_native_top_k.py#L1654-L1677).
It preserves exact BF16 storage, checks indices, dtype and shape, and compares
original words for B2/B5 fresh, warmed and eager calls. The native
[capture diagnostic](https://github.com/enpasos/jax2onnx/blob/0a9638da26585bfafc0485ee9766135cd0a991a6/scripts/_source_bf16_topk_capture.py)
and unchanged 99-case owner remain available; the focused reproduction does
not replace that owner or the original context acceptance obligations.

This is a native-reference failure, distinct from converter defects, ONNX
representation restrictions and ONNX Runtime behavior. Affected acceptance
remains failed or unverified until demonstrated on the supported official
runtime and target. Do not weaken comparisons or change the reference runtime
to obtain acceptance. Earlier custom compiler build attempts produced no
numerically accepted repair; their unapproved patch and build infrastructure
have been removed. Supported official dependencies and lock markers remain the
reference; custom builds, patches and binary overlays require separate,
explicit approval.

## ONNX Runtime kernel gaps

Some lowerings depend on kernels or behavior that ONNX Runtime lacks. Because
exports must run on every supported version, a workaround can be removed only
once the supported minimum includes the fix; a fix in the locked release alone
is not enough. Re-check this table whenever the locked ONNX Runtime or the
supported minimum changes, and update the versions it was checked against. The
`minimum-dependencies` CI job runs the related precision tests at the supported
minimum.

The table records the checked provider/version combinations and the public
LEGACY lowerings. These observations do not establish support or numerical
acceptance for every model. The private SOURCE path keeps its own typed
algorithms and admission gates, described below; it is not a public export
option.

| Gap | Checked | Affected exports | Current handling |
|:--|:--|:--|:--|
| No bfloat16 `Log` on CPU or CUDA; no bfloat16 `Exp` on CPU | CPU 1.24.1, 1.31.0; CUDA 1.31.0 | bfloat16 `log1p`, `expm1`, `sinh`, `jax.nn.log1mexp`, `jax.nn.log_sigmoid` | Computed in float32 and rounded back. The recorded precision controls check accuracy within 1 ulp; a new kernel alone does not justify changing this algorithm. |
| No float64 or bfloat16 `Sinh` on CPU; no `Sinh` on CUDA | CPU 1.24.1, 1.31.0; CUDA 1.31.0 | `jnp.sinh`; `lax.sinh` other than float32 | expm1-based form. Float32 `lax.sinh` keeps `Sinh`, which runs on CPU in CUDA sessions. |
| No float64 `Cos` on CPU | Missing in 1.24.1, present in 1.31.0 | float64 `lax.cos`, `jnp.cos` | Half-angle form `1 - 2 * Sin(x / 2)²` |
| No float64 `Erf` on CPU | CPU 1.24.1, 1.31.0 | Public LEGACY float64 exact `jax.nn.gelu`, `nnx.gelu` | The legacy export uses `Erf` (or `Gelu`, which expands to it) and fails to load; its generated metadata restricts these cases to float32. This does not describe the private SOURCE Erfc algorithm. |
| CPU `Where` returns `+0.0` when it selects `-0.0` | CPU 1.24.1, 1.31.0 | `log1p`, `expm1`, `sinh` | `log1p` and `sinh` end in `Mul` to keep the sign; `expm1(-0.0)` exports as `+0.0`. |

Private SOURCE `log1p` and `expm1` retain their existing typed elementary
algorithms and bit-preserving selection, with separate dtype/opset and numerical
checks. SOURCE `sinh` retains typed `Sinh`, so its kernel limitations still
apply. SOURCE exact GELU traces the original JAX computation through Erfc;
its owning typed implementation avoids reliance on a float64 `Erf` kernel at
the admitted opsets. Those finite implementations do not qualify arbitrary
models or override an unsupported dtype, opset, provider or numerical contract.

SOURCE floating `reduce_min` and `reduce_max` now retain a native typed
`ReduceMin`/`ReduceMax` candidate and independently reduce NaN and zero-sign
predicates. Typed `Gather`/`GatherElements` selection restores NaN propagation
and the required zero word without using floating `Where`. Integer extrema,
empty identities, no-axis reductions and public LEGACY behavior retain their
existing paths. Focused FP32/FP64 NaN/zero-sign regressions pass on official
JAX/jaxlib 0.10.2 and 0.12.0.1, and CPU ORT 1.31.0 with optimization disabled and enabled,
including reciprocal infinity signs. This is a bounded correction, not full
numerical acceptance of those operations.
BF16 artifact/schema coverage alone makes no CPU numerical acceptance claim.
For subnormal inputs, the reference remains the actual official native eager
and fresh JIT outcomes: JAX documents backend-dependent
[flush-to-zero behavior](https://docs.jax.dev/en/latest/accuracy.html).
Both native roles must agree, and the target must match each role with strict
shape, dtype, NaN-mask and zero-sign checks. Nominal mathematical subnormal-word
observations are separate diagnostics, not authority to rewrite native values
or evidence that gradual underflow was preserved.

The retained full-feed default-target comparison records numerical **FAILED**
on both tested framework variants: CPU ORT 1.31.0 preserves subnormal words where
the official eager and fresh-JIT references return signed zero. Successful
diagnostic tests are not numerical acceptance. The exporter does not
unconditionally flush inputs or replace the native reference to hide this
difference. ORT's documented
[`session.set_denormal_as_zero` setting](https://github.com/microsoft/onnxruntime/blob/v1.24.1/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h#L28-L35)
is a separate execution-target setting. With it explicitly enabled, the finite
FP16/FP32/FP64 extrema test matrix passes strict numerical comparisons on both
framework variants, retaining the full feeds, shapes, dtypes, native roles,
zero signs and both ORT optimization modes. This qualifies those cases and
settings only. Compare different settings in fresh processes because the first
session can affect the main/OpenMP thread's floating-point mode; every native
reference is materialized before its worker creates any ORT session.

The existing private FP64 opset21 profile admits `reduce_max` but rejects
`integer_pow(-1)`, including a `jnp.reciprocal` consumer. The original composed
test remains an explicit unimplemented SOURCE obligation with target numerics
UNVERIFIED; a separate max-only21 graph is checked without claiming that the
composition or a general ONNX limitation has been established.

To check a kernel, create a session for a one-node model with that operator and
dtype. For providers other than CPU, set the session option
`session.disable_cpu_ep_fallback` to `1` so the node cannot fall back to CPU. A
`NOT_IMPLEMENTED` error means the kernel is missing.

*For dependency requirements and optional dependency groups, see [`pyproject.toml`](https://github.com/enpasos/jax2onnx/blob/main/pyproject.toml). For resolved Python dependency versions and their markers, see `poetry.lock` and `uv.lock`.*
