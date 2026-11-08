# Dependencies

**Major dependency versions in the lockfiles (updated 2026-11-02):**

| Library       | Version |
|:--------------|:--------|
| [`JAX / JAXLIB`](https://github.com/jax-ml/jax) | 0.10.2 (Python 3.11/3.12); 0.12.0.1 (Python 3.13/3.14) |
| [`Flax`](https://github.com/google/flax) | 0.12.8 (Python 3.11/3.12); 0.12.12 (Python 3.13/3.14) |
| [`Equinox`](https://github.com/patrick-kidger/equinox) | 0.13.8 |
| [`onnx-ir`](https://github.com/onnx/ir-py) | 1.1.0 |
| [`onnx`](https://github.com/onnx/onnx) | 1.23.2 |
| [`onnxruntime`](https://github.com/microsoft/onnxruntime) | 1.31.1 |
| [`onnxruntime-web`](https://www.npmjs.com/package/onnxruntime-web) | 1.31.1 |

Python package versions above come from `poetry.lock`; the Web runtime version
comes from `package-lock.json`.
The supported ONNX IR minimum is **1.0.0** (`onnx-ir>=1.0.0`).
`onnxruntime-web` 1.31.1 is locked in `package-lock.json` for the
Node.js and Chromium WASM validation flows. The supported Python runtime
minimum remains ONNX Runtime 1.24.1; exports do not select operators based on
the runtime installed on the export machine.

## ONNX Runtime kernel gaps

Some lowerings depend on kernels or behavior that ONNX Runtime lacks. Because
exports must run on every supported version, a workaround can be removed only
once the supported minimum includes the fix; a fix in the locked release alone
is not enough. Re-check this table whenever the locked ONNX Runtime or the
supported minimum changes, and update the versions it was checked against.
`minimum-dependencies` CI is the minimum-version gate; local locked-version
measurements do not substitute for it.

| Gap | Checked | Affected exports | Current handling |
|:--|:--|:--|:--|
| No bfloat16 `Log` on CPU or CUDA; no bfloat16 `Exp` on CPU | CPU 1.24.1, 1.31.0; CUDA 1.31.0 | bfloat16 `log1p`, `expm1`, `sinh`, `jax.nn.log1mexp`, `jax.nn.log_sigmoid` | Computed in float32 and rounded back. Float32 is also the more accurate choice (within 1 ulp), so a new kernel alone is no reason to switch. |
| No float64 or bfloat16 `Sinh` on CPU; no `Sinh` on CUDA | CPU 1.24.1, 1.31.0; CUDA 1.31.0 | `jnp.sinh`; `lax.sinh` other than float32 | expm1-based form. Float32 `lax.sinh` keeps `Sinh`, which runs on CPU in CUDA sessions. |
| No float64 `Cos` on CPU | Missing in 1.24.1, present in 1.31.0 | float64 `lax.cos`, `jnp.cos` | Half-angle form `1 - 2 * Sin(x / 2)²` |
| No DOUBLE `Atan` on CPU | CPU 1.24.1, 1.31.0 | FP64 `atan2` | Two half-angle reductions and a DOUBLE polynomial. Six subnormal input pairs remain numerically unqualified; see Known Limitations. |
| No DOUBLE `Conv` on CPU | CPU 1.24.1, 1.31.0 | FP64 convolutions and complex128 components | Unit kernels/strides/dilations with zero padding use grouped canonical DOUBLE `Einsum`. Other DOUBLE `Conv` artifacts retain source types but fail to load on these CPU runtimes; numerics remain unqualified. |
| Transpose–MatMul fusion creates unsupported DOUBLE `com.microsoft.FusedMatMul` | CPU 1.24.1, optimizations enabled | DOUBLE pointwise convolutions | Canonical channel-first `Einsum` avoids the transpose–MatMul pattern while retaining DOUBLE work. |
| No float64 `Erf` on CPU | CPU 1.24.1, 1.31.0 | float64 exact `jax.nn.gelu`, `nnx.gelu` | None: the export uses `Erf` (or `Gelu`, which expands to it) and fails to load; float64 test variants are skipped. |
| CPU `Where` returns `+0.0` when it selects `-0.0` | CPU 1.24.1, 1.31.0 | `log1p`, `expm1`, `sinh`, floating-point sort keys | `log1p` and `sinh` end in `Mul` to keep the sign; `expm1(-0.0)` exports as `+0.0`. Sorting uses sanitized keys only to compute a permutation, then gathers original values to preserve their bits. |
| `MatMulInteger` with a zero contraction length can leave output values uninitialized | CPU 1.31.0 | INT8/UINT8 `lax.dot_general` with INT32 accumulation | A known empty contraction produces explicit INT32 zeros. Potentially empty symbolic contractions pad both operands with one zero product; this may copy the operands. Fixed positive contractions retain direct integer multiplication. |
| UINT8 × INT8 `MatMulInteger` can saturate intermediate pair sums on some x86 CPUs | CPU 1.31.0 on GitHub runners | UINT8-left/INT8-right `lax.dot_general` with INT32 accumulation | Re-encode the right operand as UINT8 by adding 128 in INT32, then use a UINT8 right zero point of 128. This keeps the full source range and exact integer accumulation. |
| BF16 `Clip` and `Where` kernels are unavailable | CPU 1.31.0 | BF16 conversion to INT8/UINT8 | Widen exactly to FP32 for NaN handling and saturation, then cast to the requested integer type. FP16, FP32 and FP64 retain their actual computation type. Minimum-runtime qualification is separate. |
| Half-kernel adaptations can remove FP16 rounding boundaries | CPU 1.24.1, 1.31.0, 1.31.1 | FP16 `Add`, `Sub`, `Mul`, `rsqrt` | Local FLOAT work with an explicit FP16 result cast. `rsqrt` keeps both `Sqrt` and `Div` in FLOAT before its single half boundary. |
| Integer `ReduceSum` can saturate instead of wrapping | CPU 1.24.1, 1.31.0, 1.31.1 | Integer reductions and window sums | Integer `CumSum` and index selection preserve exact source-width wrapping, including values above `2**53`; no floating-point intermediate. |
| Optimizing some empty-tensor graphs fails an internal `RemoveNode` invariant during session creation | CPU 1.24.1; the same metadata fixtures pass with default optimization on 1.31.1 | Empty captured arrays with reshape/arithmetic, including nested Functions | Metadata contract tests retain full ONNX checking and exact storage checks, and run empty-input inference with `ORT_DISABLE_ALL`. Other cases retain default runtime optimization. This is an external runtime optimizer limitation, not an exception to metadata preservation. |

Focused FP16 `Add`/`Sub`/`Mul` and integer reduction/window checks also pass on
the minimum stack: JAX 0.8.1, Flax 0.12.1 and CPU ONNX Runtime 1.24.1. This is
not full minimum-version acceptance. The narrowly accepted `rsqrt(65504)`
boundary remains numerically failed; ONNX 1.20.1 lacks the opset-26 `BitCast`
schema. See
[Known Limitations](../user_guide/known_limitations.md#fp16-arithmetic-boundaries)
for the failed `rsqrt` boundary and separate JAX CPU compilation limits.

ONNX Runtime documents the [UINT8 × INT8 saturation mechanism and UINT8 × UINT8 workaround](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#when-and-why-do-i-need-to-try-u8u8).
The affected AVX2/AVX512 kernels use saturating INT16 intermediate sums;
VNNI and Arm paths do not have that issue. A passing local runtime test alone
therefore does not establish correctness on the other CPU paths.

An ONNX `Cast` alone also differs from the observed JAX FP16/BF16/FP32/FP64
conversion to INT8/UINT8 at overflow, infinity and NaN boundaries. Those
conversions map NaN to zero and clamp to the destination range before casting.
Integer-to-integer conversions keep their existing wrapping behaviour.

To check a kernel, create a session for a one-node model with that operator and
dtype. For providers other than CPU, set the session option
`session.disable_cpu_ep_fallback` to `1` so the node cannot fall back to CPU. A
`NOT_IMPLEMENTED` error means the kernel is missing.

*For dependency requirements and optional dependency groups, see [`pyproject.toml`](https://github.com/enpasos/jax2onnx/blob/main/pyproject.toml). For resolved Python dependency versions, see `poetry.lock`.*
