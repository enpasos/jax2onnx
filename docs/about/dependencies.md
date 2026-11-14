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

## bfloat16 and the ONNX operator schemas

ONNX admitted bfloat16 one operator at a time. A node that reads or writes
bfloat16 through a type parameter that excludes it at the target opset makes
the model invalid: `onnx.checker.check_model(..., full_check=True)` rejects it,
and so does ONNX Runtime when it loads the file. This is a property of the ONNX
schemas rather than of a runtime, so the export decides by target opset alone.

| bfloat16 from opset | Operators (selection) |
|:--|:--|
| 16 | `Where`, `LessOrEqual`, `GreaterOrEqual`, `LeakyRelu`, `PRelu`; values through `If`, `Loop`, `Scan` |
| 20 | `ConstantOfShape`, `IsInf` |
| 22 | `Conv`, `ConvTranspose`, `AveragePool`, `MaxPool`, `GlobalAveragePool`, `GlobalMaxPool`, `InstanceNormalization`, `Sin`, `Cos`, `Tan`, `Asin`, `Acos`, `Atan`, `Sinh`, `Cosh`, `Asinh`, `Acosh`, `Atanh`, `Elu`, `Selu`, `HardSigmoid`, `HardSwish`, `Mish`, `Softplus`, `Softsign`, `Round`, `EyeLike`, `RandomNormal`, `RandomUniform` |
| 24 | `TopK` |
| 27 | `Range` |
| 28 | `Einsum`, `Celu`, `Compress`, `Unique` |
| not through opset 28 | `Shrink` |

Below that opset the converter handles the node after optimization
(`converter/bfloat16_legalization.py`):

- **Computed in float32.** Where the excluded type parameter is shared by the
  inputs and outputs of the operator, those inputs are cast to float32 and the
  results are cast back. Graph inputs and outputs, initializers, state inputs
  and Function signatures keep their bfloat16 type and payload; control-flow
  and Function bodies are covered. Every bfloat16 value is a float32 value, so
  the widening is exact and the result is rounded once at the operator.
- **Rejected.** Casts cannot fix an operator whose output type comes from an
  attribute (`RandomNormal`, `RandomUniform`, `EyeLike`, `ConstantOfShape`
  before opset 20), bfloat16 values carried through `If`, `Loop` or `Scan`
  before opset 16, or sequence operators. The export raises
  `NotImplementedError` and names the operator and the first opset that admits
  it.

From the admitting opset on, the operator stays in bfloat16.

The rules are a table in that module because the converter does not import
`onnx`. `scripts/emit_bfloat16_operator_rules.py` derives it from the ONNX
schemas, and a policy test compares it with the installed ONNX for every
operator and opset from 13. Regenerate the table when that test fails after an
ONNX upgrade.

Checked with JAX 0.12.0.1 and CPU ONNX Runtime 1.31.1: a convolution (`Conv`),
a two-operand `Einsum`, `TopK`, `MaxPool` and `Sin` computed in float32
reproduce JAX bit for bit.
`Softplus`, `Selu` and `HardSwish` each replace several JAX primitives and
round once instead of after every primitive; they differ from JAX by up to 3
adjacent bfloat16 values and stay within half a bfloat16 ulp of a float64
reference, where JAX reaches 1.2 to 2.7 ulp. A `jnp.einsum` with three operands
exports as one `Einsum`; JAX rounds the intermediate product to bfloat16, which
can cost hundreds of ulp where terms cancel, so the two results can differ by
that much. The same holds for window sums at every opset: JAX rounds each
partial sum of `lax.reduce_window` with `add` to bfloat16, while the exported
`Conv` or `AveragePool` accumulates the window before rounding.

A valid bfloat16 model is not necessarily a loadable one. Among the operators
checked, CPU ONNX Runtime 1.31.1 has bfloat16 kernels for `Cast` and for
`Identity`, `Reshape`, `Transpose`, `Concat`, `Gather`, `Slice`, `Squeeze` and
`Unsqueeze`. It has none for `Add`, `Sub`, `Mul`, `Div`, `MatMul`, `Gemm`, `Conv`, `MaxPool`,
`AveragePool`, `ReduceSum`, `ReduceMean`, `Sqrt`, `Exp`, `Relu`, `Sigmoid`,
`Tanh`, `Erf`, `Gelu`, `Softmax`, `Where`, `Expand` or `TopK`. A node computed
in float32 loads; bfloat16 arithmetic around it does not.

*For dependency requirements and optional dependency groups, see [`pyproject.toml`](https://github.com/enpasos/jax2onnx/blob/main/pyproject.toml). For resolved Python dependency versions, see `poetry.lock`.*
