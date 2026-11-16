# Dependencies

**Major dependency versions in the lockfiles (updated 2026-11-16):**

| Library       | Version |
|:--------------|:--------|
| [`JAX / JAXLIB`](https://github.com/jax-ml/jax) | 0.10.2 (Python 3.11/3.12); 0.12.0.1 (Python 3.13/3.14) |
| [`Flax`](https://github.com/google/flax) | 0.12.8 (Python 3.11/3.12); 0.12.12 (Python 3.13/3.14) |
| [`Equinox`](https://github.com/patrick-kidger/equinox) | 0.13.8 |
| [`onnx-ir`](https://github.com/onnx/ir-py) | 1.1.0 |
| [`onnx`](https://github.com/onnx/onnx) | 1.23.2 |
| [`onnxruntime`](https://github.com/microsoft/onnxruntime) | 1.32.0 |
| [`onnxruntime-web`](https://www.npmjs.com/package/onnxruntime-web) | 1.31.1 |

Python versions come from `poetry.lock`. The Web runtime comes from
`package-lock.json`, where it is locked for the Node.js and Chromium WASM
validation flows. The supported minimums are `onnx-ir` 1.0.0 and ONNX Runtime
1.24.1. Exports do not select operators by the runtime installed on the export
machine.

## ONNX Runtime kernel gaps

Some lowerings work around kernels or behaviour that ONNX Runtime lacks. Exports
must run on every supported version, so a workaround can go only when the
supported minimum includes the fix. Re-check this table whenever the locked or
the minimum ONNX Runtime changes, and update the versions. `minimum-dependencies`
CI is the minimum-version gate; a local run on the locked version does not
replace it.

| Gap | Checked (CPU unless noted) | Affected exports | Handling |
|:--|:--|:--|:--|
| No bfloat16 `Log` on CPU or CUDA; no bfloat16 `Exp` on CPU | 1.24.1, 1.31.0, 1.32.0; CUDA 1.31.0 | bfloat16 `log1p`, `expm1`, `sinh`, `jax.nn.log1mexp`, `jax.nn.log_sigmoid` | Float32 work, rounded back. Float32 is also more accurate (within 1 ulp), so a new kernel alone is no reason to switch. |
| No float64 or bfloat16 `Sinh` on CPU; no `Sinh` on CUDA | 1.24.1, 1.31.0, 1.32.0; CUDA 1.31.0 | `jnp.sinh`; `lax.sinh` other than float32 | expm1-based form. Float32 `lax.sinh` keeps `Sinh`, which runs on CPU in CUDA sessions. |
| No float64 `Cos` | Missing in 1.24.1; present in 1.31.0, 1.32.0 | float64 `lax.cos`, `jnp.cos` | Half-angle form `1 - 2 * Sin(x / 2)²` |
| No float64 `Atan` | 1.24.1, 1.31.0, 1.32.0 | float64 `atan2` | Two half-angle reductions and a float64 polynomial. Six subnormal input pairs remain unqualified; see [Known Limitations](../user_guide/known_limitations.md#fp64-atan2-subnormal-inputs). |
| No float64 `Conv` | 1.24.1, 1.31.0, 1.32.0 | float64 convolutions and complex128 components | Unit kernels, strides and dilations with zero padding use a grouped float64 `Einsum`. Other float64 `Conv` exports keep their types but fail to load; their numerics are unqualified. |
| Transpose–MatMul fusion creates an unsupported float64 `com.microsoft.FusedMatMul` | 1.24.1 with optimizations | float64 pointwise convolutions | A channel-first `Einsum` avoids the pattern and stays in float64. |
| No float64 `Erf` | 1.24.1, 1.31.0, 1.32.0 | float64 exact `jax.nn.gelu`, `nnx.gelu` | None: the export uses `Erf` (or `Gelu`, which expands to it) and fails to load; float64 test variants are skipped. |
| `Where` returns `+0.0` when it selects `-0.0` | 1.24.1, 1.31.0 | `log1p`, `expm1`, `sinh`, floating-point sort keys | `log1p` and `sinh` end in `Mul` to keep the sign; `expm1(-0.0)` exports as `+0.0`. Sorting computes the permutation from sanitized keys and gathers the original values. |
| `MatMulInteger` with an empty contraction can leave outputs uninitialized | 1.31.0 | INT8/UINT8 `lax.dot_general` with INT32 accumulation | A known empty contraction emits INT32 zeros. A possibly empty symbolic one pads both operands with one zero product, which may copy them. |
| UINT8 × INT8 `MatMulInteger` can saturate pair sums on some x86 CPUs | 1.31.0 on GitHub runners | UINT8-left, INT8-right `lax.dot_general` with INT32 accumulation | The right operand is re-encoded as UINT8 by adding 128 in INT32, with a right zero point of 128. This keeps the full range and exact accumulation. |
| No bfloat16 `Clip` or `Where` | 1.31.0, 1.32.0 | bfloat16 conversion to INT8/UINT8 | Widened exactly to float32 for NaN handling and saturation, then cast. Other float types keep their computation type. Not qualified on the minimum runtime. |
| Half-kernel adaptations can remove float16 rounding boundaries | 1.24.1, 1.31.0, 1.31.1 | float16 `Add`, `Sub`, `Mul`, `rsqrt` | Float32 work with an explicit float16 result cast. `rsqrt` keeps `Sqrt` and `Div` in float32 before its single cast. |
| No kernels for the opset-22 `RandomUniform`, `RandomUniformLike`, `RandomNormal`, `RandomNormalLike`, `Bernoulli` and `Multinomial` | 1.31.1, 1.32.0 | Lowerings that draw random numbers, such as `jax.random.normal`, `jax.random.bernoulli`, `jax.random.bits` and `lax.rng_uniform`, from opset 22, which includes the default 23 | None: the file is valid but fails to load with `NOT_IMPLEMENTED`. Export with `opset=21`. |
| Integer `ReduceSum` can saturate instead of wrapping | 1.24.1, 1.31.0, 1.31.1 | Integer reductions and window sums | Integer `CumSum` and index selection wrap exactly at the source width, also above `2**53`. |
| A `CastLike` that widens a float16 constant aborts session creation with `ORT_ENABLE_ALL` | Aborts on 1.24.1, 1.27.0, 1.28.0, 1.29.0; loads on 1.30.0, 1.31.1, 1.32.0 | float16 parameters used in float32 work, such as the scale and bias of the explicit Flax NNX and Linen LayerNorm | The export never writes `CastLike`. It uses `Cast`, which the runtime folds on a constant, and a policy test rejects `CastLike`. See [below](#the-float16-initializer-optimizer-before-onnx-runtime-130). |
| An overridable float16 initializer that is widened to float32 crashes session creation with `ORT_ENABLE_ALL` | Crashes on 1.24.1, 1.29.0; loads on 1.30.0, 1.32.0 | float16 state from `to_onnx(..., state=...)`: matrix product weights on 1.24.1, LayerNorm scale and bias on 1.24.1 and 1.29.0 | None. See [FP16 State before ONNX Runtime 1.30](../user_guide/known_limitations.md#fp16-state-before-onnx-runtime-130) for the ways to load such a file. |

Notes on the table:

- **Minimum stack.** JAX 0.8.1, Flax 0.12.1 and ONNX Runtime 1.24.1 also pass
  the focused float16 `Add`/`Sub`/`Mul` and integer reduction checks. That is
  not full minimum-version acceptance: `rsqrt(65504)` remains numerically
  failed (see
  [Known Limitations](../user_guide/known_limitations.md#fp16-arithmetic-boundaries)),
  and ONNX 1.20.1 has no opset-26 `BitCast`.
- **UINT8 × INT8.** ONNX Runtime documents the
  [saturation and the UINT8 × UINT8 workaround](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#when-and-why-do-i-need-to-try-u8u8).
  The AVX2 and AVX512 kernels use saturating INT16 sums; VNNI and Arm paths do
  not. A passing local test therefore says nothing about the other CPU paths.
- **Conversion to INT8/UINT8.** An ONNX `Cast` alone differs from JAX for
  floating inputs at overflow, infinity and NaN: JAX maps NaN to zero and
  clamps to the destination range. Integer-to-integer conversions wrap.
- **Checking a kernel.** Create a session for a one-node model with the
  operator and dtype; `NOT_IMPLEMENTED` means the kernel is missing. For other
  providers set `session.disable_cpu_ep_fallback` to `1`, so that the node
  cannot fall back to CPU.

### The float16 initializer optimizer before ONNX Runtime 1.30

With the default `ORT_ENABLE_ALL`, `FuseFp16InitializerToFp32NodeTransformer`
replaces a `Cast` from a float16 initializer to float32 by a float32
initializer. Before 1.30.0 two defects end the process instead of raising:

- It looks up the consumer by the node name of the `Cast`, which only matches
  the casts the runtime inserted itself. A `CastLike` on a float16 constant is
  expanded to a `Cast` after constant folding and fails that lookup; the
  official wheels then abort on a libstdc++ assertion. The export never writes
  `CastLike`, and a `Cast` on a constant is folded before the optimizer runs.
- It accepts an overridable initializer, which has no constant to convert, and
  crashes with a segmentation fault. A cast on such an initializer is never
  folded, whether the runtime inserted it or the export wrote it for float32
  work, so the export does not avoid this.

Measured with the PyPI wheels on Linux x86-64. Both cases load with
`ORT_ENABLE_EXTENDED`. `tests/extra_tests/test_float16_cpu_inference.py` checks
them in the `minimum-dependencies` job, in a separate process, because a crash
ends the process with a signal instead of a Python exception.

## bfloat16 and the ONNX operator schemas

ONNX admitted bfloat16 one operator at a time. A node that uses bfloat16 where
the schema of the target opset excludes it makes the model invalid:
`onnx.checker.check_model(..., full_check=True)` and ONNX Runtime reject it.
This depends on the schemas, not on a runtime, so the export decides by target
opset alone.

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
(`converter/bfloat16_legalization.py`); from that opset on, the operator stays
in bfloat16.

- **Computed in float32.** Where the inputs and outputs share the excluded type
  parameter, the inputs are cast to float32 and the results are cast back, also
  in control-flow and Function bodies. Graph inputs and outputs, initializers,
  state inputs and Function signatures keep their bfloat16 type and payload.
  The widening is exact, and the result is rounded once at the operator.
- **Rejected.** Casts cannot fix an operator whose output type comes from an
  attribute (`RandomNormal`, `RandomUniform`, `EyeLike`, `ConstantOfShape`
  before opset 20), bfloat16 values through `If`, `Loop` or `Scan` before opset
  16, or sequence operators. The export raises `NotImplementedError` with the
  operator and the first opset that admits it.

The rules are a table in that module, because the converter does not import
`onnx`. `scripts/emit_bfloat16_operator_rules.py` derives it from the ONNX
schemas, and a policy test compares it with the installed ONNX. Regenerate it
when that test fails after an ONNX upgrade.

Numerics, checked with JAX 0.12.0.1 and CPU ONNX Runtime 1.31.1:

- `Conv`, a two-operand `Einsum`, `TopK`, `MaxPool` and `Sin` computed in
  float32 reproduce JAX bit for bit.
- `Softplus`, `Selu` and `HardSwish` replace several JAX primitives and round
  once instead of after each of them. They differ from JAX by up to 3 adjacent
  bfloat16 values and stay within half an ulp of a float64 reference, where JAX
  reaches 1.2 to 2.7 ulp.
- Where JAX rounds intermediate results that the export accumulates first, the
  results can differ by hundreds of ulp when terms cancel. This applies to a
  `jnp.einsum` with three operands, which exports as one `Einsum`, and to
  window sums of `lax.reduce_window` with `add` at every opset.

A valid bfloat16 model does not have to load. Among the operators checked, CPU
ONNX Runtime 1.32.0 has bfloat16 kernels for `Cast`, `Identity`, `Reshape`,
`Transpose`, `Concat`, `Gather`, `Slice`, `Squeeze` and `Unsqueeze`. It has
none for `Add`, `Sub`, `Mul`, `Div`, `MatMul`, `Gemm`, `Conv`, `MaxPool`,
`AveragePool`, `ReduceSum`, `ReduceMean`, `Sqrt`, `Exp`, `Relu`, `Sigmoid`,
`Tanh`, `Erf`, `Gelu`, `Softmax`, `Where`, `Expand` or `TopK`. A node computed
in float32 loads; bfloat16 arithmetic around it does not.

*For dependency requirements and optional dependency groups, see [`pyproject.toml`](https://github.com/enpasos/jax2onnx/blob/main/pyproject.toml). For resolved Python dependency versions, see `poetry.lock`.*
