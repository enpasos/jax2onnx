# Known Limitations

This page summarizes the main support boundaries for `jax2onnx`.

`jax2onnx` is an export tool for JAX-derived callables and model code. It is
primarily intended to produce ONNX inference artifacts.

## Unsupported Primitives

`jax2onnx` lowers traced JAXPR primitives through registered plugins.

If a traced callable uses a primitive without a registered lowering, conversion
fails with an explicit error. In many cases, support can be added through the
plugin system.

See [Plugin System](../developer_guide/plugin_system.md) for extension details.

## Dynamic Shapes

Symbolic dimensions such as `"B"` are supported for common dynamic-batch export
patterns.

Not every JAX shape-polymorphic expression can necessarily be represented
directly in ONNX. For validation and debugging, prefer starting with concrete
input shapes and then introducing symbolic dimensions where needed.

## Inference Behavior

The exported ONNX model represents the traced callable behavior.

For modules with dropout, batch normalization, mutable state, or RNG-dependent
behavior, make the intended inference behavior explicit before export. Pass
runtime flags as explicit inputs only when those flags should remain part of the
ONNX model interface.

## Dtypes and Model Preparation

Export follows the prepared JAX computation. Shape-only input specs default to
FP32 even when x64 is enabled. For FP64, supply typed inputs and configure the
model in an effective JAX x64 context; use that same context for validation.
Explicit FP32 casts, constants, and framework dtype settings keep their meaning.
Widening stored FP32 weights does not recover precision lost when they were
created. See [Dtypes and Precision](api.md#dtypes-and-precision) for the API and
the migration from the removed `enable_double_precision` argument.

For unary `lax` math lowerings that accept an `accuracy` argument, `None` and
`AccuracyMode.DEFAULT` are supported. `AccuracyMode.HIGHEST` and explicit
`Tolerance` requests currently raise `NotImplementedError`; export does not yet
preserve their requested accuracy.

## Runtime Compatibility

### BF16 Runtime I/O

ONNX Runtime's ordinary NumPy `session.run` interface rejects BF16 arrays.
The public `allclose` helper uses that interface, so it cannot validate a
BF16 tensor input or output through this path. Export support and runtime
kernel availability do not remove this binding limitation.

For manual BF16 validation, pass raw `uint16` bits through
`OrtValue.ortvalue_from_numpy_with_onnx_type` and I/O binding, then compare the
decoded output with the prepared JAX computation. See the
[working raw-bit validation example](https://github.com/enpasos/jax2onnx/blob/main/tests/extra_tests/test_log1p_expm1_precision.py).
Validate the provider and optimization settings used for deployment.

### BF16 Operators by Opset

ONNX admitted BF16 one operator at a time: `Conv`, the pooling operators and
most activations from opset 22, `TopK` from opset 24, `Einsum` from opset 28.
Below that opset the export computes the operator in FP32 between BF16 values.
Inputs, outputs, constants and state inputs keep their BF16 type, and the file
passes `onnx.checker.check_model(..., full_check=True)`. From the admitting
opset on, the operator stays in BF16.

Some cases cannot be fixed by casts and raise `NotImplementedError` with the
required opset: BF16 `jax.random.normal` and `jnp.eye` before opset 22, and BF16
values through `lax.cond`, `lax.scan` or `lax.while_loop` before opset 16.
Raise the target opset, or produce the value in FP32 and convert it to BF16.

A valid BF16 file does not have to load. CPU ONNX Runtime 1.31.1 has no BF16
kernels for arithmetic such as `Add`, `Mul`, `MatMul` or `Conv`, so a BF16
model usually fails to load there with `NOT_IMPLEMENTED`; an operator computed
in FP32 loads. See
[bfloat16 and the ONNX operator schemas](../about/dependencies.md#bfloat16-and-the-onnx-operator-schemas)
for the operator table, the checked numerical behaviour and the kernels.

### Random Operators from Opset 22

CPU ONNX Runtime 1.31.1 has no kernels for the opset-22 versions of
`RandomUniform`, `RandomUniformLike`, `RandomNormal`, `RandomNormalLike`,
`Bernoulli` and `Multinomial`. Exports that draw random numbers, for example
through `jax.random.normal`, `jax.random.bernoulli`, `jax.random.bits` or
`lax.rng_uniform`, are valid at the default opset 23 but fail to load there with
`NOT_IMPLEMENTED`. Export them with `opset=21` to run on that runtime.

`jax.random.uniform` additionally requires opset 26 for every dtype, because
JAX builds it with `lax.bitcast_convert_type`, which lowers to `BitCast`.

### Quantized Rounding Boundaries

Small floating-point differences before integer rounding can change quantized
values, even when the integer matrix product itself is exact. Dynamic
activation quantization can therefore amplify a division difference between
JAX and ONNX Runtime. A numerical check must cover the prepared quantization
scheme as well as its integer contractions. The
[trained W8A8 example](quantized_mnist.md) uses calibrated power-of-two
activation scales and validates its complete held-out set; it does not
establish support for arbitrary Qwix recipes.

### Execution Targets

ONNX Runtime compatibility depends on:

- the operators emitted by the export,
- the target opset,
- the ONNX Runtime version,
- the execution provider,
- whether the model is intended for Python, browser/WASM, or another deployment target.

For browser/WASM deployment, use `export_mode="web"` and the Web validation
workflow.

Bit reinterpretation with `lax.bitcast_convert_type` requires opset 26, a
validator that contains the `BitCast` schema, and a runtime kernel. The minimum
ONNX package, 1.20.1, contains schemas only through opset 25; these opset-26
workflows therefore require a newer ONNX environment.

Reshapes that place a literal zero at an axis without a corresponding known
zero input dimension require opset 14 or newer. Earlier opsets interpret zero
as copying the input dimension at the same axis; these incompatible literal
zero placements are rejected. Aligned zero dimensions and explicit `-1` entries
retained in the emitted shape tensor keep ONNX's copy/inference semantics. If
JAX resolves an inferred size to zero before lowering, the same literal-zero
restriction applies. Use opset 14 or newer when symbolic reshape dimensions
may become zero at runtime.

## Numerical Differences

Small numerical differences can occur across JAX and ONNX Runtime because of
implementation details, dtype handling, precision settings, or runtime kernels.

Use `allclose(...)` with tolerances appropriate for the model and dtype. For
deployment checks, validate representative inputs rather than only zero-valued
inputs.

Normalization is especially sensitive when a group has zero or near-zero
variance. JAX and ONNX runtimes can produce different floating-point residuals
while implementing the same normalization formula, and later layers may amplify
that roundoff. Use strict parity checks on representative, nonconstant inputs;
for degenerate normalization inputs, also check finiteness and apply a tolerance
specific to the model, dtype, and runtime.

### FP16 arithmetic boundaries

On the checked ONNX Runtime CPU versions, implicit half-kernel adaptations can
remove intermediate FP16 rounding. FP16 `Add`, `Sub`, and `Mul` therefore use
local FLOAT arithmetic and an explicit FP16 result cast. FP16 `rsqrt` computes
`1 / Sqrt(x)` in FLOAT and casts once at the primitive's result boundary; its
intermediate square root is not rounded to half. Inputs and shared constants
retain their source types. This does not establish universal accuracy for all
half operations or execution providers. See the measured runtime scope in
[Dependencies](../about/dependencies.md#onnx-runtime-kernel-gaps).

Checks on 2 November 2026 found that CPU JAX 0.8.1 rounds FP16 `rsqrt(65504)`
differently from JAX 0.10.2/0.12.0.1. The lowering matches the locked references
but differs from JAX 0.8.1 by one FP16 step with ONNX Runtime 1.24.1 and 1.31.0.
Numerical parity at that boundary remains **failed**.

On CPU JAX/jaxlib 0.8.1 with ONNX Runtime 1.24.1, native result bits are `7169`
and runtime bits are `7168`. The main baseline and this exporter produce
bit-identical results across the full eight-input regression with runtime
optimizations disabled and enabled. The known failed input remains recorded
separately; other inputs retain exact comparisons. This does not qualify parity
at `65504` or change the supported dependency floors.

### FP16 attention on JAX CPU

For the masked `jax.nn.dot_product_attention` regression with input shape
`(1, 4, 2, 8)`, eager JAX works on both official JAX/jaxlib 0.10.2 and 0.12.0.1.
Fresh CPU JIT compilation raises `ValueError` with
`The precision 'F16_F16_F32' is not supported by dot_general on CPU`.
No warmed executable is available after that compilation failure.

The original eager-JAX/ONNX Runtime comparison remains at `rtol=atol=2e-3`,
with runtime graph optimizations disabled and enabled. Qualification against
all three native references—eager, freshly compiled, and warmed—remains
**unverified** for this case. The compiler error is recorded separately and is
not a successful numerical comparison.

### Explicit matrix multiplication algorithms

CPU exports support ordinary precision tuples, `None`, and
`DotAlgorithmPreset.DEFAULT`. Explicit `F32_F32_F32` and `F64_F64_F64` are
supported for FP32/FP64 operands and results through local work casts and a
final cast to the source result dtype. Other explicit algorithms or dtype
combinations currently raise `NotImplementedError`.

The checked `(2, 2)` matrix result with contraction length 2 agrees with eager,
freshly compiled, and warmed JAX and both ONNX Runtime optimization settings
on official JAX 0.10.2/0.12.0.1 CPU. A `(1, 1)` result with the same products
and `F64_F64_F64` fails in both official CPU emitters with
`ALG_DOT_F64_F64_F64`. Native execution **failed** and no warmed executable is
available. The exporter can represent the arithmetic; JAX/ONNX numerical parity
for that shape remains **unverified**.

### FP64 atan2 subnormal inputs

The checked JAX 0.10.2/0.12.0.1 CPU references agree across eager, fresh and
warmed execution, but differ from ONNX at six pairs in the full 4,265-input
FP64 `atan2(y, x)` witness. With `s` the smallest positive FP64 subnormal and
`M` the largest finite FP64 value, these are `(s, -M)`, `(s, -1)`,
`(s, s)`, `(s, -s)`, `(-s, s)` and `(-s, -s)`.

These pairs remain numerically **failed** and are outside qualified JAX parity.
The regression retains all inputs, native consensus, FP64 graph checks, NaN/Inf
and signed-zero checks. Every other pair in the witness must pass
`rtol=0, atol=4e-15` with ONNX Runtime optimizations disabled and enabled.
Exact zeros with the sign of `y` are required on the corresponding input axes
and for finite `y` with positive-infinite `x`; zeros shared by native and runtime
results must also retain their signs. The absolute tolerance can permit a
finite residual where native returns zero, so it does not promise identical
zero locations for every finite angle.
The diagnostic records the actual failing subset, including zero after a fix;
it does not emulate the native behavior or qualify numerical parity for a
failing pair. The minimum-JAX FP16 `rsqrt` failure remains a separate boundary.

### Complex calculations

Complex values use packed real and imaginary components: `complex64` retains
FP32 components and `complex128` retains FP64 components. Complex Gather
supports ordinary in-bounds and clipped indexing. Its fill mode currently
requires identical real/imaginary component bits; unequal components, including
the default `NaN + 0j` fill, raise `NotImplementedError`. This is a lowering
limitation, not an ONNX restriction.

`lax.exp` and `jnp.exp` share the complex exponential lowering. Numerical checks
cover the finite RoPE witnesses on the checked CPU stacks, including signed
zero and small FP64 increments; other complex numerical domains and execution
providers remain unqualified.

### Float64 cosine

Float32 cosine uses native ONNX `Cos`. For float64, `jnp.cos` and `lax.cos`
use the equivalent expression `1 - 2 * Sin(x / 2)²`. This supports older
ONNX Runtime CPU versions without a double-precision `Cos` kernel and avoids
losing the phase shift in `Sin(x + π/2)` for large arguments. The half-angle
formula can lose relative accuracy near cosine zeros; its regression checks
use an absolute tolerance of `2e-15` on selected finite float64 inputs, including
large arguments. NaN and infinity retain their NaN cosine results. These checks
cover ONNX Runtime CPU; validate the actual deployment provider separately.

### log1p, expm1, and related functions

ONNX has no `Log1p` or `Expm1` operator. `jnp.log1p`, `lax.log1p`,
`jnp.expm1`, `lax.expm1`, and `jax.nn.log1mexp` use compensated forms of
`Log(1 + x)` and `Exp(x) - 1`; the direct forms return zero or lose most digits
for small `|x|`. Two functions build on them. `jax.nn.log_sigmoid` uses JAX's
`min(x, 0) - log1p(exp(-|x|))`; `Log(Sigmoid(x))` returns `-inf` from `x = -20`
down on ONNX Runtime. `jnp.sinh` uses the expm1-based libm form instead of
`(Exp(x) - Exp(-x)) / 2` for every floating dtype, so it also runs on CUDA,
which has no `Sinh` kernel. `lax.sinh` keeps `Sinh` for float32 and uses the
same form for other floating dtypes, which have no ONNX Runtime float64 or
bfloat16 `Sinh` kernel. The compensated forms compute in the input dtype with
standard operators and rely on the runtime's `Log` being accurate near one.
bfloat16 is the exception: it is computed in float32 and rounded back, because
ONNX Runtime has no bfloat16 `Log` kernel; the model keeps its bfloat16 inputs
and outputs. Float16 and bfloat16 models keep their declared dtype with caller
x64 disabled or enabled. Their regression checks compare against a float64
reference: at most 8 ulp for float16, float32, and float64 sweeps with and
without graph optimizations and for every float16 input, and at most 1 ulp for
every bfloat16 input. They also cover subnormal inputs, NaN, infinities, the
`log1p` domain boundary at `-1`, and `sinh` up to its overflow threshold.
ONNX Runtime's CPU `Where` returns
`+0.0` when it selects `-0.0`, so exported `expm1(-0.0)` is `+0.0`;
`log1p(-0.0)` and `sinh(-0.0)` keep their sign. These checks cover ONNX Runtime
CPU; validate the actual deployment provider separately.

### Normalization export policy

`normalization_mode="auto"` follows a two-step policy. First, choose the
implementation with the best reproducible accuracy in a defined test
environment and lock its acceptance bounds. Then prefer a native ONNX operator
if it can replace that choice while satisfying the same bounds. Export applies
the plugin's current selection; it does not benchmark implementations for each
model. The current selection is an explicit graph for Flax NNX and Linen
GroupNorm, and Equinox, Flax NNX, and Linen RMSNorm and LayerNorm. **Fixed
comparative accuracy bounds currently cover only the selected LayerNorm cases
below.** Equivalent GroupNorm and RMSNorm bounds have yet to be established,
so the LayerNorm results must not be read as guarantees for those operators or
for other models, inputs, or hardware.

`normalization_mode="prefer_native"` requests a standard ONNX operator when
the plugin can map the operation, the builder supports the operator, and the
selected opset defines it: `LayerNormalization` from opset 17, fast-variance
`GroupNormalization` from opset 21, and `RMSNormalization` from opset 23. The
plugin falls back to its explicit graph when these conditions do not hold.
This is a representation choice: export does not measure numerical error or
fall back based on the `"auto"` accuracy limits. The selected native LayerNorm
cases have their own, looser acceptance limits in the table below. Masks,
distributed axis settings, unsupported reduction layouts, and some dtype
configurations can bypass the specialized module plugin and trace framework
operations instead.

Slow-variance GroupNorm remains explicit in every mode; native GroupNorm also
requires a supported floating dtype and a shape without statically known empty
dimensions. For symbolic dimensions that may become zero at runtime, select
`normalization_mode="force_decomposed"`. That mode chooses the explicit
primitive graph at export. Native operators can produce smaller graphs and may
be accelerated by a runtime, but they can also produce larger numerical
differences on high-offset or otherwise ill-conditioned inputs.

This choice controls the **exported graph**. A runtime may subsequently
optimize or fuse that graph. The explicit LayerNorm and RMSNorm paths square
with `Mul` instead of `Pow`; targeted tests check that ONNX Runtime does not
re-fuse these patterns into its native normalization kernels with
`ORT_ENABLE_ALL`. Other runtimes may rewrite the graphs differently. In the
tested CPU environment, the native `LayerNormalization` path has larger errors
on the selected rows with large outlier activations than the explicit graph.
GPU measurements are supplementary to the CPU acceptance gate. ONNX
`GroupNormalization` does not model Flax's negative-variance clamp or
reduction order, and deep models may amplify normalization roundoff. Validate
the chosen representation on representative inputs and the actual deployment
runtime.

ONNX Runtime 1.30 adds AVX2 kernels for native float32 LayerNorm and RMSNorm.
The LayerNorm kernel uses two-pass statistics on eligible x86 CPUs; native
accuracy therefore depends on the CPU, runtime version, and normalized width.
The selected native LayerNorm cases still exceed the explicit graph's stricter
acceptance bounds below. To measure the model-specific latency and JAX parity
tradeoff, use the [decoder benchmark](validation.md#decoder-normalization-benchmark).

The specialized explicit LayerNorm uses a two-pass variance for Equinox and
Flax NNX with `use_fast_variance=False`. Its constant-row safeguard preserves
exact centered zeros for constant rows when the computed mean is finite; the
final affine bias still applies. Fast-variance Flax NNX and Linen LayerNorm
instead follow the clamped `E[x²] - E[x]²` formula, which is sensitive to
large offsets. Flax Linen LayerNorm with `use_fast_variance=False` traces the
original JAX computation in every mode and does not use that constant-row
safeguard. It is outside the fixed comparative limits below. The specialized
explicit LayerNorm computes float16 and bfloat16 statistics in float32. For
float32 inputs, only the slow-variance reduction accumulates its squared centered
values in float64 and rounds the variance back to float32; the mean, squares,
epsilon, normalization, and affine math stay float32. This local work-dtype
choice is LayerNorm-specific and improves accuracy portability across runtime
reduction orders without guaranteeing identical results. The fixed comparisons
below cover float32 inputs;
validate other configurations for the model's inputs.

### Scope of the fixed LayerNorm limits

The [LayerNorm accuracy
test](https://github.com/enpasos/jax2onnx/blob/main/tests/extra_tests/test_layer_norm_precision.py)
exports at opset 23 and runs the same float32 inputs through ONNX Runtime's
`CPUExecutionProvider` with `ORT_ENABLE_ALL` and `ORT_DISABLE_ALL`. It uses
257 rows of 384 float32 features from `np.random.default_rng(0)`: standard
normal samples with channel 7 shifted by +1700, channel 123 by -900, and
channel 300 by +300 or -300 using the same generator. Epsilon is `1e-5`, with
the tested modules' default unit scale and zero bias. JAX x64 is disabled for
the tested functions; the independent reference uses float64. The
variants are Equinox LayerNorm, Flax NNX LayerNorm with slow or default fast
variance, and Flax Linen LayerNorm with default fast variance. For each case,
the test computes the **maximum absolute error over all outputs** against two
separate comparators: an independent float64 two-pass LayerNorm reference and
the corresponding JAX output. JAX parity therefore does not stand in for error
against the float64 reference.

The following values are **fixed acceptance limits**, not errors measured by a
particular CI run. Both limits in each row must hold for both optimization
settings:

| Export mode | LayerNorm variant | Maximum error vs. float64 reference | Maximum error vs. JAX |
| --- | --- | ---: | ---: |
| `auto`, `force_decomposed` | Equinox, Flax NNX slow variance | 4.1e-6 | 3.9e-6 |
| `auto`, `force_decomposed` | Flax NNX and Linen fast variance | 4.1e-6 | 5.8e-6 |
| `prefer_native` | Equinox, Flax NNX slow variance | 1.2e-5 | 1.2e-5 |
| `prefer_native` | Flax NNX and Linen fast variance | 1.2e-5 | 1.4e-5 |

The [GroupNorm stability
tests](https://github.com/enpasos/jax2onnx/blob/main/tests/extra_tests/test_group_norm_stability.py)
and [RMSNorm policy
tests](https://github.com/enpasos/jax2onnx/blob/main/tests/extra_tests/test_normalization_export_policy.py)
use case-specific JAX parity tolerances: for example, representative float32
GroupNorm cases use `rtol=atol=1e-5`; selected Equinox and NNX float32 RMSNorm
cases use `rtol=atol=5e-5`; and the selected float16 RMSNorm cases use
`rtol=atol=2e-3`. These checks do not compare against an independent float64
reference and do not establish fixed comparative bounds for GroupNorm or
RMSNorm.

The limits were set from measurements with ONNX Runtime 1.29 on an AMD Ryzen 9
9950X3D CPU using JAX 0.10.2 and 0.11.1, then rounded up to two significant
digits. The current lockfile's CPU accuracy jobs target Python 3.12 with JAX
and JAXLIB 0.10.2, Flax 0.12.8, and NumPy 2.4.6; and Python 3.13 with JAX and
JAXLIB 0.12.0.1, Flax 0.12.12, and NumPy 2.5.4. Both use Equinox 0.13.8, ONNX
1.23.2, ONNX IR 1.1.0, and ONNX Runtime 1.31.1. Each GitHub Actions accuracy
job records its actual dependency versions, runner image, and CPU hardware in
its job summary; use that record when interpreting a run. The CPU jobs are the
authoritative numerical acceptance gate. Changes to limits, reference
calculation, inputs, or test conditions require explicit justification and
before/after evidence. Tests must never derive or rewrite their own acceptance
bounds from the current run.

The opset only selects the ONNX schema contract; it does not assert support in a
particular runtime version. Validate the chosen `opset` and normalization mode
against the actual deployment runtime.

For example, [TensorRT 10.9's `GroupNormalization-21` importer](https://github.com/onnx/onnx-tensorrt/blob/d5dce67db7c2e64b07e055571f5ec06f7f254de2/onnxOpImporters.cpp#L2260-L2330)
internally uses a fixed rank-4 normalization core. Native GroupNorm export
therefore temporarily extends rank-2/3 inputs with singleton dimensions and
flattens the spatial dimensions of higher-rank inputs before restoring the
original shape. The standard `GroupNormalization` node and its channel-wise
`(C)` parameters remain visible and schema-conformant. TensorRT 10.15's newer
NormalizationV2 path does not require this physical-shape adaptation, but
parses the same rank-canonicalized models.

TensorRT's NormalizationV2 importer nevertheless has a separate correctness
defect for `GroupNormalization(num_groups=1)`: through at least TensorRT
10.16.1 it normalizes each channel independently instead of reducing over the
single group containing all channels. Use
`normalization_mode="force_decomposed"` for that TensorRT case until a fixed
version is verified; see [TensorRT #4756](https://github.com/NVIDIA/TensorRT/issues/4756)
and the still-open [onnx-tensorrt fix](https://github.com/onnx/onnx-tensorrt/pull/1052).

The ONNX schema function for native GroupNorm cannot execute zero-sized batch
or spatial dimensions in the checked runtimes. Statically known empty shapes
therefore fall back to the explicit graph even in `"prefer_native"` mode. If a
symbolic dimension may become zero at runtime, export with
`normalization_mode="force_decomposed"`.

## Training Is Out of Scope

`jax2onnx` exports ONNX artifacts for inference-style execution. It does not
attempt to preserve JAX training loops, optimizer state, automatic
differentiation behavior, or Python-side training control flow.

## Coverage Pages

For current coverage information, see:

- [Supported Components](supported_components.md)
- [ONNX Operator Coverage](onnx_operator_coverage.md)
- [JAX LAX Coverage](jax_lax_coverage.md)
- [JAX NumPy Coverage](jax_numpy_coverage.md)
- [Flax API Coverage](flax_api_coverage.md)
- [Equinox NN Coverage](equinox_nn_coverage.md)
