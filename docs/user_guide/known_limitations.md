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

## Runtime Compatibility

ONNX Runtime compatibility depends on:

- the operators emitted by the export,
- the target opset,
- the ONNX Runtime version,
- the execution provider,
- whether the model is intended for Python, browser/WASM, or another deployment target.

For browser/WASM deployment, use `export_mode="web"` and the Web validation
workflow.

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

### Float64 cosine

Float32 cosine uses native ONNX `Cos`. For float64, `jnp.cos` and `lax.cos`
use the equivalent expression `1 - 2 * Sin(x / 2)²`. This supports older
ONNX Runtime CPU versions without a double-precision `Cos` kernel and avoids
losing the phase shift in `Sin(x + π/2)` for large arguments. The half-angle
formula can lose relative accuracy near cosine zeros; its regression checks
use an absolute tolerance of `2e-15` on selected finite float64 inputs, including
large arguments. NaN and infinity retain their NaN cosine results. These checks
cover ONNX Runtime CPU; validate the actual deployment provider separately.

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
epsilon, normalization, and affine math stay float32. The widening is
unconditional (independent of `enable_double_precision`), LayerNorm-specific, and
improves accuracy portability across runtime reduction orders without
guaranteeing identical results. The fixed comparisons below cover float32 inputs;
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
JAXLIB 0.11.2, Flax 0.12.10, and NumPy 2.5.3. Both use Equinox 0.13.8, ONNX
1.23.0, ONNX IR 1.0.0, and ONNX Runtime 1.30.0. Each GitHub Actions accuracy
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
