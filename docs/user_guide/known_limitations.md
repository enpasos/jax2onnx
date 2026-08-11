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

For GroupNorm and Flax RMSNorm, `normalization_mode="auto"` preserves the
framework-oriented default: Flax Fast-Variance GroupNorm remains explicit
because ONNX `GroupNormalization` does not model Flax's negative-variance clamp
or reduction order, while Flax RMSNorm uses `RMSNormalization` at opset 23 or
newer except for low-precision Linen configurations that deliberately disable
FP32 statistics. Set `normalization_mode="semantic"` to emit a Fast-Variance
GroupNorm as `GroupNormalization` at opset 21 or newer. This improves graph
readability but can produce material runtime-dependent differences for
high-offset or otherwise ill-conditioned inputs. Slow-Variance GroupNorm stays
explicit in every mode.
Set `normalization_mode="decomposed"` to force the explicit reduction layout for
both plugins. The mode controls the exported ONNX graph; a runtime optimizer may
still recognize and fuse an explicit normalization pattern.

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
single group containing all channels. Use `normalization_mode="decomposed"` for
that TensorRT case until a fixed version is verified; see [TensorRT
#4756](https://github.com/NVIDIA/TensorRT/issues/4756) and the still-open
[onnx-tensorrt fix](https://github.com/onnx/onnx-tensorrt/pull/1052).

The ONNX schema function for native GroupNorm cannot execute zero-sized batch
or spatial dimensions in the checked runtimes. Statically known empty shapes
therefore fall back to the explicit graph even in `"semantic"` mode. If a
symbolic dimension may become zero at runtime, export with
`normalization_mode="decomposed"`.

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
