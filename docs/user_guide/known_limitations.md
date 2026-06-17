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
