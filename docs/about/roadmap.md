# Roadmap

## Planned

* Document and test how to export a JAX model to ONNX and run it in a mobile
  application.
* Expand automated compatibility tests for FP16, BF16 and FP64 computations,
  input dimensions that vary between calls, and non-square arrays. Document
  supported operator and runtime combinations and their numerical limits.
* Add a runnable numerical or physical simulation example that exports a
  complete computation to ONNX and executes it in ONNX Runtime without JAX,
  with numerical checks against the original model.
* Publish a reproducible example that loads a trained reinforcement-learning
  policy from an established library and exports its observation-to-action
  computation to ONNX for inference.
* Keep tables of supported JAX, Flax and Equinox APIs current as those libraries
  evolve, and add regression tests for remaining operator and model conversion
  gaps.


## Upcoming Version

### **jax2onnx 0.18.1**

* **Preserve masked softmax broadcasting under `vmap`:** Export masks that
  expand singleton logits dimensions or add sample dimensions through JAX's
  decomposition, preserving output shapes and reduction axes. Masks that fit
  the logits retain the native ONNX Softmax lowering.


## Current Version

### **jax2onnx 0.18.0**

* **Breaking change — configure precision in JAX:** Remove
  `enable_double_precision` from `to_onnx` and `allclose`; supplying it with any
  value, including `False` or `None`, raises `TypeError`. Export now retains the
  caller's JAX configuration, including when the old argument was omitted.
  `allclose` also retains matrix-precision settings instead of forcing
  `"float32"`. Removing the argument alone may change results: follow the
  [migration from v0.17.1](../user_guide/api.md#migration-from-v0171).
* **Preserve the prepared model's precision:** Model configuration, typed
  inputs and explicit casts determine computation types. Shape-only input
  specifications still default to FP32, even with JAX x64 enabled. For FP64,
  prepare both the model and typed inputs in an enabled x64 context and use
  matching settings for validation. Constants and local precision settings keep
  their meaning. Unsupported explicit matrix algorithms and unary accuracy
  requests raise errors; see [Known Limitations](../user_guide/known_limitations.md)
  and [PR #288](https://github.com/enpasos/jax2onnx/pull/288).
* **Deploy a trained quantized classifier:** Add a reproducible
  [MNIST example](../user_guide/quantized_mnist.md) with 8-bit weights and
  activations (W8A8), ordinary ONNX export and independent CPU inference without
  JAX, Flax or Qwix. On 10,000 held-out images, accuracy changes from 97.96% to
  97.93% while the ONNX file is 73.9% smaller than the FP32 version. INT8
  inference is slower and memory use is similar on the measured CPU. Qwix is
  optional and needed for preparing and exporting this example. See
  [PR #289](https://github.com/enpasos/jax2onnx/pull/289).
* **Export 8-bit integer matrix products:** Support INT8/UINT8 operands with
  INT32 accumulation. Floating-point conversions to 8-bit integers preserve
  JAX's saturation at the integer limits and map NaN to zero. Linen Dense
  preserves the configured computation for structured weights, enabling
  prepared quantized models to use the standard export API. Integer reductions
  and window sums retain integer arithmetic and wraparound at overflow. See
  [PR #287](https://github.com/enpasos/jax2onnx/pull/287) for the quantization
  foundation.
* **Preserve JAX scalar promotion:** Keep JAX's weak typing for inputs,
  captured values and ONNX Functions. A weak FP32 scalar can combine with an
  FP16 value in FP16, while a strongly typed FP32 operand requests FP32
  promotion. Scalar constants keep their traced type; `jnp.where(c, x, 0)`
  now exports with BF16 inputs and retains FP16 for FP16 inputs.
* **Preserve FP16 rounding and tensor types:** Keep FP16 rounding at the
  results of addition, subtraction and multiplication, and round `rsqrt` once
  at its result. Preserve FP16 sequence inputs in `scan` and FP16/BF16 types
  when converting channel layouts for export, including with JAX x64 enabled.
  See the tested CPU configurations and numerical exceptions in
  [Known Limitations](../user_guide/known_limitations.md#fp16-arithmetic-boundaries).
* **Improve FP64 computations:** Keep `atan2` arithmetic in FP64 and improve
  FP64 cosine for large arguments. Selected pointwise convolutions with unit
  strides and no padding can run in FP64, including complex128 components, on
  ONNX Runtime CPU. Supported convolution configurations and remaining
  numerical limits are documented in [Dependencies](dependencies.md#onnx-runtime-kernel-gaps)
  and [Known Limitations](../user_guide/known_limitations.md#fp64-atan2-subnormal-inputs).
* **Improve numerical stability near zero:** Use compensated formulas for
  `log1p`, `expm1`, `jax.nn.log1mexp` and related `sinh` implementations. Avoid
  premature `-inf` results in `jax.nn.log_sigmoid` for large negative inputs.
  FP16/BF16 models retain their input and output types; BF16 calculations use
  FP32 internally. `jax.nn.log1mexp` also supports FP16 inputs. See the
  [accuracy checks and remaining limitations](../user_guide/known_limitations.md#log1p-expm1-and-related-functions).
* **Correct `jax.nn.dot_product_attention` masking:** Combine boolean, causal,
  local-window and sequence-length masks before softmax. Large masked logits
  no longer zero the allowed weights; fully masked rows have uniform weights
  as in JAX, and rows beyond `query_seq_lengths` produce zero outputs. FP16 and
  BF16 inputs compute logits and softmax in FP32, then cast weights back before
  multiplying by the values.
* **Preserve arguments through functions and loops:** Keep dynamic keyword
  arguments and runtime flags through `jit`, nested `scan`, `while_loop` and
  `fori_loop`. Decorated functions retain explicitly supplied flags, omitted
  defaults and the correct module instance, including inside control flow.
* **Make ONNX Function reuse reliable:** Distinguish calls with different
  model state, supported static configuration, weak types or local precision
  settings. Snapshot supported mutable configuration so later changes do not
  alter earlier calls. Preserve function names and metadata. See the
  [ONNX Functions guide](../user_guide/onnx_functions.md) for reuse rules and
  `vmap` behavior.
* **Support JAX 0.12 while retaining Python 3.11–3.14:** Fix exports affected
  by newer JAX representations, including `argwhere`, `nonzero`, `polyval` and
  `sinc`. Test the locked JAX 0.10.2 stack on Python 3.11/3.12 and JAX 0.12 on
  Python 3.13/3.14. See [Dependencies](dependencies.md) for the resolved versions.
* **Improve diagnostics and plugin development:** Add typed plugin
  registration with explicit input/output binding and checks for missing or
  disconnected graph values. Existing `register_primitive` extensions remain
  supported. See the [Plugin System guide](../developer_guide/plugin_system.md)
  and [converter architecture](../developer_guide/architecture.md).
* **Clarify export and validation:** Explain when `input_params` are fixed
  during tracing or become ONNX graph inputs, how ONNX Functions retain local
  precision settings, and how to configure inference explicitly.
  Explain that `allclose` checks ONNX Runtime CPU with graph optimizations
  disabled; BF16 tensor inputs and outputs require separate validation through
  I/O binding. See [Validation](../user_guide/validation.md) and
  [BF16 runtime I/O](../user_guide/known_limitations.md#bf16-runtime-io).
* **Check bitcast compatibility:** Bit reinterpretation with `lax.bitcast_convert_type`
  requires opset 26 and compatible ONNX validation and runtime support; lower
  opsets now fail with a clear error. A bitcast to the same dtype remains
  an `Identity` at any opset. See
  [runtime compatibility](../user_guide/known_limitations.md#execution-targets).
* **Require ONNX IR 1.0.0 or newer:** Raise the dependency minimum to
  `onnx-ir>=1.0.0`.

## Past Versions

See [Past Versions](past_versions.md) for the full release archive.
