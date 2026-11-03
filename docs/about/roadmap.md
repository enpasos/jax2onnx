# Roadmap

## Planned

* Document and test how to export a JAX model to ONNX and run it in a mobile
  application.
* Test more operators and inference runtimes with different numerical
  precisions, input dimensions that vary between calls, and non-square arrays.
* Add a runnable numerical or simulation example that takes a system's state
  or numerical parameters and computes an updated state or result with ONNX
  Runtime, without installing JAX.
* Publish a reproducible example that loads a trained reinforcement-learning
  policy from an established library and exports its observation-to-action
  computation to ONNX for inference.
* Keep tables of supported JAX, Flax and Equinox APIs current as those libraries
  evolve, and add regression tests for remaining operator and model conversion
  gaps.


## Upcoming Version

### **jax2onnx 0.18.0**

* **Deploy a trained quantized classifier:** Add an optional
  [W8A8 MNIST example](../user_guide/quantized_mnist.md) with reproducible training,
  ordinary ONNX export and independent CPU inference. Compare held-out accuracy,
  actual artifact size, memory and latency; Qwix remains an optional dependency.
* **Restore affected exports with JAX 0.12:** Handle newer internal
  representations used by `argwhere`, `nonzero`, `polyval`, and `sinc`, while
  retaining compatibility with JAX 0.10.2.
* **Preserve FP16 sequence types in `scan`:** Keep each step's input in the
  sequence dtype, including when the caller enables JAX x64.
* **Follow the prepared JAX computation:** Use one precision contract through
  `to_onnx` and `allclose`. Shape-only specs remain FP32; explicitly typed FP64
  inputs require an effective JAX configuration that preserves them. Preserve
  explicit casts, shared constants, and captured precision contexts. Remove
  `enable_double_precision` and diagnose any supplied value with migration
  guidance. See [Dtypes and Precision](../user_guide/api.md#dtypes-and-precision)
  and [PR #288](https://github.com/enpasos/jax2onnx/pull/288).
* **Introduce a typed lowering core:** Add a closed native plugin context,
  explicit input/output binding, and graph ownership checks. Prepare
  legacy extension calls at registration and share child-context construction
  across Functions and control flow. See the
  [converter architecture](../developer_guide/architecture.md).
* **Reuse native ONNX Function definitions:** Finalize and cache `ir.Function`
  objects directly, preserving names, metadata, and graph ownership without
  parallel function snapshots or a second finalization step.
* **Preserve decorated-function arguments:** Keep dynamic keyword dependencies
  through `jit` and nested control flow, preserve explicit flags and omitted
  defaults, and select the correct decorated module instance. Document `vmap`
  inlining and function-reuse rules in the
  [ONNX Functions guide](../user_guide/onnx_functions.md).
* **Make static configuration reuse safe:** Snapshot supported mutable
  configuration and compare exact contents when reusing Function definitions.
  Keep static strings, booleans, and axis configuration static during shape
  inference, including under `jit`.
* **Carry runtime flags through loops:** Forward export flags explicitly through
  `scan`, `while_loop`, and `fori_loop` bodies, including nested scopes and zero
  iterations. Represent fori closure values as regular JAX operands so their
  dependencies survive tracing.
* **Preserve precision and export integrity:** Keep FP16/BF16 types across NCHW
  export bridges, protect existing graph values during output allocation, and
  retain safe import retries and precision restoration after failed exports.
* **Preserve JAX scalar promotion across export boundaries:** Retain weak
  typing through inputs, named functions, captured values, and function reuse.
  A weak FP32 scalar combined with an FP16 value can stay FP16 when JAX requires
  it, while a strongly typed FP32 operand still requests FP32 promotion.
* **Export 8-bit quantized matrix computations:** Preserve INT8/UINT8 operands
  with INT32 accumulation and match JAX's saturation and NaN-to-zero handling
  when converting floating values to 8-bit integers. Keep the configured Linen
  Dense computation for structured weights, and demonstrate ordinary export
  with a separately tested, optional Qwix W8A8 example. See
  [PR #287](https://github.com/enpasos/jax2onnx/pull/287).
* **Preserve operation-specific precision boundaries:** Keep explicit FP32
  consumers and bitcast representations in their source types. Retain FP16
  Add/Sub/Mul rounding with local FLOAT work and half result casts; FP16 Rsqrt
  has one half boundary after FLOAT Sqrt and Div. Keep integer window sums in
  exact integer arithmetic. Locked CPU runtime measurements and remaining
  qualification limits are recorded in [Dependencies](dependencies.md) and
  [Known Limitations](../user_guide/known_limitations.md).
* **Keep `log1p`, `expm1`, and functions built on them accurate:** Replace
  `Log(1 + x)` and `Exp(x) - 1`, which return zero or lose most digits for small
  inputs, with compensated forms shared by the `lax` and `jnp` lowerings and
  `jax.nn.log1mexp`. Build `jax.nn.log_sigmoid` on `log1p` instead of
  `Log(Sigmoid(x))`, which returned `-inf` for `x <= -20` on ONNX Runtime, and
  `jnp.sinh` and non-FP32 `lax.sinh` on `expm1` instead of
  `(Exp(x) - Exp(-x)) / 2`; `jnp.sinh` keeps running on CUDA, which has no
  `Sinh` kernel. Results stay within a few ulp in the input dtype without
  promoting the model; BF16 computes in FP32 internally so ONNX Runtime can run
  it. FP16 and BF16 exports of these functions keep their source dtype when the
  caller enables JAX x64, and `jax.nn.log1mexp` now exports FP16
  inputs. See
  [Known Limitations](../user_guide/known_limitations.md#log1p-expm1-and-related-functions).
* **Compute `jax.nn.dot_product_attention` like JAX:** Combine the boolean,
  causal, local-window, and sequence-length masks, and replace masked logits
  with JAX's large finite negative value before `Softmax`, instead of masking
  and renormalizing the weights afterwards. A large masked logit no longer zeroes
  the allowed weights, fully masked rows get uniform weights like JAX instead of
  zeros, and an explicit `mask` no longer drops `local_window_size`. Output rows
  beyond `query_seq_lengths` are zeroed. FP16 and BF16 inputs compute the
  logits and softmax in FP32 and cast the weights back before the value
  product.
* **Keep FP16 scalars and weakly typed `jnp.where` literals in the input
  dtype:** Scalar initializers retain the traced dtype, so FP16 graphs with
  scalar constants, such as the attention scale, stay valid.
  `jnp.where(c, x, 0)` follows JAX's weak-type promotion: BF16 `x` exports
  instead of failing, and FP16 `x` keeps FP16 instead of becoming FP32.
* **Reject `lax.bitcast_convert_type` below opset 26:** ONNX `BitCast` exists
  only from opset 26, so exports at lower opsets raise a clear error instead of
  producing an invalid model. A bitcast to the operand's own dtype lowers to
  `Identity` at any opset.
* **Require ONNX IR 1.0.0 or newer:** Align the declared dependency floor,
  package metadata, developer guidance, and minimum-version CI coverage on
  `onnx-ir>=1.0.0`.


## Current Version

### **jax2onnx 0.17.1**

* **Fix activation argument binding:** Preserve positional and keyword scalar
  options for JAX and Flax Linen GELU, ELU, CELU, and LeakyReLU exports. This restores
  Keras GELU export with its JAX backend and prevents positional CELU `alpha`
  from silently falling back to the default value. See
  [issue #270](https://github.com/enpasos/jax2onnx/issues/270).
* **Stabilize float32 LayerNorm variance:** Equinox and slow-variance Flax NNX
  explicit exports now accumulate the squared centered values in float64, then
  round the variance back to float32 before epsilon, normalization, and affine
  operations. This improves accuracy portability across runtime reduction orders
  while preserving the existing fixed CPU accuracy limits. The two additional
  casts apply independently of `enable_double_precision`; fast variance and
  FP16/BF16/FP64 statistics are unchanged. See
  [PR #272](https://github.com/enpasos/jax2onnx/pull/272).
* **Strengthen normalization regression checks:** Assert LayerNorm reduction and
  output dtypes for FP16, BF16, FP32, and FP64 at opsets 16 and 23. Correct the
  vmapped Equinox GroupNorm test's tolerance for its JAX reference while continuing
  to require exact zero exported outputs for constant groups.
* **Update development and validation tools:** Update test integrations to
  Torch 2.14.0, formatting to Black 26.5.1, type checking to mypy 2.3.1, and
  browser validation to Playwright 1.63.0; align the Ruff pre-commit hook
  with version 0.16.9. Restore strict API documentation builds with mkdocstrings
  1.0.7's compatibility fix for pymdown-extensions 12.2.
* **Measure FP16 deployment performance:** Extend the decoder normalization
  benchmark with FP16 parameters and inputs, configurable embedding widths and
  CPU threads, and a standalone RMSNorm workload. Compare static and symbolic
  exports with JAX parity, and report session initialization, first-run, and
  warmed inference timings separately.
* **Expose ConvTranspose/BatchNorm fusion:** Fold eligible immutable float32 or
  float64 channel bias Adds into an absent ConvTranspose bias input. Preserve
  observable convolution outputs, existing biases, and required padding; the
  resulting standard ONNX graph can use ONNX Runtime 1.31's BatchNorm fusion.
* **Preserve FP16 intermediate output metadata:** Keep traced float16 output
  placeholders in float16 when double-precision export is disabled. This fixes
  invalid output declarations in FP16 LinearGeneral projections and decoder
  exports.




## Past Versions

See [Past Versions](past_versions.md) for the full release archive.
