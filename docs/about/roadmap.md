# Roadmap

## Planned

* Extend target-oriented validation guidance beyond the documented ONNX Runtime
  CPU and Web/WASM flows, especially for mobile deployments where practical.
* Broaden capability-matrix coverage across dtype and shape variants, including
  BF16, dynamic dimensions, and non-square inputs.
* Add focused end-to-end deployment examples for small vision and numerical
  models.
* Add a realistic end-to-end RL deployment example based on a widely used RL
  library, loading a trained actor and exporting the inference-only
  `obs -> action` policy contract.
* Continue targeted coverage work for JAX, Flax NNX/Linen, Equinox, SotA
  examples, and physics/simulation use cases, including classification of newly
  published upstream APIs before claiming them as supported.


## Upcoming Version

### **jax2onnx 0.18.0**

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
