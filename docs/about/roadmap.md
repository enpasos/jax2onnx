# Roadmap

## Planned

* Consolidate the lowering, value-binding, and graph-access contracts into a
  readable typed core. See the
  [converter architecture](../developer_guide/architecture.md).
* Preserve FP16/BF16 types across NCHW export bridges and use the correct module
  instance for decorated function evaluation, shape inference, and batching.
  Keep graph ownership checks, import retries, and precision restoration intact.

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
