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
  examples, and physics/simulation use cases.


## Current Version


### **jax2onnx 0.16.0**


* **Harden the `onnx-ir` integration without changing export contracts:**
  Validate the resolved dependency stack against `onnx-ir` 1.0.0, replace the
  private tape-builder dependency with a local adapter based on the public
  `onnx_ir.tape.Tape` API, prohibit private `onnx_ir` imports in converter and
  plugin code, and exercise the required public surface in minimum-dependency
  CI while retaining compatibility with the declared `onnx-ir>=0.2.1` floor.
* **Keep behavior-changing `onnx-ir` 1.0 features deliberate:** Continue
  emitting IR version 10 and preserving the existing external-data artifact
  layout. Revisit IR version 11 device and sharding metadata only after graph
  optimization is placement-safe and runtime round trips are covered; consider
  external-data sharding only together with a justified public export contract.
* **Add configurable normalization export policies:** Add `normalization_mode`
  to `to_onnx(...)`: `"auto"` preserves framework-oriented defaults,
  `"prefer_native"` selects standard ONNX normalization operators when eligible
  and falls back to the explicit graph otherwise, and `"force_decomposed"`
  always emits the explicit graph. Apply the policy to Fast-Variance Flax
  GroupNorm and Flax RMSNorm according to the target opset and statistics dtype.
* **Harden opt-in native GroupNorm portability:** Preserve arbitrary channel
  axes and mapped batch dimensions, adapt rank-2/3 and higher-rank inputs around
  the native node for runtime compatibility, retain channel-wise affine
  parameters, and fall back to the explicit decomposition for slow variance or
  statically empty shapes.
* **Stabilize slow-variance GroupNorm exports:** Add a finite constant-group
  correction that keeps finite constants exactly centered while preserving the
  existing reduction semantics for non-constant high-offset groups and
  retaining NaN/Inf behavior; cover symbolic and empty shapes across Equinox
  and Flax normalization paths.
* **Expand normalization coverage and guidance:** Add structural tests across
  policies, opsets, dtypes, and ranks; verify policy propagation into ONNX
  Function bodies and control-flow contexts; extend ONNX checker and ONNX
  Runtime parity coverage for numerical and empty-shape edge cases; document
  runtime-specific tradeoffs and refresh the generated component matrix.

## Past Versions

See [Past Versions](past_versions.md) for the full release archive.
