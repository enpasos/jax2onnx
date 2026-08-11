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


## Planned Version


### **jax2onnx 0.15.2**


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

## Current Version

### **jax2onnx 0.15.1**

 
* **Support explicit Flax NNX convolution padding:** Convert canonicalized
  non-string `nnx.Conv` padding to immutable nested integer tuples before
  primitive binding, satisfying JAX's static-parameter hashability requirement
  for scalar, per-axis integer, and asymmetric before/after pair forms while
  leaving named padding modes unchanged; add regression coverage for all three
  forms.
* **Fix batched `jnp.split` exports:** Resolve positive and negative split axes
  against the logical unbatched rank under `vmap` and preserve valid
  zero-length outputs, including repeated split indices.
* **Support non-divisible Equinox adaptive pooling:** Keep the compact
  `AveragePool`/`MaxPool` path for divisible shapes and lower non-divisible 1D
  and 2D average or max targets through per-bin `Slice` plus reduction graphs,
  for both batched and unbatched inputs.
* **Harden Equinox and Flax NNX GroupNorm exports:** Use explicit,
  layout-preserving ONNX decompositions across opsets 17 through 23 while
  preserving nested and symbolic mapped batch axes, framework variance modes,
  dtype promotion, affine application, and finite, tolerance-checked behavior
  for constant, near-constant, and high-offset inputs.
* **Align Flax Linen normalization semantics:** Make `GroupNorm` and
  `InstanceNorm` preserve Linen's separate statistics, affine-parameter, and
  result dtype stages; retain the centered-variance framework path where its
  graph contract is valid and reject configurations that would require silent
  dtype narrowing.
* **Preserve Equinox PRNG and batching contracts:** Let `eqx.nn.Linear` accept
  an optional key, make deterministic and nested `eqx.nn.Sequential` exports
  ignore keys only when every layer permits it, retain keys for stochastic
  dropout, and give the unary Sequential primitive a correct batching rule.
* **Add Equinox internal primitive lowerings:** Export `unvmap_any`,
  `unvmap_all`, `unvmap_max`, `nonbatchable`, and the unbatched
  `select_if_vmap` identity case through explicit ONNX operations.
* **Correct vmapped `while_loop` semantics:** Reduce mapped predicates to the
  scalar continuation condition required by ONNX `Loop`, make that condition
  safe for empty batches, mask completed examples so only active elements
  update, and carry values used only by the condition through the Loop
  interface.
* **Preserve initializer ownership across captures:** Avoid eagerly deleting
  original initializers when inlining reshaped Equinox convolution biases and
  rotary-attention caches, so aliases and `Loop` captures remain valid.
* **Make reduction lowering opset-aware:** Centralize reduction construction
  and emit axes as attributes or tensor inputs according to each target schema
  for mean, max, min, product, sum, L1/L2, log-sum, log-sum-exp, and sum-square
  reductions used by lax, jnp, `jax.nn`, normalization, adaptive pooling,
  Equinox internal primitives, and control flow.
* **Correct reduction edge semantics:** Preserve empty-axis identities and
  unsigned result dtypes, avoid invalid L1 and sum-square fusions, booleanize
  numeric `jnp.any`/`jnp.all` inputs, use the correct identities for empty
  dimensions, and reject unsupported integer bitwise reductions explicitly.
* **Make IR graph rewrites semantics-preserving:** Refresh elementwise shape
  metadata after safe reshape folding, derive `CastLike` shape only from its
  data input, remove cast round trips only when losslessness is globally or
  statically proven, preserve casts observed by graph outputs or nested
  captures, and leave custom-domain operators untouched.
* **Isolate scatter-specific shape handling:** Remove scatter window hints from
  shared broadcast state and retain only dedicated loop-extent hints,
  preventing one scatter lowering from changing unrelated later broadcasts.
* **Complete symbolic shape metadata:** Mark dynamic `Shape` and `Gather`
  helper outputs from `broadcast_in_dim` as `INT64`, allowing nested symbolic
  normalization graphs to serialize cleanly and pass the ONNX checker.
* **Keep optional ORMQR discovery portable:** Allow plugin discovery on JAX
  versions that do not expose the ORMQR primitive while retaining the existing
  lowering where it is available.
* **Make generated-test execution self-contained:** Bootstrap the repository
  root in `scripts/generate_tests.py` so direct invocation and
  `scripts/run_all_checks.sh` work from a source checkout without an external
  `PYTHONPATH`.
* **Expand regression coverage:** Add ONNX checker and ONNX Runtime parity
  coverage for split batching and empty outputs, normalization dtype and
  numerical edge cases, adaptive pooling, Equinox key and internal-primitive
  interoperability, loop masking and captures, reduction schemas and
  identities, optimizer safety, scatter isolation, symbolic metadata, and
  ORMQR discovery.
 
## Past Versions

See [Past Versions](past_versions.md) for the full release archive.
