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

## Upcoming Version

### **jax2onnx 0.15.1**

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
* **Preserve control-flow initializer ownership:** Keep reshaped Equinox
  convolution parameters and rotary-attention caches available until normal
  dead-code elimination so captured values do not become dangling Loop inputs.
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
* **Refresh documentation and attribution:** Clarify initializer rules for ONNX
  Functions and control-flow `GraphProto` bodies, document runtime-sensitive
  normalization tolerances, regenerate component/operator coverage, and
  acknowledge @clementpoiret's returning hardening contribution.

## Current Version

### **jax2onnx 0.15.0**

* **Add deterministic RL policy exports:** Provide continuous-control and
  discrete-control `examples.rl` exports for the `obs -> action` deployment
  contract, documented with RL policy-only guidance and validated through the
  standard generated example-test path.
* **Harden generated example runtime contracts:** Add optional ONNX shape
  inference and runtime contract hooks to example metadata so deployment
  examples can validate extra concrete batch sizes, output dtype/shape, and
  domain-specific output constraints without separate test trees.
* **Add generated deployment readiness summaries:** Let generated examples run
  an integrated readiness check with checker status, strict shape-inference
  status, public dtype/shape summaries, initializer summaries, operator
  inventory, and public-dimension warnings without expanding the public API.
* **Add JAX 0.11 support:** Track the `scan` parameter change from
  `num_consts`/`num_carry` to the `ft_in`/`ft_out` flat-tree descriptors, guard
  recursive jaxpr walks against the merged `ClosedJaxpr`/`Jaxpr` type whose
  `.jaxpr` now returns itself, route the internal APIs dropped from `jax.core`
  through the compatibility layer, and add an `empty` primitive plugin for the
  new `jnp.empty` lowering.
* **Accept the Flax `out_sharding` argument:** Let the `nnx.LinearGeneral`
  monkey-patch accept and ignore the placement hint added in Flax `0.12.8`,
  matching the existing `Linear` and `Conv` patches.
* **Keep the pre-0.11 JAX path supported:** Leave the declared `jax>=0.8.1`
  floor and the Python 3.11 test row in place; since JAX `0.11`, NumPy `2.5`,
  and SciPy `1.18` all require Python 3.12, the 3.11 job now validates the full
  suite against the older JAX stack through the same compatibility layer.
* **Refresh the validation stack:** Update the documented runtime stack to JAX
  `0.11.0`, Flax `0.12.8`, ONNX Runtime `1.28.0`, and `onnxruntime-web`
  `1.27.0`; pull the transitive `protobufjs` dev dependency up to `7.6.5` to
  clear its advisories; and raise the pinned mypy to `1.20.2` with a `3.12`
  type-check target, which the NumPy `2.5` stubs require.
* **Guard the generated coverage tables:** Make `scripts/generate_readme.py`
  abort instead of silently dropping documented rows when an optional plugin
  world (MaxText, MaxDiffusion) is not registered, check every target before
  writing any of them, and allow deliberate deletions via `--allow-removals`.

## Past Versions

See [Past Versions](past_versions.md) for the full release archive.
