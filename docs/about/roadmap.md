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
