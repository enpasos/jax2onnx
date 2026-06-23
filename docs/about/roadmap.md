# Roadmap

## Planned

* Expand deployment-readiness reporting beyond the basic validation guide:
  operator inventory, dtype/shape summaries, stricter shape-inference checks,
  and reusable report artifacts for exported models.
* Extend target-oriented validation guidance beyond the documented ONNX Runtime
  CPU and Web/WASM flows, especially for mobile deployments where practical.
* Broaden capability-matrix coverage across dtype and shape variants, including
  BF16, dynamic dimensions, and non-square inputs.
* Add focused end-to-end deployment examples for small vision, RL-style, and
  numerical models.
* Continue targeted coverage work for JAX, Flax NNX/Linen, Equinox, SotA
  examples, and physics/simulation use cases.

## Upcoming Version

### **jax2onnx 0.14.2**

* **Add deterministic RL policy examples:** Provide continuous-control and
  discrete-control `examples.rl` exports that validate an `obs -> action`
  deployment contract through the standard generated example-test path.

## Current Version

### **jax2onnx 0.14.1**

* **Add targeted Opset 27 coverage:** Preserve the Opset 23 default while using
  native FP16/BF16 `Range` for `opset=27` exports and keeping older exports on
  the cast fallback; cover new JAX 0.10 resize methods where the ONNX mapping is
  exact.
* **Centralize JAX internal API compatibility:** Add a package-level JAX
  compatibility layer, migrate direct `jax.core` / `jax.extend.core` usage
  across converter and plugin code, and guard `Literal` resolution across JAX
  layout differences so JAX internal API moves are handled in one place.
* **Validate declared lower bounds in CI:** Raise the installable minimum JAX
  version to `0.8.1` for Flax/NNX `0.12.1` compatibility, add a
  minimum-dependency workflow job, and cover the lower-bound installer with
  focused tests.
* **Refresh the validation stack:** Update the documented runtime stack to JAX
  `0.10.2`, Equinox `0.13.8`, ONNX `1.22.0`, ONNX Runtime `1.27.0`, and keep
  Web validation on the latest stable `onnxruntime-web` release until a matching
  `1.27.x` Web package is published.

## Past Versions

See [Past Versions](past_versions.md) for the full release archive.
