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


## Current Version


### **jax2onnx 0.16.1**


* **Record trustworthy model provenance:** Populate exported ONNX models with
  the active `jax2onnx` producer version while handling source checkouts safely,
  so metadata identifies the converter build without changing graph semantics.
* **Harden and modernize GitHub Actions:** Pin third-party actions to immutable
  commit SHAs, declare least-privilege token access for CI and nightly jobs, and
  move workflows to the Node 24-based `actions/checkout` 7.0.1 and
  `actions/setup-python` 7.0.0 releases.
* **Automate dependency maintenance with bounded noise:** Group minor and patch
  updates for GitHub Actions and npm, rate-limit update traffic across Actions,
  Python, npm, and pre-commit dependencies, keep major upgrades isolated for
  review, and defer separate `uv` automation until lockfile synchronization has
  a defined policy.
* **Protect both supported JAX stacks:** Retain JAX/JAXLIB 0.10.2 for Python
  3.11/3.12 and 0.11.0 for Python 3.13/3.14 in the Poetry lockfile, with a CI
  guard that verifies the modern stack remains present.
* **Refresh the validation and tooling stack:** Validate against ONNX Runtime
  and `onnxruntime-web` 1.29.0, Playwright 1.62.1, pytest 9.1.1, and Ruff 0.16.4,
  with an explicit `E4`, `E7`, `E9`, and `F` lint baseline.

## Past Versions

See [Past Versions](past_versions.md) for the full release archive.
