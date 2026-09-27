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


### **jax2onnx 0.16.2**


* **Preserve transposed-convolution geometry:** Lower input-dilated
  `lax.conv_general_dilated` (used by `eqx.nn.ConvTranspose`,
  `nnx.ConvTranspose`, `jax.lax.conv_transpose`, and `nnx.Conv` with
  `input_dilation`) to ONNX `ConvTranspose` with the input dilation as its
  strides, a group-aware kernel layout, and an explicit output `Pad` where JAX
  pads beyond the kernel's reach, so exported shapes and values match JAX; fail
  export explicitly for strided input dilation, `batch_group_count > 1`, and
  complex transposed convolutions instead of emitting different semantics.
* **Make GELU exports opset-aware:** Emit ONNX `Gelu` only for opset 20 and
  newer, and lower `jax.nn.gelu` and `nnx.gelu` below opset 20 to the exact
  (`Erf`) or tanh-approximate formula so the graph validates at the requested
  opset.
* **Behavior change: explicit normalization graphs by default.**
  `normalization_mode="auto"` now exports the representation with the best
  reproducible accuracy and prefers native operators only when they meet the
  same locked accuracy bounds. For GroupNorm, Flax RMSNorm, and Equinox/Flax
  LayerNorm this is currently the explicit graph that reproduces the
  framework's statistics, instead of ONNX `LayerNormalization` (opset 17+) or
  `RMSNormalization` (opset 23+). Pass `normalization_mode="prefer_native"` to
  keep the native operators, for example for smaller graphs on accelerated
  runtimes.
* **Add precision-faithful LayerNorm export:** Equinox, Flax NNX, and Flax Linen
  LayerNorm now honor `normalization_mode`. The explicit graph follows the
  framework's statistics (two-pass variance with exact zeros for constant rows,
  or Flax's clamped fast variance, in float32 for low-precision inputs) and is
  not re-fused by ONNX Runtime, whose CPU `LayerNormalization` kernel loses
  precision on rows with very large activations. Exports below opset 17 no
  longer emit an operator the opset does not define.
* **Add native Equinox RMSNorm export:** `eqx.nn.RMSNorm` now honors
  `normalization_mode`; `"prefer_native"` emits ONNX `RMSNormalization` (plus an
  `Add` for Equinox's optional bias) at opset 23 or newer, matching Flax RMSNorm,
  while other modes and older opsets keep the explicit graph.
* **Keep explicit RMSNorm graphs explicit at runtime:** Equinox and Flax RMSNorm
  now square with `Mul` instead of `Pow`, so ONNX Runtime no longer fuses the
  explicit graph into its `SimplifiedLayerNormalization` kernel.
* **Propagate NaN through Slow-Variance GroupNorm:** A group containing NaN
  among otherwise equal values is no longer exported as exactly centered
  zeros; it yields NaN like JAX.
* **Export `jnp.cos` as `Cos` below float64:** Keep the `Sin(x + π/2)`
  workaround only for float64, which ONNX Runtime's `Cos` kernel lacks, so
  float32 rotary embeddings no longer lose accuracy to the shifted argument.


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
