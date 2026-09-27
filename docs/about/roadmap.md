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

### **jax2onnx 0.17.0**

* **Preserve transposed-convolution geometry:** Lower input-dilated
  `lax.conv_general_dilated` (used by `eqx.nn.ConvTranspose`,
  `nnx.ConvTranspose`, `jax.lax.conv_transpose`, and `nnx.Conv` with
  `input_dilation`) to ONNX `ConvTranspose` with the input dilation as its
  strides, a group-aware kernel layout, and an explicit output `Pad` where JAX
  pads beyond the kernel's reach, so exported shapes and values match JAX; fail
  export explicitly for strided input dilation and complex transposed
  convolutions, and reject `batch_group_count > 1` for all
  `lax.conv_general_dilated` lowerings.
* **Make GELU exports opset-aware:** Emit ONNX `Gelu` only for opset 20 and
  newer, and lower `jax.nn.gelu` and `nnx.gelu` below opset 20 to the exact
  (`Erf`) or tanh-approximate formula so the graph validates at the requested
  opset.
* **Keep JIT lowering compatible with JAX 0.11.1+:** Create fresh JAX
  variables through the compatibility layer, which supports the newer `Var`
  constructor and preserves quantization metadata on older JAX releases.
* **Restore gradient exports on JAX 0.11.2:** Lower its new
  `lax.one_minus_square` primitive as `(1 + x) * (1 - x)` to retain
  precision near `|x| = 1`. This fixes the generated CI cases for `tanh`,
  `acos`, `asin`, and `atanh` gradients. Cast the scalar constant to the
  input dtype so the lowering also works in float32 `lax.scan` bodies when
  double-precision export is enabled.
* **Behavior change: explicit normalization graphs by default.**
  `normalization_mode="auto"` selects the representation with the best
  reproducible accuracy in a defined test environment, then permits a native
  operator only when it meets the same fixed acceptance bounds. Export uses the
  resulting plugin policy; it does not benchmark each model. The current choice
  is explicit graphs for Flax NNX/Linen GroupNorm and Equinox/Flax NNX/Linen
  RMSNorm and LayerNorm.
  Fixed comparative bounds currently cover selected float32 LayerNorm cases on
  ONNX Runtime CPU; equivalent GroupNorm and RMSNorm bounds remain future work.
  `"prefer_native"` requests an eligible native operator, while
  `"force_decomposed"` requests the explicit graph.
* **Add precision-faithful LayerNorm export:** Equinox, Flax NNX, and Flax Linen
  LayerNorm now honor `normalization_mode`. The explicit graph follows the
  framework's statistics: two-pass variance with a constant-row safeguard
  for Equinox and slow-variance NNX, or Flax's clamped fast variance. Linen
  slow-variance LayerNorm continues to trace JAX in every mode. The tested
  explicit paths stay explicit under ONNX Runtime CPU optimizations; the native
  `LayerNormalization` path has higher error on the fixed outlier-input cases.
  Exports below opset 17 no longer emit an operator the opset does not define.
* **Add native Equinox RMSNorm export:** `eqx.nn.RMSNorm` now honors
  `normalization_mode`; `"prefer_native"` emits ONNX `RMSNormalization` (plus an
  `Add` for Equinox's optional bias) at opset 23 or newer, matching Flax
  NNX/Linen RMSNorm. Other modes and older opsets keep the explicit graph.
* **Keep explicit RMSNorm graphs explicit at runtime:** Equinox and Flax
  NNX/Linen RMSNorm now square with `Mul` instead of `Pow`. In the tested ONNX
  Runtime CPU configuration, the checked explicit paths avoid fusion into
  `SimplifiedLayerNormalization`.
* **Propagate nonfinite values through explicit normalization:** Slow-variance
  LayerNorm and GroupNorm no longer turn otherwise constant rows or groups
  containing NaN or infinity into zeros; they preserve JAX's nonfinite results.
* **Export `jnp.cos` as `Cos` below float64:** Keep the `Sin(x + π/2)`
  workaround only for float64, which ONNX Runtime's `Cos` kernel lacks, so
  float32 rotary embeddings no longer lose accuracy to the shifted argument.
* **Guard normalization accuracy in CI:** Add CPU checks on Python 3.12/JAX
  0.10 and Python 3.13/JAX 0.11 with fixed accuracy bounds for selected
  LayerNorm and cosine cases; add BF16 capability coverage and static/dynamic
  NNX decoder examples for the explicit `auto` graph. Keep CI enabled for
  documentation changes and release tags, allow manual runs, and record the
  tested commit, dependency versions, runner image, and CPU hardware.
* **Refresh locked Python dependencies:** Keep JAX/JAXLIB 0.10.2 and Flax
  0.12.8 for Python 3.11/3.12, and resolve JAX/JAXLIB 0.11.2 and Flax
  0.12.10 for Python 3.13+; update ONNX to 1.23.0 and ONNX Runtime to 1.30.0.
  The dependency guide now separates latest upstream releases from the
  `onnxruntime-web` 1.29.0 lock used in smoke tests.
* **Update test, documentation, and CI tools:** Upgrade optional test Torch to
  2.13.0, mkdocstrings to 1.0.6 while dropping the direct Griffe bound, and
  the resolved Ruff to 0.16.9; update Ruff pre-commit to 0.16.5 and
  `actions/setup-node` to 7.0.0.



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
