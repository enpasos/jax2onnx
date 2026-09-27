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
* **Improve cosine accuracy across dtypes:** Export `jnp.cos` as native
  `Cos` below float64 so float32 rotary embeddings avoid phase-shift error.
  Float64 `jnp.cos` and `lax.cos` use `1 - 2 * Sin(x / 2)²`, avoiding the
  large-angle error of `Sin(x + π/2)` while retaining compatibility with ONNX
  Runtime 1.24.1, whose CPU kernels lack double-precision `Cos`. Regression
  tests cover large arguments, cosine zeros, nonfinite values, and symbolic
  shapes with runtime optimizations enabled and disabled.
* **Preserve FP16 matrix-product types:** Keep the actual operand dtype on
  `jnp.matmul` and `jnp.dot` outputs, and cast operands when an explicit
  `preferred_element_type` requests wider accumulation. Add CPU runtime
  checks for matrix products, NNX Linear, and bias-plus-GELU with static and
  dynamic batches, including ONNX Runtime's FP32 fallback for FP16 operations.
* **Measure decoder normalization tradeoffs:** Add a reproducible CPU benchmark
  comparing `auto` and `prefer_native` on identical tiny-decoder weights and
  inputs, with static and symbolic exports, JAX parity, optimized operator
  counts, and runtime/hardware metadata. Update the native-kernel description
  for ONNX Runtime 1.30's AVX2 implementation; retain the fixed accuracy gates.
* **Guard normalization accuracy in CI:** Add CPU checks on Python 3.12/JAX
  0.10 and Python 3.13/JAX 0.11 with fixed accuracy bounds for selected
  LayerNorm and cosine cases; add BF16 capability coverage and static/dynamic
  NNX decoder examples for the explicit `auto` graph. Keep CI enabled for
  documentation changes and release tags, allow manual runs, and record the
  tested commit, dependency versions, runner image, and CPU hardware.




## Past Versions

See [Past Versions](past_versions.md) for the full release archive.
