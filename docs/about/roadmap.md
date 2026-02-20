# Roadmap

## Planned

* Broaden coverage of JAX, Flax NNX/Linen, and Equinox components.
* Expand SotA example support for vision and language models.
* Improve support for **physics-based simulations**.
* Support for **[MaxDiffusion](https://github.com/AI-Hypercomputer/maxdiffusion)**.

## Upcoming Version

### **jax2onnx 0.12.1 (Upcoming)** - Coverage expansion, AD rules, and conversion robustness

* **ONNX operator quick wins across `lax`/`jnp`:**
  Added and stabilized many lowerings in the quick-win track, including unary math/trig/hyperbolic ops, boolean/bitwise paths, reduction patterns, GlobalPool variants, and additional shape/window-related conversions.

* **Neural-network and random primitive growth:**
  Added/expanded support for activations and NN utilities (including HardSigmoid/HardSwish, SiLU/Swish, logsumexp/log_softmax, one_hot, standardize) plus random/export paths such as `jax.random.normal` and improved categorical handling with symbolic batch shaping.

* **Autodiff and transform-rule coverage:**
  Extended primitive autodiff coverage (including shape-oriented and activation paths), and implemented custom primitive transform-rule work tracked in issues **#190** and **#191**.

* **Converter and runtime stability fixes:**
  Hardened multiple regression-prone paths (reshape/flatten safety, reduce-op coverage, ORT-sensitive test behavior, bool-sort/TopK compatibility, and numeric-validation guardrails) to keep generated graphs valid and deterministic.

* **Usability and interface improvements:**
  Added customizable ONNX graph input/output naming in `to_onnx(...)`, and continued maintenance of coverage matrices and mapping docs used for release planning.

## Current Version

### **jax2onnx 0.12.0** - Layout controls, opset 23 defaults, and regression hardening

* **NCHW boundary layout support:**
  Added `inputs_as_nchw` / `outputs_as_nchw` for `to_onnx(...)` and `allclose(...)`, with layout-optimization docs/tests and transpose-cleanup improvements for Conv-heavy graphs (PR #164, #172).

* **Depth-to-space and residual-stack coverage:**
  Added `dm_pix.depth_to_space` lowering to ONNX `DepthToSpace` and expanded NNX regression examples/tests for depth-to-space and nested residual groups (PR #167, #175, #176).

* **Primitive and IR improvements:**
  Added `jax.numpy.mean` lowering to `ReduceMean`; fixed symbolic `dim_as_value` handling; and stabilized dynamic reshape folding used by CLIP/MaxText exports (PR #170, #171, #179).

* **ONNX opset 23 path for attention models:**
  Added opset >= 23 RotaryEmbedding/Attention support and made opset 23 the default in `to_onnx(...)` (PR #177).

* **Gather/scatter regression fixes:**
  Fixed scatter-add broadcast window handling and issue #52 lowering edge cases; fixed gather indexing and `vmap(dynamic_slice_in_dim)` gather lowering regressions (PR #181, #183, #184).

* **Compatibility refresh:**
  Expanded tested Python versions to 3.11-3.14 and updated runtime dependency floors (`onnx`, `onnxruntime`, `dm-pix`) for the new paths.

## Past Versions

See [Past Versions](past_versions.md) for the full release archive.
