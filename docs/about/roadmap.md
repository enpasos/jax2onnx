# Roadmap

## Planned

* Broaden coverage of JAX, Flax NNX/Linen, and Equinox components.
* Expand SotA example support for vision and language models.
* Improve support for **physics-based simulations**.
* Support for **[MaxDiffusion](https://github.com/AI-Hypercomputer/maxdiffusion)**.

## Current Version

### **jax2onnx 0.11.2** – Native IR cloning & code cleanup

* **Native `onnx-ir` graph cloning:**
  Replaced the custom `ir_clone` implementation with `onnx-ir`'s native `Graph.clone()` method, improving maintainability and leveraging upstream validation (PR #162, #163).

* **Unified IR pass infrastructure:**
  Streamlined the optimization pipeline by adopting standard `onnx-ir` passes and removing redundant custom pass logic (PR #151).

* **MaxDiffusion robustness:**
  Fixed environment-dependent crashes (`UnboundLocalError`) and corrected type annotations in the MaxDiffusion plugin stack.



## Past Versions

See [Past Versions](past_versions.md) for the full release archive.
