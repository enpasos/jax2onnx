# Roadmap

## Planned

* Broaden coverage of JAX, Flax NNX/Linen, and Equinox components.
* Expand SotA example support for vision and language models.
* Improve support for **physics-based simulations**.
 

## Upcoming Version


### **jax2onnx 0.13.0**

* **Refresh compatibility floors:** Move the documented stack to JAX `0.10.0`, Equinox `0.13.7`, and `onnx-ir` `0.2.1`; update GPT-OSS examples to the new `jnp.clip(min=..., max=...)` API and tolerate legacy `scan` payload params removed upstream.
* **Broaden attention export coverage:** Extend `jax.nn.dot_product_attention` support for grouped-query attention and unbatched `TNH` inputs by reusing the shared IR KV-head expansion path already exercised by Flax NNX.
* **Lean harder on `onnx-ir` helpers:** Replace local graph/value plumbing with `onnx-ir` helpers for bulk `rename_values`, `replace_all_uses_with`, `create_value_mapping`, constant tensor reads, tensor/NumPy dtype mapping, scalar/tensor attribute conversion, and IRBuilder mapping attributes.
* **Tighten dtype fidelity:** Preserve explicit dtype requests across casts, array creation, `full`, `eye`, `linspace`, `arange`, index outputs, bitwise casts, shifts, and `iota`, backed by exact-dtype converter tests.
* **Complete generated component coverage:** Bring generated JAX LAX, JAX NumPy, Flax Linen/NNX, and Equinox NN coverage matrices to zero missing entries, with guard tests that fail on future drift.
* **Expand JAX NumPy lowering coverage:** Add or harden coverage for `searchsorted`, `digitize`, `histogram`, `histogram2d`, `histogramdd`, `interp`, `nancumprod`, `frexp`, `ldexp`, `spacing`, 1D convolution reuse, `packbits`, `argpartition`, `lexsort`, `polyval`, `intersect1d`, triangular index helpers, `roots`, and static linear `polyfit`.
* **Add small static linear algebra lowerings:** Cover static small-matrix/system paths for `jnp.linalg.inv`, `solve`, `tensorsolve`, and `tensorinv`, including generated tests and coverage-doc refreshes.
* **Improve LAX and control-flow coverage:** Add `lax.linalg.ormqr`, strengthen gather/scatter/sort/top-k/split/reduce helper coverage, fix single-output split-style lowerings, and keep custom IO-name rewrites on the `onnx-ir` bulk rename path.
* **Improve coverage documentation quality:** Add stale-doc checks for coverage generators, snapshot/zero-missing guards, ONNX metadata-vs-lowering asymmetry guards, and an ONNX operator coverage matrix with next-action buckets for the remaining open operators.



## Current Version
 

### **jax2onnx 0.12.5**

* **Add MaxDiffusion example coverage:** Export a lightweight MaxDiffusion UNet test matrix for selected SDXL-family configs, add an optional `maxdiffusion` dependency group, and wire the SotA checks into `run_all_checks.sh` with a pinned upstream checkout flow.
* **Strengthen numeric validation for resize exports:** Stop skipping numeric validation for small static opset 9 linear resizes by lowering exact resize weights through `MatMul`, and align `jax.image.resize(..., method="nearest", antialias=True)` with JAX by treating nearest-neighbour antialiasing as a no-op.
* **Fix symbolic and dynamic index-path exports:** Preserve symbolic dimension origins in the IR and unblock dynamic `lax.slice`, `gather`, `iota`, and `rev` lowerings, including reflect-padding exports for dynamic `nnx.Conv` paths.

 
 
## Past Versions

See [Past Versions](past_versions.md) for the full release archive.
