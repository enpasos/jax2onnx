# Roadmap

## Planned

* Broaden coverage of JAX, Flax NNX/Linen, and Equinox components.
* Expand SotA example support for vision and language models.
* Improve support for **physics-based simulations**.
 

## Upcoming Version


### **jax2onnx 0.13.0**

* Restore JAX 0.10.0 compatibility by updating GPT-OSS examples to the new `jnp.clip(min=..., max=...)` API and tolerating legacy `scan` payload params removed upstream.
* Broaden `jax.nn.dot_product_attention` compatibility with grouped-query attention and unbatched `TNH` inputs by reusing the shared IR KV-head expansion path already exercised by Flax NNX.
* Reuse `onnx-ir` 0.2.1 helpers for graph rewrites, value lookup, and constant handling, including bulk `rename_values`, `replace_all_uses_with`, `create_value_mapping`, and `get_const_tensor`.



## Current Version
 

### **jax2onnx 0.12.5**

* **Add MaxDiffusion example coverage:** Export a lightweight MaxDiffusion UNet test matrix for selected SDXL-family configs, add an optional `maxdiffusion` dependency group, and wire the SotA checks into `run_all_checks.sh` with a pinned upstream checkout flow.
* **Strengthen numeric validation for resize exports:** Stop skipping numeric validation for small static opset 9 linear resizes by lowering exact resize weights through `MatMul`, and align `jax.image.resize(..., method="nearest", antialias=True)` with JAX by treating nearest-neighbour antialiasing as a no-op.
* **Fix symbolic and dynamic index-path exports:** Preserve symbolic dimension origins in the IR and unblock dynamic `lax.slice`, `gather`, `iota`, and `rev` lowerings, including reflect-padding exports for dynamic `nnx.Conv` paths.

 
 
## Past Versions

See [Past Versions](past_versions.md) for the full release archive.
