# Roadmap

## Planned

* Broaden coverage of JAX, Flax NNX/Linen, and Equinox components.
* Expand SotA example support for vision and language models.
* Improve support for **physics-based simulations**.
 


## Upcoming Version

### **jax2onnx 0.12.5**

* **Add MaxDiffusion example coverage:** Export a lightweight MaxDiffusion UNet test matrix for selected SDXL-family configs and align the Flax/JAX export path for `LogicallyPartitioned` parameters, hashable conv padding, and `jax.image.resize(..., method="nearest", antialias=True)`.


## Current Version

### **jax2onnx 0.12.4**

* **Leverage native ONNX Attention for Equinox MHA:** Export `eqx.nn.MultiheadAttention` via ONNX `Attention` when targeting opset >= 23, while keeping the existing fallback path for older opsets.
* **Add Flax NNX grouped-query attention support:** Export `nnx.dot_product_attention` and `nnx.MultiHeadAttention` when `num_heads != num_kv_heads`, using native ONNX `Attention` on opset >= 23 and a compatible grouped-K/V expansion fallback on older opsets.
* **Add Exclusive Self Attention example coverage:** Export XSA-style attention blocks on top of the existing attention lowering, preserving native ONNX `Attention` on opset >= 23 and the standard fallback path on older opsets.
* **Fix Flax NNX conv special padding export:** Align `nnx.Conv` with upstream Flax handling for `REFLECT`, `CIRCULAR`, and `CAUSAL` padding so these modes export correctly through `to_onnx`.
 
## Past Versions

See [Past Versions](past_versions.md) for the full release archive.
