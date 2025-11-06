# Complex Numbers in jax2onnx

This guide explains how `jax2onnx` handles complex tensors while staying within the ONNX specification, and how plugin authors should interact with the shared helper utilities.

## Why we need a strategy

ONNX (up to opset 21) does not provide native `tensor(complex*)` types for most operators. Arithmetic, shape, and control-flow primitives all expect real-valued tensors. The only complex-aware operator we rely on is `DFT`, which represents complex inputs and outputs as real tensors whose trailing dimension packs the real and imaginary channels (`[..., 2]`).

To stay portable across runtimes we represent every complex tensor as a real tensor with that trailing size-2 channel. Conversion never emits `Real`, `Imag`, or other custom operators—everything is expressed in terms of standard ONNX ops on real tensors.

## Helper surface (`plugins/_complex_utils.py`)

| Helper | Purpose |
| --- | --- |
| `pack_native_complex(ctx, tensor)` | Reinterpret a native `complex64/complex128` value as a packed real tensor (`[..., 2]`). Handles double-precision upgrades automatically when `enable_double_precision=True`.
| `is_packed_complex_tensor(value)` | Detect whether a value already uses the packed representation.
| `ensure_packed_real_pair(ctx, value)` | Return `(packed_tensor, base_dtype)` for both native complex inputs and already-packed tensors. Raises if the value is neither.
| `cast_real_tensor(ctx, value, target_dtype)` | Insert a `Cast` when the packed tensor must move between `FLOAT` and `DOUBLE` representations.
| `resolve_common_real_dtype(lhs, rhs)` | Pick the shared real dtype (`FLOAT` or `DOUBLE`) for binary complex operations.
| `split_packed_real_imag(ctx, value, base_dtype)` | Gather the trailing real and imaginary channels from a packed tensor, returning two real tensors.
| `pack_real_imag_pair(ctx, real, imag, base_dtype)` | Unsqueeze matching real/imag tensors and concatenate them back into the packed `[... , 2]` representation.
| `conjugate_packed_tensor(ctx, value, base_dtype)` | Flip the sign of the imaginary channel while preserving shape metadata, producing the complex conjugate of a packed tensor.
| `coerce_dim_values(dims)` | Normalise shape metadata so `onnx_ir` can stamp symbolic dimensions and integers consistently.
| `unpack_to_native_complex(...)` | Convert a packed tensor back to a native complex value (used rarely, e.g. when handing results back to JAX in test harnesses).

These helpers take care of dtype metadata, `IRBuilder` stamping, and axis bookkeeping so individual plugins only need to express the real-valued arithmetic. New complex-aware plugins should rely on them instead of ad hoc `Gather` / `Reshape` sequences so every lowering shares the same representation.

## Supported operations

- **Elementwise arithmetic** (`lax.add`, `lax.sub`, `lax.mul`, `lax.div`):
  - Detection logic looks at the JAX avals and value metadata. When a complex value is involved we normalise operands through `ensure_packed_real_pair`, align their base dtype (`FLOAT` ↔ `DOUBLE`) via `resolve_common_real_dtype` / `cast_real_tensor`, run the real-valued formulas, and use `pack_real_imag_pair` to rebuild the packed output.
  - Outputs inherit the packed representation and expose real metadata (`tensor(float)` / `tensor(double)` with trailing `2`).

- **FFT pipeline** (`lax.fft`, `jnp.fft` for FFT/IFFT/RFFT):
  - Complex inputs (FFT/IFFT) are packed, reshaped if needed, and lowered to ONNX `DFT` with `inverse` / `onesided` flags. Real inputs (RFFT) receive the trailing channel before invoking `DFT`.
  - `IRFFT` currently requires explicit `fft_lengths`. The implementation reconstructs the missing half of the spectrum, flips the imaginary channel, and runs a forward packed `DFT` before gathering the real component.
  - For `jnp.fft`, we register metadata-only primitives that reuse the same lowering when the call matches the canonical 1-D form (`axis=-1`, optional length). `irfft` keeps the original NumPy behaviour until we finish integrating the dtype-safe reconstruction path.

- **MatMul / Einsum family** (`jax.lax.dot_general`, `jnp.matmul`):
  - Operands are normalised via `ensure_packed_real_pair` and cast to a shared real dtype. The real/imag channels are split with `split_packed_real_imag`, the real-valued contraction (`Einsum` or `MatMul`) runs four times, and `pack_real_imag_pair` stitches the results back together.
  - For `dot_general`, both the batched MatMul fast-path and general `Einsum` lowering share the same helper plumbing so the trailing complex channel is never part of the contraction labels.
  - For `jnp.matmul`, the four-real flow lowers to four ONNX `MatMul` nodes before recombining; broadcasting and vector/matrix promotion match the real path.

- **Convolutions** (`jax.lax.conv_general_dilated`):
  - Inputs and kernels flow through `ensure_packed_real_pair`, are cast to a shared dtype, and have the complex channel split before any layout transposes.
  - Each of the four real-valued paths runs through the existing Conv lowering (after layout canonicalisation). Outputs are optionally transposed back to the requested layout and re-packed with `pack_real_imag_pair`.

- **Conjugation** (`jax.lax.conj`, `jnp.conj`):
  - Normalise packed/native complex inputs with `ensure_packed_real_pair`, call `conjugate_packed_tensor` to negate the imaginary channel, and return the packed output. Real inputs bypass through an `Identity`.

- **Tests**: regression coverage lives under `tests/primitives/test_lax.py::Test_fft`, `Test_add`, `Test_sub`, `Test_mul`, `Test_div`, and `tests/primitives/test_jnp.py::Test_fft/ifft/rfft`.

## Authoring new plugins with complex inputs

1. **Detect complex flows early.** Inspect JAX avals (`var.aval.dtype`) or existing value metadata. If the operand is complex, call `ensure_packed_real_pair(...)` to normalise it.
2. **Work in real space.** Once packed, treat the tensors as real arrays. Use `resolve_common_real_dtype` and `cast_real_tensor` to reconcile dtypes before running arithmetic.
3. **Stamp shapes and metadata.** Most helpers already stamp values, but if you build new tensors (e.g., concatenations) remember to call `_stamp_type_and_shape` with `coerce_dim_values(...)` so the ONNX graph carries explicit metadata.
4. **Return packed outputs.** Results should remain in `[... , 2]` form. Do not attempt to reintroduce native complex ONNX tensors—runtimes will reject them.
5. **Tests + docs.** Add `expect_graph` snippets alongside the plugin metadata and cover complex variants in the autogenerated test suites.

## Current limitations

- `jnp.fft.irfft` still delegates to the upstream implementation. The packed-real helpers need a dtype-clean reconstruction path before we can reuse the ONNX `DFT` lowering safely; track this separately if IRFFT metadata is required.
- When new primitives handle complex data (e.g., transcendental ops), follow the same recipe outlined above: convert to packed real tensors, run the pure-real arithmetic, and emit `[... , 2]` outputs.
- Convolution transpose / deconvolution paths are not yet implemented in `jax2onnx`; once a plugin lands it should reuse the same four-real structure (split, canonicalise layout, regroup, repack).
- Additional regression coverage (broadcasted shapes, reduced-precision dtypes such as `bfloat16`, and multi-group convolutions) is staged in `work_notes_complex.md` and will be brought online incrementally.
