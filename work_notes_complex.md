# Complex Bilinear Lowering (Four-Real Strategy)

## Goal
Deliver ONNX-compatible complex support for all bilinear primitives mentioned in issue #127 by lowering JAX complex tensors into real-valued graphs that operate on a trailing size-2 channel (`[..., 2]`). Only the four-real expansion will be implemented at this stage; no Gauss optimisation knob will be exposed.

## Plan
- Establish a shared helper layer (reuse `ensure_packed_real_pair`, bias handling, shape stamping) so every plugin consistently converts complex tensors into real/imag pairs and re-packs the results.
- Audit existing complex-capable paths (`add`, `sub`, `mul`, `div`, `lax.fft`, `jnp.fft`, …) and align them with the shared helpers so the representation is uniform across the converter.
- Keep `docs/dev_guides/complex_numbers.md` in sync with any helper or flow changes so plugin authors have an up-to-date reference.
- **MatMul / Einsum**: lower complex inputs by splitting the trailing channel, run real-valued ops while keeping the complex channel out of contraction labels, and repack results.
- **Conv / ConvTranspose**: mirror the split/compute/repack flow, inserting layout transposes only when the backend requires channel-first tensors.
- **Conjugation utilities**: ensure `Conj` lowers to `PackComplex(Real(X), Neg(Imag(X)))` so plugins can normalise inputs before the four-real expansion.
- **Broadcast & attribute preservation**: carry original op attributes (stride, dilation, etc.) and rely on ONNX broadcasting semantics for shape alignment.
- **Testing**: extend each plugin’s functional tests in parallel with the lowerings to cover broadcasted shapes, FP16/BF16, real-only operands, and conjugate/transpose cases.

## Notes (2025-10-30)
- Elementwise add/sub/mul/div now share the helper surface (`ensure_packed_real_pair`, `cast_real_tensor`, `split_packed_real_imag`, `pack_real_imag_pair`) so they exercise the canonical four-real flow.
- FFT (`lax.fft`, `jnp.fft`) relies on `pack_native_complex`; keep its shape stamping consistent with any helper updates.
- Dev guide mirrors the helper surface today; we’ll expand “Supported operations” after MatMul/Einsum/Conv/Conj land.

## Implementation Status
- Shared helper review: in progress — `_complex_utils.py` mapped and new split/pack helpers (`split_packed_real_imag`, `pack_real_imag_pair`) added for reuse.
- Existing complex ops helper alignment: in progress — `lax.mul`/`lax.div` now use the shared helpers; FFT path still to verify.
- Complex numbers dev-guide sync: not started.
- MatMul / Einsum four-real lowering: not started.
- Conv / ConvTranspose four-real lowering: not started.
- Conjugation helper alignment: not started.
- Plugin functional tests covering broadcast/precision matrix: not started.
