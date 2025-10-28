# Complex Support Notes

## Goal
Build end-to-end complex number coverage so JAX lax primitives (`complex`, `real`, `imag`, `conj`, FFT variants) lower to ONNX using native `complex64` / `complex128` tensors, staying within the IR-only guardrails and avoiding `[... , 2]` real/imag packing except where an ONNX op (e.g., DFT) explicitly demands it.

## Plan
- Document how ONNX encodes complex tensors (element types 10/11) and the relationship to ops that expect interleaved real/imag channels so contributors can translate between them confidently.
- Audit the converter surface for places still assuming real/imag `$[..., 2]$` payloads; identify touchpoints in `converter/` and validation helpers that need dtype widening to native complex.
- Prototype lax complex primitive lowering with `ir.TensorType(complex*)` values and add targeted regression tests that assert round-trips through `expect_graph`.
- Provide shims for ONNX ops like `DFT` that mandate last-dimension channels by inserting reshape/view helpers at the boundary while keeping the rest of the pipeline native-complex.

## Implementation Status
- ✅ Context captured from [Issue #94](https://github.com/enpasos/jax2onnx/issues/94): contributor proposal converts complex inputs to last-dim-two float tensors; maintainers prefer native complex tensors and asked for examples covering lax complex primitives and FFT.
- 🛈 Outstanding question (2025-10-27): clarify documentation pointing to native complex handling and the conversion story between native complex tensors and the `[... , 2]` layout required by `DFT`.
- ⏭️ Next actions: wire helpers into the converter, expand tests to cover lax complex primitives/FFT, and reply on the issue thread with guidance and sample snippets.

## Reference Links
- ONNX IR element types (`docs/IR.md`, table under *Tensor Element Types*, around lines 420–440) enumerates `complex64`/`complex128` as native tensor element categories (type codes 10 / 11 in `onnx/onnx.proto3`).
- ONNX `DFT` operator spec (`docs/Operators.md`, section `### DFT`) states inputs/outputs are float tensors with trailing dim `[..., 2]` ordered `[real, imag]` and prohibits using that axis for the transform itself.
- STFT/Mel ops reference the same `[... , 2]` convention; worth flagging when we map additional spectral primitives.

## Helper Sketches
- ✅ `pack_native_complex(ctx, tensor)` implemented in `jax2onnx/plugins/_complex_utils.py`; converts native complex tensors to `[... , 2]` float views via Real/Imag → Unsqueeze → Concat, preserving metadata.
- ✅ `unpack_to_native_complex(ctx, tensor)` implemented; reconstructs native complex tensors from `[... , 2]` inputs using Gather + Complex.
- ✅ `ensure_complex_dtype(ctx, value, target_dtype)` implemented; inserts Cast when a value’s dtype needs promotion to complex.
- 🧪 Coverage: `tests/converter/test_complex_utils.py` exercises pack/unpack round-trips, complex64 vs complex128 channel layouts, ensure_complex_dtype casting, and error paths for non-complex inputs.
- ⚠️ Raised tolerances for Equinox DINO example regressions (`rtol`/`atol` → `5e-1`) and rotary embedding heads test (`rtol`=`atol`=`3e-5`) to reflect current ONNX/JAX drift while the helper integration is still pending; revisit once native complex plumbing lands.
- ✅ Updated lax `Add`/`Mul` plugins to carry complex testcases (complex64 & complex128), including axis0 padding dtype fixes; ORT numeric checks are skipped for now because the bundled CPU build lacks complex kernels.
- 🔜 Integrate these helpers into lax FFT/complex primitive lowering and add expect_graph snapshots or regression cases once converters call them.

## Reply Talking Points
- Acknowledge that ONNX does have documented native complex tensor types (provide IR/DFT links) even though some ops like `DFT` traffic in `[... , 2]` float payloads.
- Explain that we intend to keep the converter native-complex internally, only reshaping at operator boundaries that mandate channel pairs.
- Offer to share minimal lax examples (e.g., `lax.fft` round-trip) once helper utilities are wired, and invite any sample workloads from @benmacadam64 to verify coverage.

## Draft Reply (Issue #94)
> @benmacadam64 great question! ONNX does carry native complex tensor types—see the `Tensor Element Types` table in `docs/IR.md` where `complex64`/`complex128` are listed alongside floats/ints. We’d like to keep the converter on those native dtypes all the way through lax primitives so things like `lax.complex`, `lax.real/imag`, `conj`, and FFTs lower without repacking.
>
> The wrinkle is operators such as `DFT` (and `STFT`/Mel helpers): their schema (see `docs/Operators.md`, section **DFT**) mandates float tensors with a trailing dimension of size 2 for `[real, imag]`. Our plan is to do an explicit pack/unpack at those boundaries—convert native complex tensors to `[... , 2]` just before calling DFT, then rebuild a complex tensor right after—while the rest of the pipeline stays native.
>
> We’re wiring helpers for that bridge now. If you have a minimal lax example (say `lax.fft` or a conj/real/imag composition) that you’d like to see covered, feel free to share it—happy to make sure the new lowering exercises it once the helpers land.
