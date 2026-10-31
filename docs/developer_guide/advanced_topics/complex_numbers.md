# Complex Numbers in jax2onnx

This guide explains how `jax2onnx` handles complex tensors while staying within the ONNX specification, and how plugin authors should interact with the shared helper utilities.

## Why we need a strategy

ONNX has only limited native complex-tensor support across operators. Most
arithmetic, shape, and control-flow primitives expect real-valued tensors. The
main complex-aware operator we rely on is `DFT`, which represents complex inputs
and outputs as real tensors whose trailing dimension packs the real and imaginary
channels (`[..., 2]`).

To stay portable across runtimes we represent every complex tensor as a real tensor with that trailing size-2 channel. Conversion never emits `Real`, `Imag`, or other custom operators—everything is expressed in terms of standard ONNX ops on real tensors.

## Helper surface (`plugins/_complex_utils.py`)

| Helper | Purpose |
| --- | --- |
| `ensure_complex_dtype(ctx, value, target_dtype)` | Cast native complex tensors between `COMPLEX64` and `COMPLEX128` when a lowering needs an exact complex dtype. |
| `pack_native_complex(ctx, tensor)` | Reinterpret a native `complex64/complex128` value as a packed real tensor (`[..., 2]`). Handles double-precision upgrades automatically when `enable_double_precision=True`. |
| `is_packed_complex_tensor(value)` | Detect whether a value already uses the packed representation. |
| `ensure_packed_real_pair(ctx, value, *, name_hint)` | Return `(packed_tensor, base_dtype)` for both native complex inputs and already-packed tensors. Raises if the value is neither. |
| `cast_real_tensor(ctx, value, target_dtype, *, name_hint)` | Insert a `Cast` when the packed tensor must move between `FLOAT` and `DOUBLE` representations. |
| `resolve_common_real_dtype(lhs, rhs)` | Pick the shared real dtype (`FLOAT` or `DOUBLE`) for binary complex operations. |
| `split_packed_real_imag(ctx, value, base_dtype, *, prefix)` | Gather the trailing real and imaginary channels from a packed tensor, returning two real tensors. |
| `pack_real_imag_pair(ctx, real, imag, base_dtype, *, name_hint)` | Unsqueeze matching real/imag tensors and concatenate them back into the packed `[... , 2]` representation. |
| `conjugate_packed_tensor(ctx, value, base_dtype, *, prefix)` | Flip the sign of the imaginary channel while preserving shape metadata, producing the complex conjugate of a packed tensor. |
| `coerce_dim_values(dims)` | Normalise shape metadata so `onnx_ir` can stamp symbolic dimensions and integers consistently. |
| `unpack_to_native_complex(...)` | Convert a packed tensor back to a native complex value (used rarely, e.g. when handing results back to JAX in test harnesses). |

These helpers take care of dtype metadata, `IRBuilder` stamping, and axis bookkeeping so individual plugins only need to express the real-valued arithmetic. New complex-aware plugins should rely on them instead of ad hoc `Gather` / `Reshape` sequences so every lowering shares the same representation.

## Private SOURCE broadcast contract

Static SOURCE `broadcast_in_dim` at opsets 23/26 preserves the declared
complex64/complex128 width and final real/imaginary channel. Logical axis mappings
may permute axes; typed Transpose, Reshape and Expand keep the channel last.
An original public call that native JAX forwards unchanged remains an alias;
the converter does not invent a broadcast for that trace.

Selected official Python 3.11/3.13 checks pass with ORT 1.31 CPU, optimizations
disabled and enabled: 56 numerical copy/capture cases, 12 rejection controls and
two broadcast-admission controls per family. Captures include noncontiguous views,
a registered plain-owner Function and an independent real trailing-two sibling.
The two product-source admission controls do not qualify complex Mul numerics.
General Mul, complete B–F, minimum ORT and public SOURCE transition remain open;
public LEGACY remains the default.

## Private SOURCE conjugation contract

Static same-width `lax.conj` at opsets 23/26 validates the native `input_dtype`,
weakness and complete physical pair before allocating or emitting nodes. The
existing Gather/Neg/Concat helper keeps the real component and negates the
imaginary component without a width conversion or floating product. Original
`jnp.conj` traces retain their native dependencies.

Selected official 3.11/3.13 checks with ORT 1.31 pass 68 numerical cases and 27
rejection controls per family, including captured views and real/imaginary
projections. The independent oracle flips only the imaginary sign bit; comparison
requires equal NaN masks and exact non-NaN component words, including signed
zeros and subnormals. NaN payloads and signs are recorded without a preservation
claim. Cross-width conversion, real-input `lax.conj`, symbolic complex scopes,
general Mul/AD, minimum ORT and public SOURCE transition remain separate work.

## Private SOURCE static tensor-layout contract

Static same-width complex `copy`, `reshape`, `transpose`, `squeeze`,
`concatenate` and `rev` at opsets 23/26 share a strict physical-pair contract.
Canonical native equations, static shape/axis laws, dtype and weakness are checked
before allocation; every input must already have the declared real width and
last size-two channel. Logical permutations append that channel, reshape keeps
literal zero extents, and reverse builds integer Range/Gather indices without
floating arithmetic or a Python-sized index vector. Concatenation checks every
input and the native all-input weakness rule. Real trailing-two tensors retain
their separate path.

The same ordered 343 checks pass on each official 3.11/3.13 family: 182 numerical
cases and 161 rejection/bypass controls. Original eager/first/warm and both ORT
1.31 CPU optimization modes match the independent copy/layout oracles in physical
dtype/shape, NaN masks and exact non-NaN words. Coverage includes empty/scalar
shapes, signed zeros, infinities, subnormals, noncontiguous inline/named captures,
real siblings and repeated outputs. NaN payload/sign preservation, symbolic
complex layout, width conversion, Mul/AD, minimum ORT, complete B–F and public
SOURCE transition remain outside this finite contract.

## Private SOURCE static selection contract

Static same-width complex `lax.select`, `lax.select_n` and original `jnp.where`
traces at opsets 23/26 select complete real/imaginary pairs. The dedicated native
`select_n` admission checks all case and selector descriptors before allocation.
Cases have equal logical shape and width; a Boolean selector has at most two
cases, while an integer selector must satisfy the native precondition
`0 <= which < number_of_cases`. Scalar selectors and selectors matching the
logical shape are supported. Result weakness depends on every case, excluding
the selector. Unsqueeze/Concat/GatherElements/Squeeze copy both component words
without floating selection or arithmetic.

The same ordered 286 checks pass on official 3.11/3.13 with ORT 1.31 CPU:
234 numerical cases and 52 contract/representation controls. Original
eager/first/warm and both target optimization modes match independent selected
word oracles in physical dtype/shape, component NaN masks and exact non-NaN
words. Coverage includes all eight integer selector widths in valid domains,
scalar/empty/trailing-two shapes, weak values and noncontiguous inline/named
captures with real and index siblings. NaN payloads/signs, out-of-domain integer
selectors, mixed widths, symbolic complex scopes, Mul/Cast/AD, minimum ORT,
complete B–F and public transition are not qualified by these checks.

## Supported operations

- **Construction and projection** (`lax.complex`, `lax.real`, `lax.imag`):
  - `lax.complex` packs matching real and imaginary tensors into the shared
    trailing-channel representation.
  - `lax.real` and `lax.imag` gather the corresponding channel for packed or
    native complex inputs; real inputs pass through the real path.

- **Elementwise arithmetic** (`lax.add`, `lax.sub`, `lax.mul`, `lax.div`):
  - Detection logic looks at the JAX avals and value metadata. When a complex value is involved we normalise operands through `ensure_packed_real_pair`, align their base dtype (`FLOAT` ↔ `DOUBLE`) via `resolve_common_real_dtype` / `cast_real_tensor`, run the real-valued formulas, and use `pack_real_imag_pair` to rebuild the packed output.
  - Outputs inherit the packed representation and expose real metadata (`tensor(float)` / `tensor(double)` with trailing `2`).

- **FFT pipeline** (`lax.fft`, `jnp.fft` for FFT/IFFT/RFFT):
  - Complex inputs (FFT/IFFT) are packed, reshaped if needed, and lowered to ONNX `DFT` with `inverse` / `onesided` flags. Real inputs (RFFT) receive the trailing channel before invoking `DFT`.
  - `IRFFT` currently requires explicit `fft_lengths`. The implementation reconstructs the missing half of the spectrum, flips the imaginary channel, and runs a forward packed `DFT` before gathering the real component.
  - For `jnp.fft`, metadata-only primitives reuse the same `lax.fft` lowering
    for canonical 1-D forms (`axis=-1`, optional length). `jnp.fft.irfft`
    follows the same packed reconstruction path as `lax.fft` and currently needs
    static length information.

- **MatMul / Einsum family** (`jax.lax.dot_general`, `jnp.matmul`):
  - Operands are normalised via `ensure_packed_real_pair` and cast to a shared real dtype. The real/imag channels are split with `split_packed_real_imag`, the real-valued contraction (`Einsum` or `MatMul`) runs four times, and `pack_real_imag_pair` stitches the results back together.
  - For `dot_general`, both the batched MatMul fast-path and general `Einsum` lowering share the same helper plumbing so the trailing complex channel is never part of the contraction labels.
  - For `jnp.matmul`, the four-real flow lowers to four ONNX `MatMul` nodes before recombining; broadcasting and vector/matrix promotion match the real path.

- **Convolutions** (`jax.lax.conv_general_dilated`):
  - Inputs and kernels flow through `ensure_packed_real_pair`, are cast to a shared dtype, and have the complex channel split before any layout transposes.
  - Each of the four real-valued paths runs through the existing Conv lowering (after layout canonicalisation). Outputs are optionally transposed back to the requested layout and re-packed with `pack_real_imag_pair`.

- **Conjugation** (`jax.lax.conj`, `jnp.conj`):
  - Normalise packed/native complex inputs with `ensure_packed_real_pair`, call `conjugate_packed_tensor` to negate the imaginary channel, and return the packed output. Real inputs bypass through an `Identity`.

- **Complex-valued outputs** (`lax.linalg.eig`, `jnp.roots`):
  - Plugins that naturally produce complex results should emit the same packed
    representation and use `_complex_utils` helpers for dtype and shape metadata.

- **Tests**: regression coverage is generated from plugin metadata and exercised
  through `tests/primitives/test_lax.py`, `tests/primitives/test_jnp.py`, and
  focused helper tests under `tests/extra_tests/converter/test_complex_utils.py`.

## Authoring new plugins with complex inputs

1. **Detect complex flows early.** Inspect JAX avals (`var.aval.dtype`) or existing value metadata. If the operand is complex, call `ensure_packed_real_pair(...)` to normalise it.
2. **Work in real space.** Once packed, treat the tensors as real arrays. Use `resolve_common_real_dtype` and `cast_real_tensor` to reconcile dtypes before running arithmetic.
3. **Stamp shapes and metadata.** Most helpers already stamp values, but if you build new tensors (e.g., concatenations) remember to call `_stamp_type_and_shape` with `coerce_dim_values(...)` so the ONNX graph carries explicit metadata.
4. **Return packed outputs.** Results should remain in `[... , 2]` form. Do not attempt to reintroduce native complex ONNX tensors—runtimes will reject them.
5. **Tests + docs.** Add `expect_graph` snippets alongside the plugin metadata and cover complex variants in the autogenerated test suites.

## Current limitations

- The packed representation is the only supported ONNX-side complex convention.
  Do not introduce native complex ONNX tensors for general arithmetic paths.
- Not every JAX primitive that accepts complex inputs has a complex-aware
  lowering. Several linear-algebra paths still reject complex inputs explicitly;
  add support case by case using the same packed-real recipe.
- `IRFFT` needs static onesided length information. If a new frontend path hides
  that length, add metadata/tests before claiming coverage.
- Convolution transpose / deconvolution paths should reuse the same four-real
  structure if complex support is added there.


## Potential optimizations

- **Gaussian 3-multiply strategy:** at the moment we always lower via the straightforward four-real expansion. The `complex_strategy` knob isn’t wired yet; it could be exposed as (`four_real` by default, `gauss` as an opt-in). The Gauss variant would replace the four real multiplies `[(a_r b_r), (a_i b_i), (a_r b_i), (a_i b_r)]` with three multiplies plus a few adds:  
  1. `p1 = a_r * b_r`  
  2. `p2 = a_i * b_i`  
  3. `p3 = (a_r + a_i) * (b_r + b_i)`  
     then reconstruct `real = p1 - p2` and `imag = p3 - p1 - p2`.  
  We’ll evaluate this once we can guarantee backend support and have regression coverage for the numerical trade-offs.
