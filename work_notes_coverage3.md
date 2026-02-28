# Work Notes: JAX LAX Coverage Checklist (v3)

## Scope
- Source list: all operators linked on the official `jax.lax` docs page: `https://docs.jax.dev/en/latest/jax.lax.html`
- Coverage signal: `jax_doc` metadata + `jaxpr_primitive` registrations in `jax2onnx/plugins/**/*.py`.

## This Pass
- [x] Added `jax.lax.approx_top_k` plugin (`jax2onnx/plugins/jax/lax/approx_top_k.py`).
- [x] Added `jax.lax.cbrt` plugin (`jax2onnx/plugins/jax/lax/cbrt.py`).
- [x] Added `jax.lax.clz` plugin (`jax2onnx/plugins/jax/lax/clz.py`).
- [x] Added `jax.lax.bessel_i0e` plugin (`jax2onnx/plugins/jax/lax/bessel_i0e.py`).
- [x] Added `jax.lax.bessel_i1e` plugin (`jax2onnx/plugins/jax/lax/bessel_i1e.py`).
- [x] Added `jax.lax.betainc` plugin (`jax2onnx/plugins/jax/lax/betainc.py`).
- [x] Added `jax.lax.custom_linear_solve` plugin (`jax2onnx/plugins/jax/lax/custom_linear_solve.py`).
- [x] Added `jax.lax.cumlogsumexp` plugin (`jax2onnx/plugins/jax/lax/cumlogsumexp.py`).
- [x] Added `jax.lax.cummax` plugin (`jax2onnx/plugins/jax/lax/cummax.py`).
- [x] Added `jax.lax.cummin` plugin (`jax2onnx/plugins/jax/lax/cummin.py`).
- [x] Added `jax.lax.erf_inv` plugin (`jax2onnx/plugins/jax/lax/erf_inv.py`).
- [x] Added `jax.lax.digamma` plugin (`jax2onnx/plugins/jax/lax/digamma.py`).
- [x] Added `jax.lax.lgamma` plugin (`jax2onnx/plugins/jax/lax/lgamma.py`).
- [x] Added `jax.lax.nextafter` plugin (`jax2onnx/plugins/jax/lax/nextafter.py`).
- [x] Added `jax.lax.igamma` plugin (`jax2onnx/plugins/jax/lax/igamma.py`).
- [x] Added `jax.lax.igammac` plugin (`jax2onnx/plugins/jax/lax/igamma.py`).
- [x] Added `jax.lax.igamma_grad_a` plugin (`jax2onnx/plugins/jax/lax/igamma.py`).
- [x] Added `jax.lax.linalg.lu_pivots_to_permutation` plugin (`jax2onnx/plugins/jax/lax/lu_pivots_to_permutation.py`).
- [x] Added `jax.lax.linalg.tridiagonal_solve` plugin (`jax2onnx/plugins/jax/lax/tridiagonal_solve.py`).
- [x] Added `jax.lax.linalg.tridiagonal` plugin (`jax2onnx/plugins/jax/lax/tridiagonal.py`).
- [x] Added `jax.lax.linalg.symmetric_product` plugin (`jax2onnx/plugins/jax/lax/symmetric_product.py`).
- [x] Added `jax.lax.linalg.cholesky` plugin (`jax2onnx/plugins/jax/lax/cholesky.py`).
- [x] Added `jax.lax.linalg.cholesky_update` plugin (`jax2onnx/plugins/jax/lax/cholesky_update.py`).
- [x] Added `jax.lax.linalg.eig` plugin (`jax2onnx/plugins/jax/lax/eig.py`).
- [x] Added `jax.lax.linalg.eigh` plugin (`jax2onnx/plugins/jax/lax/eigh.py`).
- [x] Added `jax.lax.linalg.householder_product` plugin (`jax2onnx/plugins/jax/lax/householder_product.py`).
- [x] Added `jax.lax.linalg.hessenberg` plugin (`jax2onnx/plugins/jax/lax/hessenberg.py`).
- [x] Added `jax.lax.linalg.lu` plugin (`jax2onnx/plugins/jax/lax/lu.py`).
- [x] Added `jax.lax.linalg.qr` plugin (`jax2onnx/plugins/jax/lax/qr.py`).
- [x] Added `jax.lax.linalg.schur` plugin (`jax2onnx/plugins/jax/lax/schur.py`).
- [x] Added `jax.lax.linalg.svd` plugin (`jax2onnx/plugins/jax/lax/svd.py`).
- [x] Added `jax.lax.optimization_barrier` plugin (`jax2onnx/plugins/jax/lax/optimization_barrier.py`).
- [x] Added `jax.lax.reduce_window` plugin (`jax2onnx/plugins/jax/lax/reduce_window.py`).
- [x] Added `jax.lax.population_count` plugin (`jax2onnx/plugins/jax/lax/population_count.py`).
- [x] Added `jax.lax.polygamma` plugin (`jax2onnx/plugins/jax/lax/polygamma.py`).
- [x] Added `jax.lax.reduce` plugin (`jax2onnx/plugins/jax/lax/reduce.py`).
- [x] Added `jax.lax.reduce_precision` plugin (`jax2onnx/plugins/jax/lax/reduce_precision.py`).
- [x] Added `jax.lax.rng_bit_generator` plugin (`jax2onnx/plugins/jax/lax/rng_bit_generator.py`).
- [x] Added `jax.lax.rng_uniform` plugin (`jax2onnx/plugins/jax/lax/rng_uniform.py`).
- [x] Added `jax.lax.scatter_sub` plugin (`jax2onnx/plugins/jax/lax/scatter_sub.py`).
- [x] Added `jax.lax.shift_right_arithmetic` plugin (`jax2onnx/plugins/jax/lax/shift_right_arithmetic.py`).
- [x] Added `jax.lax.zeta` plugin (`jax2onnx/plugins/jax/lax/zeta.py`).
- [x] Expanded `jax.lax.linalg.qr` plugin to support `full_matrices=True` (in addition to reduced mode), with new tall/wide full-mode testcases.
- [x] Expanded `jax.lax.linalg.eigh` plugin to support static `subset_by_index` slicing for `1x1`/`2x2` outputs.
- [x] Expanded `jax.lax.linalg.lu_pivots_to_permutation` plugin to support batched rank-2 pivots.
- [x] Expanded `jax.lax.linalg.cholesky` plugin to support rank-3 batched inputs.

## Snapshot
- Total docs operators: `201`
- Covered (direct plugin): `137`
- Covered (via alias primitive): `14`
- Composite/helper (no standalone plugin expected): `33`
- Out of scope (distributed/token/host): `17`
- Missing primitive plugins: `0`
- Missing `lax.linalg` plugins: `0`

## Priority Gap Queue
- No quick-win candidates currently marked missing by this heuristic.

## Full Checklist
Legend: `covered`, `covered_indirect`, `composite`, `out_of_scope`, `missing`, `missing_linalg`.

| Checklist | jax.lax Operator | Status | Modules (signals) | Notes |
|:--|:--|:--|:--|:--|
| [x] | `abs` | `covered` | `jax/lax/abs` | Direct plugin coverage. |
| [x] | `acos` | `covered` | `jax/lax/acos` | Direct plugin coverage. |
| [x] | `acosh` | `covered` | `jax/lax/acosh` | Direct plugin coverage. |
| [x] | `add` | `covered` | `jax/lax/add` | Direct plugin coverage. |
| [x] | `after_all` | `out_of_scope` | `-` | Distributed/token/host path; currently out of converter scope. |
| [x] | `all_gather` | `out_of_scope` | `-` | Distributed/token/host path; currently out of converter scope. |
| [x] | `all_to_all` | `out_of_scope` | `-` | Distributed/token/host path; currently out of converter scope. |
| [x] | `approx_max_k` | `covered_indirect` | `jax/lax/approx_top_k` | Covered via `approx_top_k` primitive. |
| [x] | `approx_min_k` | `covered_indirect` | `jax/lax/approx_top_k` | Covered via `approx_top_k` primitive. |
| [x] | `argmax` | `covered` | `jax/lax/argmax` | Direct plugin coverage. |
| [x] | `argmin` | `covered` | `jax/lax/argmin` | Direct plugin coverage. |
| [x] | `asin` | `covered` | `jax/lax/asin` | Direct plugin coverage. |
| [x] | `asinh` | `covered` | `jax/lax/asinh` | Direct plugin coverage. |
| [x] | `associative_scan` | `composite` | `-` | Composite/helper API; no standalone primitive plugin. |
| [x] | `atan` | `covered` | `jax/lax/atan` | Direct plugin coverage. |
| [x] | `atan2` | `covered` | `jax/lax/atan2` | Direct plugin coverage. |
| [x] | `atanh` | `covered` | `jax/lax/atanh` | Direct plugin coverage. |
| [x] | `axis_index` | `out_of_scope` | `-` | Distributed/token/host path; currently out of converter scope. |
| [x] | `axis_size` | `out_of_scope` | `-` | Distributed/token/host path; currently out of converter scope. |
| [x] | `batch_matmul` | `covered_indirect` | `jax/lax/dot_general` | Covered via `dot_general` primitive. |
| [x] | `bessel_i0e` | `covered` | `jax/lax/bessel_i0e` | Direct plugin coverage. |
| [x] | `bessel_i1e` | `covered` | `jax/lax/bessel_i1e` | Direct plugin coverage. |
| [x] | `betainc` | `covered_indirect` | `jax/lax/betainc` | Covered via `regularized_incomplete_beta` primitive. |
| [x] | `bitcast_convert_type` | `covered` | `jax/lax/bitcast_convert_type` | Direct plugin coverage. |
| [x] | `bitwise_and` | `covered` | `jax/lax/and` | Direct plugin coverage. |
| [x] | `bitwise_not` | `covered_indirect` | `jax/lax/bitwise_not` | Covered via `not` primitive. |
| [x] | `bitwise_or` | `covered_indirect` | `jax/lax/or` | Covered via `or` primitive. |
| [x] | `bitwise_xor` | `covered_indirect` | `jax/lax/xor` | Covered via `xor` primitive. |
| [x] | `broadcast` | `composite` | `-` | Composite/helper API; no standalone primitive plugin. |
| [x] | `broadcast_in_dim` | `covered` | `jax/lax/broadcast_in_dim` | Direct plugin coverage. |
| [x] | `broadcast_shapes` | `composite` | `-` | Composite/helper API; no standalone primitive plugin. |
| [x] | `broadcast_to_rank` | `composite` | `-` | Composite/helper API; no standalone primitive plugin. |
| [x] | `broadcasted_iota` | `composite` | `-` | Composite/helper API; no standalone primitive plugin. |
| [x] | `cbrt` | `covered` | `jax/lax/cbrt` | Direct plugin coverage. |
| [x] | `ceil` | `covered` | `jax/lax/ceil` | Direct plugin coverage. |
| [x] | `clamp` | `covered` | `jax/lax/clamp` | Direct plugin coverage. |
| [x] | `clz` | `covered` | `jax/lax/clz` | Direct plugin coverage. |
| [x] | `collapse` | `composite` | `-` | Composite/helper API; no standalone primitive plugin. |
| [x] | `complex` | `covered` | `jax/lax/complex` | Direct plugin coverage. |
| [x] | `composite` | `composite` | `-` | Composite/helper API; no standalone primitive plugin. |
| [x] | `concatenate` | `covered` | `jax/lax/concatenate` | Direct plugin coverage. |
| [x] | `cond` | `covered` | `jax/lax/cond` | Direct plugin coverage. |
| [x] | `conj` | `covered` | `jax/lax/conj` | Direct plugin coverage. |
| [x] | `conv` | `covered_indirect` | `jax/lax/conv` | Covered via `conv_general_dilated` primitive. |
| [x] | `conv_dimension_numbers` | `composite` | `-` | Composite/helper API; no standalone primitive plugin. |
| [x] | `conv_general_dilated` | `covered` | `jax/lax/conv` | Direct plugin coverage. |
| [x] | `conv_general_dilated_local` | `composite` | `-` | Composite/helper API; no standalone primitive plugin. |
| [x] | `conv_general_dilated_patches` | `composite` | `-` | Composite/helper API; no standalone primitive plugin. |
| [x] | `conv_transpose` | `composite` | `-` | Composite/helper API; no standalone primitive plugin. |
| [x] | `conv_with_general_padding` | `composite` | `-` | Composite/helper API; no standalone primitive plugin. |
| [x] | `convert_element_type` | `covered` | `jax/lax/convert_element_type` | Direct plugin coverage. |
| [x] | `cos` | `covered` | `jax/lax/cos` | Direct plugin coverage. |
| [x] | `cosh` | `covered` | `jax/lax/cosh` | Direct plugin coverage. |
| [x] | `cumlogsumexp` | `covered` | `jax/lax/cumlogsumexp` | Direct plugin coverage. |
| [x] | `cummax` | `covered` | `jax/lax/cummax` | Direct plugin coverage. |
| [x] | `cummin` | `covered` | `jax/lax/cummin` | Direct plugin coverage. |
| [x] | `cumprod` | `covered` | `jax/lax/cumprod` | Direct plugin coverage. |
| [x] | `cumsum` | `covered` | `jax/lax/cumsum` | Direct plugin coverage. |
| [x] | `custom_linear_solve` | `covered` | `jax/lax/custom_linear_solve` | Direct plugin coverage. |
| [x] | `custom_root` | `composite` | `-` | Composite/helper API; no standalone primitive plugin. |
| [x] | `digamma` | `covered` | `jax/lax/digamma` | Direct plugin coverage. |
| [x] | `div` | `covered` | `jax/lax/div` | Direct plugin coverage. |
| [x] | `dot` | `covered_indirect` | `jax/lax/dot_general` | Covered via `dot_general` primitive. |
| [x] | `dot_general` | `covered` | `jax/lax/dot_general` | Direct plugin coverage. |
| [x] | `dynamic_index_in_dim` | `composite` | `-` | Composite/helper API; no standalone primitive plugin. |
| [x] | `dynamic_slice` | `covered` | `jax/lax/dynamic_slice` | Direct plugin coverage. |
| [x] | `dynamic_slice_in_dim` | `composite` | `-` | Composite/helper API; no standalone primitive plugin. |
| [x] | `dynamic_update_index_in_dim` | `composite` | `-` | Composite/helper API; no standalone primitive plugin. |
| [x] | `dynamic_update_slice` | `covered` | `jax/lax/dynamic_update_slice` | Direct plugin coverage. |
| [x] | `dynamic_update_slice_in_dim` | `composite` | `-` | Composite/helper API; no standalone primitive plugin. |
| [x] | `empty` | `composite` | `-` | Composite/helper API; no standalone primitive plugin. |
| [x] | `eq` | `covered` | `jax/lax/eq` | Direct plugin coverage. |
| [x] | `erf` | `covered` | `jax/lax/erf` | Direct plugin coverage. |
| [x] | `erf_inv` | `covered` | `jax/lax/erf_inv` | Direct plugin coverage. |
| [x] | `erfc` | `covered` | `jax/lax/erfc` | Direct plugin coverage. |
| [x] | `exp` | `covered` | `jax/lax/exp` | Direct plugin coverage. |
| [x] | `exp2` | `covered` | `jax/lax/exp2` | Direct plugin coverage. |
| [x] | `expand_dims` | `composite` | `-` | Composite/helper API; no standalone primitive plugin. |
| [x] | `expm1` | `covered` | `jax/lax/expm1` | Direct plugin coverage. |
| [x] | `fft` | `covered` | `jax/lax/fft` | Direct plugin coverage. |
| [x] | `floor` | `covered` | `jax/lax/floor` | Direct plugin coverage. |
| [x] | `fori_loop` | `covered_indirect` | `jax/lax/fori_loop` | Covered via `lax.fori_loop` primitive. |
| [x] | `full` | `composite` | `-` | Composite/helper API; no standalone primitive plugin. |
| [x] | `full_like` | `composite` | `-` | Composite/helper API; no standalone primitive plugin. |
| [x] | `gather` | `covered` | `jax/lax/gather` | Direct plugin coverage. |
| [x] | `ge` | `covered` | `jax/lax/ge` | Direct plugin coverage. |
| [x] | `gt` | `covered` | `jax/lax/gt` | Direct plugin coverage. |
| [x] | `igamma` | `covered` | `jax/lax/igamma` | Direct plugin coverage. |
| [x] | `igamma_grad_a` | `covered` | `jax/lax/igamma` | Direct plugin coverage. |
| [x] | `igammac` | `covered` | `jax/lax/igamma` | Direct plugin coverage. |
| [x] | `imag` | `covered` | `jax/lax/imag` | Direct plugin coverage. |
| [x] | `index_in_dim` | `composite` | `-` | Composite/helper API; no standalone primitive plugin. |
| [x] | `index_take` | `composite` | `-` | Composite/helper API; no standalone primitive plugin. |
| [x] | `integer_pow` | `covered` | `jax/lax/integer_pow` | Direct plugin coverage. |
| [x] | `iota` | `covered` | `jax/lax/iota` | Direct plugin coverage. |
| [x] | `is_finite` | `covered` | `jax/lax/is_finite` | Direct plugin coverage. |
| [x] | `le` | `covered` | `jax/lax/le` | Direct plugin coverage. |
| [x] | `lgamma` | `covered` | `jax/lax/lgamma` | Direct plugin coverage. |
| [x] | `linalg.SvdAlgorithm` | `composite` | `-` | Composite/helper API; no standalone primitive plugin. |
| [x] | `linalg.cholesky` | `covered` | `jax/lax/cholesky` | Direct plugin coverage. |
| [x] | `linalg.cholesky_update` | `covered` | `jax/lax/cholesky_update` | Direct plugin coverage. |
| [x] | `linalg.eig` | `covered` | `jax/lax/eig` | Direct plugin coverage. |
| [x] | `linalg.eigh` | `covered` | `jax/lax/eigh` | Direct plugin coverage. |
| [x] | `linalg.hessenberg` | `covered` | `jax/lax/hessenberg` | Direct plugin coverage. |
| [x] | `linalg.householder_product` | `covered` | `jax/lax/householder_product` | Direct plugin coverage. |
| [x] | `linalg.lu` | `covered` | `jax/lax/lu` | Direct plugin coverage. |
| [x] | `linalg.lu_pivots_to_permutation` | `covered` | `jax/lax/lu_pivots_to_permutation` | Direct plugin coverage. |
| [x] | `linalg.qdwh` | `composite` | `-` | Composite/helper API; no standalone primitive plugin. |
| [x] | `linalg.qr` | `covered` | `jax/lax/qr` | Direct plugin coverage. |
| [x] | `linalg.schur` | `covered` | `jax/lax/schur` | Direct plugin coverage. |
| [x] | `linalg.svd` | `covered` | `jax/lax/svd` | Direct plugin coverage. |
| [x] | `linalg.symmetric_product` | `covered` | `jax/lax/symmetric_product` | Direct plugin coverage. |
| [x] | `linalg.triangular_solve` | `covered` | `jax/lax/triangular_solve` | Direct plugin coverage. |
| [x] | `linalg.tridiagonal` | `covered` | `jax/lax/tridiagonal` | Direct plugin coverage. |
| [x] | `linalg.tridiagonal_solve` | `covered` | `jax/lax/tridiagonal_solve` | Direct plugin coverage. |
| [x] | `log` | `covered` | `jax/lax/log` | Direct plugin coverage. |
| [x] | `log1p` | `covered` | `jax/lax/log1p` | Direct plugin coverage. |
| [x] | `logistic` | `covered` | `jax/lax/logistic` | Direct plugin coverage. |
| [x] | `lt` | `covered` | `jax/lax/lt` | Direct plugin coverage. |
| [x] | `map` | `composite` | `-` | Composite/helper API; no standalone primitive plugin. |
| [x] | `max` | `covered` | `jax/lax/max` | Direct plugin coverage. |
| [x] | `min` | `covered` | `jax/lax/min` | Direct plugin coverage. |
| [x] | `mul` | `covered` | `jax/lax/mul` | Direct plugin coverage. |
| [x] | `ne` | `covered` | `jax/lax/ne` | Direct plugin coverage. |
| [x] | `neg` | `covered` | `jax/lax/neg` | Direct plugin coverage. |
| [x] | `nextafter` | `covered` | `jax/lax/nextafter` | Direct plugin coverage. |
| [x] | `optimization_barrier` | `covered` | `jax/lax/optimization_barrier` | Direct plugin coverage. |
| [x] | `pad` | `covered` | `jax/lax/pad` | Direct plugin coverage. |
| [x] | `platform_dependent` | `out_of_scope` | `-` | Distributed/token/host path; currently out of converter scope. |
| [x] | `pmax` | `out_of_scope` | `-` | Distributed/token/host path; currently out of converter scope. |
| [x] | `pmean` | `out_of_scope` | `-` | Distributed/token/host path; currently out of converter scope. |
| [x] | `pmin` | `out_of_scope` | `-` | Distributed/token/host path; currently out of converter scope. |
| [x] | `polygamma` | `covered` | `jax/lax/polygamma` | Direct plugin coverage. |
| [x] | `population_count` | `covered` | `jax/lax/population_count` | Direct plugin coverage. |
| [x] | `pow` | `covered` | `jax/lax/pow` | Direct plugin coverage. |
| [x] | `ppermute` | `out_of_scope` | `-` | Distributed/token/host path; currently out of converter scope. |
| [x] | `precv` | `out_of_scope` | `-` | Distributed/token/host path; currently out of converter scope. |
| [x] | `psend` | `out_of_scope` | `-` | Distributed/token/host path; currently out of converter scope. |
| [x] | `pshuffle` | `out_of_scope` | `-` | Distributed/token/host path; currently out of converter scope. |
| [x] | `psum` | `out_of_scope` | `-` | Distributed/token/host path; currently out of converter scope. |
| [x] | `psum_scatter` | `out_of_scope` | `-` | Distributed/token/host path; currently out of converter scope. |
| [x] | `pswapaxes` | `out_of_scope` | `-` | Distributed/token/host path; currently out of converter scope. |
| [x] | `ragged_all_to_all` | `out_of_scope` | `-` | Distributed/token/host path; currently out of converter scope. |
| [x] | `ragged_dot` | `covered_indirect` | `jax/lax/ragged_dot_general` | Covered via `ragged_dot_general` primitive. |
| [x] | `ragged_dot_general` | `covered` | `jax/lax/ragged_dot_general` | Direct plugin coverage. |
| [x] | `random_gamma_grad` | `composite` | `-` | Composite/helper API; no standalone primitive plugin. |
| [x] | `real` | `covered` | `jax/lax/real` | Direct plugin coverage. |
| [x] | `reciprocal` | `covered_indirect` | `jax/lax/integer_pow` | Covered via `integer_pow` primitive. |
| [x] | `reduce` | `covered` | `jax/lax/reduce` | Direct plugin coverage. |
| [x] | `reduce_and` | `covered` | `jax/lax/reduce_and` | Direct plugin coverage. |
| [x] | `reduce_max` | `covered` | `jax/lax/reduce_max` | Direct plugin coverage. |
| [x] | `reduce_min` | `covered` | `jax/lax/reduce_min` | Direct plugin coverage. |
| [x] | `reduce_or` | `covered` | `jax/lax/reduce_or` | Direct plugin coverage. |
| [x] | `reduce_precision` | `covered` | `jax/lax/reduce_precision` | Direct plugin coverage. |
| [x] | `reduce_prod` | `covered` | `jax/lax/reduce_prod` | Direct plugin coverage. |
| [x] | `reduce_sum` | `covered` | `jax/lax/reduce_sum` | Direct plugin coverage. |
| [x] | `reduce_window` | `covered` | `jax/lax/reduce_window` | Direct plugin coverage. |
| [x] | `reduce_xor` | `covered` | `jax/lax/reduce_xor` | Direct plugin coverage. |
| [x] | `rem` | `covered` | `jax/lax/rem` | Direct plugin coverage. |
| [x] | `reshape` | `covered` | `jax/lax/reshape` | Direct plugin coverage. |
| [x] | `rev` | `covered` | `jax/lax/rev` | Direct plugin coverage. |
| [x] | `rng_bit_generator` | `covered` | `jax/lax/rng_bit_generator` | Direct plugin coverage. |
| [x] | `rng_uniform` | `covered` | `jax/lax/rng_uniform` | Direct plugin coverage. |
| [x] | `round` | `covered` | `jax/lax/round` | Direct plugin coverage. |
| [x] | `rsqrt` | `covered` | `jax/lax/rsqrt` | Direct plugin coverage. |
| [x] | `scaled_dot` | `composite` | `-` | Composite/helper API; no standalone primitive plugin. |
| [x] | `scan` | `covered` | `jax/lax/scan` | Direct plugin coverage. |
| [x] | `scatter` | `covered` | `jax/lax/scatter` | Direct plugin coverage. |
| [x] | `scatter_add` | `covered` | `jax/lax/scatter_add` | Direct plugin coverage. |
| [x] | `scatter_apply` | `composite` | `-` | Composite/helper API; no standalone primitive plugin. |
| [x] | `scatter_max` | `covered` | `jax/lax/scatter_max` | Direct plugin coverage. |
| [x] | `scatter_min` | `covered` | `jax/lax/scatter_min` | Direct plugin coverage. |
| [x] | `scatter_mul` | `covered` | `jax/lax/scatter_mul` | Direct plugin coverage. |
| [x] | `scatter_sub` | `covered` | `jax/lax/scatter_sub` | Direct plugin coverage. |
| [x] | `select` | `covered` | `jax/lax/select` | Direct plugin coverage. |
| [x] | `select_n` | `covered` | `jax/lax/select_n` | Direct plugin coverage. |
| [x] | `shift_left` | `covered` | `jax/lax/shift_left` | Direct plugin coverage. |
| [x] | `shift_right_arithmetic` | `covered` | `jax/lax/shift_right_arithmetic` | Direct plugin coverage. |
| [x] | `shift_right_logical` | `covered` | `jax/lax/shift_right_logical` | Direct plugin coverage. |
| [x] | `sign` | `covered` | `jax/lax/sign` | Direct plugin coverage. |
| [x] | `sin` | `covered` | `jax/lax/sin` | Direct plugin coverage. |
| [x] | `sinh` | `covered` | `jax/lax/sinh` | Direct plugin coverage. |
| [x] | `slice` | `covered` | `jax/lax/slice` | Direct plugin coverage. |
| [x] | `slice_in_dim` | `composite` | `-` | Composite/helper API; no standalone primitive plugin. |
| [x] | `sort` | `covered` | `jax/lax/sort` | Direct plugin coverage. |
| [x] | `sort_key_val` | `composite` | `-` | Composite/helper API; no standalone primitive plugin. |
| [x] | `split` | `covered` | `jax/lax/split` | Direct plugin coverage. |
| [x] | `sqrt` | `covered` | `jax/lax/sqrt` | Direct plugin coverage. |
| [x] | `square` | `covered` | `jax/lax/square` | Direct plugin coverage. |
| [x] | `squeeze` | `covered` | `jax/lax/squeeze` | Direct plugin coverage. |
| [x] | `stop_gradient` | `covered` | `jax/lax/stop_gradient` | Direct plugin coverage. |
| [x] | `sub` | `covered` | `jax/lax/sub` | Direct plugin coverage. |
| [x] | `switch` | `composite` | `-` | Composite/helper API; no standalone primitive plugin. |
| [x] | `tan` | `covered` | `jax/lax/tan` | Direct plugin coverage. |
| [x] | `tanh` | `covered` | `jax/lax/tanh` | Direct plugin coverage. |
| [x] | `tile` | `composite` | `-` | Composite/helper API; no standalone primitive plugin. |
| [x] | `top_k` | `covered` | `jax/lax/top_k` | Direct plugin coverage. |
| [x] | `transpose` | `covered` | `jax/lax/transpose` | Direct plugin coverage. |
| [x] | `while_loop` | `covered_indirect` | `jax/lax/while_loop` | Covered via `while` primitive. |
| [x] | `with_sharding_constraint` | `covered_indirect` | `jax/lax/sharding_constraint` | Covered via `sharding_constraint` primitive. |
| [x] | `zeta` | `covered` | `jax/lax/zeta` | Direct plugin coverage. |

## Next Steps
1. Implement remaining quick-win primitive gaps from the queue above.
2. For each new plugin, add metadata testcases and regenerate tests (`scripts/generate_tests.py`).
3. Re-run this script after each batch to keep this checklist current.
