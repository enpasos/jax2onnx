# JAX NumPy Coverage Checklist

## Scope
- Source list: all autosummary entries linked on the official `jax.numpy` page: `https://docs.jax.dev/en/latest/jax.numpy.html`
- Coverage signal: `jax_doc` metadata + `jaxpr_primitive` registrations in `jax2onnx/plugins/**/*.py`.

Regenerate with:

```bash
poetry run python scripts/generate_jnp_operator_coverage.py
```

## Snapshot
- Total docs entries: `440`
- Covered (direct plugin): `266`
- Covered (via alias/indirect signal): `15`
- Composite/helper entries: `60`
- Non-functional entries (dtype/type/constants): `34`
- Missing dedicated plugin coverage: `65`

## Priority Gap Queue
- No quick-win candidates currently marked missing by this heuristic.

## Full Checklist
Legend: `covered`, `covered_indirect`, `composite`, `non_functional`, `missing`.

| Checklist | jax.numpy Entry | Status | Modules (signals) | Notes |
|:--|:--|:--|:--|:--|
| [x] | `ComplexWarning` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `abs` | `covered` | `jax/numpy/abs` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `absolute` | `covered_indirect` | `jax/lax/abs, jax/numpy/abs` | Covered via alias `abs`. |
| [x] | `acos` | `covered_indirect` | `jax/numpy/arccos` | Covered via alias `arccos`. |
| [x] | `acosh` | `covered_indirect` | `jax/numpy/arccosh` | Covered via alias `arccosh`. |
| [x] | `add` | `covered` | `jax/numpy/add` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `all` | `covered` | `jax/numpy/all` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `allclose` | `covered` | `jax/numpy/allclose` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `amax` | `covered` | `jax/numpy/amax` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `amin` | `covered` | `jax/numpy/amin` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `angle` | `covered` | `jax/numpy/angle` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `any` | `covered` | `jax/numpy/any` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `append` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `apply_along_axis` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `apply_over_axes` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `arange` | `covered` | `jax/numpy/arange` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `arccos` | `covered` | `jax/numpy/arccos` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `arccosh` | `covered` | `jax/numpy/arccosh` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `arcsin` | `covered` | `jax/numpy/arcsin` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `arcsinh` | `covered` | `jax/numpy/arcsinh` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `arctan` | `covered` | `jax/numpy/arctan` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `arctan2` | `covered` | `jax/numpy/arctan2` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `arctanh` | `covered` | `jax/numpy/arctanh` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `argmax` | `covered` | `jax/numpy/argmax` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `argmin` | `covered` | `jax/numpy/argmin` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `argpartition` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `argsort` | `covered` | `jax/numpy/argsort` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `argwhere` | `covered` | `jax/numpy/argwhere` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `around` | `covered` | `jax/numpy/around` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `array` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `array_equal` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `array_equiv` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `array_repr` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `array_split` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `array_str` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `asarray` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `asin` | `covered_indirect` | `jax/numpy/arcsin` | Covered via alias `arcsin`. |
| [x] | `asinh` | `covered_indirect` | `jax/numpy/arcsinh` | Covered via alias `arcsinh`. |
| [x] | `astype` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `atan` | `covered_indirect` | `jax/numpy/arctan` | Covered via alias `arctan`. |
| [x] | `atan2` | `covered_indirect` | `jax/numpy/arctan2` | Covered via alias `arctan2`. |
| [x] | `atanh` | `covered_indirect` | `jax/numpy/arctanh` | Covered via alias `arctanh`. |
| [x] | `atleast_1d` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `atleast_2d` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `atleast_3d` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `average` | `covered` | `jax/numpy/average` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `bartlett` | `covered` | `jax/numpy/windows` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `bincount` | `covered` | `jax/numpy/bincount` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `bitwise_and` | `covered` | `jax/numpy/bitwise_and` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `bitwise_count` | `covered` | `jax/numpy/bitwise_count` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `bitwise_invert` | `covered_indirect` | `jax/numpy/bitwise_not` | Covered via alias `bitwise_not`. |
| [x] | `bitwise_left_shift` | `covered` | `jax/numpy/left_shift` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `bitwise_not` | `covered` | `jax/numpy/bitwise_not` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `bitwise_or` | `covered` | `jax/numpy/bitwise_or` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `bitwise_right_shift` | `covered` | `jax/numpy/right_shift` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `bitwise_xor` | `covered` | `jax/numpy/bitwise_xor` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `blackman` | `covered` | `jax/numpy/windows` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `block` | `covered` | `jax/numpy/composite_metadata_batch4` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `bool_` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `broadcast_arrays` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `broadcast_shapes` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `broadcast_to` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `c_` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `can_cast` | `covered` | `jax/numpy/can_cast` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `cbrt` | `covered` | `jax/numpy/cbrt` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `cdouble` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `ceil` | `covered` | `jax/numpy/ceil` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `character` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `choose` | `covered` | `jax/numpy/composite_metadata_batch3` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `clip` | `covered` | `jax/numpy/clip` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `column_stack` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `complex128` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `complex64` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `complex_` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `complexfloating` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `compress` | `covered` | `jax/numpy/compress` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `concat` | `covered_indirect` | `jax/lax/concatenate, jax/numpy/concatenate` | Covered via alias `concatenate`. |
| [x] | `concatenate` | `covered` | `jax/numpy/concatenate` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `conj` | `covered` | `jax/numpy/conj` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `conjugate` | `covered_indirect` | `jax/lax/conj, jax/numpy/conj` | Covered via alias `conj`. |
| [ ] | `convolve` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `copy` | `covered` | `jax/numpy/copy` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `copysign` | `covered` | `jax/numpy/copysign` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `corrcoef` | `covered` | `jax/numpy/composite_metadata_batch5` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `correlate` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `cos` | `covered` | `jax/numpy/cos` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `cosh` | `covered` | `jax/numpy/cosh` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `count_nonzero` | `covered` | `jax/numpy/count_nonzero` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `cov` | `covered` | `jax/numpy/composite_metadata` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `cross` | `covered` | `jax/numpy/composite_metadata` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `csingle` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `cumprod` | `covered` | `jax/numpy/cumprod` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `cumsum` | `covered` | `jax/numpy/cumsum` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `cumulative_prod` | `covered_indirect` | `jax/lax/cumprod, jax/numpy/cumprod` | Covered via alias `cumprod`. |
| [x] | `cumulative_sum` | `covered_indirect` | `jax/lax/cumsum, jax/numpy/cumsum` | Covered via alias `cumsum`. |
| [x] | `deg2rad` | `covered` | `jax/numpy/deg2rad` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `degrees` | `covered` | `jax/numpy/degrees` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `delete` | `covered` | `jax/numpy/composite_metadata` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `diag` | `covered` | `jax/numpy/diag` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `diag_indices` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `diag_indices_from` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `diagflat` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `diagonal` | `covered` | `jax/numpy/diagonal` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `diff` | `covered` | `jax/numpy/diff` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `digitize` | `covered` | `jax/numpy/digitize` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `divide` | `covered` | `jax/numpy/divide` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `divmod` | `covered` | `jax/numpy/divmod` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `dot` | `covered` | `jax/numpy/dot` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `double` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `dsplit` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `dstack` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `dtype` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `ediff1d` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `einsum` | `covered` | `jax/numpy/einsum` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `einsum_path` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `empty` | `covered` | `jax/numpy/empty` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `empty_like` | `covered` | `jax/numpy/empty_like` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `equal` | `covered` | `jax/numpy/equal` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `exp` | `covered` | `jax/numpy/exp` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `exp2` | `covered` | `jax/numpy/exp2` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `expand_dims` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `expm1` | `covered` | `jax/numpy/expm1` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `extract` | `covered` | `jax/numpy/extract` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `eye` | `covered` | `jax/numpy/eye` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `fabs` | `covered` | `jax/numpy/fabs` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `fft.fft` | `covered` | `jax/numpy/fft` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `fft.fft2` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `fft.fftfreq` | `covered` | `jax/numpy/composite_metadata` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `fft.fftn` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `fft.fftshift` | `covered` | `jax/numpy/composite_metadata` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `fft.hfft` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `fft.ifft` | `covered` | `jax/numpy/fft` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `fft.ifft2` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `fft.ifftn` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `fft.ifftshift` | `covered` | `jax/numpy/composite_metadata` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `fft.ihfft` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `fft.irfft` | `covered` | `jax/numpy/fft` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `fft.irfft2` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `fft.irfftn` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `fft.rfft` | `covered` | `jax/numpy/fft` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `fft.rfft2` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `fft.rfftfreq` | `covered` | `jax/numpy/composite_metadata` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `fft.rfftn` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `fill_diagonal` | `covered` | `jax/numpy/composite_metadata` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `finfo` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `fix` | `covered` | `jax/numpy/fix` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `flatnonzero` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [ ] | `flexible` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `flip` | `covered` | `jax/numpy/flip` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `fliplr` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `flipud` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [ ] | `float16` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `float32` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `float64` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `float_` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `float_power` | `covered` | `jax/numpy/float_power` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `floating` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `floor` | `covered` | `jax/numpy/floor` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `floor_divide` | `covered` | `jax/numpy/floor_divide` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `fmax` | `covered` | `jax/numpy/fmax` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `fmin` | `covered` | `jax/numpy/fmin` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `fmod` | `covered` | `jax/numpy/fmod` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `frexp` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `from_dlpack` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `frombuffer` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [ ] | `fromfile` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `fromfunction` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `fromiter` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [ ] | `frompyfunc` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `fromstring` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `full` | `covered` | `jax/numpy/full` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `full_like` | `covered` | `jax/numpy/full_like` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `gcd` | `covered` | `jax/numpy/composite_metadata_batch6` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `generic` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `geomspace` | `covered` | `jax/numpy/composite_metadata` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `get_printoptions` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `gradient` | `covered` | `jax/numpy/composite_metadata` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `greater` | `covered` | `jax/numpy/greater` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `greater_equal` | `covered` | `jax/numpy/greater_equal` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `hamming` | `covered` | `jax/numpy/windows` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `hanning` | `covered` | `jax/numpy/windows` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `heaviside` | `covered` | `jax/numpy/heaviside` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `histogram` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `histogram2d` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `histogram_bin_edges` | `covered` | `jax/numpy/composite_metadata` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `histogramdd` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `hsplit` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `hstack` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `hypot` | `covered` | `jax/numpy/hypot` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `i0` | `covered` | `jax/numpy/composite_metadata` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `identity` | `covered` | `jax/numpy/identity` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `iinfo` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `imag` | `covered` | `jax/numpy/imag` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `index_exp` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `indices` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `inexact` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `inner` | `covered` | `jax/numpy/inner` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `insert` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `int16` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `int32` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `int64` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `int8` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `int_` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `integer` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [ ] | `interp` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `intersect1d` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `invert` | `covered` | `jax/numpy/invert` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `isclose` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `iscomplex` | `covered` | `jax/numpy/iscomplex` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `iscomplexobj` | `covered` | `jax/numpy/iscomplexobj` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `isdtype` | `covered` | `jax/numpy/isdtype` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `isfinite` | `covered` | `jax/numpy/isfinite` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `isin` | `covered` | `jax/numpy/isin` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `isinf` | `covered` | `jax/numpy/isinf` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `isnan` | `covered` | `jax/numpy/isnan` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `isneginf` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `isposinf` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `isreal` | `covered` | `jax/numpy/isreal` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `isrealobj` | `covered` | `jax/numpy/isrealobj` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `isscalar` | `covered` | `jax/numpy/isscalar` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `issubdtype` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `iterable` | `covered` | `jax/numpy/iterable` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `ix_` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `kaiser` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `kron` | `covered` | `jax/numpy/kron` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `lcm` | `covered` | `jax/numpy/composite_metadata_batch6` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `ldexp` | `covered` | `jax/numpy/ldexp` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `left_shift` | `covered` | `jax/numpy/left_shift` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `less` | `covered` | `jax/numpy/less` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `less_equal` | `covered` | `jax/numpy/less_equal` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `lexsort` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `linalg.cholesky` | `covered` | `jax/numpy/composite_metadata_batch4` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `linalg.cond` | `covered` | `jax/numpy/composite_metadata_batch4` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `linalg.cross` | `covered` | `jax/numpy/composite_metadata` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `linalg.det` | `covered` | `jax/numpy/linalg_det` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `linalg.diagonal` | `covered` | `jax/numpy/diagonal` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `linalg.eig` | `covered` | `jax/numpy/composite_metadata_batch5` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `linalg.eigh` | `covered` | `jax/numpy/composite_metadata_batch5` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `linalg.eigvals` | `covered` | `jax/numpy/composite_metadata_batch5` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `linalg.eigvalsh` | `covered` | `jax/numpy/composite_metadata_batch5` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `linalg.inv` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `linalg.lstsq` | `covered` | `jax/numpy/composite_metadata_batch5` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `linalg.matmul` | `covered` | `jax/numpy/matmul` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `linalg.matrix_norm` | `covered` | `jax/numpy/composite_metadata_batch3` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `linalg.matrix_power` | `covered` | `jax/numpy/composite_metadata_batch4` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `linalg.matrix_rank` | `covered` | `jax/numpy/composite_metadata_batch4` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `linalg.matrix_transpose` | `covered` | `jax/numpy/matrix_transpose` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `linalg.multi_dot` | `covered` | `jax/numpy/composite_metadata_batch5` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `linalg.norm` | `covered` | `jax/numpy/linalg_norm` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `linalg.outer` | `covered` | `jax/numpy/outer` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `linalg.pinv` | `covered` | `jax/numpy/composite_metadata_batch5` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `linalg.qr` | `covered` | `jax/numpy/composite_metadata_batch4` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `linalg.slogdet` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `linalg.solve` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `linalg.svd` | `covered` | `jax/numpy/composite_metadata_batch5` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `linalg.svdvals` | `covered` | `jax/numpy/composite_metadata_batch4` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `linalg.tensordot` | `covered` | `jax/numpy/composite_metadata_batch2, jax/numpy/composite_metadata_batch3` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `linalg.tensorinv` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `linalg.tensorsolve` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `linalg.trace` | `covered` | `jax/numpy/composite_metadata_batch2, jax/numpy/composite_metadata_batch3` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `linalg.vecdot` | `covered` | `jax/numpy/composite_metadata_batch2, jax/numpy/composite_metadata_batch3` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `linalg.vector_norm` | `covered` | `jax/numpy/composite_metadata_batch3` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `linspace` | `covered` | `jax/numpy/linspace` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `load` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `log` | `covered` | `jax/numpy/log` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `log10` | `covered` | `jax/numpy/log10` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `log1p` | `covered` | `jax/numpy/log1p` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `log2` | `covered` | `jax/numpy/log2` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `logaddexp` | `covered` | `jax/numpy/logaddexp` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `logaddexp2` | `covered` | `jax/numpy/logaddexp2` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `logical_and` | `covered` | `jax/numpy/logical_and` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `logical_not` | `covered` | `jax/numpy/logical_not` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `logical_or` | `covered` | `jax/numpy/logical_or` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `logical_xor` | `covered` | `jax/numpy/logical_xor` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `logspace` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `mask_indices` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `matmul` | `covered` | `jax/numpy/matmul` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `matrix_transpose` | `covered` | `jax/numpy/matrix_transpose` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `matvec` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `max` | `covered` | `jax/numpy/max` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `maximum` | `covered` | `jax/numpy/maximum` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `mean` | `covered` | `jax/numpy/mean` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `median` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `meshgrid` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `mgrid` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `min` | `covered` | `jax/numpy/min` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `minimum` | `covered` | `jax/numpy/minimum` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `mod` | `covered` | `jax/numpy/mod` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `modf` | `covered` | `jax/numpy/modf` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `moveaxis` | `covered` | `jax/numpy/moveaxis` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `multiply` | `covered` | `jax/numpy/multiply` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `nan_to_num` | `covered` | `jax/numpy/nan_to_num` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `nanargmax` | `covered` | `jax/numpy/composite_metadata_batch3` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `nanargmin` | `covered` | `jax/numpy/composite_metadata_batch3` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `nancumprod` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `nancumsum` | `covered` | `jax/numpy/composite_metadata_batch3` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `nanmax` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `nanmean` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `nanmedian` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `nanmin` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `nanpercentile` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `nanprod` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `nanquantile` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `nanstd` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `nansum` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `nanvar` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `ndarray` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `ndarray.at` | `covered` | `jax/numpy/composite_metadata_batch6` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `ndim` | `covered` | `jax/numpy/ndim` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `negative` | `covered` | `jax/numpy/negative` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `nextafter` | `covered` | `jax/numpy/nextafter` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `nonzero` | `covered` | `jax/numpy/nonzero` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `not_equal` | `covered` | `jax/numpy/not_equal` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `number` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [ ] | `object_` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `ogrid` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `ones` | `covered` | `jax/numpy/ones` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `ones_like` | `covered` | `jax/numpy/ones_like` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `outer` | `covered` | `jax/numpy/outer` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `packbits` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `pad` | `covered` | `jax/numpy/pad` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `partition` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `percentile` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `permute_dims` | `covered` | `jax/numpy/permute_dims` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `piecewise` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [ ] | `place` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `poly` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `polyadd` | `covered` | `jax/numpy/composite_metadata_batch4` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `polyder` | `covered` | `jax/numpy/composite_metadata_batch4` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `polydiv` | `covered` | `jax/numpy/composite_metadata_batch4` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `polyfit` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `polyint` | `covered` | `jax/numpy/composite_metadata_batch4` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `polymul` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `polysub` | `covered` | `jax/numpy/composite_metadata_batch4` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `polyval` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `positive` | `covered` | `jax/numpy/positive` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `pow` | `covered_indirect` | `jax/numpy/pow` | Covered via alias `power`. |
| [x] | `power` | `covered` | `jax/numpy/pow` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `printoptions` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `prod` | `covered` | `jax/numpy/prod` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `promote_types` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `ptp` | `covered` | `jax/numpy/ptp` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `put` | `covered` | `jax/numpy/composite_metadata_batch6` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `put_along_axis` | `covered` | `jax/numpy/composite_metadata_batch6` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `quantile` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `r_` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `rad2deg` | `covered` | `jax/numpy/rad2deg` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `radians` | `covered` | `jax/numpy/radians` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `ravel` | `covered` | `jax/numpy/ravel` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `ravel_multi_index` | `covered` | `jax/numpy/composite_metadata_batch3` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `real` | `covered` | `jax/numpy/real` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `reciprocal` | `covered` | `jax/numpy/reciprocal` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `remainder` | `covered` | `jax/numpy/remainder` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `repeat` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `reshape` | `covered` | `jax/numpy/reshape` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `resize` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `result_type` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `right_shift` | `covered` | `jax/numpy/right_shift` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `rint` | `covered` | `jax/numpy/rint` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `roll` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `rollaxis` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [ ] | `roots` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `rot90` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `round` | `covered` | `jax/numpy/round` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `s_` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [ ] | `save` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `savez` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `searchsorted` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `select` | `covered` | `jax/numpy/select` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `set_printoptions` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `setdiff1d` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `setxor1d` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `shape` | `covered` | `jax/numpy/shape` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `sign` | `covered` | `jax/numpy/sign` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `signbit` | `covered` | `jax/numpy/signbit` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `signedinteger` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `sin` | `covered` | `jax/numpy/sin` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `sinc` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `single` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `sinh` | `covered` | `jax/numpy/sinh` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `size` | `covered` | `jax/numpy/size` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `sort` | `covered` | `jax/numpy/sort` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `sort_complex` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `spacing` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `split` | `covered` | `jax/numpy/split` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `sqrt` | `covered` | `jax/numpy/sqrt` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `square` | `covered` | `jax/numpy/square` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `squeeze` | `covered` | `jax/numpy/squeeze` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `stack` | `covered` | `jax/numpy/stack` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `std` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `subtract` | `covered` | `jax/numpy/subtract` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `sum` | `covered` | `jax/numpy/sum` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `swapaxes` | `covered` | `jax/numpy/swapaxes` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `take` | `covered` | `jax/numpy/take` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `take_along_axis` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `tan` | `covered` | `jax/numpy/tan` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `tanh` | `covered` | `jax/numpy/tanh` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `tensordot` | `covered` | `jax/numpy/composite_metadata_batch2, jax/numpy/composite_metadata_batch3` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `tile` | `covered` | `jax/numpy/tile` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `trace` | `covered` | `jax/numpy/composite_metadata_batch2, jax/numpy/composite_metadata_batch3` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `transpose` | `covered` | `jax/numpy/transpose` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `trapezoid` | `covered` | `jax/numpy/composite_metadata_batch3` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `tri` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `tril` | `covered_indirect` | `jax/numpy/trilu` | Covered via alias `triu`. |
| [ ] | `tril_indices` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `tril_indices_from` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `trim_zeros` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `triu` | `covered` | `jax/numpy/trilu` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `triu_indices` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `triu_indices_from` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `true_divide` | `covered` | `jax/numpy/true_divide` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `trunc` | `covered` | `jax/numpy/trunc` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `ufunc` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `uint` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `uint16` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `uint32` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `uint64` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `uint8` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `union1d` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `unique` | `covered` | `jax/numpy/unique` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `unique_all` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `unique_counts` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `unique_inverse` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `unique_values` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `unpackbits` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `unravel_index` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `unsignedinteger` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `unstack` | `covered` | `jax/numpy/unstack` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `unwrap` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `vander` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `var` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `vdot` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `vecdot` | `covered` | `jax/numpy/composite_metadata_batch2, jax/numpy/composite_metadata_batch3` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `vecmat` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `vectorize` | `covered` | `jax/numpy/composite_metadata_batch6` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `vsplit` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `vstack` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `where` | `covered` | `jax/numpy/where` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `zeros` | `covered` | `jax/numpy/zeros` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `zeros_like` | `covered` | `jax/numpy/zeros_like` | Direct plugin coverage via `jax_doc` metadata. |

## Next Steps
1. Implement missing quick-win `jax.numpy` plugins from the queue above.
2. Add metadata testcases for each new plugin and regenerate tests (`scripts/generate_tests.py`).
3. Re-run this script after each batch to keep coverage docs in sync.

