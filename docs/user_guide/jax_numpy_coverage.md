# JAX NumPy Coverage Checklist

## Scope
- Source list: all autosummary entries linked on the official `jax.numpy` page: `https://docs.jax.dev/en/latest/jax.numpy.html`
- Coverage signal: `jax_doc`/`component` metadata + `jaxpr_primitive` registrations in `jax2onnx/plugins/**/*.py`.

## Snapshot
- Total docs entries: `439`
- Covered (direct plugin): `189`
- Covered (via alias/indirect signal): `78`
- Composite/helper entries: `100`
- Non-functional entries (dtype/type/constants): `60`
- Missing dedicated plugin coverage: `12`

## Full Checklist
Legend: `covered`, `covered_indirect`, `composite`, `non_functional`, `missing`.

| Checklist | jax.numpy Entry | Status | Modules (signals) | Notes |
|:--|:--|:--|:--|:--|
| [x] | `ComplexWarning` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `abs` | `covered` | `jax/numpy/abs` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `absolute` | `covered_indirect` | `jax/lax/abs, jax/numpy/abs` | Covered via alias or lower-level primitive `abs`. |
| [x] | `acos` | `covered` | `jax/numpy/acos` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `acosh` | `covered` | `jax/numpy/acosh` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `add` | `covered` | `jax/numpy/add` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `all` | `covered` | `jax/numpy/all` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `allclose` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `amax` | `covered` | `jax/numpy/amax` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `amin` | `covered` | `jax/numpy/amin` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `angle` | `covered_indirect` | `jax/lax/atan2, jax/numpy/atan2` | Covered via alias or lower-level primitive `atan2`. |
| [x] | `any` | `covered` | `jax/numpy/any` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `append` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `apply_along_axis` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `apply_over_axes` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `arange` | `covered` | `jax/numpy/arange` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `arccos` | `covered_indirect` | `jax/lax/acos, jax/numpy/acos` | Covered via alias or lower-level primitive `acos`. |
| [x] | `arccosh` | `covered_indirect` | `jax/lax/acosh, jax/numpy/acosh` | Covered via alias or lower-level primitive `acosh`. |
| [x] | `arcsin` | `covered_indirect` | `jax/lax/asin, jax/numpy/asin` | Covered via alias or lower-level primitive `asin`. |
| [x] | `arcsinh` | `covered_indirect` | `jax/lax/asinh, jax/numpy/asinh` | Covered via alias or lower-level primitive `asinh`. |
| [x] | `arctan` | `covered_indirect` | `jax/lax/atan, jax/numpy/atan` | Covered via alias or lower-level primitive `atan`. |
| [x] | `arctan2` | `covered_indirect` | `jax/lax/atan2, jax/numpy/atan2` | Covered via alias or lower-level primitive `atan2`. |
| [x] | `arctanh` | `covered_indirect` | `jax/lax/atanh, jax/numpy/atanh` | Covered via alias or lower-level primitive `atanh`. |
| [x] | `argmax` | `covered` | `jax/numpy/argmax` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `argmin` | `covered` | `jax/numpy/argmin` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `argpartition` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `argsort` | `covered_indirect` | `jax/lax/sort, jax/numpy/sort` | Covered via alias or lower-level primitive `sort`. |
| [x] | `argwhere` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `around` | `covered_indirect` | `jax/lax/round` | Covered via alias or lower-level primitive `round`. |
| [x] | `array` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `array_equal` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `array_equiv` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `array_repr` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `array_split` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `array_str` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `asarray` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `asin` | `covered` | `jax/numpy/asin` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `asinh` | `covered` | `jax/numpy/asinh` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `astype` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `atan` | `covered` | `jax/numpy/atan` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `atan2` | `covered` | `jax/numpy/atan2` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `atanh` | `covered` | `jax/numpy/atanh` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `atleast_1d` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `atleast_2d` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `atleast_3d` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `average` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `bartlett` | `covered` | `jax/numpy/windows` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `bincount` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `bitwise_and` | `covered` | `jax/numpy/bitwise_and` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `bitwise_count` | `covered_indirect` | `jax/lax/population_count` | Covered via alias or lower-level primitive `population_count`. |
| [x] | `bitwise_invert` | `covered_indirect` | `jax/numpy/bitwise_not` | Covered via alias or lower-level primitive `bitwise_not`. |
| [x] | `bitwise_left_shift` | `covered` | `jax/numpy/bitwise_left_shift` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `bitwise_not` | `covered` | `jax/numpy/bitwise_not` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `bitwise_or` | `covered` | `jax/numpy/bitwise_or` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `bitwise_right_shift` | `covered` | `jax/numpy/bitwise_right_shift` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `bitwise_xor` | `covered` | `jax/numpy/bitwise_xor` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `blackman` | `covered` | `jax/numpy/windows` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `block` | `covered` | `jax/numpy/composite_metadata_batch4` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `bool_` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `broadcast_arrays` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `broadcast_shapes` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `broadcast_to` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `c_` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `can_cast` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `cbrt` | `covered_indirect` | `jax/lax/cbrt` | Covered via alias or lower-level primitive `cbrt`. |
| [x] | `cdouble` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `ceil` | `covered` | `jax/numpy/ceil` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `character` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `choose` | `covered` | `jax/numpy/composite_metadata_batch3` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `clip` | `covered` | `jax/numpy/clip` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `column_stack` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `complex128` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `complex64` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `complex_` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `complexfloating` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `compress` | `covered` | `jax/numpy/compress` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `concat` | `covered_indirect` | `jax/lax/concatenate, jax/numpy/concatenate` | Covered via alias or lower-level primitive `concatenate`. |
| [x] | `concatenate` | `covered` | `jax/numpy/concatenate` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `conj` | `covered` | `jax/numpy/conj` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `conjugate` | `covered_indirect` | `jax/lax/conj, jax/numpy/conj` | Covered via alias or lower-level primitive `conj`. |
| [x] | `convolve` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `copy` | `covered_indirect` | `jax/lax/copy` | Covered via alias or lower-level primitive `copy`. |
| [x] | `copysign` | `covered` | `jax/numpy/copysign` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `corrcoef` | `covered` | `jax/numpy/composite_metadata_batch5` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `correlate` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `cos` | `covered` | `jax/numpy/cos` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `cosh` | `covered` | `jax/numpy/cosh` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `count_nonzero` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `cov` | `covered` | `jax/numpy/composite_metadata` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `cross` | `covered` | `jax/numpy/composite_metadata` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `csingle` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `cumprod` | `covered` | `jax/numpy/cumprod` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `cumsum` | `covered` | `jax/numpy/cumsum` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `cumulative_prod` | `covered_indirect` | `jax/lax/cumprod, jax/numpy/cumprod` | Covered via alias or lower-level primitive `cumprod`. |
| [x] | `cumulative_sum` | `covered_indirect` | `jax/lax/cumsum, jax/numpy/cumsum` | Covered via alias or lower-level primitive `cumsum`. |
| [x] | `deg2rad` | `covered_indirect` | `jax/lax/mul` | Covered via alias or lower-level primitive `mul`. |
| [x] | `degrees` | `covered_indirect` | `jax/lax/mul` | Covered via alias or lower-level primitive `mul`. |
| [x] | `delete` | `covered` | `jax/numpy/composite_metadata` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `diag` | `covered` | `jax/numpy/diag` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `diag_indices` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `diag_indices_from` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `diagflat` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `diagonal` | `covered` | `jax/numpy/diagonal` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `diff` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [ ] | `digitize` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `divide` | `covered` | `jax/numpy/divide` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `divmod` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `dot` | `covered` | `jax/numpy/dot` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `double` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `dsplit` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `dstack` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `dtype` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `ediff1d` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `einsum` | `covered` | `jax/numpy/einsum` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `einsum_path` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `empty` | `covered_indirect` | `jax/lax/broadcast_in_dim` | Covered via alias or lower-level primitive `broadcast_in_dim`. |
| [x] | `empty_like` | `covered_indirect` | `jax/lax/broadcast_in_dim` | Covered via alias or lower-level primitive `broadcast_in_dim`. |
| [x] | `equal` | `covered` | `jax/numpy/equal` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `exp` | `covered` | `jax/numpy/exp` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `exp2` | `covered` | `jax/numpy/exp2` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `expand_dims` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `expm1` | `covered` | `jax/numpy/expm1` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `extract` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `eye` | `covered` | `jax/numpy/eye` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `fabs` | `covered` | `jax/numpy/fabs` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `fft.fft` | `covered` | `jax/numpy/fft` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `fft.fft2` | `covered_indirect` | `jax/lax/fft, jax/numpy/fft` | Covered via alias or lower-level primitive `fft`. |
| [x] | `fft.fftfreq` | `covered` | `jax/numpy/composite_metadata` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `fft.fftn` | `covered_indirect` | `jax/lax/fft, jax/numpy/fft` | Covered via alias or lower-level primitive `fft`. |
| [x] | `fft.fftshift` | `covered` | `jax/numpy/composite_metadata` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `fft.hfft` | `covered_indirect` | `jax/numpy/fft` | Covered via alias or lower-level primitive `irfft`. |
| [x] | `fft.ifft` | `covered` | `jax/numpy/fft` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `fft.ifft2` | `covered_indirect` | `jax/numpy/fft` | Covered via alias or lower-level primitive `ifft`. |
| [x] | `fft.ifftn` | `covered_indirect` | `jax/numpy/fft` | Covered via alias or lower-level primitive `ifft`. |
| [x] | `fft.ifftshift` | `covered` | `jax/numpy/composite_metadata` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `fft.ihfft` | `covered_indirect` | `jax/numpy/fft` | Covered via alias or lower-level primitive `rfft`. |
| [x] | `fft.irfft` | `covered` | `jax/numpy/fft` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `fft.irfft2` | `covered_indirect` | `jax/numpy/fft` | Covered via alias or lower-level primitive `irfft`. |
| [x] | `fft.irfftn` | `covered_indirect` | `jax/numpy/fft` | Covered via alias or lower-level primitive `irfft`. |
| [x] | `fft.rfft` | `covered` | `jax/numpy/fft` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `fft.rfft2` | `covered_indirect` | `jax/numpy/fft` | Covered via alias or lower-level primitive `rfft`. |
| [x] | `fft.rfftfreq` | `covered` | `jax/numpy/composite_metadata` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `fft.rfftn` | `covered_indirect` | `jax/numpy/fft` | Covered via alias or lower-level primitive `rfft`. |
| [x] | `fill_diagonal` | `covered` | `jax/numpy/composite_metadata` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `finfo` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `flatnonzero` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `flexible` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `flip` | `covered_indirect` | `jax/lax/rev` | Covered via alias or lower-level primitive `rev`. |
| [x] | `fliplr` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `flipud` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `float16` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `float32` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `float64` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `float_` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `float_power` | `covered_indirect` | `jax/lax/pow, jax/numpy/pow` | Covered via alias or lower-level primitive `pow`. |
| [x] | `floating` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `floor` | `covered` | `jax/numpy/floor` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `floor_divide` | `covered` | `jax/numpy/floor_divide` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `fmax` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `fmin` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `fmod` | `covered` | `jax/numpy/fmod` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `frexp` | `covered` | `jax/numpy/frexp` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `from_dlpack` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `frombuffer` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `fromfile` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `fromfunction` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `fromiter` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `frompyfunc` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `fromstring` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `full` | `covered` | `jax/numpy/full` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `full_like` | `covered_indirect` | `jax/lax/broadcast_in_dim` | Covered via alias or lower-level primitive `broadcast_in_dim`. |
| [x] | `gcd` | `covered` | `jax/numpy/composite_metadata_batch6` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `generic` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `geomspace` | `covered` | `jax/numpy/composite_metadata` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `get_printoptions` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `gradient` | `covered` | `jax/numpy/composite_metadata` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `greater` | `covered` | `jax/numpy/greater` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `greater_equal` | `covered` | `jax/numpy/greater_equal` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `hamming` | `covered` | `jax/numpy/windows` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `hanning` | `covered` | `jax/numpy/windows` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `heaviside` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [ ] | `histogram` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `histogram2d` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `histogram_bin_edges` | `covered` | `jax/numpy/composite_metadata` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [ ] | `histogramdd` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `hsplit` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `hstack` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `hypot` | `covered_indirect` | `jax/lax/sqrt, jax/numpy/sqrt` | Covered via alias or lower-level primitive `sqrt`. |
| [x] | `i0` | `covered` | `jax/numpy/composite_metadata` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `identity` | `covered_indirect` | `jax/lax/iota` | Covered via alias or lower-level primitive `iota`. |
| [x] | `iinfo` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `imag` | `covered_indirect` | `jax/lax/imag` | Covered via alias or lower-level primitive `imag`. |
| [x] | `index_exp` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `indices` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `inexact` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `inner` | `covered_indirect` | `jax/lax/dot_general` | Covered via alias or lower-level primitive `dot_general`. |
| [x] | `insert` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `int16` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `int32` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `int64` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `int8` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `int_` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `integer` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [ ] | `interp` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `intersect1d` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `invert` | `covered` | `jax/numpy/invert` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `isclose` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `iscomplex` | `covered_indirect` | `jax/lax/imag` | Covered via alias or lower-level primitive `imag`. |
| [x] | `iscomplexobj` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `isdtype` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `isfinite` | `covered` | `jax/numpy/isfinite` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `isin` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `isinf` | `covered_indirect` | `jax/lax/eq` | Covered via alias or lower-level primitive `eq`. |
| [x] | `isnan` | `covered_indirect` | `jax/lax/ne` | Covered via alias or lower-level primitive `ne`. |
| [x] | `isneginf` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `isposinf` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `isreal` | `covered_indirect` | `jax/lax/imag` | Covered via alias or lower-level primitive `imag`. |
| [x] | `isrealobj` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `isscalar` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `issubdtype` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `iterable` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `ix_` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `kaiser` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `kron` | `covered_indirect` | `jax/lax/mul` | Covered via alias or lower-level primitive `mul`. |
| [x] | `lcm` | `covered` | `jax/numpy/composite_metadata_batch6` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `ldexp` | `covered` | `jax/numpy/ldexp` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `left_shift` | `covered` | `jax/numpy/left_shift` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `less` | `covered` | `jax/numpy/less` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `less_equal` | `covered` | `jax/numpy/less_equal` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `lexsort` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `linalg.cholesky` | `covered` | `jax/numpy/composite_metadata_batch4` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `linalg.cond` | `covered` | `jax/numpy/composite_metadata_batch4` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `linalg.cross` | `covered` | `jax/numpy/composite_metadata` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `linalg.det` | `covered` | `jax/numpy/linalg_det` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `linalg.diagonal` | `covered` | `jax/numpy/diagonal` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `linalg.eig` | `covered` | `jax/numpy/composite_metadata_batch5` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `linalg.eigh` | `covered` | `jax/numpy/composite_metadata_batch5` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `linalg.eigvals` | `covered` | `jax/numpy/composite_metadata_batch5` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `linalg.eigvalsh` | `covered` | `jax/numpy/composite_metadata_batch5` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [ ] | `linalg.inv` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `linalg.lstsq` | `covered` | `jax/numpy/composite_metadata_batch5` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `linalg.matmul` | `covered` | `jax/numpy/matmul` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `linalg.matrix_norm` | `covered` | `jax/numpy/composite_metadata_batch3` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `linalg.matrix_power` | `covered` | `jax/numpy/composite_metadata_batch4` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `linalg.matrix_rank` | `covered` | `jax/numpy/composite_metadata_batch4` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `linalg.matrix_transpose` | `covered_indirect` | `jax/lax/transpose, jax/numpy/transpose` | Covered via alias or lower-level primitive `transpose`. |
| [x] | `linalg.multi_dot` | `covered` | `jax/numpy/composite_metadata_batch5` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `linalg.norm` | `covered` | `jax/numpy/linalg_norm` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `linalg.outer` | `covered` | `jax/numpy/outer` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `linalg.pinv` | `covered` | `jax/numpy/composite_metadata_batch5` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `linalg.qr` | `covered` | `jax/numpy/composite_metadata_batch4` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `linalg.slogdet` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [ ] | `linalg.solve` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `linalg.svd` | `covered` | `jax/numpy/composite_metadata_batch5` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `linalg.svdvals` | `covered` | `jax/numpy/composite_metadata_batch4` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `linalg.tensordot` | `covered` | `jax/numpy/composite_metadata_batch2, jax/numpy/composite_metadata_batch3` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [ ] | `linalg.tensorinv` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `linalg.tensorsolve` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `linalg.trace` | `covered` | `jax/numpy/composite_metadata_batch2, jax/numpy/composite_metadata_batch3` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `linalg.vecdot` | `covered` | `jax/numpy/composite_metadata_batch2, jax/numpy/composite_metadata_batch3` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `linalg.vector_norm` | `covered` | `jax/numpy/composite_metadata_batch3` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `linspace` | `covered` | `jax/numpy/linspace` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `load` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `log` | `covered` | `jax/numpy/log` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `log10` | `covered_indirect` | `jax/lax/log, jax/numpy/log` | Covered via alias or lower-level primitive `log`. |
| [x] | `log1p` | `covered_indirect` | `jax/lax/log1p` | Covered via alias or lower-level primitive `log1p`. |
| [x] | `log2` | `covered_indirect` | `jax/lax/log, jax/numpy/log` | Covered via alias or lower-level primitive `log`. |
| [x] | `logaddexp` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `logaddexp2` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `logical_and` | `covered_indirect` | `jax/lax/and` | Covered via alias or lower-level primitive `and`. |
| [x] | `logical_not` | `covered_indirect` | `jax/lax/bitwise_not` | Covered via alias or lower-level primitive `not`. |
| [x] | `logical_or` | `covered_indirect` | `jax/lax/or` | Covered via alias or lower-level primitive `or`. |
| [x] | `logical_xor` | `covered_indirect` | `jax/lax/xor` | Covered via alias or lower-level primitive `xor`. |
| [x] | `logspace` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `mask_indices` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `matmul` | `covered` | `jax/numpy/matmul` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `matrix_transpose` | `covered_indirect` | `jax/lax/transpose, jax/numpy/transpose` | Covered via alias or lower-level primitive `transpose`. |
| [x] | `matvec` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `max` | `covered` | `jax/numpy/max` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `maximum` | `covered` | `jax/numpy/maximum` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `mean` | `covered` | `jax/numpy/mean` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `median` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `meshgrid` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `mgrid` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `min` | `covered` | `jax/numpy/min` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `minimum` | `covered` | `jax/numpy/minimum` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `mod` | `covered_indirect` | `jax/lax/rem` | Covered via alias or lower-level primitive `rem`. |
| [x] | `modf` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `moveaxis` | `covered` | `jax/numpy/moveaxis` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `multiply` | `covered_indirect` | `jax/lax/mul` | Covered via alias or lower-level primitive `mul`. |
| [x] | `nan_to_num` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `nanargmax` | `covered` | `jax/numpy/composite_metadata_batch3` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `nanargmin` | `covered` | `jax/numpy/composite_metadata_batch3` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [ ] | `nancumprod` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `nancumsum` | `covered` | `jax/numpy/composite_metadata_batch3` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `nanmax` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `nanmean` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `nanmedian` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `nanmin` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `nanpercentile` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `nanprod` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `nanquantile` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `nanstd` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `nansum` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `nanvar` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `ndarray` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `ndarray.at` | `covered` | `jax/numpy/composite_metadata_batch6` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `ndim` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `negative` | `covered_indirect` | `jax/lax/neg` | Covered via alias or lower-level primitive `neg`. |
| [x] | `nextafter` | `covered_indirect` | `jax/lax/nextafter` | Covered via alias or lower-level primitive `nextafter`. |
| [x] | `nonzero` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `not_equal` | `covered_indirect` | `jax/lax/ne` | Covered via alias or lower-level primitive `ne`. |
| [x] | `number` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `object_` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `ogrid` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `ones` | `covered` | `jax/numpy/ones` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `ones_like` | `covered_indirect` | `jax/lax/broadcast_in_dim` | Covered via alias or lower-level primitive `broadcast_in_dim`. |
| [x] | `outer` | `covered` | `jax/numpy/outer` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `packbits` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `pad` | `covered` | `jax/numpy/pad` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `partition` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `percentile` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `permute_dims` | `covered_indirect` | `jax/lax/transpose, jax/numpy/transpose` | Covered via alias or lower-level primitive `transpose`. |
| [x] | `piecewise` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `place` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `poly` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `polyadd` | `covered` | `jax/numpy/composite_metadata_batch4` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `polyder` | `covered` | `jax/numpy/composite_metadata_batch4` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `polydiv` | `covered` | `jax/numpy/composite_metadata_batch4` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [ ] | `polyfit` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `polyint` | `covered` | `jax/numpy/composite_metadata_batch4` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `polymul` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `polysub` | `covered` | `jax/numpy/composite_metadata_batch4` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `polyval` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `positive` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `pow` | `covered` | `jax/numpy/pow` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `power` | `covered` | `jax/numpy/pow` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `printoptions` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `prod` | `covered` | `jax/numpy/prod` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `promote_types` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `ptp` | `covered_indirect` | `jax/lax/reduce_max` | Covered via alias or lower-level primitive `reduce_max`. |
| [x] | `put` | `covered` | `jax/numpy/composite_metadata_batch6` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `put_along_axis` | `covered` | `jax/numpy/composite_metadata_batch6` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `quantile` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `r_` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `rad2deg` | `covered_indirect` | `jax/lax/mul` | Covered via alias or lower-level primitive `mul`. |
| [x] | `radians` | `covered_indirect` | `jax/lax/mul` | Covered via alias or lower-level primitive `mul`. |
| [x] | `ravel` | `covered_indirect` | `jax/lax/reshape, jax/numpy/reshape` | Covered via alias or lower-level primitive `reshape`. |
| [x] | `ravel_multi_index` | `covered` | `jax/numpy/composite_metadata_batch3` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `real` | `covered_indirect` | `jax/lax/real` | Covered via alias or lower-level primitive `real`. |
| [x] | `reciprocal` | `covered_indirect` | `jax/lax/integer_pow` | Covered via alias or lower-level primitive `integer_pow`. |
| [x] | `remainder` | `covered_indirect` | `jax/lax/rem` | Covered via alias or lower-level primitive `rem`. |
| [x] | `repeat` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `reshape` | `covered` | `jax/numpy/reshape` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `resize` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `result_type` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `right_shift` | `covered` | `jax/numpy/right_shift` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `rint` | `covered_indirect` | `jax/lax/round` | Covered via alias or lower-level primitive `round`. |
| [x] | `roll` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `rollaxis` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [ ] | `roots` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `rot90` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `round` | `covered_indirect` | `jax/lax/round` | Covered via alias or lower-level primitive `round`. |
| [x] | `s_` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `save` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `savez` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `searchsorted` | `covered` | `jax/numpy/searchsorted` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `select` | `covered` | `jax/numpy/select` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `set_printoptions` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `setdiff1d` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `setxor1d` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `shape` | `covered` | `jax/numpy/shape` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `sign` | `covered` | `jax/numpy/sign` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `signbit` | `covered_indirect` | `jax/lax/shift_right_arithmetic` | Covered via alias or lower-level primitive `shift_right_arithmetic`. |
| [x] | `signedinteger` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `sin` | `covered` | `jax/numpy/sin` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `sinc` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `single` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `sinh` | `covered` | `jax/numpy/sinh` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `size` | `covered` | `jax/numpy/size` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `sort` | `covered` | `jax/numpy/sort` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `sort_complex` | `covered_indirect` | `jax/lax/sort, jax/numpy/sort` | Covered via alias or lower-level primitive `sort`. |
| [x] | `spacing` | `covered` | `jax/numpy/spacing` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `split` | `covered` | `jax/numpy/split` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `sqrt` | `covered` | `jax/numpy/sqrt` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `square` | `covered_indirect` | `jax/lax/square` | Covered via alias or lower-level primitive `square`. |
| [x] | `squeeze` | `covered` | `jax/numpy/squeeze` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `stack` | `covered` | `jax/numpy/stack` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `std` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `subtract` | `covered_indirect` | `jax/lax/sub` | Covered via alias or lower-level primitive `sub`. |
| [x] | `sum` | `covered` | `jax/numpy/sum` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `swapaxes` | `covered_indirect` | `jax/lax/transpose, jax/numpy/transpose` | Covered via alias or lower-level primitive `transpose`. |
| [x] | `take` | `covered` | `jax/numpy/take` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `take_along_axis` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `tan` | `covered` | `jax/numpy/tan` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `tanh` | `covered` | `jax/numpy/tanh` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `tensordot` | `covered` | `jax/numpy/composite_metadata_batch2, jax/numpy/composite_metadata_batch3` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `tile` | `covered` | `jax/numpy/tile` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `trace` | `covered` | `jax/numpy/composite_metadata_batch2, jax/numpy/composite_metadata_batch3` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `transpose` | `covered` | `jax/numpy/transpose` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `trapezoid` | `covered` | `jax/numpy/composite_metadata_batch3` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `tri` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `tril` | `covered_indirect` | `jax/numpy/trilu` | Covered via alias or lower-level primitive `triu`. |
| [x] | `tril_indices` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `tril_indices_from` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `trim_zeros` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `triu` | `covered` | `jax/numpy/trilu` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `triu_indices` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `triu_indices_from` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `true_divide` | `covered_indirect` | `jax/lax/div` | Covered via alias or lower-level primitive `div`. |
| [x] | `trunc` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `ufunc` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `uint` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `uint16` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `uint32` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `uint64` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `uint8` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `union1d` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `unique` | `covered` | `jax/numpy/unique` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `unique_all` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `unique_counts` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `unique_inverse` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `unique_values` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `unpackbits` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `unravel_index` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `unsignedinteger` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `unstack` | `covered` | `jax/numpy/unstack` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `unwrap` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `vander` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `var` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `vdot` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `vecdot` | `covered` | `jax/numpy/composite_metadata_batch2, jax/numpy/composite_metadata_batch3` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `vecmat` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `vectorize` | `covered` | `jax/numpy/composite_metadata_batch6` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `vsplit` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `vstack` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `where` | `covered` | `jax/numpy/where` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `zeros` | `covered` | `jax/numpy/zeros` | Direct plugin coverage via `jax_doc` or `component` metadata. |
| [x] | `zeros_like` | `covered_indirect` | `jax/lax/broadcast_in_dim` | Covered via alias or lower-level primitive `broadcast_in_dim`. |

