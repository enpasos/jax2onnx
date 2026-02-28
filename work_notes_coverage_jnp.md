# Work Notes: JAX NumPy Coverage Checklist

## Scope
- Source list: all autosummary entries linked on the official `jax.numpy` page: `https://docs.jax.dev/en/latest/jax.numpy.html`
- Coverage signal: `jax_doc` metadata + `jaxpr_primitive` registrations in `jax2onnx/plugins/**/*.py`.

Regenerate with:

```bash
poetry run python scripts/generate_jnp_operator_coverage.py
```

## Snapshot
- Total docs entries: `440`
- Covered (direct plugin): `143`
- Covered (via alias/indirect signal): `7`
- Composite/helper entries: `60`
- Non-functional entries (dtype/type/constants): `34`
- Missing dedicated plugin coverage: `196`

## Priority Gap Queue
- [ ] `argmax`
- [ ] `argmin`
- [ ] `diag`
- [ ] `dot`
- [ ] `ones`
- [ ] `pad`
- [ ] `zeros`

## Full Checklist
Legend: `covered`, `covered_indirect`, `composite`, `non_functional`, `missing`.

| Checklist | jax.numpy Entry | Status | Modules (signals) | Notes |
|:--|:--|:--|:--|:--|
| [x] | `ComplexWarning` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `abs` | `covered` | `jax/numpy/abs` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `absolute` | `covered_indirect` | `jax/lax/abs, jax/numpy/abs` | Covered via alias `abs`. |
| [ ] | `acos` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `acosh` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `add` | `covered` | `jax/numpy/add` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `all` | `covered` | `jax/numpy/all` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `allclose` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `amax` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `amin` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `angle` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `any` | `covered` | `jax/numpy/any` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `append` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `apply_along_axis` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `apply_over_axes` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `arange` | `covered` | `jax/numpy/arange` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `arccos` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `arccosh` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `arcsin` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `arcsinh` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `arctan` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `arctan2` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `arctanh` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `argmax` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `argmin` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `argpartition` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `argsort` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `argwhere` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `around` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `array` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `array_equal` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `array_equiv` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `array_repr` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `array_split` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `array_str` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `asarray` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [ ] | `asin` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `asinh` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `astype` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [ ] | `atan` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `atan2` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `atanh` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `atleast_1d` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `atleast_2d` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `atleast_3d` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [ ] | `average` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `bartlett` | `covered` | `jax/numpy/windows` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `bincount` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `bitwise_and` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `bitwise_count` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `bitwise_invert` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `bitwise_left_shift` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `bitwise_not` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `bitwise_or` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `bitwise_right_shift` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `bitwise_xor` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `blackman` | `covered` | `jax/numpy/windows` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `block` | `covered` | `jax/numpy/composite_metadata_batch4` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `bool_` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `broadcast_arrays` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `broadcast_shapes` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `broadcast_to` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `c_` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [ ] | `can_cast` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `cbrt` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
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
| [ ] | `copy` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `copysign` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `corrcoef` | `covered` | `jax/numpy/composite_metadata_batch5` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `correlate` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `cos` | `covered` | `jax/numpy/cos` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `cosh` | `covered` | `jax/numpy/cosh` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `count_nonzero` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `cov` | `covered` | `jax/numpy/composite_metadata` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `cross` | `covered` | `jax/numpy/composite_metadata` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `csingle` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `cumprod` | `covered` | `jax/numpy/cumprod` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `cumsum` | `covered` | `jax/numpy/cumsum` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `cumulative_prod` | `covered_indirect` | `jax/lax/cumprod, jax/numpy/cumprod` | Covered via alias `cumprod`. |
| [x] | `cumulative_sum` | `covered_indirect` | `jax/lax/cumsum, jax/numpy/cumsum` | Covered via alias `cumsum`. |
| [ ] | `deg2rad` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `degrees` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `delete` | `covered` | `jax/numpy/composite_metadata` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `diag` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `diag_indices` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `diag_indices_from` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `diagflat` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [ ] | `diagonal` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `diff` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `digitize` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `divide` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `divmod` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `dot` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `double` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `dsplit` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `dstack` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `dtype` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `ediff1d` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `einsum` | `covered` | `jax/numpy/einsum` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `einsum_path` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `empty` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `empty_like` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `equal` | `covered` | `jax/numpy/equal` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `exp` | `covered` | `jax/numpy/exp` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `exp2` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `expand_dims` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [ ] | `expm1` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `extract` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `eye` | `covered` | `jax/numpy/eye` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `fabs` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
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
| [ ] | `fix` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `flatnonzero` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [ ] | `flexible` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `flip` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `fliplr` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `flipud` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [ ] | `float16` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `float32` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `float64` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `float_` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `float_power` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `floating` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `floor` | `covered` | `jax/numpy/floor` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `floor_divide` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `fmax` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `fmin` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `fmod` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `frexp` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `from_dlpack` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `frombuffer` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [ ] | `fromfile` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `fromfunction` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `fromiter` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [ ] | `frompyfunc` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `fromstring` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `full` | `covered` | `jax/numpy/full` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `full_like` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `gcd` | `covered` | `jax/numpy/composite_metadata_batch6` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `generic` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `geomspace` | `covered` | `jax/numpy/composite_metadata` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `get_printoptions` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `gradient` | `covered` | `jax/numpy/composite_metadata` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `greater` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `greater_equal` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `hamming` | `covered` | `jax/numpy/windows` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `hanning` | `covered` | `jax/numpy/windows` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `heaviside` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `histogram` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `histogram2d` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `histogram_bin_edges` | `covered` | `jax/numpy/composite_metadata` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `histogramdd` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `hsplit` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `hstack` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [ ] | `hypot` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `i0` | `covered` | `jax/numpy/composite_metadata` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `identity` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `iinfo` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `imag` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `index_exp` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `indices` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `inexact` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [ ] | `inner` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `insert` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `int16` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `int32` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `int64` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `int8` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `int_` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [x] | `integer` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [ ] | `interp` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `intersect1d` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `invert` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `isclose` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [ ] | `iscomplex` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `iscomplexobj` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `isdtype` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `isfinite` | `covered` | `jax/numpy/isfinite` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `isin` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `isinf` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `isnan` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `isneginf` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `isposinf` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [ ] | `isreal` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `isrealobj` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `isscalar` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `issubdtype` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [ ] | `iterable` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `ix_` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `kaiser` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [ ] | `kron` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `lcm` | `covered` | `jax/numpy/composite_metadata_batch6` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `ldexp` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `left_shift` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `less` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `less_equal` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `lexsort` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `linalg.cholesky` | `covered` | `jax/numpy/composite_metadata_batch4` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `linalg.cond` | `covered` | `jax/numpy/composite_metadata_batch4` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `linalg.cross` | `covered` | `jax/numpy/composite_metadata` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `linalg.det` | `covered` | `jax/numpy/linalg_det` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `linalg.diagonal` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
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
| [ ] | `linalg.matrix_transpose` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
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
| [ ] | `log10` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `log1p` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `log2` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `logaddexp` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `logaddexp2` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `logical_and` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `logical_not` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `logical_or` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `logical_xor` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `logspace` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `mask_indices` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `matmul` | `covered` | `jax/numpy/matmul` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `matrix_transpose` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `matvec` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `max` | `covered` | `jax/numpy/max` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `maximum` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `mean` | `covered` | `jax/numpy/mean` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `median` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `meshgrid` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `mgrid` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `min` | `covered` | `jax/numpy/min` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `minimum` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `mod` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `modf` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `moveaxis` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `multiply` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `nan_to_num` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
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
| [ ] | `ndim` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `negative` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `nextafter` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `nonzero` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `not_equal` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `number` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [ ] | `object_` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `ogrid` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [ ] | `ones` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `ones_like` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `outer` | `covered` | `jax/numpy/outer` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `packbits` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `pad` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `partition` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `percentile` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `permute_dims` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
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
| [ ] | `positive` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `pow` | `covered_indirect` | `jax/numpy/pow` | Covered via alias `power`. |
| [x] | `power` | `covered` | `jax/numpy/pow` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `printoptions` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `prod` | `covered` | `jax/numpy/prod` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `promote_types` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [ ] | `ptp` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `put` | `covered` | `jax/numpy/composite_metadata_batch6` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `put_along_axis` | `covered` | `jax/numpy/composite_metadata_batch6` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `quantile` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `r_` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [ ] | `rad2deg` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `radians` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `ravel` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `ravel_multi_index` | `covered` | `jax/numpy/composite_metadata_batch3` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `real` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `reciprocal` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `remainder` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `repeat` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `reshape` | `covered` | `jax/numpy/reshape` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `resize` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [x] | `result_type` | `non_functional` | `-` | Constant/dtype/type helper entry; no standalone plugin expected. |
| [ ] | `right_shift` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `rint` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `roll` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `rollaxis` | `composite` | `-` | Composite/helper API; typically lowered through other primitives. |
| [ ] | `roots` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `rot90` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `round` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
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
| [ ] | `signbit` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
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
| [ ] | `square` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `squeeze` | `covered` | `jax/numpy/squeeze` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `stack` | `covered` | `jax/numpy/stack` | Direct plugin coverage via `jax_doc` metadata. |
| [x] | `std` | `covered` | `jax/numpy/composite_metadata_batch2` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `subtract` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [x] | `sum` | `covered` | `jax/numpy/sum` | Direct plugin coverage via `jax_doc` metadata. |
| [ ] | `swapaxes` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
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
| [ ] | `true_divide` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `trunc` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
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
| [ ] | `zeros` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |
| [ ] | `zeros_like` | `missing` | `-` | Missing dedicated `jax.numpy` plugin coverage. |

## Next Steps
1. Implement missing quick-win `jax.numpy` plugins from the queue above.
2. Add metadata testcases for each new plugin and regenerate tests (`scripts/generate_tests.py`).
3. Re-run this script after each batch to keep coverage docs in sync.

