# Dependencies

**Selected major dependency versions:**

| Library       | Version |
|:--------------|:--------|
| [`JAX`](https://github.com/jax-ml/jax) | 0.12.0.1 |
| [`Flax`](https://github.com/google/flax) | 0.12.10 |
| [`Equinox`](https://github.com/patrick-kidger/equinox) | 0.13.8 |
| [`onnx-ir`](https://github.com/onnx/ir-py) | 1.1.0 |
| [`onnx`](https://github.com/onnx/onnx) | 1.23.2 |
| [`onnxruntime`](https://github.com/microsoft/onnxruntime) | 1.31.0 |
| [`onnxruntime-web`](https://www.npmjs.com/package/onnxruntime-web) | 1.30.0 |

The latest releases are not necessarily selected for every supported Python
version; Poetry resolves compatible versions for each environment.
The modern Poetry stack uses JAX and jaxlib **0.12.0.1** and ONNX IR
**1.1.0** (updated 2026-10-22).
The supported minimums are **Python 3.11**, **JAX/jaxlib 0.10.2** and
**Flax 0.12.8**. `poetry.lock` retains JAX/jaxlib **0.10.2** with Flax **0.12.8**
for Python 3.11/3.12, and the modern stack above for Python 3.13/3.14.
`uv.lock` selects JAX/jaxlib **0.10.2** on Python 3.11 and **0.12.0** on
Python 3.12+, with Flax **0.12.8** and ONNX IR **1.0.0**; this separate lock
retains its earlier resolved versions.
The minimum-dependency CI job installs every direct runtime dependency at its
declared lower bound on Python 3.11; a locked environment with the minimum
framework versions can still use newer versions of other dependencies.
The supported ONNX IR minimum is **1.0.0** (`onnx-ir>=1.0.0`).
`onnxruntime-web` 1.30.0 is locked in `package-lock.json` for the
Node.js and Chromium WASM validation flows. The supported Python runtime
minimum remains ONNX Runtime 1.24.1; exports do not select operators based on
the runtime installed on the export machine.

## Native CPU BF16 conversion in the JAX 0.10.2 cohort

CI549 and CI551 captured an original JAX CPU BF16 TopK failure before ONNX
conversion. On the affected hosted CPU, the first B2 native call ordered the
distinct positive BF16 subnormal words `0001` and `007f` as indices `[0, 1]`
instead of `[1, 0]`; the returned selected-input words themselves were retained.
Passing local runs do not erase that hosted counterexample.

The pinned XLA nested CPU emitter selects a different F32/BF16 conversion route
from its ordinary emitter when certain native target features are available.
The branch-owned patch makes the nested factory use the existing integer
round-to-nearest-even and exact BF16 widening implementation consistently with
the ordinary emitter. It changes neither CPU features nor the TopK algorithm,
source program, runtime flags or numerical comparisons. This is a compiler
repair candidate until an actual corrected build passes the affected workload;
the particular instruction executed in the failed call has not been established.

The inputs are recorded in `build_support/jaxlib/cpu-bf16-build-recipe.json`:
JAX `990e6a0b84138346e6a38785412f36356e0e5dc3`, XLA
`5a9e73cbd92530cac2ac36f4736a774b2412afe2`, its existing LLVM source revision,
the exact compiler patch, Bazel 7.7.0, hermetic toolchain and Python requirement
locks. The build uses the upstream `--local_xla_path` and `--python_version`
interfaces. Its wheel has the visible local version
`0.10.2+jax2onnx.bf16.<recipe-digest-prefix>`; official 0.10.2 bytes are never
renamed or silently treated as corrected.

The separate **Corrected CPU jaxlib build pilot** workflow runs on pushes to
`precision-migration` that change its six build-support, controller, workflow,
test or dependency-documentation paths. Optional user-initiated manual dispatch
also remains available. It builds Linux x86_64 cp311 and cp312 candidates and
does not replace ordinary CI. Push trials use the exact completed-wheel cache;
manual dispatch can explicitly request a cache miss. It
first measures actual capacity and stops unavailable unless at least 12 GiB
memory and 64 GiB disk are free. The owned build is limited to 12 GiB aggregate
RSS, up to three concurrent Bazel jobs, a 1 GiB JVM and 300 minutes, with live
floors of 2 GiB available memory and 4 GiB disk. Bazel runs in batch mode with a
fresh owned output root; the public Bazel cache is read-only, without remote execution.
Linux subreaper adoption and PID/starttime-bound pidfds keep accounting and
cleanup complete when compiler descendants create their own sessions.
The earlier [one-job cp311 and cp312 trials](https://github.com/enpasos/jax2onnx/actions/runs/40643345683)
reached the 300-minute limit during the wheel build; neither reached compiler
tests or produced an accepted corrected wheel. Three jobs are a bounded
scheduling trial, with completion and memory use still unproved under the
unchanged limits. No native numerical repair is qualified. Resource or log-limit
exhaustion retains a failed trial, not a repaired-wheel fallback.

Completed wheels and manifests use an exact recipe/platform/ABI cache key with
no prefix restore. The standalone controller checks source archives, patch
pre/postimages, tool versions, resolved dependency archives, compiler regression
and manylinux tests, then every wheel RECORD hash, version, ABI and native
member identity. A cache hit repeats wheel-byte verification. Neither these
checks nor a caller-supplied manifest grant private SOURCE numerical authority;
bit-for-bit build reproducibility also requires a separate matching build.
An accepted cache manifest is committed only after final source/environment and
owned-process cleanup fences; it binds the durable final-success evidence.
Failed trials keep unverified wheel bytes and logs without an accepted marker.

Installation is a later, explicit native-overlay step after independent review
of the actual wheel hashes and renewed private build-authority records. In an
owned environment, before any JAX import, the eventual command is:

```bash
sha256sum --check /absolute/path/to/independently-reviewed-wheel.sha256
poetry run python -m pip install --no-deps --force-reinstall /absolute/path/to/reviewed-corrected-jaxlib.whl
```

The paths above are placeholders for a retained, independently accepted wheel,
not currently published package coordinates. Preserve the original environment
and its binary receipts before creating that overlay. Keep the base Poetry/uv
locks and Python markers unchanged in this stage. Python 3.13/3.14 retain their
original modern artifacts; the separate uv resolution is not rewritten.
An expiring Actions cache or artifact URL is not a permanent package dependency.

Qualification still requires the unchanged original 99-case native owner and
385-case context request, complete B2/B5 fresh/warmed/eager inputs, exact words
and indices, stable ties, signed zero and the original NaN policies. The affected
hosted target and all four supported full CI cohorts, plus existing support
checks, must pass on their reviewed binaries. The new focused compiler tests
add obligations; they do not weaken or replace these gates.

## ONNX Runtime kernel gaps

Some lowerings depend on kernels or behavior that ONNX Runtime lacks. Because
exports must run on every supported version, a workaround can be removed only
once the supported minimum includes the fix; a fix in the locked release alone
is not enough. Re-check this table whenever the locked ONNX Runtime or the
supported minimum changes, and update the versions it was checked against. The
`minimum-dependencies` CI job runs the related precision tests at the supported
minimum.

The table records the checked provider/version combinations and the public
LEGACY lowerings. These observations do not establish support or numerical
acceptance for every model. The private SOURCE path keeps its own typed
algorithms and admission gates, described below; it is not a public export
option.

| Gap | Checked | Affected exports | Current handling |
|:--|:--|:--|:--|
| No bfloat16 `Log` on CPU or CUDA; no bfloat16 `Exp` on CPU | CPU 1.24.1, 1.31.0; CUDA 1.31.0 | bfloat16 `log1p`, `expm1`, `sinh`, `jax.nn.log1mexp`, `jax.nn.log_sigmoid` | Computed in float32 and rounded back. The recorded precision controls check accuracy within 1 ulp; a new kernel alone does not justify changing this algorithm. |
| No float64 or bfloat16 `Sinh` on CPU; no `Sinh` on CUDA | CPU 1.24.1, 1.31.0; CUDA 1.31.0 | `jnp.sinh`; `lax.sinh` other than float32 | expm1-based form. Float32 `lax.sinh` keeps `Sinh`, which runs on CPU in CUDA sessions. |
| No float64 `Cos` on CPU | Missing in 1.24.1, present in 1.31.0 | float64 `lax.cos`, `jnp.cos` | Half-angle form `1 - 2 * Sin(x / 2)²` |
| No float64 `Erf` on CPU | CPU 1.24.1, 1.31.0 | Public LEGACY float64 exact `jax.nn.gelu`, `nnx.gelu` | The legacy export uses `Erf` (or `Gelu`, which expands to it) and fails to load; its generated metadata restricts these cases to float32. This does not describe the private SOURCE Erfc algorithm. |
| CPU `Where` returns `+0.0` when it selects `-0.0` | CPU 1.24.1, 1.31.0 | `log1p`, `expm1`, `sinh` | `log1p` and `sinh` end in `Mul` to keep the sign; `expm1(-0.0)` exports as `+0.0`. |

Private SOURCE `log1p` and `expm1` retain their existing typed elementary
algorithms and bit-preserving selection, with separate dtype/opset and numerical
checks. SOURCE `sinh` retains typed `Sinh`, so its kernel limitations still
apply. SOURCE exact GELU traces the original JAX computation through Erfc;
its owning typed implementation avoids reliance on a float64 `Erf` kernel at
the admitted opsets. Those finite implementations do not qualify arbitrary
models or override an unsupported dtype, opset, provider or numerical contract.

To check a kernel, create a session for a one-node model with that operator and
dtype. For providers other than CPU, set the session option
`session.disable_cpu_ep_fallback` to `1` so the node cannot fall back to CPU. A
`NOT_IMPLEMENTED` error means the kernel is missing.

*For dependency requirements and optional dependency groups, see [`pyproject.toml`](https://github.com/enpasos/jax2onnx/blob/main/pyproject.toml). For resolved Python dependency versions and their markers, see `poetry.lock` and `uv.lock`.*
