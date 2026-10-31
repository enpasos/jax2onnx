# Fixed-Source Precision Preparation: Portable Finite Proof

The internal [portable proof](fixed_source_preparation.py) demonstrates one
explicit numerical redesign before JAX tracing and a separate model-configuration
case. It does not implement a generic production transformer or make arbitrary
fixed-source models compatible. The ordinary public API and both legacy double
options remain unchanged; full compatibility is still required before phase F.

## Three different precision choices

The original exact witness contains `x.astype(jnp.float32)`. For the fixed strong
FP64 input `[1e8, 1e8+1, -1e8, 0, 16777216]`, `(y + 1) - y` returns FP32
`[0,0,0,1,0]`. The finite external adapter accepts only that exact source preimage
and explicit authorization to replace the selected cast with FP64. It changes
the AST before compiling a new native-evaluable JAX callable; the prepared
result is FP64 `[1,1,1,1,1]`. The original function/source remain unchanged.
This intentionally changes the computation. It is not source-preserving export
of the original FP32 calculation, equivalence or recovery of lost precision.

The unchanged minimum-precision example instead computes
`promote_types(x.dtype, float32)` before tracing. FP16/BF16 inputs therefore use
FP32 work, while FP64 inputs retain FP64. Its four recorded dtypes distinguish
that input-dependent rule from a fixed cast.

The NNX Linear(128,64) configuration case captures actual FP32 kernel/bias values,
clones the model, changes its computation dtype and uses complete `set_value`
replacement for exact FP64 widenings of the same stored parameters. It never
reinitializes new FP64 weights. Wider storage cannot recover information already
lost in FP32. Configuration is separate from replacing an explicit source cast.

## Protected work and native references

The twelve cases are original/prepared exact witnesses, their two intermediate
variants, their two protected-consumer variants, four minimum-precision input
dtypes and original/prepared Linear. Protected consumers retain FP16/BF16 rounding,
FP32 accumulation, an unchanged rounded sibling, the UINT32 bitcast, shared
values and deliberate castback. The accepted function-body operation order is
retained: sum precedes bitcast.

Each run freezes source/feeds, known exact/minimum expectations, settings and
criteria before evaluating its cases. All native arrays are saved before the
first export; actual target arrays are saved before comparison. Each stack has
30 eager/JIT output pairs, six known expected outputs and 82 saved NPY files.
Exact cases use bitwise type/shape/payload comparisons. Linear's finite native
comparison uses `2*gamma128*sum(abs(x*k)) + 4*eps*abs(bias)`, with
`gamma128 = 128*eps/(1-128*eps)` and the chosen computation dtype's epsilon.
This is not a general matrix or accumulator guarantee.

The harness preserves five hidden/global/local/effective JAX settings, original
callables and active precision policy. Tests cover nested contexts and exceptions,
interleaved original/prepared JIT calls, unchanged original weights and protected
sibling bytes. It introduces no import-time seeding, cache clearing, environment
assignment or CPU-affinity mutation. The measured HIGHEST context is imposed by
this proof; it is not a requirement of every original NNX Linear.

## Export and runtime results

Independent v3 review accepts all 470 frozen pins and the exact two new files.
The maintained tests execute the proof and pass **11 cases per stack**; these
are not receipt-only tests. Configured Black/Ruff, mypy on both files and module
help have persisted successful receipts.

| Per supported stack | Observed result |
| --- | --- |
| Ordinary audited SOURCE export attempts | 12, at opset 23, with the default validator. |
| Structurally checked artifacts | 4: two exact witnesses and two intermediate variants. |
| Actual ORT sessions/runs | 4/4: two exact artifacts, each with CPU disabled/all optimization. |
| Target outputs | Four saved actual Y arrays, each bitwise equal to its own frozen native reference. |
| Remaining first barriers | Eight: two protected sums, four minimum means, two HIGHEST dots. |
| Other ten programs | Zero ORT runs; target numerical acceptance remains unverified/None. |

Across both stacks this is 24 export attempts, eight sessions and eight actual
runs. Each exact artifact's same serialized bytes serve both CPU optimization
settings, with one sequential intra/inter-op thread and fallback disabled.
Original graphs retain Cast/Add/Sub and FP32 output; prepared graphs retain
Add/Sub and FP64 output. Source reports retain runtime/numerical UNVERIFIED;
separate target reports hold the finite observed acceptance. The intermediate
artifacts have structural evidence only.

Protected bitcast remains in the native calculation but is not reached by these
audited exports, which first stop at sum. No broader bitcast, reduction, HIGHEST,
GPU or all-model acceptance follows. Aggregate PoC `validated` means that this
bounded observation completed successfully, not that every case was admitted.
No diagnostic validator replaces the default audited path.

## Reproduce from tracked files

Use an existing supported environment from the repository root:

```bash
PYTHONPATH=$PWD python -m internal_docs.precision.poc.fixed_source_preparation --output /outside/repo/new-run
PYTHONPATH=$PWD python -m pytest -q -n 0 tests/extra_tests/converter/test_fixed_source_preparation_poc.py
```

The output directory must be new and outside the resolved repository path.
Generated arrays, contracts, graphs and reports go there; no generated ONNX
artifact belongs in Git. The CLI records actual affinity/environment and does
not require the owner's CPU numbering or historical `/tmp` archives. This is an
internal finite demonstration, not a public preparation switch.

The exact two PoC files are committed in the combined fourteen-path
named/preparation/cache/inventory checkpoint `8f823ac3`. MAIN focused checks
pass 2,191 cases per stack; the complete primary suite passes 8,772 tests and
all seven actual code hooks pass. These root-owned checks are separate from
the owner's eleven-test/native/target epochs above. The first full failure and
remaining older-stack benchmark failure are recorded in the
[integration findings](named_runtime_parameters_findings.md#actual-main-integration-boundary).

## Historical epochs remain separate

The accepted temporary stage-1 archive recorded twelve native cases per stack.
Its first indexed parameter assignment failed to replace the storage dtype;
that failure and unchanged contract remain preserved, followed by the explicit
whole-value `set_value` repair. Missing original process receipts were not
reconstructed; later receipt-only checks are distinct executions. Temporary
stage 2 used the default audited validator, recorded the same two exact witnesses
under CPU disabled/all, and retained sum/mean/HIGHEST barriers. It did not
establish a generic transformer or complete model compatibility.

Portable v1 copied the protected context in the wrong evaluation order and
first encountered the opset-26 bitcast barrier. Its maintained test failed,
although its aggregate report said `validated`; that status does not erase the
failure. The v1 harness was not execution-pinned, so the later reconstructed
context is explicitly labeled reconstruction. V2 restored the accepted body AST
and passed eleven primary-stack tests. V3 persisted actual native arrays before
comparison and prefroze known exact/minimum expectations, with fresh dual-stack
eleven-pass receipts. The final source, v1 failure and v2 result are not relabeled
as the same epoch. No numerical criterion, input domain or protected barrier was
weakened to obtain the final result.

The [named runtime-parameter packet](named_runtime_parameters_findings.md) is a
separate shared prerequisite. Completing unchanged models still needs native
framework computations, their actual runtime controls/state and independent
numerical evidence. The [migration plan](../precision_migration_plan.md#next-implementation-work)
retains that work and the general fixed-source preparation requirement.

## Frozen evidence

| Artifact | SHA256 |
| --- | --- |
| `/tmp/jax2onnx-fixed-source-precision-preparation-decision.md` | `2c8a6596186cb9a8151d22740201966a0617912d3a0220121640c5f43ef19779` |
| `/tmp/jax2onnx-preparation-portable-final-v3/final-manifest.json` | `2eb360a7d214800c81cdc47422da6c1ee2a3ebc0712513c7100f8831127b305c` |
| `/tmp/jax2onnx-preparation-portable-astra-review-v3.md` | `24e701c23b8f25c564b7d055e0b26f74a933eaf668774c25c191590b8d822ace` |
| `/tmp/jax2onnx-preparation-portable-astra-verification-v3.json` | `440b89e4b1490a1df8968f0e660dfd9fdf2b81c4c1c82dddf5a13b7c3f7ed3bf` |
| `/tmp/jax2onnx-preparation-portable-final-v3/native-target-process-receipts.json` | `d12acd2774de404c3c207fcd2d05fbf39baf0e6a3678e30b728632ec59f1b5c6` |
| `/tmp/jax2onnx-preparation-portable-order-proof-v2/order-proof.json` | `362f9fec1141900c8965b23662d0b70dca332944c7485cabd44b2abd31139947` |
| `/tmp/jax2onnx-fixed-source-preparation-poc-20261018/preparation_poc/final-manifest.json` | `07cf51be62b3b25542dd016c2933ce6db9f765d95d036fd5ae637bd43afb803f` |
| `/tmp/jax2onnx-fixed-source-preparation-stage1-astra-review.md` | `c538330f95894a71f81101516c9067f4b869adc1ceca6b86d0dc165c74ea5ffe` |
| `/tmp/jax2onnx-fixed-source-preparation-poc-20261018/preparation_stage2/final-manifest.json` | `cd2dfab320e2203ed18fccf03b2cc9c33d75f389cfd21d3a5f01e1e8c131e03a` |
| `/tmp/jax2onnx-fixed-source-preparation-stage2-astra-review.md` | `46431869a0382b29f87f77f202f074f87f35ab256e311d36ef039c19df322600` |
| `/tmp/jax2onnx-preparation-portable-main-application-v1.json` | `bcb6817cdbc7c00e85bcf8ca53d0c7c27601d0070b1956eaa67f28437863d7cc` |
