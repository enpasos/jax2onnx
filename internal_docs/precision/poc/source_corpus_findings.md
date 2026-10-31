# First source-corpus diagnostic batch

Status: measured local evidence, 2026-10-18. The first B3b batch observed exactly
36 cells on each supported stack through the shared private SOURCE facade.
This is artifact evidence; every cell retains unverified lowering, runtime and
numerical acceptance. Public admission and legacy behavior are unchanged.

## Frozen scope and results

The batch used the inventory at `158ce625`, SHA256
`9ac003ba1e05b94cec26c16985892c506b617909534de320e223796c3cc8cd2b`:
1,802 raw rows, 3,604 x64-off/on cells, and then 186 handwritten-test files with
888 test functions. Later handwritten-test additions do not change these raw
identities or contracts. The fixed selection is ordinal zero for add,
convert_element_type, sin, reshape, transpose, gather, reduce_sum, dot_general,
conv_general_dilated, jit, pjit, lax.fori_loop, while, scan, complex, real, imag
and fft. All 18 selected rows have unresolved numerical criteria.

Both stacks produced the same outcomes; counts below are **per stack**.

| Outcome | Cells | Meaning |
| --- | ---: | --- |
| Diagnostic artifact checked | 26 | Schema and artifact checks passed; lowering remains unaudited. |
| Export implementation gap | 4 | Ordinary FP32 dot_general and complex64 FFT each failed in both x64 contexts. |
| Structural form pending | 3 | Gather with x64 enabled and both scan cells exceed the bounded diagnostic validator. |
| Trace lineage pending | 2 | Both pjit cells contain an unsupported partition-spec fingerprint. No export was attempted. |
| Source configuration conflict | 1 | Gather's FP64/INT64 inputs conflict with x64 disabled; preparation rejects narrowing. |

Independent original-JAX references passed for 35 cells; the configuration
conflict failed preparation. Structural diagnostics passed for 30 cells and
remain unverified for six. Schema passed for 26; artifact checks passed for 26,
failed for four and remain unverified for six. These stage counts overlap and
must not be added. All 3,568 unselected cells remain unchanged, including 126
recipe-pending cells.

## Concrete gaps

| Case | Original contract and observed boundary | Next work |
| --- | --- | --- |
| dot_general | FP32 `[3,4]` and `[4,2]` produce FP32 `[3,2]`, with `preferred_element_type=None`. SOURCE currently implements the earlier bounded FP16/preferred-FP32 matrix case and rejects this ordinary FP32 case. | Audit and implement the ordinary matrix contract without transforming its source or inputs. |
| FFT | Original input and output are logical complex64 `[4]`. The shared representation stores real/imaginary pairs, but FFT lowering still requires a native complex IR dtype. | Repair the representation boundary; this is not an invalid real-valued input recipe. |
| gather, x64 off | Original FP64 data and INT64 indices would be narrowed by JAX. | Retain the configuration conflict and supplied-payload evidence; do not treat it as a lowering defect. |
| gather, x64 on | The original trace includes `mul(out_dtype=None)`, outside the diagnostic table. | Review that exact parameter form with positive and malformed/non-None tests before rerunning this cell. |
| scan | The plugin-activated trace includes `jax.numpy.zeros`, outside the explicit diagnostic families. | Inspect that form before extending the validator; no scan artifact is claimed here. |
| pjit | Original trace fingerprinting records three unresolved `spec: unsupported jax.P` entries. | Add a bounded, stable partition-spec representation before exporting these cells. |

The diagnostic validator is substantive and recursive; unknown forms remain
pending. A successful diagnostic never enlarges `source-static-v1`. The
`source-diagnostic-v1` report and strict reader prevent unaudited lowering from
being relabeled as validated.

## Evidence and isolation

The repaired tooling passed 125 focused tests per stack and an independent
58-test rerun per stack. Review closed two findings: restore all five recorded
JAX settings, including hidden global changes under caller contexts; and classify
a valid vmapped while predicate as pending rather than malformed. Known
malformed scalar while signatures still fail.

The batch's 14 isolation checks passed on each stack. Selection and raw-contract
preflight ran before model execution. Diagnostics supplied no numerical samples
and no optimization setting; an `InferenceSession` deny guard observed zero
ORT session attempts. Audited add/reduce_sum controls retained their artifact
hashes before and after the batch, default sin admission stayed excluded, and
caller settings were restored. No exporter, JAX result, model bytes or checker
was substituted. No retry changed a recorded failure into success.

Each cell retains immutable stage history, requested/observed/export settings,
environment versions, original typed trace and digest, logical/physical input
and reference bytes, row/scenario identity, artifact digest and diagnostic
scopes. The maintained tooling is
[source_precision_probe.py](../../../scripts/source_precision_probe.py), with
[independent preparation](../../../tests/extra_tests/converter/source_precision_support.py)
and [structural diagnostics](../../../tests/extra_tests/converter/source_precision_diagnostics.py).
Its default action only creates a plan; execution requires an explicit bounded
selection. The local batch runner is supplementary evidence, not a maintained
repository entry point.

| Stack | Recorded environment |
| --- | --- |
| Python 3.12.9 | JAX/JAXLIB 0.11.2, NumPy 2.5.4, Flax 0.12.10, ONNX 1.23.2, onnx-ir 1.0.0, ORT 1.31.0 |
| Python 3.11.12 | JAX/JAXLIB 0.10.2, NumPy 2.4.6, Flax 0.12.8, ONNX 1.22.0, onnx-ir 1.0.0, ORT 1.29.0 |

Both used the same frozen checkout with CPU and one-thread numerical-library
settings. Installed distribution metadata reported jax2onnx 0.17.1 and 0.16.1,
respectively; those metadata values are not the imported checkout revision.
Local evidence files are `/tmp/jax2onnx-b3b-artifact-36-findings.md` and
`/tmp/jax2onnx-b3b-artifact-36-{py312,py311}.{full,compact}.json`, with corresponding
logs. These temporary files are not portable repository dependencies.

## Checkpoint boundary

C3b promotion-context replay, C3c finite complex-loop representations and
required SOURCE finalization have independent focused acceptance. Their wider
selections passed 359, 404 and 523 tests per stack, respectively; C3c also passed
75 generated tests per stack. Independent review ran all 52, 121 and 54 new
tests, respectively. These selections overlap and are not a combined total.

The latest integrated primary-stack run passes **5,650 tests** (13,104 warnings,
356.07 seconds). A preceding run ended with one failure and 5,647 passes because
collection enriched raw registration dictionaries in place. The exact-contract
preflight correctly detected that drift. The accepted shallow-copy repair
preserves generated metadata and variants; 127 focused tests pass per stack,
including independent reruns of both new regressions and generated-collection
controls.

C3b, C3c and finalization are committed at `ea6a8da7`, `8ba3c681` and `f1ee71b3`;
B3 tooling, metadata ownership and the inventory refresh are committed at
`3dbec1b8`, all with actual hooks passing. The refreshed inventory has 191
handwritten-test files, 944 functions and 192 migration additions, preserving
all 752 original anchors, all 1,802 raw contracts and optional entries. Fresh
inventory comparisons pass on both stacks. This does not rebind the measured
batch: its original `9ac003ba...` baseline and 36-cell evidence remain intact.
The earlier 5,363-test passing run belongs to `158ce625`.
The [work plan](../precision_implementation_work_plan.md) owns current validation
and commit status. No result here completes phases B-F.


## Two-cell matrix supplement on 2026-10-18

A separate observation after the D2a lowering repair repeats exactly
`plugin::dot_general`, ordinal 0, in the original raw-x64-off/on scenarios on
each stack. The original FP32 `[3,4]` and `[4,2]` operands, FP32 `[3,2]` result,
`preferred_element_type=None`, factories, reference bytes, trace fingerprints,
settings and environment fields are unchanged. The inventory remains the
historical `158ce625` baseline with SHA256 `9ac003ba...` above. The first 36-cell
batch and its failed ordinary-dot observations remain intact.

| Supplement gate | Result on each stack |
| --- | --- |
| Independent source reference and structural diagnostic | VALIDATED for both cells |
| ONNX schema | VALIDATED for both cells |
| Artifact acceptance | FAILED for both cells: unchanged raw callback requires Gemm with zero bias; the bounded SOURCE lowering emits MatMul |
| Lowering, runtime and numerics | UNVERIFIED; numerical criteria remain unresolved and zero ORT sessions ran |

The raw helper still records the generic `implementation-gap` category with
`Raw metadata structural callback failed`. The precise next task is to audit
and migrate the SOURCE structural oracle. No raw metadata, historical result or
criterion was changed to force a pass. There is no accepted artifact digest,
newly closed corpus row or numerical acceptance from this supplement.

Independent inspection confirms preserved prior observations, all recorded
reference fields, unchanged before/after control artifacts, zero ORT attempts
and all isolation invariants. The separate fresh plan has only these two new
observations; it does not reset the earlier 36-cell batch. Imported module paths
and hashes were verified against production snapshot
`8463f856ca832d9f75e192d7b3c15faf4dfcf842560c2daca0c8e2bf795b0c65`.
Those bytes are now committed at `c528a894`/`2011b140`; the recorded observation
checkout `182e03f6` correctly identifies their earlier uncommitted observation.

Local records are
`/tmp/jax2onnx-d2a-b3-dot-8463f856ca83-py{312,311}.{compact,full}.json`;
independent inspection is `/tmp/jax2onnx-d2a-b3-dot-astra-review.md`. These paths
are provenance, not maintained repository commands. The
[matrix and optimizer findings](matrix_and_optimizer_findings.md) record the
bounded implementation, distinct runtime samples and current checkpoint checks.
