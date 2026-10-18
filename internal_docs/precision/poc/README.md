# Precision Contract Experiments

Status: bounded private integration, 2026-10-18, beyond `main` at `fae17822`.
The converter carries SOURCE policy through selected real lowerings and scopes.
Public options and legacy defaults retain their meanings; no public source mode
is available. These experiments inform the
[target design](../precision_target_state.md) and
[migration plan](../precision_migration_plan.md).

| Experiment | Question | Evidence |
| --- | --- | --- |
| [Source entry point](source_export.py), [demo](demo.py) | Can real lowerings preserve the traced calculation? | [Integration findings](source_integration_findings.md): inputs, scopes, call-time settings, cache isolation and current tests. |
| [Numerical contract](numerical_contract.py) | Do result types, wider sums and intermediate rounding survive execution? | [Numerical findings](numerical_contract_findings.md): actual exports, independent controls and failed runtime acceptance. |
| [Migration calls](migration_ux.py) | Can callers predict legacy/source differences and diagnostics? | [UX findings](migration_ux_findings.md): input defaults, tracing configuration, weak inputs and option conflicts. |
| Source corpus | What does the first frozen 36-cell batch actually establish? | [Corpus findings](source_corpus_findings.md): 26 artifacts per stack, explicit failures/pending cells, and unverified lowering/runtime/numerics. |
| Fixed-source parameters | Which existing runtime controls must survive unchanged model implementations? | [Compatibility findings](fixed_source_compatibility_findings.md): all 22 raw rows, original state prerequisites, real Function/If/Loop dependencies and the later three-artifact/19-pending diagnostic pass per stack. |
| Recording, packaging and array creation | Do shared export options and native promotion preserve the source contract? | [Option and array findings](option_and_array_creation_findings.md): committed C4a/C4c and D1a/D1b checks, real named BOOL operands, native sequence binding and explicit target failures. |
| Ordinary matrices and Dropout cleanup | Do bounded matrix contracts and the shared optional-input rewrite preserve actual values? | [Matrix and optimizer findings](matrix_and_optimizer_findings.md): committed D2a and shared repair, exact subnormal failures, corrected witness provenance and separate corpus evidence. |
| Static layouts and native binary binding | Can external storage and native promotion preserve unchanged source computations? | [Layout and binary findings](layout_and_native_binary_findings.md): committed binder/Mul and C4b/observer repairs, passing 6,351-test integration after ten retained regressions, signed-zero failures and future symbolic work. |
| Shared symbolic batch and Min/Max feasibility | Can one artifact preserve a shared batch contract, and what does the finite selection audit establish? | [Shared-batch findings](symbolic_batch_and_minmax_findings.md): focused Add/named-composition acceptance, strict sample binding, retained BF16/subnormal failures, measured selection costs and planned artifact-only corpus tooling. |
| Min/Max lowering and corpus replay | Does the finite zero-sign representation survive real export, and how far do unchanged raw scenarios progress? | [Checkpoint findings](minmax_and_corpus_replay_findings.md): committed lowerer and repaired B3 tooling, retained target failures/cost and artifact-only outcomes; 6,466-test integrated run and actual hooks pass. |
| Native trig and fixed corpus numerics | Does native dispatch preserve the source, and do the fixed FP32 samples meet all approved comparisons? | [Trig/numerical findings](native_trig_and_corpus_numerics_findings.md): committed binding and fixed sample evidence; gather gaps and unverified general lowering remain. |
| Native NNX Linear and reshape | Can unchanged model methods preserve native tracing and static diagnostic artifacts? | [Linear/reshape findings](native_linear_reshape_findings.md): committed native adapters, exact parameters and named aliases; 544 main checks per stack and a 6,620-test full pass. |
| Audited ordinary matrices | Can the finite matrix representation pass actual admission and retain the fixed numerical contract? | [Admission findings](ordinary_matrix_admission_findings.md): static FP32/FP64 audit, 750 main checks per stack and fresh audited FP32 sample results; historical failures remain. |

## Run it

From the repository root in the installed Poetry environment:

```sh
JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  poetry run python -m internal_docs.precision.poc.demo

JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  poetry run python -m internal_docs.precision.poc.numerical_contract

JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  poetry run python -m internal_docs.precision.poc.migration_ux

JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  poetry run pytest -q tests/extra_tests/converter/test_source_*.py \
    internal_docs/precision/poc
```

The demos keep models in memory. Source and UX implementations now live in
[private test support](../../../tests/extra_tests/converter/precision_test_support.py)
and [UX test support](../../../tests/extra_tests/converter/precision_ux_support.py);
the old module paths remain runnable compatibility entry points. The durable
source regressions live in `tests/extra_tests/converter` and run in normal CI.
Supplemental experiments in this folder still need an explicit test invocation.
See [recorded validation](source_integration_findings.md#recorded-validation)
for historical integration results and the
[preceding committed checkpoint](matrix_and_optimizer_findings.md#validation-and-provenance)
for the historical 6,183-test run. The [layout/binary checkpoint](layout_and_native_binary_findings.md)
records the passing 6,351-test run and three commits with actual hooks, retaining
the earlier failed full run and separate future work.
The [shared-batch checkpoint](symbolic_batch_and_minmax_findings.md) has 697 owner
and 241 independent focused checks per stack. Commit `80cb68b2` passed actual
hooks and that complete primary-stack run passed 6,391 tests. The preceding
[checkpoint](minmax_and_corpus_replay_findings.md), `fc3f9323`, commits Min/Max and
B3 after independent review, 323 focused passes per stack, both main replays and
inventory comparisons, a 6,466-test full run and all seven actual hooks.
The [trig/numerical checkpoint](native_trig_and_corpus_numerics_findings.md),
`d2292b89`, retains its 6,579-test full pass and scoped numerical observations.
The preceding [Linear/reshape checkpoint](native_linear_reshape_findings.md),
`791664f8`, retains its 6,620-test full pass and native adapter prerequisite.
The [ordinary-matrix audit](ordinary_matrix_admission_findings.md) now admits
bounded static FP32/FP64 dots through the real SOURCE profile. Main checks pass
750 cases per stack; 24 fresh audited artifacts and 48 CPU runs per stack pass
the unchanged selected FP32 criteria. Strict subnormal failures remain.
Independent post-run review accepts these samples.
The full primary-stack run passes **6,728 tests**; commit `45881d8d`
passes all seven actual hooks.

## Observable source behavior

For exact FP32 work after FP64 input `1e8`:

```python
def exact_region(x):
    y = x.astype(jnp.float32)
    return (y + 1) - y
```

JAX and the source experiment return FP32 `0`. Legacy double export preserves
the initial conversion but widens subsequent arithmetic and returns FP64 `1`.
The source contract preserves that model's chosen computation. To choose a
local minimum instead, model code can use `jnp.promote_types(x.dtype, jnp.float32)`;
typed export inputs instantiate that rule.

| Model choice | Current experimental evidence |
| --- | --- |
| Mixed FP64 calculation and shared rounded FP32 value | Direct mixed outputs, arithmetic and same-width bitcast retain the intended types and selected values. |
| FP16 work with selected FP32 arithmetic and cast-back | Chosen input, intermediate and output boundaries are retained on the tested samples. |
| Weak scalar plus explicitly FP16 one | Weak input `2048` returns FP16 `2048`; the strong input with the same shape/dtype returns FP32/FP64 `2049`. |
| Local x64 or matrix-precision setting around a named function | Captured settings survive body retracing and distinguish specialization. Cached source traces cannot leak into ordinary/legacy calls. |
| FP16 matrix inputs with preferred FP32 result | Bounded default-precision matrix lowering preserves FP16 inputs and returns FP32 `2049`. |
| FP32 squares followed by selected FP64 variance work | Actual source exports preserve exact variance `8388609` inline, in a Function and in both conditional branches. |
| FP16 square then sum | The graph is valid and executes, but the tested ORT targets miss the required rounding. Sample acceptance rejects it. |

The [review findings](source_integration_findings.md#review-findings) explain
three repaired defects: lost local tracing context, cached branch leakage
between policies, and legacy algebraic fusion changing source arithmetic.
Source now retains separate operations where those fusions lacked numerical
justification. The known FP16 runtime failure persists without converter fusion.

The [follow-up review](source_integration_findings.md#follow-up-review-on-2026-10-18)
closed two additional gaps: the shared optimizer now honors SOURCE eligibility,
and function specialization includes weak metadata in captured state and static
arguments. Both fixes have regular-suite regression coverage on both stacks.
The new exact FP32 reciprocal sample still differs under ORT optimization;
that profile fails its exact sample acceptance even with the correct source
graph. See the recorded numerical evidence before choosing a runtime profile.

## Scope and limitations

The private harness uses the actual tracing, IR context, plugins, optimizer,
post-processing and serialization. It checks static floating inputs and a small
primitive set at opsets 23/26, including selected Function/If cases. The finite
shared-B extension adds equal rank-two floating inputs through Add and named
composition. Broader symbolic operations, loops, scan, JIT regions, effects and
unaudited operations remain excluded.
Bodies are validated recursively. Same-width BitCast requires explicit opset26;
different-width bitcasts need a different representation and are rejected.
No opset is raised automatically. These are experiment limits, not universal
ONNX support classifications. Separate diagnostic regressions now cover finite
noncomplex/complex loops and JIT cases without broadening this product profile;
the [work plan](../precision_implementation_work_plan.md) records their exact gates.

The audited matrix profile retains its FP16-operands/preferred-FP32 branch for
positive static rank-2/rank-3 standard axes. It now also admits same-type
FP32/FP64 ordinary matrices at ranks 2–4, with one contracting/free axis per
operand and ordered paired batches. Both branches require default precision;
explicit algorithms, symbolic/general contractions and other pending forms stay
excluded. The FP16 branch and its widening behavior are unchanged, with no
native low-precision hardware or performance promise. The original D2a records
remain diagnostic; new audited artifacts and their sample outcomes are recorded
separately. Exact subnormal failures persist on the recorded CPUs.

BF16 structural validity does not establish runtime kernels. Function coverage
does not establish all nested combinations or every JAX setting. Names/layouts,
parameter materialization, other return modes and concurrent export remain
outside the harness. The normal public pipeline still uses its legacy policy. The shared private
facade separately supports names, reports, primitive recording and
IR/proto/standard/web-file returns. The new array-creation diagnostics preserve
native promotion/weakness, sequence arithmetic and selected Boolean CPU results.
Private scalar-BOOL runtime declarations use one real root binding map, including
framework Dropout, local defaults and static literals. These diagnostics do not
broaden the audited product profile. Exact sequence failures, BF16 runtime limits
and dynamic `arange` compatibility remain open. The shared Dropout optimizer
defect is repaired with bounded optional-input cleanup and preserved shared uses;
see the [repair evidence](matrix_and_optimizer_findings.md#shared-dropout-optimizer-repair).
Static NCHW boundaries and the corrected observer/naming dependency are committed
after independent review, the passing fresh full run and actual hooks. The ten
earlier integration failures remain recorded. Shared-B Add now has focused
implementation evidence; all 310 raw symbolic rows and seven NCHW rows remain
open. The finite dense Min/Max lowerer now passes independent review at opset 23
for equal FP16/FP32/FP64 operands. New zero-sign samples pass, while strict
subnormal/grid failures remain and no default profile is expanded. B3 tooling
selects 13 unchanged rows/26 artifact-only cells; side and main replays each check
24 artifacts and retain two unresolved outcomes per stack, with zero ORT.
That historical artifact checkpoint adopts no numerical criterion. The new
separate FP32 numerical counterpart retains the historical packets' None
criteria/history and records all 24 selected cells under the approved three-predicate criterion.
Its sample results do not audit general lowering or change default admission.
The gather revision produces no artifact: the native trig barrier is removed,
while x64-off's conflict and x64-on's custom-Where gap remain. The
[next implementation work](../precision_migration_plan.md#next-implementation-work)
completes remaining options and full source coverage. Web packaging has exact
serialized-payload evidence, while browser execution remains unverified.

## Numerical acceptance is a separate result

The numerical demo reports actual source exports separately from changed-source
candidates and independent hand-built graphs. A candidate passing one example
is not an automatic exporter repair. In particular, wider arithmetic with an
explicit FP16 intermediate cast passes the square/sum sample, while the original
composition fails on both recorded CPU runtimes and both optimization settings.

The [private CPU sample gate](../../../tests/extra_tests/converter/precision_runtime_support.py)
checks a supplied artifact, input/reference sample and execution profile.
`require_sample_acceptance` rejects failed and unverified results. The demo
shows this rejection for the original FP16 sample. Kernel execution failure is
distinct from numerical failure. An accepted sample does not certify all inputs
or every composition of allowed primitives.

The calling experiment uses `precision="source"` only as a private spelling.
Its supplied/normalized/traced properties, actual warnings and legacy comparison
remain useful before any public interface is finalized. Neither it nor the
CPU sample gate is a general deployment capability engine. Public availability
requires the full phase-F acceptance gate in the migration plan.
