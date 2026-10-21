# Precision Contract Experiments

The latest [native transformed-callable checkpoint](source_integration_findings.md#native-transformed-callable-capture)
retains original `vmap`/traceback state and exact caller axis context. Pretyping
focused checks pass **441 PRIMARY / 437 SECONDARY / 437 MINIMUM cases**;
actual production typing on **559 files** and the **17-factory RNG inventory**
subsequently pass. Final typed-source affected selections pass **919 cases each
on SECONDARY and DIRECTMIN**, zero failures/errors/skips. The current full
PRIMARY suite passes **12,762 unique cases**, retaining all 12,600 prior IDs and
adding 162 wrapper cases, with zero failures/errors/skips. Documentation, normal
hooks and current commit gates remain pending. Original MLP2 source/export
validates, while its raw callbacks remain FAILED, criteria remain None and
runtime/numerics UNVERIFIED.

Preceding precision and shared file-storage work is recorded in
[Native Equinox Dropout and shared static uniform](source_integration_findings.md#native-equinox-dropout-and-shared-static-uniform).
That storage-v3 source epoch has 13 Python paths and 1,108 held inputs. The same
five storage cases pass on MINIMUM and PRIMARY; seven direct quality checks
pass. All 26 affected owners pass with **1,752 supported minimum cases** and
**1,813 cases on each modern stack**, zero failures/errors/skips. The 61 exact
minimum target-capability cases remain outside positive qualification.
The full PRIMARY suite records **12,600 PASS / 0 FAIL / 0 ERROR / 0 SKIP of 12,600 cases**, in 1451.264381 seconds. No complete minimum or model-numerical
acceptance is claimed.

At the qualification cut recorded here (2026-10-21T20:59:01.968117+00:00), strict
documentation, rendered-link checks, normal commit hooks, staging/qualified-byte
checks and commit had not yet run. Later closeout evidence is recorded separately.
The [completed storage-v3 closeout](source_integration_findings.md#storage-v3-local-closeout) records local commit
`dde196609d` and its completed documentation, seven normal hooks and held-byte gates.

The shared serializer corrects ordinary public file exports as well as private
precision work, preserving root-external/Function-inline storage and exact
payloads across ONNX versions. Public precision policy/options/defaults remain
LEGACY. Earlier minimum/storage/literal-helper failures, diagnostic artifact
variants and focused source epochs remain separate; raw callbacks/criteria and
UNVERIFIED runtime/numerics are retained.

The immutable c686 map retains **98 ordinary terminals / one original incomplete /
3,605 unstarted of 3,704**. It remains earlier-source product coverage, not current
runtime/numerical qualification. The accepted four-cell source-only context
capture is bound to earlier `b945a957` bytes and does not authorize current
numerical execution; the separate R2 hold remains. All fixed sources, existing
functionality and full B–F acceptance remain required.

Private same-width complex64/complex128 Add is committed at `18f3fcd7`, with
finite native/CPU qualification for static shapes at opsets 23/26. It retains
declared pair storage and JAX's chosen width. The [latest checkpoint](source_integration_findings.md#complex-add-and-partial-ordering-checkpoint)
separates those samples from the partial initial raw map and the **FAILED**
ordering-kernel diagnostics. Public behavior remains **LEGACY**; all existing
functionality, fixed sources and full B–F acceptance remain open.

The preceding private [CNN, pooling, scientific FP64 and raw-replay checkpoint](source_integration_findings.md#cnn-pooling-scientific-fp64-and-raw-replay-checkpoint)
is committed at **`118ddd8b`** after independent finite-checkpoint acceptance,
corrected dual-stack checks, the complete PRIMARY suite and all seven normal
code hooks. It retains the unchanged full-CNN nonzero comparison, native pooling
and selected scientific FP64 at opset 21. The raw lane's 49 maintained controls
do not run the full 1,852-row / 3,704-caller map. Public behavior stays **LEGACY**;
the full B–F migration remains open.

The preceding explicit FP64 Linear code checkpoint is `b2a866bb`. Its strict
documentation build passed, and its eight-document update is committed at
`97929998`. The canonical findings retain its original runtime and failed epochs.

Earlier saved-model evidence: the original six source selectors have independent structural and
finite CPU-sample acceptance for eight saved artifacts on each supported stack:
**384 returns / 16 sessions per stack**, **768 returns / 32 sessions total**.
Their immutable 117-derived source epoch is separate from current MAIN Conv/Scan
qualification. Six original callbacks per stack remain FAILED, two PASSED; stored
export reports remain UNVERIFIED. Public behavior stays **LEGACY**.

Conv and the Scan metadata remedy are committed at **`3223d080`** after
independent finite-checkpoint acceptance. Earlier focused failure epochs are retained with corrected secondary counts. Latest
PRIMARY passes 687 cases; SECONDARY has 684 PASS / three legacy runtime-fusion
SKIP, pytest exit 0 but strict wrapper exit 99. All 586 converter cases pass per
stack. The reviewed fixture correction preserves production. The complete PRIMARY suite passes
**10,504 cases / 31,512 phases**, with zero failures/errors/skips and all 1,091
source/input files held through all seven normal code hooks and commit. The
worktree is clean at commit verification. The preceding committed loop
checkpoint remains `c033d301`, with its accepted 477 focused cases per stack,
10,395-case full run, and seven-hook evidence retained below.

The [preceding elementary findings](source_integration_findings.md#elementary-and-reduction-checkpoint)
separate finite Asin positives, eight failed Sign numerical reports, 44
numerically unverified original compositions and two failed raw Gather callbacks
with separate positive finite comparisons. These experiments inform the
[target design](../precision_target_state.md) and
[migration plan](../precision_migration_plan.md).

The first rows separate current implementation qualification, accepted saved
samples and the preceding committed loop checkpoint.
The remaining rows retain the preceding elementary/tensor/index/Erfc/inventory
scopes and earlier experiments with their original evidence.

| Experiment | Question | Evidence |
| --- | --- | --- |
| Original native transformed callables | Can replay retain original `vmap`/traceback dependencies and exact caller context? | [Wrapper findings](source_integration_findings.md#native-transformed-callable-capture): finite pretyping three-stack focused passes; actual typing/RNG, typed-source SECONDARY/DIRECTMIN 919 each and full PRIMARY 12,762 pass; current docs/hooks/commit pending. Original MLP2 callbacks remain FAILED and runtime/numerics UNVERIFIED. |
| Native Equinox Dropout and shared static uniform | Do original random/state, branch-result and tensor-storage contracts survive the same sources and settings? | [Current findings](source_integration_findings.md#native-equinox-dropout-and-shared-static-uniform): current same5 storage and quality7 pass; affected owners pass supported minimum1752 and both modern1813. Exact61 minimum capability cases remain requirements; full PRIMARY has 12,600 cases with 0 failures, 0 errors and 0 skips; docs/hooks/stage/commit were unrun at the recorded qualification cut. Earlier failures remain retained. |
| Complex Add and ordering kernels | Which declared complex widths preserve finite samples, and where do bare ordering kernels fail? | [Latest findings](source_integration_findings.md#complex-add-and-partial-ordering-checkpoint). The [standalone ordering probe](ordering_kernel_probe.py) preserves its predeclared inputs, exact criteria and FAILED results; it performs no SOURCE export or original-row replay. |
| CNN, native pooling, scientific FP64 and raw replay | Do unchanged originals retain exact source structure and separately declared numerical criteria? | [Current findings](source_integration_findings.md#cnn-pooling-scientific-fp64-and-raw-replay-checkpoint): reviewed implementation; corrected scientific 141 cases / 423 phases pass per stack. The failed PRIMARY diagnostic1798 and its 49 raw-lane / 46 CNN controls are retained. Corrected dual37 and full PRIMARY pass; all seven normal code hooks pass and the exact 28-path checkpoint is committed at `118ddd8b`. Raw callbacks/criteria=None and native-bitwise subnormal FAILED evidence remain separate. |
| Explicit FP64 Linear preparation | Can the unchanged full Linear receive explicitly selected FP64 computation/storage through ordinary SOURCE validation? | [Latest findings](source_integration_findings.md#explicit-fp64-linear-preparation-checkpoint): eleven applied paths; accepted dual936/2,808 on unchanged production, fresh dual145/435 and full PRIMARY v4 10,596/31,788 PASS. Raw0/raw3 retain 24 prepared cases / 96 target returns per stack, a priori Fraction bounds, failed raw Gemm callbacks, criteria=None and UNVERIFIED reports. Full v3 failures retained; independent readiness and seven normal hooks pass, with exact qualified bytes committed at `b2a866bb`. Strict docs build passed; docs committed at `97929998`. |
| Preceding native Conv/Scan integration | Do actual physical shapes survive child Function lowering while original sources/options remain intact? | [Latest findings](source_integration_findings.md#original-six-saved-samples-and-native-conv-integration): committed at `3223d080`; earlier failed epochs corrected and retained. V3 PRIMARY 687 PASS; SECONDARY 684 PASS/3 unchanged legacy SKIP, pytest exit 0 / strict wrapper exit 99. All 586 converter cases pass per stack; legacy post-skip numerics unexecuted. Complete PRIMARY 10,504 cases / 31,512 phases PASS, zero failures/errors/skips; all seven normal hooks pass and all 1,091 inputs hold through commit, with a clean worktree at verification. |
| Original-six saved samples, both stacks | Do all eight original artifacts per stack preserve their frozen native/state/flag comparisons? | [Latest findings](source_integration_findings.md#original-six-saved-samples-and-native-conv-integration): independent bounded structure and 384 actual returns / 16 CPU sessions per stack. Six raw callbacks FAILED, two PASSED per stack; stored reports UNVERIFIED and PRIMARY exit99 retained. |
| Masked FP64 saved sample | Does the unchanged artifact pass its original criterion on both ORT stacks? | [Latest findings](source_integration_findings.md#masked-fp64-saved-sample): four bitwise-identical target returns, eight comparisons / 48 scalars against saved PRIMARY eager/JIT references, four sample-API validations; original rtol=5e-8 / atol=1e-12. No fresh SECONDARY native or whole-family claim for this saved artifact; later Issue52 opset-21 evidence is recorded in the current checkpoint. |
| Ordinary native loop admission | Do actual native loop bodies and typed roles survive the shared SOURCE validator? | [Latest findings](source_integration_findings.md#loop-admission-and-first-saved-model-samples): actual dual477 / 1,431 PASS; finite scan at opsets 23/26 and while at opsets 18/23/26, original metadata and all 51 raw obligations retained. Complete PRIMARY 10,395/31,185 and seven normal code hooks PASS; committed at `c033d301`. |
| First saved original014/0 | Does this unchanged saved artifact preserve its complete bounded Function structure and exact captured target samples? | [Latest findings](source_integration_findings.md#loop-admission-and-first-saved-model-samples): independent structural acceptance; actual two CPU sessions / 32 calls pass the original reference comparisons. Raw test/callback FAILED and earlier report UNVERIFIED; later full saved-artifact follow-through is recorded above, while whole-model acceptance remains open. |
| Elementary/reduction and same-width Asin | Do bounded native calls and owning lowerings preserve the reviewed typed/value contracts? | [Current findings](source_integration_findings.md#elementary-and-reduction-checkpoint): actual 183/549 PASS per stack; finite Asin F32/F64 and B=2/5 positives; eight Sign numerical reports FAILED, 44 compositions UNVERIFIED, both raw Gather callbacks FAILED with separate positive finite comparisons. Code `11748451`, full PRIMARY 10,237/30,711, dual focused 1,937/5,811 and seven normal hooks PASS; numerical limits remain. |
| Common tensor/index operations | Do unchanged raw tensor/index contracts preserve values, weakness, shapes and index semantics? | [Current findings](source_integration_findings.md#tensor-index-erfc-and-optional-inventory-checkpoint): 296 cases per stack; 63 original rows retained; 56 selected rows / 112 tested caller contexts contain 109 positive artifact contexts and three native/configuration negatives; seven other rows were deferred in that selector; the later elementary selection separately checks the two trigonometric Gather rows. |
| Source-typed Erfc | Does the reviewed expression preserve typed structure, and which configured CPU outputs pass? | [Current findings](source_integration_findings.md#tensor-index-erfc-and-optional-inventory-checkpoint): graph/schema evidence at opsets 23/26; configured CPU numerical evidence at opset 23 only, with finite direct/ordinary/named positives; full-seven-output and native-agreement gates FAILED, low-precision numerics UNVERIFIED. |
| Original GPT_MLP numerics | Does the unchanged ordinal-0 FP32 model pass finite criteria in both caller-x64 contexts? | [Current findings](source_integration_findings.md#elementary-and-reduction-checkpoint): four actual focused15-v2 caller/stack cells independently accepted at strict 1e-5, original 768 → 3,072 → 768 model/state/seed/zero-Dropout/approximate-GELU contract and typed cubic body preserved; export reports keep runtime/numerics UNVERIFIED, bounded sample receipts carry acceptance. No FP64 or all-model claim. |
| Optional inventory retention | Do absent/filter-selected optional environments preserve every reviewed obligation? | [Current findings](source_integration_findings.md#tensor-index-erfc-and-optional-inventory-checkpoint): 1,852 rows / 3,704 caller obligations; initial CLI 4 PASS / 2 FAIL; supplementary secondary MaxDiffusion defaults fails on missing tokamax. No optional SOURCE acceptance. |
| Native Sum | Do original reduction options and computation types survive interception? | [Sum findings](source_integration_findings.md#native-sum-forwarding-and-measured-composition): committed native forwarding, accepted MAIN integration and retained BitCast barriers. |
| Native LayerNorm and mean | Do original framework calls retain their native normalization and reduction? | [Forwarding findings](source_integration_findings.md#layernorm-and-mean-native-forwarding): accepted call preservation, retained full-world regressions, and the historical default-export barriers. |
| Named runtime parameters | Can declared keyword values keep their real types and variable bindings through the root, Function and If? | [Named-parameter findings](named_runtime_parameters_findings.md): audited binder, finite sample evidence and separate all-22 progress ledger. |
| [Fixed-source preparation](fixed_source_preparation.py) | Can an explicit pretrace choice change one fixed cast while preserving protected work and actual weights? | [Preparation findings](fixed_source_preparation_findings.md): portable finite proof, distinct native references, two exact target witnesses and retained profile barriers. |
| [Source entry point](source_export.py), [demo](demo.py) | Can real lowerings preserve the traced calculation? | [Integration findings](source_integration_findings.md): inputs, scopes, call-time settings, cache isolation and current tests. |
| Original MLP state and bounded normalization | Do original model controls, explicit state and native normalization survive the private pipeline? | [Current acceptance](source_integration_findings.md#mlp-state-and-bounded-normalization-acceptance): finite CPU model/state evidence, permanent regression and the actual exported-body proof limit. |
| Named NNX Function state | Can original trace-local owner updates survive named calls and unblock unchanged models? | [NNX findings](source_integration_findings.md#nnx-function-state-checkpoint): 13 new and 45 retained cases pass per stack; original-model exploration reaches the measured matrix barrier with no target execution. The code checkpoint `437d41fe`, complete 9,557-test primary suite, secondary 58-case selection and all seven normal hooks pass. |
| Normalization options and tensor matrices | Do unchanged options and tested matrix compositions retain the source contract? | [Joint findings](source_integration_findings.md#joint-normalization-and-matrix-integration): options 131, focused 159, matrix 44 and retained 348 pass on both stacks; checkpoint `39ddee8c` passes primary suite (9,667 cases) and all seven normal hooks; six corrected checks pass on both stacks. Original graph callbacks and complete models remain open. |
| Shared-batch normalization | Do the original default LayerNorm rows retain their shapes, parameters, opsets and numerical contract? | [Shared-batch findings](source_integration_findings.md#shared-batch-normalization-checkpoint): accepted finite checks and code checkpoint `bb7a7d76`; retained failed epochs and remaining family/model work. |
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
| Leading-B matrices and bias | Can one native symbolic batch survive MatMul, singleton Add and the original Linear bias path? | [Leading-B findings](leading_batch_matrix_bias_findings.md): reviewed finite semantics, exact raw-source provenance, repaired symbolic preparation and separately gated five-row numerics. |
| Native Exp and Leading-B Mul | Do native weak typing and explicit broadcast survive exponential/multiplication composition? | [Pointwise findings](pointwise_mul_exp_findings.md): accepted four-path semantics and 1,489 actual-main checks per stack, with fixed finite CPU samples and separate BF16 structural limits. |
| Native Where and typed JIT | Can native selection and pure compiled bodies preserve actual typed inputs, constants and scopes? | [Where/JIT findings](where_jit_admission_findings.md): accepted private semantics, 2,661 pre-repair main checks and 89 fresh restoration checks per stack; retained strict zero-sign failures and the raw gather callback. The later [copy-based repair](source_integration_findings.md#exact-floating-selection) addresses the bounded floating selection case. |
| Gather diagnostic counterpart | Can the original indexing/trigonometry source receive an exact structural check while preserving its raw failure? | [Gather findings](gather_counterpart_findings.md): separate SOURCE structure validation for one saved representation; raw callback FAILED and product/runtime/numerics UNVERIFIED. |
| Typed values and masks | Can native discrete values, masks and static complex components retain their declared semantics? | [Typed findings](typed_values_and_masks_findings.md): committed bounded rules, retained raw/cast failures, accepted integration and new original-model boundary evidence. |

The [current checkpoint record](source_integration_findings.md#loop-admission-and-first-saved-model-samples)
binds the actual loop result and both retained failed focused epochs. The
preceding elementary record keeps its nine failed samples and separate
callback/composition limits; a passing diagnostic test does not accept a failed
numerical gate. The accepted `11748451` code checkpoint and all earlier failed
full/model/Erfc epochs remain historical evidence.

The original-six/eight-artifact saved samples are now accepted on both stacks;
original callbacks and stored reports retain their distinct outcomes. The
PRIMARY capture failure and missing final origins remain historical limits.
The Conv/Scan and explicit FP64 Linear checkpoints are committed, including
the latter's strict documentation build and documentation commit `97929998`.
The current CNN/scientific/raw-replay code checkpoint is independently accepted
and committed at `118ddd8b`. Its nonzero CNN predicate is finite sample evidence, with provider-wide
algorithms **UNVERIFIED**. Exact source subnormal selection does not turn a failed
native-bitwise sample into numerical acceptance. General fixed sources,
optional models, low precision and the full B–F migration remain open.

## Run it

The standalone ordering feasibility probe uses the installed stack and writes
only to a new directory outside the repository:

```bash
poetry run python internal_docs/precision/poc/ordering_kernel_probe.py --out /outside/repo/new-ordering-run --opset 23
```

It saves every native/target observation before comparison and exits nonzero
when a selected check remains **FAILED**. Bare-kernel execution does not admit
an operation to SOURCE or supply missing raw criteria. Its tracked executable
body is identical to the reviewed v2 probe used for the recorded dual-stack run.

The finite preparation proof is now reproducible from tracked files:

```bash
PYTHONPATH=$PWD python -m internal_docs.precision.poc.fixed_source_preparation --output /outside/repo/new-run
```

Use a new output directory outside the repository. The command records fresh
native arrays, contracts, graphs and actual target outputs. It exercises the
ordinary private audited exporter and records remaining barriers; it does not
make the prepared computation equivalent to the original or enable a public
precision mode. See the [preparation findings](fixed_source_preparation_findings.md)
for the exact scope and maintained test command.

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
post-processing and serialization. The current checkpoint adds the bounded
common tensor/index and Erfc families at the recorded opsets, types and shapes;
its exact scope and open original rows are linked above. Generic SOURCE
admission accepts explicit opsets 16, 18, 21, 23, 24, 26 and 27, with
operation/type-specific eligibility. Earlier static floating Function/If
samples retain their recorded opsets 23/26. The finite
shared-B extension adds equal rank-two floating inputs through Add and named
composition. The Leading-B checkpoint adds one atomic native B at axis zero of
rank-two/three matrix and bias contracts, with FP32/FP64 MatMul and positive
static tails. The Pointwise extension adds native floating Exp and equal-tail
shared-B/static-1 Mul at those ranks, with exact source weakness.
The later Where/JIT checkpoint adds static BOOL dependencies, bounded floating
selection and pure closed JIT bodies. Broader symbolic operations, unsupported JIT forms, effects and unaudited
operations retain temporary gates. Native scan/while now have the finite
ordinary SOURCE contracts in the [latest checkpoint](source_integration_findings.md#loop-admission-and-first-saved-model-samples),
including static complex/empty storage and actual recursive bodies. All 51 raw
loop rows remain obligations. Same-width BitCast still requires explicit
opset 26; other widths and unaudited forms are not admitted. No opset is raised
automatically. The earlier diagnostic-loop records retain their historical
scope; the loop packet's complete 10,395-case PRIMARY run and all seven normal
code hooks now pass at committed checkpoint `c033d301`.

The audited matrix profile retains its FP16-operands/preferred-FP32 branch for
positive static rank-2/rank-3 standard axes. It now also admits same-type
FP32/FP64 ordinary matrices at ranks 2–4, with one contracting/free axis per
operand and ordered paired batches. Both branches require default precision;
explicit algorithms, general contractions and symbolic forms beyond the finite
Leading-B extension stay excluded. The FP16 branch and its widening behavior
are unchanged, with no native low-precision hardware or performance promise. The original D2a records
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
earlier integration failures remain recorded. Shared-B Add and the finite
Leading-B matrix/bias semantics have focused implementation evidence. The
310-row symbolic inventory and all seven NCHW rows still require broader
acceptance accounting. The finite dense Min/Max lowerer now passes independent
review at opset 23 for equal FP16/FP32/FP64 operands. New zero-sign samples pass, while strict
subnormal/grid failures remain and no default profile is expanded. B3 tooling
selects 13 unchanged rows/26 artifact-only cells; side and main replays each check
24 artifacts and retain two unresolved outcomes per stack, with zero ORT.
That historical artifact checkpoint adopts no numerical criterion. The new
separate FP32 numerical counterpart retains the historical packets' None
criteria/history and records all 24 selected cells under the approved three-predicate criterion.
Its sample results do not audit general lowering or change default admission.
The historical gather revision stopped at custom Where. The newer Where/JIT
supplement preserves the complete native typed trace but still fails the
unchanged raw callback for the false constant's broadcast shape. X64-off's
configuration conflict remains. The later [Gather counterpart](gather_counterpart_findings.md)
adds a separate checked diagnostic representation with no runtime or numerical
acceptance. The [next implementation work](../precision_migration_plan.md#next-implementation-work)
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
