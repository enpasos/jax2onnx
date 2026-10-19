# Precision Contract Experiments

Status: bounded private integration, 2026-10-19, beyond the public-behavior
baseline at `fae17822`.
The converter carries SOURCE policy through selected real lowerings and scopes.
Public options and legacy defaults retain their meanings; no public source mode
is available. These experiments inform the
[target design](../precision_target_state.md) and
[migration plan](../precision_migration_plan.md).

| Experiment | Question | Evidence |
| --- | --- | --- |
| Named runtime parameters | Can declared keyword values keep their real types and variable bindings through the root, Function and If? | [Named-parameter findings](named_runtime_parameters_findings.md): audited binder, finite sample evidence and separate all-22 progress ledger. |
| [Fixed-source preparation](fixed_source_preparation.py) | Can an explicit pretrace choice change one fixed cast while preserving protected work and actual weights? | [Preparation findings](fixed_source_preparation_findings.md): portable finite proof, distinct native references, two exact target witnesses and retained profile barriers. |
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
| Leading-B matrices and bias | Can one native symbolic batch survive MatMul, singleton Add and the original Linear bias path? | [Leading-B findings](leading_batch_matrix_bias_findings.md): reviewed finite semantics, exact raw-source provenance, repaired symbolic preparation and separately gated five-row numerics. |
| Native Exp and Leading-B Mul | Do native weak typing and explicit broadcast survive exponential/multiplication composition? | [Pointwise findings](pointwise_mul_exp_findings.md): accepted four-path semantics and 1,489 actual-main checks per stack, with fixed finite CPU samples and separate BF16 structural limits. |
| Native Where and typed JIT | Can native selection and pure compiled bodies preserve actual typed inputs, constants and scopes? | [Where/JIT findings](where_jit_admission_findings.md): accepted private semantics, 2,661 pre-repair main checks and 89 fresh restoration checks per stack; retained strict zero-sign failures and the raw gather callback. The later [copy-based repair](source_integration_findings.md#exact-floating-selection) addresses the bounded floating selection case. |
| Gather diagnostic counterpart | Can the original indexing/trigonometry source receive an exact structural check while preserving its raw failure? | [Gather findings](gather_counterpart_findings.md): separate SOURCE structure validation for one saved representation; raw callback FAILED and product/runtime/numerics UNVERIFIED. |
| Typed values and masks | Can native discrete values, masks and static complex components retain their declared semantics? | [Typed findings](typed_values_and_masks_findings.md): committed bounded rules, retained raw/cast failures, accepted integration and new original-model boundary evidence. |

The [named runtime-parameter checkpoint](named_runtime_parameters_findings.md)
removes the private facade's blanket parameter exclusion. Positional and named
values now share actual typed root arguments, native signature binding, symbolic
scope, exported names, reports and sample feeds. The independently reviewed
owner packet passes **1,026 affected-area cases per stack**. Finite FP32/BOOL
samples exercise root, Function and If scopes at B=2/5 with True/False/True:
each stack records six validation-API sessions/36 runs and six separate direct
capture sessions/36 runs. Those observations do not establish complete models.

The closed owner ledger preserves all 22 original model contracts and their
actual first barriers, with criteria None and runtime, numerics and product
support unverified. Its zero-input/B=1 native references measure tracing progress.
NNX MLP now reaches BatchNorm, Linen MLP reaches Dense, and GPT full/embeddings
reach zero-rate Dropout; the old before-factory parameter guard is no longer
these models' current first barrier. Original missing-key and immutable-state
errors remain recorded.

The separate [fixed-source preparation PoC](fixed_source_preparation_findings.md)
packages a finite, explicitly authorized pretrace cast change and actual NNX
parameter configuration. Its 11 maintained tests pass on each stack. Each stack
records twelve ordinary audited export attempts and four actual CPU runs of the
two exact witnesses; eight sum/mean/HIGHEST profile barriers remain. This shows
bounded preparation feasibility, not a generic transformer or all-model support.

Code checkpoint `8f823ac3` commits the named binder, portable preparation proof
and reviewed instance-lifetime cache repair: fourteen code/inventory paths,
with 871 Python files and all 27 existing precision documents bound. MAIN focused checks pass **2,191 cases and 6,573
setup/call/teardown phases per stack**, with settings, callables and source bytes
restored. The inventory records **220 manual files, 1,363 functions and 611
migration additions**; all 1,357 immediate predecessor functions and 752 original
anchors remain. Raw 1,802 rows/3,604 cells and nonmanual fields are unchanged.
Both inventory CLIs and the configured style/type checks pass.

The first complete run retains **8,765 passes and one DINO B14 failure**. A
controlled instance-address reuse defect is repaired, but address reuse in that
original event was not observed. The older-stack FP16 decoder benchmark also
remains failed; pristine-baseline comparison proves that specific failure
predates this packet. Details and separate owner counts are in the
[retained failure record](named_runtime_parameters_findings.md#instance-lifetime-and-retained-failures).
The fresh complete primary suite passes **8,772 tests** with no failures,
errors or skips (13,773 warnings; 570.05 pytest seconds). Every previous 8,766
identity and six lifetime regressions are present. All seven actual code hooks
pass without bypass; frozen source/checkpoint/document bytes stay unchanged.

The [exact floating-selection update](source_integration_findings.md#exact-floating-selection) repairs the already-admitted
SOURCE Boolean-selector, two-arm FP16/FP32/FP64 case by copying the selected
bits through INT32 GatherElements. Native calls, admission and other dtype/scope
boundaries stay unchanged. The reviewed owner packet passes 179 cases per stack,
including strict zero-sign and NaN-payload checks. It adds 38 cases while
retaining the original test anchors and strict criteria. The two-arm stack and
index tensor have substantial memory cost. Historical direct-Where failures remain recorded.

Independent review accepts the integrated selection repair and the strict
current Gather graph oracle. The original Gather graph and failed first full
run remain recorded; this tooling update adds no Gather numerical acceptance.
Actual MAIN focused checks pass 1,911 cases per stack. Style, configured typing
and both inventory checks pass; the inventory has 220 files, 1,372 functions
and 620 migration additions, retaining all earlier anchors and raw rows.
The complete primary suite passes all 8,819 expected cases without errors,
failures or skips. Code commit `778dce3c` passes all seven actual hooks without
bypass, with the six changed paths and all frozen source/document hashes
preserved. The separate unchanged NNX/Linen MLP implementation remains outside
MAIN, with model runtime and numerical acceptance pending. Complete model
coverage and the public transition remain open.

The preceding [Typed checkpoint](typed_values_and_masks_findings.md) retains
its independent acceptance, 8,659-test full result and code commit `0d8f9d19`.
Its RAW65, cast/subnormal, Where and matrix-context results remain unchanged.
Public LEGACY behavior and both double-precision options are unchanged. The
full B–F migration still requires complete model/control coverage, compatible
fixed-source preparation and one ordinary source-preserving public call.

The preceding [Gather diagnostic counterpart](gather_counterpart_findings.md) checks one unchanged
`plugin::gather`, ordinal 0, source and its actual FP64/INT64 inputs, reference,
native traces and saved ONNX representation. Its exact graph of twenty nodes
and thirteen constants retains native broadcasts, index adaptation and the existing
FP64 cosine expansion `1 - 2*Sin(shared*0.5)^2`. It does not contain a direct Cos
node. The original raw callback still reports **FAILED**; the separate SOURCE
structure check is **VALIDATED**. Diagnostic schema/reload checks pass under
`source-diagnostic-v1`, while product admission, lowering, runtime and numerical
acceptance remain **UNVERIFIED**. The x64-off configuration conflict is unchanged.

The six-file tooling/test implementation has independent semantic acceptance.
Owner focused checks pass 249 cases on each stack, including ninety new Gather
cases. Both existing 26-cell replays and ordinary two-cell probes preserve
raw outcomes and historical provenance with zero ORT attempts. Criteria and
optimization remain None; no numerical approval or historical repin occurs.
Independent review also accepts the saved owner artifact observations on both
stacks, without numerical acceptance. Actual MAIN integration is independently
accepted: both stacks pass the same 249 focused cases and reproduce the bounded Gather diagnostic in the 26-cell
replay and separate ordinary probes, with all 864 Python hashes unchanged.
Eight other matrix cells per stack have unresolved fresh reference differences;
Gather’s own reference and artifact remain byte-identical to the owner evidence.
That Gather checkpoint’s inventory records **214 files, 1,264 functions and 512 migration
additions**, preserving all 752 original functions, zero removals and 1,802 raw
rows/3,604 cells. Both inventory CLI checks pass. The fresh complete primary-stack suite passes **7,326 tests** (13,130 warnings,
442.22 pytest seconds), with no failures, errors or skips. Code commit `e60ba9ab`
passes all seven actual repository hooks; all seven checkpoint paths and 864 Python hashes remain unchanged after commit.

Public LEGACY precision policy, both double-precision options and report schema
1 retain their meanings. The existing audited SOURCE profile is unchanged.
That checkpoint’s strict Where zero-sign failures remain recorded, and BF16
structural evidence still does not establish runtime support. Coherent typed operator families,
broader model coverage and the full B–F transition remain unfinished.

The preceding [native Where and typed JIT checkpoint](where_jit_admission_findings.md)
has independent semantic and actual-main acceptance. The private SOURCE path
preserves original `jnp.where` calls and admits bounded floating/BOOL selection,
static BOOL inputs and recursively validated pure closed JIT bodies. Actual
constants, formal bindings, weak types and native symbol scopes are checked
before lowering; version-sensitive inline hints use the compatibility layer.
Public LEGACY precision policy, `source-static-v1` and report schema 1 remain
unchanged.

Before the shared x64 restoration repair, both actual-main selections pass
**2,661 cases**, including the complete prior
2,658 and 1,489 selections. Each records 7,983 passing test phases and all 49
observed test modules: 44 from MAIN and five frozen external files. All 862
Python hashes, settings, original callables and policy are preserved. The
independent selection passes 200 cases per stack; the packet adds 216 permanent
cases. These overlapping counts are not additive. The separately reviewed
shared x64 repair preserves the hidden global beneath caller-local overrides
and adds 87 permanent cases. Both post-repair main selections
pass 89 cases with hidden x64 True; five actual MAIN imports, all 863 Python
hashes and restored state are verified. Independent integration review accepts
these results.

That checkpoint’s inventory records **213 files, 1,254 functions and 502 additions**,
retaining all 752 original anchors, zero removals and 1,802 raw rows/3,604 cells.
Both inventory CLI checks pass. The fresh complete primary-stack suite passes
**7,236 tests** (13,130 warnings, 400.95 pytest seconds), with no failures, errors
or skips. All fourteen checkpoint paths and all 863 Python hashes stay exact.
Code commit `ae820b1a` passes all seven actual repository hooks; the tested
checkpoint hashes remain unchanged after commit.

At that checkpoint, strict Where negative-zero samples were **FAILED** for
FP16/FP32/FP64 in both CPU settings on both measured stacks, with a raising
acceptance guard. Those original records remain unchanged; the later copy-based
SOURCE repair is described above.
Ordinary finite/BOOL success and BF16 structural evidence are separate. The
original gather source now has matching complete native/activated typed traces,
but its unchanged raw callback still rejects the false constant's broadcast
shape. That supplement added no accepted artifact or ORT result and kept its None
criterion.
At that checkpoint, the separate Gather counterpart was still being authored.
Its later diagnostic result is recorded above; coherent typed operator families,
broader symbolic/model coverage and the full B–F migration remain unfinished.

The preceding [native Exp and Leading-B Mul checkpoint](pointwise_mul_exp_findings.md)
has independent semantic and actual-main acceptance. Native `jnp.exp` now keeps
the original SOURCE call and weak typing; the private audited profile validates
floating Exp and finite static/shared-batch Mul composition. Source casts and
explicit broadcasts remain visible. Main checks pass **1,489 cases per stack**,
including all previous 990 cases, fourteen frozen review guards and 117 new
permanent cases. The separate independent selection passes 153 cases per stack;
these overlapping counts are not additive.

That checkpoint’s inventory records **210 files, 1,217 functions and 465 additions**, with
all 752 original anchors, zero original removals and 1,802 raw contracts/3,604
cells retained. Both inventory CLI checks pass. The complete primary-stack suite passes **6,933 tests** (13,130
warnings, 403.51 seconds). Commit `7c68300b` passes all seven actual hooks;
all five code/inventory paths and all 860 Python hashes remain unchanged.
The fixed Exp/Mul samples establish bounded CPU evidence; BF16 structural
checks do not establish a working CPU kernel. Neither the five-row Leading-B
nor the twelve-row numerical packet is rerun or repinned. Public LEGACY,
`source-static-v1` and report schema 1 remain unchanged. Where/JIT was still
pending at that checkpoint; its bounded follow-up is recorded above. Broader
phases D/E/F remain unfinished.

The preceding [Leading-B matrix/bias checkpoint](leading_batch_matrix_bias_findings.md)
extends the private audited path to one actual leading symbolic batch at ranks
two/three: canonical FP32/FP64 MatMul, equal-tail Add with a shared B or a static
leading one, and the native static bias reshape. The five original MatMul,
GPT Add and NNX Linear anchors keep their callables, weights, bias and runtime
operand ports. Production semantics, repaired tooling and actual-main binding
have independent acceptance; main checks pass **990 cases per stack**. The
preceding owner run passes 1,004 cases per stack, including those 990 permanent
cases and 14 frozen review guards; these overlapping counts are not additive.

The separately released v3 observation records ten symbolic artifacts, twenty
sessions and forty actual runs per stack; all three approved FP32 predicates
pass for every output; independent result review accepts these samples. Earlier
preparation failures and a later wrapper reporting error remain recorded.
That checkpoint’s inventory records **209 files, 1,197 functions and 445 additions**,
retaining all 752 original anchors and 1,802 raw contracts. Both inventory CLI
checks pass. The complete primary-stack suite passes **6,816 tests** (13,130
warnings, 431.98 seconds). Commit `718f448e` passes all seven actual hooks;
the twelve frozen code/inventory files and all 859 Python hashes are unchanged.
Public LEGACY, `source-static-v1` and report schema 1 stay unchanged. Broader acceptance across
the 310 symbolic rows and final phase F remain unfinished.

## Run it

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
post-processing and serialization. It checks static floating inputs and a small
primitive set at opsets 23/26, including selected Function/If cases. The finite
shared-B extension adds equal rank-two floating inputs through Add and named
composition. The Leading-B checkpoint adds one atomic native B at axis zero of
rank-two/three matrix and bias contracts, with FP32/FP64 MatMul and positive
static tails. The Pointwise extension adds native floating Exp and equal-tail
shared-B/static-1 Mul at those ranks, with exact source weakness.
The later Where/JIT checkpoint adds static BOOL dependencies, bounded floating
selection and pure closed JIT bodies. Broader symbolic operations, loops, scan,
unsupported JIT forms, effects and unaudited operations remain excluded from
audited admission.
Bodies are validated recursively. Same-width BitCast requires explicit opset26;
different-width bitcasts need a different representation and are rejected.
No opset is raised automatically. These are experiment limits, not universal
ONNX support classifications. Separate diagnostic regressions now cover finite
noncomplex/complex loops and broader JIT forms without broadening this product
profile;
the [work plan](../precision_implementation_work_plan.md) records their exact gates.

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
