# Computation Precision Current State

Date: 2026-10-17. Source snapshot:
[`fae17822e0114e6c00840b8f78adedb7ec36f74e`](https://github.com/enpasos/jax2onnx/commit/fae17822e0114e6c00840b8f78adedb7ec36f74e),
which includes the merged consumer-adaptation fix from PR #275 and the native
typed lowering changes from PR #274.

The public-behavior and implementation analysis below retains that baseline.
The later local branch integration is identified separately: private converter
code now carries `PrecisionPolicy.SOURCE`, while the public API and legacy
defaults remain unchanged. See the
[integration findings](poc/source_integration_findings.md) for its bounded scope.

The committed [native Sum checkpoint](poc/source_integration_findings.md#native-sum-forwarding-and-measured-composition)
`2d1ea944` forwards the captured original before legacy dtype, axis, mask,
initial-value or integer-promotion handling. Its full-plugin-world tests preserve
native JAX behavior on both stacks, and independent review accepts the completed
MAIN integration. Both protected preparation examples reach the float32-to-uint32
BitCast representation check at unchanged opset 23. Their exports remain rejected,
and the two HIGHEST-dot profile exclusions remain. The existing profile, public
defaults and precision flags are unchanged. The linked findings record completed
validation separately from unresolved whole-model and target numerical support.

The committed [LayerNorm and mean checkpoint](poc/source_integration_findings.md#layernorm-and-mean-native-forwarding)
preserves the original calls for NNX, Linen and Equinox under the private SOURCE
policy. SOURCE now reaches native normalization and reduction before the legacy
wrappers can cast inputs, select axes or replace the computation. A retained NNX
witness with FP32 inputs and FP16 affine parameters demonstrates the benefit:
the original computation promotes those parameters to FP32; the old wrapper
instead narrowed the input to FP16. The new activated trace matches the original.

This is independently accepted forwarding, with no new profile admission or
public API behavior. Actual default SOURCE LayerNorm exports still stop at `square`, so
LayerNorm-containing model export, runtime and numerical acceptance remain open.
The separate MLP implementation remains outside MAIN. The linked findings retain
the full evidence and current integration-validation status.

The [named runtime-parameter checkpoint](poc/named_runtime_parameters_findings.md)
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

The separate [fixed-source preparation PoC](poc/fixed_source_preparation_findings.md)
packages a finite, explicitly authorized pretrace cast change and actual NNX
parameter configuration. At that checkpoint, its 11 maintained tests passed on
each stack. Each stack recorded twelve ordinary audited export attempts and four
actual CPU runs of the two exact witnesses, with eight sum/mean/HIGHEST profile
barriers. This shows
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
[retained failure record](poc/named_runtime_parameters_findings.md#instance-lifetime-and-retained-failures).
The fresh complete primary suite passes **8,772 tests** with no failures,
errors or skips (13,773 warnings; 570.05 pytest seconds). Every previous 8,766
identity and six lifetime regressions are present. All seven actual code hooks
pass without bypass; frozen source/checkpoint/document bytes stay unchanged.

The [exact floating-selection update](poc/source_integration_findings.md#exact-floating-selection) repairs the already-admitted
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

The preceding [Typed checkpoint](poc/typed_values_and_masks_findings.md) retains
its independent acceptance, 8,659-test full result and code commit `0d8f9d19`.
Its RAW65, cast/subnormal, Where and matrix-context results remain unchanged.
Public LEGACY behavior and both double-precision options are unchanged. The
full B–F migration still requires complete model/control coverage, compatible
fixed-source preparation and one ordinary source-preserving public call.

A separate native-reference investigation reproduces the earlier four-row
OWNER/MAIN matrix hash split by changing one recorded XLA CPU flag, with equal
inputs and typed traces. The historical owner's flag was not recorded, so its
cause remains unproved. This is context evidence, not a new tolerance or ONNX
numerical acceptance. Independent review accepts the finite controlled observation;
it does not establish the historical owner's environment. The [Typed findings](poc/typed_values_and_masks_findings.md#native-reference-context)
record the bounded follow-up while the Gather history stays unchanged.

The preceding [Gather diagnostic counterpart](poc/gather_counterpart_findings.md) checks one unchanged
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

The preceding [native Where and typed JIT checkpoint](poc/where_jit_admission_findings.md)
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

The preceding [native Exp and Leading-B Mul checkpoint](poc/pointwise_mul_exp_findings.md)
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

The preceding [Leading-B matrix/bias checkpoint](poc/leading_batch_matrix_bias_findings.md)
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

The local follow-up reviewed on 2026-10-18 also carries SOURCE policy into the
shared optimizer and includes promotion metadata from captured state in
function keys. The [follow-up findings](poc/source_integration_findings.md#follow-up-review-on-2026-10-18)
record the repaired cases and further runtime failures. They do not change the
public baseline analyzed below.

The branch has committed shared-facade report/naming/serialization and CPU
sample integration (`3b3a48e0`), the evidence module (`ccc861b0`), typed
initializers (`b2974c3d`), independent reference preparation (`206277ac`) and
complex-constant storage (`3e904b46`). Discrete comparisons, logical/physical
boundaries and finite noncomplex loops are committed at `52389e29`, `530561d1`
and `1461e4b9`. Their `158ce625` scenario-plan checkpoint passed 5,363 integrated
primary-stack tests; that is historical regression evidence.

Further local promotion-context replay, finite complex-loop representations
and required SOURCE finalization have passed independent focused review on both
stacks. Their selections passed 359, 404 and 523 tests per stack, respectively;
complex loops also passed 75 generated tests per stack. Independent review ran
all 52, 121 and 54 new cases, respectively. These overlapping selections do not
establish family-wide support. Known CPU Where negative-zero failures remain
failed under their bitwise sample criteria on ORT 1.31/1.29, with optimization
off and on. Diagnostic lowering remains unverified.

The [first corpus batch](poc/source_corpus_findings.md) measured 36 original
x64-off/on cells per stack, including 26 checked artifacts, four export gaps,
five pending cases and one configuration conflict. No ORT sessions ran and no
numerical acceptance was claimed. The other 3,568 cells remain unchanged.
The [fixed-source review](poc/fixed_source_compatibility_findings.md) classified
all 22 parameterized models using 44 serialized LEGACY checks in total and 20
manual scope checks in total. It identifies capabilities the new runtime-input
contract must retain without editing supplied source implementations.

The preceding integrated checkpoint passed **5,650 tests** (13,104 warnings,
356.07 seconds). An earlier run failed one preflight while 5,647 tests passed:
test generation had mutated raw registration metadata. The accepted repair
copies each testcase before enrichment, preserving generated behavior and raw
contracts. Its focused selection passes 127 tests per stack; both new
regressions pass independent reruns, and generated-collection controls pass.

C3b, C3c and required finalization are committed at `ea6a8da7`, `8ba3c681` and
`f1ee71b3`; B3 tooling, metadata ownership and the reviewed inventory refresh
are committed at `3dbec1b8`. All four commits passed their actual hooks. Package
typing (541 files), RNG registration, Ruff, path markers and strict MkDocs pass.
The `3dbec1b8` inventory recorded 191 handwritten-test files, 944 functions and
192 migration additions, retaining all 752 original anchors and 1,802 raw
contracts. The first B3 batch remains bound to its earlier baseline and has not
been rerun.

Private recording/web packaging is now committed at `8a82d510`; SOURCE
`where`/`full` promotion and weak-type repairs are committed at `223f24bf`.
Both passed independent review and actual repository hooks. Recording reuses
the shared logger; IR/proto/file checks preserve exact 1.2 MB FP64 payloads
through root and nested Function constants, including signed zero and NaN bits.
The array repairs preserve native JAX promotion and canonical constants;
Boolean selection uses logical operations with exact CPU samples. At that checkpoint, audited
product admission excluded `where`/`full`; the newer bounded Where/JIT admission
is described above, while `full` remains outside the audited profile. BF16 and neutral Boolean
Where kernel failures remain separate target evidence. See the
[curated findings](poc/option_and_array_creation_findings.md).

The C4a/D1a checkpoint passed **5,803 tests** (13,104 warnings, 382.81 seconds)
and recorded 192 handwritten-test files, 963 functions and 211 migration additions.

C4c (`4c1c57fe`) now traces positional and real scalar-BOOL named operands in one
prepared root, retaining actual unused formals without artificial sinks. One
checked binding map drives names, reports and IR/proto/reloaded boundaries.
SOURCE framework Dropout consumes actual operands; defaults and static literals
stay local. Whole-root external `lax.cond` preparation covers a bounded unchanged
Python-branch source without freezing its runtime control. The all-22 diagnostic
pass yields three checked artifacts and 19 pending rows per stack: zero ORT
executions and unverified lowering/runtime/numerics. It does not establish
stochastic or full-model acceptance.

D1b (`05037961`) dispatches SOURCE `arange`/`linspace` to captured native JAX before
argument normalization. It preserves native promotion, omitted-step arithmetic,
integer flooring and explicit conversion boundaries. Four exact FP16/FP32
`linspace` cells per stack still fail on ORT and ONNX ReferenceEvaluator; BF16
runtime limits and native-untraceable, LEGACY-positive dynamic `arange` remain
open. Public LEGACY and audited admission are unchanged. The C4c/D1b checkpoint
passed **6,035 tests** (13,110 warnings, 402.20 seconds), with an inventory of
195 handwritten-test files, 1,003 functions and 251 migration additions.

The subsequent shared optimizer repair (`c528a894`) removes only the proven-false
optional training input of a matched standard Dropout. It preserves shared and
observable `Not(True)` outputs and avoids the formerly unowned `false_const`.
D2a (`2011b140`) adds diagnostic same-dtype FP32/FP64 matrix lowering for positive
static ranks 2–4, one contracting/free axis per operand and default precision.
Native SOURCE `matmul` binding retains JAX promotion and precision parameters.
The [matrix and optimizer findings](poc/matrix_and_optimizer_findings.md) record
the exclusions, preserved operands and failed exact subnormal samples.
A separate two-cell corpus supplement now passes schema checking but fails the
unchanged Gemm/zero-bias structural callback; that supplement accepts no artifact
or row.
Static NCHW boundaries are committed at `7609fc2e`, with their corrected shared
observer/naming dependency at `f02c0b4c` and native binary/Mul-None at `258e90d6`.
The [new findings](poc/layout_and_native_binary_findings.md) distinguish that
scope from the independently accepted native add/min/max binder and separate
Mul-None diagnostic repair. The binder passes 366 independent tests per stack;
its eight historical signed-zero failures per stack remain recorded. Broader options and
models remain open.

The historical Dropout/D2a complete primary-stack run passed **6,183 tests** (13,110 warnings,
406.62 seconds), with no exclusions. Required typing covers 541 files; RNG
registration and global Ruff pass. Both Dropout/D2a commits passed actual hooks. The
checkpoint inventory records 197 handwritten-test files, 1,023 functions and 271
migration additions. The next full run on optimizer `8fd3cff6` failed ten tests,
with 6,335 passed and 13,110 warnings in 632.85 seconds. Detached removed-Transpose
use links caused all ten failures, reproduced unchanged on both stacks. Final
`0f8b60c5` bytes use current local consumers while retaining output/nested guards;
owner and independent checks pass the same 238 cases per stack. These prefixes
are file hashes, not commits. Root typing (541 files plus RNG) and global Ruff pass.
That checkpoint's inventory records 200 files, 1,074 functions and 322 additions, retaining
all 752 original anchors, all 1,802 raw contracts and optional entries. Fresh
CLI comparisons pass on both stacks. The fresh complete primary-stack run passes
**6,351 tests** (13,110 warnings, 466.07 seconds). All three new code commits passed
actual configured hooks; all 17 frozen code/inventory hashes remain unchanged.
The `d1dd3162` documentation checkpoint records those results. Shared-B
implementation `80cb68b2` preserves two equal floating `(B,K)` inputs through audited Add
and named composition. One artifact passes native eager/JIT samples at B=2/5;
the evidence layer rejects inconsistent bindings before session creation/run,
even where raw ORT would broadcast. Strict IR/proto/file/report signatures,
native symbol scopes, source weakness and contextual cache behavior are covered.
BF16 checker/report success retains a FAILED CPU Add load result. Other symbolic
operations/layouts and all 310 raw symbolic rows remained open at that checkpoint.

The [new findings](poc/symbolic_batch_and_minmax_findings.md) record 697 owner and
241 independent checks per stack, including all 40 new cases. Root typing
(541 files plus RNG), global Ruff and both inventory CLI comparisons pass. The
new inventory records 201 files, 1,093 functions and 341 additions, with all 752
original anchors and 1,802 raw contracts unchanged. The complete primary-stack
run passes **6,391 tests** (13,110 warnings, 444.36 seconds); `80cb68b2` passed all
seven actual configured hooks. All twelve tested code/inventory hashes remain
unchanged after commit. Public LEGACY, the private profile identifier and report
schema remain unchanged. These are the preceding checkpoint results.

The [preceding checkpoint findings](poc/minmax_and_corpus_replay_findings.md) record an
independently accepted SOURCE Min/Max lowerer for equal FP16/FP32/FP64 at opset 23.
Owner and independent selections each pass the same 132 tests per stack,
including 29 new cases. New zero-sign artifacts pass bitwise native eager/JIT
samples; the old eight failed observations per stack remain history. Four
subnormal witness artifacts per stack and each FP32/FP64 grid's 108 inherited
non-NaN mismatches remain FAILED. Actual graphs have 15 total nodes for Min and
12 for Max, with one Min or Max operator and no new floating Cast; the earlier
one-workload cost evidence remains scoped.

The repaired B3 revision has independent acceptance for 13 unchanged rows/26
artifact-only cells. At that checkpoint, side and main replays each check 24 artifacts per
stack, retaining gather-off's configuration conflict and gather-on's custom-sine
diagnostic gap. All 26 lowering/runtime/numerical gates remain unverified, with
zero ORT. A separate negative reproduction overwrote its own preparation
snapshot; earlier valid distinct-path records remain intact. The repaired guard
and fresh preparation hashes are verified. This packet changes no default
admission or numerical criterion.

Commit `fc3f9323` combines both reviewed packets and the manual inventory. Main
focused checks pass 323 per stack; the full primary-stack run passes **6,466 tests**
(13,110 warnings, 413.65 seconds). All seven actual configured hooks pass, and all
nine frozen code/test/inventory hashes match after commit. Typing (542 files plus
RNG), global Ruff and both inventory CLI comparisons pass. The inventory records
203 files, 1,114 functions and 362 additions, retaining all 752 original anchors
and 1,802 raw contracts. Those historical results are unchanged.

The [ordinary-matrix admission findings](poc/ordinary_matrix_admission_findings.md)
record the preceding static audit: same-type FP32/FP64 ordinary dots with positive
static ranks 2–4, valid ordered paired axes and default precision now use the
actual audited SOURCE profile. Main checks pass **750 tests per stack**.
The separate fresh numerical packet creates 24 audited artifacts and records
48 CPU runs per stack; all three unchanged FP32 sample predicates pass with
zero pending cells. Historical diagnostic reports and failed raw Gemm callbacks
remain unchanged. These results do not establish general numerical acceptance;
strict subnormal failures and their raising gates remain.

That checkpoint’s inventory records **207 files, 1,175 functions and 423 additions**,
with all 752 original anchors, zero removals and 1,802 raw contracts/3,604 cells
retained.
Typing (542 files plus RNG), global Ruff, pinned-source Black and both inventory
CLI comparisons and independent numerical post-run review pass. The integrated
primary-stack run passes **6,728 tests** (13,120 warnings, 370.40 seconds).
Commit `45881d8d` passes all seven actual hooks on the first attempt, with the
24 frozen source hashes unchanged. The preceding
[Linear/reshape checkpoint](poc/native_linear_reshape_findings.md), `791664f8`,
retains its 6,620-test full pass and unchanged native adapter prerequisite.
Public LEGACY, the private profile identifier and report schema are unchanged.

The preceding [trig/numerical checkpoint](poc/native_trig_and_corpus_numerics_findings.md),
`d2292b89`, retains its 6,579-test full pass, actual hooks and scoped numerical
observations: 96 recorded outputs and 288 approved comparisons across both
stacks. Its failures, criteria and unverified general lowering remain unchanged.

The separately approved criterion requires exact output count/FP32 dtype/shape and all three
comparisons: `allclose(Y, Rraw)`, `allclose(Rraw, Y)` and
`allclose(Rharness, Y)`, with `rtol=atol=1e-5` and `equal_nan=True`.
`Rharness` evaluates the unchanged native callable on the same feeds under
x64 disabled and matmul precision `float32`; it is a separately labelled
context reference, not the legacy export or its original random-input run.
Every predicate consumes the same captured Y. Historical None criteria, raw
Gemm callback failures and stricter failed samples remain unchanged.

The historical trig gather supplement retains x64-off's configuration conflict; x64-on
passes the former sine barrier and stops at `scope 0/equation 11
(jax.numpy.where)`. Its first eleven equations and typed boundary/output
signatures match the native trace, including native Sin/Cos; the complete
traces differ at Where. No artifact or ORT result is added. A primary observer
list/tuple assertion failed after persisting evidence; that exit-1 run remains,
with a separate persisted-JSON correction. The secondary v2 driver passes.
The later Where/JIT supplement described above preserves the complete trace
but still fails the unchanged raw broadcast-constant callback, with zero ORT.
The later [Gather counterpart](poc/gather_counterpart_findings.md) separately
checks the exact diagnostic representation and preserves that raw failure.
The [work plan](precision_implementation_work_plan.md) owns the remaining gates.

For observable behavior and export choices, read the
[user view](precision_current_state_user_view.md). All companion documents are
listed in the [precision overview](README.md).

Computation precision is governed by overlapping tracing, lowering, and
postprocessing rules. The double-precision setting promotes many values, while
explicit casts and selected scopes preserve lower precision. The resulting
behavior is not one uniform dtype contract. The proposed
[target state](precision_target_state.md) and
[migration plan](precision_migration_plan.md) describe how to resolve this.
The merged fix repairs selected consumer type mismatches within the existing
policy; it does not implement that broader target.
The local experiments and committed private integration include core changes. Their successful
samples, known gaps, and separate feasibility controls are summarized below;
they do not establish a released source mode.

## One Argument Controls Several Decisions

The public `enable_double_precision` argument defaults to `False`. Its current
meaning combines tracing configuration, defaults for unspecified dtypes, IR
type selection, constant conversion, and final export postprocessing.

The public facade uses `_temporary_x64`, an alias of
`conversion_api._force_jax_x64`. This context sets `jax_enable_x64` to the
argument's Boolean value and restores the previous setting afterward. Thus
`False` actively disables x64 during tracing, including when the caller had
enabled it. `True` permits JAX to trace FP64 computations; enabling x64 alone
does not make an explicitly FP32 JAX calculation FP64.

`_as_sds_list` assigns FP32 to shape-only input specifications under `False` and
FP64 under `True`. Explicit `ShapeDtypeStruct` dtypes initially survive this
normalization. Later tracing and lowering still apply their own rules, so this
initial preservation does not establish an end-to-end dtype guarantee. See
[conversion_api.py](../../jax2onnx/converter/conversion_api.py) and
[user_interface.py](../../jax2onnx/user_interface.py).

The baseline normalization reconstructs `ShapeDtypeStruct` from shape and dtype
and drops `weak_type`. That JAX property affects type promotion, so retaining
the dtype alone can change the computation JAX traces. Public legacy calls
retain those observed normalization rules. The initial source experiment
inherited the same loss in both the UX wrapper and shared tracing helper.
The later private source policy now preserves the property through both steps,
including direct helper calls, and reports supplied/normalized/traced flags.
The [regression](poc/migration_ux_findings.md#weak-input-properties-preserved-on-the-source-path)
returns FP16 `2048` for a weak scalar plus explicit FP16 `1`, while a strong
input with the same shape/dtype returns FP32 or FP64 `2049`. Both supported
stacks cover this source correction; it does not change legacy normalization.

The [calling experiment](poc/migration_ux_findings.md) compares real public calls
with the argument omitted, explicitly `False`, and explicitly `True`, for
shape-only inputs, typed specifications, NumPy arrays, and JAX arrays under
both ambient x64 settings. Omission and `False` currently select the same
behavior. An FP64 `ShapeDtypeStruct` on those routes is truncated during JAX
tracing: JAX 0.10.2 is silent for the tested case; JAX 0.11.2 warns. Both warn
for a model-internal `.astype(jnp.float64)` under disabled x64. These are
observed version differences, not diagnostics guaranteed by the exporter.

The shared `numpy_dtype_to_ir_with_float_policy` maps FP32 to ONNX `DOUBLE`
when the argument is `True`. It preserves FP16, BF16, and FP64 in its own
mapping. An unspecified dtype also follows the argument. Therefore this helper
does more than translate a dtype already chosen by JAX. Conversely, its FP64
case does not downcast under `False`; tracing and context allocation have
additional behavior. See [ir_utils.py](../../jax2onnx/ir_utils.py).

Historically, this combination was intentional. The
[v0.6 implementation](https://github.com/enpasos/jax2onnx/commit/b646501ae07737e011a4f12e460c9c14f8da4773)
described a double export mode and a single precision downcast mode. Subsequent
fixes introduced important exceptions. Recasting the existing argument as only
an x64 permission would consequently change observable behavior.

## Lowering Has Several Type Authorities

The current native context separates `require_value_for_var` for inputs,
`lookup_value_for_var` for inspection, and `allocate_value_for_var` for writable
outputs. The compatibility helper `get_value_for_var` still reuses an existing
binding or allocates one. Allocating an output no longer means blindly reusing
an input or an already produced result. These ownership improvements do not
remove the precision policy: allocation still consults the Boolean and the
traced `aval.dtype`. See [ir_context.py](../../jax2onnx/converter/ir_context.py) and
[typing_support.py](../../jax2onnx/converter/typing_support.py).

Plugin output types can then follow a traced dtype, an allocated placeholder,
an operand, or a plugin-specific rule. For example,
[`convert_element_type`](../../jax2onnx/plugins/jax/lax/convert_element_type.py)
uses the exact traced target dtype, whereas general context allocation can
promote an FP32 placeholder. Final output binding also keeps existing floating
types in cases where they differ from its policy-derived target. Reading only
the JAXPR or only the final public output type therefore cannot establish the
precision of the whole exported calculation.

The implemented
[`promote_float32_operand`](../../jax2onnx/plugins/_precision_utils.py) helper
now adapts a FLOAT operand when its caller explicitly requires DOUBLE. The
adapter inserts a separate Cast and keeps the original binding available to
other consumers. Native Neg, selected unary and real binary operations,
reduction helpers, and concatenate paths use this adaptation. The explicit
FP64-to-FP32 rounding remains before the consumer's FP32-to-FP64 conversion.
This is a bounded compatibility repair, rather than a global precision rewrite
or a new interpretation of every `astype` expression.

Explicit dtype preservation is a tested boundary. The
[#217 cast fix](https://github.com/enpasos/jax2onnx/commit/e76a90eac15d4cef8dc5f8e5d06d7f70e2720912)
deliberately retained explicit cast targets. Current
[exact-dtype tests](../../tests/extra_tests/converter/test_exact_dtype_casts.py)
require FP32 graph outputs under `enable_double_precision=True` for casts and
selected constructors: zeros, ones, eye, full, linspace, and arange. They do
not establish preservation of every intermediate in arbitrary compositions.
The older bitcast target assertion is structural, and plugin metadata still
skips numerical validation. A new regression additionally checks a complete
exported FP32-to-UINT32 bitcast at opset 26 through the checker and CPU runtime,
while the same FP32 value also feeds a DOUBLE consumer. This is specific runtime
evidence, rather than general bitcast coverage.

FP16 and BF16 need separate consideration. Their shared mapping preserves
their types, and the context's disabled-double output allocation exempts both
from its general floating downcast. The FP16 placeholder regression covers
both Boolean settings. These facts do not prove complete FP16/BF16 constant,
operator, or runtime coverage. In particular, context constant promotion still
widens NumPy floating arrays, including FP16, under double export; the mapper's
exceptions are not universal rules for every path.

## Constants and Nested Graphs Add Further Rules

Closed-over constants are reconciled with their JAX variables and can then be
promoted before binding. `IRContext.bind_const_for_var` applies another
promotion path. At export time, `postprocess_ir_model` promotes FP32
initializer payloads, tensor-valued `Constant` attributes, and constant-backed
node outputs when requested. It recursively visits control-flow subgraphs and
ONNX Function bodies. This occurs after the converter's optimization pass.
See [ir_postprocess.py](../../jax2onnx/converter/ir_postprocess.py).

These paths can turn a precision decision into more than a metadata change:
they alter stored tensor payloads. The postprocessor does not implement a
complete typed graph transformation that reconciles every consumer and cast.
The existing
[constant-promotion tests](../../tests/extra_tests/converter/test_f64_constant_promotion.py)
require DOUBLE initializers and numerical agreement for selected captured
constant expressions. Separate
[postprocessing tests](../../tests/extra_tests/converter/test_ir_postprocess.py)
check direct payload promotion. Neither suite establishes arbitrary mixed
precision correctness.

Child contexts inherit the precision setting. Control-flow contexts also
inherit `_keep_function_float32`; selected scan paths enable that flag to
retain lower precision body values. Function boundaries have explicit formal
inputs, while eligible If branches can capture lexical values. These scope
rules and scan exceptions must be considered together with recursive late
promotion. The
[scan regression](../../tests/extra_tests/framework/test_one_minus_square_scan.py)
exercises a float32 body under double export, so preserving only public
outputs would miss an existing requirement.

## Numerical Accuracy Already Has Local Policies

Storage dtype, operation dtype, and accumulator precision are distinct. The
specialized explicit LEGACY LayerNorm lowering already widens selected internal
work independently of `enable_double_precision`: FP16/BF16 statistics use
FP32, and the FP32 slow-variance reduction accumulates squared centered values
in FP64 before rounding the variance to FP32. The remaining FP32 statistics
and affine computation retain FP32. This is a local numerical algorithm,
rather than a request to turn the entire model into FP64.

The [documented limits](../../docs/user_guide/known_limitations.md) and
[LayerNorm tests](../../tests/extra_tests/test_layer_norm_precision.py) distinguish
error against JAX from error against an independent FP64 reference. Their
fixed bounds cover specified FP32 cases, inputs, opset, runtime provider, and
optimization settings. They provide no general guarantee of bitwise equality,
universal improvement from widening, or equivalent bounds for every
normalization family.

The private SOURCE forwarding checkpoint preserves the actual framework's
normalization choices instead of selecting this legacy algorithm. Native
trace/eager/JIT equality establishes that interception behavior; it does not
transfer the legacy ONNX sample bounds to SOURCE or certify an exported
LayerNorm-containing model.

## What the Precision PoCs Establish

The [numerical findings](poc/numerical_contract_findings.md) compare JAX eager
and JIT execution, ONNX ReferenceEvaluator, and ONNX Runtime CPU with optional
optimization disabled and enabled. Both installed stacks reproduce the
observations: JAX 0.11.2 / ORT 1.31.0 and JAX 0.10.2 / ORT 1.29.0. The findings
record exact versions and inputs. The artifact origin matters:

| Case and origin | ONNX validity | Execution and numerical observation | What remains open |
| --- | --- | --- | --- |
| Public export: FP16 batched `dot_general`, preferred FP32 result | Full checker passes. | JAX returns FP32 `2049`; exported `MatMul` returns FP16 `2048` on ReferenceEvaluator and both ORT settings. | This public legacy result-type gap remains; the private source lowering below handles a bounded subset. |
| Private source export: default-precision FP16 rank-2/rank-3 matrix operands, explicitly preferred FP32 result | Full checker passes. | Operand casts precede an FP32 `MatMul`; the `2049` witness and additional finite-value cases retain FP32 results while shared input values remain FP16. | Other layouts, dtypes, explicit precision algorithms, targets, and performance require separate work. |
| Changed JAX source: explicitly FP32 matrix operands, public export | Full checker passes. | All tested paths return FP32 `2049`. | Sample agreement does not make this a validated automatic lowering or establish its hardware and performance properties. |
| Hand-built normalization control: FP32 centering/squares, FP64 variance sum and division, FP32 output | Full checker passes. | JAX, independent staged reference, ReferenceEvaluator, and both ORT settings agree on variance `8388609`; a NumPy FP32-only sum gives `8388608`. | This original control establishes feasibility; the local source integration below now exports the complete callable. |
| Private source export: the same selected-FP64 normalization, inline, in a Function, and in either conditional branch | Full checker passes after save/reload. | Six boundary/optimization cases retain exact centered values, squares, and variance, with a tolerance on normalized outputs. | This covers the stated calculation and inputs, rather than general normalization families or deployment targets. |
| Source-PoC export: FP16 square then sum | Full checker passes; ReferenceEvaluator agrees with JAX on the sample. | JAX returns zero squares and zero sum. Both ORT settings return zero squares but sum `2**-23`. | The tested runtime fails the sample's required intermediate rounding despite successful execution. |
| Hand-built unfused FP16 `Mul -> ReduceSum` control | Full checker passes. | The same ORT rounding discrepancy occurs. | This rules out blaming converter fusion alone; it does not identify the underlying cause. |
| Changed JAX source: explicit FP32 multiplication, FP16 rounding, FP32 sum, FP16 result; source-PoC export | Full checker passes. | Every tested path returns zero with the original FP16 output signatures. | The arithmetic change is only a bounded source candidate, not a general exporter repair. |

For normalization, exact centered values, squares, and variance are checked;
only the final normalized output uses a tolerance. A broad relative tolerance
on variance would accept losing the very terms the experiment is designed to
preserve. The constant input row also has an exact zero result.

The FP16 discrepancy remains when optional ORT graph optimization is disabled.
That setting does not establish the arithmetic performed inside a kernel.
Neither ReferenceEvaluator agreement nor the unfused control proves a general
rounding guarantee or identifies a converter fusion bug.

The [calling experiment](poc/migration_ux_findings.md) adds input normalization,
actual tracing observations, option conflicts, and warning reporting around
the existing entry points. Explicit FP64 inputs with x64 disabled are rejected before tracing
on the candidate source route; a model-internal FP64 conversion can already
have been truncated by JAX and is reported through its actual warning.

The later [private integration](poc/source_integration_findings.md) replaces
the local allocation subclass with `PrecisionPolicy.SOURCE` in the real
`IRContext`. It reaches audited `onnx_function` and two-branch `cond` children,
preserving selected mixed outputs, shared FP32 values, and FP32/FP64 constants
through serialization and CPU execution. Floating `gt` and Boolean-to-INT32
selector conversion cover the tested condition wiring. Recursive validation
checks branches and actual retraced function bodies. The later Where/JIT
checkpoint adds bounded pure closed JIT bodies; broader loops/JIT forms and
other unaudited operations remain excluded.

The source function variant carries flattened result abstract values and
reconstructs structured returns. It now captures `jax_enable_x64` and
`jax_default_matmul_precision` at each call site, replays them while tracing the
body, and includes them in function cache signatures alongside weak flags,
shape/dtype, and captures. Already-typed FP64 formals survive a local x64-off
scope; recreating the formals must not truncate a value that already existed.
The retraced result signature is checked against the original call.

Review exposed two distinct faults that the local implementation now repairs:
replaying a function under ambient rather than call-site x64 could return FP32
`0` instead of `1`, and JAX's cached conditional branch could retain a source
primitive during a later public legacy export. A private tracing discriminator
now separates ordinary JAX, legacy export, and source export caches. It uses
JAX's trace/JIT cache context through the compatibility layer rather than
clearing caches. The
[context regressions](../../tests/extra_tests/converter/test_source_function_context.py)
cover both local x64 directions, unchanged result dtypes, mixed call-site
settings in one graph, nested functions, scoped matrix requirements, failure
restoration, and export sequences around ordinary eager/JIT execution.

The [input](../../tests/extra_tests/converter/test_source_inputs.py) and
[scope](../../tests/extra_tests/converter/test_source_scopes.py) regressions now
live in the regular repository suite. They cover static arguments, structured
returns, repeated aliases, and input passthrough as well as mixed types.
The bounded source
[matrix lowering](../../tests/extra_tests/converter/test_source_dot_precision.py),
[normalization integration](../../tests/extra_tests/converter/test_source_normalization.py),
and [arithmetic-boundary checks](../../tests/extra_tests/converter/test_source_arithmetic_fusion.py)
also run there. The source path retains `Add -> Div` and `Mul -> ReduceSum`
instead of applying the legacy Mean and ReduceSumSquare substitutions without
an audited numerical equivalence argument.

The [sample acceptance helper](../../tests/extra_tests/converter/precision_runtime_support.py)
records the actual CPU runtime, optimization mode, architecture, thread counts,
and comparison result. It rejects failed or unverified sample acceptance and
keeps execution errors separate. It does not prevent export or implement a
public target-selection API. In particular, the known FP16 square/sum model
still exports and executes, but fails acceptance against its exact reference
on the recorded runtimes. That numerical defect remains unresolved.

Historical validation totals are recorded in the
[integration findings](poc/source_integration_findings.md#recorded-validation);
the [latest checkpoint](poc/option_and_array_creation_findings.md#validation-and-provenance)
records the complete 5,803-test run.
A green suite can include tests that recognize the known numerical failure;
it does not establish model-wide or deployment-wide precision compliance.

## ONNX Feasibility and Runtime Evidence

Precision support has separate schema, model-validity, runtime, and numerical
boundaries. The [ONNX Conv schema](https://onnx.ai/onnx/operators/onnx__Conv.html)
allows DOUBLE tensors, but this does not establish a corresponding kernel in
every execution provider. Focused standalone ONNX probes on 2026-10-17 used
Python 3.12.9, ONNX 1.23.2, ONNX Runtime 1.31.0 CPU, opset 23, IR version 10,
disabled runtime graph optimization, and one thread:

| Minimal operation | Full ONNX checker | ONNX Runtime CPU result |
| --- | --- | --- |
| FP64 `MatMul` | Passed | Loaded and executed |
| FP64 `Conv` | Passed | Load failed with `NOT_IMPLEMENTED` |
| BF16 `Add` and BF16 `Neg` | Passed | Both loads failed with `NOT_IMPLEMENTED` |

The BF16 probes used FLOAT inputs, explicit casts to BF16, the tested operation,
and a cast back to FLOAT. These are narrow kernel availability observations,
not complete jax2onnx exports or universal statements about these dtypes.

The [standard BitCast schema](https://onnx.ai/onnx/operators/onnx__BitCast.html)
starts at opset 26 and requires equal input/output bit widths. A separate
minimal FLOAT `[1]` to INT32 `[1]` reinterpretation failed checker and runtime
loading at opset 23, but passed both and returned `1065353216` for `1.0` at
opset 26. The current
[bitcast plugin](../../jax2onnx/plugins/jax/lax/bitcast_convert_type.py)
emits `BitCast` without an opset gate, while the public default remains opset
23 and its metadata skips numerical validation. The successful probe at 26
and the merged exported-model regression do not establish full JAX bitcast
coverage, especially for unequal bit widths.

Native [LayerNormalization](https://onnx.ai/onnx/operators/onnx__LayerNormalization.html)
also separates data and statistics: its data type parameter includes DOUBLE,
but its statistics type parameter allows FLOAT or BF16. Default `stash_type=1`
specifies FP32 normalization statistics. Thus DOUBLE input/output declarations
alone do not promise FP64 accumulation. This schema observation is not a
demonstrated CPU accuracy regression.

Runtime math settings add another boundary. ONNX Runtime's
[CUDA `use_tf32` option](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#use_tf32)
allows reduced-precision multiplication for certain FP32 matrix products and
convolutions while retaining FP32 tensor types. This is a documented provider
behavior, not a result of the CPU probes above; validation evidence needs the
relevant provider settings as well as the model's tensor dtypes.

Current CPU validation helpers and `export_mode="web"` have narrower roles:
the former execute selected comparisons on CPU, and the latter selects a
serialization profile. Neither is a general runtime capability or numerical
acceptance check. See [user_interface.py](../../jax2onnx/user_interface.py).

## Observable Behavior on Current Main

Focused checks at this snapshot used JAX 0.11.2 on CPU, ONNX 1.23.2,
ONNX Runtime 1.31.0, JAX x64 enabled, and `enable_double_precision=True`.
They show the distinction between source computation and later export promotion:

| Model expression | Tracing input dtype | Exported output dtype |
| --- | --- | --- |
| `x.astype(jnp.float32)` | FP32 | DOUBLE, with no Cast node |
| `x.astype(jnp.float32)` | FP64 | FLOAT, through a Cast node |
| `x.astype(jnp.result_type(x.dtype, jnp.float32))` | FP16 or BF16 | FLOAT, through a Cast node |
| `x.astype(jnp.result_type(x.dtype, jnp.float32))` | FP64 | DOUBLE, with no Cast node |
| `-x.astype(jnp.float32)` | FP64 | DOUBLE, after FP32 rounding and a consumer Cast |

The same-type FP32 cast disappears during JAX tracing. Later promotion therefore
cannot consistently distinguish an explicit source cast from an unchanged
FP32 value. The minimum-precision expression illustrates why its input dtype
must already reflect the intended precision when JAX traces the model. A
literal `astype(jnp.float32)` requests exactly FP32; it cannot also communicate
that FP32 is only a minimum for numerical stability. Preserving that cast
alone therefore does not resolve the minimum-precision intention. The proposed
[FP32 model-side example](precision_target_state_user_view.md#example-exactly-fp32-or-at-least-fp32)
uses computation configuration or a minimum-precision expression before
tracing, rather than having export silently reinterpret an exact conversion.
This is one example of operation-specific intent, rather than a general
requirement to widen lower precision values to FP32.

The original failing example from PR #275 now has consistent operand and
result types. For an FP64 input, `return -x.astype(jnp.float32)` follows:

```text
Cast: DOUBLE input -> FLOAT result
Cast: FLOAT result -> DOUBLE consumer operand
Neg:  DOUBLE input -> DOUBLE result
```

The merged runtime regression checks this result against negation of the input
rounded to FP32 and converted back to FP64. A direct output of the same rounded
value remains FP32, including when a DOUBLE consumer also uses it.

## Completed Legacy Repair and Remaining Boundaries

[PR #275](https://github.com/enpasos/jax2onnx/pull/275) is merged in this snapshot.
Its final implementation uses explicit consumer adapters, including the native
Neg path, rather than promoting values as a side effect of general lookup.
Graph output binding continues to require the existing computed value.

The focused
[exact-dtype suite](../../tests/extra_tests/converter/test_exact_dtype_casts.py)
passed all 26 cases locally at this snapshot. It covers Neg and Add after an
explicit FP32 cast; selected unary, real binary, reduction, and concatenate
consumers; shared direct outputs; function and layout boundaries; boolean and
integer results; and the opset-26 bitcast above. Runtime cases first run the
full ONNX checker, then execute the serialized model on CPU with optimization
disabled and one thread. FP16 Neg has runtime coverage here; BF16 Neg checks
types and model validity because the tested CPU kernel is unavailable.

The tests establish that these selected consumers can compute in DOUBLE after
the required FP32 rounding without changing the canonical FP32 value. They do
not establish source-type preservation for the full program. A fresh check of
`y = x.astype(jnp.float32); return (y + 1) - y` with FP64 input `1e8` produced
zero in JAX eager execution and one in ONNX Runtime CPU under this legacy
policy. This difference illustrates why restoring loadable, internally
consistent types is separate from preserving every source arithmetic step.

Finally, generated `_f64` tests rewrite floating inputs and selected factory
and expected dtypes to FP64. See [t_generator.py](../../tests/t_generator.py).
They supply useful FP64 coverage but cannot substitute for tests that retain
intentional mixtures of dtypes and observable rounding boundaries. The
current system consequently has substantial precision coverage alongside
context-dependent rules; it has no single faithful mixed precision contract
established across the whole pipeline.
