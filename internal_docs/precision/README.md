# Computation Precision

These documents describe current precision behavior, the proposed target, and
the full migration plan. Each topic has a user view focused on model preparation,
export options, and observable results, plus a technical view for implementation
work.

JAX model code and configuration define computation, accumulation, output types,
and deliberate conversions. The export call supplies the input shapes, dtypes,
and precision-relevant tracing properties and selects the ONNX representation.
Runtime configuration controls the execution environment. The target preserves
the resulting numerical contract within documented support limits. Examples
include FP16 or BF16 work, selective wider arithmetic for stability, and
scientific calculations requiring FP64.

| Topic | User view | Technical view |
| --- | --- | --- |
| Current behavior | [Precision today](precision_current_state_user_view.md) | [Current state](precision_current_state.md) |
| Proposed behavior | [Precision in the target design](precision_target_state_user_view.md) | [Target state](precision_target_state.md) |
| Proposed transition | [Moving to the target behavior](precision_migration_plan_user_view.md) | [Migration plan](precision_migration_plan.md) |

The authorized support floor is **Python >=3.11 / JAX and JAXLIB >=0.10.2 /
Flax >=0.12.8**. Dependency declarations and both checked lockfiles are applied;
retained metadata from an unpublished pre-restart wheel declares those bounds.
Existing locked package versions and Python variants are retained. Python 3.11 remains supported. The
SECONDARY stack is the exact framework-floor target; its newer ONNX/ORT and
other dependencies do not establish all-direct-dependency-minimum acceptance.
ORT 1.24.1 and the other declared bounds are unchanged.

Earlier JAX 0.8.1/Flax 0.12.1 runs retain their actual outcomes as historical
evidence; they are no longer a supported-floor release gate. This decision
does not fix numerical/provider gaps or remove any existing model, fixed source
or numerical requirement from the single-interface migration target.

The preceding exact native platform-selector and `eye` forwarding checkpoint
is committed at **`b7eb923a`**. Its PRIMARY full passes
**11,685 unique cases**; the same **110 focused cases** pass on all three JAX
stacks and at actual ONNX Runtime 1.24.1. Strict documentation, all seven normal
hooks and qualified-byte verification pass.

The preceding generalized-contraction/support-floor checkpoint is committed at
**`7b70b22f`**. Current private work preserves the original Equinox Linear call,
shared native contraction/transpose layouts and exact output-axis sample/report
binding for one input-derived batch size. Mixed FP32/FP64 dots follow native
FP64 or explicitly preferred FP32 work through local consumer conversions;
stored parameters, inputs and returned aliases keep their widths. A guarded
shared representation also preserves static empty additions.

Current regression coverage is accounted for across explicit run partitions.
The original failed PRIMARY full and SECONDARY affected runs remain retained;
corrected whole-owner reruns pass on both stacks and at actual direct minima.
The capability-scoped minimum result and its supplement are separate receipts.
Runtime failures remain separate from these test results: some FP64 artifacts validate with optimization disabled
while default ALL fails load, and subnormal comparisons retain their exact
failed bits. No default, source cast, fallback or numerical criterion changes.
The [checkpoint record](poc/source_integration_findings.md#equinox-forwarding-shared-symbolic-contraction-and-output-axes)
binds exact receipts, capability exclusions, failed history and remaining
quality/documentation/commit gates.

Public behavior stays **LEGACY**. Complete models, fixed-source preparation,
original numerical requirements and full B–F acceptance remain open. The target
remains one ordinary source-preserving export contract across existing
functionality and fixed third-party sources. The full product map belongs on a
clean committed checkpoint; the separately frozen callable-preparation packet
remains unapplied and native-unqualified.

The preceding private contraction work preserves ordered contracting/batch axes,
scalar and empty results, native result dtypes and exact calls through
LinearGeneral/DenseGeneral and JAX/NNX/Linen einsum. That preceding checkpoint
records a PRIMARY full pass of **12,061 unique cases** with zero failures/errors/skips. The SECONDARY
full run records **12,054 PASS / two SKIP of 12,056 unique cases**, zero
failures/errors; its skips and five feature-conditional PRIMARY-only cases
are explicitly accounted for. The unchanged sixteen-owner local CI replay at
all direct dependency minima passes **all 465 cases**. Checkpoint acceptance
requires final quality, strict documentation and normal commit checks. The
[current contraction record](poc/source_integration_findings.md#generalized-native-contractions-and-product-map-smoke)
retains every failed epoch, exact test identity and evidence digest, and
separates these checks from whole-model numerical acceptance.

The resumable SOURCE-map coordinator has actual two-cell and resume smoke
qualification on the preceding committed snapshot. It retains three measured
profile exclusions and the complete **1,852-row / 3,704-caller** denominator;
no complete map or runtime/numerical acceptance is claimed. Run the full map on
a separately pinned contraction checkpoint, then prioritize remaining shared
barriers. Numerical criteria, optional models, the 104 registrations without
metadata and fixed-source workflows remain separate obligations. Public
behavior stays **LEGACY** and full B–F gates remain open. The support-floor
change does not waive numerical acceptance or existing functionality. Earlier
checkpoints below retain their historical evidence scopes and support policies.

Private same-width complex64/complex128 Add is committed at `18f3fcd7`, with
finite native/CPU qualification for static shapes at opsets 23/26. It retains
declared pair storage and JAX's chosen width. The [latest checkpoint](poc/source_integration_findings.md#complex-add-and-partial-ordering-checkpoint)
separates those samples from the partial initial raw map and the **FAILED**
ordering-kernel diagnostics. Public behavior remains **LEGACY**; all existing
functionality, fixed sources and full B–F acceptance remain open.

The preceding private [CNN, pooling, scientific FP64 and raw-replay checkpoint](poc/source_integration_findings.md#cnn-pooling-scientific-fp64-and-raw-replay-checkpoint)
is committed at **`118ddd8b`** after independent finite-checkpoint acceptance,
corrected dual-stack checks, the complete PRIMARY suite and all seven normal
code hooks. It retains the unchanged full-CNN nonzero comparison, native pooling
and selected scientific FP64 at opset 21. The raw lane's 49 maintained controls
do not run the full 1,852-row / 3,704-caller map. Public behavior stays **LEGACY**;
the full B–F migration remains open.

The corrected selection passes **1,798 cases / 5,394 phases per stack**; full
PRIMARY passes **10,884 cases / 32,652 phases**, with zero failures/errors/skips.

The preceding explicit FP64 Linear code checkpoint is `b2a866bb`. Its strict
documentation build passed, and its eight-document update is committed at
`97929998`. The canonical findings retain its original runtime and failed epochs.

The private elementary/reduction selection now passes **183 cases / 549
phases on each supported stack**, with zero test failures/errors/skips and
independent actual-evidence acceptance. It includes finite FP32/FP64 Asin
numerical evidence from the same-width rational/half-angle lowering. Passing
cases include diagnostics: **eight original Sign numerical reports remain
FAILED**, and 44 original composition artifacts remain numerically
**UNVERIFIED**. Both original Gather raw graph callbacks remain **FAILED**,
with separately verified finite numerical comparisons. The
[canonical checkpoint record](poc/source_integration_findings.md#elementary-and-reduction-checkpoint)
separates those gates and preserves failed epochs.

The preceding accepted local code checkpoint **`11748451`** is committed. Its
complete PRIMARY suite
passes **10,237 cases / 30,711 phases**, and the focused selection passes
**1,937 cases / 5,811 phases on each supported stack**, with zero test
failures/errors/skips in those executions. All seven normal code hooks pass;
86 committed Python paths (63 production and 23 test/support) match the qualified
bytes, all **923 Python / 163 other input pins** hold through hooks and commit,
and the worktree is clean at commit verification. The focused and full results
have independent finite-checkpoint acceptance. These results retain the separate
failed numerical gates; they do not complete the precision migration.

The native loop checkpoint **`c033d301`** is committed after independent
acceptance: **477 cases / 1,431 phases per stack** and the complete PRIMARY
suite's **10,395 cases / 31,185 phases** pass, with zero failures/errors/skips.
All seven normal code hooks pass, preserving the 1,088 qualified source/input
files through commit. The reviewed scan/while contracts use ordinary SOURCE
admission. The [latest findings](poc/source_integration_findings.md#loop-admission-and-first-saved-model-samples)
retain the two failed focused epochs and distinguish each gate.

Public behavior stays **LEGACY**. The inventory remains **1,852 rows / 3,704
caller obligations**, including all 318 elementary raw contracts / 636 caller
obligations and all 50 optional model obligations. The failed strict Erfc
full/native gates, unverified low-precision and complete original-model work,
optional-environment prerequisites, general fixed-source preparation and
scientific FP64 remain open. The full B–F single-contract migration remains the
goal over all existing functionality and fixed third-party sources, with no
permanent public mode/profile selector or loss of existing capabilities.

All eight saved artifacts across the original six normalized014/015 selectors
now have independent structural and finite CPU-sample acceptance on both
supported stacks: **384 actual returns / 16 sessions per stack**, 768 returns
in total. The frozen source snapshot is separate from current MAIN. Six raw
callbacks per stack remain **FAILED**, two remain **PASSED**, and stored export
reports remain numerically **UNVERIFIED**. The [latest findings](poc/source_integration_findings.md#original-six-saved-samples-and-native-conv-integration) distinguish those results, the retained PRIMARY capture failure,
and the separate criteria for original015.

Native FP32 Conv and the Scan physical-metadata remedy are committed at
**`3223d080`** after independent finite-checkpoint acceptance.
Two failed focused epochs remain recorded; the reviewed test-only oracle
correction preserves production bytes. Latest PRIMARY passes 687 cases;
SECONDARY has 684 passes / three unchanged legacy runtime-fusion skips, pytest
zero but strict-wrapper exit 99. All 586 converter-selection cases pass on each
stack. The complete PRIMARY suite passes **10,504 cases / 31,512 phases**,
with zero failures/errors/skips. All seven normal code hooks pass, all 1,091
source/input files hold through commit, and the worktree is clean at commit
verification. The later Linear checkpoint and the current CNN/scientific work
have separate source and qualification records.

The new CNN comparison uses a fixed, a priori forward-roundoff predicate for
its recorded nonzero samples. Provider-wide arithmetic guarantees remain
**UNVERIFIED**. Original raw callbacks remain **FAILED** and missing raw
`criteria=None` is unchanged; independent finite predicates do not replace them.
The corrected scientific selection passes 141 cases / 423 phases on each stack,
including the unchanged Issue52 payload and its original opset-21 tolerance.
Paired Min/Max subnormal diagnostics preserve exact source operand selection
while recording the unmet native-bitwise criterion as **FAILED**.
See the [current evidence and pending gates](poc/source_integration_findings.md#cnn-pooling-scientific-fp64-and-raw-replay-checkpoint).

The FP16 repair has static acceptance, with native/target and cost checks
pending. The 22 original compositions / 44 artifacts still need explicit
comparison criteria and numerical acceptance. General fixed sources, optional
models and broader scientific/low-precision support remain open.

## What the experiments establish

The [internal experiments](poc/README.md) provide evidence for selected CPU
samples on both supported JAX stacks. Their findings distinguish implemented behavior from remaining
migration gates:

| Finding | Consequence |
| --- | --- |
| The source path preserves chosen mixed dtypes and the selected exact-FP32 example returns `0`, where legacy double export returns `1`. | Preserve source computation as a distinct contract; explain the numerical change when migrating. |
| Public legacy matrix export loses the preferred FP32 result; the private source path now preserves it for bounded FP16 matrix cases. | Preserve input, result and algorithm requirements separately; wider operands have costs that remain unmeasured. |
| A valid source-PoC graph runs on CPU but misses required FP16 intermediate rounding, even with optional runtime optimization disabled. | Runtime execution and numerical acceptance are separate gates. A known numerical failure cannot be certified by matching tensor types. |
| Selected static variance/std and NNX/Linen/Equinox LayerNorm examples pass source and CPU checks. | Acceptance covers the recorded FP16/FP32/FP64 samples and serialized boundaries; the named-boundary proof compares the actual exported body, and broader normalization/model coverage remains open. |
| The calling experiment preserves actual legacy calls and exposes source-call differences, conflicts, and tracing warnings. | Use this evidence for an explicit transition to one normal export contract. The experiment's selector remains private. |
| The private source path now preserves JAX `weak_type`: weak scalar plus explicit FP16 `1` returns FP16 `2048`; a strong input with the same shape and dtype returns FP32 or FP64 `2049`. | Keep supplied, normalized, and traced properties visible. Legacy normalization retains its observed compatibility rules. |
| Source policy reaches selected functions and conditionals, preserves call-time x64 and matrix settings, and isolates cached JAX traces from legacy export. | Durable regressions now belong to regular CI collection. Other settings, scope combinations and export boundaries still need evidence. |
| The CPU sample API checks artifacts from the actual private facade and rejects failed or unverified results. | The known FP16 rounding failure cannot pass sample acceptance. The report covers the supplied artifact, samples and settings, not model-wide deployment readiness. |

The proposed source contract uses FP32 for shape-only inputs. This is an input
default, independent of computation precision. Typed inputs and the caller's
JAX configuration instantiate the model's rules. The public facade retains its
current behavior during development. The final transition requires full
existing-functionality coverage and clear migration errors for the old
`enable_double_precision` argument; no permanent public precision mode is added.
No release is assigned. Existing return types will carry
a metadata report, with a separate CPU sample-validation API for scoped
execution and numerical evidence.

The [private integration findings](poc/source_integration_findings.md),
[numerical findings](poc/numerical_contract_findings.md), and
[calling findings](poc/migration_ux_findings.md) record the exact samples,
versions, controls, and exclusions. A hand-built control or changed-source
example does not establish an automatic exporter implementation. Passing the
experiment tests includes correctly recognizing known gaps and expected errors.
The review found and repaired local function-context loss, cached source/legacy
trace leakage, and an unsafe arithmetic fusion on the source path. Matrix
result preservation and selected wider normalization now have bounded positive
evidence. Required FP16 runtime rounding remains a failed gate on the recorded
targets; the explicit wider example is not an automatic repair. The
preceding code checkpoint `1533d60b` remains accepted. The
[next work](precision_migration_plan.md#next-implementation-work) continues the
remaining inventory mapping, end-to-end models and fixed-source workflows
while preserving original source, inputs and criteria.
Remaining registered families, scopes and export boundaries require the full contract.

The [follow-up review on 2026-10-18](poc/source_integration_findings.md#follow-up-review-on-2026-10-18)
also closed a policy-blind optimizer rewrite and function reuse that ignored
weak typing in captured model state. SOURCE now selects eligible optimizer
passes explicitly and preserves that captured metadata. A further FP32
reciprocal sample still fails its exact acceptance criterion when ORT
optimization is enabled; a captured FP16 Add probe also loses rounding under
both recorded settings. These remain separate runtime observations.

Both views distinguish ONNX validity, preservation of the intended calculation,
runtime execution, and numerical acceptance. The proposed handling of limits is
to use a validated alternative within the selected target, require an explicit
choice for target or precision changes, and report a concrete failure when no
suitable implementation exists. An untested deployment target remains untested.
