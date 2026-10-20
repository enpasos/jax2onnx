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
permanent public mode/profile selector or loss of existing capabilities. The first saved original-model artifact now has independent bounded structural
acceptance and passing comparisons against its exact captured CPU samples. The
original raw graph callback stays **FAILED**, and the earlier stored export
report stays numerically **UNVERIFIED**. Other original cells and model-wide
acceptance remain open; see the [separate saved-artifact result](poc/source_integration_findings.md#loop-admission-and-first-saved-model-samples).

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
remaining original-model cells under their unchanged criteria and qualifies the
separate native Conv candidate. That candidate is still external and static-only.
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
