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

The supported floor is **Python >=3.11, JAX/JAXLIB >=0.10.2 and Flax >=0.12.8**.
The exact framework-floor stack and the all-direct-dependency-minimum stack
are separate validation targets; ORT 1.24.1 remains the declared runtime floor.

The last fully green checkpoint remains **`fe214590`**; the latest stored
LOCAL WIP checkpoint is **`ce87c2e0`**. That normal local commit completed strict
docs, all seven hooks and exact staged review, with a clean worktree at commit.
Later reviewed fixture, captured-state and corpus-provenance repairs pass their
complete affected selections on both tested stacks; current typing/RNG passes.
These are source-bound focused results. The historical b90 full-suite failure
remains retained, and the full suite has not been rerun. An intermediate WIP
commit preserves reviewed unfinished work after the required checks; it does
not establish a fully green milestone or complete B–F. Public options/defaults
remain **LEGACY**.

The [canonical current status and evidence ledger](poc/source_integration_findings.md#current-checkpoint)
owns exact source/dependency epochs, result scopes and next gates. The
[user snapshot](precision_current_state_user_view.md#local-branch-snapshot)
explains the consequences for exporters. All existing functionality, every
fixed supplied JAX source, the full inventory, numerical contracts and phases
B–F remain required through one uniform source-preserving export experience.
Only demonstrated representation/target limits may restrict support; temporary
implementation gaps do not reduce the migration goal.

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
