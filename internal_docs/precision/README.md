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

The private checkpoint retains the ROOT **786-case / 2,358-phase PASS per
stack** on unchanged production. A separate final-byte test-repair selection
now passes **29 cases / 87 phases per stack**, with zero failures/errors/skips
and held current source/input pins. These are distinct executions. Bounded
tensor/index operations, source-typed Erfc graphs at opsets 23/26 and finite
original GPT_MLP FP32 evidence retain their recorded scope. The complete PRIMARY
suite now passes **10,047 cases / 30,141 phases**, with zero failures/errors/skips
and held current pins. Local code checkpoint **`1533d60b`** passes all seven
normal code hooks. Independent final review **accepts this finite code
checkpoint** and its retained scope limits.
The [canonical checkpoint record](poc/source_integration_findings.md#tensor-index-erfc-and-optional-inventory-checkpoint)
keeps passing finite scopes, failed full-suite epochs and unfinished migration
gates separate.

Configured CPU numerical evidence at **opset 23 only** is positive for the
measured direct Erfc, ordinary composition and named shared-batch scopes.
Opset 26 has graph/schema evidence; its CPU numerics remain unverified. The strict seven-output and native
reference-agreement gates remain **FAILED** because JAX eager/JIT disagree on
FP32/FP64 subnormal prefix values. FP16/BF16 numerics remain **UNVERIFIED**;
production FTZ and runtime policy are unchanged. The old six-Function-model Erfc
barrier is historical and needs a fresh complete-model run after this change.

The inventory retains **1,852 rows / 3,704 caller obligations**, including all
1,802 original core rows and 50 optional MaxText/MaxDiffusion obligations.
Optional imports establish inventory evidence only; configured secondary import
failures remain environment work. Public behavior stays **LEGACY** until the
full existing-functionality B–F gates pass, including original models, general
fixed-source preparation, FP16/BF16 and scientific FP64. No permanent public
precision-mode or partial-profile choice is planned.

Earlier accepted checkpoints, their complete-suite and hook receipts, and all
failed epochs are retained in the [canonical findings](poc/source_integration_findings.md)
and [experiment index](poc/README.md).

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
current code checkpoint `1533d60b` is independently accepted. The
[next work](precision_migration_plan.md#next-implementation-work) is to address
the separate elementary candidate's recorded failures and qualify it, then migrate
the remaining registered families, original models, scopes and export boundaries
through the shared private path.

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
