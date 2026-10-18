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

The [implementation work plan](precision_implementation_work_plan.md) defines
coverage of the existing functionality, completion gates, roles, and local
commits. The completed normal `to_onnx` call has one rule: JAX determines the
computation and export preserves it throughout. The bounded `source-static-v1`
path remains internal and is an intermediate checkpoint. The shared converter
and CPU evidence module are committed; full migration is still in progress.
The private facade, typed constants, logical/physical boundaries, discrete
comparisons and finite noncomplex loop repairs are committed. The `158ce625`
checkpoint passed 5,363 integrated primary-stack tests. Promotion-context,
complex-loop and required-finalization repairs are now committed at `ea6a8da7`,
`8ba3c681` and `f1ee71b3`. Diagnostic tooling, the metadata-ownership repair and
reviewed inventory additions are committed at `3dbec1b8`. Private recording/web
packaging and array-creation promotion repairs are now committed at `8a82d510`
and `223f24bf`, with independent review and actual repository hooks passing.
Private scalar-BOOL runtime binding is committed at `4c1c57fe`; native
`arange`/`linspace` binding is committed at `05037961`. Both passed independent
review and actual hooks. The latest complete primary-stack run passes **6,035
tests**. The inventory retains every raw contract and original anchor, with
195 handwritten-test files, 1,003 functions and 251 migration additions.

The [option and array-creation findings](poc/option_and_array_creation_findings.md)
record exact large-payload serialization, recording behavior, real named BOOL
operands, native JAX promotion/sequence binding and Boolean selection evidence.
Named runtime operands and these array-creation families still require private
diagnostic admission; recording/web options work through the private audited
facade. BF16 target failures, failed exact `linspace` samples and dynamic `arange`
compatibility work remain visible. A separate shared Dropout optimizer defect
is still open. These changes do not establish browser, stochastic or model-wide
numerical acceptance.

The [first source-corpus batch](poc/source_corpus_findings.md) measured 36 cells
per stack: 26 checked artifacts, four export gaps, five pending cases and one
x64 configuration conflict. Every cell keeps lowering, runtime and numerics
unverified; the other 3,568 cells remain unchanged. The
[fixed-source compatibility review](poc/fixed_source_compatibility_findings.md)
classifies all 22 parameterized models: 16 have real dynamic controls, five GPT
configurations are flag-insensitive at zero dropout, and one SpectralNorm case
is static. The subsequent C4c diagnostic pass produces three checked artifacts
and 19 pending rows per stack, with zero ONNX Runtime executions and all
lowering/runtime/numerical gates unverified. These are migration requirements,
not completed SOURCE support. Existing export options, remaining operators and
full-model acceptance stay open. Only phase F completes the transition.

The current-state documents preserve the analysis of `main` at `fae17822` on
2026-10-17, including the merged fix from
[PR #275](https://github.com/enpasos/jax2onnx/pull/275), and separately identify
the newer local branch experiment. That experiment changes private converter
code; the public API and legacy defaults remain unchanged.
The target documents describe the complete semantic design. The work plan
defines its implementation across existing operations, scopes, and options;
temporary unaudited exclusions are outstanding work, not technical limits.
Read down either column for the related analysis and migration guidance.

## What the experiments establish

The [internal experiments](poc/README.md) provide evidence for selected CPU
samples on both supported JAX stacks. Their findings distinguish implemented behavior from remaining
migration gates:

| Finding | Consequence |
| --- | --- |
| The source path preserves chosen mixed dtypes and the selected exact-FP32 example returns `0`, where legacy double export returns `1`. | Preserve source computation as a distinct contract; explain the numerical change when migrating. |
| Public legacy matrix export loses the preferred FP32 result; the private source path now preserves it for bounded FP16 matrix cases. | Preserve input, result and algorithm requirements separately; wider operands have costs that remain unmeasured. |
| A valid source-PoC graph runs on CPU but misses required FP16 intermediate rounding, even with optional runtime optimization disabled. | Runtime execution and numerical acceptance are separate gates. A known numerical failure cannot be certified by matching tensor types. |
| FP32 squares with an FP64 variance sum now work through the source pipeline as well as the independent control. | Selected serialized inline/function/conditional samples establish end-to-end feasibility; broader normalization remains unverified. |
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
[next work](precision_migration_plan.md#next-implementation-work) is to complete
the coverage inventory and source-reference tests, then migrate the remaining
lowerings, scopes, and export boundaries through the shared private path.

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
