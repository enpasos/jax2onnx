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

The [native Exp and Leading-B Mul checkpoint](poc/pointwise_mul_exp_findings.md)
has independent semantic and actual-main acceptance. Native `jnp.exp` now keeps
the original SOURCE call and weak typing; the private audited profile validates
floating Exp and finite static/shared-batch Mul composition. Source casts and
explicit broadcasts remain visible. Main checks pass **1,489 cases per stack**,
including all previous 990 cases, fourteen frozen review guards and 117 new
permanent cases. The separate independent selection passes 153 cases per stack;
these overlapping counts are not additive.

The inventory now records **210 files, 1,217 functions and 465 additions**, with
all 752 original anchors, zero original removals and 1,802 raw contracts/3,604
cells retained. Both inventory CLI checks pass. The complete primary-stack suite passes **6,933 tests** (13,130
warnings, 403.51 seconds). Commit `7c68300b` passes all seven actual hooks;
all five code/inventory paths and all 860 Python hashes remain unchanged.
The fixed Exp/Mul samples establish bounded CPU evidence; BF16 structural
checks do not establish a working CPU kernel. Neither the five-row Leading-B
nor the twelve-row numerical packet is rerun or repinned. Public LEGACY,
`source-static-v1` and report schema 1 remain unchanged. Where/JIT work is a
separate, unaccepted follow-up; broader phases D/E/F remain unfinished.

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
review and actual hooks. The shared Dropout optimizer repair is committed at
`c528a894`; bounded ordinary FP32/FP64 matrix lowering is committed at `2011b140`.
Both also passed independent review and actual hooks. That committed
checkpoint’s complete primary-stack run passes **6,183 tests**, with
197 handwritten-test files, 1,023 functions and 271 migration additions.
The [layout and native-binary checkpoint](poc/layout_and_native_binary_findings.md)
is committed on `precision-migration`: shared observer/naming repair `f02c0b4c`,
binder and Mul-None `258e90d6`, and static NCHW `7609fc2e`. All passed independent
review and actual configured hooks. The initial full run failed ten regressions;
the live-observer correction passes 238 owner and independent checks per stack.
Its root-owned inventory records 200 files, 1,074 functions and 322 additions,
retaining all 752 original anchors and 1,802 raw contracts. The fresh complete
primary-stack run passes **6,351 tests**, with 13,110 warnings in 466.07 seconds.
That checkpoint is recorded at `d1dd3162`. The
[shared-batch checkpoint](poc/symbolic_batch_and_minmax_findings.md) implements
audited Add and named composition for two equal `(B,K)` inputs, with one artifact
checked at B=2/5 and strict per-sample symbol binding. Owner checks pass 697 per
stack and independent checks 241, including all 40 new cases. Commit `80cb68b2`
passed all actual configured hooks; the complete primary-stack run passes
**6,391 tests** (13,110 warnings, 444.36 seconds). Its inventory records 201 files, 1,093 functions
and 341 additions, preserving every raw contract and original anchor. These
historical results remain unchanged.

The [Min/Max and corpus-replay findings](poc/minmax_and_corpus_replay_findings.md)
record checkpoint `fc3f9323`, with independently accepted signed-zero lowering
and repaired B3 tooling. Main focused checks pass 323 tests per stack; both
26-cell replays check 24 artifacts and retain two unresolved outcomes, with zero
ORT. The complete primary-stack run passes **6,466 tests** (13,110 warnings,
413.65 seconds), and all seven actual configured hooks pass. Typing (542 files
plus RNG), global Ruff and both inventory CLI comparisons pass. The inventory
records 203 files, 1,114 functions and 362 additions, retaining all original
anchors and raw contracts. These are historical checkpoint results.

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

The [option and array-creation findings](poc/option_and_array_creation_findings.md)
record exact large-payload serialization, recording behavior, real named BOOL
operands, native JAX promotion/sequence binding and Boolean selection evidence.
Named runtime operands and these array-creation families still require private
diagnostic admission; recording/web options work through the private audited
facade. BF16 target failures, failed exact `linspace` samples and dynamic `arange`
compatibility work remain visible. The
[matrix and optimizer findings](poc/matrix_and_optimizer_findings.md) record the
bounded diagnostic matrix contract and repaired optional Dropout input cleanup.
Selected subnormal matrix samples still fail exact comparison. These changes do
not establish browser, stochastic or model-wide numerical acceptance. Static
NCHW boundary handling and its shared observer/naming dependency have passed
independent review. They do not establish whole-model layout support.
Native add/min/max binding preserves JAX promotion. The eight historical
signed-zero failures per stack remain recorded; new opset-23 FP16/FP32/FP64
artifacts now pass those bitwise samples. Four strict subnormal witness artifacts
per stack and 108 inherited FP32/FP64 grid mismatches remain failed. The dense
selection has measured cost and adds no default admission. Shared-B Add remains
finite; broader acceptance across the 310 symbolic and seven NCHW corpus rows
remains open. The B3 revision
selects 13 unchanged rows/26 artifact-only cells, retaining four explicit SOURCE
matrix counterparts and original callback failures. That artifact checkpoint
adopts no numerical criterion and runs no corpus runtime checks. The new,
separate numerical counterpart preserves the historical packets' None criteria
and history while recording results under `source-corpus-fp32-conjunction-v1`. The trig gather
supplement still produces no artifact or runtime result: x64-off retains its
configuration conflict and x64-on now reaches the custom-Where diagnostic gap.
The remaining 3,580 cells and 126 unresolved recipe scenarios stay visible.

The [first source-corpus batch](poc/source_corpus_findings.md) measured 36 cells
per stack: 26 checked artifacts, four export gaps, five pending cases and one
x64 configuration conflict. Every cell keeps lowering, runtime and numerics
unverified; the other 3,568 cells remain unchanged. A later two-dot-cell
supplement passes schema checking but still fails the unchanged raw structural
callback; it adds no accepted artifact or row. The
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
