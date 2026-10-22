# Computation Precision Migration Plan

Status: accepted plan for the complete migration, updated 2026-10-22. The final
normal `to_onnx(...)` call preserves the JAX computation through one consistent
contract across existing functionality and fixed sources, without a permanent
public precision mode/profile selector or capability loss. The private
source-static milestone is implementation scaffolding, not completion scope.

Public precision behavior remains LEGACY. The [canonical current ledger](poc/source_integration_findings.md#current-checkpoint)
separates completed `fe214590` qualification from later directed results. The
unchanged repeated-Function/If recipe now passes all three stacks. Portable BF16
source words and strict native NaN failures remain separate. The current PRIMARY
Transformer owner passes; the fresh current suite and full migration remain open. The implementation
plan retains all coverage obligations; no release version or deadline is set.

Read the [user migration plan](precision_migration_plan_user_view.md) for the
proposed caller experience, the [current analysis](precision_current_state.md)
for existing public behavior and the [target design](precision_target_state.md)
for the complete numerical contract. The plan below is independent of any
particular implementation checkpoint.

## Migration objective and boundaries

The target is to lower the typed JAX graph produced under a documented tracing
configuration. Its dtypes, precision-relevant primitive parameters, numeric
casts, bitcasts, and boundary signatures define the source contract. Input
normalization must preserve precision-relevant properties before that graph
is traced. ONNX adaptations follow each operator's operand roles and preserve
that contract under documented numerical policies. This goes beyond tensor
dtype fidelity without promising bitwise agreement for every JAX/runtime
calculation.

Support is finite and must be documented for the operation, dtype, shape, opset,
and deployment conditions being claimed. Every stage separates four checks:
ONNX schema validity; preservation of source semantics, including rounding and
accumulation; runtime kernel availability under the specified version, provider,
hardware, and settings; and numerical acceptance. Record each check as validated,
failed, or unverified, with its evidence. A failed gate names the cause:
representation restriction, lowering defect, unavailable kernel, or numerical
failure. Unknown support is not a demonstrated failure, and a demonstrated
failure must not be described as merely unverified.

The migration must preserve the model's chosen computation, accumulation, and
output types, together with deliberate conversions in each region. A model may
remain entirely in a lower precision without any FP32 stage. Representative
cases also include FP16/BF16 bulk computation with selected FP32 stability
regions and a deliberate return to the lower dtype, or scientific and physics
computation requiring FP64. These examples do not exhaust the supported policies.

Choose precision rules in model configuration before tracing. The
[target examples](precision_target_state.md#example-exact-fp32-or-at-least-fp32)
distinguish an exact conversion from a local minimum. Export inputs instantiate
such rules; they do not rewrite fixed casts, parameter types, or third-party
constraints. Configure JAX x64 before creating values or tracing FP64 work.

Export performs no implicit numerical FP32-to-FP64 transformation. Explicit
JAX-side preparation is required compatibility work for supported fixed-source
workflows that existing configuration cannot express. It must produce a defined
prepared computation and independent native reference before ordinary export,
while the supplied source remains unchanged. Preserve exact casts, accumulation,
stored parameter values and output contracts unless the requested preparation
explicitly changes them. Widening stored weights cannot recover lost precision.
No general transformation API is implemented or assumed, and a particular
model's intended numerical changes must not be inferred from an FP32 dtype.
This requirement is settled; implementation and acceptance remain open.
Development does not change `enable_double_precision`. Its legacy behavior includes
temporarily disabling JAX x64 when false, selecting defaults for shape-only
inputs, and applying the existing export precision rules. Compatibility with
those rules is distinct from preserving the existing conversion capabilities.
The completed migration retains those capabilities under one precision rule;
it does not retain a permanent second public precision policy. Both explicit
legacy Boolean values will receive an actionable migration error at the final
transition. FP64 remains available through model configuration, JAX x64, and
typed values. An export-wide widening rule cannot also preserve every exact
FP32 cast, so equivalent old numbers or unchanged call syntax are not promised.

## Current gates and order of work

| Stage | Evidence available | Remaining gate |
| --- | --- | --- |
| 1. Inventory | The reconciled baseline retains 1,852 rows / 3,704 caller obligations: all 1,802 unchanged original core rows plus 45 MaxText and five MaxDiffusion obligations. | Complete optional SOURCE/model/numerical evidence in prepared upstream environments; retain configured import failures, name collisions and known failures explicitly. |
| 2. Legacy repair | PR #275 is merged and its selected consumers are covered. | Audit remaining consumer gaps without reinterpreting the legacy flag. |
| 3. Internal source path | Weak input metadata, bounded preferred-FP32 matrix results, and model-chosen FP64 normalization work have source-export evidence. Unvalidated algebraic fusions are disabled under SOURCE. | Use the internal profile as a safe starting point; migrate every existing operator family and distinguish unfinished work from demonstrated representation or runtime limits. |
| 4. Scopes and artifacts | Selected functions/branches, local precision settings and cache isolation have durable regressions. The committed private facade preserves names, reports and proto/IR/file serialization; actual artifacts reach the CPU sample gate and integrated regression checks pass. | Preserve all existing scopes, dynamic shapes, parameters, layout options, normalization choices, and artifact formats under the same contract. |
| 5. Public transition | One normal export contract, explicit legacy-argument migration errors, metadata reports, and separate sample evidence are selected. | Complete phases B-F of the work plan before changing normal public behavior; the internal phase A does not complete the goal. |

Review exposed two function-context defects and one arithmetic-fusion defect.
They are repaired privately and have regressions; see the
[review findings](poc/source_integration_findings.md#review-findings).
The subsequent [Astra Ultra review](poc/source_integration_findings.md#follow-up-review-on-2026-10-18)
found two further defects at optimizer and captured-state cache boundaries.
6.1 Sol Ultra implemented their bounded repairs: explicit optimizer eligibility
and SOURCE keys that include captured promotion metadata. Independent review
accepts those repairs; numerical acceptance remains a separate gate.
Experiments may inform later stages before an earlier stage is complete;
the public transition requires the complete existing-functionality acceptance
matrix, not only the bounded internal profile.

## Next implementation work

Current 9134 adds only reviewed local typing annotations/key rename; fresh
typing/RNG and both scoped 279-case SECONDARY/DIRECTMIN selections pass. The reviewed b90 live-branch restoration passes the 331-case PRIMARY
joint gate, including all 44 new controls, unchanged Transformer 52 and repeated
Function/If 2. Original Cast bytes/artifacts, endpoint bindings, mathematics and
final precision equality remain unchanged. The earlier two Transformer 26
failures are closed in this scope, while historical receipts remain intact.

The same ordinary full suite is terminal and failed. All 54 model/state runs
plus 2 guards pass, while remaining failures stay unresolved in the canonical
ledger. These positives do not complete full-suite or migration acceptance. Do not inherit the
historical 19 PASS/37 unfinished result. Fresh current 9134
SECONDARY and DIRECTMIN each pass 279 cases, and typing/RNG passes. The proposed LOCAL
WIP commit may preserve unfinished changes after relevant actual checks, all
seven hooks and exact staged review; it is separate from an accepted checkpoint
and need not repair every full-suite failure or require another broad run first.
A green full standard
suite would close this checkpoint gate, not the full migration. The complete
map, 63 opaque originals, 104 metadata-free registrations, 752 original manual
anchors, fixed-source/numerical workflows and coordinated ordinary export/
allclose/browser/generator cutover remain required. Public behavior stays LEGACY
until that complete uniform transition is qualified.

Preserve original factories/configuration, parameters, source casts, RNG, state,
aliases, callbacks, frozen criteria and actual target evidence. Prepared and
original computations keep separate native references and retained outcomes.
Current three-stack recipe success does not complete every model or scope.
Installed-capability diagnostics retain actual failed schema/load outcomes;
passing diagnostic assertions never makes the failed capability available.

Complete all22 original owners, the unchanged1,852-row/3,704-caller inventory,
optional environments, broader operators/scopes/options, fixed supplied JAX
sources and scientific/low-precision contracts before the public transition.
Only demonstrated representation and target limits define support boundaries.
Temporary missing implementation is unfinished work, and the final experience
remains one uniform source-preserving converter contract across the full B–F
acceptance matrix.

The implementation follows these acceptance requirements:

1. **Record and enforce the initial subset.** List operations/parameters, dtypes,
   shapes, scope combinations, tracing settings, opsets, and claimed execution
   conditions. The current allowlist is an exploration boundary, not proof for
   arbitrary compositions. Keep known failing FP16 compositions and the
   optimized FP32 reciprocal sample excluded from acceptance under their
   recorded exact-sample requirements. A sample-only guard must not become a
   claim that every input or every accepted graph was validated.
2. **Exercise the shared export boundary privately.** Integrate source policy
   with input/output names and the return formats selected for the initial
   subset. Reject the deferred layout and parameter options. Test the actual common path
   beside legacy, including failure restoration and saved artifacts. Deferred
   boundary options and scopes must fail explicitly until migrated. Avoid a second independently
   maintained product export pipeline.
3. **Connect validation evidence to the selected caller experience.** Keep
   successful artifact creation separate from target execution and numerical
   acceptance. Carry a versioned metadata report accessible through
   `get_precision_report`; provide `validate_source_samples` and its acceptance
   guard for evidence tied to an artifact, supplied samples, criteria, and CPU
   settings. Known failures remain failed. This integration now has fourteen
   actual-facade CPU cases covering accepted controls and numerical witnesses;
   the full existing-functionality matrix remains open.

These requirements complete internal phase A only. Phases B-F then establish an
independent source oracle, migrate the complete operation and public-option
inventory, validate representative full models and targets, and switch the
normal public API to the single contract. A missing implementation, an unaudited
plugin, or an untested existing option is outstanding work, not a technical
boundary. The final gate permits only evidenced representation/tracing limits,
separately reported runtime or numerical failures, and explicitly scoped missing
deployment evidence. It requires zero unresolved migration gaps.

Preserve every metadata row with its registration/module/testcase provenance;
identical display names are not evidence of equivalent tests. Add regression
and boundary suites outside the registry to the matrix. Record optional
MaxText/MaxDiffusion discovery and execution status explicitly. Reuse raw test
metadata and deterministic factories, but build independent JAX references:
the legacy generator rewrites dtypes and x64 settings and cannot serve as the
source oracle. Frozen per-case comparison criteria follow strict output dtype
and shape checks. Skipped numerical validation remains unverified.
The initial scenario plan leaves missing, one-sided and legacy-variant-only
criteria unresolved; it does not invent exact comparisons. Freeze reviewed
criteria before numerical runs. Integer and Boolean outputs always require
exact values, including mixed outputs whose floating values use tolerances.

## Stage 1 Inventory and freeze observable behavior

Status: partial inventory with reproducible observations, not a full support
matrix.

Map every precision decision from public input normalization through tracing,
constant binding, plugin emission, child scopes, optimization, and export
post-processing. Record which decisions change the traced program and which
rewrite its lowered representation. Include explicitly typed inputs, weak
scalars, closed-over constants, constructors, complex representations, and
normalization plugins. Record precision-relevant input properties such as
`weak_type` alongside shape and dtype. Legacy normalization retains the observed
loss for compatibility; the private source path now preserves the property.
Keep the former source failure as a regression, rather than freezing its
changed numerical result as desired source behavior.

Inventory the feasible implementations and their restrictions alongside legacy
behavior. Use the [current feasibility evidence](precision_current_state.md#onnx-feasibility-and-runtime-evidence)
as bounded examples: same-width `BitCast` requires opset 26 rather than the
current default 23; a tested FP64 `Conv` lacks an ONNX Runtime 1.31 CPU kernel;
and FP32 tensor types alone do not establish arithmetic precision when settings
such as TF32 affect execution. These are specific observations, not universal
support classifications for those operators.

Inventory model-chosen computation, accumulation, and output types per region
and the intended role of each conversion. A local minimum precision is one
possible policy, not a global floor. A hardcoded `astype(float32)` does not
establish a minimum. Where source changes are possible, validate the intended
JAX behavior before export.

Turn examples into focused regressions covering both legacy flag values and
ambient x64 settings. Include lower-precision computation without FP32 promotion,
FP16/BF16 bulk work with selective FP32 regions and explicit casts back, and
scientific FP64 cases. For the local FP32 example, contrast exact and
minimum FP32 after FP64 inputs, using rounding-sensitive values that distinguish
`FP64 -> FP32 -> FP64` from uninterrupted FP64 arithmetic. Capture valid legacy
behavior; retain invalid models as known failures, never desired results.

Exit criterion: the inventory separates intentional legacy behavior, defects,
feasible alternatives, unsupported cases, and unverified targets. Existing
failures remain identifiable when later stages change graph construction.

## Stage 2 Repair invalid legacy type combinations

Status: partially completed. PR #275 supplies the bounded legacy repair on
`fae17822`. Selected arithmetic, unary, reduction, and concatenation lowerings
explicitly adapt FP32 operands when their consumer requires FP64, preserving
preceding FP32 rounding and the original binding. Direct FP32 outputs and
unaffected lower-precision paths retain their existing contracts. The public
`enable_double_precision` behavior remains the compatibility policy.

This is evidence for those covered cases, not completion of every feasibility
check below. Remaining gaps must follow the inventory; the merge does not
establish universal consumer coverage or generic runtime-aware failure handling.

Use the native `require_value_for_var`, `allocate_value_for_var`, and
`lookup_value_for_var` roles. Final outputs require existing bindings; do not use
writable output allocation as an output lookup. General lookup must
not silently substitute a promoted value. A plugin can explicitly request an
operand adaptation through a shared typed helper; the original variable binding
remains available to other consumers. Any reuse of adapted values must respect
graph ownership and destination dtype.

Derive adaptations from operator schemas and semantics. Casting every operand
to the result dtype is incorrect: comparisons return booleans, gather indices
are integers, and control-flow conditions have distinct roles. Protect bitcast
input and output representation constraints; numeric casts cannot replace
reinterpretation semantics.

Validate the repaired consumer against its ONNX schema and actual supported
execution conditions. Preserving an explicit cast is insufficient if the next
operation still has no usable implementation. Add successful fallback cases and
expected failures; an invalid model must not become the accepted compatibility
result. Where no feasible implementation exists, fail before reporting a
successful export, naming the model location, operation, operand/result dtypes,
opset, declared runtime, failed requirement, and available alternatives.

Exit criterion: focused numerical and structural tests establish valid legacy
models, preserved explicit rounding and outputs, and correct native ownership.
The merged repair does not establish the target contract or justify a new
public flag. Mark Stage 2 complete only for the explicitly recorded scope and
evidence; retain unresolved cases in the inventory.

## Stage 3 Build an internal path that preserves source dtypes

Status: experimental, with open acceptance failures. The
[integration experiment](poc/README.md) demonstrates selected type-preserving
paths, not a completed precision contract.
[source_export.py](poc/source_export.py) reuses the real tracing helper, primitive
dispatch and plugins, optimizer, finalization, post-processing, and serialization.
The private `PrecisionPolicy.SOURCE` now selects traced output dtypes in the
real context and propagates through existing child-context creation. Source
input preparation preserves weak metadata; constant bindings and
post-processing suppress legacy widening. The caller's active JAX x64 setting
is left unchanged. This replaces the initial local allocation subclass and
changes private converter/plugin code. The public entry point and its legacy
defaults remain unchanged.

The [demo](poc/demo.py) compares source and exported types and results on chosen
samples. For example, exact FP32 arithmetic after FP64 input `1e8` returns `0`
in FP32, while the merged legacy double export returns `1` in FP64. Other cases
exercise selective FP32 work in FP16 computation, deliberate output conversions,
mixed FP64/FP32 outputs, wider reduction work, and a shared value used by a
same-width bitcast. These demonstrate numerical behavior as well as type labels.

The initial sample evidence below is bounded to explicitly typed static inputs,
a small primitive allowlist and opsets 23 or 26. Current generic SOURCE admission
accepts 16, 18, 21, 23, 24, 26 and 27, with per-operation/type checks; that later
admission does not broaden the original samples. Some CPU samples agree; a new FP16 composition
fails the required runtime rounding. An allowlist of individually covered
primitives is therefore not a support claim for every composition. BF16 has
structural evidence in the source prototype, with separate known kernel
limitations recorded in the current-state analysis. Opset-23 bitcasts and
different-width bitcasts are rejected. Audited `onnx_function` calls and
two-branch `cond` are included, with floating `gt` and Boolean-to-INT32 selector
conversion. The added arithmetic paths include Div, BroadcastInDim, both square-root
plugins, and bounded matrix products. Source lowering keeps separate arithmetic
operations instead of unvalidated Mean, LpNormalization, ReduceSumSquare, and
ReduceL2 fusions. A recursive validator checks both branches and actual retraced
function bodies before lowering. The later shared-B Add/named-composition packet
adds its finite symbolic contract, and the later Where/JIT checkpoint admits
bounded pure closed JIT bodies. The later native loop packet admits its finite
scan/while contracts through the same recursive validator. Remaining loop/JIT
forms, broader symbolic shapes, effects and other unaudited operations retain
temporary gates, without implying that ONNX cannot support them. This advances Stage 3; it does not
complete the stage, establish general numerical equivalence, or make a public
mode available.

The [numerical experiment](poc/numerical_contract_findings.md) and
[input regressions](poc/migration_ux_findings.md#weak-input-properties-preserved-on-the-source-path)
supply concrete acceptance evidence and remaining gates:

| Gate | Observed evidence | Required next result |
| --- | --- | --- |
| Input promotion before tracing | Repaired privately: both normalization steps retain `weak_type` under SOURCE. Weak inputs return FP16 `2048`; strong inputs with the same shape/dtype return FP32/FP64 `2049`, matching original JAX references. | Retain array/specification, direct-helper/UX, both-x64 and both-stack regressions, plus weak-specific function signatures and source/legacy isolation. Extend precision-relevant metadata coverage with each newly audited input path. |
| Matrix result and accumulation | Public legacy export still loses the preferred FP32 result. The private source lowering now preserves FP16 inputs and returns FP32 `2049` for bounded default-precision matrix cases. | Keep the admitted result/algorithm distinction explicit; separately validate additional shapes, dtypes, algorithms, targets, and the cost of wider operands. |
| Selected wider normalization | Both the separate ONNX control and the actual source export preserve FP32 squares, FP64 variance work, and exact variance `8388609`. Source checks cover inline, Function, and both If branches. | Retain exact intermediates/variance and constant-row zero, and extend numerical coverage only with justified final-output tolerance. |
| Intermediate FP16 rounding | The source graph now retains Mul then ReduceSum, but ORT still returns sum `2**-23`. The private sample gate rejects it on both recorded runtimes and rejects unverified samples. | Preserve or exclude this composition for a claimed deployment. Integrate the exclusion/evidence with any eventual public target interface; the sample gate is not a general capability engine. |
| Candidate source changes | Explicit wider operands and explicit FP16 rounding between wider operations pass their selected samples. | Establish a bounded semantic/numerical policy before implementing an automatic alternative. A changed JAX example or hand-built graph alone does not satisfy this gate. |

The bounded matrix and normalization rows now have positive source evidence;
the FP16 runtime failure and broader coverage remain open source-contract
gates even though the experiment suite passes. A diagnostic test can accept a
future runtime improvement without
requiring the known discrepancy to persist. An observed numerical failure
remains failed evidence until the implementation or target is revalidated.

The [calling experiment](poc/migration_ux_findings.md) already exercises
shape-only inputs, typed specifications, arrays, ambient x64 states, conflicts,
and actual trace warnings. Carry its comparisons into the internal path while
keeping its argument names private. Retain the observed three differences:
shape-only defaults, tracing configuration, and late legacy promotion.

The 46 input regressions now compare weak and strong inputs with the same shape
and dtype using JAX arrays and `ShapeDtypeStruct`, both ambient x64 settings,
and both supported stacks. They check original JAX references, actual traced
properties, serialized CPU results, and all three real public legacy routes.
Metadata is retained through both `migration_ux._source_inputs` and
`conversion_api._as_sds_list` under SOURCE, and diagnostics expose supplied,
normalized, and traced weak flags. Keep these comparisons as the implementation
broadens. A prototype rejection must identify an experiment limit rather than
an ONNX restriction.

The explicit private policy separates source tracing/lowering from the legacy
Boolean. Existing calls default to `LEGACY`, and source lowering uses traced
aval dtypes through the audited allocation, constant, and finalization paths.
Keep the source path private while completing its operator and pipeline audit;
disabling one legacy promotion site alone would not establish the target
contract. Wider scope support must inherit the policy and validator through the
real child-context creation path, rather than reconstructing a local context
override.

The target path uses the caller's effective JAX configuration. Enabling x64
affects defaults as well as FP64 availability, so conversion must not silently
change it. Explicit FP64 input specifications under disabled x64 need actionable
errors before tracing. Surface JAX warnings about model-internal FP64 requests
separately: those may already have become FP32 in the traced graph. An FP32
trace alone cannot establish an earlier FP64 intention. The proposed source
contract uses FP32 for shape-only inputs and preserves explicitly typed inputs
or arrays. That default does not constrain model-internal precision. Explicit
FP64 inputs need typed specifications or existing FP64 arrays; model-internal
FP64 work can also follow explicit source conversions with x64 enabled.
For scoped overrides, including the legacy
path, restore prior state after success, exceptions, and nested exports.
For function retracing, preserve the effective configuration of the original
call, including model-local overrides. Configuration read later while lowering
is not a substitute. The private function path captures and replays x64 and
default matrix precision and separates ordinary/legacy/source JAX trace caches.
These tested settings do not establish preservation of every JAX option.
Concurrent configuration isolation requires separate validation.

Acceptance must distinguish the three responsibilities explicitly. The model
defines precision rules; typed export inputs and active JAX configuration
instantiate them during tracing; export selects an ONNX implementation of that
computation. Exercise the same dtype-dependent callable with different typed
inputs, checking its actual computation and output types. Also check that fixed
model casts retain their meanings and that an FP32 input does not acquire FP64
arithmetic merely from a legacy export setting.

Give operator adaptations explicit input and output contracts. Complex storage
representations need semantic checks rather than universal equality between
source and IR dtypes. Preserve native ownership checks and IR-only construction.
Test this path beside the legacy path using internal entry points, without
exposing a partially working mode.

Select implementations within the allowed target policy. When a direct native
implementation is unsuitable, prefer validated standard decompositions or
representation adapters that preserve canonical bindings, required conversions,
and the agreed numerical behavior. Local implementation conversions must not
become a global dtype rewrite. In particular, do not hide a missing FP64 kernel
behind FP64-to-FP32-to-FP64 casts.
A precision reduction requires a separate, explicit numerical transformation.

Changing opset, falling back to another execution provider, or introducing custom
operators requires explicit permission in the target policy. A custom operator
also changes portability and needs its runtime dependency documented. Do not
automatically raise the model's opset or treat a CPU fallback as evidence of
execution on the requested accelerator.

With no declared runtime, an export meeting its representation and lowering
requirements may proceed with runtime/numerical checks marked unverified.
For a declared target, a known missing kernel or numerical failure requires a
validated alternative or an error. A valid model that executes but fails its
numerical gate cannot count as target-ready. Unverified support must remain
visibly unverified; deployment readiness requires the missing evidence.

Runtime sessions remain separately configured by the caller. Acceptance must
compare declared requirements with the runtime/provider, hardware, math modes,
and optimization settings actually used. A declaration is not evidence that
export configured the session or that a requested provider executed the model.

Include adversarial intermediate-rounding samples, with runtime optimization
both enabled and disabled. A valid, runnable graph can still fail this numerical
gate. Also distinguish the result type of a matrix product from its accumulation
algorithm. An explicit wider-input representation needs its own correctness and
cost evidence; a successful sample does not establish a general translation of
every JAX precision option.

Exit criterion: the documented support subset covers the representative precision
policies and passes the four checks for its claimed targets. Cases outside that
subset produce the appropriate explicit failure or unverified status, without
changing source precision to force success.

## Stage 4 Complete scopes and finalization

Status: selected boundaries, local numerical contexts, and cache isolation have
durable evidence; broader boundaries remain open.
The real child-context path now carries the explicit policy and validator.
The [scope regressions](../../tests/extra_tests/converter/test_source_scopes.py) export the same mixed
FP64/explicit-FP32 calculation inline, inside one `onnx_function`, and across a
two-branch `cond`. Shared rounded values, direct mixed-type outputs, FP32/FP64
constants, and both branch choices survive save/reload and CPU execution under
both tested runtime optimization settings. All 16 scope cases pass on both
stacks, including static arguments, structured return trees, repeated result
aliases, and formal-input passthrough. Success establishes these cases only;
the later Where/JIT checkpoint adds its separately checked pure bodies. Broader
loop/scan/JIT and public boundary handling require their own work.
The review's local-x64 witness now retains its intended result, including when
input/output types stay FP32. Source function metadata captures x64 and default
matrix precision at the call and replays them for body tracing. Cache signatures
distinguish those settings and weak flags. Separate JAX trace-cache namespaces
prevent a cached source branch from entering a later legacy or ordinary-JAX
trace. Retain these regressions, including scoped settings around existing FP64
values, failure restoration, and source/legacy interleaving.

The SOURCE function variant preserves mixed result avals and reconstructs
selected tuple/list/dict return trees. Weak and strong calls produce distinct
function specializations. Broader nested-scope combinations still need durable
acceptance before being admitted.

Before exposing the path, verify these boundaries together:

- **Constants:** payload, declared dtype, literal materialization, constant
  folding, and post-processing agree. No late pass silently promotes a constant
  or preserves incompatible metadata.
- **Functions:** callers, formal inputs, body results, and return values agree.
  Cache keys distinguish dtype signatures and any tracing configuration that
  changes the body. Nested functions inherit the selected contract explicitly.
- **Control flow:** captured values, branch outputs, loop-carried state, and scan
  results retain compatible types. Iteration counters and conditions keep their
  separate schema roles.
- **Export boundaries:** direct outputs, layout transposes, custom names, and
  dynamic parameter materialization preserve dtypes. Optimizers and recursive
  post-processing retain required numeric casts and bitcast representations,
  including entry to FP32 stability regions and deliberate returns to FP16/BF16.

Follow the existing [builder rules](../../docs/developer_guide/advanced_topics/onnx_ir_builder.md)
and [subgraph contracts](../../docs/developer_guide/advanced_topics/subgraph_input_handling.md).
Check serialized output as well as raw IR: a correct intermediate graph is
insufficient if finalization changes its precision.

Apply feasibility checks recursively to function and control-flow bodies. A
fallback must preserve captures, signatures, state, and required rounding in
every affected scope. After any allowed opset or implementation change,
revalidate the complete model; changing its opset declaration alone does not
make existing nodes compatible with a different schema. Record graph-size,
memory, and performance costs separately from numerical acceptance.

Exit criterion: every scope combination, boundary option, and return mode
admitted by the declared initial subset satisfies the same contract through
serialization. Deferred boundaries are explicitly rejected rather than
implicitly accepted. Known infeasible target cases do not return a successful
artifact satisfying that target; unverified deployment support is not reported
as validated. Completing this bounded gate does not claim universal scope or
public-option coverage. Completing the overall migration additionally requires
the full existing-functionality scope and boundary matrix in phases C-F.

## Stage 5 Implement and document the selected public transition

Status: final single-contract architecture accepted; public cutover pending
phase F of the [implementation work plan](precision_implementation_work_plan.md).
M5 integrates the shared facade privately and does not change public behavior.
Its implementation and focused evidence are reviewed and committed at
`3b3a48e0`; the integrated primary-stack regression passes. This checkpoint does
not establish the full source-coverage matrix required for public cutover.

Make the normal `to_onnx(...)` call preserve the source contract only after the
full existing-functionality gates pass. Do not expose a public precision mode
or permit an individual lowering to fall back to legacy arithmetic. Preserve
omission versus an explicitly supplied legacy flag with an internal sentinel.
The final argument behavior is:

| Call | Required transition behavior |
| --- | --- |
| Legacy argument omitted | Preserve caller JAX configuration, typed and weak input properties, and the traced computation; shape-only inputs default to FP32. |
| Explicit `enable_double_precision=False` | Reject before tracing with instructions to omit the flag and configure any intended JAX x64 behavior explicitly. |
| Explicit `enable_double_precision=True` | Reject before tracing with instructions to configure JAX x64 before constructing model values and supply the intended FP64 model/parameter/input types. |

This is a deliberate public behavior and argument transition, with release
notes, replacement examples, and migrated repository callers. Preserving
conversion capabilities does not promise identical old arithmetic or unchanged
call syntax. Hardcoded FP32 casts remain FP32; requesting FP64 instead needs
explicit model configuration or preparation of a different JAX computation,
with its own reference. Cover fixed third-party
sources through external configuration/preparation and caller wrappers, without
requiring edits to their implementation. Account for previous FP64 workflows
under this constraint before cutover. Historical legacy fixtures can
remain private references, but every genuine existing capability needs a source
counterpart before cutover.

Migrate `input_params` to one explicit runtime-input contract as well. Each
declared value becomes a genuine typed argument in the whole-model JAX trace;
do not discover dynamicness from decorated-function usage or attach synthetic
flags after specializing the root. Fixed keyword configuration moves to a
closure, `functools.partial`, or model construction. This deliberately replaces
the old mixture of fixed root kwargs and selectively promoted Function flags.
Preserve all 22 existing metadata-model capabilities and the handwritten static
configuration cases through runnable migration examples. Unsupported Python
branching on a declared runtime value needs an actionable tracing diagnostic;
it does not make the previous static calculation unavailable. Verify both BOOL
values against original JAX, defaults/literal overrides, unused arguments,
Function/If/Loop forwarding, names and report arity. The existing mixed
static-root/dynamic-Function witness must never produce a misleading artifact.
The reviewed named binder implements this mapping for the private audited path;
the unchanged all-22 sources expose their individual barriers. The later
NNX/Linen MLP and explicit Dropout checks provide finite model/state acceptance;
complete all-22 model and stochastic coverage remains open.
The tracing strategy must pass compatibility examples using those existing
models with unchanged sources. A static wrapper cannot replace a previously
working dynamic control. If a supported baseline model would require a source rewrite, keep the
case open and implement an appropriate semantics-preserving adapter; do not
close it as a newly declared tracing or ONNX limit. The
[completed 22-row classification](poc/fixed_source_compatibility_findings.md)
now supplies that compatibility baseline: 16 live positive-rate controls, five
zero-rate GPT configurations and one static SpectralNorm calculation. Numerical
and stochastic SOURCE acceptance still precedes completion of the new contract.

Preserve existing return types and store the
versioned export report under metadata key `jax2onnx.precision`.
`get_precision_report(model_or_path)` exposes supplied/normalized/traced/exported
properties, warnings, settings, and scoped checks. Export leaves runtime and
numerical checks unverified. The separate CPU `validate_source_samples` API
records actual target settings and independent reference comparisons; its
acceptance guard certifies only the bound artifact and samples. Full interface
details and external-data identity requirements are in the work plan.

Document the supported deployment conditions and the distinction between valid
ONNX export and validated execution. The public transition must explain target
restrictions, permitted implementation alternatives, and how missing evidence is
reported. It must not promise arbitrary operation/dtype combinations on every
provider. Explain migration with representative cases: entirely lower-precision
models, selective FP32 regions, and true FP64 computation. Preserve exact casts;
suggest minimum-precision source changes only where intended and under caller control.
Deployment choices are independent of a request to change model precision.

Migration examples must explain three distinct behavior changes: shape-only
input defaults, preservation of the caller's JAX configuration, and removal of
late legacy promotion after JAX has chosen types. The last can change arithmetic
and output dtypes even when explicit input specifications are unchanged.
Requesting FP64 computation then requires suitable model rules and tracing
inputs, rather than relying on export to rewrite an FP32 computation.
Also document corrections to input preparation, including preserving
`weak_type`: fixing a known defect can change the traced computation even when
the supplied shape and dtype stay the same.

Migrate `allclose`, browser validation, and their reference/input preparation to
the same public contract. Do not silently cast runtime outputs to the expected
dtype or override JAX configuration to make a comparison pass. Logical/physical
adapters such as complex packing and NCHW layout remain explicit. Browser
execution evidence must use an independent JAX reference; CPU execution alone
does not validate the JavaScript target.

Do not introduce a separate `explicit_float32` switch. If demand later justifies
rewriting hardcoded FP32 models, assess a separate graph-wide numerical
transformation with defined coverage and exclusions. It is not a prerequisite
for dtype-faithful export.

## Release validation and rollback

Use the following matrix to guide meaningful coverage rather than multiplying
every case indiscriminately. The reproducible inventory, all public options,
extra regression suites, and representative end-to-end models define the
required migration scope. The internal static profile is only an early gate.

| Dimension | Required evidence |
| --- | --- |
| Contract | One normal source-preserving public path across the complete baseline; historical legacy comparisons explain intentional differences |
| Inputs and tracing | Typed FP16/BF16/FP32/FP64 where supported, shape-only defaults, weak/strong inputs with the same shape and dtype, both ambient x64 states; preserve promotion properties or reject unsupported inputs before tracing |
| Workloads | Lower precision without FP32 promotion; selective FP32 with deliberate cast-back; scientific FP64 examples |
| Semantics | Model-chosen computation/accumulation/output types; local exact vs minimum FP32 example; rounding-sensitive casts and constants |
| Operator roles | Floating data, integer indices, boolean conditions, mixed-result operators |
| Representation | Bitcast widths and shapes; structural validity and runtime support reported separately |
| Boundaries | Direct outputs, layout bridges, functions, branches, loops, scan |
| Compatibility | Both repository-supported JAX stacks through the compatibility layer |
| ONNX limits | Supported opsets and schema/type restrictions; valid decompositions and explicit upgrade requirements |
| Deployment target | Runtime/version, provider, hardware, optimization and math settings including TF32 where relevant |
| Evidence | Validated, failed with a concrete cause, or unverified for each of the four checks |
| Failure behavior | Missing kernels, forbidden alternatives, unknown readiness; no invalid model accepted as success |
| Fallbacks | Preserved rounding and signatures, complete-model revalidation, separate memory/latency measurements |

Restore changed JAX configuration in test teardown, including failure paths.
Keep fixtures deterministic and use bounded parallelism. Start with focused
regressions; run the full suite, linting, formatting, typing, builder-policy
checks, and strict documentation build before shipping implementation changes.
Use appropriate numeric tolerances and retain structural assertions for casts
whose loss could be hidden by benign test values. Compare accuracy and
performance independently for the representative workload cases. Neither faster
execution nor lower memory use establishes adequate numerical behavior, and improved
accuracy alone does not establish an acceptable deployment cost. Export does
not autotune those tradeoffs or select new model precision automatically.

Release notes must distinguish earlier validity fixes from the new normal
precision contract and its explicit argument/default changes. Publish runtime
limitations and migration examples with the corresponding behavior. Before
cutover, the public legacy path remains available during development. After
cutover, comparison or rollback can use a retained earlier artifact or release;
the new exporter does not switch individual computations back to legacy rules.
No stage is complete merely because a branch builds or a CI run has started.
