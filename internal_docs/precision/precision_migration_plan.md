# Computation Precision Migration Plan

Status: proposed migration with bounded local implementation, 2026-10-18.
PR #275 is merged. Private source policy now preserves weak inputs, selected
function/branch calculations, call-time x64 and matrix-precision settings, and
isolation from cached legacy/ordinary JAX traces. Optimizer selection follows
the policy, and captured-state keys retain weak typing. Source matrix and wider
normalization samples export end to end. Durable regressions are in the normal
test suite. Known runtime-rounding failures are rejected by a separate sample
acceptance gate; automatic repair remains open. No public source mode is available.

For the transition from an export user's perspective, read the
[user view](precision_migration_plan_user_view.md). All companion documents are
listed in the [precision overview](README.md).

The migration retains the existing public precision rules while validating an
export path for the source numerical contract. The [experiments](poc/README.md)
make the behavioral differences reviewable; their passing tests also include
recognized numerical failures. Each stage therefore needs explicit acceptance
evidence beyond the experiment's test result. No release version is assigned.

Read the [current-state analysis](precision_current_state.md) for the existing
behavior and the [target-state design](precision_target_state.md) for the proposed
contract. The implementation baseline is `fae17822`, which merges PR #275 on
top of PR #274's native lowering architecture. The fix uses explicit operand
adaptations in selected lowerings; it does not introduce the target precision
mode or a general runtime capability engine. The newer
[private integration](poc/source_integration_findings.md) is separate local
work beyond that baseline, not a merged or released public capability.

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

This plan adds no automatic numerical FP32-to-FP64 transformation. It also does
not initially change `enable_double_precision`. Its legacy behavior includes
temporarily disabling JAX x64 when false, selecting defaults for shape-only
inputs, and applying the existing export precision rules. Compatibility with
those rules is distinct from the proposed source-dtype contract.

## Current gates and order of work

| Stage | Evidence available | Remaining gate |
| --- | --- | --- |
| 1. Inventory | Public-call comparison and numerical counterexamples cover selected paths on both stacks. | Complete the operation and configuration inventory; retain known failures separately from compatibility expectations. |
| 2. Legacy repair | PR #275 is merged and its selected consumers are covered. | Audit remaining consumer gaps without reinterpreting the legacy flag. |
| 3. Internal source path | Weak input metadata, bounded preferred-FP32 matrix results, and model-chosen FP64 normalization work have source-export evidence. Unvalidated algebraic fusions are disabled under SOURCE. | Define the admitted numerical subset and target exclusions; broader operators/algorithms and automatic FP16 rounding repair remain open. |
| 4. Scopes and artifacts | Selected functions/branches, call-time x64 and matrix-precision settings, and source/legacy cache isolation have durable regressions in the normal CI collection. | Validate every scope combination and public boundary option admitted by the initial source interface; defer others explicitly. |
| 5. Public transition | The UX candidate distinguishes omitted and explicit legacy options and rejects conflicts. | Finalize and publish an interface only after the supported subset meets its earlier gates. |

Review exposed two function-context defects and one arithmetic-fusion defect.
They are repaired privately and have regressions; see the
[review findings](poc/source_integration_findings.md#review-findings).
The subsequent [Astra Ultra review](poc/source_integration_findings.md#follow-up-review-on-2026-10-18)
found two further defects at optimizer and captured-state cache boundaries.
6.1 Sol Ultra implemented their bounded repairs: explicit optimizer eligibility
and SOURCE keys that include captured promotion metadata. Independent review
accepts those repairs; numerical acceptance remains a separate gate.
Experiments may inform later stages before an earlier stage is complete;
public availability still requires the acceptance path for its declared subset.

## Next implementation work

The immediate review follow-ups have reached this checkpoint:

| Work package | Implemented result | Remaining limit |
| --- | --- | --- |
| Preserve function context | Capture and replay call-time x64 and default matrix precision; specialize functions accordingly. Separate ordinary JAX, legacy, and source trace/JIT caches. | Other configuration settings and concurrent export are not generally validated. |
| Preserve captured-state semantics | SOURCE specialization distinguishes weak/strong JAX arrays in supported instance state and nested static arguments while reusing equivalent states. | Other captured metadata and unsupported object forms still require an audit. |
| Apply policy during optimization | Explicit registry eligibility excludes four unaudited arithmetic/reduction rewrites under SOURCE, including an enabled legacy reciprocal rewrite. | Eligible structural passes do not certify runtime arithmetic; further numerical implementations require evidence. |
| Put regressions in CI | Durable input, context, scope, numerical, rejection, and isolation tests live under `tests/extra_tests/converter`; PoC modules reuse private test helpers. | Supplemental diagnostic demos remain an explicit internal invocation. |
| Preserve preferred FP32 matrix result | Local FP32 operand casts and FP32 MatMul preserve the original FP16 inputs for the admitted rank-2/rank-3 default-precision cases. | Other shapes, types, precision algorithms, and deployment costs need separate evidence. Public legacy behavior is unchanged. |
| Export wider normalization | The actual JAX callable passes inline/function/conditional serialization and CPU checks, including exact variance `8388609`. | This is model-chosen FP64 work on selected samples, not general normalization support or adoption of a legacy accuracy policy. |
| Enforce sample acceptance | The private CPU gate rejects failed and unverified samples, including the original FP16 square/sum and optimized FP32 reciprocal cases; load/run errors remain distinct. | The gate checks supplied samples. There is no automatic rewrite, model-wide capability database, or public target interface. |

The next bounded implementation should connect this private foundation to the
existing export boundary while retaining explicit scope limits:

1. **Record and enforce the initial subset.** List operations/parameters, dtypes,
   shapes, scope combinations, tracing settings, opsets, and claimed execution
   conditions. The current allowlist is an exploration boundary, not proof for
   arbitrary compositions. Keep known failing FP16 compositions and the
   optimized FP32 reciprocal sample excluded from acceptance under their
   recorded exact-sample requirements. A sample-only guard must not become a
   claim that every input or every accepted graph was validated.
2. **Exercise the shared export boundary privately.** Integrate source policy
   with input/output names, layout bridges, parameter materialization, and the
   return formats selected for the initial subset. Test the actual common path
   beside legacy, including failure restoration and saved artifacts. Deferred
   boundary options and scopes must fail explicitly. Avoid a second independently
   maintained product export pipeline.
3. **Connect validation evidence to the proposed caller experience.** Keep
   successful artifact creation separate from target execution and numerical
   acceptance. Define how an application supplies or obtains the required
   evidence and how known failures block a target-ready claim. The existing
   CPU sample helper is a test tool, not the final deployment API.

Public selection follows these gates. Loops, `scan`, JIT regions, and unused
export options may remain deferred if the initial interface enforces those
boundaries. Full support for every repository operation and return option is
not a prerequisite for a deliberately bounded first release. Broader numerical
work can proceed independently, with no implicit changes to model precision.

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

The evidence is bounded: explicitly typed static inputs, a small primitive
allowlist, and opsets 23 or 26. Some CPU samples agree; a new FP16 composition
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
function bodies before lowering. Loops, `scan`, JIT regions, symbolic shapes,
effects, and other unaudited operations remain outside the experiment, without
implying that ONNX cannot support them. This advances Stage 3; it does not
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
aliases, and formal-input passthrough. Success establishes these cases only; loop,
scan, JIT, and the remaining public boundary handling require their own work.
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
public-option coverage.

## Stage 5 Choose and document the public transition

Status: private UX exercised; no public selection finalized.

Prototype calling conventions and diagnostics privately during Stage 3, including
omitted versus explicitly supplied legacy options. Use those experiments to
check whether users can predict input, traced, and exported output types before
committing to an interface.

Only after the previous gates pass, finalize and publish how callers opt into
the source-dtype contract. A distinct, explicit export-mode selection is a
possible transition; this plan reserves no parameter name. Keep the existing
boolean's behavior available during the transition and define incompatible
option combinations as errors before tracing.

The private candidate tests this compatibility table:

| Selection | Required transition behavior |
| --- | --- |
| Legacy argument omitted | Keep the existing default, including temporary x64 disabling and shape-only FP32 input. |
| Explicit legacy `False` | Preserve the same current behavior while recording that the choice was explicit. |
| Explicit legacy `True` | Preserve the x64 override, shape-only FP64 input default, and legacy lowering/promotion rules. |
| Explicit source selection | Use caller JAX configuration, proposed shape-only FP32 input default, and preserved typed inputs; suppress legacy graph widening. |
| Source selection plus either explicit legacy Boolean | Reject before tracing, explaining which argument to omit. |

The experiment uses `precision="source"` only as a private spelling. Its
successful calls validate the comparison harness and bounded examples, not a
general public mode. Keep the proposed FP32 input fallback distinct from the
still-open selection syntax and release decision.

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

Design argument handling so omission and an explicitly supplied `False` remain
distinguishable when needed. The current default cannot be silently repurposed
to mean both legacy x64 disabling and source-context preservation. Any later
default switch or removal requires a deliberate breaking-change decision,
deprecation guidance, and examples showing replacement calls.

Do not introduce a separate `explicit_float32` switch. If demand later justifies
rewriting hardcoded FP32 models, assess a separate graph-wide numerical
transformation with defined coverage and exclusions. It is not a prerequisite
for dtype-faithful export.

## Release validation and rollback

Use the following matrix to guide meaningful coverage rather than multiplying
every case indiscriminately:

| Dimension | Required evidence |
| --- | --- |
| Contract | Legacy and internal source-dtype paths; valid differences explained |
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

Release notes must distinguish a legacy validity fix from opt-in source-dtype
support and from any eventual breaking default change. Publish the supported
runtime limitations and migration examples with the corresponding behavior.
Until the transition is accepted, rollback means withdrawing the opt-in path
while retaining tested legacy behavior. After a default change, provide the
documented compatibility route while it remains supported. No stage is complete
merely because a branch builds or a CI run has started.
