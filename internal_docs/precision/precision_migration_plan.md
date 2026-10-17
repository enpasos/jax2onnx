# Computation Precision Migration Plan

Status: proposed migration, 2026-10-17. The bounded legacy repair in PR #275 is
merged; Stage 3 has begun with an internal integration proof of concept. The
public source-dtype export contract remains proposed.

For the transition from an export user's perspective, read the
[user view](precision_migration_plan_user_view.md). All companion documents are
listed in the [precision overview](README.md).

The migration retains the existing public precision rules while validating a
source-dtype-based export path. A bounded legacy repair has landed, and the
[internal experiment](poc/README.md) now exercises a small subset through the
existing conversion pipeline. Each stage has acceptance criteria; no release
version is assigned.

Read the [current-state analysis](precision_current_state.md) for the existing
behavior and the [target-state design](precision_target_state.md) for the proposed
contract. The implementation baseline is `fae17822`, which merges PR #275 on
top of PR #274's native lowering architecture. The fix uses explicit operand
adaptations in selected lowerings; it does not introduce the target precision
mode or a general runtime capability engine.

## Migration objective and boundaries

The target is to lower the typed JAX graph produced under a documented tracing
configuration. Its dtypes, numeric casts, bitcasts, and boundary signatures are
authoritative. Necessary ONNX adaptations must preserve those semantics and
follow each operator's operand roles. This promises dtype fidelity, not bitwise
agreement between JAX and an ONNX runtime.

Support is finite and must be documented for the operation, dtype, shape, opset,
and deployment conditions being claimed. Every stage separates four checks:
ONNX schema validity; preservation of source semantics, including rounding and
accumulation; runtime kernel availability under the specified version, provider,
hardware, and settings; and numerical acceptance. Record each check as checked,
known unsupported, or unverified, with its evidence. One passing check cannot
stand in for the others.

The migration must preserve the model's chosen computation, accumulation, and
output types, together with deliberate conversions in each region. A model may
remain entirely in a lower precision without any FP32 stage. Representative
cases also include FP16/BF16 bulk computation with selected FP32 stability
regions and a deliberate return to the lower dtype, or scientific and physics
computation requiring FP64. These examples do not exhaust the supported policies.

Choose those policies in model configuration before tracing. As a local example,
when a particular region needs at least FP32, floating inputs can use
`jnp.promote_types(x.dtype, jnp.float32)`:

| Input dtype | Computation dtype for at least FP32 |
| --- | --- |
| FP16, BF16, FP32 | FP32 |
| FP64 | FP64 |

This does not require FP64 inputs for a low-precision model. Request FP64 inputs
where the model needs them, with JAX x64 enabled for FP64 computation. An exact
FP32 conversion still means FP32 rounding. Change it to a minimum only when that
is the intended behavior and the caller controls the model. Preserving source
types cannot reveal an unexpressed minimum requirement or remove hardcoded
third-party constraints.
Casting existing weights to a wider dtype cannot recover lost information.

This plan adds no automatic numerical FP32-to-FP64 transformation. It also does
not initially change `enable_double_precision`. Its legacy behavior includes
temporarily disabling JAX x64 when false, selecting defaults for shape-only
inputs, and applying the existing export precision rules. Compatibility with
those rules is distinct from the proposed source-dtype contract.

## Stage 1 Inventory and freeze observable behavior

Map every precision decision from public input normalization through tracing,
constant binding, plugin emission, child scopes, optimization, and export
post-processing. Record which decisions change the traced program and which
rewrite its lowered representation. Include explicitly typed inputs, weak
scalars, closed-over constants, constructors, complex representations, and
normalization plugins.

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
`lookup_value_for_var` roles. Final outputs require existing bindings; allocation
can replace a produced value with an unproduced placeholder. General lookup must
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

Status: begun, with the [integration experiment](poc/README.md).
[source_export.py](poc/source_export.py) reuses the real tracing helper, primitive
dispatch and plugins, optimizer, finalization, post-processing, and serialization.
A local context subclass preserves the existing writable-output allocation
rules and assigns those placeholders their traced dtypes. Non-promoting input
and constant bindings and post-processing suppress legacy widening; the caller's
active JAX x64 setting is left unchanged. Production code and the public entry
point are unchanged.

The [demo](poc/demo.py) compares source and exported types and results on chosen
samples. For example, exact FP32 arithmetic after FP64 input `1e8` returns `0`
in FP32, while the merged legacy double export returns `1` in FP64. Other cases
exercise selective FP32 work in FP16 computation, deliberate output conversions,
mixed FP64/FP32 outputs, wider reduction work, and a shared value used by a
same-width bitcast. These demonstrate numerical behavior as well as type labels.

The evidence is bounded: explicitly typed static inputs, a small primitive
allowlist, and opsets 23 or 26. The README records CPU execution settings and
sample agreement; BF16 has structural evidence only. Opset-23 bitcasts and
different-width bitcasts are rejected. Functions, control flow, symbolic shapes,
and other unaudited operations are outside the experiment, without implying
that ONNX cannot support them. This begins Stage 3; it does not complete the
stage, establish general numerical equivalence, or make a public mode available.

Next, replace the local allocation override with an explicit internal conversion
configuration that separates tracing context from export precision policy and
can be carried through nested scopes. The experiment's local use of the legacy
flag `False` only suppresses promotion; it is not the proposed public contract.
Keep the source-dtype path private until its
complete pipeline is tested. Normal lowering on that path uses traced aval
dtypes; it must bypass legacy promotion rather than inherit whichever behavior
the old boolean happens to select. This includes input/value allocation,
constant payloads, and late post-processing; disabling only one promotion site
does not establish the target contract.

The target path uses the caller's effective JAX configuration. Enabling x64
affects defaults as well as FP64 availability, so conversion must not silently
change it. Explicit FP64 requests under disabled x64 need actionable errors
rather than silent truncation. Shape-only inputs keep a documented FP32 default;
this input fallback does not establish a minimum computation dtype. FP64 needs
typed specifications. For scoped overrides, including the legacy
path, restore prior state after success, exceptions, and nested exports.
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

With no declared runtime, a standards-valid export may proceed without a runtime
guarantee. For a declared target, a known unsupported implementation requires a
validated alternative or an error. Unverified support must remain visibly
unverified; a request for deployment readiness requires the missing validation
rather than an unsupported success claim.

Runtime sessions remain separately configured by the caller. Acceptance must
compare declared requirements with the runtime/provider, hardware, math modes,
and optimization settings actually used. A declaration is not evidence that
export configured the session or that a requested provider executed the model.

Exit criterion: the documented support subset covers the representative precision
policies and passes the four checks for its claimed targets. Cases outside that
subset produce the appropriate explicit failure or unverified status, without
changing source precision to force success.

## Stage 4 Complete scopes and finalization

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

Exit criterion: the entire path, including nested scopes and all return modes,
satisfies the same contract. Known infeasible target cases do not return a
successful artifact; unverified deployment support is not reported as validated.

## Stage 5 Choose and document the public transition

Only after the previous gates pass, decide how callers opt into the source-dtype
contract. A distinct, explicit export-mode selection is a possible transition;
this plan reserves no parameter name. Keep the existing boolean's behavior
available during the transition and define incompatible option combinations as
errors before tracing.

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
| Inputs and tracing | Typed FP16/BF16/FP32/FP64 where supported, shape-only defaults, both ambient x64 states |
| Workloads | Lower precision without FP32 promotion; selective FP32 with deliberate cast-back; scientific FP64 examples |
| Semantics | Model-chosen computation/accumulation/output types; local exact vs minimum FP32 example; rounding-sensitive casts and constants |
| Operator roles | Floating data, integer indices, boolean conditions, mixed-result operators |
| Representation | Bitcast widths and shapes; structural validity and runtime support reported separately |
| Boundaries | Direct outputs, layout bridges, functions, branches, loops, scan |
| Compatibility | Both repository-supported JAX stacks through the compatibility layer |
| ONNX limits | Supported opsets and schema/type restrictions; valid decompositions and explicit upgrade requirements |
| Deployment target | Runtime/version, provider, hardware, optimization and math settings including TF32 where relevant |
| Evidence | Checked, known unsupported, and unverified for each of the four checks |
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
