# Source Precision Implementation Work Plan

Status: accepted implementation contract, 2026-10-18. Completion means consistent
source-precision behavior across the existing functionality, with genuine
technical limits stated explicitly. The private foundation, shared converter,
and CPU evidence module are committed and reviewed. Full coverage remains open.
The bounded private facade, canonical initializer repair, and deterministic
attention fixtures are reviewed and committed; inventory/reference infrastructure
is committed. A fixed
source-scenario plan, logical/physical boundary checks, exact discrete sample
comparison, and finite noncomplex loop repairs have passed independent review;
the repairs are committed and that checkpoint's integrated run passes all
5,363 tests. The scenario plan is committed at `158ce625`.
Further promotion-context, complex-loop and strict-finalization repairs are
committed at `ea6a8da7`, `8ba3c681` and `f1ee71b3`. B3 diagnostics, the reviewed
metadata-ownership repair and inventory additions are committed at `3dbec1b8`.
All four commits passed actual repository hooks. C4a recording/web packaging is
committed at `8a82d510`; D1a array-promotion repairs and the reviewed manual
inventory refresh are committed at `223f24bf`, also with independent review and
actual hooks passing. C4c scalar-BOOL binding is committed at `4c1c57fe`; D1b
native sequence binding is committed at `05037961`. Both passed independent
review and actual hooks. The shared Dropout optimizer repair is committed at
`c528a894`; D2a ordinary FP32/FP64 matrices are committed at `2011b140`, also with
independent review and actual hooks passing. That complete primary-stack
run passed **6,183 tests**. Native add/min/max binding and the separate Mul-None
diagnostic dependency are committed at `258e90d6`; the shared observer/naming
repair is committed at `f02c0b4c` and C4b static layout at `7609fc2e`. All passed
independent review and actual configured hooks. After ten regressions in the
initial full run, the corrected complete primary-stack run passes **6,351 tests**
(13,110 warnings, 466.07 seconds).
The [layout/binary findings](poc/layout_and_native_binary_findings.md) retain
those distinctions. Shared-B packet `80cb68b2` has 697 owner and 241 independent
focused passes per stack, including 40 new cases. Its complete primary-stack run
passes **6,391 tests** (13,110 warnings, 444.36 seconds), and actual configured
hooks pass. The [new findings](poc/symbolic_batch_and_minmax_findings.md)
record the bounded implementation, completed Min/Max feasibility and the next
accepted artifact-only corpus tooling plan. An earlier failed preflight exposed raw metadata
mutation, now repaired; a later preflight was deliberately interrupted after
typing exposed defects.
Interrupted runs are not counted as passing.
The first 36-cell diagnostic batch has run on both stacks, with every
lowering/runtime/numerical gate still unverified. Broader source-corpus execution
and migration remain open.
The completed normal `to_onnx` call will have one source-preserving contract.
The bounded source path remains internal during migration. The committed public
facade still has legacy behavior; no release version is assigned.

The [target contract](precision_target_state.md) defines the numerical semantics.
The [migration plan](precision_migration_plan.md) retains implementation history,
and its [user view](precision_migration_plan_user_view.md) explains adoption.
The bounded `source-static-v1` work is phase A of this plan. Completing that
phase does not complete the migration or its persistent goal.

## Completion contract

Overall completion requires all of the following:

1. Normal `to_onnx(...)` preserves the source precision contract throughout the existing trace, lowering,
   optimization, finalization, naming, and serialization pipeline. Tests and
   examples call that pipeline rather than maintaining another exporter. There
   is no permanent public precision/profile selection and no primitive-specific
   fallback to legacy precision rules.
2. Every existing registered primitive/framework interception, metadata testcase,
   supported scope, and export option is accounted for in a reproducible baseline
   inventory. Source behavior is implemented wherever the requested contract can
   be represented. Typed operations, precision parameters, and nested bodies are
   checked; a primitive-name allowlist is not the completion criterion.
3. Input properties, constants, deliberate conversions, precision parameters,
   and mixed results retain the traced source contract. Local implementation
   adaptations preserve canonical values, other consumers, and direct outputs.
4. Supported caller and model-local JAX settings, weak typing, captured-state
   specialization, and source/legacy trace-cache isolation survive the shared
   path. Exporter-managed state is restored on success and failure.
5. Every source artifact carries a versioned export report that survives
   serialization. Final boundary dtypes and ONNX representation are checked.
   Runtime execution and numerical acceptance remain unverified at export.
6. A public CPU sample-validation helper produces evidence tied to the exact
   artifact, samples, criteria, and execution settings. Its acceptance guard
   rejects failed, unverified, or mismatched evidence.
7. The inventory has no unresolved migration gaps, unaudited cases, temporary
   profile exclusions, or unexplained source-test skips. Remaining limitations
   identify a concrete representation, tracing, or execution constraint and its
   evidence; a missing implementation is not a technical impossibility.
8. Public API documentation, limitations, migration examples, changelog, complete
   source/legacy regression checks on both supported stacks, typing, lint, actual
   pre-commit hooks, and strict documentation checks pass. Independent review
   accepts the complete inventory and representative end-to-end results.
9. Important accepted checkpoints are committed locally with their evidence.
   Completion does not imply a push, merge, publication, or deployment.
10. Existing workflows with fixed, externally supplied JAX sources remain
    available without editing their implementation. Demonstrate external
    configuration/preparation and caller wrappers for static configuration,
    supported runtime flags and previous FP64 use cases. A static binding does
    not replace a working dynamic control. A strategy requiring source rewrites
    leaves a migration gap until an appropriate supported adapter exists.

A private prototype, bounded-profile release, or plan update cannot complete
this contract. Existing loops/JIT, symbolic shapes, layout bridges, parameters,
framework plugins, and serialization modes are part of the migration. New
functionality beyond the existing baseline and a generic runtime capability
database are outside it. Runtime-specific failures require a validated permitted
alternative or a precise failed-target diagnostic; they do not require a claim
of bitwise agreement on every provider or input.
Record preexisting negative or unsupported contracts separately from positive
conversion capabilities. Preserving the baseline does not require turning every
previously rejected operation into new functionality; an existing implementation
gap still must not be described as an ONNX impossibility.

## Coverage baseline and acceptance accounting

Freeze the initial inventory against `ccc861b0`, then record any intentional
baseline additions during implementation. Reconcile live `PLUGIN_REGISTRY` and
`EXAMPLE_REGISTRY` metadata with statically discovered plugin registrations and
the generated `tests/primitives` and `tests/examples` collections. Missing
optional dependencies or unprepared MaxText/MaxDiffusion checkouts must not
silently remove entries from the denominator. Use the existing preparation
wrapper when regenerating coverage documentation.

An inventory row identifies a primitive or framework entry, metadata testcase,
source module, dtype/shape/precision-parameter contract, opset, test selector,
and applicable scope or facade option. Record the existing baseline outcome and
the source migration outcome separately. Count test variants and registered
entries separately; many passing variants do not cover a missing operator.
Freeze actual contract fields, including original input/output types and shapes,
factory configuration, supplied-value fingerprints, and numerical criteria;
unchanged names or presence bits cannot certify an unchanged capability.
Keep original regression identities separate from migration tests added after
the baseline revision. Opaque recipes remain visible and require review.

The initial live inventory finds 605 registry names, including 104 without
metadata; 501 metadata entries and 110 example entries produce 1,802 raw
testcase rows and 1,768 unique `(context, component, testcase)` display names.
Legacy parameter expansion produces 3,195 raw variants and 3,119 unique named IDs.
Preserve all 1,802 rows using registration/example key, source module, and
testcase ordinal. Name collisions are not proven aliases: some distinct rows
currently overwrite the same generated test name. Deduplicate only after
proving equivalence and retaining every original provenance record. There are 310 symbolic
rows, 22 using `input_params`, seven using NCHW options, 41 with nondefault
normalization, and 29 skipping numerical validation. Recorded opsets span
9, 16, 18, 19, 21, 22, 23, 24, and 27; phase A's 23/26 restriction is not the
full migration boundary. These are inventory counts, not source-pass counts.
Optional MaxText/MaxDiffusion registrations are absent from this import
environment and require separate prepared discovery before the denominator is
closed. B1's committed inventory retains those unresolved discovery entries.

| Migration classification | Completion meaning |
| --- | --- |
| Source implementation validated | Tracing/representation/lowering checks pass with referenced tests; runtime/numerical evidence retains its separate scope. |
| Genuine representation or tracing limit | A specific unsupported semantic requirement is identified, with a schema/configuration explanation or minimal reproduction and assessment of available standard alternatives. |
| Known runtime limitation | The source artifact is valid, but the named runtime/version/provider/settings fail loading or the stated numerical criterion. Other targets remain separately assessed. |
| Runtime unverified | No execution evidence for that target. This does not prevent a valid source export and does not establish deployment readiness. |
| Implementation gap or unaudited case | Work remains. A temporary profile exclusion, missing test, skipped collection, or unfinished lowering cannot close the row. |

The final gate requires zero unresolved rows in the last category. It does not
require a Cartesian product of every dtype, shape, opset, and runtime. Cover the
existing tested contracts, precision-sensitive variants exposed by the audit,
and representative compositions. Integer/Boolean data, indices and shape values,
and the existing complex representations are part of this denominator.

Existing metadata identities, deterministic inputs, callable factories, and
structural assertions are reusable. Their legacy precision oracles are not
automatically reusable: `tests/t_generator.py` currently widens floating inputs
and expected dtypes for its double variants, and `allclose` coerces comparison
types. A source test lane must prepare model/input dtypes and JAX settings
explicitly, retain deliberate casts and weak metadata, compare strict output
types/shapes, and derive source references without legacy widening. Existing
fixed numerical criteria remain explicit; skipped numerical checks count as
unverified evidence, not a successful comparison.

Keep historical legacy fixtures only for comparison. Moving an existing
positive FP64 case to a private legacy suite does not preserve its capability
in the new exporter. Add its source counterpart with explicit JAX/model/input
configuration, and explain any deliberately changed arithmetic.

## Phase A: source-static-v1 integration checkpoint

This profile bounds the first integration checkpoint, not the overall goal.
Its identifier records intermediate support scope; it is not another caller option.
Eligibility depends on typed operations and attributes, rather than primitive
names alone. Static graph validity and audited lowering rules do not establish
numerical acceptance for every composition or execution target.

| Dimension | Admitted initial scope |
| --- | --- |
| Inputs | Static shape tuples/lists, `jax.ShapeDtypeStruct`, and concrete NumPy/JAX arrays. Shape-only inputs default to FP32; typed inputs retain dtype and `weak_type`. Other array-like forms fail before tracing. |
| Input dtypes | FP16, BF16, FP32, FP64. Explicit FP64 inputs with JAX x64 disabled fail before tracing. |
| Floating operations | Audited add, subtract, multiply, negate, divide, square root, broadcast, numeric conversion, and reduce-sum contracts; floating comparisons and the selector conversion required by `cond`. |
| Matrix operation | Default-precision FP16 operands with an explicitly preferred FP32 result, positive static rank-2/rank-3 shapes, matching batch axis 0 for rank 3, and contracting axes `(-1, -2)`. Other dtypes, axes, shapes, or precision algorithms fail explicitly. |
| Bitcast | FP32 to UINT32 at opset 26, with equal word width and preserved shape. Broader bitcast support requires a separate audit. |
| Scopes | Inline computation, audited `@onnx_function` specializations and captured/static state, and two-branch `cond`. Apply the same profile to nested bodies. |
| Outputs | Flattened structured results in existing output order, including mixed admitted floating types, Boolean comparison results, and the admitted UINT32 bitcast result. Integer arithmetic is excluded. |
| ONNX opsets | 23 and 26. The caller selects any required newer opset explicitly. |
| Export boundary | Initially `proto`, `ir`, and standard `file` returns, model name and custom input/output names; C4a now adds private primitive recording and web-file packaging through the same owners. Browser execution is a separate gate. |
| JAX configuration | Caller x64 and default matrix precision, including supported model-local scopes captured by named-function tracing. Explicit unsupported matrix algorithms fail. |

During phase A, the source path rejects symbolic dimensions, effects, loops/scan/JIT regions,
unlisted operations or precision parameters, nonempty `input_params`, nonempty
NCHW input/output options, and nondefault normalization
selection. Empty/omitted parameter and layout options retain their ordinary
meaning. The initial recording/web guards have been removed in reviewed C4a.
Remaining exclusions are temporary migration gaps unless the inventory records
a genuine technical limit. Existing captured-state forms and supported
configuration behavior require audit in later phases. New concurrent-export
guarantees are not required beyond the existing functionality.

This table records the initial phase-A scope. The later shared-B packet keeps
the persisted identifier and report schema while adding one audited equal-shape
Add/named-composition contract; its explicit gates below do not retroactively
broaden the historical phase-A evidence.

Retain appropriate NaN, infinity, and signed-zero coverage. BF16 graph validity
does not establish an available CPU kernel. No operation is silently widened
to a global minimum precision. The existing matrix adapter widens its own
operands under its documented contract; its memory and performance costs are
separate from correctness evidence.

## One public contract and explicit transition

The final normal call is `to_onnx(fn, inputs=..., ...)`. JAX model/configuration
and the export's tracing inputs determine the computation; every lowering and
boundary preserves it. No public `precision="source"`, `precision="legacy"`,
or subset selector is introduced. During development the source policy and
legacy reference route remain private. The current public behavior changes only
at the completed, reviewed transition in phase F.

Retain an omission sentinel temporarily for the old `enable_double_precision`
argument solely to produce a useful migration error. Neither explicit Boolean
continues to select a second numerical policy in the completed interface.

| Call at the completed transition | Required behavior |
| --- | --- |
| Old precision argument omitted | Preserve caller JAX configuration, input properties, and the traced source contract throughout; shape-only inputs default to FP32. |
| Explicit `enable_double_precision=False` | Fail before tracing with instructions to remove the flag and choose JAX/model/input dtypes explicitly. Export no longer disables caller x64. |
| Explicit `enable_double_precision=True` | Fail before tracing with the FP64 migration recipe: configure JAX x64 before model/array creation, then request the required model/parameter/input computation types. |

This is an explicit default/API transition and needs release notes and runnable
migration examples. It is distinct from preserving existing export capabilities.
FP64 scientific computation, low-precision models, mixed regions, existing
operators, and export options remain required. Blanket legacy widening has no
universally equivalent source-preserving replacement: a hardcoded FP32 cast
continues to mean FP32. Computing in FP64 instead requires explicit model
configuration or preparation of a different JAX computation, with its own
reference. Do not hide that difference behind another exporter mode. An earlier
release may announce deprecation, but final completion cannot leave two public
precision contracts. A supported previous release remains a reproducible
historical reference, not a fallback inside the new exporter.

Existing `proto`, `ir`, and `file` return types remain intact. A versioned JSON
`PrecisionReport` is stored under model metadata key `jax2onnx.precision` and
obtained through `jax2onnx.precision.get_precision_report(model_or_path)`
reader. Version 1 records:

- the implemented support profile, exporter/JAX versions, opset, caller x64, and
  default matrix precision;
- supplied, normalized, traced, and exported input properties, plus traced and
  exported outputs; properties include name, shape, dtype, and weak typing;
- trace warnings, including model-internal FP64 requests truncated by JAX;
- selected implementation adaptations and their scope;
- representation and lowering checks, and separately unverified runtime and
  numerical checks, with category, status, and detail.

Check statuses are `validated`, `failed`, and `unverified`. An audited lowering
status describes conformity to the profile, not arbitrary numerical equivalence.
Verify boundary types after facade postprocessing and naming, and run the full
ONNX checker on the serialized equivalent even when returning IR. Metadata must
survive file save/reload.

The completed contract uses a general source profile rather than asking callers
to choose an operator subset. Evolve the report's profile identifier while
retaining readers for earlier `source-static-v1` artifacts. Complete existing
facade options and validation conveniences under the same source semantics;
`allclose` must use the same single source contract, not apply legacy x64/matrix
overrides or hide dtype mismatches. Its old flag needs the same explicit
migration handling. Browser validation distinguishes CPU/browser
agreement from a JAX source-reference check.

Structured failures identify option/input validation, JAX configuration or
tracing, profile exclusion, ONNX representation/lowering, runtime execution, or
numerical comparison. Include the available graph or function/branch location,
operation, requested dtypes/precision parameters, opset, failed requirement, and
supported next action. A profile exclusion is not a claim that ONNX cannot
represent the operation. Preserve trace warnings alongside observed types.

## CPU sample evidence boundary

The committed `jax2onnx.precision.validate_source_samples(...)` helper is separate
from export. Each `SourceSample` supplies concrete positional or named inputs,
expected outputs, and `SampleCriteria`. Exact, tolerant, and bitwise criteria
have distinct meanings; tolerant criteria require explicit `rtol` and `atol`.
Check output dtypes and shapes strictly. A tolerant final-output check cannot
certify an unobserved required rounding boundary.
Integer and Boolean outputs require exact values under every criterion;
floating tolerances cannot excuse a changed discrete result in a mixed tuple.

The initial target is the installed ONNX Runtime CPU provider, with checked
provider selection and one thread in each pool. Select `optimization="disabled"`
or `"all"` explicitly; `None` executes nothing and produces unverified evidence.
Record the actual runtime version, provider/options, architecture, optimization
level, and thread counts. Never change provider, opset, or model precision as an
implicit fallback.

References are supplied explicitly so validation cannot silently recompute JAX
under different configuration or change weak typing. Acceptance requires
nonempty samples. Keep representation failure, load/run failure, and numerical
mismatch distinct. Aggregate evidence retains failed samples; another passing
sample cannot overwrite them. Accepted results cover only the recorded samples
and execution target.

The `SourceValidationReport.require_sample_acceptance(model_or_path=None)` guard
checks the retained original artifact or an explicitly supplied artifact before
accepting the recorded samples. Bind evidence to artifact and sample digests,
references, comparison criteria, and actual execution settings. Snapshot mutable
sample data or detect changes; evidence cannot be reused to certify changed
samples, criteria, or settings.

Artifact identity includes tensor contents. Explicit local paths can resolve
external tensor data, then inline it before hashing and execution. Reject
absolute/traversing sidecar paths and symlinks. In-memory artifacts with
unresolved external data fail explicitly. An unresolved protobuf reference
alone cannot identify its tensor payload. Unsupported serialization or artifact
sizes produce a concrete limitation rather than a fallback identity.

Retain the FP16 square/sum, FP16 weak-bias/FP32 scaling, and optimized FP32
reciprocal witnesses through the common public path. Their observed numerical
failures are failed evidence under their recorded criteria, even when the model
loads. A newer runtime may pass a witness when it actually preserves the
expected result. Also retain accepted lower-precision, mixed matrix, and
scientific FP64 variance samples. Passing tests that recognize a known failure
do not establish deployment success.

## Milestones, ownership, and local commit ledger

Implementation and repairs use GPT-6.1 Sol Ultra. Planning and independent
acceptance reviews use GPT-6 Astra Ultra. The coordinator owns integration,
evidence recording, and local commits after each accepted gate. Documentation
planning work is separate from production-code ownership.

| Checkpoint | Scope and acceptance gate | Status | Local commit |
| --- | --- | --- | --- |
| Foundation | Private source contexts, selected scopes, precision/captured-state fixes, optimizer eligibility, and durable regressions; evidence remains bounded as recorded in the integration findings. | Committed | `63abe1c1` |
| M1: accepted execution plan | This finite contract, linked technical/user guidance, planned API and exclusions; documentation/link review and repository hooks. | Committed | `5b106e60` |
| M2: common private pipeline | Common converter body, SOURCE context/recursive validator, constant binding, strict output checks, and policy-aware x64 handling; remove manual test graph assembly. Focused tests on both stacks and Astra review pass. | Committed | `5a1858a3` |
| M3: enforced internal profile | Product-owned typed intermediate profile and structured contextual errors; validate shapes, parameters, value roles, and nested scopes. Positive and rejection tests pass on both stacks, with independent review. | Committed | `dab1bde4` |
| M4: CPU sample evidence | Artifact reader/identity, explicit samples/criteria, execution/numerical statuses, identity-aware acceptance guard, failure witnesses and accepted controls. Dedicated tests and API review pass. | Committed | `ccc861b0` |
| M5: private facade integration | After M2-M4 acceptance, exercise normal facade normalization, reports/warnings, names, and proto/IR/file serialization through one policy-aware facade body; full checker and save/reload evidence. Keep the bounded mode internal. | Committed | `3b3a48e0` |
| M6: phase A acceptance | Validate and review actual facade artifacts through the CPU sample API, complete integrated checks, record remaining inventory work, and commit the checkpoint. The overall goal stays active. | Bounded integration reviewed; integrated standard regression passed | `3b3a48e0` |
| B prerequisite: deterministic fixtures | Replace two unseeded attention input-fixture producers with isolated named-seed generators; preserve Gaussian domain, dtype/shape, causal masks, and numerical criteria without touching global RNG state. | Committed | `32554a9e` |
| B1/B2: baseline and source oracle | Preserve raw row provenance and typed contract fingerprints; prepare independent original-JAX references without legacy coercion; reject input canonicalization before model construction. | Committed | `206277ac` |
| B3: fixed initial scenario plan | Preserve original factory configuration and typed inputs in paired x64-off/on cells; freeze provenance, criteria status, and bounded plan/prepare/product stages without a broad corpus run. | Committed | `158ce625` |
| C1: canonical initializers | Carry SOURCE policy into builders; preserve typed payloads and exact duplicate identity, require typed floating coefficients, and validate actual function Constant payload/metadata on reuse. | Committed | `b2974c3d` |
| C2a: complex constant storage | Pack actual initializer and Constant tensor payloads together with metadata, preserving the selected component dtype and checking cache consistency before mutation. | Committed | `3e904b46` |
| Sample comparison: discrete outputs | Compare integer/Boolean values exactly even when floating outputs use tolerance; retain strict dtype/shape checks and artifact-bound acceptance. | Committed | `52389e29` |
| C2b: typed representation boundary | Encode logical complex values into actual typed real/imaginary payloads; check physical boundaries exactly, reject input canonicalization before tracing, and label private diagnostic artifacts as unaudited. | Committed | `530561d1` |
| C3a: finite noncomplex loops | Preserve original JAX fori tracing, typed while/scan carries and static extents; materialize correctly shaped zero-trip scanned outputs after body validation. | Committed | `1461e4b9` |
| C3b: promotion context | Replay local NumPy dtype/rank promotion, specialize function caches, and record actual caller settings in compatible optional report fields. | Independently accepted; integrated run and actual hooks pass | `ea6a8da7` |
| C3c: finite complex loop representations | Apply shared physical descriptors to scan/while formals, outputs, masks and stacked extents; retain exact canonical values and explicit runtime sample failures. | Independently accepted; integrated run and actual hooks pass | `8ba3c681` |
| Required SOURCE finalization | Stop on required shape, Function postprocessing or explicit attribute failures before returning a successful artifact/report. | Independently accepted; integrated run and actual hooks pass | `f1ee71b3` |
| B3b: artifact diagnostic runner | Bind immutable stage history to exact rows, settings, original traces, typed values and artifacts; inspect only explicitly known structural families. | Independently accepted; first 36 cells per stack measured; metadata repair, integrated run and actual hooks pass | `3dbec1b8` |
| C4a: recording and packaging | Reuse shared logging and serialization; preserve exact large root/nested payloads, names/reports and primary failures; remove real stale sidecars in web mode. | Independently accepted; both-stack focused checks, complete integrated run and actual hooks pass | `8a82d510` |
| D1a: array promotion | Preserve native `where` promotion and `full` weakness, immutable canonical operands and Boolean logical selection; retain diagnostic-only admission and failed target evidence. | Independently accepted; both-stack focused/generated checks, complete integrated run and actual hooks pass; reviewed manual inventory included | `223f24bf` |
| C4c: real scalar-BOOL parameters | Trace one prepared root with actual positional/named operands; share checked bindings across names/reports/boundaries and preserve framework controls/defaults/literals. | Private diagnostic packet independently accepted; both-stack focused/legacy checks, complete integrated run and actual hooks pass; all-22 model acceptance remains open | `4c1c57fe` |
| D1b: native sequence binding | Dispatch original `arange`/`linspace` arguments before normalization; preserve native typed arithmetic while retaining failed target criteria and dynamic compatibility gaps. | Independently accepted; both-stack focused/legacy checks, complete integrated run and actual hooks pass; reviewed manual inventory included | `05037961` |
| Shared Dropout optimizer repair | Omit only a proven-false optional training input; preserve shared/captured/observable values and existing traversal. | Independently accepted; both-stack focused/legacy checks, corrected witness provenance, complete integrated run and actual hooks pass | `c528a894` |
| D2a: ordinary matrices | Preserve same-dtype FP32/FP64 rank-2–4/default-precision matrices, actual batch/contracting order and native matmul binding; retain exact subnormal failures. | Private diagnostic packet independently accepted; both-stack focused/generated checks, complete integrated run and actual hooks pass; no matrix-family or B3 row closure | `2011b140` |
| Native add/min/max binding | Forward original SOURCE arguments through one private binder opt-in; preserve native promotion, weakness, errors and transforms. | Independently accepted; 366 focused/generated checks per stack, 62 new cases; integrated run and actual hooks pass; eight signed-zero target failures per stack retained | `258e90d6` |
| Diagnostic Mul-None dependency | Admit only the observed None spelling under the existing typed floating rule; retain native abstract checks. | Separately accepted; 84 checks per stack, 18 new safeguards; integrated run and actual hooks pass; no production/profile change | `258e90d6` |
| C4b static NCHW and shared observers | Compose root physical layout without changing canonical source values; protect observable transpose values and lexical naming. | 36 layout and 57 observer cases; ten failed-full regressions retained, final correction independently passes 238 cases per stack; fresh full run and actual hooks pass | Optimizer `f02c0b4c`; layout `7609fc2e` |
| Shared symbolic batch | Preserve equal `(B,K)` floating Add and named composition, exact symbolic boundaries and fresh sample bindings; one artifact at B=2/5. | 697 owner and 241 independent checks per stack, all 40 new cases; full run and actual hooks pass; all 310 raw symbolic rows remain open | `80cb68b2` |
| Next B3 revision supplement | Replay 13 exact unchanged rows/26 artifact-only cells per stack; preserve raw outcomes and four explicit SOURCE matrix counterparts; propose criteria separately. | Accepted tooling plan only; no run total, runtime release or criterion adoption | Planned |

M2 has one owner for the coupled converter body and shared test harness. M3
owns the source profile and its tests. M4 proceeds independently in
`jax2onnx/precision.py` and dedicated evidence tests; it must not introduce ONNX
protobuf imports into converter or plugin modules. Agree the plain report
schema before integration. M5 integrates the private facade path after those reviews.
Record actual commit IDs and evidence when gates complete; pending entries are
not completion claims.

## Full migration phases and review gates

Each phase is implemented in reviewable local commits paired with relevant
tests and documentation. GPT-6.1 Sol Ultra implements; GPT-6 Astra Ultra reviews
the acceptance evidence. Work may overlap where files and dependencies permit.

| Phase | Concrete work | Required exit evidence |
| --- | --- | --- |
| B: inventory and source test lane | Freeze/reconcile the baseline; add policy-aware source materialization, strict type/shape checks, independent JAX references, and separately counted numerical evidence to the metadata test infrastructure. | Every baseline entry has an owner/status/test selector; missing optional imports are visible; representative legacy/source cases prove the new oracle does not reproduce legacy coercion. |
| C: values, scopes, and export options | Audit canonical values/constants and floating/integer/Boolean/complex roles; migrate named functions, `cond`, while/fori/scan, JIT/pjit/custom-call bodies, symbolic dimensions, NCHW bridges, `input_params`, names, recording, and proto/IR/standard/web file paths. | Relevant existing framework/converter/control-flow/boundary regressions run under SOURCE; nested mixed-type state, captured flags, dynamic shapes and serialization preserve their source contract. Temporary exclusions are removed as each area passes. |
| D: numerical and framework coverage | Migrate remaining elementwise/transcendental, indexing/scatter, reduction/window, matrix/convolution, linear algebra, random, complex, and framework-specific lowerings. Audit high-level patches/abstract evaluation as well as emitted IR. Validate or disable numerical rewrites by source contract. | Every existing plugin/parameter family is implemented or has a reviewed genuine-limit record. Matrix operand/result/accumulation/algorithm requirements and normalization statistics/casts have independent adversarial evidence. |
| E: complete-model and target evidence | Run the reconciled metadata/examples corpus through SOURCE; cover representative existing Flax NNX/Linen, Equinox, attention/CNN, nested-control-flow, symbolic-shape, and scientific workloads. Exercise exact versus minimum rules, low-precision work, selective FP32 and scientific FP64. | Model-level artifact/report checks and applicable numerical criteria pass; schema/lowering defects, unavailable kernels, numerical failures and unverified targets remain distinct. No model is waived merely because its operators pass individually. |
| F: single-contract transition and final acceptance | Reconcile inventory counts and exceptions; make normal export and validation consistently source-preserving; implement actionable old-flag migration errors; complete API/coverage/limitations/changelog and both-stack regression checks; perform independent final review. | Zero unresolved migration/unaudited rows or unexplained skips; genuine limits have evidence and actionable diagnostics; no public partial-profile or legacy fallback remains; all required local checks pass at the final reviewed commit. Only this gate completes the goal. |

Start B immediately alongside phase A. Its first commits are the inventory and
the source test/reference lane. C can then proceed in separate commits for
canonical value roles, inline/named scopes, loops/scan, and facade/dynamic
boundaries. D can run in parallel by disjoint plugin groups after agreeing shared
dtype/adaptation helpers; shared helper changes have one owner. Framework
normalization and attention depend on the corresponding matrix/reduction work.
E starts as groups become available and finishes only after the complete
inventory is accounted for. F is an integrated gate, not a replacement for
focused acceptance at each checkpoint.

Use the existing tests as concrete coverage anchors: `tests/primitives`,
`tests/examples`, converter tests, `tests/extra_tests/loop`,
`tests/extra_tests/scan`, return-mode/layout/parameter tests, normalization
policy tests, and the source precision regressions. Record exact selectors and
case counts in the inventory as it is generated. Do not declare all variants
supported from a source-code search for a precision flag.

The following packets make the broad phases dispatchable. Each inventory row
records its concrete selectors and evidence; the table is not a substitute for
the full registered-case denominator.

| Packet | Implementation and regression anchors | Acceptance boundary |
| --- | --- | --- |
| B1: reproducible baseline | A compact tracked manifest plus `scripts/source_precision_inventory.py`; raw registry/example provenance and contract snapshots, name collisions, original versus added regression identities, and optional integration status. | Regeneration is deterministic; no row disappears because its display name collides or its optional environment is unavailable. Changes to types, shapes, supplied values, factory recipes, or comparison criteria are detected. |
| B2: independent source oracle | Raw metadata preparation under explicit caller x64, model/factory dtype, input properties, concrete symbolic sizes, and matrix precision; original JAX trace/reference; frozen per-case criteria. | No legacy input/output coercion. Diagnostic probes outside audited product admission remain `audit_pending` even when they execute successfully. |
| B3: measured source scenarios | Freeze raw factory/input recipes and both caller x64 contexts before execution; run explicit bounded stages and retain every unexecuted or unresolved cell. | Preparation, artifact creation, audited lowering, and numerical acceptance retain separate evidence. Missing comparison criteria prevent numerical acceptance. |
| C1: canonical values and builders | Reviewed repair for initializer payload/type policy, duplicate constants, child contexts, and function capture tests; actual inline/nested relu6 regression without broadening product admission. | SOURCE preserves supplied typed payloads, including an FP64 rounding witness, NaN bits and signed zero; policy survives child/function builders. Preserve existing legacy comparisons privately. |
| C2: logical/physical boundaries and value roles | One typed representation descriptor shared by core checks, facade reports and reference/feed adapters; ordinary integer/Boolean results and existing complex pairs; input canonicalization checks. | Identity mappings remain exact; complex64/complex128 map explicitly to FP32/FP64 with a trailing real/imaginary axis. Encode actual constant payloads as well as metadata. Explicit input requests that JAX would narrow fail before tracing. |
| C3a: noncomplex loop repair | Original JAX fori tracing, mixed-type nested-loop regression, typed while/scan carries, and static/zero-trip scanned results. | Original trace dtype, shape and weak typing agree before lowering; actual artifact samples retain discrete values and output extents. This finite packet does not establish complex-loop support. |
| C3b: local promotion context | Capture, specialize on, and replay actual dtype/rank-promotion settings in named functions; record caller settings without inventing values for older reports. | Local standard/allow scopes survive strict/raise callers; original negative behavior, cache isolation and state restoration remain intact on both stacks. |
| C3c: complex loop representations | Audit scan/while/fori complex formals, carries and scanned outputs against the shared logical/physical descriptor, including empty results and nested functions. | Physical channel axes and component dtypes survive every loop boundary with correct actual payloads. C2b and C3a passing separately cannot close this packet. |
| C4: remaining facade options | Dynamic batch/symbolic names, function call parameters, recording, NCHW models, normalization options, and web packaging. | Existing options preserve the same contract through logical/physical adapters and serialization; none falls back to legacy precision rules. |
| D/E: frameworks and complete models | Linen CNN/parameterized MLP; NNX GRU, dynamic NCHW attention/residual models and decoder; Equinox inference models; GPT/ViT examples; scientific mixed-FP32/FP64 variance. | Run actual model compositions with their original contracts and independent JAX references. Structural success and CPU/browser numerical evidence remain separate. |

B2 may use a recursive structural diagnostic validator in test tooling to
discover gaps outside the internal product profile. It cannot replace product
admission with a no-op or turn an unaudited operation into validated support.
Missing implementation and failed source preparation remain explicit pending
rows. Numerical skips retain their existing reason and an unverified outcome.
Browser comparisons use a JAX reference, not Python ONNX Runtime as an implicit
source oracle. Optional model configuration filters are recorded and reconciled
before final acceptance; unavailable local environments are not passing tests.

C2 derives the physical representation from the logical JAX aval and the chosen
adapter, never from a coincidental trailing dimension of size two. Check direct
passthrough, captured constants, mixed tuples, real/imaginary channels and signed
zero, and named functions against independent JAX references. Existing complex
constant metadata/payload inconsistencies must be repaired, not hidden by a
relaxed boundary checker. With x64 disabled, explicit int64, uint64 and complex128
requests require the same canonicalization check as explicit FP64 requests.
C3 audits concrete dtype-default and promotion settings as they affect existing
cases; it does not require a generic framework for every JAX configuration flag.
The recorded local dtype/rank-promotion failures are repaired in C3b: named
functions replay model-local `standard` or `allow` scopes beneath an outer
`strict` or `raise` setting and include both settings in specialization keys.
Other transformed scopes and dynamic loop contracts remain in the phase C
inventory beyond the finite reviewed packets.
C3b adds optional `caller_numpy_dtype_promotion` and
`caller_numpy_rank_promotion` report fields with actual caller values. Older
schema-1 artifacts read missing fields as `None` (not recorded), without inferred
defaults. Function specialization records and replays its actual local settings
separately from the top-level caller context.

The initial B3 grid retains all 1,802 rows as 3,604 provenance-distinct cells,
one for each caller x64 setting. Both cells retain the same typed input bytes
and original factory recipe; a requested model-dtype override is a separate,
explicit scenario. The 63 opaque recipes remain unresolved. Seventeen raw rows
have both general `rtol` and `atol`; the other 1,785 have unresolved criteria.
Missing, one-sided, or legacy-variant-only criteria are not converted into an
implicit exact requirement. Freeze reviewed family criteria and case exceptions
before numerical runs, preserving integer/Boolean exactness and original dtype
checks. Existing legacy tolerances are review evidence, not a coercing oracle.

Default B3 invocation only writes a plan. Preparation and product-export stages
require an explicit bounded selection; neither establishes corpus acceptance.
The first diagnostic batch is fixed at ordinal zero for 18 registered origins:
add, convert_element_type, sin, reshape, transpose, gather, reduce_sum,
dot_general, conv_general_dilated, jit, pjit, lax.fori_loop, while, scan, complex,
real, imag, and fft. Both x64 contexts produce 36 cells per stack; the other
3,568 remain unchanged and unexecuted, including 126 recipe-pending cells. The
[first measured batch](poc/source_corpus_findings.md) produced the same outcomes
on both stacks: 26 checked artifacts, four export implementation gaps, three
structural forms pending, two trace-lineage cases pending and one source x64
configuration conflict. Original references passed for 35 cells. All 36 keep
lowering, runtime and numerical acceptance unverified; zero ORT sessions ran.
History retains actual settings, environment, logical/physical bytes, original
trace fingerprints, lineage and artifact evidence. No criteria were invented.

Diagnostic lowering uses the shared facade and a real recursive validator.
Its fixed `source-diagnostic-v1` report keeps lowering `unverified` even when
representation checks and samples pass. The strict reader rejects a diagnostic
report claiming validated lowering. Audited product admission remains unchanged;
successful probes do not silently broaden `source-static-v1`.

The reviewed strict-finalization repair stops SOURCE on required shape and
Function postprocessing failures and on explicit attributes that cannot be
applied to surviving nodes. The checker-valid LeakyRelu default witness proves
why schema validity alone cannot certify the intended computation. Removed
nodes and overrides belonging to another graph remain legitimate skips; hit
counter bookkeeping stays best effort. Broader D/F admission still requires
the full lowering audit alongside C4 options and the wider D/E/F gates.

C4 parameter migration uses an explicit whole-model contract: `input_params`
declares real typed runtime arguments; model construction, closures and partial
application bind static configuration. Do not preserve the legacy inference of
dynamicness from ONNX Function usage. Trace the root once with the declared
runtime values and carry the actual formals through boundaries and reports.
Never add synthetic dynamic flags to a specialized root. Preserve all 22
existing metadata-model capabilities and the handwritten root-only static
cases with runnable migration recipes; changed argument syntax is documented
separately from a lost capability. Python branching on a declared runtime flag
requires JAX control flow through a supported framework or external caller
adapter; use a static binding only for an intended fixed setting. Defaults and literal
overrides stay local, unused inputs may be pruned, and both BOOL values must
match independent original-JAX references through Function/If/Loop scopes.
The first implemented packet covers scalar BOOLs through private diagnostics;
unimplemented parameter contracts stay open and cannot become a final public
partial profile.
The [fixed-source compatibility review](poc/fixed_source_compatibility_findings.md)
has classified all 22 baseline models and the static/manual witnesses: 16 rows
have live positive-rate dynamic controls, five GPT rows are flag-insensitive at
zero dropout, and one SpectralNorm row is static False. Serialized LEGACY checks
passed for all 44 models in total, 22 per stack, plus 20 manual checks in total,
ten per stack. This is a compatibility baseline, not SOURCE or stochastic
numerical acceptance. Reject a tracing strategy
that preserves their names but loses an existing dynamic control. External
configuration and wrappers may change export call syntax; modifying the supplied
implementation cannot be assumed. Previous fixed-source FP64 workflows also
require an explicit compatible preparation recipe and independent reference,
rather than an instruction to edit hardcoded casts. These decisions remain open
acceptance work; no new generic transformation API is assumed by this plan.
The user has been asked whether external preparation must also change hardcoded
FP32 calculations to FP64 in fixed sources. That scope question is pending;
continue independent scope, evidence and finalization work without assuming
the answer or weakening source preservation in the ordinary export call.


C4a is committed at `8a82d510`. Exactly three SOURCE guard removals enable
recording and web packaging through the existing logger and serializer. Exact
1.2 MB FP64 root and nested Function payloads retain rounding-sensitive values,
signed zero and NaN bits across IR/proto/file returns. Tests cover real external
data, same-path web sidecar cleanup, names/reports, off/on/off recording, a single
root flush through Function/If/Loop scopes and best-effort log failures that
preserve the primary result/error. Browser execution remains a separate gate.

C4c is committed at `4c1c57fe`. The coupled packet changes root tracing and
binding metadata in `conversion_api.py`, facade names/reports in
`user_interface.py`, SOURCE ownership in `ir_context.py` and `plugin_system.py`,
and the NNX/Equinox Dropout lowerers. Linen inherits the NNX lowerer. A protocol
declaration exposes the existing training-mode helper: seven production files
in total, six with behavioral changes, plus three test files. Existing Function
argument packing and ordinary If/Loop operand wiring are reused; no optimizer
traversal or report schema change was needed.

The complete unchanged callable is traced once with positional specs followed
by real typed scalar-BOOL named operands. Actual Boolean scalar declarations are
accepted without numeric coercion; other unimplemented named contracts fail
explicitly. Closures, construction and partial application hold intended static
settings. The private facade/core handshake requires explicit diagnostic
validation, including false-valued validator objects; direct core calls also
require explicit runtime-parameter permission. Audited product admission and
LEGACY behavior remain unchanged. Default/literal callee settings remain local,
regardless of whether their names match a declared runtime argument.

One immutable ordered binding tuple holds actual root vars, normalized specs,
positional indices/keyword keys and trace indices. Every genuine root formal is
retained initially, including unused flags, using internal `in_N` names through
optimization. The tuple resolves against finalized/cloned inputs with strict
arity, order and shared representation checks. Input-name declarations remain
positional-only. Final naming and output-alias separation receive the complete
input target map, including named ports, and apply one collision-checked rename.
Reports use that mapping for supplied, normalized, traced and exported properties,
including proto/file rechecks. Same-typed input/output reorder corruptions fail.

SOURCE skips late input-parameter materialization and promotion by name. Named
Function captures and Dropout resolve actual typed vars; a name-only flag request
fails rather than appending a port. SOURCE adds no unused Function sink. Actual
input retention is not liveness evidence. Check real call/formal bindings and
If/Loop dependencies through a positive-rate Dropout control to a model output.
The existing legacy promotion, sinks and pruning stay unchanged until phase F.

Preserve all 22 compatibility rows and their original factories/dtypes. The
sixteen positive-rate controls remain dynamic; five zero-rate GPT configurations
remain flag-insensitive; SpectralNorm's supported False case uses static partial
preparation. Equinox's missing-key False failure and original NNX/Linen state/key
behavior stay explicit. For genuine Python branching in an unchanged pure
source, test a concrete external `lax.cond` adapter that selects the complete
original computation with concrete branch flags. Do not build an automatic
combinatorial wrapper engine or freeze a working dynamic control. Stochastic
JAX/ONNX masks, sequences and state advancement are separate unresolved contracts.

Focused regressions cover root-only and mixed root/Function control, defaults,
literal overrides, unused inputs, zero-trip and nested loops, names/aliases and
failure restoration on both stacks. Original references run outside plugin
activation. The all-22 preparation/trace/export pass produced three checked
framework Dropout artifacts and 19 pending rows per stack, preserving the exact
missing audit/lowering gates. No ORT sessions ran; lowering, runtime and numerics
remain UNVERIFIED. Concrete `B=1` probes do not close symbolic-batch coverage.
Extend the contract to all remaining parameter types and complete all-22 model
acceptance before the final transition.

The separate shared optimizer defect is repaired at `c528a894`. The old
same-graph `Not(True)`/Dropout match could introduce an unowned `false_const`,
including in other uses of that matched value. The bounded repair changes only
the matched standard Dropout's optional training input to omission. It deletes
the Not only after all matched consumers are handled and no uses, captures or
graph outputs retain it. It introduces no replacement constant or global-use
rewrite. Standard-domain guards, same-graph proof, dynamic/overridable-input
exclusions and existing direct-False/nonrecursive behavior remain explicit.
C4c's builder-owned literal handling remains unchanged. Valid checker/runtime
witnesses and scope controls are recorded in the
[matrix and optimizer findings](poc/matrix_and_optimizer_findings.md).

D1a is committed at `223f24bf`: `where` uses native JAX operand-aware promotion,
`full` retains weak fills, and SOURCE consumer casts leave canonical constants
unchanged. Boolean selection uses `Not`/`And`/`And`/`Or` with actual pairwise
shapes and selected exact CPU results. Numeric selection continues to emit Where
with JAX-promoted operand types;
`where`/`full` still require diagnostics rather than audited product admission.
The existing trace-cache namespace separates native/LEGACY/SOURCE; no extra
cache mechanism was needed. The
[curated findings](poc/option_and_array_creation_findings.md) retain kernel failures
and the precise regression scope.

D1b is committed at `05037961`. SOURCE dispatches original `arange`/`linspace`
arguments to captured native JAX before normalization, preserving dtype/weakness,
omitted-step behavior, explicit conversions, appended endpoints and integer
flooring. Native positional/device forms and strict-promotion errors retain
their original meaning; LEGACY code and policy-scoped cache behavior are preserved. Existing dynamic `arange` capability
remains compatibility work even when native JAX rejects data-dependent bounds.
Selected FP16/FP32 native `linspace` traces and ONNX intermediate types match,
but exact JAX samples differ from ORT and ONNX ReferenceEvaluator on both stacks.
These remain failed exact criteria. Independent compiled-arithmetic review of
the selected witnesses observes JAX CPU reciprocal substitution, reassociation
and FP32 fused multiply-add relative to the preserved typed graph; it does not
explain every additional ORT FP16 discrepancy. The
[curated diagnosis](poc/option_and_array_creation_findings.md#native-sequences-and-retained-exact-failures)
keeps original dynamic-endpoint references unchanged. Do not call this an
ORT-only defect or ONNX impossibility, widen the computation, or loosen failed
criteria. BF16 Add loading/result-interface failures and the installed opset27
runtime limit stay separate from numerical comparisons. Ordinary cross-runtime
floating reproducibility and deliberately required rounding remain distinct
gates; the accepted binding packet does not close family-wide numerical work.

D2a is committed at `2011b140`. The diagnostic lowering covers same-type
FP32/FP64 operands and results, preferred type omitted or equal, default precision,
and equal positive static ranks 2–4 with one contracting/free axis per operand.
Checked permutations preserve the declared paired-batch order, using fresh typed
Transpose consumers where needed. Input/output representations and source result
shape are checked before arithmetic. Canonical values remain unchanged; no Gemm
zero bias, result cast or global widening is introduced. The earlier finite
FP16/preferred-FP32 route retains its local operand casts.

SOURCE `matmul` forwards original arguments to captured native JAX, retaining
promotion, weak types, strict errors and precision parameters. LEGACY and audited
product admission remain unchanged. Only unspecified `out_sharding=None` is
accepted; unknown semantic fields and non-None placement stay explicit exclusions.
Vector/broadcast/general contractions, other result types, algorithms and broader
wrappers remain work. Four exact subnormal target observations per stack remain
FAILED: native eager/JIT positive-zero bits 0 versus CPU ORT bits 2 for FP32/FP64
with optimization off/all. No criterion or reference is relaxed. A separate
recheck of the two original B3 dot cells passes schema but fails the unchanged
Gemm/zero-bias callback; structural-oracle migration remains open and no corpus
row closes. The [dated supplement](poc/source_corpus_findings.md#two-cell-matrix-supplement-on-2026-10-18)
preserves original histories and unresolved numerical criteria.

The committed C4b packet implements selected static logical rank-four
inputs/outputs with NCHW external layout through the shared representation
boundary, while the original JAX computation remains NHWC. Independent review
accepts the layout and corrected observer/naming dependency within its focused
scope; the fresh full run passes after the ten retained regressions. Real storage maps
`(N,H,W,C)` to `(N,C,H,W)`; complex storage maps `(N,H,W,C,2)` to `(N,C,H,W,2)`
without moving or inferring the real/imaginary axis. Ordinary real trailing-two
data remains real. Preserve canonical source/child values, dtype/weakness,
output aliases, custom names and final IR/proto/reloaded reports. Select only
positional inputs and flattened outputs; named scalar BOOL operands must not
shift layout indices. Use one explicit descriptor through the core/facade, with
no skipped boundary checks or primitive-admission expansion. Symbolic layouts,
normalization policy and raw-metadata layout replay require later packets; a
boundary implementation alone cannot establish all layout-dependent models.
The dependency protects input/intermediate/inverse-output observations and later
orphan cleanup. Native NameFix preserves only unambiguous visible-ancestor
node-input aliases; ambiguity fails before mutation and no graph rewiring occurs.
Current local-consumer membership excludes detached removed-node use links while
retaining graph-output and reachable nested-capture guards.
Superseded freezes and invalid direct-child-output fixtures remain identified
in the [new findings](poc/layout_and_native_binary_findings.md).

The native add/min/max packet forwards original arguments through one opt-in
shared helper, preserving all other 46 callers and LEGACY registration paths.
The separate diagnostic Mul-None rule retains exact native abstract evaluation;
actual min/max gradients now validate four CPU samples each per stack. Eight
signed-zero observations per stack remain FAILED, with exact target/payload
bounds. No default-profile or whole-family admission follows.

The shared-B packet is now implemented and has passed focused independent
review. Two `(B,K)` floating inputs share one positive atomic batch dimension
through audited Add and named composition; one unchanged artifact validates
B=2/5. Native/string declarations retain actual scope provenance, and named
formals remap correctly without merging unrelated scopes. Strict normalized
IR/proto/file/report signatures preserve dtype, weakness, order and aliases.
The sample layer binds inputs before session creation and each run, then checks
reference/actual outputs against the same fresh binding. It rejects B=1/B=5
broadcasting even though raw ORT executes it. BF16 checker/report success retains
a FAILED CPU Add load result. Commit `80cb68b2`, the complete primary-stack run
and actual configured hooks pass their integration gates.

No symbolic NCHW, broadcasting, affine expressions, other operations or control
flow are admitted by this packet. All 310 symbolic raw rows and seven NCHW rows
remain: the only raw symbolic declaration with two inputs of rank two and
identical shapes includes Divide/Mean.
Keep existing dynamic positive witnesses and fixed sources, without a new
profile mode, runtime Assert layer or source rewrite.

After this checkpoint, the accepted next packet is tooling for 13 exact raw
rows/26 artifact-only cells per stack: five dot, three matmul, four Add and one
FP64 gather row. Bind origin, ordinal, raw contract and actual typed equation
parameters. Exactly four dot rows receive explicit SOURCE MatMul counterparts
that preserve input/axis/no-bias/no-unused-input guarantees; execute and retain
the original Gemm callback failure separately. Keep other callbacks unchanged,
callback absence explicit, and the gather-off configuration conflict. Create a
new revision supplement; never reset failed histories or overwrite original
36-cell/two-dot evidence. The new selection leaves 3,578 cells unselected.

The same tooling packet produces a criterion proposal ledger only. Preserve
original unresolved criteria and record reference-first legacy versus
actual-first SOURCE allclose, FP32/FP64 variant thresholds, dtype conversions
and forced harness float32 matrix-precision settings. x64-on does not turn the
twelve strong-FP32 rows into FP64; gather needs its own audit. No numerical
execution, criterion adoption or public SampleCriteria change is released.
Review any proposed conjunction before outcomes. Ownership is limited to a
fixed corpus-oracle helper/test, the private source-support seam and maintained
probe tooling; numerical lowerers, metadata, profiles and diagnostic families
are outside the packet.

Min/Max sparse scalar/empty/output-only and the single FP16 cost comparison are
complete feasibility evidence: 72 exact edges and 12 scoped grid checks per
stack. Six raw rank-zero failures remain, with the rank-one adapter demonstrated
only in PoCs; inherited FP32/FP64 non-NaN mismatches remain failed. Dense costs
1.242–1.542× baseline and sparse/dense 0.963–1.220× on that one workload, with no
consistent sparse benefit; logical buffer sizes are not allocator measurements.
A later GatherElements lowerer candidate is recommended for bounded integration
and semantic/resource-cost review, without a production release. The small standard
Add/Neg/Where construction has a semantic argument but fails strict CPU checks
and changes other measured results. Gather alternatives have material or
unverified cost and are not selected for an unconditional rollout. Current export
has no runtime-target argument: `validate_source_samples` and the raising
`require_sample_acceptance` guard establish separate artifact-bound evidence.
Preserve failed samples and criteria; ordinary tolerances do not waive required
sign/rounding behavior. No Min/Max production packet, new target API, admission
or expensive rollout is approved.

Checkpoint evidence on 2026-10-18:

- M2's focused converter/source selection passes 299 tests on each supported
  stack. This includes the reviewed fix making optimizer failures fatal under
  SOURCE regardless of legacy environment settings.
- The integrated primary-stack standard regression run passes 4,995 tests
  (13,102 warnings, 387.20 seconds). It includes M3-M6, C1, and the deterministic
  fixtures, and excludes the then-new B1/B2 test modules.
  Their focused checks are separate evidence; this is not the final source
  corpus or phase F acceptance run.
- M4's dedicated evidence module passes 40 tests on each supported stack.
  Independent Astra review accepts its bounded metadata and sample-evidence
  contract. The public single-contract transition remains phase F.
- M3's internal profile passes 41 tests on each supported stack. Independent
  review and a primary-stack rerun accept its bounded typed admission rules.
- M5's reviewed fixes preserve enclosing function/branch diagnostics and accept
  empty layout selections. The affected selection passes 246 tests on each
  supported stack. Fourteen actual-facade CPU cases cover accepted controls and
  known numerical witnesses through IR/proto/file artifacts.
- C1's initializer and existing builder selection passes 162 tests on each
  supported stack and in independent primary-stack review. Actual inline/nested
  relu6 samples preserve FP16/FP32/FP64 results under both CPU optimization modes;
  BF16 artifact validity remains separate from its observed unavailable CPU kernel.
- The two deterministic attention fixture recipes pass three dedicated tests
  on each stack. Independent combined facade, sample, common-core and fixture
  verification passes 112 tests.
- At the phase A checkpoint, package typing covers 540 files; RNG registration,
  global Ruff, path markers, strict MkDocs and document links pass. The M3, C1,
  M5/M6, and fixture commits pass the actual pinned repository hooks. These
  checks do not certify the remaining migration or phase F.
- B1/B2 review found two reference-preparation defects: NumPy FP64 scalars lost
  their typed identity in contract fingerprints, and inferred typed inputs could
  be narrowed by JAX before their original dtype was recorded. Both are repaired.
  The 53 focused tests pass on each supported stack and in independent primary
  review, including exact scalar bytes, pre-conversion rejection, no model
  invocation on invalid input, and caller-state restoration. Fresh read-only
  baseline comparisons pass on both stacks; at `206277ac`, independent review
  confirms SHA256
  `fd3a7c50b8fa75dacc0c8cb62b5f00c495e42b86a51a70bdf50c1de6e341829e`.
  The baseline retains 1,802 raw rows, 63 unresolved opaque recipes, 752 original
  handwritten-test anchors and 90 migration additions. Optional environments
  and whole-corpus source execution remain unverified. The infrastructure is
  accepted and committed at `206277ac`. Later migration-test additions require
  an explicit baseline refresh without dropping or changing raw contracts.
- C2a's shared complex-constant repair is accepted and committed at `3e904b46`.
  The helper/packing selection passes 99 tests and the affected generated
  complex-operation selection passes 69 tests on each supported stack.
  Independent review passes the 93 new packing tests.
- B3's initial plan/preparation packet passes 53 focused tests on each stack
  and independent review. This includes 14 new plan tests and the 39 existing
  source-preparation checks. No broad source-corpus run or B3 numerical run has
  occurred.
- C2b and the separate discrete-comparison repair pass the same 976-test
  selection on each supported stack. Independent review runs all 138 new
  representation tests and 10 real-artifact discrete sample tests. Package
  typing covers 541 files; RNG, Ruff, pinned Black and builder policy checks
  pass. These changes are committed at `530561d1` and `52389e29`.
- C3a repairs the probed noncomplex while/fori/scan dtype and static-shape
  defects and the runtime zero-trip shape defect. Independent review passes
  its 60 new source-loop cases. Primary-stack selections pass 193 broader tests
  and 135 new/generated tests; the older-stack selection passes 269 tests.
  These selections overlap and have different totals; they are not summed or
  presented as identical matrices. The repair is committed at `1461e4b9`. At that
  checkpoint, complex loops and local promotion replay remained open; their
  subsequent finite C3b/C3c evidence is recorded below. Broader scopes stay open.
  Reverse scan is already rejected by the legacy path: record that preexisting
  unsupported contract separately, without calling it an ONNX impossibility.
- The historical integrated primary-stack run at the `158ce625` checkpoint,
  with no test exclusions, passes 5,363 tests (13,104 warnings, 502.81 seconds),
  including B1/B2 and these packets.
  Global Ruff, path markers and strict MkDocs also pass. This is regression
  evidence, not source-corpus or phase F acceptance.
- The `158ce625` inventory refresh retains all 1,802 raw rows, unchanged
  registered contracts, 63 opaque recipes and all 752 original handwritten-test
  anchors. It now records 186 handwritten-test files, 888 test functions and
  136 migration additions; no original anchor was removed. Fresh read-only
  comparisons pass on both stacks; optional integrations remain unresolved.
- C3b passes the same 359-test selection on both stacks; independent review
  passes all 52 new promotion-context and report-compatibility tests.
- C3c passes 404 focused tests and 75 generated loop tests on each stack;
  independent review passes its 121 new cases. Diagnostic lowering remains
  unverified. Actual bitwise sample evidence retains CPU Where's selected
  negative-zero failure on ORT 1.31/1.29 under both optimization settings.
  Broadcast, full-condition, Expand and flattened standard alternatives retain
  that failure; finite masking controls and unbatched exact-payload cases pass.
- Required finalization passes the same 523-test selection on both stacks;
  independent review passes its 54 new cases, including defaulted-attribute
  semantics, failure causes and preserving preexisting output files.
- The complete package typing check passes 541 files and RNG registrations
  with all three production packets frozen. B3b's initial independent review
  found a global-setting leak and a valid batched-while misclassification;
  the repaired tooling selection passes 125 tests on each stack. Independent
  review passes 58 tests on each stack and closes both findings. The fixed
  36-cell artifact batch is measured separately in the corpus findings.
- The preceding integrated primary-stack checkpoint passes **5,650 tests**, with 13,104
  warnings in 356.07 seconds. The preceding run's one failure and 5,647 passes
  exposed raw dictionary mutation by `tests/t_generator.py` during collection.
  The accepted shallow-copy repair preserves enrichment, variants, typed values
  and factory identities without weakening preflight. The 127-test focused
  selection passes on each stack, as do independent reruns of both new
  regressions and generated-module collection controls.
- The `3dbec1b8` inventory refresh changes only manual-test accounting: 191 files,
  944 test functions and 192 migration additions. All 752 original anchors,
  all 1,802 raw contracts and optional entries remain unchanged; no original
  anchor was removed. Fresh CLI comparisons pass on both stacks. The first B3
  batch remains bound to its earlier `9ac003ba...` baseline, not this refresh.
- C3b (`ea6a8da7`), C3c (`8ba3c681`), finalization (`f1ee71b3`) and B3/metadata
  ownership/inventory (`3dbec1b8`) all pass actual repository hooks. Package
  typing covers 541 files; RNG registrations, global Ruff, path markers and
  strict MkDocs pass. These checkpoints do not complete C4 or phases D/E/F.
- C4a passes 485 focused tests per stack. Independent review reruns all 23 new
  cases: 16 recording and seven facade/serialization cases. The logger and
  serializer implementations remain unchanged.
- D1a passes 453 focused tests and 47 generated LEGACY controls per stack;
  independent review reruns all 134 new cases. Twenty Boolean export cases have
  exact CPU samples with optimization off/all across inline/Function/If/Loop
  scopes. BF16 and neutral LEGACY Boolean Where kernel failures are recorded
  separately; a passing test count does not turn them into runtime support.
- The C4a/D1a complete primary-stack suite passes **5,803 tests**, with 13,104
  warnings in 382.81 seconds. Required package typing passes 541 files, RNG
  registration and global Ruff pass, and both C4a (`8a82d510`) and D1a
  (`223f24bf`) pass actual repository hooks.
- The `223f24bf` refresh changes only manual accounting: 192 files, 963 test
  functions and 211 migration additions. All 752 original anchors, 1,802 raw
  contracts and optional entries remain unchanged; no original anchor was
  removed. Fresh CLI comparisons pass on both stacks. The historical first B3
  batch remains bound to its original baseline and was not rerun.

- C4c has 107 new cases; independent review passes 110 selected cases on each
  stack, including three existing trace-helper controls, then the one affected
  regression after the final behavior-neutral typing repair. Owner selections
  pass 726 cases per stack, including 45 LEGACY controls. These overlapping
  counts are regression evidence, not a support denominator. The all-22 batch
  retains three checked artifacts and 19 pending rows per stack, zero ORT runs
  and unverified lowering/runtime/numerics.
- D1b passes 508 focused and 70 generated/LEGACY controls per stack; independent
  review passes all 125 new cases. Four exact linspace cells per stack remain
  failed under ORT off/all and ONNX ReferenceEvaluator. BF16 and dynamic arange
  target/compatibility failures stay explicit. The actual pinned Black hook
  changed one redundant-parentheses line; independent full-AST comparison,
  including type comments, confirms no behavior change.
- The C4c/D1b complete primary-stack suite passes **6,035 tests**, with 13,110
  warnings in 402.20 seconds, with no exclusions. Required package typing passes
  541 files, RNG registration and global Ruff pass, and C4c (`4c1c57fe`) and D1b
  (`05037961`) pass actual configured hooks. The earlier preflight interrupted
  after 1,215 passes during the typing repair is not a completed run.
- C4c's reviewed manual inventory records 194 files, 990 functions and 238
  migration additions. D1b's final refresh records 195 files, 1,003 functions and
  251 additions. All 752 original anchors, 1,802 raw contracts and optional
  entries remain unchanged. Fresh CLI comparisons pass on both stacks. The
  historical B3 batch remains bound to its `158ce625` baseline and was not rerun.

- The shared Dropout optimizer packet has 34 new cases. Owner selections pass
  339 tests on each stack; independent review passes 121, including the new cases
  and relevant existing controls. The valid original witness passes pre/post
  checker and exact inference data comparisons on both stacks. Separate maintained
  regression fixtures cover inference masks. A standalone Python 3.11 run
  initially imported an older checkout; that failed record is
  retained and corrected by explicit current-checkout path/hash verification.
  These checks do not claim stochastic-mask equivalence.
- D2a has 115 new cases. Owner checks pass 390 focused and 90 generated/integration
  controls per stack; independent review passes 435 per stack. Selected exact
  targets record 24 positive observations and four subnormal failures per stack.
  Those failures retain successful schema/runtime checks and rejected exact
  sample acceptance. Counts overlap and are not a matrix-family support measure.
- The Dropout/D2a complete primary-stack suite passes **6,183 tests**, with 13,110
  warnings in 406.62 seconds and no exclusions. Required typing passes 541 files
  plus RNG registration; global Ruff passes. Both `c528a894` and `2011b140` pass
  actual configured hooks. The log is
  `/tmp/jax2onnx-precision-dropout-d2a-full-py312.log`.
- The optimizer inventory records 196 handwritten-test files, 1,009 functions and
  257 migration additions; D2a's final inventory records 197 files, 1,023 functions
  and 271 additions. Both-stack comparisons pass. All 752 original anchors,
  1,802 raw contracts and optional entries are unchanged. The original B3 batch
  remains bound to `158ce625`; its separate two-cell supplement adds no accepted
  artifact, row or numerical criterion.

- The next native binary packet has 62 new cases. Owner checks pass 339 focused
  and 27 unchanged generated controls per stack; independent review passes the
  combined 366 each. Original eager/JIT references, settings, typed constants,
  cache orders, gradients and vmap are checked. Eight min/max signed-zero sample
  observations per stack remain FAILED; they are not numerical acceptances.
- The separate Mul-None diagnostic dependency has 18 new safeguards. Owner and
  independent selections pass 84 each stack; native abstract evaluation and
  unknown/non-None exclusions remain intact. Structural acceptance is separate
  from the binder's actual gradient runtime evidence.
- The initial C4b/observer candidate passed 783 owner checks before a typing-only
  variable rename; `8fd3cff6` then passed 51 owner observer cases, 572 independent
  checks and six original valid witness replays per stack. Its subsequent full
  run failed ten tests, with 6,335 passed and 13,110 warnings in 632.85 seconds;
  `/tmp/jax2onnx-precision-layout-binary-full-py312.log` retains that failure.
  Detached removed-Transpose use links caused all ten unchanged failures on both
  stacks. The earlier focused passes and superseded freezes are not integrated
  acceptance.
- Final optimizer `0f8b60c5` changes only the local-user expression to current
  graph consumers; output/nested/NameFix guards stay unchanged. Six new cases in
  one function bring observer coverage to 57, alongside 36 layout cases. Owner
  and independent selections each pass the same 238 tests per stack: ten original
  failures, 57 observer, 36 layout and 135 related controls. Counts are not additive.
  Ten collection-only controls and one injected-name check remain qualified.
  Root typing passes 541 files plus RNG, and global Ruff passes. File-hash prefixes
  are not commit IDs.
- The layout/binary complete primary-stack run passes **6,351 tests**, with 13,110 warnings
  in 466.07 seconds and exit status zero. The log is
  `/tmp/jax2onnx-precision-live-observer-full-py312.log`; all 17 code/inventory hashes
  in `/tmp/jax2onnx-precision-live-observer-full-freeze.sha256` match before and after
  commits. Optimizer `f02c0b4c`, binder/Mul-None `258e90d6` and static NCHW `7609fc2e`
  each passed all actual configured hooks, without bypasses or formatter upgrades.
- The layout/binary inventory refresh records 200 files, 1,074 functions and 322
  migration additions; both CLI comparisons pass. All 752 original anchors and
  1,802 raw contracts remain unchanged. The historical B3 batch and two-cell
  supplement, all-22 three-checked/19-pending results, and failed matrix,
  linspace and FP16 rounding criteria are unchanged. No row closes automatically.
- The shared-B packet adds 40 cases in 19 test functions. On the final eleven
  frozen paths, owner checks pass 697 each stack (8 warnings, 46.54/48.84 seconds);
  independent checks pass 241 each (16.44/18.51 seconds), including all new cases
  and four updated obsolete guards. Earlier 315/693/695 and fixture failures
  are preparation history. Before/after hashes match; counts overlap.
- Root final typing passes 541 files plus RNG, global Ruff passes, and both
  inventory CLI comparisons pass. Manual accounting now records 201 files,
  1,093 functions and 341 migration additions, with no raw/anchor changes or
  removals. The complete primary-stack run passes **6,391 tests**, with 13,110
  warnings in 444.36 seconds and exit status zero; log
  `/tmp/jax2onnx-symbolic-batch-root-full-py312.log`. Commit `80cb68b2` passed all
  seven actual configured hooks, recorded in
  `/tmp/jax2onnx-symbolic-batch-code-commit.log`. All twelve hashes in the root
  full freeze match after commit; the historical 6,351-test result is unchanged.

## Validation and final review

Use deterministic focused tests before broad validation. The final matrix covers
both supported Python/JAX stacks, legacy/source sequencing, ordinary JAX cache
isolation, both ambient x64 states, weak/strong inputs and captured values,
mixed results, source casts/constants, nested scopes, omitted/conflicting
options, supported artifacts, and meaningful rejection diagnostics.

Phase A uses the full standard suite on the primary stack and affected source,
facade, compatibility, and framework suites on Python 3.11/JAX 0.10.2. Overall
phase F runs the integrated source/legacy suites on both supported stacks,
including the reconciled generated corpus. Record genuine environment-specific
limits explicitly rather than silently shrinking collection. Use bounded workers
and restore modified JAX configuration in fixtures. Complete typing, Ruff,
actual repository pre-commit hooks, strict documentation build, and relative
link checks. Respect formatter versions in the hooks; changing their pins is
outside this implementation.

Public examples demonstrate entirely low-precision work, selected FP32 work
with deliberate cast-back, and scientific FP64 computation. Explain exact
versus minimum precision as model choices. Keep failed runtime examples visible
and latency/memory assessment separate from numerical acceptance.

Final review assesses code, tests, diagnostics, documentation, and evidence
together. Starting a full test run or committing intermediate work does not
complete the feature. Phase A may finish while the overall goal remains active.
Completion requires phase F's full existing-functionality accounting and its
reviewed technical-limit register.
