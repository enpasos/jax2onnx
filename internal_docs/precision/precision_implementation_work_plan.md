# Source Precision Implementation Work Plan

Status: accepted implementation contract, 2026-10-18. Completion means consistent
source-precision behavior across the existing functionality, with genuine
technical limits stated explicitly. The private foundation, shared converter,
and CPU evidence module are committed and reviewed. Full coverage remains open.
The bounded private facade, canonical initializer repair, and deterministic
attention fixtures are reviewed and committed. The integrated primary-stack
regression passes; inventory/reference infrastructure is committed. A fixed
source-scenario plan, logical/physical boundary checks, exact discrete sample
comparison, and finite noncomplex loop repairs have passed independent review;
the repairs are committed and the full integrated run passes all 5,363 tests.
The scenario-plan checkpoint accompanies this update.
Source-corpus execution and broader migration remain open.
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
| Export boundary | `proto`, `ir`, and standard `file` return modes; model name and custom input/output names. |
| JAX configuration | Caller x64 and default matrix precision, including supported model-local scopes captured by named-function tracing. Explicit unsupported matrix algorithms fail. |

During phase A, the source path rejects symbolic dimensions, effects, loops/scan/JIT regions,
unlisted operations or precision parameters, nonempty `input_params`, nonempty
NCHW input/output options, `export_mode="web"`, and nondefault normalization
selection. Empty/omitted parameter and layout options retain their ordinary
meaning. These are temporary migration exclusions unless the inventory records
a genuine technical limit. Existing captured-state forms and supported
configuration behavior require audit in later phases. New concurrent-export
guarantees are not required beyond the existing functionality.

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
continues to mean FP32 and requires an intentional model change if FP64 is
wanted. Do not hide that difference behind another exporter mode. An earlier
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
| B3: fixed initial scenario plan | Preserve original factory configuration and typed inputs in paired x64-off/on cells; freeze provenance, criteria status, and bounded plan/prepare/product stages without a broad corpus run. | Independently accepted; included in this update | This checkpoint |
| C1: canonical initializers | Carry SOURCE policy into builders; preserve typed payloads and exact duplicate identity, require typed floating coefficients, and validate actual function Constant payload/metadata on reuse. | Committed | `b2974c3d` |
| C2a: complex constant storage | Pack actual initializer and Constant tensor payloads together with metadata, preserving the selected component dtype and checking cache consistency before mutation. | Committed | `3e904b46` |
| Sample comparison: discrete outputs | Compare integer/Boolean values exactly even when floating outputs use tolerance; retain strict dtype/shape checks and artifact-bound acceptance. | Committed | `52389e29` |
| C2b: typed representation boundary | Encode logical complex values into actual typed real/imaginary payloads; check physical boundaries exactly, reject input canonicalization before tracing, and label private diagnostic artifacts as unaudited. | Committed | `530561d1` |
| C3a: finite noncomplex loops | Preserve original JAX fori tracing, typed while/scan carries and static extents; materialize correctly shaped zero-trip scanned outputs after body validation. | Committed | `1461e4b9` |

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
The recorded local dtype/rank-promotion failures are C3b work: named-function
retracing currently loses a model-local `standard` or `allow` scope beneath an
outer `strict` or `raise` setting. Other existing transformed scopes and dynamic
loop contracts remain in the phase C inventory beyond the finite C3a packet.
C3b will add optional `caller_numpy_dtype_promotion` and
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
3,568 remain scheduled. Record environment, actual tracing settings, prepared
input/reference fingerprints and original signatures with stage evidence.
No numerical session is part of this first artifact-only batch.

Diagnostic lowering uses the shared facade and a real recursive validator.
Its fixed `source-diagnostic-v1` report keeps lowering `unverified` even when
representation checks and samples pass. The strict reader rejects a diagnostic
report claiming validated lowering. Audited product admission remains unchanged;
successful probes do not silently broaden `source-static-v1`.

Before broad phase D/F acceptance, audit required SOURCE finalization and
function-attribute processing for swallowed errors. A required transformation
failure must not leave a successful lowering claim merely because the artifact
still passes the schema checker. This audit remains open alongside C4 options
and the wider D/E/F gates.

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
  presented as identical matrices. The repair is committed at `1461e4b9`. Complex loops,
  further recursive scopes and local promotion settings remain open work.
  Reverse scan is already rejected by the legacy path: record that preexisting
  unsupported contract separately, without calling it an ONNX impossibility.
- The full integrated primary-stack run, with no test exclusions, passes 5,363
  tests (13,104 warnings, 502.81 seconds), including B1/B2 and these packets.
  Global Ruff, path markers and strict MkDocs also pass. This is regression
  evidence, not source-corpus or phase F acceptance.
- The explicitly refreshed inventory retains all 1,802 raw rows, unchanged
  registered contracts, 63 opaque recipes and all 752 original handwritten-test
  anchors. It now records 186 handwritten-test files, 888 test functions and
  136 migration additions; no original anchor was removed. Fresh read-only
  comparisons pass on both stacks; optional integrations remain unresolved.

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
