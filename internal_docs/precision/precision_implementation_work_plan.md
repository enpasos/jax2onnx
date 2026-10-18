# Source Precision Implementation Work Plan

Status: accepted implementation contract, 2026-10-18. The private foundation is
committed at `63abe1c1`. The shared private pipeline and CPU evidence module
have passed independent review. Product profile and facade integration remain.
The planned public selection is `precision="source"`; it is
unavailable until milestone M5 passes review. No release version is assigned.

This plan defines completion of an initial public source-precision feature for
static numerical kernels. The [target contract](precision_target_state.md)
describes the wider semantic design. The [migration plan](precision_migration_plan.md)
retains implementation history, and its [user view](precision_migration_plan_user_view.md)
explains the transition. Additional operators, execution targets, and export
options have separate future gates.

## Completion contract

Completion requires all of the following:

1. `to_onnx(..., precision="source")` uses the existing trace, lowering,
   optimization, finalization, naming, and serialization pipeline. Tests and
   examples call that pipeline rather than maintaining another exporter.
2. The `source-static-v1` profile is enforced for typed equations and their
   precision-relevant parameters, recursively through admitted functions and
   both conditional branches. Unsupported inputs/options fail before tracing;
   unsupported traced operations fail before their lowering.
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
7. Public API documentation, limitations, migration examples, changelog, focused
   dual-stack tests, full primary-stack tests, typing, lint, actual pre-commit
   hooks, and strict documentation checks pass. Independent review accepts the
   supported behavior and its stated limits.
8. Important accepted checkpoints are committed locally with their evidence.
   Completion does not imply a push, merge, publication, or deployment.

A private prototype or a plan update alone cannot complete this contract.
Automatic repair of known runtime rounding failures, all-plugin coverage,
loops/JIT, CUDA support, and a general capability database are outside it.

## Initial source-static-v1 profile

The profile identifier records support scope; it is not another caller option.
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

The source path rejects symbolic dimensions, effects, loops/scan/JIT regions,
unlisted operations or precision parameters, nonempty `input_params`, nonempty
NCHW input/output options, `export_mode="web"`, and nondefault normalization
selection. Empty/omitted parameter and layout options retain their ordinary
meaning. Other captured object forms, concurrent exports, and arbitrary JAX
configuration overrides are not part of this support claim.

Retain appropriate NaN, infinity, and signed-zero coverage. BF16 graph validity
does not establish an available CPU kernel. No operation is silently widened
to a global minimum precision. The existing matrix adapter widens its own
operands under its documented contract; its memory and performance costs are
separate from correctness evidence.

## Planned public selection and diagnostics

The selected spelling is `to_onnx(..., precision="source")`. It becomes available
only at M5. An internal omission sentinel distinguishes an omitted legacy flag
from an explicitly supplied Boolean.

| Selection | Required behavior |
| --- | --- |
| Precision and legacy flag omitted | Preserve the existing legacy default. |
| Precision omitted, legacy flag explicitly `False` or `True` | Preserve the corresponding legacy behavior. |
| `precision="source"`, legacy flag omitted | Preserve caller JAX configuration, source tracing properties, and the supported source contract; default shape-only inputs to FP32. |
| Source plus either explicit legacy Boolean | Reject before tracing, including explicit `False`; explain that the legacy argument must be omitted. |
| Unknown precision value | Reject before tracing. |

Existing `proto`, `ir`, and `file` return types remain intact. A versioned JSON
`PrecisionReport` is stored under model metadata key `jax2onnx.precision` and
obtained through the planned `jax2onnx.precision.get_precision_report(model_or_path)`
reader. Version 1 records:

- the `source-static-v1` profile, exporter/JAX versions, opset, caller x64, and
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

Structured failures identify option/input validation, JAX configuration or
tracing, profile exclusion, ONNX representation/lowering, runtime execution, or
numerical comparison. Include the available graph or function/branch location,
operation, requested dtypes/precision parameters, opset, failed requirement, and
supported next action. A profile exclusion is not a claim that ONNX cannot
represent the operation. Preserve trace warnings alongside observed types.

## CPU sample evidence boundary

The planned `jax2onnx.precision.validate_source_samples(...)` helper is separate
from export. Each `SourceSample` supplies concrete positional or named inputs,
expected outputs, and `SampleCriteria`. Exact, tolerant, and bitwise criteria
have distinct meanings; tolerant criteria require explicit `rtol` and `atol`.
Check output dtypes and shapes strictly. A tolerant final-output check cannot
certify an unobserved required rounding boundary.

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
| M3: enforced profile and reports | Product-owned typed profile, private facade integration, input normalization, structured errors/warnings, report metadata, names and three return modes, full checker and save/reload evidence. Review passes before public selection. | Pending | Pending |
| M4: CPU sample evidence | Artifact reader/identity, explicit samples/criteria, execution/numerical statuses, identity-aware acceptance guard, failure witnesses and accepted controls. Dedicated tests and API review pass. | Implemented and reviewed; checkpoint commit pending | Pending |
| M5: public opt-in and migration | After M2-M4 acceptance, expose source selection, omission/conflict handling, report/evidence helpers, API/limitations/changelog, and runnable public examples. Astra review passes. | Pending | Pending |
| M6: final acceptance | Required local validation, final independent review, accurate remaining limits, and final checkpoint; no required work for the declared profile remains. | Pending | Pending |

M2 has one owner for the coupled converter body and shared test harness. M3
owns source profile/facade changes and tests. M4 proceeds independently in
`jax2onnx/precision.py` and dedicated evidence tests; it must not introduce ONNX
protobuf imports into converter or plugin modules. Agree the plain report
schema before integration. M5 integrates the public surface after those reviews.
Record actual commit IDs and evidence when gates complete; pending entries are
not completion claims.

Checkpoint evidence on 2026-10-18:

- M2's focused converter/source selection passes 299 tests on each supported
  stack. This includes the reviewed fix making optimizer failures fatal under
  SOURCE regardless of legacy environment settings.
- The primary-stack standard regression run passes 4,692 tests. Its collection
  preceded the three new optimizer-failure cases and excludes the separately
  tested M4 module. Final acceptance will run the integrated standard suite.
- M4's dedicated evidence module passes 40 tests on each supported stack.
  Independent Astra review accepts its bounded metadata and sample-evidence
  contract; public-path integration remains an M5 gate.
- Package typing and RNG registration checks, Ruff, strict MkDocs, and all 60
  relative links in the six planning documents pass. Local commits use the
  actual pinned repository hooks.

## Validation and final review

Use deterministic focused tests before broad validation. The final matrix covers
both supported Python/JAX stacks, legacy/source sequencing, ordinary JAX cache
isolation, both ambient x64 states, weak/strong inputs and captured values,
mixed results, source casts/constants, nested scopes, omitted/conflicting
options, supported artifacts, and meaningful rejection diagnostics.

Run the full standard suite on the primary stack and affected source, facade,
compatibility, and framework suites on Python 3.11/JAX 0.10.2. Use bounded workers
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
complete the feature. The accepted initial scope can finish while documented
broader target work remains deferred.
