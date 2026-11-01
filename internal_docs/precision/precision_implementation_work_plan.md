# Computation precision implementation work plan

Updated 2026-10-31. This document retains the complete accepted **B–F** scope.
Completion means one source-preserving ordinary export contract across existing
functionality and unchanged supplied sources. A private profile, PoC, green
regression suite or partial model result cannot replace that scope.

Read the [target contract](precision_target_state.md) for semantics and the
[migration plan](precision_migration_plan.md) for sequencing and caller changes.
The [current state](precision_current_state.md) summarizes implemented behavior;
the [handoff](implementation_handoff.md) retains exact evidence and failed
predecessors. Commands, lane selections and runtime/resource accounting belong
in the [CI execution plan](ci_execution_plan.md), under the binding
[CI gate](precision_ci_gate.md). This work plan is a coverage and acceptance
contract, not a chronological test ledger.

## Settled decisions

- The unmerged branch may adopt the source-preserving contract incrementally.
  Keeping the previous public LEGACY default until full qualification is not a
  separate requirement. Retain legacy code only for concrete CI or meaningful
  regression/reference needs, with removal criteria. Review/test actual API
  changes normally; incomplete migration alone does not require deferral or
  separate permission. Full compatibility, numerical validation and independent acceptance
  remain prerequisites for merge. No permanent public selector, partial-profile
  product or per-operation legacy fallback is introduced.
- Use official supported dependencies. Custom builds, patches, binary overlays
  or substitutions of JAX, jaxlib, XLA or LLVM require separate explicit user
  approval and are currently prohibited. The unapproved compiler infrastructure
  has been removed; preserve the concise upstream reproduction.
- Keep original supplied/third-party sources, model dimensions, input values,
  state/control behavior and numerical requirements. External configuration,
  preparation and caller adapters must preserve supported workflows, including
  prior FP64 uses. No generic transformation API is assumed.
- The user retains strict reference agreement. Official eager/fresh-JIT/warmed-JIT
  contradictions remain FAILED until the required roles agree. Choosing another
  reference rule would require a separate explicit decision. The named BF16
  publication exception does not change mandatory scientific or final gates.
- A rejection can contain unfinished behavior, but does not complete its positive
  migration obligation. Missing references, criteria, execution or implementation
  never become acceptance.
- Reuse existing helpers, runners and evidence/report types. Remove superseded
  experiments and duplication; introduce infrastructure only for a concrete
  requirement. Implementation uses Sol Ultra and independent review Astra Ultra.

<a id="current-binding-minimum-version-obligation"></a>

## Supported dependencies and cohorts

Python 3.11 remains supported. The framework floor is JAX/JAXLIB 0.10.2 and
Flax/NNX 0.12.8. Poetry locks 0.10.2/0.12.8 on Python 3.11/3.12 and
JAX/JAXLIB 0.12.0.1 with Flax 0.12.10 on Python 3.13/3.14. The uv lock retains
its distinct 0.10.2-on-3.11 and 0.12.0-on-3.12+ markers and Flax 0.12.8.
Preserve each lock, rather than normalizing them to one environment.

Version-sensitive APIs go through `jax2onnx/_compat/jax.py`. Test actual supported
cohorts, not only a convenient local Python/framework pair. Minimum dependency
acceptance is separate, including ONNX IR 1.0.0, ONNX 1.20.1 and ORT 1.24.1.
Locked ORT 1.31.0 success does not qualify those minima. The
[dependency table](../../docs/about/dependencies.md) owns checked kernel and
behavior gaps and distinguishes LEGACY algorithms from SOURCE implementations.

## Completion contract

The final review must establish all of the following:

1. The ordinary trace/lower/optimize/finalize/name/serialize pipeline preserves
   source types, precision parameters, exact constants, deliberate rounding and
   required accumulation across the complete existing functionality.
2. Every existing registry entry, metadata case, manual regression, scope and
   export option has a retained contract, executable owner and explicit outcome.
   No migration gap, unaudited case, temporary exclusion or unexplained skip is
   left unresolved. Genuine limits have concrete evidence and diagnostics.
3. Caller/model-local context, weak typing, captured-state specialization, live
   trace ownership and configuration restoration survive success and failure.
   Completed exports do not retain model instances and weights through caches.
4. Unchanged fixed sources retain their supported static, dynamic and scientific
   workflows through explicit preparation/adapters, with independent references.
5. Actual returned artifacts preserve interfaces and carry versioned precision
   metadata through serialization. Representation and lowering are separate from
   runtime/numerical verdicts.
6. Sample acceptance binds the actual artifact and tensor contents, immutable
   inputs/references/criteria and actual execution settings. Failed, unverified,
   changed or incomplete evidence cannot pass.
7. Complete original model, corpus, minimum-runtime, browser and resource
   obligations pass at the required scope; smaller examples are supplementary.
8. Ordinary export, `allclose`, browser validation, generated tests, repository
   callers and documentation move together to one public contract with explicit
   old-flag migration errors.
9. Required local quality checks and the entire accepted suite pass on the exact
   final commit across all supported cohorts. Independent review accepts the
   semantic evidence, coverage and measured CI budgets; meaningful commits and
   authorized non-force pushes preserve the result.

Preexisting negative/unsupported contracts remain separately identified; the
migration does not require inventing new functionality for every formerly
rejected JAX operation. A positive existing capability cannot be moved into a
private legacy suite and called preserved. A missing implementation also cannot
be relabeled as an ONNX impossibility.

## Coverage baseline and acceptance accounting

The [current inventory epoch](current_inventory_epoch.md) and its
[predecessor mapping](source_inventory_epoch_mapping.json) retain **1,856 raw
rows / 3,712 caller obligations** in the modern observation, with explicit
cohort differences. Preserve the original 1,802 core contracts, 50 optional
models (45 MaxText and five MaxDiffusion), all 752 original manual anchors,
104 metadata-free registrations and 63 historical opaque-recipe obligations.
Seven recipes now have explicit metadata; that does not accept their models.
Registration-only entries remain obligations without invented testcase rows.

Reconcile live plugin/example registries, static registrations and generated
collections. Record origin, ordinal, module, callable/factory, complete typed
contract, input bytes, shape, weak metadata, precision parameters, opset,
structural callback, numerical criteria, selector and applicable scope/options.
Display-name collisions do not establish equivalence. Keep baseline tests
separate from additions and preserve reviewed changes in the epoch mapping.
Do not refresh a manifest simply to make drift disappear.

Retain the original 310 symbolic rows, 22 runtime-parameter rows, seven NCHW
rows, 41 nondefault-normalization rows and 51 loop rows. Original opsets include
9, 16, 18, 19, 21, 22, 23, 24 and 27; an internal admitted-opset subset cannot
define completion. New opset-26 evidence does not accept an original opset-23
contract. Optional import/configuration failures stay visible, with genuine
upstream checkouts and complete model discovery. Use the existing preparation
wrapper when regenerating coverage tables.

Each row keeps baseline and SOURCE outcomes separately:

| Classification | Meaning for acceptance |
| --- | --- |
| Source implementation validated | Scoped trace/representation/lowering evidence exists; runtime and numerical verdicts remain separate. |
| Genuine representation or tracing limit | A concrete semantic restriction has schema/configuration evidence or a minimal reproduction, including assessment of standard alternatives. |
| Known runtime/numerical failure | The named target fails load, execution or the required numerical criterion; another passing target does not erase it. |
| Unverified target | Required evidence is absent; this is not deployment readiness. |
| Migration gap or unaudited case | Implementation or acceptance work remains and blocks completion. |

Final coverage is not an indiscriminate Cartesian product. It includes every
existing tested contract, precision-sensitive variants identified by the audit
and representative compositions. The full inventory currently has global ERROR
and zero accepted scientific cells; accounting controls do not close it. The
1,838 missing tolerance pairs remain unresolved, and the existing 18 declared
pairs require scientific review rather than automatic adoption.

## Full migration phases and review gates

| Phase | Required work | Exit evidence |
| --- | --- | --- |
| B: inventory and source references | Complete provenance, independent source preparation, frozen criteria and actual outcome accounting. | Every obligation is retained and assigned; source references do not reproduce legacy coercion, and missing evidence remains explicit. |
| C: values, scopes and options | Canonical constants/values, logical/physical representations, functions, control flow, transformed bodies, dynamic shapes, runtime parameters, layouts and artifacts. | Existing boundary combinations preserve the contract through actual serialized artifacts; temporary exclusions do not remain as completion scope. |
| D: operators and frameworks | All existing typed operator/parameter families, high-level patches and abstract evaluation, numerical algorithms, state/RNG and transformed functionality. | Independently reviewed semantic/numerical evidence or a genuine-limit record for each original contract. |
| E: complete models and targets | Original model/corpus workloads, prepared scientific workflows, native references, CPU/minimum-runtime/browser execution and resources. | Complete original inputs and criteria pass; valid artifacts or isolated operators cannot stand in for model acceptance. |
| F: public transition | Integrated ordinary export/validation, migration errors, callers, docs and complete inventory reconciliation. | Zero unresolved migration gaps; complete exact-final-head suites, budgets and independent acceptance. |

Phase A's `source-static-v1` identifier records the first private integration
scope. Its narrow static operations, selected functions/branches and early
artifact evidence are historical milestones. Later finite extensions retain
their own scope; none replaces phases B–F. Preserve report compatibility for
those artifacts when evolving toward a general SOURCE contract.

## B: independent source preparation and numerical evidence

**B1 — Reproducible inventory.** Use the current manifest and predecessor mapping
with `scripts/source_precision_inventory.py`. Preserve full raw contracts,
manual anchors, optional status and cohort differences. Any unresolved difference between the frozen inventory and an older supported
JAX cohort remains explicit until reviewed; do not normalize it away. Inventory generation is identity/accounting evidence, not model execution.

**B2 — Original references.** Reuse original callables, deterministic factories,
input bytes and configuration. Execute references outside plugin activation,
with both caller-x64 settings and their actual precision/promotion context.
The legacy generator's input/expected-dtype rewriting and comparison casts are
not a SOURCE oracle. Explicit preparation scenarios retain their distinct
reference and requested change; x64-on alone does not change a strong FP32 model.

Preserve supplied JAX carriers, typed graph provenance and relevant native
options when inspecting references. A synthetic callable cannot certify the
original model. Audited admission and diagnostic probing remain distinct:
a successful diagnostic outside the audited product contract remains
`audit_pending`/unverified, even when its model loads.

**B3 — Actual numerical acceptance.** Freeze per-case criteria before execution.
Missing, one-sided or legacy-variant-only tolerances do not become exact/default
criteria. Check output count/tree, dtype and shape before values; integer/Boolean
outputs are exact under every floating comparison mode. Retain required
reference-first and actual-first comparisons where an accepted criterion uses
both, rather than silently changing comparison direction.

For the required Native3/ORT2 matrix, compare original eager, fresh-JIT and
warmed-JIT roles and both ORT optimization settings on the same artifact and
feeds. Required native roles must agree. Materialize references before ORT
sessions can alter the floating-point environment. A default target and an
explicit denormal-as-zero target are separate claims.

Exercise finite nonzero values, positive/negative zero, NaNs, signed infinities,
subnormals and rounding boundaries where relevant. Check zero sign bits and
NaN masks explicitly; numerical exact equality and `allclose` are insufficient
for signs. Include downstream reciprocal signs and other consumers exposing a
required intermediate. Permit legitimate all-zero results. Preserve NaN storage
words for literal/artifact identity where required, while using NaN-payload
output equality only when justified by that particular contract.

Retain known FP16 square/sum, weak-bias/scaling, optimized reciprocal, sequence,
subnormal matrix, extrema and BF16 TopK witnesses. Green diagnostic/rejection
tests are not equivalence proofs. Another sample's success cannot overwrite a
failed gate. Original structural callback failures remain recorded beside
reviewed SOURCE counterparts; source-semantic checks must eventually replace
inappropriate legacy fusion/count assumptions without hiding their history.

## C: values, contexts, scopes and export interfaces

**C1 — Canonical values and constants.** Maintain one typed binding per source
variable. Lookup does not change precision; output binding uses an existing
computed value. Explicit local consumer conversions preserve all other uses.
Constant metadata and actual initializer/Constant payloads agree, including
rounding-sensitive FP64 values, signed zero and required stored words. Protect
duplicate identity, captures, child builders and constant folding. No late global
promotion may rewrite SOURCE arithmetic. Keep IR-only construction and native
builder ownership rules.

**C2 — Logical and physical representations.** Use a shared typed descriptor in
core, facade, reports and reference/feed adapters. Complex64/complex128 have
FP32/FP64 real/imaginary storage with an explicit trailing component axis;
never infer complex values from an ordinary dimension of size two. Verify
passthrough, constants, channels, signed zero, mixed tuples and nested functions,
including actual storage. Explicit float64, int64, uint64 and complex128 input
requests that JAX would canonicalize under x64-off fail before model execution.

Numeric conversion, bitcast and arithmetic promotion have different semantics.
Preserve equal/unequal-width bitcast shapes and bits, packed-complex contracts
and original opsets. Same-type Identity or an opset-26 equal-width success does
not close unfinished different-width cases or minimum-runtime evidence.

**C3 — Context, functions and control flow.** Preserve supported caller and local
x64, default matrix precision and NumPy dtype/rank-promotion settings. Capture
and replay the actual call-site context, including local standard/allow scopes
under strict/raise callers. Specialization includes weak typing, captures and
precision-relevant settings even when output signatures match. Restore managed
state after success, exceptions, nested calls and finalization; old reports with
missing context fields retain “not recorded,” not inferred defaults.

Preserve ordinary-JAX/LEGACY/SOURCE trace isolation and legitimate live-trace
ownership. Completed/failed exports must release model/weight ownership without
unconditional global cache clearing. Audit `onnx_function`, cond, while, fori,
scan, JIT/pjit/custom-call bodies, nested combinations and existing transforms,
including AD. Keep schema differences behind compatibility helpers.

Loop obligations include typed conditions/counters, mixed carries, captured
values, zero-trip/static/empty results, scanned outputs and source weak typing.
Complex channel axes and component types must survive formals, carries and
stacked outputs. Separate complex representation and real-loop successes do
not prove complex loops. Preserve symbolic provenance, derived/constrained
shapes and original dynamic capabilities; finite leading-B results do not close
all symbolic rows.

**C4 — Options and shared facade.** Retain names, aliasing, output order/trees,
function parameters, symbolic sizes, normalization choices, recording and
proto/IR/standard-file/web-file interfaces. Verify signatures after naming,
optimization and finalization, then check the actual serialized equivalent.
Required attributes, Function postprocessing or shape failures must stop export.
Meaningful errors survive best-effort logging failures.

NCHW adapters preserve the fixed logical-to-physical permutation:
`(N,H,W,C)` to `(N,C,H,W)`; complex storage keeps its component axis last,
`(N,H,W,C,2)` to `(N,C,H,W,2)`. Named scalar inputs must not shift positional
layout indices. Verify coordinate-sensitive values, symbolic batches, aliases,
shared canonical values and IR/proto/reloaded reports.

Named `input_params` are actual typed runtime arguments throughout the root and
nested scopes. Static model configuration uses construction, closures or partial
application. Preserve native signature/default errors, literal overrides,
unused-input behavior, input-name/report arity and both Boolean values through
Function/If/Loop. Never infer dynamicness from decoration, add synthetic flags
to a specialized root or count a retained input as proof of live control.

Preserve all 22 fixed-source metadata models plus handwritten static witnesses.
Their baseline contains 16 positive-rate dynamic controls, five zero-dropout
GPT configurations and one static-False SpectralNorm calculation. Keep original
NNX/Linen state/key behavior and Equinox's missing-key failure explicit. Use a
supported external `lax.cond`/framework adapter around unchanged calls when a
Python branch needs dynamic control. Do not create a generic combinatorial
wrapper engine or freeze a working runtime control.

## D: complete operators, numerical algorithms and frameworks

The remaining work includes every existing family and its admitted parameters:

| Family | Required semantic evidence |
| --- | --- |
| Constructors, promotion and elementwise/transcendental operations | Native argument binding, weak/default types, exact casts, domain/special values, sequence rounding and original dynamic capabilities. High-level monkey-patches and abstract evaluation must preserve the supplied source, not only emitted IR types. |
| Indexing, gather, scatter, slicing, sorting and selection | Integer/index roles, shape and bounds contracts, duplicate/update semantics, exact discrete results and bit-preserving floating selection where required. Finite indexing or complex selection does not close the full family. |
| Reductions and windows | Source accumulation/result types, axes/identities/empty cases, NaN propagation, signed-zero ties, subnormals and downstream effects. Configured runtime success does not qualify default execution. |
| Matrix products, convolution and pooling | Native axis/batch order, layouts, operand/result/accumulation requirements and explicit precision algorithms separately. Wider operands require correctness and cost evidence; FP32 output alone proves no accumulation algorithm. |
| Normalization and attention | Original computation/statistics types, local casts, masks and selected implementations through nested scopes. Preserve original numerical criteria and existing options/opsets; LEGACY accuracy policies are not automatic SOURCE authority. |
| Complex, FFT and linear algebra | Actual component storage and source arithmetic, conversions, conjugation, multiplication, transforms, decomposition algorithms and special-value behavior across existing shapes/scopes. Finite broadcast/conjugation/layout/select acceptance does not establish general complex support. |
| Random, state, AD and framework-specific paths | Actual keys, key reuse rules, mutable state, callback/static preparation and transformed computations for existing Flax NNX/Linen, Equinox and other registered integrations. An early rejection or similar output distribution does not complete the positive contract. |

Audit optimizer rewrites against the source contract. Preserve deliberate
intermediate rounding and its downstream use; matching input/output types or a
loose final tolerance is insufficient. Validate or disable unproved fusions
such as Mean, normalization or fused reduction substitutions. Required provider
settings, hardware and algorithm limits belong in evidence rather than dtype
metadata. Standard decompositions may be used only with documented, validated
semantics; no silent precision reduction, opset upgrade or target fallback.

NNX preparation either rejects unsupported mutable static payloads before any
execution/mutation or provides reviewed isolation and complete state transport.
Retain the accepted containment controls and complete the remaining positive
state/RNG migration independently.

## E: original models, artifacts and execution targets

Run the complete reconciled metadata/examples corpus and representative original
NNX/Linen, Equinox, attention/CNN, recurrent, residual/decoder, GPT/ViT,
nested-control-flow, symbolic-shape and scientific models. Preserve the original
full inputs, weights, factories, comparison criteria and optional configurations.
The following anchors remain mandatory alongside the full inventory:

- All original 22 state-preparation models and both caller contexts, with genuine
  live controls, state feedback, repeated calls, reset/replay and original keys.
  Retain all cases in `test_source_original22_state_preparation.py`.
- The complete `test_source_model_native_bindings_exports.py` owner, including
  its ten full-model cases and negative controls. Preserve isolated original
  invocations, complete artifact transport and replay.
- Original NNX/Linen MLP dimensions 30/20/10, seed 17, Dropout rate 0.1,
  running-stat BatchNorm, GELU and captured parameters. Finite accepted sequences
  do not replace complete model/state/RNG obligations.
- Functions014/015 and their original parameters, decorators and
  LayerNorm → Linear → exact GELU → Dropout → Linear → Dropout computation.
  Preserve both live Dropout sites, Function014's fixed B=5 and each symbolic
  artifact's B=2/B=5 schedule. Earlier barriers do not prove a later barrier.
- All 30 original LayerNorm rows/60 caller contexts, including nondefault options,
  original graph callbacks and opsets, the six opset-16 cells, captured scale/bias
  reshapes and one reused artifact/session at B=2/B=5 where declared.
- Original SimpleLinear, AttentionCore, GPT_MLP, CNN/pooling and scientific
  witnesses, including exact versus minimum precision, mixed FP32/FP64 variance
  and explicit full Linear FP64 preparation. Keep raw callback failures and
  original missing criteria beside separately approved source comparisons.
- All 45 MaxText and five MaxDiffusion obligations in their prepared upstream
  environments. An export-only testcase needs full checker evidence and cannot
  claim numerical acceptance; missing optional setup cannot remove its rows.

Historical local memory stops and crashes remain unresolved observations unless
supported evidence establishes their cause and closure. Keep the original B=2
and B=5 payloads; returned stage bodies are not completed test/model acceptance.
Hosted success does not waive a local resource hold. Schedule complete expensive
cases in reviewed lanes; never shrink their inputs or split away required
comparisons to fit a budget.

### Artifact reports and sample acceptance

Preserve existing return types. A versioned report under metadata key
`jax2onnx.precision`, read by `get_precision_report(model_or_path)`, records:

- Source/exporter/framework versions, implemented scope, opset and actual caller
  precision/promotion settings, with backward-compatible absent-field handling.
- Supplied, normalized, traced and exported input properties and output signatures,
  including names, logical/physical shapes, dtypes and relevant weak typing.
- Trace warnings, implementation adaptations, locations and separate
  representation/lowering/runtime/numerical checks.

Runtime and numerical checks remain unverified when only export ran. Report
schema validity or an audited profile is not proof of every composition.
Required boundary checks occur after facade postprocessing/naming, with full
ONNX checker and save/reload evidence even when returning IR.

`validate_source_samples` takes explicit inputs, independent expected outputs
and criteria. Exact, tolerant and bitwise comparisons remain distinct; tolerant
criteria require both `rtol` and `atol`. Require nonempty samples, strict output
structure/dtypes/shapes and the B-phase numerical rules. Record actual CPU
provider/options, version, hardware/architecture, optimization mode and thread
counts. `optimization=None` executes nothing and stays unverified. CPU evidence
does not certify browser or accelerator execution.

The acceptance guard must bind the artifact, all tensor contents, sample and
reference bytes, criteria and settings, detecting mutable changes. Local file
identity resolves permitted external data; reject absolute/traversing sidecars,
symlinks and unresolved in-memory external tensors. Preserve large root
initializers and inline Function constants, exact payloads, web sidecar cleanup,
recording and primary-error behavior. Unsupported sizes/formats require clear
limits, not a weaker identity. BF16 transport must read actual returned OrtValues
and storage words, never unwritten requested buffers.

Minimum ONNX/ORT checks retain their own executable lane. Browser Node and
Chrome validation requires independent JAX references, complete original
conditions and qualified bounded execution; Python ORT agreement alone does
not establish source agreement. Keep provider fallbacks explicit and revalidate
the complete model after an allowed opset or implementation change.

## Executable acceptance ownership

Use the existing commands and runners. The execution plan supplies complete
argument forms, source/context checks and artifact locations. A missing verifier
below is work to implement and review, not permission to substitute exit zero.

| Obligation | Existing entrypoint/owner | Required verdict |
| --- | --- | --- |
| Current inventory | `scripts/source_precision_inventory.py` with the current manifest and genuine optional setup. | Complete identity/contract reconciliation on official families; no automatic refresh to hide drift. |
| Caller plan and replay | `scripts/source_precision_probe.py` with explicitly current baseline/adapters and a reviewed complete schedule. | Every assigned preparation/export/lowering/artifact outcome is inspected. Planning or process exit zero cannot certify pending/failed cells. |
| Current metadata numerics | Existing metadata pytest owner with `--source-metadata-acceptance`, exact selected cases/current adapters, and the full retained ledger. | Native references, reviewed criteria, actual ORT results and complete phase accounting for every obligation. Current global ERROR/zero scientific acceptance must be resolved. |
| Historical finite replay/numerics | Existing corpus replay and `scripts/source_precision_numerical_probe.py` with their authenticated historical inputs and fixed criteria. | Only the original bounded evidence; unavailable history stays unverified. Never reheader it as current full-corpus acceptance. |
| Full CPU/model suites | `scripts/run_pytest_batches.py` and complete existing generated, converter, framework, loop/scan and model owners. | Exact complete/disjoint collection and phase outcomes, genuine model numerical verdicts and resource closure. |
| Web | `scripts/run_onnxruntime_web_full.sh` and `scripts/run_onnxruntime_web_chrome_full.sh`, with original setup/conditions and reviewed bounded workers. | Complete Node/Chrome behavior against independent references. Existing unbounded settings do not establish resource qualification. |
| Quality and documentation | Pinned normal hooks, configured Ruff and Black 26.10.1, typing, RNG/policy checks, strict MkDocs and relative links. | Actual successful applicable checks; no hook bypass. Reviewed formatter and dependency transitions retain their original evidence epochs. |

The metadata proof-producer remains WIP. Its finite storage, report and consumer
controls must not accept production rows through synthetic examples. Complete
current replay adapters, scientific criteria and artifact-bound verdicts remain
required. Preserve the coverage obligations for all original, manual and
prepared-LEGACY counterparts while integrating results through existing reports;
that does not require permanent legacy runtime paths.

## Temporary legacy bridges

The current implementation has concrete CI/reference dependencies. This is a
source-inspection assessment, not a claim that replacement checks have run:

| Existing dependency | Temporary purpose | Removal or migration criterion |
| --- | --- | --- |
| `tests/t_generator.py` supplies the old flag to `to_onnx` and `allclose`, and prepares `_f64` variants. | Preserve main-equivalent generated coverage and its original contracts while the independent source lane is completed. | Account for each original capability with reviewed SOURCE preparation/references and explicit CI mapping. Retarget or isolate only the remaining historical comparisons; do not delete positive coverage to make the new API pass. |
| `test_exact_dtype_casts.py` and related legacy constant/scan regressions check the former widening and consumer-rounding policy. | Keep meaningful regression evidence for those old repairs and explain intentional arithmetic changes. | Preserve the witness and required new source behavior, then remove the obsolete live route when a retained fixture/artifact or source counterpart supplies the needed comparison. |
| `test_source_inputs.py`, `test_source_function_context.py` and facade tests compare omitted/false/true routes and cache interleavings. | Protect input/context ownership and source/legacy cache isolation while both routes exist. | Once the old runtime route is removed, retain ordinary-JAX/SOURCE isolation and meaningful failure/restoration controls; replace obsolete route assertions with the new migration-error contract. |
| Historical replay and explicitly bound legacy reference data. | Reproduce the original evidence without changing its oracle. | Keep the minimum authenticated reference material still needed. Do not preserve an entire duplicate product pipeline solely for historical data. |

These needs do not justify keeping every old lowering, promotion helper or
public default. Remove a legacy path when its concrete purpose ends. Changes to
baseline ownership, call behavior or comparisons still need normal review and
qualification; this documentation neither changes code nor activates a CI gate.

## F: coordinated public transition and final review

The branch may transition ordinary export and validation together before all
other migration work finishes. Complete integrated acceptance remains required
before merge and release. In the completed interface, omitted
`enable_double_precision` selects the one source-preserving contract; both
explicit Boolean values produce an actionable pre-trace migration error.
Shape-only inputs default to FP32; typed inputs and caller settings retain their
meaning. Provide runnable scientific-FP64 and fixed-source preparation examples,
explaining deliberate changes from legacy widening without promising identical
old numbers. No `explicit_float32` switch or public partial profile is introduced.

Migrate `allclose`, browser reference/input preparation and generated tests with
the exporter. Do not coerce outputs to expected dtypes or override JAX settings
to obtain agreement. Update API docs, coverage, limitations, examples and
changelog together. Explain low-precision-only models, selected FP32 regions
with cast-back, true FP64 work, exact versus minimum precision, and real target
limits. Keep correctness separate from latency/memory cost.

Routine PR lanes preserve main-equivalent coverage and focused affected extras.
Complete added suites, models, replay, numerical validation, minimum-runtime and integrations
remain required at significant milestones, when assumptions change and on the
exact final commit across supported cohorts. Each successful Linux STANDARD
job must meet the complete 20-minute budget, including setup/hooks/upload.
Extra budgets and resource limits are separate; no count or short subprocess
duration replaces the gate. Retain original checks until replacements have
exact-head evidence and independent activation review.

Final review considers code, original capability accounting, actual references,
numerical outcomes, artifacts, diagnostics, documentation and measured costs
together. The final commit must be documented, committed and independently
accepted with all required checks passing. The BF16 publication exception does
not waive final acceptance. If an official upstream or environment limit blocks
a required gate, present the evidence and the necessary decision explicitly;
keep the goal active and do not narrow the scope silently.
