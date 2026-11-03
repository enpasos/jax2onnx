# A simpler precision architecture

Status: completed and merged. Step 1 ([PR #287](https://github.com/enpasos/jax2onnx/pull/287)),
Step 2 ([PR #288](https://github.com/enpasos/jax2onnx/pull/288)) and the trained
classifier demonstration ([PR #289](https://github.com/enpasos/jax2onnx/pull/289))
are on `main`. The accepted architecture, API migration and practical example
are implemented; there is no remaining implementation work in this programme.
[PR #277](https://github.com/enpasos/jax2onnx/pull/277) is closed without merge;
its implementation and unresolved findings remain historical evidence.
See the [completed classifier plan](quantized_classifier_plan.md) and
[user-facing example](../../docs/user_guide/quantized_mnist.md) for its results.
The goals, delivery plans and dated status reports below record the original
contract and development history. They are not an active work queue; earlier
failed or incomplete results retain their original scope. Post-merge CI and
administrative closure of issue #276 are separate from implementation completion.

The replacement for [PR #277](https://github.com/enpasos/jax2onnx/pull/277)
uses one consistent precision rule in the existing exporter, demonstrated by a
small, quantized JAX model that can run independently through ONNX Runtime.
The aim is a simpler user interface and a codebase that remains maintainable.
A successful demonstration alone would not establish the architectural change.

PR #277 and its implementation programme remain frozen. Its branch, findings and
failed or incomplete acceptance results remain historical evidence. This
replacement started from `main` without importing that implementation wholesale.
It does not reclassify the earlier outstanding requirements as satisfied. It
develops the direction discussed in
[issue #276](https://github.com/enpasos/jax2onnx/issues/276).

## Goal and user experience

The JAX model defines how each calculation chooses its computation and
accumulation precision, including deliberate conversions and output dtypes.
The export call supplies the inputs and their tracing properties and selects
the ONNX representation. Export preserves the resulting computation
within the capabilities of the selected ONNX version and execution target.

Users configure their model in JAX, Flax or another supported framework, supply
typed inputs, and call the ordinary `to_onnx(...)`. They do not choose an
exporter-wide arithmetic dtype, a SOURCE/LEGACY mode or an operator profile.
Quantization, when wanted, prepares the JAX model before export. Export does not
recalibrate weights or silently choose a different quantization scheme.

For example, a model may use INT8 matrix operands, accumulate in INT32, then
scale the result in FP32. Another model may deliberately round an FP64 value
to FP32 and continue computing in FP32. Both follow the same export rule.
The first example must not become a special Qwix path.

Input declarations retain their types and precision-relevant properties such
as JAX `weak_type`. Shape-only inputs have a documented FP32 default; that
default does not govern the rest of the model. JAX x64 settings belong to model
preparation and tracing. An explicit FP64 input incompatible with those
settings receives a clear error. Export does not silently change the caller's
configuration or lose a local precision setting at a function boundary.

Preserving the computation includes its required conversions and accumulation
behaviour. It does not promise identical floating-point bits across arbitrary
kernels and hardware. Numerical criteria must match the operation and the
execution target. A known violation cannot be hidden by a wider final tolerance.

## Architecture and hard boundaries

Python 3.11 and 3.12 are mandatory primary user platforms: the user reports that
they account for 95% of users. The implementation and dependency requirements
must retain both. A newer JAX or Flax release that requires Python 3.12 cannot
become the unconditional minimum while Python 3.11 remains supported. Existing
Python 3.13 and 3.14 coverage remains required as well. This platform requirement
does not by itself decide how far back to support JAX and Flax releases.

Keep the existing [IR converter and plugin architecture](../../docs/developer_guide/architecture.md).
Replace competing precision policies inside it rather than add a second
converter or a separate precision subsystem.

| Existing responsibility | Required rule |
| --- | --- |
| Input preparation and tracing | Preserve declared inputs, relevant JAX configuration and precision information needed by nested traces. |
| Values and constants | Keep one canonical typed binding for each traced value. Constant data and its declared type agree. Looking up a value never changes its type. |
| Operator plugins | Implement source operand, result, conversion and accumulation requirements. Tensor output dtype alone does not prove accumulation precision. |
| Functions and control flow | Preserve the same bindings, signatures and precision rules through functions, branches and loop state. |
| Optimization and serialization | Preserve observable rounding and conversions. Do not globally promote or narrow already typed constants and values. |

A lowering may create explicit intermediate conversions for its operation.
Those conversions must respect the source contract and must not replace the
canonical value seen by another consumer. Unknown types must receive a specific
unsupported-representation error, never silently become FP32.

Apply these rules across existing supported functionality. A missing
implementation is reported as missing; it is not described as an ONNX
impossibility. Supported operations must not alternate between old and new
arithmetic policies. The demonstration does not justify a callable whitelist
or a private export path that ordinary models cannot use.

Use existing diagnostics and validation facilities. Distinguish ONNX validity,
correct lowering, runtime availability and numerical acceptance. A schema-valid
model is not a deployment guarantee. Unknown runtime behaviour stays unverified;
a known failure stays failed. Operator-specific limitations belong with the
operator and the existing dependency documentation.

Only official supported JAX/jaxlib builds are permitted. Compiler patches,
binary overlays and custom JAX, XLA or LLVM builds require separate explicit
permission and are outside this proposal. Respect existing resource limits.

## Practical demonstration

The candidate is a trained MNIST digit classifier with a small
Dense–ReLU–Dense Flax model. Qwix prepares one specified post-training W8A8
configuration: 8-bit weights and activations, INT32 accumulation, and FP32
scales and results. The exact recipe and compatible versions must be established
in the feasibility step; general Qwix support is not implied.

The supplied callable captures prepared weights and accepts typed FP32 image
inputs. Ordinary `to_onnx(...)` produces an artifact that ONNX Runtime CPU can
load and execute in an environment containing neither JAX nor Qwix. The
exporter receives the actual prepared computation, without rewriting it into
a special demonstration implementation.

The intended benefit is portable quantized inference with a smaller artifact
than the corresponding FP32 model. Verify that weights are actually stored in
the quantized representation rather than keeping FP32 weights and quantizing
them on each invocation. Measure actual artifact bytes and runtime memory.
Report latency, but do not promise that a small INT8 model runs faster.

Compare two different properties separately:

- **Export fidelity:** ONNX agrees with the quantized JAX computation under
  predefined numerical criteria, including integer accumulation and scaling.
- **Task quality:** the quantized classifier retains an agreed level of
  accuracy relative to the useful FP32 classifier on held-out data.

Training, calibration and evaluation data are separate. Fix the model,
quantization recipe, data split and numerical and accuracy thresholds before
implementation acceptance. If a threshold fails, investigate the cause rather
than relax it to obtain a green result.

Training and dataset downloads are preparation tasks, not routine CI work.
Use reproducible weights and a small fixed offline regression fixture, with
provenance, licences and checksums. Evaluate task quality on the agreed held-out
set at demonstration acceptance; the small fixture does not replace it.
Large datasets and checkpoints stay outside git. Generated `.onnx` and
`.onnx.data` files must not be committed.

Qwix belongs in a pinned optional example environment. It must not become a
mandatory dependency, load during ordinary converter imports, or silently
change supported dependency cohorts. Whether the proposed example can fit
those constraints is part of feasibility.

## Public transition and scope

The current `enable_double_precision` option changes tracing, defaults,
constants and arithmetic. Replacing that policy is a behaviour change, not
merely removing an argument. Enabling JAX x64 cannot replace every previous
widening effect, especially inside unchanged third-party sources.

Before changing the public API, list affected existing workflows and their
concrete migration paths. Use model configuration, typed parameters and inputs
where they express the intended computation. Do not promise automatic rewriting
of fixed casts. An existing supported workflow without an acceptable migration
path is an unresolved decision, not permission to remove it silently.

The intended end state is one rule at an explicit, announced API version
boundary, with migration examples and an actionable error for the obsolete
argument. The released API stays unchanged until that transition is qualified.
Inside the implementation PR, preserving the old arithmetic is not a separate
requirement; keep temporary compatibility only where an existing CI check
actually needs it during the migration.
Any temporary compatibility code needs a removal point before the new contract
is released; permanent dual arithmetic modes are not the solution.

This project does not include redesigning `input_params`, general stateful-model
adapters, automatic precision rewriting of external model sources, a new
artifact-certification framework, or support for every quantization format.
FP8, FP4, INT4 and additional quantized deployment targets wait for concrete
requirements. Existing non-precision features and integrations retain their
regression coverage; these scope limits do not authorize breaking them.

## Acceptance criteria

All of the following are needed to complete a replacement implementation:

1. **One public rule.** The ordinary export API follows the model's precision
   choices throughout its supported computation. There is no hidden per-model
   or per-operator switch back to legacy arithmetic. Migration documentation,
   public API tests and release notes describe the same behaviour.
2. **General invariants.** Focused regressions cover source casts and rounding,
   weak inputs, exact constants, shared consumers, mixed outputs, functions,
   branches and loops. Include the known FP32-rounding counterexample from
   #277. Tests inspect relevant intermediates and their consumers where final
   outputs alone would miss a violation.
3. **Operation correctness.** INT8 products retain INT32 accumulation, including
   inputs that distinguish it from an inappropriate floating implementation.
   Check shapes, dtypes and discrete values exactly, and use operation-specific
   criteria for floating results. Check zero signs, NaN locations and exceptional
   values where the operation's contract requires them; all-zero results are
   not universally invalid.
4. **A complete useful example.** The trained classifier exports through the
   normal API, executes on the pinned CPU runtime without the training stack,
   meets quantized-JAX fidelity and task-quality criteria, and demonstrates
   the claimed artifact-size benefit. Report measured performance separately.
5. **Existing capability.** Affected operators and complete model workflows
   pass their regressions. Run the full suite, pinned hooks and documentation
   checks at implementation milestones and before release across the supported
   Python/JAX cohorts. Core and operator regressions cover those cohorts;
   the optional Qwix demonstration initially qualifies its one pinned supported
   environment. Do not drop supported cases or shrink model inputs to obtain a
   pass. Any intentional API change is reviewed explicitly.
6. **Practical CI.** Preserve main-equivalent routine coverage. Every Linux
   STANDARD job finishes within 20 minutes from job start through final cleanup,
   excluding only queue time. Added expensive validation has a separate,
   explicit integration job and runs at relevant milestones and final acceptance.
   Report aggregate work as well; splitting jobs must not conceal growing cost.
7. **Maintainability.** Independent review confirms that implicit promotion
   paths and competing policy branches have been removed, additions have clear
   owners in the existing pipeline, and ordinary converter use has no Qwix
   dependency. A contributor can explain the precision rule without a second
   parallel implementation or a large catalogue of special cases.

For claimed numerical acceptance, retain matching official JAX eager,
fresh-JIT and warmed-JIT references and validate the runtime settings being
claimed. Matching means meeting the predefined operation-specific criteria,
including exact properties where required, rather than universal bitwise
identity. A contradiction between required references remains a failure; it is
not repaired by choosing the convenient result. Historical failures in #277
remain recorded as such, and are not retroactively passed by this narrower
proposal. Green diagnostic tests do not establish numerical correctness.

## Step 1 — completed foundation

[PR #287](https://github.com/enpasos/jax2onnx/pull/287) is merged into `main`
at `8d502aac`. It preserves weak scalar promotion through ordinary inputs,
named functions, captures and function reuse. It also preserves 8-bit integer
matrix operands with INT32 accumulation, saturating floating-to-8-bit casts,
and the configured Linen computation for structured weights. These changes
use the ordinary converter; Qwix remains an optional example dependency.

The exact witnesses distinguish weak FP32 `2048 + FP16(1)`, which returns
FP16 `2048`, from the strongly typed FP32 result `2049`, and an INT8 contraction
whose INT32 result is `16,790,289`. The fixed-weight Qwix model exports its two
integer products through the same API. Its FP32 comparison uses
`rtol=1e-6`, `atol=1e-6`; native eager, fresh-JIT and warmed-JIT results and both
ORT optimization settings are required. Hosted CPU testing also established
the need for the portable UINT8 encoding of UINT8-left/INT8-right products.

All nine checks in [the accepted CI run](https://github.com/enpasos/jax2onnx/actions/runs/41413918783)
passed on `1e6c884f`, including the complete Python 3.11–3.14 jobs, minimum
dependencies and optional Qwix integration. The longest job took 19:02.
Earlier local full-suite attempts on the checkpoint and unchanged main stopped
at the 4 GiB memory limit; those attempts remain incomplete.

The subsequent dependency-compatibility commit `0723b2fe` repairs affected
exports on the locked JAX 0.10.2 and 0.12.0.1 cohorts and fixes FP16 sequence
metadata in `scan`.
Its local standard suite passed in sequential groups: 4,754 passed and two
existing optional skips. Another 119 focused tests passed on each JAX cohort,
along with all pinned hooks and the strict documentation build. This is
separate evidence from PR #287's hosted run; it does not establish hosted
acceptance for the newer dependency commit.

The global `enable_double_precision` behaviour still exists. In particular,
explicit FP32 arithmetic can become FP64 arithmetic during double export.
The trained classifier, held-out task accuracy and artifact-size benefit also
remain outstanding. Step 1 establishes useful export improvements, not the
complete replacement architecture or the complete demonstration.

## Step 2 — completed public precision contract

The following contract and delivery plan describe the work completed in #288.
Their original implementation instructions are retained for reference.

### Purpose and user experience

Step 2 should make the existing exporter consistently follow the computation
prepared in JAX. Its concrete benefit is reliable mixed-precision export:
explicit rounding, constants and nested functions keep the same meaning,
and users configure precision once in their model. Removing the competing
global policy should also make future lowerings easier to maintain.

The working quantized example does not by itself require this transition.
Step 2 is justified by the proposed correctness, user-interface and maintenance
improvements. Its acceptance must demonstrate those improvements directly;
passing the quantized example alone is insufficient.

For a model configured for FP64, the intended export call is:

```python
with jax.enable_x64():
    model = to_onnx(
        configured_model,
        inputs=[jax.ShapeDtypeStruct((2,), jnp.float64)],
    )
```

The model's parameters and operations must also express the intended precision.
This context permits FP64; it does not rewrite explicit FP32 operations.
Ordinary FP32 and quantized exports use the same API with their own typed inputs.

### Public API boundary and migration

The coordinated API boundary is the upcoming **0.18.0** release, as reviewed
at the first implementation checkpoint. The package is currently 0.17.1;
update both version declarations with the qualified transition. No release
date or merge is implied. The transition follows these rules:

| Existing workflow | Proposed rule and migration |
| --- | --- |
| Omitted precision flag | Preserve the caller's effective JAX configuration during export and restore it after success or failure. |
| Shape-only inputs, including inside an x64 context | Use the documented FP32 input default. Request FP64 with typed inputs. |
| A declared input type that configured JAX tracing would narrow | Raise an actionable error before tracing. Scoped `jax.enable_x64()` is the portable FP64 preparation on both checked cohorts. |
| Explicit FP64 permission with global x64 disabled | Respect it when the official JAX tracing API preserves that input type. The checked JAX 0.12 cohort does; JAX 0.10 abstract input tracing still narrows it and receives the diagnostic. |
| Explicit `enable_double_precision=True` or `False` | Reject the obsolete argument with a migration message at the announced boundary. Configure the model and inputs, then omit the argument. |
| Fixed FP32 casts or parameters in third-party code | Preserve them. Use the framework's actual dtype configuration where available; a workflow needing automatic rewriting remains an explicit compatibility decision. |
| Local precision context around a named function | Preserve effective x64, matrix precision, explicit-x64 permission and NumPy promotion mode for delayed traces and function reuse. Keep explicit primitive precision parameters. |
| `allclose(...)` validation | Use the same model and JAX configuration as export, with the same obsolete-argument diagnostic. Remove its independent x64 override and forced matrix-precision default at the same boundary. |

Apply the pre-trace compatibility check to other supported types that JAX would
otherwise narrow, including 64-bit integers; FP64 is the principal UX example.

Do not promise equivalence between an old globally widened computation and an
unchanged FP32 model. First demonstrate migration of an ordinary FP32 workflow,
a configurable FP64 workflow and a fixed-cast third-party-style workflow.
Include an actual existing full-model double-export case with its complete
inputs and established numerical criteria, not only scalar examples. Retain
fixed supplied callables rather than rewriting them to force a plugin path.
An important existing workflow without an acceptable migration blocks the
transition decision.

### One coordinated implementation

Keep one implementation in the existing pipeline. The static inventory finds
roughly 90 production files reading or exposing the current policy. Most leaf
changes are mechanical, but tracing, constant ownership and a few lowerings
require semantic changes. Removing the public argument alone is insufficient.

| Owner | Required change |
| --- | --- |
| `user_interface.py`, `converter/conversion_api.py` | Coordinate export and validation, input defaults and x64 checks. Stop imposing a global arithmetic dtype. |
| `ir_utils.py`, `converter/ir_builder.py`, `converter/ir_context.py`, `converter/ir_postprocess.py` | Map traced dtypes exactly and keep constant payloads and declarations consistent. Remove global promotion, narrowing and scope exceptions. Unknown or missing traced types get an explicit representation error. |
| `plugins/plugin_system.py`, `converter/function_scope.py` | Preserve effective context for delayed function traces and cache identity, retaining Step 1's weak types and capture handling. |
| Existing operator and control-flow plugins | Consume source types and signatures. Remove legacy-only promotion helpers; keep conversions required by the operation or its evidenced runtime implementation. |
| `tests/t_generator.py` and existing regressions | Construct intentional typed model/input variants and check their actual JAX output types. Remove automatic FP64 expected-output rewriting. Review each changed legacy expectation. |

The feasibility check addressed three concrete risks. Their implementations
still need production review and complete regression coverage:

- `jnp.where` can narrow a traced FP64 result and mutate a shared constant while
  adapting one consumer. Establish a distinct local conversion without changing
  the canonical constant seen by other consumers.
- `scan` contains dtype-based scope heuristics and even selects an integer
  result type from the global precision flag. Use the traced carry, sequence
  and output signatures while retaining necessary ONNX interface casts.
- Integer `reduce_window_sum` selects floating convolution precision from the
  flag. Demonstrate exact source integer behaviour or identify the necessary
  local lowering change before claiming that this policy can be removed safely.

Keep attention's required work precision, BF16 runtime adaptations and INT8/INT32
accumulation where their source and operation semantics require them. Do not
replace these with a generic floating dtype rule, a second converter, a
SOURCE/LEGACY switch or a model-specific export path.

### Delivery plan

The checkpoints belong to one coordinated implementation PR. They are useful
review and commit boundaries, not separately releasable precision modes.

1. **Establish the baseline and retain real precision coverage.** Start from
   fresh `main`, verify the dependency-compatibility fix and preserve local user
   changes. Confirm the supported minimum and locked dependency variants rather
   than infer them from an old document. Record the baseline commit and existing
   CI coverage in the PR. Review the generated FP64 cases: prepare genuinely
   FP64-configurable models and inputs through existing factories, and preserve
   deliberate fixed FP32, mixed and half computations. Review the remaining NNX
   MLP graph expectation against its source calculation. Keep its complete
   inputs and numerical criteria. Confirm the API boundary and migration
   examples. Open a draft PR with a short goal and scope when implementation
   begins.

   **Exit:** an independently reviewed fixture strategy, representative complete
   FP32/FP64 and fixed-cast workflows, and a justified structural expectation.
   An unresolved coverage loss or incompatible important workflow blocks the
   broad mechanical migration. The 39 prototype tests alone do not pass this
   checkpoint.

2. **Make types and tracing consistent in the existing converter.** Adapt the
   reviewed prototype's exact type mapping, canonical bindings, constant payload
   handling and function-context capture. Keep the finite context fields in
   existing tracing and cache ownership; route version differences through
   `_compat/jax.py`. Coordinate input preparation, `to_onnx` and `allclose`.
   Test configuration restoration after failures as well as successful calls.
   Remove shared promotion and narrowing; do not retain them behind a new mode.
   Integrate the small prototype lowering and framework repairs needed for
   these contract regressions before migrating the remaining consumers.

   **Exit:** all semantic scenarios covered by the 39 feasibility probes pass
   through the ordinary test path on both locked JAX cohorts. The genuinely
   FP64-prepared complete-model cases and the reviewed NNX structural case have
   actually run and passed. Equivalent consolidation of tests is welcome;
   retaining the number 39 is not a requirement. A fixture strategy or unchanged
   test names alone cannot satisfy this gate. Independent review accepts the
   ownership changes before broad migration of the remaining plugins.

3. **Migrate lowerings and remove obsolete policy.** Review all existing
   consumers of the old flag, including leaf plugins, control flow, optimizer
   passes, validation and test generation. Reuse local `where` and `scan`
   simplifications and the integer-window implementation where qualified.
   Finish operator-owned FP16 rounding and framework dtype handling, including
   exceptional values and the runtime settings actually claimed. Preserve
   intentional source conversions and required accumulation. Remove obsolete
   arguments, dead helpers, no-op compatibility signatures and scope guesses.

   **Exit:** no active competing arithmetic policy remains. Focused regressions
   pass on both JAX cohorts, existing actual FP64 coverage is retained, and
   changed optimizer and framework paths have explicit semantic review.
   A test may not become green by merely asserting a known numerical failure.

4. **Qualify complete supported workflows and document the change.** Run the
   complete standard suites across Python 3.11–3.14, minimum-dependency checks,
   existing support jobs, pinned hooks and strict documentation build. Retain
   Step 1's optional pinned Qwix integration in its existing job. Check the
   existing artifact interfaces, including functions, serialization and packed
   complex results. Update the API documentation, migration examples, dependency
   limitations and upcoming release notes together.

   **Exit:** independent review accepts correctness, compatibility and the
   complexity removed; local required checks pass without reducing supported
   cases, model inputs or numerical criteria.

5. **Accept the exact final PR revision.** Publish reviewed checkpoints and
   inspect their actual GitHub checks. Repair a red revision before publishing
   further feature changes; necessary repair pushes remain possible. At final
   acceptance, all required jobs must reach terminal `SUCCESS` on the exact PR
   head, including every complete Python job and support job. Each Linux
   STANDARD job must finish within 20 minutes. Review the final diff for
   temporary scaffolding and undocumented behaviour changes.

   **Exit:** the committed implementation, documentation, independent review
   and hosted results all refer to that same final revision. Merging remains
   the user's decision. This completes Step 2, not the later trained-model
   demonstration.

### Reuse from #277

Use current `main` and Step 1 as the implementation foundation. The frozen
`precision-migration` revision `3aaed9ff` is a source of specific examples and
small independently applicable repairs, not a branch to merge or cherry-pick
wholesale. Several of its witnesses are already in the feasibility prototype.

The files below are under `tests/extra_tests/converter/` on that frozen revision.
They identify useful starting material, not instructions to copy every test.

| Material from #277 | Selective reuse in the ordinary pipeline |
| --- | --- |
| `test_source_precision_preparation.py` | Reuse explicit FP32 rounding, its FP64 counterpart, weak/strong promotion and mixed-output examples. Retain the original model bodies, inputs and meaningful assertions. |
| `test_source_function_context.py` | Adapt equal-signature calls under different local contexts, correct cache reuse, nesting and restoration after errors. |
| `test_source_initializers.py` | Extract typed payload, shared-consumer, clone/serialization and duplicate-constant tests, including relevant signed-zero and NaN-payload distinctions. |
| `test_source_loop_precision.py` | Adapt mixed carry types, captured constants, scans without sequence inputs, and zero-trip, repeated and nested outputs. |
| `test_source_optimizer_precision.py` | Reuse concrete rounding and shared-intermediate counterexamples to review existing optimization passes. Do not copy policy-based pass exclusions. |
| Runtime findings and small production fixes | Keep evidenced limitations in the existing dependency/operator documentation. Review each code fragment against current ownership, retain only a necessary repair, and validate it afresh. |

Port small model bodies, fixtures and assertions into existing test modules.
Remove SOURCE/LEGACY wrappers, callable profiles, admission checks and custom
reporting machinery. Do not reproduce tests that only establish separation
between the old modes. Existing Step 1 fixes must not be implemented twice.
Every ported numerical claim needs current validation; a historical green test,
especially a test confirming a failure, is not inherited acceptance.

Keep failures and incomplete investigations identifiable, including upstream
reference contradictions. Reuse should save rediscovery, not turn an old
failure into a pass or justify an expanded test-product matrix.

### Step 2 acceptance

- The ordinary API and `allclose` follow the approved rules above, including
  configuration restoration after errors and useful migration errors.
- The [FP32 cancellation witness from #277](https://github.com/enpasos/jax2onnx/blob/3aaed9ff/tests/extra_tests/converter/test_source_precision_preparation.py)
  returns FP32 zero for FP64 input `2**24` after an explicit FP32 cast and
  `(rounded + FP32(1)) - rounded`. An intentionally FP64 counterpart returns
  FP64 one. Required official native references must agree before either
  exported result is accepted.
- Focused checks preserve constant payloads, shared consumers, mixed outputs,
  explicit casts, weak promotion, named-function context/reuse, branches and
  loops. Add distinct witnesses for the three risks above. Check exact shapes,
  types, integers and relevant zero signs; use predefined operation-specific
  criteria for other floating results.
- Step 1's integer-accumulation and optional Qwix regressions continue to pass.
  A trained classifier and its accuracy, size and deployment measurements are
  a later demonstration milestone, not a prerequisite for this API-contract PR
  and not fulfilled by it.
- Existing operators and complete model inputs retain coverage on the supported
  cohorts. Migrate tests that intentionally encoded the old policy against the
  approved new source behaviour; do not widen numerical tolerances to obtain a
  pass. Review actual typed fixtures and source results: unchanged `_f64` names
  and test counts are insufficient. Every removed or changed structural
  assertion needs a source-based justification. Existing native-reference
  contradictions remain failures.
- Full suites, minimum-dependency checks, pinned hooks and strict documentation
  builds pass at the reviewed checkpoint. Each Linux STANDARD job remains under
  20 minutes with existing coverage. Reuse the existing optional integration job
  and focused tests; do not add a new evidence framework or a large test-product
  matrix.
- Independent review confirms that global promotion and scope heuristics are
  gone, operation-specific conversions have clear owners, and ordinary imports
  need no Qwix. No permanent parallel precision policy or temporary no-op
  scaffolding remains. The final review explains which competing rules were
  deleted and why each necessary runtime adaptation belongs to its operator;
  net line count alone is not a maintainability verdict.
- All required hosted checks reach terminal `SUCCESS` on the exact final PR
  head, including all full Python 3.11–3.14 jobs and support jobs. A previous
  head, a partial run, skipped required work or a cancelled job cannot satisfy
  this gate. Green CI is necessary and does not replace the semantic and
  maintenance review above.

Passing these gates completes Step 2. Completion of the entire concept still
requires the useful trained-model demonstration and its separate acceptance.

### Feasibility result — 2 November 2026

**The architecture is technically feasible in the existing converter.** The
prototype needed no second exporter, global precision mode or compiler patch.
Source types, constants, function context and operation-owned conversions fit
the current responsibilities. This supports a bounded implementation proposal;
it does not qualify a release or establish that the larger change is worth its
cost today.

The final focused set passed **39 tests on each of the official JAX 0.10.2 and
0.12.0.1 cohorts** on Python 3.12 and 3.14 respectively, using ORT 1.31.0 and
1.31.1. Numerical witnesses compare eager, fresh-JIT and warmed-JIT references
with CPU ORT optimizations disabled and enabled; input diagnostics are checked
separately. The tested scope is:

| Question | Observed result |
| --- | --- |
| Explicit FP32 rounding inside an FP64-input computation | Exact FP32 zero; the deliberately FP64 counterpart returns exact FP64 one. |
| Existing complete-model migration | Linen MLP in FP32 and explicitly configured FP64, plus the NNX MLP FP64 workflow, pass with complete `(7, 30)` inputs, original generator seeds and established `1e-7` FP64 criteria. Existing FP32 weights are retained and widened explicitly for the configured Linen model. |
| Fixed supplied FP32 computation | The unchanged GPT-OSS RMSNorm class keeps its internal FP32 calculation on `(2, 64)` FP64 inputs and returns the source result within `1e-7`. This does not provide automatic FP64 rewriting of that class. |
| Delayed function tracing and reuse | Four finite JAX precision settings survive ordinary, unique and nested function calls, with source weak types and captures retained. |
| Shared constants, scan and integer windows | Consumer-local conversions preserve other users; INT32/INT64 and unsigned sums retain exact wrapping, including values above `2**53`, dilation, symbolic extents and empty outputs. |
| Existing Step 1 witnesses | Weak/strong scalar results remain FP16 `2048` / FP32 `2049`; the 1,041-term integer contraction witness remains INT32 `16,790,289`. |
| Configured framework precision | An additional Dense witness exposed an ignored Flax computation dtype. Using Flax's own `promote_dtype` preparation distinguishes exact FP32 `2**24` from FP64 `2**24 + 1`. |

Several operation-specific repairs were necessary. Integer ORT `ReduceSum`
saturated in the tested overflow cases; signed `CumSum` followed by selection
of the final prefix preserved the source sum. The tested UINT32 and UINT64
values use same-width signed work values and an exact final cast. The initial mixed-type FP64
BatchNorm representation had no matching CPU kernel; tracing the original Flax
calculation supplied a working source-faithful representation.

The FP16 case is especially important. ORT 1.31.1 removed an intermediate half
rounding boundary even with ordinary optimizations disabled, returning `1`
where all required JAX references returned `0`. The prepared runtime graph
confirmed the change. A small Add/Sub-owned representation using FP32 work and
explicit FP16 result casts passed the original witness and a shared-input,
signed-zero witness on both cohorts. The original unmodified runtime models
remain failed. This establishes the tested repair, not qualification of all
half operators, exceptional values, subnormals, providers or performance.

The test-generator prototype also runs existing cases with caller-owned JAX
configuration and checks each actual source output type, retaining the existing
packed-complex interface. The final existing-test sample is **nine passed and
one failed per cohort**. The remaining NNX MLP FP64 failure is its unchanged
`BatchNormalization` graph expectation after the source calculation was
expanded. Its separate complete numerical witness passes; the structural
expectation still requires review and was not waived.

Most importantly, unchanged `_f64` names and test counts do not prove unchanged
FP64 coverage. Some supplied fixtures explicitly contain FP32 values or fixed
FP32 model choices. Their intended polymorphic FP64 variants need explicit
preparation through existing fixtures and factories, while deliberate mixed
and half types must stay fixed. This review is a substantial part of the
remaining work. Minimum dependencies, full regressions, broader operator and
framework-binding behaviour, and hosted CI cost remain unqualified.

The reviewed prototype and raw outcomes are retained locally under
`tmp/precision-step2-feasibility-20261102/`; they are not part of the release
implementation. Earlier failed attempts and runtime counterexamples remain in
that record. The normal working branch contains this concept update; the
experimental code lives in a separate disposable worktree.

### Original effort estimate and implementation notes

The user activated this goal on 2 November 2026. The five delivery checkpoints
govern implementation; activation does not authorize a release or merge.
Astra's preliminary estimate remains **seven to twelve engineering days**,
including fixture migration, framework-binding review, minimum dependencies,
complete regressions, documentation and independent review. Reuse from #277
reduces rediscovery; it has not established a reliable smaller estimate. The
estimate excludes classifier training and the later quality, artifact-size and
deployment demonstration. It is not an elapsed-time or token forecast.

Review progress and the remaining estimate at the first checkpoint and after
the first two engineering days of implementation. Report concrete completed
work, unresolved failures and any growth in scope. Do not manufacture a
completion percentage from passing tests. Any hard time or token budget must
be agreed separately. The previous four-hour cap applied only to the completed
feasibility check; it does not carry over to implementation.

Do not expand a difficult lowering into general arithmetic emulation, another
policy engine, compiler changes or an unrelated operator project. Present the
specific blocker and smallest alternatives for a user decision while leaving
the affected acceptance gate unresolved. The same applies if genuine existing
coverage or the 20-minute STANDARD limit cannot be retained. A technical
limitation does not authorize silent scope reduction or completion of the goal.

The local `precision-contract-step2` branch was created from the fetched main
at `8d502aac`. A fresh fetch at implementation start still returned that main
revision without the dependency fix. The branch therefore includes the already
reviewed compatibility commit `0723b2fe`, cherry-picked as `639f1429`; their
committed trees are identical. Continue on `precision-contract-step2`; preserve
PR #277 and its branch as frozen historical material.

Early validation passed source-precision regressions on both locked JAX
cohorts, including complete static and dynamic FP64 Linen and NNX MLP workflows
with strict structural checks. Independent Astra review accepted that bounded
gate for proceeding with policy cleanup. It did not qualify the full suites,
minimum dependencies or hosted CI for the precision transition.

[Draft PR #288](https://github.com/enpasos/jax2onnx/pull/288) is open. Its first
published head `d3fc3aa2` contained the compatibility repair and this contract.
All nine jobs in
[CI 618](https://github.com/enpasos/jax2onnx/actions/runs/41465407265) passed on
that exact head. The longest STANDARD job was Python 3.12 at 17:18, measured
from GitHub's job start through completion. These results qualify that
foundation head; precision-transition qualification follows the requirements below.

The core precision migration, production-policy cleanup and fixture migrations
are implemented. Local regression checks finished without assertion failures
and with the existing skips unchanged; only the resource-incomplete Qwen case
below lacks a completed result. Focused checks passed on both official locked
JAX cohorts and the supported minimum, including canonical pointwise Einsum.
Qualification requires the unchanged complete hosted Python 3.11–3.14 STANDARD
matrix, support/minimum/integration checks, and every Linux STANDARD job within
20 minutes excluding queue time. [PR #288](https://github.com/enpasos/jax2onnx/pull/288)
provides the live check results.

In local qualification attempts on the same locked Python 3.12 stack,
`qwen3_omni_30b_a3b` hit the 4 GiB guard on both the implementation snapshot and
exact clean main `8d502aac`, without an assertion result. This establishes a
pre-existing local execution limit, with no evidence of equal peak memory or
performance. Astra approved completing local checks before a reviewed draft
checkpoint push with hosted coverage unchanged. The existing full hosted
Python 3.12 MaxText job can qualify only this case's existing export test:
main marks it `skip_numeric_validation`, so a passing job does not qualify
runtime execution or numerics, which remain unverified.

Bounded repairs isolate patched export tracing through JAX's public cache
namespace without changing precision settings or clearing the native cache.
The `lax.conj(real)` lowering now preserves its actual complex result by packing
the real input with a positive-zero imaginary component; source types also
distinguish real tensors ending in dimension 2 from packed complex tensors.
Shape-specific matrix algorithm compilation limits remain
[documented separately](../../docs/user_guide/known_limitations.md#explicit-matrix-multiplication-algorithms).

On 2 November 2026, the user amended acceptance for exactly six FP64
`atan2(y, x)` input pairs: `(s, -M)`, `(s, -1)`, `(s, s)`, `(s, -s)`,
`(-s, s)` and `(-s, -s)`, where `s` is the smallest positive subnormal and `M`
the largest finite FP64 value. The full 4,265 inputs, eager/fresh/warmed native
consensus, DOUBLE graph checks, shape and exceptional-value checks remain
mandatory. Both ONNX Runtime optimization settings use the original
`rtol=0, atol=4e-15`; every mismatch must lie within those six bit-exact input
pairs. Zero-location and sign checks enforce input-defined analytic zeros and
the signs of zeros shared by both results; finite residuals within the original
tolerance need not have the same zero location as native. This clarifies the
existing criterion without adding a runtime boundary or another exception.
A passing boundary diagnostic records any remaining differences as
`FAILED_IN_APPROVED_BOUNDARY`, not numerical qualification, and never emulates
the native behavior.

The user delegated assessment of similar bounded cases with Astra review and
main-equivalent nonregression. This amendment covers only the six named pairs;
all other cases and the existing Python 3.11–3.14 and dependency requirements
remain mandatory. The separately accepted minimum-JAX `rsqrt` diagnostic below
remains numerically unqualified.

Minimum-stack checks on 2 November 2026 originally failed at a real numerical
boundary: all three official JAX 0.8.1 references return FP16
`0.003910064697265625` for `rsqrt(65504)`, whereas the current lowering returns
`0.00390625`. Cross-checks
with ONNX Runtime 1.24.1 and 1.31.0 show that the split follows JAX, rather than
the runtime version. JAX 0.10.2 and 0.12.0.1 agree with the lowering. An isolated
comparison with main `8d502aac`, using its valid `enable_double_precision=False`
export, found bit-identical main/current results on all eight inputs under both
ONNX Runtime optimization settings.

Under the user's delegation, Astra accepted only this diagnostic boundary for
the finite eight-input case on CPU JAX/jaxlib 0.8.1 with CPU ONNX Runtime
1.24.1: input FP16 bits `0x7bff` (`65504`), native result bits `7169`, runtime
result bits `7168`. The full native consensus and all other comparisons remain
exact; any other mismatch fails. Exceptional cases and all other dependency
combinations keep the existing strict check. A diagnostic pass records the
actual mismatch as `FAILED_IN_APPROVED_BOUNDARY`, not numerical qualification;
exact agreement is also allowed. Production arithmetic and the JAX 0.8.1 /
Flax 0.12.1 floors remain unchanged, with all four Python cohorts required.

### Completed Step 2 goal — historical contract

The following text records the contract activated on 2 November 2026 and
completed through #288. Its scope ended at Step 2. The separately accepted
trained-model demonstration was subsequently completed through #289.

```text
Implement Step 2 of the precision architecture in
/home/enpasos/projects/jax2onnx on precision-contract-step2, based on fresh main
and internal_docs/precision/precision_architecture_concept.md. Use its public
migration rules, delivery checkpoints, Step 2 acceptance criteria and hard
boundaries as the contract. Use gpt-6.1-sol with ultra reasoning for
implementation and tests, and gpt-6-astra with ultra reasoning for architecture
decisions and independent checkpoint/final reviews. Additional use of
gpt-6-astra with ultra reasoning is explicitly authorized whenever needed for
analysis, difficult technical decisions or independent assessment during this
goal; no separate permission is required.

Deliver one precision contract through the existing to_onnx and allclose APIs:
the prepared JAX model determines computation, accumulation, deliberate
conversions and result types; typed inputs and effective JAX context retain
their meaning. Coordinate the announced API boundary and provide an actionable
diagnostic for the retired argument. Remove global promotion/narrowing, competing
policy branches and temporary scaffolding. Keep necessary conversions owned by
their operations, canonical typed values unchanged, and existing artifact
interfaces and non-precision functionality intact.

Follow the five delivery checkpoints. First qualify real FP64 fixture coverage
and complete existing model migrations, including the remaining NNX MLP graph
expectation; passing the 39 feasibility tests is insufficient. Selectively reuse
#277's small regression witnesses, findings and independently applicable fixes
as specified in the reuse table. Do not merge its frozen implementation or
restore SOURCE/LEGACY controllers, profiles or evidence/reporting infrastructure.

Retain supported cases, complete model inputs, dependency cohorts and existing
numerical criteria. Use official JAX/jaxlib only; no compiler patches, overlays
or custom builds. Preserve user changes and machine stability, including the
existing single native worker and 4 GiB RAM limit. Do not weaken checks, silently
rewrite fixed supplied sources or treat native-reference contradictions as
passes. Limit runtime claims to settings actually validated. Unknown, incomplete
and failed results remain distinguishable. Do not build a second exporter,
general arithmetic-emulation layer or new mandatory Qwix dependency.

Run focused dual-JAX checks for semantic changes. At reviewed implementation
milestones and final acceptance, run the complete standard suites, supported
minimum-dependency checks, pinned hooks, strict documentation build and existing
support/integration jobs. Preserve main-equivalent STANDARD coverage; every
Linux STANDARD job must finish within 20 minutes, excluding queue time only.
Keep additional expensive checks separate and avoid redundant full reruns when
no change or unresolved concern justifies them.

Create meaningful reviewed and validated commits. On activation, ordinary
commits and non-force pushes to precision-contract-step2 and opening/updating
its one draft implementation PR are authorized. After each push, inspect the
actual checks on that exact head; repair regressions before further feature
pushes. Necessary repair pushes are allowed. Do not merge, force-push, publish a
release, post unrequested comments or otherwise modify GitHub outside this scope.

Report progress against the checkpoints and reassess effort at the first one
and after two engineering days. Escalate any need to expand the architecture,
weaken coverage or exceed the runtime bound; leave unresolved work unfinished.
The seven-to-twelve-day estimate is provisional, not a token or spending budget.

Finish only when the whole Step 2 scope is implemented, documented, independently
reviewed and committed, with all required PR checks at terminal SUCCESS on the
exact final head, including the complete Python 3.11–3.14 and support jobs.
Passing CI alone is insufficient. Classifier training, held-out accuracy,
artifact-size benefit and standalone deployment demonstration are a separate
later milestone and must not be reported as completed by this goal.
```

## Lessons and references

PR #277 established useful counterexamples but coupled precision preservation
to a much broader migration and evidence programme. Preserve its findings,
separate exporter defects from upstream/runtime behaviour, and keep decisions
and evidence proportionate to the release claim. Maintain this one concept
document and ordinary focused tests rather than duplicate current-state,
handoff and execution histories.

- [#275: explicit FP32 values and double export](https://github.com/enpasos/jax2onnx/pull/275)
  shows the conflict between source rounding and global promotion.
- [#278: stable elementary functions](https://github.com/enpasos/jax2onnx/pull/278)
  and [#279: attention and low-precision types](https://github.com/enpasos/jax2onnx/pull/279)
  show the value of operation-specific fixes within the existing architecture.
- [Qwix](https://github.com/google/qwix) documents W8A8 model preparation;
  [its accumulator selection](https://github.com/google/qwix/blob/main/qwix/_src/core/qarray.py)
  separates integer accumulation from the floating result.
- [ONNX MatMulInteger](https://onnx.ai/onnx/operators/onnx__MatMulInteger.html)
  provides a candidate representation for 8-bit operands and INT32 output.
  That schema does not establish complete Qwix export or runtime performance.
- [JAX numerical differences under JIT](https://docs.jax.dev/en/latest/faq.html#jit-changes-the-exact-numerics-of-outputs)
  explain why reference disagreement must be distinguished from an export bug.
