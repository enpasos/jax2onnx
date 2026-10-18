# Computation Precision Migration Plan: User View

Status: accepted plan for the complete precision transition, 2026-10-18.
The completed normal `to_onnx(...)` call will preserve the JAX computation
through one consistent rule across the existing functionality. There will be
no public precision-mode or partial-profile choice. The
[implementation work plan](precision_implementation_work_plan.md) requires the
full coverage gates before that public transition. PR #275's bounded repair is
merged. The local branch now preserves
weak inputs and selected calculations through real functions and branches
using a private policy. Local JAX settings and successive export modes now have
regression coverage. Bounded matrix and selected-FP64 normalization cases also
export through this path. Public legacy options and defaults retain their rules;
the full migration and deployment evidence remain unfinished.
No release or deadline is promised.

The [Typed values and masks checkpoint](poc/typed_values_and_masks_findings.md)
adds reviewed private rules for Boolean and integer inputs, comparisons,
conversions and selection, plus selected static complex values. Both supported
stacks pass 2,047 focused checks on the actual migration checkout; independent
integration review accepts the checkpoint. The complete primary-stack suite
passes 8,659 tests, and code commit `0d8f9d19` passes all repository hooks after
a comment-only path-marker correction. The original GPT and embeddings
integer inputs now pass input checks, but export still stops at the named-parameter
restriction before model construction. This is progress towards existing models,
not completed model support.

The named-parameter packet is released for isolated implementation to preserve
real runtime controls through unchanged model sources. Independent review,
integration and model acceptance remain pending. Fixed-source FP64 workflows also require explicit JAX-side preparation
where existing configuration is insufficient; deliberate casts, accumulation and
stored weights remain protected unless a numerical change is explicitly chosen.
That preparation is required work, not an available automatic transformation.
Public calls and both current double-precision options keep their behavior.

The preceding [Gather diagnostic checkpoint](poc/gather_counterpart_findings.md) checks
a saved ONNX representation for one unchanged indexing-and-trigonometry example.
It preserves the original JAX call, input values and types, and reference result.
The example’s original graph check still fails; a separate source-structure
check validates the specific representation. This is diagnostic progress, with
runtime behavior and numerical accuracy still unverified.
The diagnostic tooling is committed and independently reviewed. Both supported
stacks pass 249 focused checks, and the fresh complete primary-stack suite
passes 7,326 tests.

Public export calls and both existing double-precision options keep their
meanings. The diagnostic result does not establish general Gather support or
complete model coverage.

The preceding [Where and JIT checkpoint](poc/where_jit_admission_findings.md) adds reviewed
private support for selected conditional calculations and compiled JAX bodies.
It keeps the original JAX call, input types, constants and actual branch inputs.
Both supported stacks pass 2,661 main checks before a separately reviewed
repair of shared x64 state restoration. That repair preserves caller settings
beneath a local override; 89 fresh main checks pass on each stack with
independent integration acceptance. The fresh complete primary-stack suite
passes **7,236 tests**. Code commit `ae820b1a` passes all seven repository hooks.

Some strict floating-point samples still lose the sign of negative zero on the
measured CPUs. Those samples remain failed and cannot pass sample acceptance;
passing ordinary finite or Boolean examples does not change that result. The
original gather example preserves the complete native trace, but still fails
its unchanged graph check. The later diagnostic counterpart is described above;
it adds no runtime or numerical acceptance. Further operator families and the
final transition remain unfinished.

The preceding [Exp and multiplication checkpoint](poc/pointwise_mul_exp_findings.md)
adds reviewed private support for selected exponential and multiplication
calculations, including a shared variable batch size. The source’s input types,
weak typing, casts and explicit broadcasting are preserved. Main checks pass
1,489 cases on each supported stack, with independent integration acceptance.
Selected FP32/FP64 CPU samples pass their fixed criteria; this does not promise
identical results for every value or support on every device.
The complete primary-stack run passes **6,933 tests**. Commit `7c68300b`
passes all seven repository hooks.
Public calls and both existing double-precision options keep their behavior.
Its historical results remain unchanged; broader model coverage is unfinished.

The preceding [Leading-B matrix/bias checkpoint](poc/leading_batch_matrix_bias_findings.md)
adds reviewed private support for selected matrix and bias calculations with a
shared variable batch size. It keeps the five original MatMul, GPT Add and
Linear examples, including their real parameters and runtime inputs. Focused
checks pass on both stacks, and integration on the local branch is independently
accepted. Fresh artifacts also pass the selected FP32 samples at batch sizes two
and five in both CPU settings; independent review accepts these samples.
That checkpoint’s complete primary-stack run passes **6,816 tests**; commit `718f448e` passes
all seven repository hooks. These five examples do not establish support for
every symbolic shape or complete model. Public calls and defaults keep their
existing behavior.

Reviewed private work now covers names, reports, file formats, typed constants,
selected noncomplex and complex loops, local promotion settings and required
finalization failures. The new focused repairs and diagnostic tooling passed
independent review and are committed locally with repository hooks passing.
Private recording and web-file packaging now have reviewed coverage, alongside
native JAX promotion for `where`/`full` and selected Boolean results. These
[bounded changes](poc/option_and_array_creation_findings.md) are committed with
repository hooks passing; public behavior and audited operation admission stay
unchanged. Real scalar-Boolean runtime binding and native `arange`/`linspace`
binding now also have reviewed private coverage. Selected ordinary FP32/FP64
matrix calculations and a shared Dropout cleanup repair are now reviewed and
committed too. That complete run passed **6,183 tests**. The subsequent
[layout and native-binary checkpoint](poc/layout_and_native_binary_findings.md)
has accepted binder/Mul checks. Its initial full run failed ten regressions;
the corrected layout/observer packet passes focused review on both stacks.
The fresh complete primary-stack run passes **6,351 tests**, and the three code
commits passed actual repository hooks. The refreshed inventory
retains every original contract and test anchor; these results do not complete
source coverage.

The committed private shared-batch implementation has passed focused review on both
stacks: one Add/named-function artifact runs with batch sizes two and five, and
validation rejects inconsistent batches. Its complete primary-stack run passes
**6,391 tests**, and the code commit passed actual repository hooks.
The [shared-batch findings](poc/symbolic_batch_and_minmax_findings.md) retain that
evidence. The [preceding checkpoint](poc/minmax_and_corpus_replay_findings.md),
`fc3f9323`, commits the private Min/Max zero-sign correction and repaired
artifact-only corpus tooling. Both stacks pass 323 focused checks; the complete
primary-stack run passes **6,466 tests**, and all actual hooks pass. This
checkpoint changes no default operation admission or numerical criteria.

The [ordinary-matrix findings](poc/ordinary_matrix_admission_findings.md) record
the bounded implementation: static FP32/FP64 matrices reach
audited SOURCE export. Both stacks pass 750 main checks; fresh artifacts pass
the fixed FP32 samples in 48 CPU runs per stack. Old diagnostic records and
exact subnormal failures remain intact. Independent post-run review accepts
these samples. The full primary-stack run passes **6,728 tests**; commit `45881d8d`
passes all seven actual hooks. Broader matrix, symbolic and
model acceptance remain separate work. The [preceding numerical checkpoint](poc/native_trig_and_corpus_numerics_findings.md)
retains its scoped sample results, original criteria and failures.

The [first corpus batch](poc/source_corpus_findings.md) measured 36 source scenarios
per stack, producing 26 checked artifacts while retaining ten failed or pending
outcomes. No runtime or numerical acceptance was claimed. The
[fixed-source review](poc/fixed_source_compatibility_findings.md) classifies all 22
parameterized models so their existing runtime controls and static settings can
be preserved without editing the supplied implementation. The subsequent
private diagnostic pass produces three checked artifacts and 19 pending cases
per stack, with no ONNX Runtime executions or numerical acceptance. Broader operation,
option and full-model acceptance remain unfinished. There is no new public
precision choice.

This document explains how someone exporting models would prepare for and
evaluate the transition. See the [document guide](README.md),
[current behavior](precision_current_state_user_view.md), and
[proposed target behavior](precision_target_state_user_view.md). The
[technical migration plan](precision_migration_plan.md) describes the underlying
work and acceptance criteria.

## What would change, and when

PR #275 has repaired selected cases where explicit FP32 values feed consumers
under the existing double-precision rules. Existing export options, including
`enable_double_precision`, retain their meanings. This is a bounded fix, not
general support for every operation or runtime at every dtype.

The [experiments](poc/README.md) give practical reasons to keep the
transition explicit:

| What was observed | What must happen before relying on the proposed behavior |
| --- | --- |
| After an FP64 input `1e8`, deliberate FP32 arithmetic returns `0` in JAX and the source experiment, while legacy double export returns `1` in FP64. | Explain this expected contract change and compare application results with the intended JAX model as well as the previous artifact. |
| Public matrix export loses a requested FP32 result; the bounded private source lowering now retains it. A separate FP16 square/sum example still misses required intermediate rounding. | Keep the successful matrix cases and the rejected square/sum sample separate. Validate result types and rounding on the actual target. |
| Complete source exports of the selected FP64 variance calculation now succeed inline, in a named function, and in a conditional. | Retain these regression cases while extending normalization coverage and checking the intended deployment. |
| Source input preparation now preserves weak typing: weak scalar `2048` plus explicit FP16 `1` returns FP16 `2048`; a strong input with the same shape/dtype returns FP32 or FP64 `2049`. | Keep the original input properties in comparisons. The source correction has regression coverage on both stacks; public legacy normalization retains its observed behavior. |
| Selected mixed FP64/FP32 calculations survive serialization inside a named function and either branch of a conditional. | Retain these bounded source/legacy checks while completing other scopes, operations, and deployment evidence. |
| Local JAX x64 and matrix-precision settings survive named-function boundaries, and cached conditionals no longer mix successive source and legacy exports. | Keep the contextual and sequential-export regressions in the regular suite as supported cases expand. |

The calling experiment makes omitted/false/true legacy settings,
source-preserving calls, warnings, and conflicts directly comparable. Its
private selector is an experiment interface. Positive results cover selected
CPU samples, including audited named functions and two-branch conditionals.
Broader loops/JIT forms, general deployment diagnostics and operation coverage
remain open beyond the later bounded Where/JIT checkpoint. BF16 model validity alone does
not establish execution support. The private CPU sample check records execution
settings and refuses to accept a failed or untested numerical sample. The known
FP16 square/sum result remains a failure on the recorded runtimes. Passing the
tests includes recognizing that failure; it is not a deployment certificate.

The first bounded artifact batch is recorded; the next work resolves its measured
gaps, continues the frozen scenario plan in bounded batches, completes
unresolved recipes and comparison criteria, and migrates the remaining
operators, scopes and options. Missing comparison criteria remain unverified;
they are not silently replaced with exact agreement. Integer and Boolean
results must remain exact even when floating outputs use a tolerance.
Runtime failures remain visible under their recorded criteria.
The first private parameter packet now carries real named Boolean inputs
through the complete prepared call, including framework Dropout and local
Function defaults. An external whole-call `lax.cond` adapter preserves a bounded
unchanged Python-branch implementation and both runtime choices; static binding
is appropriate only for an intended fixed setting. Broader model and stochastic
contracts remain open. Native array-sequence binding is implemented, but recorded
exact floating-point discrepancies and dynamic `arange` compatibility remain
unresolved; matching graph types does not establish matching runtime results.
The [matrix findings](poc/matrix_and_optimizer_findings.md) likewise retain exact
failures for selected subnormal values. A separate recheck of two original
matrix scenarios now passes schema checking but still fails the existing
structural callback; it establishes no new corpus acceptance. Static NCHW
input/output boundaries and the shared optimizer/naming dependency have passed
independent review. Eight historical min/max signed-zero observations per stack
remain failed under their original criteria; new artifacts from the bounded
lowerer now pass those samples. Subnormal failures and a measured execution-cost
increase remain visible. One shared batch dimension through Add and named
composition has private implementation evidence with unchanged source and
multiple batches. Broader acceptance across 310 symbolic and seven NCHW rows
remains open;
symbolic layouts, normalization and broader model coverage require further work.
Native sine/cosine and the newer Where/JIT binding preserve the complete native
trace of the unchanged gather example. Its x64-on scenario still fails the raw
structural check for a broadcast constant; x64-off retains its configuration
conflict. The later diagnostic counterpart validates a separate representation
without runtime acceptance. Ordinary static matrix
admission now has main evidence. The bounded Leading-B matrix/bias extension
is committed at `718f448e` and preserves the five unchanged MatMul, GPT Add and
NNX Linear sources, including their real weights, bias and configuration.
Their selected FP32 CPU samples pass at batch sizes two and five in 40 runs per
stack, with independent acceptance. Broader dynamic-shape and model requirements
remain open. The Exp/multiplication extension now also has independent
integration acceptance and 1,489 passing main checks per stack. Its own fixed
CPU samples do not replace the preceding five-source observation. Bounded
Where/JIT semantics and actual-main integration are now independently accepted
for the private path. The later Gather diagnostic check preserves that example
without establishing product or runtime support. The committed Typed checkpoint now adds
bounded value-family rules with independent integration acceptance. Named runtime
parameters and unchanged-model preparation are the next shared step after that
checkpoint; complete Gather and broader model acceptance remain open.
An unreviewed operation or an unfinished implementation remains open work; it
cannot be relabeled as a technical limit to complete the migration.

After the full coverage gates pass, normal export follows the new rule.
Release notes and migration examples must explain the changed default and old
arguments. Current public behavior stays unchanged during development.

## Understand the completed call behavior

The completed `to_onnx(...)` has one precision contract. JAX and your model
choose the computation; export preserves it throughout. The final transition
handles the former Boolean argument explicitly:

| Call | Effect after the transition |
| --- | --- |
| Omit `enable_double_precision` | Use your active JAX configuration, preserve typed and weak input properties and the traced computation, and use FP32 for shape-only inputs. |
| Pass `enable_double_precision=False` | Receive a migration error before tracing: remove the argument and set any intended JAX x64 configuration explicitly. |
| Pass `enable_double_precision=True` | Receive a migration error before tracing: enable JAX x64 before constructing values and configure the intended model, parameter, and input dtypes. |

FP32 is only the shape-only input default, not a minimum computation precision.
FP64 scientific workloads remain supported. Export no longer widens arbitrary
FP32 arithmetic after tracing: exact FP32 casts remain exact FP32 casts.
Migration must also cover fixed third-party sources, using external configuration,
preparation or caller wrappers. Previous FP64 workflows need explicit examples
with unchanged sources; enabling x64 alone does not rewrite hardcoded casts. The externally
prepared JAX computation supplies the comparison reference. This compatibility
work is required before the final transition; a general transformation is not
implemented. A requested change must preserve intentional rounding, accumulation
and stored weights unless that change is explicitly part of the prepared model.
Preserving conversion capabilities does not promise unchanged old numbers or
call syntax.

## Fixed configuration and runtime parameters

In the completed API, `input_params` means named runtime inputs everywhere in
the model. A value must not be fixed in one part and changeable in another.
Use ordinary model construction, a closure, or `functools.partial` for fixed
configuration. For example, bind `deterministic=True` with
`partial(model, deterministic=True)` when it should stay fixed, and export that
configured callable. This retains the static calculation without an extra
runtime input.

To change a flag at execution time, declare it through `input_params`. For
Python branches in fixed sources, use a supported caller or framework adapter
that selects the complete computation, or bind the value externally when it
is intentionally static. Moving code into a named function
does not change these rules. Unused runtime inputs may be removed.

Some existing calls specialize their keyword values at the root but promote
them inside a named function. Those calls need an explicit static binding or a
consistently dynamic model. Migration tests must retain their existing model
capabilities, including dropout and other configuration-sensitive examples.
Those examples must use unchanged given model sources. If a previously supported
runtime flag needs source edits under the proposed tracing strategy, that is
unfinished migration work requiring a different supported adapter, not a new
ONNX limitation. A static binding is a replacement only for an intended static
configuration; it does not replace a working runtime control.
The [22-model classification](poc/fixed_source_compatibility_findings.md) now
identifies which controls are dynamic, flag-insensitive or static. It does not
establish numerical or stochastic SOURCE acceptance. This parameter transition
is released for isolated implementation; independent review, integration and
model acceptance remain pending. It is not available through the committed
bounded private profile.
The next shared implementation will start with the unchanged NNX MLP, then Linen
and Function examples. It must preserve real runtime flags through one artifact,
including variable batches where the original model declares them. A successful
input check alone does not mean that construction, export or numerical execution
of the complete model succeeds. Training behavior needs separate checks of the
model's actual keys, repeated calls and state changes. A live Dropout operation
or similar output distributions alone does not prove equivalent behavior; this
is not a promise of identical random streams on every runtime.

## Implementation coverage and reports

The internal `source-static-v1` identifier is retained for the small audited
scope, now including the finite shared-batch Add extension. It is an
implementation checkpoint, not the public feature
scope. The completed transition must cover the existing operators, frameworks,
functions, branches, loops, JIT regions, dynamic shapes, parameters, layouts,
normalization choices, and artifact formats. Each keeps the same precision rule.
Existing complex-number and layout representations remain available through
explicit adapters.

Real limits identify the operation, type, shape, opset, or runtime condition that
causes them. For example, the direct ONNX bitcast representation needs a suitable
opset; a valid BF16 graph does not establish a CPU kernel. Missing migration
work is tracked separately and blocks completion.

The committed `get_precision_report(model_or_path)` reads a versioned report from
model metadata without changing the existing return types. It shows supplied,
normalized, traced, and exported types/properties, tracing settings, warnings,
and the scope of completed checks when export attaches a report. The reviewed
private facade attaches it through the shared export path; normal public behavior
changes only after full migration. Execution and numerical acceptance remain
unverified when only an artifact has been exported.

The separate `validate_source_samples` helper checks concrete inputs and
independent expected outputs on ONNX Runtime CPU with an explicitly chosen
optimization setting. Its report records the artifact, samples, criteria, and
actual runtime settings. `require_sample_acceptance()` rejects failed or
unverified evidence and verifies artifact identity. Passing checks cover those
samples only; a generous output tolerance cannot establish an unobserved
intermediate rounding requirement. The known failing samples remain failures
on the recorded targets.

## Prepare a useful comparison

Keep a record of the export that works for your application:

- Save the model version, weights, export options, JAX configuration, and
  explicitly requested input dtypes. Distinguish shape-only specifications from
  specifications that also name a dtype. Retain JAX input properties such as
  `weak_type` when present, since shape and dtype alone may not describe the
  same tracing input.
- Record the exported model's input and output types and representative outputs.
  Include examples that exercise your chosen precisions and deliberate rounding,
  whether they stay entirely at a lower precision or use higher-precision regions.
- Keep the exported artifact, runtime version, execution provider, hardware, and
  relevant runtime settings. If it is deployed, retain a reference to that
  working deployment for comparison. Settings such as TF32 can affect numerical
  behavior even with FP32 tensors.

This gives you a baseline even when the original JAX model and the existing
export do not calculate every intermediate value at the same precision.

## Choose the precision your model needs

Start with your workload's requirements. Choose the computation, accumulation,
and output types for each part of the model, including deliberate conversions.
A model may stay entirely at a lower precision without an FP32 stage. Other
examples include FP16/BF16 bulk work with selected FP32 stability regions and a
deliberate conversion back, or scientific calculations requiring true FP64.
Configure these choices before export; request FP64 inputs where needed and
enable JAX x64 for FP64 computation.

The typed inputs supplied in the export call also matter: they determine how
JAX applies rules based on an input's dtype. Changing that specification can
change the chosen computation and result types without changing the model's
code. Fixed parameter types and explicit casts still follow their model rules.

As a local example, a region using FP32 may need either exact FP32 rounding or
at least FP32 precision. If you control the model and intend the latter, express
that choice for the region with an operation such as
`jnp.promote_types(x.dtype, jnp.float32)`:

| Floating input | Chosen dtype for at least FP32 |
| --- | --- |
| FP16, BF16, FP32 | FP32 |
| FP64 | FP64 |

This example is not a general minimum precision requirement. It does not require
FP64 inputs for a low-precision model. Keep exact FP32 conversions where that
rounding is intentional.
The exporter cannot infer a minimum requirement from a hardcoded FP32 cast.
Preserving the model's types does not resolve an intention its code never
expressed, or remove fixed precision choices in someone else's model.

This proposal does not automatically choose precision or tune the tradeoff
between speed and accuracy. Converting previously rounded weights to a wider
dtype also cannot recover lost information.

## Choose where the model must run

Support would cover documented combinations of operations, dtypes, ONNX opsets,
and deployment environments. Every model needs support for each operation at
its requested dtype.

The transition would distinguish four questions: is the file valid for its ONNX
opset; does it preserve the intended calculation and rounding; can your runtime
execute it under the specified settings; and are its numerical results
acceptable? Each answer should identify what was validated, what failed, and
what remains unverified, together with the reason. A missing kernel differs
from a calculation that runs but produces unacceptable precision.

Without a declared runtime, an ONNX file can be valid while execution support
remains unverified. For a declared target, a known unsupported case needs a
suitable alternative or a clear error. The same applies to a known numerical
failure on that target, even if execution succeeds. An unverified case needs
validation before it can be described as ready for that deployment.

Configure the actual runtime session separately. Its provider, hardware, and
settings must match the requirements used for validation. Declaring a target
does not configure the session, and choosing an ONNX opset does not select its
execution provider or math settings.

## Check three important differences

Today, shape-only inputs with `enable_double_precision=True` receive an FP64
default. The proposed FP32 fallback applies when an input dtype is unspecified;
it does not impose a minimum precision on the model. Request FP64 with explicitly
typed input specifications and JAX x64 enabled.
An explicit FP64 input request with x64 disabled would produce an actionable
error instead of silently becoming FP32.
An FP64 request inside model code is a different case: JAX may warn and trace
FP32 instead. The proposed diagnostics surface that warning. If FP64 was
intended, enable x64 and repeat the export; preserving the resulting FP32 graph
cannot fulfill the earlier request.

Today, omitting the option or passing `enable_double_precision=False` temporarily
disables JAX x64 during export. The proposed behavior uses your existing JAX
configuration. That can affect values whose types the model leaves unspecified.
Check the configuration used when creating your JAX reference outputs as well
as when exporting.

The existing double-precision policy can also promote FP32 values after JAX has
chosen their types. The proposed behavior removes that later promotion. For
example, a model doing FP32 arithmetic on explicitly typed FP32 inputs would
retain that arithmetic, where the existing export policy may have widened it
to FP64. This difference can affect computation and output types even if the
input specification stays the same. For intended FP64 computation, configure
the model and, where needed, its typed inputs accordingly; an export option
would not rewrite an FP32 calculation into FP64. A model can also choose FP64
work internally while retaining a lower-precision input interface.

## Evaluate the completed transition

When the completed transition becomes available, follow its published migration
instructions for normal export and validation helpers. Compare the JAX model
and the new export using the same intended
dtypes, model settings, and input values. Check output types, representative
predictions, and application behavior in your intended runtime. Use numerical
tolerances appropriate to the computation; identical types do not guarantee
identical floating-point results.

Include cases where deliberate rounding changes a later result. If the model
requires that intermediate boundary, an acceptable final-output tolerance
alone cannot prove it was kept. Check the boundary and its consumer, using
runtime optimization settings representative of deployment. Switching optional
optimization off does not by itself establish the arithmetic used by kernels.

For dtype-dependent model rules, compare the choices produced by the typed
inputs you actually pass to export. Use the same model settings and JAX
configuration for the JAX reference. Confirm the execution session's actual
settings as well as its declared target before drawing a deployment conclusion.

For scalar JAX inputs, also retain whether they are weakly typed. Rebuilding
them solely from shape and dtype can change JAX's promotion choices before
export begins. The private source experiment now
[preserves this property](poc/migration_ux_findings.md#weak-input-properties-preserved-on-the-source-path)
and reports it before and after normalization and during actual tracing.
Its function implementation also distinguishes weak and strong signatures and
retains local JAX precision settings. Cached conditional bodies are checked
across source/legacy export sequences and ordinary JAX execution. When
comparing results, keep the original JAX input properties rather than
reproducing their earlier loss in the reference calculation. This source
correction is private during development and becomes part of the single normal
contract at cutover. `allclose` and browser validation must follow the same
input and reference rules; they cannot hide dtype mismatches by casting results.

If a limit is encountered, the exporter could use a validated alternative that
preserves the intended precision and rounding. A newer opset, another provider,
or a custom operator would require your explicit choice or previously allowed
deployment policy. A custom operator also adds a runtime dependency. The complete
model would need to be checked again after such a change.

An FP64 operation would not silently run in FP32 and then convert its result
back to FP64. If there is no suitable solution, the error should identify where
the problem occurs, the operation and dtypes, the chosen opset/runtime, and
available alternatives. Reducing the model's precision is a separate decision.

Compare separately with your previous exported artifact to understand any
change from its precision behavior. Check that an entirely lower-precision model
gains no unintended FP32 regions. Other examples include checking selective FP32
regions and casts back, or the FP64 calculations a scientific application needs.
Evaluate latency and memory separately from numerical correctness: a faster
result still needs adequate accuracy, and a more accurate result may have an
unacceptable deployment cost. A model loading
successfully is only the first check.

During development, the public API keeps its previous behavior. Retain your
working artifact and environment as a comparison and rollback reference for
the announced transition. After cutover, the exporter applies one contract
throughout; it does not switch individual operations back to old precision
rules. Report unexpected differences against the intended JAX computation.
The migration guidance must distinguish deliberate changes from implementation
defects and demonstrate replacement calls for existing FP64 use cases.
