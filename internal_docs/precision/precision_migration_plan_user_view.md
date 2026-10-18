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

The latest local step preserves promotion metadata in stored model state and
configuration when sharing named functions. It also makes optional exporter
optimizations respect the source calculation. The shared private conversion
pipeline and separate CPU sample-validation API are committed; integration with
all existing export options and calculations remains required work.
The private facade's naming, report, file-format and CPU sample integration is
reviewed and committed. The full integrated primary-stack regression passes
5,363 tests. The new inventory/reference modules separately pass 53 focused tests on
each stack and independent review; the full source corpus still needs execution
and reconciliation. The reference infrastructure and complex-constant repair
are committed. Typed complex boundaries, exact integer/Boolean comparisons,
selected loop repairs and a fixed scenario plan have now passed independent
review and are included in that full run. The repairs are committed locally;
the scenario-plan checkpoint accompanies this update.
Complex loops, local promotion settings and the remaining export options and
calculations still need migration. Diagnostic artifacts do not claim audited
lowering. This remains internal work, not a new public precision choice.

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
Loops, JIT regions, general deployment diagnostics, and broader operation
coverage remain open. BF16 model validity alone does
not establish execution support. The private CPU sample check records execution
settings and refuses to accept a failed or untested numerical sample. The known
FP16 square/sum result remains a failure on the recorded runtimes. Passing the
tests includes recognizing that failure; it is not a deployment certificate.

The next work runs the frozen scenario plan in bounded batches, completes
unresolved recipes and comparison criteria, and migrates the remaining
operators, scopes and options. Missing comparison criteria remain unverified;
they are not silently replaced with exact agreement. Integer and Boolean
results must remain exact even when floating outputs use a tolerance.
Runtime failures remain visible under their recorded criteria.
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
FP32 arithmetic after tracing: exact FP32 casts remain exact FP32 casts. A model
with fixed third-party casts may therefore need a deliberate source change to
compute in FP64. Preserving conversion capabilities does not promise unchanged
old numbers or call syntax.

## Implementation coverage and reports

The internal `source-static-v1` profile exercises a small set of static
calculations first. It is an implementation checkpoint, not the public feature
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
