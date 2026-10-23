# Computation Precision Migration Plan: User View

Status: accepted plan for the complete transition, updated 2026-10-23. The final
normal `to_onnx(...)` call preserves the selected JAX computation through one
consistent rule across all existing functionality and fixed supplied sources,
without a permanent selector or capability loss.

Public behavior and `enable_double_precision` remain unchanged. This unfinished
WIP continues **`d0835d02`**; **`fe214590`** remains the last fully green
checkpoint. The private export path now restores four caller settings across
early argument handling and final naming, reports and saving, with complete
paired checks passing within their recorded scope. The [technical ledger](poc/source_integration_findings.md#current-checkpoint)
keeps exact sources and outcomes. This private repair does not promise arbitrary
JAX configuration restoration or change public flag behavior.

An intermediate WIP does not complete the full inventory, fixed-source,
numerical or public transition work. The interrupted full run remains incomplete;
software resource guards do not prove general stability. SimpleLinear's original
failed structure check remains binding, and public options stay LEGACY until
complete acceptance of the coordinated B–F transition.

The plan below describes the completed experience independently of checkpoints.
Precision choices stay in model configuration, explicit source/preparation and
JAX settings; numerical acceptance stays tied to the actual target and supplied
comparisons. Only demonstrated ONNX or target limits constrain support. Missing
implementation remains work to complete. The implementation plan retains the
full scope; no release or deadline is promised.

## What would change, and when

The transition makes the model's intended arithmetic the ordinary export rule.
Existing public options retain their meanings until the complete migration is
qualified. Selected samples demonstrate why the change needs explicit guidance:

| Observed behavior | Consequence for migration |
| --- | --- |
| Deliberate FP32 arithmetic after an FP64 input can differ from legacy widening. | Compare against the intended JAX model and explain expected changes from the previous artifact. |
| Typed and weak scalar inputs can produce different JAX promotions. | Preserve the actual input properties and tracing configuration. |
| An artifact can load and run while missing an intermediate rounding requirement. | Validate the supplied numerical criterion on the actual runtime target. |
| Original models need their real state, runtime flags and complete computation. | Keep those contracts intact; a smaller example or edited source does not establish model support. |

The [current ledger](poc/source_integration_findings.md#current-checkpoint)
records implemented behavior, finite accepted samples and remaining failures.
Integer and Boolean comparisons remain exact; floating comparisons retain their
stated criteria. Missing criteria or execution evidence remain unverified.
An unfinished implementation remains open work and cannot be treated as a
technical impossibility.

After all coverage and validation gates pass, normal export follows the new
rule. Release notes and migration examples will explain the changed default
and old argument. No release or deadline is assigned.

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
establish numerical or stochastic SOURCE acceptance. The reviewed private binder now
uses real typed named inputs across the whole trace. The earlier parameter-only
ledger recorded progress past the guard without model numerical acceptance.
The original NNX/Linen MLPs now have finite model/state acceptance. The current
normalization work retains all 30 original LayerNorm callables, including their
parameters, dynamic batches, options and declared ONNX versions. Their bounded
SOURCE counterparts pass; the original graph checks remain open. The NNX adapter
preserves tested state changes without rewriting the model. The earlier
six-Function-model exploration stopped at Erfc before producing a complete model
artifact. The new bounded Erfc implementation needs fresh follow-through on
those original models; no later barrier is established by that earlier run.
The original GPT_MLP now has finite FP32 numerical evidence under both
caller-x64 settings. Keep the other original models, numerical requirements and
live controls through the remaining work.
Each declared runtime flag must remain live through one
artifact, including variable batches where the original model declares them. A successful
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
