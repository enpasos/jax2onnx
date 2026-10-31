# Computation precision: user migration plan

The planned change makes the selected JAX computation the rule for ordinary
export. Your model chooses its computation, accumulation, rounding and output
types; the exporter preserves those choices throughout. This is the accepted
transition plan, updated 2026-10-31, **not behavior already available publicly**.

The documented checkpoint still has the old `to_onnx` behavior and
`enable_double_precision` argument. The unmerged PR may move to the new contract
as implementation progresses; it need not preserve that old default until all
work is finished. Any old paths remain only for specific CI or meaningful
reference comparisons and are removed when no longer needed. Complete
implementation, compatibility, scientific validation and independent acceptance
are required before merge. The result has one public precision contract, with
no permanent source/legacy selector or per-operation fallback. Existing operations, models, dynamic inputs,
state, layouts and artifact formats remain part of the required migration.
The [current guide](precision_current_state_user_view.md) explains today's
behavior; the [target guide](precision_target_state_user_view.md) explains the
completed design. No release date is assigned.

## What will change

| Call | Behavior after the completed transition |
| --- | --- |
| Omit `enable_double_precision` | Preserve the caller's JAX settings, input typing properties and traced computation. Shape-only inputs default to FP32. |
| Pass `enable_double_precision=False` | Receive a migration error before tracing. Remove the argument and choose any intended JAX x64 setting explicitly. |
| Pass `enable_double_precision=True` | Receive a migration error before tracing. Enable JAX x64 before constructing values and configure the required model, parameter and input types. |

FP32 is the default for an unspecified input dtype, not a minimum computation
precision. Entirely FP16/BF16 models, selected FP32 stability regions and
scientific FP64 work all remain required. The final exporter will stop widening
arbitrary FP32 arithmetic after JAX has chosen its types. Consequently, preserving
conversion capabilities does not promise the old arithmetic or unchanged call
syntax.

Three differences deserve a deliberate comparison: shape-only inputs will no
longer become FP64 because of the export flag; export will preserve the caller's
JAX settings; and exact source conversions will no longer be followed by blanket
legacy promotion. Preserving JAX weak typing can also correct promotion behavior
even when input shape and dtype stay the same.

## Prepare the intended JAX computation

Choose model and layer dtypes before export. For true FP64 computation, enable
JAX x64 before creating arrays, configure the model's computation and
accumulation settings, and supply typed FP64 inputs where appropriate. FP64 work
can also occur inside a model with a lower-precision input interface.

An exact FP32 conversion stays exact. If you control the numerical design and
want a local minimum of FP32, a rule such as
`jnp.promote_types(x.dtype, jnp.float32)` retains FP64 inputs while widening
FP16/BF16 inputs to FP32. The [worked example](precision_target_state_user_view.md#example-exactly-fp32-or-at-least-fp32)
shows the difference. Export will not infer that intention or recover precision
already lost from stored weights.

Fixed and third-party JAX sources are part of the compatibility requirement.
The migration must provide appropriate external model configuration, preparation
or caller wrappers without requiring edits to those supplied implementations.
Previous FP64 workflows need runnable replacement examples. If a fixed FP32
choice must intentionally change, an explicit supported preparation step must
produce a well-defined JAX computation and reference first. No general automatic
transformation API is available or assumed. A missing adapter remains unfinished
migration work.

## Separate fixed settings from runtime controls

In the completed interface, `input_params` will mean named runtime inputs
throughout the complete model. Model construction, a closure or
`functools.partial` binds a fixed setting. For example,
`partial(model, deterministic=True)` is appropriate when that choice should
remain fixed for every execution.

A runtime flag must select the complete corresponding computation, including
inside functions, branches and loops. If an unchanged source uses a Python
branch, a supported external or framework adapter must express its dynamic JAX
control flow. Static binding does not replace a previously working runtime
control. Defaults and explicit literal overrides keep their normal local meaning;
unused inputs may be removed.

All 22 original parameterized metadata models and handwritten static cases
remain required. Positive-rate Dropout controls must stay live; the original
zero-rate and static configurations must retain their meaning. Training behavior
also needs actual state and RNG checks, repeated calls, feedback and reset/replay.
A live flag or similar output distributions alone do not prove equivalent
stochastic behavior. Small examples do not replace the original full models.

## Keep a useful comparison baseline

Before the announced transition, retain:

- The original model and weights, configuration, typed inputs and relevant JAX
  properties such as weak typing, plus the settings used to compute references.
- A working exported artifact and its options, output structure/dtypes and
  representative predictions.
- Runtime version, provider, hardware and optimization/math settings, together
  with application-specific numerical criteria.

Compare the new export with the intended JAX computation, and separately with
the previous artifact to understand deliberate changes. Do not silently cast
outputs or change reference settings to make a comparison pass. The migration
also covers `allclose` and browser validation under the same rule; browser
acceptance requires an independent JAX reference.

Include inputs that expose important rounding, accumulation and signed-zero
behavior. A broad final-output tolerance cannot prove a required intermediate
rounding step. Check that intermediate and a later consumer. Valid all-zero
answers are allowed; positive and negative zero need explicit sign checks when
they affect the computation. NaN masks and signed infinities require their own
checks where applicable, without imposing universal NaN-payload equality.

## Understand the validation report

The completed design distinguishes four questions: is the ONNX representation
valid; does the lowering preserve the calculation; can the selected runtime
execute it; and do the outputs meet the model's numerical criteria? Reports
identify validated, failed and unverified checks separately. Export alone leaves
runtime and numerical acceptance unverified.

The private implementation already has a metadata reader and a CPU sample
validation helper. Their accepted evidence covers the bound artifact, inputs,
references, criteria and actual runtime settings. They do not make public SOURCE
export available or certify untested models. Missing references, missing criteria
or omitted numerical execution can never be acceptance.

A known runtime or numerical failure needs a validated permitted alternative
or a clear failure diagnostic. A different opset, provider or custom operator
requires an allowed deployment choice and complete-model revalidation. Running
required FP64 work in FP32 and converting the answer back is not a silent repair.
Changing model precision is a separate numerical decision.

## When the transition is ready

Completion requires all existing functionality and original models, independently
reviewed scientific evidence, supported Python/JAX versions and minimum runtimes,
complete browser/artifact checks, and the full accepted suite on the exact final
commit. Routine CI has a separate 20-minute limit for each Linux STANDARD job;
expensive validation runs in dedicated lanes without losing its final obligations.
The [technical work plan](precision_implementation_work_plan.md) and
[CI gate](precision_ci_gate.md) define those requirements.

The current decision keeps official JAX reference contradictions failed until
the required results agree. Changing that rule would need a separate explicit
decision. The narrow BF16 exception permits
reviewed development publication, not final scientific acceptance. Custom
JAX/jaxlib/XLA/LLVM changes are not authorized.

Release notes will document the coordinated argument/default changes and provide
runnable migration examples. Keep the previous artifact and environment for
comparison or rollback. After transition, the new exporter applies one precision
contract throughout; it will not switch individual calculations back to the old
policy.
