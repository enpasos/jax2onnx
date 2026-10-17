# Computation Precision Migration Plan: User View

Status: proposed transition, 2026-10-17. PR #275's bounded repair is merged. The
proposed behavior has no public export mode; an internal proof of concept
exercises selected cases. No release or deadline is promised.

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

A first [internal proof of concept](poc/README.md) now exercises selected models
through the actual conversion pipeline. For exact FP32 arithmetic after an FP64
input of `1e8`, its result is `0` in FP32, matching the JAX calculation; the
existing double export produces `1` in FP64. This illustrates a change in the
calculation being preserved, rather than merely a different output type.

The experiment also covers chosen low/mixed-precision examples and deliberate
output conversions. It is limited to simple operations and explicitly typed,
fixed-shape inputs. Its runtime observations cover the recorded CPU samples;
BF16 execution remains unverified. Nested functions and control flow still need
integration work. Existing public calls are unchanged, and this internal
experiment is not a supported export option.

After the documented supported cases have been validated, users could explicitly
choose the new behavior. Its argument names have not been decided. Any later
change to the default would need separate release notes, migration examples,
and compatibility guidance. There is no need to rewrite a working export call
because this proposal exists.

## Prepare a useful comparison

Keep a record of the export that works for your application:

- Save the model version, weights, export options, JAX configuration, and
  explicitly requested input dtypes. Distinguish shape-only specifications from
  specifications that also name a dtype.
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
acceptable? Each answer should identify what was checked, what is known to be
unsupported, and what remains unverified.

Without a declared runtime, an ONNX file can be valid while execution support
remains unverified. For a declared target, a known unsupported case needs a
suitable alternative or a clear error. An unverified case needs validation
before it can be described as ready for that deployment.

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
the model and its typed inputs accordingly; an export option would not rewrite
an FP32 calculation into FP64.

## Evaluate an optional transition

When an explicit choice becomes available, follow its published migration
instructions. Compare the JAX model and the new export using the same intended
dtypes, model settings, and input values. Check output types, representative
predictions, and application behavior in your intended runtime. Use numerical
tolerances appropriate to the computation; identical types do not guarantee
identical floating-point results.

For dtype-dependent model rules, compare the choices produced by the typed
inputs you actually pass to export. Use the same model settings and JAX
configuration for the JAX reference. Confirm the execution session's actual
settings as well as its declared target before drawing a deployment conclusion.

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
result still needs adequate accuracy, and a more
accurate result may have an unacceptable deployment cost. A model loading
successfully is only the first check.

During the transition, the proposal keeps the previous export behavior available.
Keep your working artifact and options while evaluating the alternative. If
results are unsuitable, restore that known working choice and report the
observed difference. Future guidance must explain how to request compatibility,
including whether an omitted option differs from explicitly passing `False`.
Conflicting old and new settings should produce a clear error. Any eventual
removal of compatibility would require a separately announced transition.
