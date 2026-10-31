# Computation precision: what works today

The public exporter still uses its existing precision behavior. The planned
source-preserving behavior is being developed privately and is **not a public
export mode**. This page describes the current interface and its limitations,
as of 2026-10-31. The [target user guide](precision_target_state_user_view.md)
describes the intended result; the [migration guide](precision_migration_plan_user_view.md)
explains how the transition will work. The unmerged PR may change its ordinary
export path while implementation continues. Complete functionality and validation
are required before merge; this page does not announce a release.

## Choosing the current export setting

`to_onnx` currently exposes `enable_double_precision=False` by default. In an
ordinary call, false temporarily disables JAX x64 and true enables it; the
previous setting is restored afterward, including on failure. A surrounding
local JAX configuration scope can take precedence, so use the same effective
settings for the JAX result you compare against.

| Existing call | Ordinary x64 setting during export | Default for shape-only inputs |
| --- | --- | --- |
| Argument omitted | Disabled | FP32 |
| `enable_double_precision=False` | Disabled | FP32 |
| `enable_double_precision=True` | Enabled | FP64 |

For example, `to_onnx(fn, [(4,)])` uses an FP32 input, while the same call with
`enable_double_precision=True` uses FP64. Declare an explicit
`jax.ShapeDtypeStruct` when the input dtype matters. An FP64 declaration alone
does not override disabled x64: JAX may narrow it during tracing.

The flag also affects constants and lowered arithmetic. Double export widens
many floating values, but explicit conversions, selected constructors and
low-precision paths retain exceptions. It does not provide a uniform rule for
models that intentionally mix precisions. Configure the comparison model
consistently and check the actual exported result.

JAX scalars can carry `weak_type`, a property that changes how they combine
with explicitly typed operands. Current public input preparation can lose that
property. Equal shape and dtype therefore do not always imply the same traced
calculation. The private migration corrects this, but that correction has not
changed the public interface.

## A small example of the current behavior

```python
import jax
import jax.numpy as jnp
from jax2onnx import to_onnx


def cast_only(x):
    return x.astype(jnp.float32)


model = to_onnx(
    cast_only,
    [jax.ShapeDtypeStruct((4,), jnp.float64)],
    enable_double_precision=True,
)
```

This FP64-to-FP32 conversion retains an FP32 output in the focused regressions.
If the input specification is changed to FP32, JAX can remove the redundant
cast, and the existing double-export policy can produce an FP64 output instead.
The source expression alone does not determine the public export's arithmetic.

The earlier loading failure for `-x.astype(jnp.float32)` is repaired for the
covered paths. With an FP64 input and double export, the value is rounded to
FP32, widened, and then negated in FP64. Returning the same rounded value
separately still gives an FP32 output. The repair makes these models valid
without promising that every later operation runs at the source model's FP32
precision.

For a numerical example, consider:

```python
def rounding_sensitive(x):
    y = x.astype(jnp.float32)
    return (y + 1) - y
```

With FP64 input `1e8`, the intended FP32 arithmetic returns zero in JAX. The
tested current double export returns one because the later arithmetic is wider.
This is why the migration needs a coherent rule throughout the model.

## Validate the calculation on its deployment target

A valid ONNX file, a model that loads, and an accurate result are separate
achievements. Input/output dtypes do not prove the intermediate rounding or
accumulation used by the runtime. Some retained examples load successfully but
lose a required FP16 rounding step; others differ only when runtime optimization
is enabled. Disabling optimization is not a general fix.

Use the intended JAX model, weights, inputs and configuration as the reference.
Check output structure, shapes and dtypes before numerical values. Include
finite nonzero inputs, zeros, infinities, NaNs, subnormals and rounding boundaries
where they matter to the operation. An all-zero answer can be correct.

`allclose` and ordinary exact numerical equality treat positive and negative
zero as equal. If the model depends on the sign, check it explicitly and include
a later operation that reveals the difference, such as reciprocal producing
positive or negative infinity. Check NaN locations when required; comparing
all NaN payload bits is not a universal numerical requirement.

Record the runtime version, provider, hardware and optimization/math settings.
CPU success does not certify a browser or accelerator. Web packaging does not
establish browser execution. The [dependency table](../../docs/about/dependencies.md#onnx-runtime-kernel-gaps)
records specific kernel and behavior gaps and distinguishes current public
handling from private migration work. A result on the locked runtime does not
establish support on the minimum supported runtime.

## Local branch snapshot

The private path, called SOURCE in the technical documents, preserves selected
mixed-type calculations, weak inputs, constants and local precision settings
through real functions and control flow. It also has reviewed finite complex
broadcast, conjugation, layout and selection support. Completed-export ownership
has been repaired without clearing JAX caches globally. Unsupported NNX mutable
static payloads are rejected before the original model runs or changes; general
state migration remains unfinished.

The main numerical blockers are still visible. Default-runtime extrema
comparisons remain failed for subnormal cases. Some official JAX eager and JIT
results disagree on zero signs, and the official BF16 TopK counterexample remains
open. A configured runtime profile or a passing diagnostic does not settle those
broader failures. Only official supported dependencies are currently permitted;
custom compiler builds need separate explicit approval.

At the documented 2026-10-31 published checkpoint, focused regression lanes
passed, but the complete scientific gate and CI timing qualification across the
supported Python/JAX versions were unfinished. The [technical current state](precision_current_state.md)
explains these boundaries. Test counts do not measure how much of the migration
is complete.

## What to do now

Keep using the documented public export interface, and retain a working artifact
and its environment as a comparison. Make model precision choices explicit where
the framework supports them. For FP64 work, configure JAX before creating values
and use the intended model, parameter and input types. Widening already rounded
weights cannot recover their original precision.

Keep an intentional FP32 cast when its rounding matters. If a calculation should
use *at least* FP32 instead, express that rule through model configuration or a
supported preparation step; the exporter cannot infer it from a hardcoded cast.
The [target example](precision_target_state_user_view.md#example-exactly-fp32-or-at-least-fp32)
explains the distinction.

The final transition must preserve existing functionality, including unchanged
third-party sources, dynamic controls and scientific FP64 workflows. The
documented checkpoint has no public API change to migrate, and no release date
is assigned.
