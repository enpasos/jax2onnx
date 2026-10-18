# Computation Precision: Current User Experience

Public behavior snapshot: `main` at `fae17822`, 2026-10-17, after the merged
double-consumer repair. A newer local branch experiment is identified separately;
it changes private converter code, while public options and defaults stay the same.

This document explains the precision behavior currently available to JAX model
exporters. See the [technical details](precision_current_state.md) for evidence
and implementation analysis, or the [precision overview](README.md) for the
complete document set. The [target user experience](precision_target_state_user_view.md)
and [migration guide](precision_migration_plan_user_view.md) describe proposed
changes, rather than capabilities already available.
The current repair makes selected double-precision calculations load and run
consistently; the broader precision contract remains a proposal.

On the local migration branch, reviewed private changes preserve typed constants,
reports, names, file formats and selected loop calculations. New focused checks
also cover local promotion settings, complex loop representations and failures
during required finalization. Public calls still use the behavior described
below; these are internal migration steps.

The [first corpus batch](poc/source_corpus_findings.md) checked 36 source scenarios
per stack and produced 26 valid diagnostic artifacts. Four export gaps, five
pending cases and one x64 configuration conflict remain visible. No runtime or
numerical acceptance was claimed. A separate
[fixed-source review](poc/fixed_source_compatibility_findings.md) identifies the
runtime controls and static configurations that all 22 parameterized examples
must retain with their implementations unchanged.

Private recording and single-file web packaging now have regression coverage,
as do native JAX promotion for `where`/`full` and selected Boolean results. The
[findings](poc/option_and_array_creation_findings.md) distinguish these successes
from remaining BF16 execution and floating-point comparison failures. They do
not establish browser support or broaden the audited operation profile.

Private migration checks now also carry real named Boolean arguments through
the complete prepared call and framework Dropout, while preserving local defaults
and static settings. Native `arange`/`linspace` binding retains JAX's chosen
calculation. Selected exact sequence comparisons, BF16 execution and dynamic
`arange` compatibility remain open. The all-22 model diagnostic pass yields
three checked artifacts and 19 pending cases per stack, with no runtime or
numerical acceptance.

Private diagnostics now also cover selected FP32/FP64 matrix calculations, and
a shared Dropout cleanup defect is repaired. The
[matrix findings](poc/matrix_and_optimizer_findings.md) retain exact failures for
selected very small values; matching graph types alone does not establish
matching results. Static NCHW input/output layout handling is planned next.
These changes do not broaden the public precision interface or establish
complete matrix or model support.

The latest complete run passes **6,183 tests**. These reviewed changes are
committed locally and passed repository hooks. The refreshed inventory retains
all original contracts and test anchors. The
[work plan](precision_implementation_work_plan.md) tracks the remaining operators,
options and full-model validation. The migration is not complete.

## Choosing the Export Setting

`to_onnx` currently exposes `enable_double_precision=False` by default. During
export, this setting temporarily disables JAX x64, even if your application has
already enabled it. Passing `True` temporarily enables x64. The previous JAX
setting is restored after the call, including when export raises an error.
Consequently, enabling x64 elsewhere in your application does not replace this
export argument.

| Existing call | JAX setting during export | Input default when only a shape is supplied |
| --- | --- | --- |
| Argument omitted | x64 disabled | FP32 |
| `enable_double_precision=False` | x64 disabled | FP32 |
| `enable_double_precision=True` | x64 enabled | FP64 |

The argument also selects the input dtype when you provide only a shape. For
example, `to_onnx(fn, [(4,)])` uses a float32 input, while
`to_onnx(fn, [(4,)], enable_double_precision=True)` uses a float64 input. If the
input dtype matters, declare it explicitly with `jax.ShapeDtypeStruct` instead
of relying on this default.

Double export promotes many floating values, but it does not make every value
float64. Explicit conversions, selected array constructors, and some operations
retain lower precision. Float16 and bfloat16 also have their own supported
cases. The argument therefore does not currently offer one uniform rule for
models that intentionally mix floating types.

An explicit FP64 input does not override the disabled x64 setting on the old
omitted/false route. In the tested case, JAX truncates an FP64 input
specification: JAX 0.10.2 does so silently, while JAX 0.11.2 warns. A requested
FP64 cast inside model code warns and becomes FP32 on both tested stacks.
Configure the JAX reference consistently when comparing with such exports.

Scalar inputs can also carry JAX's `weak_type` property, which affects how their
type combines with another operand. For example, `jnp.asarray(2048.0)` creates
such a scalar. Public legacy input preparation loses that property. Matching
shape and dtype alone therefore may not reproduce the intended JAX promotion.
The private source experiment now preserves it: in the
[tested addition](poc/migration_ux_findings.md#weak-input-properties-preserved-on-the-source-path),
a weak scalar plus explicitly FP16 `1` returns FP16 `2048`, while a strong
input with the same shape and dtype returns FP32 or FP64 `2049`. Both source
routes have regression coverage on both supported JAX stacks. That correction
does not alter public legacy normalization.

The [local integration](poc/source_integration_findings.md) also preserves
selected mixed-type calculations through real named functions and two-branch
conditionals, including constants and both runtime branch choices. Its private
function variant handles mixed and structured results and distinguishes weak
and strong input signatures. It also retains local JAX x64 and matrix-precision
settings across a named function. Sequential source and legacy exports now
have regression checks around cached conditionals and ordinary JAX execution.
These selected serialized CPU cases are an intermediate step toward the complete
migration.
The local follow-up also preserves weak typing in stored model parameters and
nested configuration values when reusing a named function. Optional exporter
optimizations now respect the selected source calculation too. These repairs
have separate regression coverage; they do not certify runtime precision.
No public source option is available; loops, JIT regions, broader operations,
and deployment validation remain open.

A further FP32 reciprocal example agrees with JAX under the tested ORT sessions
with optimization off, but differs when optimization is on. A captured FP16
addition also loses its intended intermediate rounding in both profiles.
The [follow-up findings](poc/source_integration_findings.md#follow-up-review-on-2026-10-18)
record the exact cases. Disabling runtime optimization is therefore not a
general precision remedy.

## Explicit Conversions Can Depend on the Input Type

Consider this small export:

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

For this function, focused checks at the stated snapshot produced:

| Explicit input specification | Exported output dtype |
| --- | --- |
| float64 | float32 |
| float32 | float64 |

In the second case, JAX can remove the redundant float32-to-float32 conversion
before export, and double export subsequently promotes the value. The same
source `astype` expression therefore does not guarantee the same exported
output dtype for both input specifications.

The previous loading failure for `return -x.astype(jnp.float32)` is repaired.
With a float64 input and double export enabled, the value is first rounded to
float32, then converted back to float64 for negation. The exported result is
float64. Returning that rounded value directly as another output still returns
float32; sharing it with the negation does not change that direct output.

Focused CPU tests also cover selected arithmetic, unary, sum, concatenation,
function, and output-layout cases. This repair preserves the existing double
export policy. It does not guarantee that every later operation follows the
source model's float32 arithmetic: rounding to float32 and then doing arithmetic
in float64 can produce a different answer from continuing entirely in float32.

## Precision and Deployment Need Separate Checks

An exported model's input and output dtypes do not fully describe its internal
arithmetic. Some normalization calculations deliberately use a wider
accumulator while returning the original output dtype. Such choices can improve
accuracy on tested inputs without promising identical results across JAX and
ONNX runtimes.

The [numerical experiment](poc/numerical_contract_findings.md) makes three
different situations visible:

| Example | Observed behavior | Meaning for an export user |
| --- | --- | --- |
| FP16 matrix operands with a requested FP32 result | JAX returns FP32 `2049`; public legacy export returns FP16 `2048`. The private source path now returns FP32 `2049` for the tested matrix cases. | A bounded private repair exists; it has no public export switch and makes no general matrix-algorithm or performance promise. |
| FP16 square followed by a sum | JAX rounds the squares to zero and sums to zero. The tested ONNX Runtime CPU sessions expose zero squares but return a nonzero sum. | Matching input/output types and successful execution do not establish required intermediate rounding. |
| FP32 normalization with a selected FP64 variance sum | The private source path now exports the complete tested calculation, including inside a named function or conditional. | The selected wider sum retains the small contributions in these samples; broader normalization support remains to be established. |

The matrix comparison distinguishes the public exporter from the private
experiment. The square/sum case uses the private path and does not establish
the behavior of every public FP16 export. Normalization now has complete source
exports as well as the original separately constructed feasibility control.
All are bounded CPU observations on the versions recorded in the findings.

A private sample check now records the CPU execution settings and distinguishes
an acceptable result, a numerical mismatch, and an untested sample. It refuses
to mark the known square/sum failure as accepted. Exporting the file still
succeeds; the check does not repair that runtime result or certify other inputs.

The tested square/sum example can pass after explicitly changing its model
arithmetic. That change remains a candidate requiring its own validation; it
is not an automatic repair available from an export switch. Execution cost
and applicability to other inputs still need assessment.

Check the exported model on the runtime and execution provider you will deploy.
Small direct ONNX models tested with ONNX 1.23.2 and ONNX Runtime 1.31.0 CPU
illustrate the limits:

| Operation and dtype | Observed result |
| --- | --- |
| float64 matrix multiplication | Loaded and ran |
| float64 convolution | Model was valid, but loading failed: implementation unavailable |
| bfloat16 addition or negation | Models were valid, but loading failed: implementation unavailable |

These checks cover the tested operations, not complete exported models or other
devices. Bit reinterpretation also depends on the selected ONNX operator set:
a minimal float32-to-int32 example failed at the current default, opset 23,
and ran at opset 26. See the [runtime evidence](precision_current_state.md#onnx-feasibility-and-runtime-evidence)
for the exact scope. Web export packaging likewise does not certify browser
execution. Model validity, successful loading, and output accuracy are separate
checks.

Use representative inputs, including numerically difficult cases relevant to
your model. Compare against JAX running with the intended settings, and assess
the error against your task's accuracy requirements. A dtype match or one
successful example cannot replace that comparison. Existing numerical tests
cover particular configurations rather than every model and deployment.

Precision choices apply to each operation's computation, accumulation, output,
and deliberate conversions. A model may intentionally retain float16 or
bfloat16 throughout. Other examples include selected float32 steps for
stability or scientific calculations requiring float64. There is no universal
float32 minimum.

One concrete choice is exactly float32 versus at least float32:
`astype(jnp.float32)` expresses the former. Keeping that conversion unchanged
retains its rounding from float64 to float32, even when a later operation uses
float64. The conversion alone cannot communicate a minimum-precision request.

Express the intended computation types in model configuration or model code
before export. The proposed
[FP32 exact-or-minimum example](precision_target_state_user_view.md#example-exactly-fp32-or-at-least-fp32)
illustrates one such choice; it does not remove the current export limitations.
For float64 work, also configure inputs and parameter creation accordingly.
Changing existing float32 parameters to float64 cannot recover precision
already lost when they were created.
