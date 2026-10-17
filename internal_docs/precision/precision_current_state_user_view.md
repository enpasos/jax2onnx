# Computation Precision: Current User Experience

Behavior snapshot: 2026-10-17, after the merged double-consumer repair.

This document explains the precision behavior currently available to JAX model
exporters. See the [technical details](precision_current_state.md) for evidence
and implementation analysis, or the [precision overview](README.md) for the
complete document set. The [target user experience](precision_target_state_user_view.md)
and [migration guide](precision_migration_plan_user_view.md) describe proposed
changes, rather than capabilities already available.
The current repair makes selected double-precision calculations load and run
consistently; the broader precision contract remains a proposal.

## Choosing the Export Setting

`to_onnx` currently exposes `enable_double_precision=False` by default. During
export, this setting temporarily disables JAX x64, even if your application has
already enabled it. Passing `True` temporarily enables x64. The previous JAX
setting is restored after the call, including when export raises an error.
Consequently, enabling x64 elsewhere in your application does not replace this
export argument.

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
