# Computation Precision Target Behavior for Users

Status: proposed behavior, 2026-10-17. The public selection of this behavior has
not been specified.

The JAX model defines how each calculation chooses its computation and
accumulation precision, including deliberate conversions and output dtypes.
The export call supplies the input shapes and dtypes used to trace that model
and selects its ONNX representation. Export preserves the resulting computation
within the capabilities of the selected ONNX version and execution target.

A model may keep calculations in FP16 or BF16, use wider arithmetic for selected
sensitive steps, or require FP64 for scientific calculations. These choices
belong to the model's numerical design.

## What you set in JAX

Use model code and the dtype settings offered by your framework to choose:

- the types used to create and store parameters;
- computation and accumulation types for individual operations or layers;
- explicit conversions, including deliberate rounding;
- the type returned by the model or a particular calculation.

For example, `x.astype(jnp.float32)` requests a conversion to FP32. A layer's
computation or accumulation setting can select its internal precision. A later
conversion can return its result to a lower dtype. The exporter preserves the
behavior these choices express.

Configure JAX before creating arrays and tracing. For FP64 work, enable
`jax_enable_x64` in JAX and request FP64 where the model needs it. This makes
FP64 available; individual dtypes still follow the model's rules. Converting
already rounded parameters to a wider type preserves their stored values but
cannot recover lost information.

## What you set in the ONNX export call

| Export choice | What it controls |
| --- | --- |
| Callable or configured model | The JAX computation to export. |
| Input shapes and dtypes through `inputs` | The inputs JAX uses for tracing and the exported model's input interface. |
| ONNX operator-set version through `opset` | The operator schemas available for representing that computation. |
| Supported implementation options | The ONNX representation, such as a native normalization operator or an explicit sequence of operations, subject to its numerical requirements. |
| Names, layouts, and output format | The interface and packaging of the exported artifact. |

Input dtypes are the connection between the export call and JAX. When model
code chooses a computation dtype from `x.dtype`, a different input specification
can change that choice during tracing. Fixed parameter types and explicit casts
still follow their own model settings. The exported output dtype follows the
JAX computation and its deliberate output conversions.

A model can return results of different dtypes. For example, it can return an
FP64 result from one calculation and an explicitly rounded FP32 result from
another. Both choices belong in the exported output interface.

Typed input specifications retain their requested types. Shape-only
specifications use FP32 as their documented input default. An explicit FP64
request with JAX x64 disabled produces an error explaining the conflict. Export
uses the caller's active JAX configuration and restores that configuration on
return, including after a failed export.

Selecting an ONNX implementation must preserve the model's numerical contract.
For example, choosing a native operator requires suitable computation and
accumulation behavior as well as matching input and output dtypes.

## Example: exactly FP32 or at least FP32

A particular calculation may use FP32 for numerical stability. Its JAX code
can express either an exact FP32 choice or a minimum of FP32:

| Input to the calculation | Exactly FP32 | At least FP32 |
| --- | --- | --- |
| FP16 or BF16 | FP32 | FP32 |
| FP32 | FP32 | FP32 |
| FP64 | FP32, with the requested rounding | FP64 |

An exact conversion is `y = x.astype(jnp.float32)`. An addition `y + y` then
uses FP32, including when the original input was FP64.

This JAX function instead expresses a minimum of FP32 for its local work:

```python
import jax
import jax.numpy as jnp


def centered_values(x):
    compute_dtype = jnp.promote_types(x.dtype, jnp.float32)
    y = x.astype(compute_dtype)
    return y - jnp.mean(y, axis=-1, keepdims=True)


input_spec = jax.ShapeDtypeStruct((8,), jnp.float16)
```

The corresponding export choices are:

| Export argument | Value in this example |
| --- | --- |
| Callable | `centered_values` |
| `inputs` | `[input_spec]` |
| `opset` | `23` |

The input specification selects FP16 inputs. The function's own rule selects
FP32 computation and returns an FP32 result. With JAX x64 enabled and an FP64
input specification, the same rule selects FP64 computation and output. The
input argument instantiates the rule written in JAX; the opset selects the ONNX
operators used to implement it.

If the model should return to the input dtype after the calculation, assign the
centered expression to `result` and return `result.astype(x.dtype)`. This makes
that output conversion a separate, explicit model decision.

A model or layer setting can express the same intention when the framework
offers one. A fixed FP32 conversion retains its exact meaning, including in a
third-party model; expressing a different intention requires a suitable model
setting or source change.

## What you set in the execution environment

Choose the runtime version, execution provider, hardware, and relevant runtime
options when configuring deployment. The ONNX opset identifies the model's
operator rules; runtime settings determine how the execution environment runs
that model. A declared deployment target lets compatibility checks assess those
requirements. The deployed runtime must use the settings covered by validation.

For example, ONNX Runtime's CUDA provider can use [TF32 for certain FP32 matrix
multiplications and convolutions](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#use_tf32),
with reduced multiplication precision while tensor types remain FP32. Its
`use_tf32` setting belongs to the runtime session configuration. Include that
setting when establishing and validating the numerical behavior you require.

## Checking the result

Four separate checks determine whether an export meets your needs:

1. Can the selected ONNX version represent the model's operations, types, and
   shapes?
2. Does the exporter have a correct implementation, including the required
   rounding and accumulation behavior?
3. Can your runtime version and execution provider run that implementation with
   the selected hardware and settings?
4. Do the actual outputs meet the numerical tolerances established for your
   model and inputs?

Results identify what was validated, what is known to be unsupported, and what
remains unverified, together with the relevant versions and settings. Without a
declared execution target, runtime compatibility remains unverified. Matching
dtypes alone does not establish numerical agreement between JAX and a runtime.
Compare representative and difficult inputs against JAX using the intended
model and input configuration.

See the [technical deployment contract](precision_target_state.md#onnx-feasibility-and-deployment-contract)
for implementation requirements.

## When the requested operation is unsupported

A validated alternative sequence of ONNX operations can implement a calculation
if it preserves the required types, rounding, and accumulation behavior. Any
internal conversions retain the contract for other uses of the same value and
for the model's outputs. Evaluate the alternative's memory and execution costs
on the intended workload.

A newer opset, a different provider, a CPU fallback, or a custom operator must
be allowed by your deployment choices. The exporter explains the relevant
requirements before relying on such an alternative.

If no suitable implementation exists for the allowed target, export fails with
the location, operation, dtypes, target, and known alternatives. Changing the
model's requested precision is a separate numerical decision, followed by
validation of the resulting model.

See [handling an unsupported operation](precision_target_state.md#handling-an-unsupported-operation)
for the technical rules. The [precision overview](README.md) links the related
current-state analysis and migration documents.

For a runnable, deliberately limited illustration, see the
[internal experiment](poc/README.md). Its examples make selected consequences
visible; they do not provide a generally available export mode.
