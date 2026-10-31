# Computation Precision Target State

Status: normative semantic design, 2026-10-18. The normal export API applies
this contract consistently across the existing functionality. The
[implementation work plan](precision_implementation_work_plan.md) defines the
coverage and final transition gates. Its bounded source profile is internal
intermediate work; the public transition is not complete and no release is assigned.

For model preparation and expected export behavior, read the
[user view](precision_target_state_user_view.md). All companion documents are
listed in the [precision overview](README.md).

The target is to export a JAX computation while preserving its chosen value
types, accumulation requirements, output types, and deliberate conversions.
Model code and configuration define the precision rules. The input
specifications supplied to export and the active JAX configuration determine
how those rules apply during tracing. The exporter implements the resulting
contract within the capabilities of the selected ONNX version and deployment
target, and identifies unsupported cases.

There is one public precision contract. Callers do not select an operator
profile or combine source and legacy arithmetic policies. Existing conversion
operations, scopes, layouts, parameter handling, and artifact formats retain
their capabilities under this contract. An unaudited lowering or unfinished
migration is open work; a genuine technical limit has a specific explanation
and evidence.

## Where precision is configured

JAX preparation, the export call, and the execution session have distinct roles:

| Configuration place | Choices | Effect |
| --- | --- | --- |
| JAX model and caller configuration | Stored parameter types; operation, accumulation, and output dtypes; explicit conversions; `jax_enable_x64` before array creation and tracing. | Defines the typed computation that JAX traces and the numerical boundaries export must preserve. |
| ONNX export call | Callable; input shapes, dtypes, and precision-relevant tracing properties; ONNX opset; options selecting a semantics-preserving implementation. | Supplies tracing inputs, defines the ONNX input interface, and selects a supported representation of the computation. |
| Runtime session and deployment configuration | Runtime version, execution provider, hardware, math modes, and optimization settings. | Determines how the exported operators execute and which numerical and performance properties require validation. |

This contract accommodates entirely FP16 or BF16 computation, selected wider
regions in a low-precision model, and scientific computation requiring FP64.
Computation, accumulation, and output precision can differ within one
operation. A default for unspecified input types does not impose a computation
precision on the model.

Unchanged, externally supplied JAX sources are an acceptance requirement.
Supported existing workflows must remain available through external model
configuration/preparation, input declarations and caller-side bindings.
Rewriting the supplied model's implementation is not an implicit migration
prerequisite. An external change to a fixed numerical choice must be explicit
and produce a well-defined prepared JAX computation/reference; x64 availability
does not itself change hardcoded FP32 casts. The exporter then follows the same
contract for that prepared computation throughout.

Input dtypes supplied to the export call cross the JAX/ONNX boundary: they
determine the arguments used for JAX tracing and the exported input signature.
They can affect model branches or dtype choices based on `x.dtype`. They do not
globally rewrite stored parameters or explicit casts. Output dtypes follow the
computation JAX traces, including any conversions chosen by the model.

`input_params` declares named runtime inputs. Trace them as real typed JAX
arguments throughout the entire root and its nested scopes. Static configuration
belongs in model construction, a closure, or `functools.partial`; it must not be
inferred from whether a value reaches an ONNX Function. Explicit literals and
omitted nested defaults retain their normal static JAX meaning. Unused runtime
inputs may be pruned, but no parameter may be attached dynamically to an already
specialized root graph. Existing framework adapters or explicit external caller
adapters can express dynamic JAX branches around unchanged source calls with
concrete Boolean values. The runtime input must select the complete matching
computation, covering all its uses. Unsupported direct Python control flow
receives guidance to use an available adapter or bind an intended static
configuration externally. A static binding does not replace a runtime control.
The prepared root must satisfy JAX tracing requirements; a missing adapter for
supported functionality is an implementation gap, not an ONNX impossibility.

The export call's implementation choices must preserve that computation's
contract. Naming, layout, and artifact options can change the interface or
packaging while preserving precision. Declared deployment requirements may
constrain capability checks and implementation selection. Export reports
representation and lowering evidence; separate validation records the selected
execution settings, reference samples, and numerical criteria. Runtime session settings are
configured separately. An ONNX opset declaration does not configure a provider
or its arithmetic settings.

## ONNX feasibility and deployment contract

Preserving tensor dtypes is necessary but insufficient. Assess the selected
implementation against four distinct checks. A validated deployment claim
requires all four to pass; an ONNX export must satisfy the first two and state
which deployment checks remain unverified:

| Check | Required evidence |
| --- | --- |
| ONNX representation | The selected domains and opsets permit the operations, operand types, attributes, shapes, and graph signatures. The serialized model passes schema and applicable shape/type checks. |
| Semantic implementation | The lowering preserves required conversions, rounding boundaries, bit representations, and accumulation requirements, including any documented numerical implementation policy. |
| Runtime execution | The specified runtime version and execution provider can execute the implementation with the required settings and hardware. |
| Numerical acceptance | Representative and difficult inputs satisfy the documented criteria on that execution target. |

Support evidence is scoped to that combination. Record each check as validated,
failed, or unverified, with the reason and its evidence. A failure can mean an
unavailable representation or kernel, an incorrect lowering, or unacceptable
numerical behavior. These are different causes; none alone establishes that
ONNX cannot support the source operation. Passing one check does not imply
the others. Include relevant shape and attribute restrictions; a successful
sample does not establish support for every composition of those operators.

An export without a declared runtime may establish ONNX validity and the
documented lowering contract, but must leave runtime compatibility and numerical
acceptance on that runtime unverified. When a target is declared, known
incompatibilities must select a supported implementation or fail. Unknown
compatibility must remain visible; a deployment-readiness claim requires the
missing execution and numerical validation. Export need not benchmark every
model or invent support information for unavailable hardware.

A known numerical failure on a declared target is a failed gate even when the
model passes the checker and executes. Select a validated alternative or
report the target as failing the contract. Do not mark an observed failure as
merely unverified, and do not report a successful export satisfying that target
without resolving the failure. ONNX validity remains a separate observation;
it does not certify the failed target.

ONNX representation and runtime support can differ. ONNX permits DOUBLE for
[Conv](https://onnx.ai/onnx/operators/onnx__Conv.html), while a particular runtime
may lack that kernel. Standard
[BitCast](https://onnx.ai/onnx/operators/onnx__BitCast.html) starts at opset 26 and
requires equal input and output bit widths; a different-width source bitcast
needs a separately justified representation. Raising the opset alone does not
implement that representation.

Operation semantics can also impose a precision limit. Native
[LayerNormalization](https://onnx.ai/onnx/operators/onnx__LayerNormalization.html)
allows DOUBLE inputs and output, but its statistics type is constrained
separately and default `stash_type=1` specifies FP32 standardization. A lowering
requiring FP64 statistics must choose an implementation that provides them;
matching input and output dtypes cannot establish this requirement.

## Tracing configuration

The caller configures `jax_enable_x64` before creating arrays and tracing the
model. Enabling x64 makes 64-bit types available and can change JAX's defaults
for unspecified types. Explicit model dtypes continue to govern the
computation; enabling x64 does not convert every FP32 operation to FP64. See
[Default dtypes and the X64 flag](https://docs.jax.dev/en/latest/101/default_dtypes.html).

Export starts with the caller's effective JAX configuration and preserves
scoped configuration choices made by the model. The JAX reference used for
numerical validation must use the same configuration at each call boundary,
input dtypes, and precision-relevant tracing properties.

Input normalization preserves JAX properties that affect precision choices,
including `weak_type` on arrays and `ShapeDtypeStruct`. Two inputs with the same
shape and dtype can lead to different promotion and output types when one is
weakly typed. Preserve this property through every normalization step before
tracing; if a supported path cannot do so, reject that input with an actionable
error before invoking the model. Reconstructing an input from shape and dtype
alone must not silently change its promotion behavior.

`weak_type` is a JAX tracing property, not an additional ONNX tensor dtype.
The resulting ONNX graph preserves the concrete types, operations, and
conversions selected by JAX with the original input properties. Export
diagnostics distinguish the supplied and effective tracing properties when
they affect those choices; matching the boundary dtypes alone cannot validate
input normalization.

Explicit input specifications retain their requested dtypes. If an explicit
FP64 input request would be truncated because x64 is disabled, export fails with
an actionable error before tracing instead of silently changing that request.
Arrays already created with reduced precision cannot be reconstructed from an
earlier, higher-precision intention.

Model-internal dtype requests are subject to JAX tracing as well. If JAX warns
that an FP64 conversion was truncated with x64 disabled, surface that warning
with the tracing diagnostics. Preserving the resulting FP32 graph does not
establish that the original FP64 request was fulfilled. The caller must correct
the JAX configuration and trace again when FP64 was intended; export cannot
recover that intention from the typed graph alone.

Shape-only input specifications use a documented FP32 default. Explicitly typed
specifications select the requested supported types, including FP16, BF16, FP32,
and FP64. The input default is distinct from the dtype JAX chooses for an
untyped constant inside the model under its active configuration.

Any deliberately scoped tracing override must restore the previous
configuration on success, failure, and nested export. Function bodies and
control-flow subgraphs retain the effective configuration of their original
call, including a local override surrounding that call. If lowering traces a
body again after that scope has ended, it must reproduce the original tracing
contract. Function specialization and reuse must distinguish any such choices
that change the body, even when input and output dtypes stay the same.
If a configuration cannot be preserved, reject the affected case explicitly.
Export must not introduce a new precision choice merely because a body is
retraced under a different ambient configuration.

## The traced graph defines the type contract

The source contract includes the typed JAX graph, explicit conversions, and
precision-relevant primitive parameters. For a directly representable tensor,
a JAX value's traced dtype determines the dtype of its canonical ONNX value.
Operations after a conversion follow the traced types and operation
requirements. Copying only input/output dtypes is insufficient when the
primitive separately requests an accumulation type or matrix algorithm.

For example:

```python
import jax.numpy as jnp


def add_after_cast(x):
    y = x.astype(jnp.float32)
    return y + y
```

With an FP64 input, the target export is:

```text
FP64 input -> Cast to FP32 -> Add in FP32 -> FP32 output
```

If a later JAX operation promotes a value back to FP64, its traced conversion
is part of the contract as well. The exporter does not infer promotion from the
mere presence of another consumer.

An independent FP64 branch can coexist with this FP32 calculation. If the model
returns both results, its ONNX output signature contains both dtypes. A shared
FP32 value can also feed arithmetic, a direct output, and a same-width bitcast;
each use retains the source value's type and rounding boundary.

JAX may omit a redundant cast from FP32 to FP32. In that case, preserving the
FP32 input and its consumers already preserves the traced computation. There
is no need to recover whether the source code originally contained `astype`.

Some source types need a representation adapter, such as the
[representation of complex values](../../docs/developer_guide/advanced_topics/complex_numbers.md).
Those adapters must specify their storage types, shapes, and conversion rules.
They do not justify arbitrary numeric promotion elsewhere in the graph.

## Example: exact FP32 or at least FP32

An exact FP32 conversion and a request for at least FP32 precision
express different model requirements. `x.astype(jnp.float32)` requests an FP32
value even when `x` is FP64. A sensitive calculation in an FP16/BF16 model may
instead need FP32 as a minimum, with FP64 retained when the same calculation
receives FP64. Model code can express that minimum explicitly:

```python
import jax.numpy as jnp


def centered_values(x):
    compute_dtype = jnp.promote_types(x.dtype, jnp.float32)
    y = x.astype(compute_dtype)
    return y - jnp.mean(y, axis=-1, keepdims=True)
```

For floating inputs, this chooses FP32 for FP16, BF16, and FP32 inputs, and FP64
for FP64 inputs. The dtype observed by this expression must already be the
intended local dtype when JAX traces the model; a later export rewrite cannot
change an earlier decision based on `x.dtype`. See
[JAX promote_types](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.promote_types.html).

The example returns its computation dtype. If the surrounding model requires
the result in the original storage dtype, model code can assign the centered
expression to `result` and explicitly return `result.astype(x.dtype)`. That
rounding is a separate model choice; the exporter must not insert it merely
because most of the model uses FP16 or BF16.

Models may instead expose a computation dtype and a separate accumulation dtype.
An accumulation dtype governs internal work such as a reduction; the result can
then have a different, explicitly chosen output dtype. Configure these choices
before tracing. FP64 computation can be configured inside a model whose external
inputs include lower-precision or integer values. Changing input or parameter
storage alone does not necessarily change a layer's accumulation policy, and
converting existing FP32 weights to FP64 does not recover information already
lost when they were created.

Preserving a literal FP32 cast alone does not fulfill a minimum-precision
intention for FP64 computation: it still narrows the value to FP32. Model owners
must express the minimum through dtype configuration or code before tracing;
the exporter cannot infer that intention from an exact cast. If a third-party
model fixes a calculation to FP32 and offers no suitable configuration, it
remains FP32 under this contract. Changing that choice requires configuring or
modifying the model before tracing.

## Value access and operation lowering

The converter maintains one canonical binding for each JAX variable in its
scope. Lookup returns that binding without changing its numerical type.
Literal materialization and lexical capture may create local IR values, but
must preserve the agreed type and representation.

The [native lowering interface](../../docs/developer_guide/architecture.md)
separates input access through `require_value_for_var`, inspection through
`lookup_value_for_var`, and writable output allocation through
`allocate_value_for_var`. Graph-output binding must refer to an existing
computed value.

Plugins may create temporary values and explicit conversions required by their
ONNX implementation. Shared helpers can construct and cache these conversions,
but the caller must specify the required type and purpose. A helper must not
replace the canonical binding with whichever representation a consumer last
requested. Reuse must respect graph ownership and lexical scope.

| Operation role | Required treatment |
| --- | --- |
| Arithmetic operands | Follow the source operation's type rules and the selected ONNX schema. |
| Comparison operands | Preserve their numeric types; a boolean result does not imply boolean operands. |
| Indices, axes, and conditions | Follow their integer or boolean contracts independently of the data tensor. |
| Numeric cast | Preserve the requested target type and its conversion semantics. |
| Bitcast | Preserve input and output bit representations and the corresponding shape relationship. |
| Wider accumulator | Use separate intermediate values and return to the specified result type at the documented boundary. |
| Matrix product | Preserve operand types, requested result type, and accumulation/algorithm requirements separately. A result type cannot substitute for an algorithm contract. |

Each implementation records which requirements are directly representable,
which use a validated decomposition, and which cannot be met. Widening both
operands before multiplication may be a candidate standard representation, but
does not establish the same low-precision hardware algorithm or execution cost.
Casting only the result cannot restore precision lost in earlier arithmetic.

Bitcasts deserve particular care: widening a numeric value can preserve its
number while changing all of its bits and its word size. An arithmetic
promotion rule is therefore not a valid bitcast rule. See
[JAX bitcast_convert_type](https://docs.jax.dev/en/latest/_autosummary/jax.lax.bitcast_convert_type.html).

## Handling an unsupported operation

Apply the following rules whenever a candidate implementation fails any required
check, including numerical acceptance on a declared target:

1. Select an available, validated standard ONNX implementation that preserves
   the contract. This can be a native operator, a decomposition into supported
   operators, or a documented representation adapter. Mathematical equivalence
   alone is insufficient; required rounding and numerical behavior must survive.
2. Keep implementation conversions local to that operation. Preserve canonical
   bindings, other consumers, direct outputs, and function or loop signatures.
   Widening intermediate arithmetic is permitted only under a documented,
   validated numerical policy that respects explicit precision requirements.
3. If the implementation requires a newer opset, a different provider, CPU
   fallback, or a custom operator, require that choice to be allowed by the
   caller's target configuration. Do not silently raise the opset or introduce a
   new runtime dependency. Revalidate the complete model after changing its
   opset; a custom operator must declare its domain and runtime requirements.
4. Treat a change in the model's requested precision as a separate, explicitly
   chosen numerical transformation. `FP64 -> FP32 -> operation -> FP64` cannot
   stand in for required FP64 computation. Converting back does not recover
   information lost in the intermediate calculation.
5. If no permitted implementation is available, fail the export with an
   actionable diagnostic. Identify the model or graph location, operation,
   requested types, opset, applicable runtime target, failed requirement, and
   known alternatives. Do not return a known invalid model as a successful
   export or certify a numerically failing implementation for its declared target.

Diagnostics identify the phase as well as the failed requirement: option or
input validation, JAX tracing, ONNX representation/lowering, runtime execution,
or numerical acceptance. Trace warnings are preserved alongside the observed
types. A prototype exclusion is identified as such, without being classified
as an ONNX limitation. Unknown runtime support remains unverified; deployment
readiness requires the missing evidence.

These rules define a bounded implementation repertoire, not a requirement to
invent an arbitrary emulation for every JAX operation. An alternative may cost
more memory or execution time. Document relevant costs separately from
correctness and evaluate them on the intended workload. Resource limits in a
declared target also constrain the choice of implementation.

One unsupported operation can prevent the whole model from meeting its
contract. Apply the checks inside functions and control-flow bodies as well as
the outer graph. A branch not taken by the validation samples still needs a
valid implementation if it remains in the exported model.

## Constants and graph boundaries

Constant payloads and their IR metadata must describe the same dtype. The
source graph determines whether a stored FP32 constant stays FP32 or passes
through a conversion before an FP64 consumer. Constant folding may evaluate
that conversion when it preserves the required value and representation.

Post-processing must not globally promote constants after their consumers have
been typed. Optimizers must preserve observable conversions, including the
rounding in FP64 to FP32 to FP64; that sequence is not generally an identity.

The same contract applies to graph inputs and outputs, layout transposes,
function arguments and results, captured values, branch results, and loop
state. A layout change changes axis order, not precision. A branch or loop
signature must agree with its body values without acquiring a context-specific
precision interpretation.

Function reuse also preserves precision-relevant properties of captured model
state and static arguments. Equal shapes, dtypes, and stored numbers do not
establish equivalent computations when JAX promotion metadata such as
`weak_type` differs. Cache keys retain that metadata through the supported
containers and pytrees; truly equivalent states can still share a definition.
Output-signature equality alone cannot detect changed internal arithmetic.

## Numerical correctness

Type preservation does not promise bitwise equality between JAX and every ONNX
runtime. Reduction ordering, kernel choices, and deliberate implementation
details can produce numerical differences. Validation must establish suitable
bounds for the actual operation, dtype, and runtime.

Distinguish differences allowed by those bounds from violations of a required
rounding boundary. Computing a product in a wider type and rounding only after
its reduction can differ from rounding the product before summing. A wider
accumulator does not by itself authorize removing that earlier rounding.
If a contract requires that intermediate boundary, a loose final-output
tolerance cannot waive it. Test the intermediate and the downstream use that
depends on it; merely returning the rounded value through another branch is
insufficient. Exact equality can be appropriate for a chosen rounding witness
without promising bitwise agreement for arbitrary reductions.
Check sensitive intermediates and their consumers, as well as final outputs,
with the runtime optimization settings being claimed. Disabling optional graph
optimizations does not by itself prove the arithmetic used by runtime kernels.

For matrix products, preserve operand, result, and accumulation requirements
separately. An FP32 output alone is not evidence of FP32 accumulation. An
implementation that widens inputs before multiplication needs its own numerical
and cost assessment; it cannot be assumed to retain a native low-precision
matrix algorithm. See [JAX matmul precision](https://docs.jax.dev/en/latest/201/precision.html).

Runtime settings are part of this evidence. For example, ONNX Runtime's
[CUDA `use_tf32` option](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#use_tf32)
allows reduced-precision multiplication for some FP32 matrix products and
convolutions while keeping FP32 tensor types. Required multiplication precision
must therefore be translated into supported runtime settings or reported as
unverified or unsupported. Tensor metadata alone cannot enforce that setting.
Record provider options, relevant hardware, and optimization settings with
validation results; runtime fusion may change the executed computation.

An explicitly documented lowering may use a wider internal accumulator while
preserving source input types, output types, and required conversion boundaries.
That freedom must be justified for the particular source operation. It cannot
override an explicit accumulation type, required rounding, or algorithm merely
because a wider calculation improves a sample's accuracy.
The [LayerNorm accuracy policy](../../docs/user_guide/known_limitations.md#normalization-export-policy)
illustrates why such choices need operation-specific evidence. Its bounds do
not establish a global guarantee for all normalization or mixed-precision
graphs, or automatically authorize that lowering under the source contract.

Structural checks must cover dtypes, casts, signatures, and constant payloads.
Numerical checks must include rounding-sensitive inputs and appropriate NaN,
infinity, and signed-zero behavior. A model loading successfully is necessary
but insufficient evidence of correct precision. Tests must also verify that
unsupported combinations fail with the intended diagnostic instead of silently
changing precision or overstating deployment readiness.

## Related reading

- [Current state](precision_current_state.md)
- [Migration plan](precision_migration_plan.md)
- [Internal experiments](poc/README.md), with their own coverage and
  validation limits; it does not establish implementation of this full design.
