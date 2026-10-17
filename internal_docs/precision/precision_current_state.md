# Computation Precision Current State

Date: 2026-10-17. Source snapshot:
[`fae17822e0114e6c00840b8f78adedb7ec36f74e`](https://github.com/enpasos/jax2onnx/commit/fae17822e0114e6c00840b8f78adedb7ec36f74e),
which includes the merged consumer-adaptation fix from PR #275 and the native
typed lowering changes from PR #274.

For observable behavior and export choices, read the
[user view](precision_current_state_user_view.md). All companion documents are
listed in the [precision overview](README.md).

Computation precision is governed by overlapping tracing, lowering, and
postprocessing rules. The double-precision setting promotes many values, while
explicit casts and selected scopes preserve lower precision. The resulting
behavior is not one uniform dtype contract. The proposed
[target state](precision_target_state.md) and
[migration plan](precision_migration_plan.md) describe how to resolve this.
The merged fix repairs selected consumer type mismatches within the existing
policy; it does not implement that broader target.
The broader design goal is to respect the computation, accumulation, and output
dtypes chosen for each operation, together with deliberate conversions. A model
may intentionally retain FP16 or BF16 throughout. Selected FP32 stability steps
and scientific or physics calculations requiring FP64 are further examples;
the goal sets no universal FP32 minimum.

## One Argument Controls Several Decisions

The public `enable_double_precision` argument defaults to `False`. Its current
meaning combines tracing configuration, defaults for unspecified dtypes, IR
type selection, constant conversion, and final export postprocessing.

The public facade uses `_temporary_x64`, an alias of
`conversion_api._force_jax_x64`. This context sets `jax_enable_x64` to the
argument's Boolean value and restores the previous setting afterward. Thus
`False` actively disables x64 during tracing, including when the caller had
enabled it. `True` permits JAX to trace FP64 computations; enabling x64 alone
does not make an explicitly FP32 JAX calculation FP64.

`_as_sds_list` assigns FP32 to shape-only input specifications under `False` and
FP64 under `True`. Explicit `ShapeDtypeStruct` dtypes initially survive this
normalization. Later tracing and lowering still apply their own rules, so this
initial preservation does not establish an end-to-end dtype guarantee. See
[conversion_api.py](../../jax2onnx/converter/conversion_api.py) and
[user_interface.py](../../jax2onnx/user_interface.py).

The shared `numpy_dtype_to_ir_with_float_policy` maps FP32 to ONNX `DOUBLE`
when the argument is `True`. It preserves FP16, BF16, and FP64 in its own
mapping. An unspecified dtype also follows the argument. Therefore this helper
does more than translate a dtype already chosen by JAX. Conversely, its FP64
case does not downcast under `False`; tracing and context allocation have
additional behavior. See [ir_utils.py](../../jax2onnx/ir_utils.py).

Historically, this combination was intentional. The
[v0.6 implementation](https://github.com/enpasos/jax2onnx/commit/b646501ae07737e011a4f12e460c9c14f8da4773)
described a double export mode and a single precision downcast mode. Subsequent
fixes introduced important exceptions. Recasting the existing argument as only
an x64 permission would consequently change observable behavior.

## Lowering Has Several Type Authorities

The current native context separates `require_value_for_var` for inputs,
`lookup_value_for_var` for inspection, and `allocate_value_for_var` for writable
outputs. The compatibility helper `get_value_for_var` still reuses an existing
binding or allocates one. Allocating an output no longer means blindly reusing
an input or an already produced result. These ownership improvements do not
remove the precision policy: allocation still consults the Boolean and the
traced `aval.dtype`. See [ir_context.py](../../jax2onnx/converter/ir_context.py) and
[typing_support.py](../../jax2onnx/converter/typing_support.py).

Plugin output types can then follow a traced dtype, an allocated placeholder,
an operand, or a plugin-specific rule. For example,
[`convert_element_type`](../../jax2onnx/plugins/jax/lax/convert_element_type.py)
uses the exact traced target dtype, whereas general context allocation can
promote an FP32 placeholder. Final output binding also keeps existing floating
types in cases where they differ from its policy-derived target. Reading only
the JAXPR or only the final public output type therefore cannot establish the
precision of the whole exported calculation.

The implemented
[`promote_float32_operand`](../../jax2onnx/plugins/_precision_utils.py) helper
now adapts a FLOAT operand when its caller explicitly requires DOUBLE. The
adapter inserts a separate Cast and keeps the original binding available to
other consumers. Native Neg, selected unary and real binary operations,
reduction helpers, and concatenate paths use this adaptation. The explicit
FP64-to-FP32 rounding remains before the consumer's FP32-to-FP64 conversion.
This is a bounded compatibility repair, rather than a global precision rewrite
or a new interpretation of every `astype` expression.

Explicit dtype preservation is a tested boundary. The
[#217 cast fix](https://github.com/enpasos/jax2onnx/commit/e76a90eac15d4cef8dc5f8e5d06d7f70e2720912)
deliberately retained explicit cast targets. Current
[exact-dtype tests](../../tests/extra_tests/converter/test_exact_dtype_casts.py)
require FP32 graph outputs under `enable_double_precision=True` for casts and
selected constructors: zeros, ones, eye, full, linspace, and arange. They do
not establish preservation of every intermediate in arbitrary compositions.
The older bitcast target assertion is structural, and plugin metadata still
skips numerical validation. A new regression additionally checks a complete
exported FP32-to-UINT32 bitcast at opset 26 through the checker and CPU runtime,
while the same FP32 value also feeds a DOUBLE consumer. This is specific runtime
evidence, rather than general bitcast coverage.

FP16 and BF16 need separate consideration. Their shared mapping preserves
their types, and the context's disabled-double output allocation exempts both
from its general floating downcast. The FP16 placeholder regression covers
both Boolean settings. These facts do not prove complete FP16/BF16 constant,
operator, or runtime coverage. In particular, context constant promotion still
widens NumPy floating arrays, including FP16, under double export; the mapper's
exceptions are not universal rules for every path.

## Constants and Nested Graphs Add Further Rules

Closed-over constants are reconciled with their JAX variables and can then be
promoted before binding. `IRContext.bind_const_for_var` applies another
promotion path. At export time, `postprocess_ir_model` promotes FP32
initializer payloads, tensor-valued `Constant` attributes, and constant-backed
node outputs when requested. It recursively visits control-flow subgraphs and
ONNX Function bodies. This occurs after the converter's optimization pass.
See [ir_postprocess.py](../../jax2onnx/converter/ir_postprocess.py).

These paths can turn a precision decision into more than a metadata change:
they alter stored tensor payloads. The postprocessor does not implement a
complete typed graph transformation that reconciles every consumer and cast.
The existing
[constant-promotion tests](../../tests/extra_tests/converter/test_f64_constant_promotion.py)
require DOUBLE initializers and numerical agreement for selected captured
constant expressions. Separate
[postprocessing tests](../../tests/extra_tests/converter/test_ir_postprocess.py)
check direct payload promotion. Neither suite establishes arbitrary mixed
precision correctness.

Child contexts inherit the precision setting. Control-flow contexts also
inherit `_keep_function_float32`; selected scan paths enable that flag to
retain lower precision body values. Function boundaries have explicit formal
inputs, while eligible If branches can capture lexical values. These scope
rules and scan exceptions must be considered together with recursive late
promotion. The
[scan regression](../../tests/extra_tests/framework/test_one_minus_square_scan.py)
exercises a float32 body under double export, so preserving only public
outputs would miss an existing requirement.

## Numerical Accuracy Already Has Local Policies

Storage dtype, operation dtype, and accumulator precision are distinct. The
specialized explicit LayerNorm lowering already widens selected internal
work independently of `enable_double_precision`: FP16/BF16 statistics use
FP32, and the FP32 slow-variance reduction accumulates squared centered values
in FP64 before rounding the variance to FP32. The remaining FP32 statistics
and affine computation retain FP32. This is a local numerical algorithm,
rather than a request to turn the entire model into FP64.

The [documented limits](../../docs/user_guide/known_limitations.md) and
[LayerNorm tests](../../tests/extra_tests/test_layer_norm_precision.py) distinguish
error against JAX from error against an independent FP64 reference. Their
fixed bounds cover specified FP32 cases, inputs, opset, runtime provider, and
optimization settings. They provide no general guarantee of bitwise equality,
universal improvement from widening, or equivalent bounds for every
normalization family.

## ONNX Feasibility and Runtime Evidence

Precision support has separate schema, model-validity, runtime, and numerical
boundaries. The [ONNX Conv schema](https://onnx.ai/onnx/operators/onnx__Conv.html)
allows DOUBLE tensors, but this does not establish a corresponding kernel in
every execution provider. Focused standalone ONNX probes on 2026-10-17 used
Python 3.12.9, ONNX 1.23.2, ONNX Runtime 1.31.0 CPU, opset 23, IR version 10,
disabled runtime graph optimization, and one thread:

| Minimal operation | Full ONNX checker | ONNX Runtime CPU result |
| --- | --- | --- |
| FP64 `MatMul` | Passed | Loaded and executed |
| FP64 `Conv` | Passed | Load failed with `NOT_IMPLEMENTED` |
| BF16 `Add` and BF16 `Neg` | Passed | Both loads failed with `NOT_IMPLEMENTED` |

The BF16 probes used FLOAT inputs, explicit casts to BF16, the tested operation,
and a cast back to FLOAT. These are narrow kernel availability observations,
not complete jax2onnx exports or universal statements about these dtypes.

The [standard BitCast schema](https://onnx.ai/onnx/operators/onnx__BitCast.html)
starts at opset 26 and requires equal input/output bit widths. A separate
minimal FLOAT `[1]` to INT32 `[1]` reinterpretation failed checker and runtime
loading at opset 23, but passed both and returned `1065353216` for `1.0` at
opset 26. The current
[bitcast plugin](../../jax2onnx/plugins/jax/lax/bitcast_convert_type.py)
emits `BitCast` without an opset gate, while the public default remains opset
23 and its metadata skips numerical validation. The successful probe at 26
and the merged exported-model regression do not establish full JAX bitcast
coverage, especially for unequal bit widths.

Native [LayerNormalization](https://onnx.ai/onnx/operators/onnx__LayerNormalization.html)
also separates data and statistics: its data type parameter includes DOUBLE,
but its statistics type parameter allows FLOAT or BF16. Default `stash_type=1`
specifies FP32 normalization statistics. Thus DOUBLE input/output declarations
alone do not promise FP64 accumulation. This schema observation is not a
demonstrated CPU accuracy regression.

Runtime math settings add another boundary. ONNX Runtime's
[CUDA `use_tf32` option](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#use_tf32)
allows reduced-precision multiplication for certain FP32 matrix products and
convolutions while retaining FP32 tensor types. This is a documented provider
behavior, not a result of the CPU probes above; validation evidence needs the
relevant provider settings as well as the model's tensor dtypes.

Current CPU validation helpers and `export_mode="web"` have narrower roles:
the former execute selected comparisons on CPU, and the latter selects a
serialization profile. Neither is a general runtime capability or numerical
acceptance check. See [user_interface.py](../../jax2onnx/user_interface.py).

## Observable Behavior on Current Main

Focused checks at this snapshot used JAX 0.11.2 on CPU, ONNX 1.23.2,
ONNX Runtime 1.31.0, JAX x64 enabled, and `enable_double_precision=True`.
They show the distinction between source computation and later export promotion:

| Model expression | Tracing input dtype | Exported output dtype |
| --- | --- | --- |
| `x.astype(jnp.float32)` | FP32 | DOUBLE, with no Cast node |
| `x.astype(jnp.float32)` | FP64 | FLOAT, through a Cast node |
| `x.astype(jnp.result_type(x.dtype, jnp.float32))` | FP16 or BF16 | FLOAT, through a Cast node |
| `x.astype(jnp.result_type(x.dtype, jnp.float32))` | FP64 | DOUBLE, with no Cast node |
| `-x.astype(jnp.float32)` | FP64 | DOUBLE, after FP32 rounding and a consumer Cast |

The same-type FP32 cast disappears during JAX tracing. Later promotion therefore
cannot consistently distinguish an explicit source cast from an unchanged
FP32 value. The minimum-precision expression illustrates why its input dtype
must already reflect the intended precision when JAX traces the model. A
literal `astype(jnp.float32)` requests exactly FP32; it cannot also communicate
that FP32 is only a minimum for numerical stability. Preserving that cast
alone therefore does not resolve the minimum-precision intention. The proposed
[FP32 model-side example](precision_target_state_user_view.md#example-exactly-fp32-or-at-least-fp32)
uses computation configuration or a minimum-precision expression before
tracing, rather than having export silently reinterpret an exact conversion.
This is one example of operation-specific intent, rather than a general
requirement to widen lower precision values to FP32.

The original failing example from PR #275 now has consistent operand and
result types. For an FP64 input, `return -x.astype(jnp.float32)` follows:

```text
Cast: DOUBLE input -> FLOAT result
Cast: FLOAT result -> DOUBLE consumer operand
Neg:  DOUBLE input -> DOUBLE result
```

The merged runtime regression checks this result against negation of the input
rounded to FP32 and converted back to FP64. A direct output of the same rounded
value remains FP32, including when a DOUBLE consumer also uses it.

## Completed Legacy Repair and Remaining Boundaries

[PR #275](https://github.com/enpasos/jax2onnx/pull/275) is merged in this snapshot.
Its final implementation uses explicit consumer adapters, including the native
Neg path, rather than promoting values as a side effect of general lookup.
Graph output binding continues to require the existing computed value.

The focused
[exact-dtype suite](../../tests/extra_tests/converter/test_exact_dtype_casts.py)
passed all 26 cases locally at this snapshot. It covers Neg and Add after an
explicit FP32 cast; selected unary, real binary, reduction, and concatenate
consumers; shared direct outputs; function and layout boundaries; boolean and
integer results; and the opset-26 bitcast above. Runtime cases first run the
full ONNX checker, then execute the serialized model on CPU with optimization
disabled and one thread. FP16 Neg has runtime coverage here; BF16 Neg checks
types and model validity because the tested CPU kernel is unavailable.

The tests establish that these selected consumers can compute in DOUBLE after
the required FP32 rounding without changing the canonical FP32 value. They do
not establish source-type preservation for the full program. A fresh check of
`y = x.astype(jnp.float32); return (y + 1) - y` with FP64 input `1e8` produced
zero in JAX eager execution and one in ONNX Runtime CPU under this legacy
policy. This difference illustrates why restoring loadable, internally
consistent types is separate from preserving every source arithmetic step.

Finally, generated `_f64` tests rewrite floating inputs and selected factory
and expected dtypes to FP64. See [t_generator.py](../../tests/t_generator.py).
They supply useful FP64 coverage but cannot substitute for tests that retain
intentional mixtures of dtypes and observable rounding boundaries. The
current system consequently has substantial precision coverage alongside
context-dependent rules; it has no single faithful mixed precision contract
established across the whole pipeline.
