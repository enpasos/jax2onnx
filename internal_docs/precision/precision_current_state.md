# Computation Precision Current State

Public precision-policy analysis baseline: 2026-10-17. Source snapshot:
[`fae17822e0114e6c00840b8f78adedb7ec36f74e`](https://github.com/enpasos/jax2onnx/commit/fae17822e0114e6c00840b8f78adedb7ec36f74e),
which includes the merged consumer-adaptation fix from PR #275 and the native
typed lowering changes from PR #274.

The local support floor is **Python >=3.11, JAX/JAXLIB >=0.10.2 and Flax >=0.12.8**.
Framework-floor and all-direct-minimum evidence remain separate.

## Current private implementation

The last completed checkpoint remains **`fe214590`**. The [canonical ledger](poc/source_integration_findings.md#current-checkpoint)
owns current source/dependency epochs, exact directed counts and remaining
gates. Three-stack results now include lexical bindings, exact Constant origins
and the unchanged repeated-Function/If recipe with both CPU modes and three
native comparisons. These qualify their recipe and owner selection only.

Private work preserves original callable/state preparation, deliberate cast
rounding, more layouts and typed TopK widths/empty results. BF16 portable
original-word checks pass on both capable stacks; strict native NaN word/index
comparisons retain their independent FAILED outcomes. Previous model samples
do not qualify changed code or dependency epochs.

Current code 9134 adds only reviewed local typing annotations/key rename and
passes fresh typing/RNG and both scoped 279-case paired selections. The actual full-suite and focused
results below remain source-bound to b90. Its reviewed live integer-CSE branch
restoration has an actual
PRIMARY joint gate of 331 PASS, including 197 engine cases with 44 new controls,
unchanged Transformer 52 and repeated Function/If 2. The exact proof covers both
Gather-index and floating-Where-mask branches and actual downstream redirects.
Original Cast bytes/artifacts, final precision equality, bridge/math helpers
and numerical criteria remain unchanged. This resolves the prior two selected
Transformer 26 failures prospectively; e6 and earlier failed runs remain history.

The single ordinary PRIMARY full suite is terminal and failed at b90. All 54 model/state runs plus 2 guard controls pass, as do Transformer 52
and MAIN If 2; the other failures remain unresolved. Their exact counts, IDs and
owner accounting appear only in the canonical ledger. No full-suite acceptance
is claimed. A local WIP commit may preserve unfinished changes
after relevant checks, all seven normal hooks and staged review; acceptance
requires the separate complete gates. Fresh 9134 typing/RNG and both scoped
SECONDARY/DIRECTMIN selections pass; earlier e6 results remain source-scoped. A full standard pass would not
complete the current 3,704-cell map, opaque/metadata-free/manual accounting, all
fixed-source/numerical workflows or coordinated public cutover. The canonical
ledger owns exact gates and the corrected earlier ROOT attribution.

Public precision policy/defaults remain **LEGACY**. All existing functionality,
fixed supplied JAX sources, the unchanged inventory, original owners and phases
B–F remain required through one uniform source-preserving contract. The public
analysis below retains its baseline and does not announce a cutover.

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

The baseline normalization reconstructs `ShapeDtypeStruct` from shape and dtype
and drops `weak_type`. That JAX property affects type promotion, so retaining
the dtype alone can change the computation JAX traces. Public legacy calls
retain those observed normalization rules. The initial source experiment
inherited the same loss in both the UX wrapper and shared tracing helper.
The later private source policy now preserves the property through both steps,
including direct helper calls, and reports supplied/normalized/traced flags.
The [regression](poc/migration_ux_findings.md#weak-input-properties-preserved-on-the-source-path)
returns FP16 `2048` for a weak scalar plus explicit FP16 `1`, while a strong
input with the same shape/dtype returns FP32 or FP64 `2049`. Both supported
stacks cover this source correction; it does not change legacy normalization.

The [calling experiment](poc/migration_ux_findings.md) compares real public calls
with the argument omitted, explicitly `False`, and explicitly `True`, for
shape-only inputs, typed specifications, NumPy arrays, and JAX arrays under
both ambient x64 settings. Omission and `False` currently select the same
behavior. An FP64 `ShapeDtypeStruct` on those routes is truncated during JAX
tracing: JAX 0.10.2 is silent for the tested case; JAX 0.11.2 warns. Both warn
for a model-internal `.astype(jnp.float64)` under disabled x64. These are
observed version differences, not diagnostics guaranteed by the exporter.

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
specialized explicit LEGACY LayerNorm lowering already widens selected internal
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

The private SOURCE path preserves the framework's normalization choices.
The later [bounded normalization acceptance](poc/source_integration_findings.md#mlp-state-and-bounded-normalization-acceptance)
adds selected static FP16/FP32/FP64 exports and CPU comparisons. Its named-scope
payload proof evaluates the actual exported body eagerly and with JIT;
whole-carrier JIT equality remains unestablished. The recorded SOURCE criteria
are separate from these legacy bounds and do not certify arbitrary
LayerNorm-containing models.

## What the Precision PoCs Establish

The [numerical findings](poc/numerical_contract_findings.md) compare JAX eager
and JIT execution, ONNX ReferenceEvaluator, and ONNX Runtime CPU with optional
optimization disabled and enabled. Both installed stacks reproduce the
observations: JAX 0.11.2 / ORT 1.31.0 and JAX 0.10.2 / ORT 1.29.0. The findings
record exact versions and inputs. The artifact origin matters:

| Case and origin | ONNX validity | Execution and numerical observation | What remains open |
| --- | --- | --- | --- |
| Public export: FP16 batched `dot_general`, preferred FP32 result | Full checker passes. | JAX returns FP32 `2049`; exported `MatMul` returns FP16 `2048` on ReferenceEvaluator and both ORT settings. | This public legacy result-type gap remains; the private source lowering below handles a bounded subset. |
| Private source export: default-precision FP16 rank-2/rank-3 matrix operands, explicitly preferred FP32 result | Full checker passes. | Operand casts precede an FP32 `MatMul`; the `2049` witness and additional finite-value cases retain FP32 results while shared input values remain FP16. | Other layouts, dtypes, explicit precision algorithms, targets, and performance require separate work. |
| Changed JAX source: explicitly FP32 matrix operands, public export | Full checker passes. | All tested paths return FP32 `2049`. | Sample agreement does not make this a validated automatic lowering or establish its hardware and performance properties. |
| Hand-built normalization control: FP32 centering/squares, FP64 variance sum and division, FP32 output | Full checker passes. | JAX, independent staged reference, ReferenceEvaluator, and both ORT settings agree on variance `8388609`; a NumPy FP32-only sum gives `8388608`. | This original control establishes feasibility; the local source integration below now exports the complete callable. |
| Private source export: the same selected-FP64 normalization, inline, in a Function, and in either conditional branch | Full checker passes after save/reload. | Six boundary/optimization cases retain exact centered values, squares, and variance, with a tolerance on normalized outputs. | This covers the stated calculation and inputs, rather than general normalization families or deployment targets. |
| Source-PoC export: FP16 square then sum | Full checker passes; ReferenceEvaluator agrees with JAX on the sample. | JAX returns zero squares and zero sum. Both ORT settings return zero squares but sum `2**-23`. | The tested runtime fails the sample's required intermediate rounding despite successful execution. |
| Hand-built unfused FP16 `Mul -> ReduceSum` control | Full checker passes. | The same ORT rounding discrepancy occurs. | This rules out blaming converter fusion alone; it does not identify the underlying cause. |
| Changed JAX source: explicit FP32 multiplication, FP16 rounding, FP32 sum, FP16 result; source-PoC export | Full checker passes. | Every tested path returns zero with the original FP16 output signatures. | The arithmetic change is only a bounded source candidate, not a general exporter repair. |

For normalization, exact centered values, squares, and variance are checked;
only the final normalized output uses a tolerance. A broad relative tolerance
on variance would accept losing the very terms the experiment is designed to
preserve. The constant input row also has an exact zero result.

The FP16 discrepancy remains when optional ORT graph optimization is disabled.
That setting does not establish the arithmetic performed inside a kernel.
Neither ReferenceEvaluator agreement nor the unfused control proves a general
rounding guarantee or identifies a converter fusion bug.

The [calling experiment](poc/migration_ux_findings.md) adds input normalization,
actual tracing observations, option conflicts, and warning reporting around
the existing entry points. Explicit FP64 inputs with x64 disabled are rejected before tracing
on the candidate source route; a model-internal FP64 conversion can already
have been truncated by JAX and is reported through its actual warning.

The later [private integration](poc/source_integration_findings.md) replaces
the local allocation subclass with `PrecisionPolicy.SOURCE` in the real
`IRContext`. It reaches audited `onnx_function` and two-branch `cond` children,
preserving selected mixed outputs, shared FP32 values, and FP32/FP64 constants
through serialization and CPU execution. Floating `gt` and Boolean-to-INT32
selector conversion cover the tested condition wiring. Recursive validation
checks branches and actual retraced function bodies. The later Where/JIT
checkpoint adds bounded pure closed JIT bodies; broader loops/JIT forms and
other unaudited operations remain excluded.

The source function variant carries flattened result abstract values and
reconstructs structured returns. It now captures `jax_enable_x64` and
`jax_default_matmul_precision` at each call site, replays them while tracing the
body, and includes them in function cache signatures alongside weak flags,
shape/dtype, and captures. Already-typed FP64 formals survive a local x64-off
scope; recreating the formals must not truncate a value that already existed.
The retraced result signature is checked against the original call.

Review exposed two distinct faults that the local implementation now repairs:
replaying a function under ambient rather than call-site x64 could return FP32
`0` instead of `1`, and JAX's cached conditional branch could retain a source
primitive during a later public legacy export. A private tracing discriminator
now separates ordinary JAX, legacy export, and source export caches. It uses
JAX's trace/JIT cache context through the compatibility layer rather than
clearing caches. The
[context regressions](../../tests/extra_tests/converter/test_source_function_context.py)
cover both local x64 directions, unchanged result dtypes, mixed call-site
settings in one graph, nested functions, scoped matrix requirements, failure
restoration, and export sequences around ordinary eager/JIT execution.

The [input](../../tests/extra_tests/converter/test_source_inputs.py) and
[scope](../../tests/extra_tests/converter/test_source_scopes.py) regressions now
live in the regular repository suite. They cover static arguments, structured
returns, repeated aliases, and input passthrough as well as mixed types.
The bounded source
[matrix lowering](../../tests/extra_tests/converter/test_source_dot_precision.py),
[normalization integration](../../tests/extra_tests/converter/test_source_normalization.py),
and [arithmetic-boundary checks](../../tests/extra_tests/converter/test_source_arithmetic_fusion.py)
also run there. The source path retains `Add -> Div` and `Mul -> ReduceSum`
instead of applying the legacy Mean and ReduceSumSquare substitutions without
an audited numerical equivalence argument.

The [sample acceptance helper](../../tests/extra_tests/converter/precision_runtime_support.py)
records the actual CPU runtime, optimization mode, architecture, thread counts,
and comparison result. It rejects failed or unverified sample acceptance and
keeps execution errors separate. It does not prevent export or implement a
public target-selection API. In particular, the known FP16 square/sum model
still exports and executes, but fails acceptance against its exact reference
on the recorded runtimes. That numerical defect remains unresolved.

Historical validation totals are recorded in the
[integration findings](poc/source_integration_findings.md#recorded-validation);
the [historical option checkpoint](poc/option_and_array_creation_findings.md#validation-and-provenance)
records its complete 5,803-test run. The
[current checkpoint](poc/source_integration_findings.md#tensor-index-erfc-and-optional-inventory-checkpoint)
records the new 10,047-case primary pass and local code checkpoint `1533d60b`
with seven passing normal code hooks and accepted independent final code-evidence
review.
A green suite can include tests that recognize the known numerical failure;
it does not establish model-wide or deployment-wide precision compliance.

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
