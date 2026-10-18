# Recording, Packaging and Array-Creation Findings

Date: 2026-10-18. C4a is committed at `8a82d510`, D1a at `223f24bf`, C4c at
`4c1c57fe` and D1b at `05037961`, including reviewed manual inventory additions.
All passed independent review and actual repository hooks. This is private
migration evidence. Public `to_onnx` still uses LEGACY; the new named-parameter
and array-creation diagnostics do not broaden audited `source-static-v1`
admission. Only phase F completes the transition.

## Recording and packaging

C4a removes three SOURCE option guards. It reuses the existing shared logger,
scope ownership, error handling, serializer and external-data threshold.

- Recording off/on/off produces identical serialized artifacts and reports for
  root, nested Function, If, Loop and combined cases. Nested scopes flush once
  at the root. Real log-write failures preserve successful artifacts and the
  original lowering/finalization failure, including preexisting output files.
- An exact 1,200,000-byte FP64 payload contains `1 + 2**-30`, positive/negative
  zero and two distinct NaN payloads. Tests follow the actual root-to-outer-to-inner
  Function calls to the nested Constant, independently checking its bytes and
  the root initializer across IR, proto and saved/reloaded file returns.
- Standard serialization produces a real external-data sidecar. A subsequent
  web export to the same path removes that sidecar and retains exact inline
  root and nested payloads, custom names and report metadata.

These tests establish artifact behavior, not browser execution or numerical
acceptance. Diagnostic loop bodies retain UNVERIFIED lowering. The implementation
does not introduce a separate logger or serializer.

Maintained regressions:
[recording](../../../tests/extra_tests/converter/test_conversion_primitive_recording.py)
and [facade/serialization](../../../tests/extra_tests/converter/test_source_facade.py).

## Real named Boolean operands

C4c traces positional inputs and real scalar-BOOL named operands in one complete
prepared root. One immutable binding tuple connects actual source vars, specs,
positional indices/keyword names, final inputs and report entries. Genuine unused
formals remain ports without artificial sinks. Checked rebinding after graph
cloning, complete alias/name mapping and atomic renaming preserve ordering through
IR, proto and saved/reloaded boundaries, including same-typed port reversals.

The private facade/core handshake requires explicit diagnostic validation;
false-valued validators still work. SOURCE skips late parameter materialization
and promotion by name. NNX/Linen/Equinox Dropout consumes actual typed operands,
while Function defaults, static literals and captures retain their local meaning.
Name-only flag requests fail instead of appending ports. Public LEGACY behavior
and default audited admission remain unchanged.

Focused tests follow real positive-rate controls through Function/If/Loop scopes
to outputs. Exact deterministic inference and zero-rate references remain
separate from finite training mask/scaling checks; no equality of JAX/ONNX random
streams is claimed. An external whole-call `lax.cond` adapter preserves an
unchanged Python-branch source and both live runtime choices. Static partial
application covers an intended fixed setting only. Original NNX/Linen state and
Equinox key prerequisites remain explicit.

The [all-22 diagnostic pass](fixed_source_compatibility_findings.md#c4c-diagnostic-pass-over-all-22-rows)
yields three checked artifacts and 19 pending rows per stack. No ORT sessions ran
in that batch; all lowering/runtime/numerical gates remain UNVERIFIED. Focused
successes do not establish full-model or stochastic acceptance.

A confirmed separate shared optimizer defect remains open: a same-graph
`Not(True)` feeding Dropout can be replaced by an unowned `false_const`, also
breaking other uses of that matched value. Generic Boolean graphs without that
Dropout match are not affected by this pass. C4c folds actual SOURCE literals
with builder-owned constants. A minimal shared optimizer repair is planned,
without a production-fix claim in this checkpoint.

Maintained regressions:
[runtime parameters](../../../tests/extra_tests/converter/test_source_runtime_parameters.py),
[framework Dropout](../../../tests/extra_tests/converter/test_source_dropout_parameters.py)
and [trace helpers](../../../tests/extra_tests/converter/test_conversion_trace_helpers.py).

## Native promotion and immutable operands

D1a resolves SOURCE `where` result dtype and weakness from the original operands
using native JAX promotion. It preserves standard/strict promotion behavior and
passes resolved properties as hashable primitive parameters. Lowering consumes
the actual output aval. Separate Cast consumers leave shared canonical INT32
constants unchanged, including `[16777217, 2147483647]` returned or used by integer
and FP64 consumers through Function/If/Loop scopes.

SOURCE `full` retains the converted fill's weak type. A Python weak fill therefore
combines with strong FP16/BF16/FP32/INT32 data as native JAX does; explicitly typed
fills remain strong. Existing policy-scoped tracing caches isolate ordinary JAX,
LEGACY and SOURCE without a new cache mechanism or clearing caches between calls.
LEGACY binding and lowering rules remain unchanged.

Original unpatched JAX traces and references precede export. The finite local
diagnostic checks typed signatures, recursive scopes, def-use and exact resolved
parameters; malformed metadata fails. It does not grant audited product admission.
The [maintained tests](../../../tests/extra_tests/converter/test_source_array_creation_promotion.py)
check source properties, serialized artifacts and separately scoped CPU samples.

## Boolean selection and target failures

Installed CPU ORT 1.31/1.29 lacks a kernel for the neutral Boolean-output
`Where(16)` artifacts, despite full ONNX checker validity. The public LEGACY
control retains that failure. SOURCE Boolean results now use
`(condition AND x) OR ((NOT condition) AND y)`. Each intermediate has its actual
pairwise broadcast shape. Twenty real exported cases cover the truth table,
scalars, broadcasting, numeric conditions and empty dimensions across inline,
Function, If and Loop scopes. Their samples match original JAX exactly with
runtime optimization disabled and enabled on both stacks.

This alternative uses four logical nodes in place of one Where; latency and
memory costs are unmeasured. Numeric selection continues to emit Where with
JAX-promoted operand types.
Recorded BF16 Where kernels are still unavailable on both installed CPUs under
both optimization settings. BF16 artifact/type checks without CPU samples do not
claim numerical acceptance. Kernel-negative tests retain explicit evidence and
also accept a future kernel only if its sample matches exactly. None of these
observations establishes universal ONNX or deployment support.

## Native sequences and retained exact failures

D1b dispatches SOURCE `arange`/`linspace` to captured original JAX before argument
normalization. It preserves native dtype/weakness, strict-promotion errors,
original positional/device forms and explicit conversion boundaries. Omitted
versus explicit arange step retains the distinct low-precision iota/arithmetic
sequence. Linspace retains weighted endpoints, exact endpoint append, floor
before integer conversion and native empty/singleton behavior. Tests compare
unpatched native typed programs and references before export, then check serialized
artifacts and separate CPU samples. Same-callable native/LEGACY/SOURCE cache orders
cover inline/Function/If/Loop without clearing caches or changing LEGACY lowerers.

Four exact linspace cells per stack (these two dtypes with x64 off/on) preserve
the original trace and ONNX intermediate types but fail their numerical criterion
on both JAX 0.11.2 and 0.10.2 stacks:

| Original computation | Original JAX eager/JIT | ONNX ReferenceEvaluator |
| --- | --- | --- |
| FP16 `linspace(-10, 3, 7)`, element 5 | `0.830078125` | `0.8349609375` |
| FP32 `linspace(-1, 1, 5, endpoint=False)`, element 3 | `0.20000003278255463` | `0.20000004768371582` |

ORT with optimization off/all also fails these exact samples; the FP16 probe
has additional differing elements. JAX eager/JIT agree for the recorded samples.
This is not evidence of an ORT-only defect or an ONNX impossibility.

Independent review of native StableHLO, optimized HLO, LLVM/disassembly and bounded
native controls establishes a narrow compiler observation: these selected
witnesses show JAX CPU reciprocal substitution, reassociation and FP32 fused
multiply-add relative to the preserved typed graph. For FP16 weight index 5,
`FP16(5 * FP16(1/6))` has bits `0x3aaa`, while `FP16(5/6)` has `0x3aab`.
For the FP32 sample, dynamic endpoints give index-3 bits `0x3e4ccccf`, while
endpoints embedded in the outer jitted function give `0x3e4cccd0` on both stacks
with x64 off/on. The observed fused arithmetic reproduces that narrow difference.

Constant endpoints and exposed intermediates can change compilation. They are
diagnostic controls, not replacement oracles for the original dynamic-endpoint
call. This evidence does not explain every additional ORT FP16 discrepancy or
justify a general reciprocal/FMA emulation. The original exact comparisons remain
FAILED; no algorithm, endpoint staticness, reference or criterion was changed.

Preserve mandatory dtype, weak-type, integer-floor, endpoint and explicit-cast
semantics. Ordinary cross-runtime floating reproducibility and required rounding
need separate reviewed criteria. Do not widen the computation, replace the native
algorithm or relax these exact failures. Native binding acceptance does not close
family-wide numerical work. Missing corpus criteria remain unresolved; a later
justified criterion must not erase this evidence.

BF16 arange with omitted step currently fails CPU Add loading; explicit step
reaches the NumPy result-interface failure. Both retain UNVERIFIED numerical
acceptance and failed overall target evidence. Data-dependent arange remains a
separate unfinished compatibility capability: public LEGACY FLOAT32/opset23
passes exact CPU samples off/all, while native SOURCE tracing raises
`ConcretizationTypeError`. BF16/opset23 is schema-positive but fails the result
interface. Legacy opset27 artifacts are schema-positive but the installed runtimes
reject that unreleased opset. These are distinct target/implementation gaps, not
retirement of the existing dynamic capability or ONNX-impossibility claims.

The [maintained sequence regressions](../../../tests/extra_tests/converter/test_source_array_creation_sequences.py)
record these outcomes with unchanged criteria. Their narrow recursive diagnostic
rejects malformed typed/parameter/placement forms and keeps lowering UNVERIFIED;
it does not broaden product admission.

The local replay evidence is `/tmp/jax2onnx-d1b-rounding-py312.jsonl` and
`/tmp/jax2onnx-d1b-rounding-py311.jsonl`, with standalone ONNX paths recorded
there. These are local diagnostic artifacts, not maintained runnable tools or
committed models. The values above preserve the findings independently of those
temporary files. The reviewed compiled diagnosis is recorded in
`/tmp/jax2onnx-d1b-compiled-rounding-{py312,py311}.jsonl` and
`/tmp/jax2onnx-d1b-semantic-assessment.md`; the bounded observations above do not
depend on retaining those temporary files.

## Validation and provenance

The tested stacks are Python 3.12/JAX 0.11.2 and Python 3.11/JAX 0.10.2.
Selections overlap and must not be added into a support count.

| Packet | Focused selection on each stack | Independent new-case rerun |
| --- | --- | --- |
| C4a | 485 passed; 23 new cases | 23 passed on each stack |
| D1a | 453 passed; 47 generated LEGACY controls also passed | 134 passed on each stack |
| C4c | 726 passed across owner selections, including 45 LEGACY controls; 107 new cases | 110 passed per stack, including three existing trace-helper controls; one affected regression per stack after the typing-only repair |
| D1b | 508 focused and 70 generated/LEGACY controls passed | All 125 new cases passed per stack |

C4c comprises six behavioral production files, one existing-capability protocol
declaration and three test files. D1b comprises two dispatch files and one test
module. The final pinned Black change to that module removes redundant parentheses;
independent complete-AST comparison, including type comments, confirms unchanged
behavior. Production hashes are unchanged.

The latest complete primary-stack suite passes **6,035 tests**, with 13,110
warnings in 402.20 seconds and no exclusions. Required package typing passes
541 files, RNG registration and global Ruff pass, and both new commits pass
actual configured hooks. The earlier preflight was intentionally interrupted
after typing found defects; its 1,215 passes are not a complete-run result.
This regression result includes tests that correctly retain failed target
evidence; it is not a source-corpus or deployment acceptance count.

C4c's reviewed inventory has 194 handwritten-test files, 990 functions and 238
migration additions. D1b's final inventory has 195 files, 1,003 functions and
251 migration additions. All 752 original anchors, 1,802 raw contracts and optional
entries remain unchanged; no original anchor was removed. Fresh inventory CLI
comparisons pass on both stacks. The
[first B3 batch](source_corpus_findings.md) was not rerun and remains bound to its
historical `158ce625` baseline. The [22-row compatibility review](fixed_source_compatibility_findings.md)
also remains a migration obligation rather than completed SOURCE acceptance.

The prior C4a/D1a full run is `/tmp/jax2onnx-precision-c4a-d1a-full-py312.log`.
The new complete run is `/tmp/jax2onnx-precision-c4c-d1b-full-final-py312.log`;
independent focused logs are `/tmp/jax2onnx-c4c-astra-py{312,311}.log`,
`/tmp/jax2onnx-c4c-astra-typing-repair-final-py{312,311}.log` and
`/tmp/jax2onnx-d1b-astra-py{312,311}.log`. Maintained regression commands run the
linked test modules with the repository's normal pytest setup; these log paths
are provenance, not dependencies of the tests.

The [work plan](../precision_implementation_work_plan.md) retains the remaining
parameter/model coverage, shared optimizer repair and numerical gates. Broader
options, framework/random contracts, full-model validation and the final public
transition remain open.
