# Recording, Packaging and Array-Creation Findings

Date: 2026-10-18. C4a is committed at `8a82d510`; D1a and the reviewed manual
inventory refresh are committed at `223f24bf`. Both passed independent review
and actual repository hooks. This is private migration evidence. Public
`to_onnx` still uses LEGACY, and `where`/`full` remain excluded from the audited
`source-static-v1` profile. Only phase F completes the transition.

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

## Future sequence work: exact failures remain visible

The proposed D1b `arange`/`linspace` work is not part of these commits. Captured
original JAX binding preserves native typed arithmetic more faithfully than
rebuilding it from NumPy promotion or delta stepping. Existing dynamic `arange`
capability remains required compatibility work even where native tracing rejects
data-dependent bounds.

Additional local diagnostic probes preserve the original `linspace` trace and
ONNX intermediate types but fail their exact numerical criterion on both JAX
0.11.2 and 0.10.2 stacks:

| Original computation | Original JAX eager/JIT | ONNX ReferenceEvaluator |
| --- | --- | --- |
| FP16 `linspace(-10, 3, 7)`, element 5 | `0.830078125` | `0.8349609375` |
| FP32 `linspace(-1, 1, 5, endpoint=False)`, element 3 | `0.20000003278255463` | `0.20000004768371582` |

ORT with optimization off/all also fails these exact samples; the FP16 probe
has additional differing elements. JAX eager/JIT agree for the recorded samples.
This is not evidence of an ORT-only defect or an ONNX impossibility. Compiler
contraction/reassociation and low-precision evaluation are hypotheses requiring
compiled-IR and arithmetic investigation. Observing additional intermediates may
itself change fusion, so diagnostic observations cannot replace the original
reference.

Preserve mandatory dtype, weak-type, integer-floor, endpoint and explicit-cast
semantics. Ordinary cross-runtime floating reproducibility and required rounding
need separate reviewed criteria. Do not widen the computation, replace the native
algorithm or relax these exact failures to close D1b. Missing corpus criteria
remain unresolved; a later justified criterion must not erase this evidence.

The local replay evidence is `/tmp/jax2onnx-d1b-rounding-py312.jsonl` and
`/tmp/jax2onnx-d1b-rounding-py311.jsonl`, with standalone ONNX paths recorded
there. These are local diagnostic artifacts, not maintained runnable tools or
committed models. The values above preserve the findings independently of those
temporary files.

## Validation and provenance

The tested stacks are Python 3.12/JAX 0.11.2 and Python 3.11/JAX 0.10.2.
Selections overlap and must not be added into a support count.

| Packet | Focused selection on each stack | Independent new-case rerun |
| --- | --- | --- |
| C4a | 485 passed; 23 new cases | 23 passed on each stack |
| D1a | 453 passed; 47 generated LEGACY controls also passed | 134 passed on each stack |

The latest complete primary-stack suite passes **5,803 tests**, with 13,104
warnings in 382.81 seconds. Required package typing passes 541 files, RNG
registration and global Ruff pass, and both commits pass actual pinned hooks.
This regression result includes tests that correctly retain failed target
evidence; it is not a source-corpus or deployment acceptance count.

The reviewed inventory has 192 handwritten-test files, 963 functions and 211
migration additions. All 752 original anchors, 1,802 raw contracts and optional
entries remain unchanged; no original anchor was removed. Fresh inventory CLI
comparisons pass on both stacks. The
[first B3 batch](source_corpus_findings.md) was not rerun and remains bound to its
historical baseline. The [22-row compatibility review](fixed_source_compatibility_findings.md)
also remains a migration obligation rather than new SOURCE acceptance.

Local logs: `/tmp/jax2onnx-precision-c4a-d1a-full-py312.log`,
`/tmp/jax2onnx-c4a-astra-new-py312.log`,
`/tmp/jax2onnx-c4a-astra-new-py311.log`,
`/tmp/jax2onnx-d1a-astra-py312.log` and
`/tmp/jax2onnx-d1a-astra-py311.log`. Maintained regression commands run the linked
test modules with the repository's normal pytest setup; these log paths are
provenance, not dependencies of the tests.

The [work plan](../precision_implementation_work_plan.md) specifies the next
coupled C4c root-binding packet and D1b numerical gates. Broader options,
framework/random contracts, full-model validation and the final public transition
remain open.
