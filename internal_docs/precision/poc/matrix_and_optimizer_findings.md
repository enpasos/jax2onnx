# Ordinary Matrix and Shared Optimizer Findings

Date: 2026-10-18. The shared Dropout optimizer repair is committed at `c528a894`;
D2a ordinary matrix lowering is committed at `2011b140`. Both passed independent
review and actual configured hooks. Public `to_onnx` remains LEGACY, and audited
`source-static-v1` admission is unchanged. The new matrix cases use private
diagnostics with lowering UNVERIFIED. This checkpoint does not complete a matrix
family, the source corpus or phase F.

## Bounded ordinary matrix contract

SOURCE `dot_general` now handles same-dtype FP32/FP64 operands and results, with
preferred result type omitted or equal to that dtype and default precision.
Both operands must have equal positive static rank 2–4, one contracting axis and
one free axis each, and paired batch axes. Dimension sizes, source result shape
and actual physical dtype/shape are checked before arithmetic emission.

Fresh typed Transpose consumers, only where needed, arrange the left operand as
batch/free/contract and the right as batch/contract/free. They preserve the
source batch-tuple order, including nonleading and differently positioned batch
axes. Canonical inputs, captured constants and their other consumers remain
unchanged. The result uses MatMul without a Gemm zero bias, result cast or global
widening. The earlier bounded FP16/preferred-FP32 route retains its two local
operand Casts and FP32 MatMul.

SOURCE `jnp.matmul` forwards original arguments to captured native JAX before
custom normalization. Native promotion, weak typing, strict-promotion errors,
preferred-result fields and precision settings survive interception. The LEGACY
body and policy-scoped cache behavior remain unchanged. The lowerer accepts only
unspecified `out_sharding=None`; non-None placement and unknown semantic fields,
including unaudited `out_dtype`, are rejected explicitly.

Remaining work includes vectors, broadcast/general contractions, zero or symbolic
extents, other operand/result combinations, low-precision/discrete/complex
results, explicit precision algorithms and broader wrappers. These implementation
exclusions do not retire existing capabilities or establish ONNX impossibility.
Maintained regressions cover [ordinary matrices](../../../tests/extra_tests/converter/test_source_ordinary_matrix.py)
and the [earlier precision contract](../../../tests/extra_tests/converter/test_source_dot_precision.py).
They compare original unpatched JAX traces/references before activation, inspect
actual typed artifacts, and cover inline/named/nested/conditional scopes,
serialization, shared constants and native/LEGACY/SOURCE cache orders.

## Exact subnormal failures remain failed

Selected exact target checks record 24 validated observations and four failed
observations on each stack. The four failures are this subnormal sample for
FP32/FP64, each under CPU ORT optimization disabled/all:

| Property | Recorded value |
| --- | --- |
| Left operand | Shape `[1,2]`, two smallest positive subnormals, each with bit pattern 1 |
| Right operand | Shape `[2,1]`, two ones of the same dtype |
| Original JAX eager and JIT | Shape `[1,1]`, same dtype, positive-zero bit pattern 0 |
| CPU ORT result | Shape `[1,1]`, same dtype, bit pattern 2 |
| Evidence | Schema and runtime execution VALIDATED; bitwise numerical comparison FAILED |

The source settings are x64 disabled for FP32 and enabled for FP64. Recorded
targets are x86_64 CPUExecutionProvider with ORT 1.31/1.29, sequential execution,
one intra/inter-op thread, and no extra session configuration. The numerical
acceptance guard rejects the failed samples. Tests retain the exact reference
and observed payload evidence; an arbitrary mismatch cannot satisfy the expected
failure branch. A future genuine exact match must pass the original criterion.

Other selected exact samples include dyadics, range/mantissa-sensitive values,
cancellation, NaN/Inf and signed zero. Separate deterministic 64-term reductions
use criteria fixed before export: FP32 `rtol=2e-6, atol=1e-5`, FP64
`rtol=2e-14, atol=1e-13`. Those sample criteria do not relax the failed bitwise
subnormal checks or establish universal matrix reproducibility. No flush cause,
ORT-only defect or impossibility of a faithful alternative is inferred here.
Latency and memory costs have not been measured.

The [two-cell B3 supplement](source_corpus_findings.md#two-cell-matrix-supplement-on-2026-10-18)
is separate from these runtime samples. Its original ordinary FP32 cells now
pass schema checking but fail the unchanged raw Gemm/zero-bias callback. It adds
no accepted artifact or corpus row; runtime and numerical gates remain unverified.

## Shared Dropout optimizer repair

The former same-graph `Not(True)`/Dropout rewrite could replace uses with an
unowned `false_const`. The repaired pass omits only input slot 2, the optional
training input, of a matched standard-domain Dropout. Omission retains ONNX's
inference default. It creates no constant, replacement name or global-use
rewrite. The existing later cleanup may trim the empty trailing slot.

After all matched Dropout consumers are handled, the Not can be removed only
when no remaining use, nested capture or graph output retains its result. Shared
and observable values keep their original identity and metadata. Same-graph
proof and standard-domain guards exclude unrelated/custom operators. Dynamic
inputs, overridable graph-input initializers, `Not(False)`, direct-False behavior
and generic Boolean graphs without a Dropout match remain unchanged. Existing
traversal stays nonrecursive; no new optimizer policy or stochastic rewrite is
introduced.

Valid fixtures cover top-level, Function, If and Loop graphs, retained captures
and outputs, custom-domain Functions, and both existing precision policies.
The original valid witness passes full ONNX checking before and after the pass,
then exact CPU inference data checks on both stacks. Separate maintained
regression fixtures cover inference masks. These deterministic checks do not
establish equivalence of JAX/ONNX training masks or random streams.
Maintained tests are the [new regression module](../../../tests/extra_tests/framework/test_dropout_training_mode_optimization.py)
and [existing optimizer controls](../../../tests/extra_tests/framework/test_ir_optimizations.py).

One initial standalone Python 3.11 invocation imported an older editable checkout;
its failed execution and empty output were retained, correcting the initial
success claim. Fresh runs explicitly selected this repository and verified the
imported optimizer path and frozen hash before executing the unchanged witness.
Both passed. Independent pytest runs from the repository are separate evidence.
The correction is recorded in
`/tmp/jax2onnx-dropout-optimizer-witness-evidence-correction.md`, with verified
`/tmp/jax2onnx-dropout-optimizer-repaired-witness-py{312,311}-verified.json` and
adjacent execution records. These are local provenance, not test dependencies.

## Validation and provenance

| Packet | New cases | Owner checks on each stack | Independent checks on each stack |
| --- | --- | --- | --- |
| Shared Dropout optimizer | 34 | 339 across focused and original LEGACY controls | 121 including new cases and existing controls |
| D2a ordinary matrices | 115 | 390 focused plus 90 generated/integration controls | 435 including all new cases and 45 generated matrix controls |

These selections overlap and must not be added into a support count. Both stacks
use CPU execution: Python 3.12.9/JAX 0.11.2/NumPy 2.5.4/ONNX 1.23.2/ORT 1.31.0
and Python 3.11.12/JAX 0.10.2/NumPy 2.4.6/ONNX 1.22.0/ORT 1.29.0, with onnx-ir
1.0.0. The complete primary-stack suite passes **6,183 tests**, with 13,110
warnings in 406.62 seconds and no exclusions. Required typing passes 541 files
plus RNG registration; global Ruff and both commits' actual configured hooks
pass. Passing tests include correct recognition of failed target evidence.

The optimizer inventory records 196 handwritten-test files, 1,009 functions and
257 migration additions. D2a's final inventory records 197 files, 1,023 functions
and 271 additions. Both-stack CLI comparisons pass: all 752 original anchors,
1,802 raw contracts and optional entries remain unchanged. The first B3 batch
remains bound to its `158ce625` baseline. The all-22 model diagnostic batch remains
three checked artifacts and 19 pending rows per stack, with zero ORT sessions
and unverified lowering/runtime/numerics.

The full-run log is `/tmp/jax2onnx-precision-dropout-d2a-full-py312.log`.
Independent logs are `/tmp/jax2onnx-dropout-optimizer-astra-py{312,311}.log` and
`/tmp/jax2onnx-d2a-astra-py{312,311}.log`. Frozen reviews are
`/tmp/jax2onnx-dropout-optimizer-astra-review.md` and
`/tmp/jax2onnx-d2a-astra-review.md`; selected numerical records are
`/tmp/jax2onnx-d2a-numerical-evidence-py{312,311}.jsonl`.
These paths preserve local provenance; maintained checks use the linked test
modules with the repository's normal pytest setup.

The [work plan](../precision_implementation_work_plan.md) keeps broader matrix,
parameter/model, numerical and final-transition gates open. The next static
NCHW boundary packet is planned only. Symbolic layouts, normalization policy,
remaining operators and full-model/stochastic acceptance still need evidence.
