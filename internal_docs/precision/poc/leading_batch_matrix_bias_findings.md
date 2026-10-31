# Leading-B Matrix and Bias Findings

Status: committed finite private checkpoint `718f448e`, 2026-10-18. The seven-path
production/test semantics, separate v2 provenance/preparation repair and
actual-main binding have independent acceptance. Main checks pass 990 cases
on both supported stacks. The separately released v3 observation records ten
symbolic artifacts and forty actual runs per stack; every returned output passes
all three fixed FP32 comparisons. Independent review accepts these finite
observations.
Earlier preparation failures and the v3 wrapper’s post-return reporting error
remain recorded. The inventory is refreshed and both CLI comparisons pass.
The complete primary-stack run passes 6,816 tests; the code commit passes all
seven actual repository hooks. Public LEGACY behavior is unchanged.

The prerequisite [native Linear/reshape adapters](native_linear_reshape_findings.md)
remain committed at `791664f8`. The preceding
[ordinary-matrix audit](ordinary_matrix_admission_findings.md), `45881d8d`, retains
its own 750 main checks per stack, 6,728-test full pass, actual hooks and accepted
fixed FP32 observations. These results are historical evidence for their own
bytes and samples.

## Finite source contract

One actual native atomic B is allowed at axis zero of rank-two/three tensors;
all remaining extents are positive static integers. Symbol agreement uses the
actual `SymbolicScope`, before conversion to ONNX dimension names. Equal text
from different native scopes is not sufficient. Concrete samples must bind B
to one positive integer consistently across every declared occurrence.

| Operation | Audited extension |
| --- | --- |
| MatMul | `(B,K) @ (K,N) -> (B,N)` or `(B,M,K) @ (B,K,N) -> (B,M,N)`, with canonical contracting/batch axes and positive static K/M/N. Both operands and result have the same FP32 or FP64 dtype. |
| Add | Equal-rank operands with equal static tails and either the same native B or one explicitly static leading 1; either operand order is supported. |
| Bias reshape | The resolved native static `[N] -> [1,N]` reshape, positive N, unchanged floating dtype/weakness and the supported unpermuted parameter contract. |

The matrix path retains default precision and rejects unsupported algorithm,
placement, accumulation or unknown parameters. It emits one MatMul while
preserving the source type and operand ports. There is no new arithmetic Cast,
sample specialization, dummy bias or callback bypass. This scope does not
promise a particular hardware accumulation strategy or general matrix accuracy.
The old static ordinary-matrix and FP16/preferred-FP32 branches are unchanged.

The export/profile and the direct lowerer check their own typed boundaries.
Tests cover actual named/nested functions, captured state, aliases/cache order,
nonzero bias, FP64 mantissa preservation, deliberate source casts and weak
values. The runtime sample boundary now accepts rank three while preserving
strict shared-B and static-tail checks. Two inputs both declared B cannot use
B=1 and B=5 merely because a runtime could broadcast them; a declared static
leading one remains legal and must stay one.

Audited artifacts retain profile `source-static-v1` and report schema 1. Their
representation/lowering status is validated within this finite contract;
export alone leaves runtime and numerical status unverified. Normal public
`to_onnx` continues to use LEGACY behavior during migration.

## Five unchanged source anchors

The native feasibility audit retained one original callable/model per row and
x64 scenario, with symbolic and concrete typed traces and native eager/JIT
references at B=2 and B=5. All 20 native eager/JIT pairs per stack match bytes
for those samples. That is source-reference evidence, not ONNX runtime evidence.

| Raw identity | Original computation |
| --- | --- |
| `plugin::jax.numpy.matmul`, ordinal 6, `matmul_dynamic_a` | `(B,3) @ (3,4)` |
| `plugin::jax.numpy.matmul`, ordinal 5, `matmul_dynamic` | `(B,3,4) @ (B,4,5)` |
| `example::examples.gpt::GPT_broadcast_add`, ordinal 0 | `(B,4,5) + (1,4,5)` |
| `plugin::nnx.linear`, ordinal 0, `linear_symbolic_batch` | `(B,128)` through actual Linear kernel, native bias reshape and Add to `(B,64)` |
| `plugin::nnx.linear`, ordinal 3, `linear_no_bias` | `(B,128)` through the actual Linear kernel to `(B,64)`, with bias absent |

Both original NNX factories retain their default FP32 kernel and their original
bias choice. The same actual instance serves both batches, traces and export;
no alternate dtype factory or fabricated parameter replaces it. The original
bias is zero, so separate nonzero-bias regressions check preservation too.
The raw MatMul RHS and GPT singleton operand remain runtime ports with genuine
feed values. Their static shapes do not make their values constants.

Raw registrations, contracts, callbacks and `None` criteria retain their bytes
and meaning. The raw Gemm-oriented structural failures remain recorded. Exact
SOURCE Dot/Linear counterparts provide separately identified structural checks
for the real native trace, ports, wiring and parameter payloads. This does not
turn an unchanged raw callback failure into a raw callback pass.

## Provenance repair and rejected preparation evidence

The first seven-path review accepts the finite semantics but retains **900
passed / 22 failed out of 922 cases per stack**. All 59 new cases pass. The
22 failures arise from the original strict whole-module Dot provenance gate:
14 directly reject changed source bytes and eight fail their positive-control
prerequisite. Those results remain immutable and are not waived.

The separate four-path tooling packet versions the exact current Dot
counterparts to v2 and the literal four-control gate to v3. Whole-file/path,
raw-contract and typed-structure checks stay strict. The expected control digest
is a reviewed literal; validation does not derive its own expected value from
the record under test. The old thirteen-row selector, original numerical
helper, strict seventeen-source historical CLI and previous JSON remain
unchanged. The initial repaired selection passes all 987 permanent cases on
both stacks, including the original 22 failures.

Independent review then rejects one preparation defect. Supplied concrete
descriptors could conceal an actual symbolic `original_jaxpr`, allowing a
sample-specialized artifact to be attached to the wrong original evidence.
The v2 correction derives the validation decision from safely unwrapped actual
input, constant and output avals as well as supplied descriptions. Contradictory
symbolic evidence and malformed original bodies/avals fail preparation before
export, including when criteria are `None`. Existing count, scope, dtype,
weakness and concrete-reference checks remain intact. Only the support helper
and its new regression change after that rejected review; the accepted seven
paths and the current counterpart/probe versions stay byte-identical.

The v2 owner selection passes **1,004 cases on each stack**: **990 permanent
repository/generated controls plus 14 frozen independent review guards**.
These are overlapping reruns, not additional coverage to add across owner and
review executions. No skips, xfails or omissions are recorded. Configured
mypy passes 542 source files; recorded Black, Ruff, RNG and policy checks pass.
Five effective/hidden JAX settings, actual import/source bindings, original
callables and LEGACY policy are restored. Final independent delta review passes
all 17 selected durable/review guards per stack and accepts the repair.
Actual-main binding is also accepted. Root-owned main checks pass all 990
permanent cases per stack; the full 859-module import/source map, restored settings and
four literal v3 controls remain bound to the same frozen source. The separate
14 review guards are not counted as permanent main cases.

## Separately approved numerical observation

Criterion `source-leading-b-fp32-conjunction-v1` fixes all five raw identities,
x64-off/on scenarios, B=2/B=5 feeds and CPU optimization disabled/all. Per stack,
the complete denominator is **10 symbolic artifacts, 20 feed/reference pairs,
20 sessions and 40 actual runs**. One artifact serves both batches; no batch
specialization, warmup, retry or tolerance search is part of the observation.

Each returned output Y must have exactly the required output count, FP32 dtype,
shape and shared-B binding. The same captured Y must satisfy all three checks:
`allclose(Y, Rraw)`, `allclose(Rraw, Y)` and `allclose(Rharness, Y)`, each with
`rtol=atol=1e-5` and `equal_nan=True`. The comparator does not cast, flatten or
broadcast values. `Rharness` uses the same native instance and byte-identical
feeds with x64 disabled, matmul precision `float32`, standard dtype promotion
and allowed rank promotion. It is a labelled context reference, not the legacy
export or an alternate result chosen after observing Y.

All artifacts and references must complete preparation before target sessions
begin. Execution requires final accepted source/review bindings, fresh exact
controls and a separate recorded release. The observer’s static preflight
accepts its finite design but establishes neither execution release nor results.
The first two observer attempts retain preparation failures from tuple/list
serialization comparisons: v1 stops before export; v2 produces ten checked
artifacts per stack. Both retain zero sessions/runs and all 40 target slots
pending per stack. Source/artifact and array bytes remain unchanged. These
harness failures supply no production or numerical failure result.

The separately released v3 observer completes all ten artifacts, twenty
feed/reference pairs, twenty sessions and forty actual runs on each stack.
All forty slots are validated, with one run per slot and all 120 approved
same-output predicates passing. Source files, saved payloads and all recorded
settings are restored or unchanged. Independent review accepts the evidence and
recomputes all 120 predicates per stack from the saved outputs. All eight raw
Linear callback failures remain separate from accepted SOURCE counterparts.
The exported reports retain their original unverified runtime/numerical fields;
this separate observation supplies the finite execution evidence.
After the observer returns normally through its successful final guard, the
outer wrapper reads an obsolete `status` key and exits 1. The original error
and complete result files remain immutable; a separate metadata reconciliation
records this distinction without any further inference calls. The numerical
observation itself is not retried.

The original raw `None` criteria, old twelve-row approval, strict subnormal
failures, BF16 failures and their raising acceptance gates remain unchanged.
These finite FP32 CPU samples leave FP64/GPU runtime, broader inputs and
model-wide numerical acceptance unverified. They establish no general hardware
accumulation guarantee.

## Integration and remaining migration work

The complete primary-stack suite passes **6,816 tests**, with 13,130 warnings
in 431.98 seconds and exit 0. XML records zero failures, errors or skips; all
12 frozen code/inventory files and all 859 Python hashes remain unchanged.
Commit `718f448e23ba57ca155bde358fcde7954bb18755` passes all seven configured
hooks on the first attempt, without edits, skips or bypass. All twelve frozen
code/inventory files and all 859 Python hashes remain unchanged after hooks;
the checkout is clean at the recorded code checkpoint.

The refreshed inventory contains 209 files, 1,197 functions and 445 migration
additions, retaining all 752 original anchors
with zero removals and all 1,802 raw contracts/3,604 scenario cells. Only counts,
manual tests and their provenance change; the raw contract fields stay identical.
Both actual inventory CLI comparisons pass; the secondary retains the baseline
conditional `one_minus_square` entry rather than removing it. These comparisons
verify inventory identity, not SOURCE acceptance for every row.
Production/tooling, actual-main binding, numerical observations and the full
suite retain their separate evidence. These completed checkpoint gates do not
complete the overall migration.

The separate native Exp/weakness and actual-B Mul composition packet has
independent focused acceptance and awaits its own main integration.
Where/gather remains separate: custom Where and JIT/select trace differences and the existing x64 conflict require their own
evidence. They are unfinished work, not demonstrated impossibilities.

The five anchors do not close the 310-row symbolic inventory. Seven NCHW rows,
all 22 parameterized models with their actual runtime controls, 41 normalization
rows, 104 registrations without metadata, and prepared optional
MaxText/MaxDiffusion discovery still need acceptance accounting. Other symbolic
ranks/axes, affine symbols, general reshapes and contractions, and broader
framework compositions remain migration work. The
[implementation work plan](../precision_implementation_work_plan.md) retains one
normal source-preserving UX across the existing functionality and fixed source
implementations, including FP64. Both explicit `enable_double_precision` Boolean
values receive migration errors only at final phase F; there is no permanent
public mode or per-operation LEGACY fallback.

## Frozen evidence

These local artifacts record this checkpoint. Historical records retain their
own source bindings and outcomes; later acceptance must be recorded separately.

| Evidence | Local artifact and SHA256 |
| --- | --- |
| Native five-anchor feasibility | `/tmp/jax2onnx-leading-b-feasibility-audit.md`, `610f3124ffe9bebb279d93c80f673fbeaada7a544833be4549b94e2381ced03b` |
| Accepted seven-path semantics | `/tmp/jax2onnx-leading-b-astra-review.md`, `3d7fade7206f6669a135a488ee50b75f513cb9a617bfe46ab28bf3952a9991b2` |
| Seven-path source manifest | `/tmp/jax2onnx-leading-b-final-manifest.json`, `45d56238192cb8b44356991b2806f0ecf36574af020e3666ef33ed562a2fae8b` |
| Initial tooling rejection | `/tmp/jax2onnx-leading-b-tooling-astra-review.md`, `85f3ab149b5cd1cc1461c02ccc2427401eee56721f451dd856ee6c83cbee4375` |
| Final v2 owner manifest | `/tmp/jax2onnx-leading-b-tooling-final-v2-combined-manifest.json`, `564352bd0feae5d06060270c0bdbe102724b551bbbc0ad8f662f003a9bd43336` |
| Accepted v2 tooling | `/tmp/jax2onnx-leading-b-tooling-v2-astra-review.md`, `4180dfb9c5fc5fc300d68ab76fd5f7699b979210b6b2312da0f4fff732670d54` |
| Actual-main integration acceptance | `/tmp/jax2onnx-leading-b-main-integration-astra-review.md`, `39a7ec8527211e34a349eefbe05c5e25981e1239e914c096543230477a03c548` |
| Main checks and controls | `/tmp/jax2onnx-leading-b-main-integration-results.json`, `86f2d1a26e22ed247c115b4d31f3cbbc4bd07b44e40dbd1deb84c6d10337daf1` |
| Preserved histories and both observer attempts | `/tmp/jax2onnx-leading-b-prior-history-v4.json`, `f8ab35f0465c48f52445937e92ab541557ee28efe63ab236b071dd859b6250e1` |
| Independent numerical acceptance | `/tmp/jax2onnx-leading-b-main-runtime-astra-review.md`, `1ca74875bf9e6b7c5974bc488f9142d974682c154dac1b798bf8eaabb66ab53a` |
| Complete v3 observations and retained wrapper error | `/tmp/jax2onnx-leading-b-main-runtime-reconciliation.json`, `a296fb762457d2c11bfd078ddb7f1f7d906d657896fe4dca3e64f2d81f573aed` |
| Both inventory CLI comparisons | `/tmp/jax2onnx-leading-b-main-inventory-verification.json`, `7352c8bac5b06199932ef3bdc25a077d35046aa66ba20943f7ac2a83e62fa021` |
| Committed code, actual hooks and final frozen hashes | `/tmp/jax2onnx-leading-b-main-code-checkpoint.json`, `3a0ca1d511986f568fd67ba95a16e71492b47134950c174ad76369c10cc059f9` |
| Complete primary-stack run | `/tmp/jax2onnx-leading-b-main-full-result.json`, `d8642ecfcc155356d35d68c0984a73afcabc0f77cbef6e4384bf563195b0e359` |
| Twelve-path code/inventory freeze | `/tmp/jax2onnx-leading-b-main-full-freeze.json`, `3547078b413e885f4fcb4d90de7a0cda59a8d82c43db6f230025275f738dbff8` |
| Fixed numerical approval | `/tmp/jax2onnx-leading-b-numerical-criterion-approval.json`, `2ffb31beab2c6a972acc7053fbceb5b66836a7527e72ae69cd311ab5b2b97cb3` |
| Corrected v3 observer static acceptance | `/tmp/jax2onnx-leading-b-observation-v3-astra-preflight.md`, `85b5f0ab704c43d2b80776d6aec0da7d33da5d41ba9025a90b0d7a35ad452a74` |
| Original observer static preflight | `/tmp/jax2onnx-leading-b-observation-astra-preflight.md`, `95df222665e83e5a1633734533428ec42a759deb76c5f1a674bdb429392be083` |
