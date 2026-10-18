# Audited ordinary matrices and fresh numerical observations

Date: 2026-10-18. Commit `45881d8d` contains the independently accepted finite
implementation. The actual-main checks, numerical post-run review, 6,728-test
full run and all seven actual code hooks pass. Public export remains LEGACY; the private
`source-static-v1` identifier and report schema are unchanged. The preceding
[native Linear/reshape checkpoint](native_linear_reshape_findings.md),
`791664f8`, remains a committed prerequisite.

## Admitted representation

The actual SOURCE audit now admits effect-free native `dot_general` with two
same-type FP32 or FP64 operands and one same-type result, equal ranks 2–4 and
strictly positive static dimensions. Each operand has one contracting and one
free axis; its remaining axes form ordered, matched batch pairs. Axis validity,
uniqueness, disjointness, paired sizes and the actual output shape are checked.
Precision must be absent/None or a native DEFAULT pair; a preferred result is
absent/None or the same dtype. Placement is absent/None only. Unknown parameters,
explicit algorithms and other precision forms remain excluded.

The existing lowerer permutes the left operand to batch/free/contract order and
the right to matched-batch/contract/free order using fresh values, then applies
same-type MatMul. These permutations are bijections; equal paired batch sizes
prevent implicit broadcasting, and output order matches the native contraction.
Physical dtype/shape checks precede emission. Canonical operands, captured
payloads, explicit casts, weakness and scoped source settings remain intact.
This is a tensor-representation argument, not a promise about hardware
accumulation order, subnormal handling or NaN payloads.

Production changes are confined to `converter/source_profile.py`. The other five
packet paths maintain live corpus controls and regression assertions, including
`test_source_matrix_admission.py`. Matrix lowerers and native JNP binding are
unchanged; the existing FP16-operands/preferred-FP32 audit branch retains its
AST. Symbolic/unequal-rank/broadcast matrices, vectors, general contractions,
zero extents, ordinary FP16/BF16, mixed types and non-None placement remain
implementation/audit work, not demonstrated ONNX impossibilities.

## Regression and boundary evidence

| Execution | Python 3.12 | Python 3.11 |
| --- | --- | --- |
| Independent isolated review | 750 passed, 73 warnings, 54.61s | 750 passed, 64 warnings, 51.34s |
| Actual-main checks | 750 passed, 74 warnings, 82.97s | 750 passed, 65 warnings, 54.32s |

Durations are pytest summaries. Each selection contains 99 new-module cases,
115 unchanged ordinary-matrix cases, 34 FP16 cases, 45 generated controls and
457 related controls; repeated runs are not additive coverage. Tests use the
real audited facade without a diagnostic validator. They cover native lax/JNP
entry, arbitrary valid axes/batch order, explicit casts, weak values, constants,
returned aliases, named/nested functions, both conditional branches, IR/proto/file
outputs, local settings and successive SOURCE/LEGACY calls. Actual IR child
formal types are checked directly; FunctionProto does not encode those fields.
The five effective/hidden JAX settings, callable identities and policy restore
around the actual-main checks; imported source hashes match the accepted packet.

Preparation failures remain recorded: an opaque-object comparison error, a
pre-existing LEGACY local-x64 leak reproduced separately with negation alone,
and invalid test assumptions about typed FunctionProto formals. Corrected tests
retain SOURCE restoration and inspect actual IR. The original conditional review
also required the new test's path-comment header. The accepted V2 delta is an
exact prepend with unchanged AST; marker/Black/Ruff checks pass, and the main
750-case runs execute the final header-bearing file.

Four live controls, Add0 and Dot0 with x64 off/on, use the fixed revision token
`source-static-add-ordinary-dot-v2`. Each exports through actual audited
admission. Dot0 requires the separately reviewed SOURCE MatMul structural
counterpart and retains its original Gemm callback as FAILED. These controls
create no ORT sessions; their export reports validate representation/lowering
and leave runtime/numerics UNVERIFIED. The historical numerical CLI still
rejects the changed 17-source binding. Its helper, criteria and old records
were not loosened to accommodate this revision.

## Fresh audited sample results

A separate, explicitly released root orchestration creates 24 new audited
artifacts per stack and runs each once per CPU optimization mode, disabled/all:
48 observations per stack, zero pending cells or preflight errors, and all seven
packet invariants true. The exact twelve original rows are dot_general ordinals
0/1/2/3/6, JNP Add 0/1/2/3 and JNP MatMul 2/4/8, each with x64 off/on. Eight rows
exercise the new matrix admission; four Add rows retain existing admission.
All computations remain strong FP32 in both x64 scenarios.

Every original feed, raw reference (`Rraw`) and historical-context reference
(`Rharness`) is byte-identical to the retained earlier packet. The unchanged
`source-corpus-fp32-conjunction-v1` criterion first requires exact output count,
FP32 dtype and shape, then all three comparisons against the same retained `Y`:
`allclose(Y, Rraw)`, `allclose(Rraw, Y)` and `allclose(Rharness, Y)`, each with
`rtol=atol=1e-5` and `equal_nan=True`. Rharness uses the unchanged native callable
on the same feeds under x64 disabled and matmul precision `float32`; it is not a
claim to reproduce unrelated historical random draws.

Independent offline post-run review fully checks all 48 fresh model artifacts
and recomputes all **288 predicates on 96 recorded outputs** across the two
stacks. All pass. CPU-only targets are ORT 1.31.0/1.29.0, sequential execution,
one inter/intra thread, memory patterns off and fallback disabled. Each recorded
cell/mode has one session run attempt and one success. No retry or replacement
output is used to obtain acceptance.

| Stack | Fresh result SHA256 |
| --- | --- |
| Python 3.12 | `1e586e720122810cef56f6b265939f752eb83d7ebcbccfaa13a69f5692a72c38` |
| Python 3.11 | `5386ddddd9c6db01a20deee32f67b00df243cf82833b9ed6f6ac563fd81b99f8` |

The new export reports carry audited representation/lowering VALIDATED;
runtime/numerics remain UNVERIFIED at export and become VALIDATED only in the
separate sample-result records. Old diagnostic artifacts and their UNVERIFIED
lowering are unchanged. Raw structural outcomes remain eight FAILED, twelve
passed and four absent per stack; the eight Gemm failures retain their explicit
SOURCE counterparts. Original `None` criteria, proposal ledgers and previous
numerical outputs are preserved, not relabeled as audited evidence.

The fresh FP32 samples establish no universal matrix, FP64 or model numerics.
Four audited strict-subnormal target observations per stack still fail
(FP32/FP64, both CPU modes), with exact hashes and raising acceptance gates
retained alongside their diagnostic counterparts. Earlier FP16 rounding,
linspace and Min/Max failures remain in their dated findings.

## Checkpoint and remaining work

The manual inventory is **207 files, 1,175 functions and 423 additions**. Only
its three manual accounting sections change; all 752 original anchors, zero
removals, 1,802 raw contracts and 3,604 cells remain intact. Both actual inventory
CLI comparisons pass. Inventory application follows the completed numerical
packets; it is not represented as part of their earlier six-file runtime state.
Typing (542 files plus RNG), global Ruff and pinned-source Black pass. The
Black package's development version label is resolved by comparing all 43
source files to configured Black 25.1.0 commit `8a737e727ac5ab2f1d4cf5876720ed276dc8dc4b`;
no formatter upgrade or hook bypass was used.

The integrated primary-stack run passes **6,728 tests**, with 13,120 warnings
in 370.40 seconds (pytest summary). All 24 frozen source hashes match after
the run and after all seven actual code hooks, which pass on the first attempt.
Code is committed at `45881d8d7c6f6bfab7c3b4a44979515c484e3156`. The preceding
6,620-test full pass belongs to the Linear/reshape checkpoint. Documentation
build, hooks and commit remain separate root gates.

The next seven-path isolated code work adds shared leading-B MatMul, leading-1
Add and static bias reshape through five unchanged MatMul/GPT Add/NNX Linear
sources. It remains isolated implementation in progress; no corpus-helper
expansion, new corpus result or leading-B numerical acceptance is claimed. Its approved
numerical criteria stay unchanged. All 310 raw symbolic rows, seven NCHW rows,
126 pending recipes and 3,580 unselected corpus cells retain their open status.
The 22-model batch remains three checked artifacts and 19 pending rows per
stack with zero ORT. The goal remains one normal source-preserving public UX
covering all existing functionality and fixed sources without capability loss.

Local evidence: `/tmp/jax2onnx-ordinary-matrix-main-integration-astra-review.md`
(SHA256 `e892a298d112f66b08c20ce0e6d9d2b6290d3f89cda7b90d3999dfc8351321c4`),
`/tmp/jax2onnx-audited-matrix-main-runtime-astra-review.md`
(SHA256 `a8875deb5ccdac254531738fb5579ee446d7a76821c0ec492dd2a8f54c6d4249`),
the `preparation.json`, `result.json` and `execution.json` records under
`/tmp/jax2onnx-audited-matrix-main-runtime-py{312,311}/`, and
`/tmp/jax2onnx-black-pinned-source-provenance-20261018.json`. The final code
checkpoint is `/tmp/jax2onnx-ordinary-matrix-admission-code-checkpoint.json`
(SHA256 `f2a9e8921c2931c770fdbabbb2653cb9c085445288992bd344310e92ae8cc3c9`);
its full-run and hook evidence remains separately retained.
