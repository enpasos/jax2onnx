# Native NNX Linear and Reshape Findings

Date: 2026-10-18. The independently accepted three-file adapter packet is committed
at `791664f8`. Both supported stacks pass 544 focused checks; the complete primary
run passes 6,620 tests. All seven actual hooks pass after the comment-only
path-header normalization recorded below.
The preceding [trig/numerical checkpoint](native_trig_and_corpus_numerics_findings.md),
`d2292b89`, and its completed 6,579-test run remain historical evidence.

## Implemented boundary

Under SOURCE, `nnx.Linear.__call__` and `jnp.reshape` forward their original
arguments exactly once to the captured native implementation, before custom
normalization or parameter extraction. Native argument forms, promotion, weak
typing, precision, exceptions and deliberate casts therefore reach the existing
JAX primitives. The complete LEGACY signatures and bodies remain unchanged.
The shared binder, registration, lowerers, profile, diagnostics and public API
are untouched; no new precision mode or permanent fallback is introduced.

The production paths are `jax2onnx/plugins/flax/nnx/linear.py` and
`jax2onnx/plugins/jax/numpy/reshape.py`. The new regression module is
`tests/extra_tests/converter/test_source_nnx_linear_reshape_dispatch.py`.
This removes the custom Linear/dummy-zero and custom bias-reshape barriers for
the original `plugin::nnx.linear` ordinals 0 and 3. Their unchanged
`Linear(128,64)` recipes retain requested dtype handling, seed 0 and exact FP32
kernel/bias payloads under both original x64 contexts.

## What the checks establish

The 41 new cases exercise the original bias/no-bias models and a separately
identified nonzero-bias control. Native and activated complete typed traces,
constants and respective eager/JIT results agree. Biased computation remains
`dot_general → reshape → add`; no-bias computation remains `dot_general`, with
no synthetic bias operand. Source settings and callback choices are retained.
The symbolic B cases establish **native tracing only**; concrete B=2/5 cases are
separate static diagnostic exports, not one accepted symbolic artifact.

Actual SOURCE exports use the existing `DiagnosticValidator` and full checker.
They preserve MatMul, required bias addition, original parameter bytes and
IR/protobuf/file boundaries. A real named Function retains MatMul/Reshape/Add,
owned child values and returned kernel/bias aliases under local settings.
Constant-folded bias storage does not alter the original or returned parameter.
Native reshape argument/error behavior, differentiation, batching, weak input,
explicit dtype/precision and captured-reference controls are also covered;
trace controls do not admit extra lowerer forms. Nondefault dot precision
remains an actual existing-lowerer rejection.

Native→LEGACY→SOURCE→native and reverse cache sequences run without clearing
caches, including two parameter-distinct models and named reuse. Nested
activation and failures restore public callables and policy. The test fixture
restores all five JAX global settings and effective enclosing contexts; the
main runner separately verifies matching effective/global settings before and
after the complete selection.

The new module denies `onnxruntime.InferenceSession`. Its reports remain
`source-diagnostic-v1`, with lowering, runtime execution and numerical acceptance
UNVERIFIED. Existing generated LEGACY runtime controls keep their original
criteria; their passes add no SOURCE Linear numerical acceptance.

## Validation and retained preparation failures

| Evidence | Python 3.12 / JAX 0.11.2 | Python 3.11 / JAX 0.10.2 |
| --- | --- | --- |
| Independent isolated-checkout selection | 544 passed, 102 warnings, 65.15s | 544 passed, 102 warnings, 67.24s |
| Actual-main selection | 544 passed, 102 warnings, 90.93s | 544 passed, 102 warnings, 66.72s |

Times are pytest summaries. Each selection contains the same 41 new cases,
72 unchanged generated Linear/reshape controls and 431 adjacent matrix,
binary/trig, diagnostic, Function/context, symbolic, AD and policy checks.
These overlapping selections are not additive coverage counts. Main records
retain exact imports, all 13 owned/predecessor/critical hashes, test IDs/results,
and before/after state; all match the accepted candidate and restore correctly.

The initial owner run remains recorded as 35 passes and three failures. One
callback assertion assumed keyword dimension numbers although native NNX
supplied them positionally; the corrected assertion checks that actual call.
Two new LEGACY named multi-result compositions exposed an existing Function
`multiple_results` error. A separate lax.neg-only witness, which invokes neither
changed adapter, reproduces it on both stacks. The approved single-result named
Linear covers cross-policy cache reuse, while the SOURCE multi-result named
alias artifact remains required and passes. No unrelated Function repair,
flattened source or weakened diagnostic gate is part of this packet.

Typing (542 files plus RNG) and global Ruff pass. The manual inventory records
**206 files, 1,159 functions and 407 additions**, retaining all 752 original
anchors, zero removals and all 1,802 raw rows/3,604 cells. Raw/optional sections
remain unchanged. Both actual inventory CLI comparisons pass, retaining the
older stack's conditional `one_minus_square` baseline entry. The complete
primary run passes **6,620 tests, 13,120 warnings in 452.13 seconds**, exit zero;
log `/tmp/jax2onnx-native-linear-reshape-main-full-py312.log`.

The first actual commit attempt failed only the new test's required path marker;
its six other hooks passed. `/tmp/jax2onnx-native-linear-reshape-code-commit.log`
retains that failure. The only correction prepends the repository path comment
and a blank line. Exact suffix equality and AST identity are recorded in
`/tmp/jax2onnx-native-linear-reshape-marker-normalization.json`. The test hash
changes from `f95e366ab93fe5bddf5893ef56752c9cb7460009457c0d35404cb771184a07f7`
to `5d09cc55d62ba39583653408930d34b3f6da7162a8c6b8c4ba03d58c603bbdde`;
production and inventory bytes are unchanged. The full/focused runs belong to
the pre-header freeze, `/tmp/jax2onnx-native-linear-reshape-main-full-freeze.sha256`;
the committed bytes are bound by
`/tmp/jax2onnx-native-linear-reshape-main-final-freeze.sha256`. No functional
rerun was needed for the comment. The retry log,
`/tmp/jax2onnx-native-linear-reshape-code-commit-retry.log`, records all seven
actual hooks passing and commit `791664f8cb469601c4d07d6b161d01a158832069`.

Local evidence records, with exact commands and payload/source bindings:

| Record | SHA256 |
| --- | --- |
| Independent review, `/tmp/jax2onnx-native-linear-reshape-astra-review.md` | `5dbb0f6c0c89cbfe1009ddfb5ee82844d80b6f42f987cdce8c556172351eb61e` |
| Main Python 3.12, `/tmp/jax2onnx-native-linear-reshape-main/py312-execution.json` | `20115082ff5c30efb5d9f8fc5e5c22934cc8c798dbbb471dca0742dbbabb80fc` |
| Main Python 3.11, `/tmp/jax2onnx-native-linear-reshape-main/py311-execution.json` | `ab72ed1d6f8d2df051629b1ec191e8d134d29f95b970d84338b25c06a3aa8ad4` |

`/tmp/jax2onnx-native-linear-reshape-main-preparation.json` binds those records
to the three accepted source/test hashes and the manual inventory hash
`c757e3667c33f48e560f6759be4e701a13f4d5870e3527c8638425eeb76c88c9`.
The completed code checkpoint is
`/tmp/jax2onnx-native-linear-reshape-code-checkpoint.json`, SHA256
`9ddc74a53cfb4602577160ba7df2f4928e86f95eef60334a676a715a850ea0ac`.
The full log SHA256 is
`889b2e558366a89dd3565e2fa2e887163ed6333768983d87c8077aa8c1495c96`.
These are local review records, not packaged documentation assets.

## Remaining implementation

Ordinary-matrix audited admission is a separately accepted six-path implementation
in an isolated checkout at `d2292b89`; it is not integrated into main. Its
comment-only header delta has passed independent review. Fresh audited numerical
observations have not run and remain separately controlled, with
historical numerical tooling and results pinned. The later shared leading-B
matrix/bias-broadcast packet still needs symbolic artifacts, strict binding,
a substantive bias-reshape audit and scoped numerical evidence through the
five unchanged MatMul, GPT Add and NNX Linear anchors.

All 310 symbolic rows, seven NCHW rows and the 22-model requirements remain
open. The model packet still records three checked artifacts and 19 pending
rows per stack with zero ORT. Earlier 96-output/288-comparison numerical evidence
retains its original scope; no criterion, historical failure or raw-corpus row
is changed by these adapter tests. Public LEGACY remains unchanged. The final
goal remains one normal source-preserving interface across all existing
functionality and fixed supplied sources, with no lost baseline capability.
