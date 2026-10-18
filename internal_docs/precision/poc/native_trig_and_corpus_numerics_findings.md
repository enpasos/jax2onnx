# Native Trig Binding and Fixed Corpus Numerical Findings

Date: 2026-10-18. Native SOURCE sine/cosine dispatch and a separate, preapproved
FP32 corpus comparison have passed independent review. Main focused checks pass
on both supported stacks. Commit `d2292b89` passes all seven actual configured
hooks, and the fresh integrated full run passes **6,579 tests** after the
reviewed facade-assertion repair. The failed first run and preceding `fc3f9323`
checkpoint's 6,466-test result remain historical evidence.

This work changes no default operation admission or public export option. The
normal public facade still uses LEGACY. The intended final interface remains
one source-preserving call across existing functionality, including unchanged
externally supplied JAX sources. The bounded results below do not complete that
migration or turn temporary exclusions into technical impossibilities.

## Native binding and the unchanged gather source

The two production changes in `jax2onnx/plugins/jax/numpy/sin.py` and `cos.py` opt into
the existing captured-original SOURCE binding. Original arguments reach native
JAX before custom normalization, preserving native promotion, weak typing,
settings and errors. Other bindings and LEGACY retain their prior behavior.
The new trig regressions and the precise old binder-guard updates cover typed
traces, source constants, local scopes, cache order, differentiation and batching.
Actual main runs pass **553 tests per stack**; the independent review verifies
the recorded imports, hashes, settings, logs and XML.

A new revision observation replays only `plugin::gather`, ordinal 0, under its
two original scenarios. The original source, feeds, references and settings
retain their historical fingerprints:

| Scenario | Observed boundary | New artifact/runtime evidence |
| --- | --- | --- |
| `raw-x64-off` | Original typed-FP64 input conflicts with disabled x64. | None. |
| `raw-x64-on` | Diagnostic validation stops at `scope 0/equation 11 (jax.numpy.where)`. | None. |

The first eleven equations, including native Sin/Cos, match exactly between the
native and activated traces. Root types, weakness, effects, constants and final
output signatures also match. The complete traces differ: the native last
operation uses JIT `_where`, while activation retains the custom Where adapter.
No diagnostic rule is bypassed. Both cells keep schema, artifact, lowering,
runtime and numerical gates unverified, with **zero ORT attempts**. This is
progress through a patch-induced barrier, not a new accepted gather artifact.
DOUBLE cosine's half-angle lowering and broader numerical forms remain unaudited.

The primary observer compared decoded JSON lists with live tuples and exited 1
after saving the evidence. Its original driver, log, false flags and supplement
remain unchanged. A separate persisted-JSON correction proves the exact prefix
and typed boundaries without rerunning the source or export. The secondary v2
observer performs that normalization and exits successfully. Neither result is
described as whole-trace equality or a successful primary driver run.

## Approved numerical counterpart

The separate three-file tooling packet adds
`tests/extra_tests/converter/source_precision_corpus_numerics.py`,
`scripts/source_precision_numerical_probe.py` and
`tests/extra_tests/converter/test_source_precision_corpus_numerics.py`.
It consumes the accepted artifact records and preserves their immutable
preparation bindings. Main focused runs pass **247 tests per stack**
(18.31/16.16 seconds in pytest summaries). These checks overlap prior controls; their
counts are not added to earlier test totals.

The predeclared selection contains twelve original rows:

| Original registration | Ordinals | Original off/on cells |
| --- | --- | ---: |
| `plugin::dot_general` | 0, 1, 2, 3, 6 | 10 |
| `plugin::jax.numpy.add` | 0, 1, 2, 3 | 8 |
| `plugin::jax.numpy.matmul` | 2, 4, 8 | 6 |

The rows remain distinct, including identical-looking contracts. Each selected
trace is strong FP32 under both original x64 settings; x64-on is not FP64 source
coverage. `source-corpus-fp32-conjunction-v1` was approved before target outputs.
It requires exact output count, FP32 dtype and shape, followed by all three
predicates with `rtol=atol=1e-5` and `equal_nan=True`:

1. `allclose(Y, Rraw)`.
2. `allclose(Rraw, Y)`.
3. `allclose(Rharness, Y)`.

All three consume the **same retained actual target output Y**. `Rraw` is the
unchanged original native source reference. `Rharness` evaluates that same
native callable on the same feeds under x64 disabled and matmul precision
`float32`. It is a separately labelled historical-context reference, not a
legacy ONNX output or the original legacy random-input run. Raw and historical
reference bytes happen to agree for these fixed samples; no general equivalence
of the two settings is inferred. Existing `SampleCriteria` behavior is unchanged.

An explicit release authorized one packet per stack. Each completed packet has
**24 prepared cells, zero pending cells and 48 actual CPU runs**, one run per
cell with optimization disabled and one with all optimizations enabled. Every
output passes the exact contract and all three comparisons. All seven packet
invariants pass, including unchanged code, historical files and preparation,
restored caller settings and bounded execution. The independent post-run review
recomputed **288 predicates over 96 recorded outputs**, checked 469 evidence
hashes and accepted the observations without another runtime or source execution.

| Stack | JAX / NumPy | ONNX / ONNX Runtime | Recorded target |
| --- | --- | --- | --- |
| Python 3.12 | 0.11.2 / 2.5.4 | 1.23.2 / 1.31.0 | CPU, optimization disabled/all |
| Python 3.11 | 0.10.2 / 2.4.6 | 1.22.0 / 1.29.0 | CPU, optimization disabled/all |

Both targets use CPUExecutionProvider only, disabled fallback, sequential
execution, one thread per pool and disabled memory patterns. Runtime and
numerics are validated **for these samples and settings**. General diagnostic
lowering remains UNVERIFIED; neither operators nor models gain default admission.

## Retained failures and remaining denominator

The historical artifact packets, raw None criteria and proposal ledger remain
unchanged. In particular, the eight raw Gemm callback failures per stack remain
failed beside their separately reviewed SOURCE MatMul structural counterparts.
The new numerical results do not rewrite the original 36-cell batch, its two-dot
supplement, the 26-cell replay or the gather observations.

The approved tolerant comparison permits NaN and signed-zero equality. It does
not discharge the stricter failed FP16 rounding, linspace, matrix-subnormal,
Min/Max subnormal/grid or historical signed-zero criteria. Those artifacts,
references and outcomes remain recorded in their dated findings.

The remaining **3,580 unselected cells** and **126 recipe-pending scenarios**
retain their boundaries. All **310 symbolic rows**, **seven NCHW rows** and
**22 fixed-source model rows** remain open; the model packet still has three
artifact-checked and 19 pending rows per stack, with zero ORT. No row closes by
inference from primitive tests or these twelve numerical samples.

## Validation, provenance and next work

Typing (542 files plus RNG), global Ruff and both actual inventory CLI
comparisons pass. The manual inventory records **205 files, 1,146 functions and
394 additions**, preserving all 752 original anchors, zero removals and all
1,802 raw contracts/3,604 cells. The older stack retains its conditional
`one_minus_square` entry.

The first integrated full run ends with **6,577 passed, two failed**, 13,110
warnings and 397.31 seconds. Both failures are existing facade assertions that
expect empty precision metadata; native Sin correctly reports `accuracy=None`.
The original log/XML and eight-path freeze remain at
`/tmp/jax2onnx-trig-numerics-main-full-py312.log` and
`/tmp/jax2onnx-trig-numerics-main-full-freeze.sha256`. A one-file facade-test
repair now asserts exact native `sin` and diagnostic parameters
`{"accuracy": "None"}`, replacing the old primitive alternative/empty metadata
expectations. Owner and independent selections each pass 159 tests per stack;
the independent runs take 17.69/17.20 seconds with eight warnings and no setting
leaks. The review is `/tmp/jax2onnx-source-facade-native-fix-astra-review.md`.
No production change or numerical-packet rerun was needed.

The fresh integrated primary-stack run passes **6,579 tests**, with 13,110
warnings in 384.83 seconds and exit zero; log
`/tmp/jax2onnx-trig-numerics-corrected-full-py312.log`. Commit `d2292b89` passes
all seven actual configured hooks, recorded in
`/tmp/jax2onnx-trig-numerics-code-commit.log`. All nine frozen code/test/inventory
hashes match after commit; `/tmp/jax2onnx-trig-numerics-code-checkpoint.json`
retains the completed facts. These results add no default admission or broader
numerical guarantee.

The frozen local records retain exact commands, imports, preparation and target
bindings. They are local review artifacts, not packaged documentation assets:

| Record | SHA256 |
| --- | --- |
| Primary numerical result, `/tmp/jax2onnx-corpus-numerics-main-py312-packet-result.json` | `c855571310545d4c603b4e173ea4c84169b6a7c9a63e1a48c8c1f821d15da898` |
| Secondary numerical result, `/tmp/jax2onnx-corpus-numerics-main-py311-packet-result.json` | `792e83a5348d0afc4cbf6fbaf7f96e7dc70cf3ca5538f05b873449537f9b9ade` |
| Final post-run review, `/tmp/jax2onnx-corpus-numerics-main-astra-review.md` | `a233f9efbc0013565787a180c21e30f7d636216104c276cd88076aa04a5f7c2a` |
| Final offline verification, `/tmp/jax2onnx-corpus-numerics-main-astra-verification.json` | `2ca17383b4274e4a3875abe2bbb39606eece4e5c18e656eb6f375bf288179e69` |

The separate trig review is
`/tmp/jax2onnx-native-trig-main-astra-review.md`; its supplements preserve the
primary observer failure and separate correction. Initial numerical review
records remain under their `-initial` names; the hashes above identify the final
review. The manual inventory hash is
`7c24e8fb261f5f53920f8e1f81c7dd4c66b25325a45a2a3ec005b4abddea9041`.

Next comes the accepted six-path audited ordinary-matrix admission packet,
with precise obsolete-guard maintenance and fresh audited numerical observations.
Its isolated implementation has started from `d2292b89`; it has no implementation
acceptance or main admission change. The existing numerical CLI and historical records stay pinned to their reviewed
revision. The three-path native NNX Linear/reshape prerequisite is independently
accepted in its isolated checkout: 544 tests pass per stack (102 warnings,
65.15/67.24 seconds in pytest summaries), including 41 new cases. It awaits main integration and is
not part of `d2292b89`. Its review is
`/tmp/jax2onnx-native-linear-reshape-astra-review.md`; static diagnostic artifacts
and native symbolic traces add no numerical or profile admission. These steps support a later
shared leading-B matrix/bias-broadcast packet through the five unchanged MatMul,
GPT Add and NNX Linear anchors, including bias and no-bias forms. Native tracing
feasibility does not replace symbolic artifact, binding or numerical evidence.
The [work plan](../precision_implementation_work_plan.md) retains the full
existing-functionality completion contract.
