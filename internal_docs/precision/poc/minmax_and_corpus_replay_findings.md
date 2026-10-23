# Min/Max Lowering and Corpus Replay Checkpoint

2026-10-18, `precision-migration`, commit `fc3f9323`. Min/Max and repaired B3
passed independent review and main integration. Combined focused checks pass
323 per stack (86.81/41.90 seconds); the complete primary-stack run passes
**6,466 tests** (13,110 warnings, 413.65 seconds). All seven actual configured
hooks pass. Typing covers 542 files plus RNG; global Ruff and both inventory
CLI comparisons pass. The inventory records 203 files, 1,114 test functions and
362 migration additions. All 752 original anchors, 1,802 raw contracts/3,604
cells and optional entries are retained, with no removals.
The preceding shared-batch checkpoint `80cb68b2`/`658edbe9` passed 6,391 tests;
that historical result is unchanged.

## Finite Min/Max implementation

The pair-local [helper](../../../jax2onnx/plugins/jax/lax/_min_max_utils.py), called
by the existing Min/Max lowerers, repairs zero-tie signs under private SOURCE at
opset 23 for equal FP16/FP32/FP64 operands/results. It verifies canonical types and
broadcast shapes, uses fresh values and builder-owned scoped constants, and
preserves explicit source conversions. GatherElements selects `x+y` for Max
zero ties or `-((-x)+(-y))` for Min; other elements retain the original Min/Max
result. The only added Cast converts a Boolean mask to INT64.

Actual facade graphs contain 15 total nodes for Min and 12 for Max, with one
Min or Max operator. Scalars, empty results, broadcasting, shared constants,
returned aliases, weak FP16 promotion and one
named Function with actual conditional children are covered. Symbolic metadata
is preserved internally without admitting symbolic Min/Max. LEGACY, BF16,
integers and other opsets retain their previous route; their remaining coverage
is migration work, not a permanent fallback policy.

Owner and independent runs each pass the same **132 tests on each stack**: 29
new [lowerer cases](../../../tests/extra_tests/converter/test_source_min_max_zero_sign.py),
62 unchanged binder cases, 27 generated controls and 14 IR/builder policy checks.
Independent times are 40.45/39.36 seconds on JAX 0.11.2/0.10.2 and ORT 1.31.0/1.29.0.
Four-file and imported-source hashes match before/after. These counts overlap;
the separate integrated result is recorded above. Default admission is unchanged.

New zero-sign artifacts pass unchanged bitwise criteria against original native
eager/JIT references on CPU optimization disabled/all. The eight historical
signed-zero failures per stack remain recorded. Separate grid checks preserve
baseline bytes outside ties, including target NaN bytes, and native NaN
classification. FP16 has no non-NaN mismatch on this grid; each FP32/FP64 grid
retains **108 inherited non-NaN mismatches**, still FAILED. No native NaN payload
or signaling promise is added.

Four strict subnormal witness artifacts per stack also remain FAILED: Min
returns the negative minimum subnormal where native gives negative zero; Max
returns the positive minimum subnormal where native gives positive zero, for
FP32 and FP64. Schema/runtime checks pass and the numerical acceptance guard
raises. BF16 retains its measured CPU load failure. Diagnostic lowering reports
remain UNVERIFIED; the repair does not certify an entire family or model.

The [earlier finite cost audit](symbolic_batch_and_minmax_findings.md) remains the
cost evidence: dense selection took 1.242–1.542× baseline on one FP16 Min broadcast
workload, without a consistent sparse advantage. This is not a universal ratio
or unchanged low-precision performance. The choice/selector alone require
`(2w+8)N` logical bytes for N results of width w; allocator peaks were not measured.
The actual graph matches that construction, so no new timing search was run.

## B3 artifact-only revision

The tooling selects 13 unchanged raw rows and their 26 x64-off/on cells: five
dot, three matmul, four Add and one FP64 gather row. Four dot rows have explicit
SOURCE MatMul structural counterparts; original Gemm callbacks still run and
their failures/history remain separate. Other callbacks and missing callbacks
retain their original meaning. Fixed source, supplied bytes, original typed
traces, references and settings remain authoritative.

A separate negative reproduction exposed a preparation-path alias defect and
overwrote its own snapshot; earlier valid distinct-path owner records remain
intact. The repaired guard rejects identical, normalized-parent and
symlink-parent aliases before loading or execution. Owner checks pass 191 per
stack; independent review carries forward 188 prior checks and passes the three
new alias controls on final bytes. Counts overlap. The request-changes report
and negative witness remain; the overwritten snapshot is not acceptance evidence.

Fresh distinct-path side and main replays each check 24 artifacts per stack,
retaining gather-off's FP64/x64 conflict and gather-on's pending custom
`jax.numpy.sin` diagnostic boundary. Main replays pass all twelve invariants and
six prior comparisons per stack. Preparation bytes match their declared hashes;
selected source/reference/history identities match the earlier reviewed replay.

All 26 cells keep lowering, runtime and numerics UNVERIFIED, with zero ORT
constructor attempts. The criterion ledger is proposal-only: it retains
reference-first legacy versus actual-first SOURCE comparison, variant thresholds
and forced harness matrix-precision settings without adopting them. A validated
artifact or test pass does not approve a numerical criterion.

The original 36-cell batch, two-dot supplement and all-22 results remain unchanged.
The latter still has three checked artifacts and 19 pending rows per stack,
with zero ORT. All 310 symbolic and seven NCHW raw rows remain open. There is no
new default admission, public SOURCE mode or row closure by inference.

## Next work and provenance

The next accepted binder plan covers native SOURCE `jnp.sin`/`jnp.cos` opt-ins,
focused tests and two obsolete binary-test guards. Its separate side
implementation is excluded from this checkpoint. Native trig diagnostic forms
already exist; the fixed gather
pipeline is expected to reach a later custom-Where boundary. DOUBLE cosine's
existing half-angle arithmetic needs a separate audit. No pipeline/source
rewrite, table expansion or criterion change is implied.

A separate `source-corpus-fp32-conjunction-v1` mapping is now approved for twelve
unchanged strong-FP32 rows/24 cells per stack. It requires exact dtype/shape and
all three fixed tolerance comparisons against raw-source and separately labelled
historical-harness references. The historical None criteria, proposal ledger
and public SampleCriteria remain unchanged. Implementation is separate; actual
execution awaits its frozen-code independent review. No numerical outcome is
claimed as part of this artifact checkpoint.

Min/Max review: `/tmp/jax2onnx-minmax-lowering-final-astra-review.md`, SHA256
`7ef5ce7d7dc5eadbbf84698fa117563d0a8adbeeeb6a795e1341b3567ed407ba`.
Frozen patch: `3d09c0c0227d1671fe182fa04be9b66dc0c07e2139e08aea82b122152a5102a2`.
The owner manifest and independent verification index retain actual artifacts,
failed-sample payloads and both-stack import provenance. B3 follow-up review:
`/tmp/jax2onnx-b3-corpus-path-fix-astra-review.md`, SHA256
`18f09a0a5cceacc693eb97ca2010792b0dbc73b0a966505fe3647382b37b9cb1`;
final freeze `b726710a066b90587161ab9aafba5aa71dcf6f3462fb0b650e9ca4d7067236ce`.
Main full log: `/tmp/jax2onnx-corpus-minmax-main-full-py312.log`; actual hook log:
`/tmp/jax2onnx-corpus-minmax-main-code-commit.log`. All nine frozen code/test/inventory
hashes match after commit `fc3f932315a39e9e631e7fd0788a220e48440c77`. The separate
numerical approval is recorded in
`/tmp/jax2onnx-b3-fp32-numerical-criterion-approval.json`, SHA256
`40188c1f0aa34edb3324c8b28494d07226bf59f98898bbb3eeee382323a92ee0`.

Public LEGACY behavior remains unchanged during this private work. The
[work plan](../precision_implementation_work_plan.md) still requires one normal
source-preserving export contract across all existing functionality and fixed
sources, without lost baseline capabilities or a permanent mixed user experience.
