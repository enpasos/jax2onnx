# Shared Symbolic Batch and Min/Max Feasibility Findings

Date: 2026-10-18. Shared-batch code checkpoint `80cb68b2` on `precision-migration`,
after `d1dd3162`.
The finite shared-batch implementation has passed owner and independent focused
checks on both stacks. The complete primary-stack run passes 6,391 tests and
the code commit passed all seven actual configured hooks. Min/Max
selection evidence is feasibility only; the next accepted tooling direction is
a bounded artifact-only corpus replay. Public export remains LEGACY. None of
these results completes the uniform source-preserving migration or closes a
raw model row by inference.

## One shared batch through audited Add

The private path now admits two equal-shaped floating inputs `(B,K)`, with one
shared positive atomic batch symbol and positive static `K`, for audited Add
and actual named Function composition. The unchanged source exports once; the
same FP32 artifact validates exact original native eager/JIT references at
`B=2` and `B=5`, with CPU optimizations disabled/all and an unchanged artifact
digest. There is no hidden re-export or per-sample specialization.

Native symbol objects retain their actual unconstrained scope. Mixed native
and string declarations share that verified scope in either input order;
unrelated scopes cannot be merged merely because their printed names agree.
Named Function formals/results retain correctly remapped symbol origins and
independent IR value ownership. The implementation does not parse expression
representations to infer an atomic symbol.

The same logical/physical contract is checked in input preparation, the core,
audited profile, IR/protobuf/standard-file reload and JSON reports. Dimensions
are truthful integers or symbol strings, with exact `dim_value`/`dim_param`
checks. Names, aliases, order, dtype and source weakness remain visible. Actual
child settings and SOURCE/LEGACY/SOURCE cache sequences have controls without
cache clearing. The existing private profile identifier `source-static-v1` and
report schema remain unchanged; the identifier now documents this bounded
extension rather than exposing another user choice.

The [maintained tests](../../../tests/extra_tests/converter/test_source_symbolic_batch.py)
also establish selected exact FP16 and mantissa-sensitive FP64 samples; FP64
with x64 disabled fails before tracing. BF16 artifacts pass checker/report
checks, but both installed CPU runtimes fail Add session loading with
`NOT_IMPLEMENTED`. The evidence remains FAILED and the acceptance guard raises;
no dtype replacement or comparison relaxation is used.

## Symbolic annotations need sample contract checks

Raw ORT accepts `B=1` for one input and `B=5` for the other and broadcasts them
for this artifact. That execution is outside the exported shared-B contract.
The SOURCE sample path therefore checks the exact ordered report/artifact
signature and binds each input symbol before runtime loading/session creation
and before every run. Each sample gets a fresh positive binding for `B`.
Reference and actual output shapes must match that binding too.
The actual-output mismatch control injects an invalid session result; it tests
the validation mechanism and does not identify a new ORT output defect.

Unequal batches, zero B, wrong K/rank/dtype, forged reference batches and changed
report symbols fail their appropriate input/shape contract. Invalid samples do
not execute; mixed invalid/valid sequences execute only their valid B=2/5
samples and retain a failed aggregate gate. Numerical acceptance remains
unverified for samples that never run. Unknown artifact dimensions do not pass
as an exact symbolic declaration. No ONNX runtime Assert machinery is added;
artifacts outside this relevant SOURCE report contract retain prior handling.

This is an Add/named-composition foundation. Dynamic NCHW, operand broadcasting,
affine or constrained expressions, independent/nonleading symbols, new output
symbols, other symbolic operations and control flow remain excluded pending
their own implementation. Existing static zero extents, static discrete forms,
Function/cond/loop controls, C4b layouts, named BOOL inputs and positive dynamic
LEGACY witnesses remain covered. These temporary gaps are not ONNX impossibility
claims or a permanent fallback plan.

All **310 raw symbolic rows remain open**. The only raw symbolic declaration
with two inputs of rank two and identical shapes is `plugin::div` ordinal 3,
`div_add_half_fuses_to_mean_symbolic`: its unchanged
source includes division and its structural callback requires Mean. It is not
an Add-only row. The rank-three GPT broadcast row is also outside this packet.
Neither changing a source nor specializing B may manufacture corpus acceptance.
All seven original NCHW rows likewise retain their separate model/replay work.

## Validation and provenance

The implementation owns six production paths: compatibility symbol inspection,
source representation/profile, conversion core/facade and sample evidence.
One new module contains **40 cases and 19 test functions**; four existing modules
change only obsolete guard/diagnostic assertions while preserving their test
identities. The frozen packet contains eleven paths.

| Check | Python 3.12 / JAX 0.11.2 | Python 3.11 / JAX 0.10.2 |
| --- | --- | --- |
| Owner final focused selection | 697 passed, 8 warnings, 46.54 s | 697 passed, 8 warnings, 48.84 s |
| Independent selection, including all 40 new cases and four updated guards | 241 passed, 16.44 s | 241 passed, 18.51 s |

These selections overlap and are not additive coverage totals. The eleven
before/after hashes match. Earlier owner 315/693/695 selections and fixture
failures are preparation history, not acceptance of final bytes. Root's final
typing check passes 541 files plus RNG registration, and global Ruff passes.
The complete primary-stack run passes **6,391 tests**, with 13,110 warnings in
444.36 seconds and exit status zero. Commit `80cb68b2` passed all seven actual
configured hooks. All twelve frozen code/inventory hashes match the tested
bytes after commit. The preceding 6,351-test layout/binary run remains historical.

The root-owned inventory refresh records **201 handwritten-test files, 1,093
functions and 341 migration additions**. Only manual accounting changes: both
actual CLI comparisons pass, all 752 original anchors and 1,802 raw contracts
remain unchanged, and there are no removals. Counts do not close any raw row.

Local evidence includes `/tmp/jax2onnx-symbolic-batch-implementation-handoff.md`,
`/tmp/jax2onnx-symbolic-batch-final-before.sha256`, the identical final-after
snapshot, final owner and independent per-stack logs, and root's
`/tmp/jax2onnx-symbolic-batch-root-full-freeze.sha256`. The passing full log is
`/tmp/jax2onnx-symbolic-batch-root-full-py312.log`; actual hook results are in
`/tmp/jax2onnx-symbolic-batch-code-commit.log`.
Independent code review and final-byte acceptance are recorded in
`/tmp/jax2onnx-symbolic-batch-astra-review.md`, with no blocking finding within
this finite scope.

## Min/Max: finite selection questions answered

The separate read-only sparse audit completes the previously missing scalar,
empty and output-only checks. Each stack records **72 exact edge observations**
and **12 scoped output-only grid observations**. The latter preserve frozen
inspected output bytes, zero-tie signs, outside-tie baseline bytes and NaN
classification; they are not complete native-equivalence acceptances.

Six raw rank-zero artifacts per stack retain their GatherND checker failures.
The already proposed rank-one adapter passes the scalar PoC cells with four
extra nodes; production integration remains unimplemented. FP32/FP64 grids
retain **108 non-NaN native mismatches per operation/grid/optimization setting**
from inherited subnormal behavior. FP16 has no non-NaN mismatch on this grid.
Historical eight signed-zero failures per stack, cheap-Where failures and all
prior source/reference criteria remain unchanged. No native NaN-payload or
signaling promise is added.

One coordinated quiet-window workload compares FP16 Min broadcasting
`(256,1)` with `(1,256)`, yielding 65,536 outputs and four zero ties. Both repairs
match native eager/JIT and preserve all baseline bytes outside the ties;
baseline misses one zero sign. Across the two stacks and CPU disabled/all,
GatherElements costs **1.242–1.542× baseline**, and sparse/GatherElements is
**0.963–1.220×**, with no consistent sparse speed advantage. Ten warmups and 50
timed calls exclude session creation and file I/O. Cooperating tests paused;
exclusive host tenancy and universal performance are not established. Older
FP32 cost observations remain separate historical evidence.

Raw node counts are 1/15/12 and saved ORT counts 1/19/16 for baseline/dense/sparse.
The repairs gain provider-inserted floating Cast nodes. Dense selection's two
logical tensors are 262,144-byte choices and 524,288-byte INT64 indices. Sparse's
two coordinate layouts are 64 bytes each here, with 8-byte selected updates,
but still needs full masks/base/correction and output copying; coordinates grow
with rank and the data-dependent tie count. These are logical sizes, not measured
allocator usage, peak memory or aliasing. Sparse can cost more at dense ties.

The accepted finite review recommends GatherElements as a later bounded lowerer
candidate because its selection storage depends on shape, it handles scalars
directly, and sparse shows no consistent latency benefit. Actual exporter
typed-constant/scope integration, target samples and semantic/resource-cost
review remain gates. There is no Min/Max production release, blanket admission,
expensive unconditional rollout, new precision mode or target selector.
Current `SampleCriteria` semantics remain intact: exact values equate signed
zeros; bitwise criteria distinguish their signs and NaN payloads. The existing
artifact-bound sample validator and raising acceptance guard separate generic
export from target certification.

The finite evidence and independent decision are retained in
`/tmp/jax2onnx-minmax-sparse-finite-freeze.json` and
`/tmp/jax2onnx-minmax-sparse-finite-astra-review.md`. Their scope closes these
feasibility questions, not Min/Max numerical migration.

## Next accepted tooling packet

After this checkpoint, the accepted plan selects **13 unchanged raw rows and
26 artifact-only x64-off/on cells per stack**: five dot rows, three matmul rows,
four Add rows and the historical FP64 gather row. The gather-off configuration
conflict remains. No pass total or numerical outcome is predeclared. Exact
origin/ordinal/contract identities and original typed parameters determine
selection; equal hashes or display names alone are insufficient.

Exactly four dot rows receive separately reviewed SOURCE MatMul structural
counterparts. Preserve the original Gemm/zero-bias callback and record its
actual failure independently. The counterpart must check the precise input
edges and RHS transpose where required, FP32 types/shapes, no unused inputs and
absence of bias, hidden casts or extra arithmetic. It does not assert universal
bitwise equivalence to a different Gemm-plus-zero computation. Other callbacks
stay unchanged, and absent callbacks remain explicitly absent.

A separate revision supplement retains the historical `158ce625` baseline,
original 36-cell files and two-dot supplement byte-for-byte. It does not reset
failed history or reserialize old records with new defaults. The new selection
leaves 3,578 of 3,604 cells unselected. Runtime constructor denial stays active;
lowering/runtime/numerical gates do not acquire acceptance from a structural
counterpart. All-22 fixed-source evidence remains three checked artifacts and
19 pending rows per stack, with zero ORT runs.

The numerical deliverable is a **criterion proposal ledger only**, reviewed
before any runtime execution or adoption. Legacy comparison uses
`allclose(reference, actual)`; SOURCE uses `allclose(actual, reference)`, so
copying tolerance numbers does not recover the same asymmetric predicate.
The legacy FP32 defaults are 1e-5/1e-5 and the FP64 variant defaults 1e-7/1e-7;
record actual variant applicability, conversions and forced harness
`default_matmul_precision("float32")` settings. The twelve raw matrix/Add rows
remain strong FP32 even with x64 enabled; gather needs its separate FP64 audit.
No threshold is adopted, no historical criteria are filled in, and no public
`SampleCriteria` behavior changes. Any proposed stricter conjunction needs
explicit prior review, not calibration against exported outcomes.

The accepted plan and exact selectors are
`/tmp/jax2onnx-next-b3-corpus-replay-plan.md` and
`/tmp/jax2onnx-next-b3-replay-candidates.json`. This is tooling work, not a new
operation/profile expansion. Fixed source implementations, every existing
capability and the eventual single normal export contract remain the goal;
temporary private exclusions cannot become a permanent mixed user experience.
