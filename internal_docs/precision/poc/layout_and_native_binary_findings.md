# Static Layout and Native Binary Binding Findings

Date: 2026-10-18. Committed checkpoint on `precision-migration`:
optimizer `f02c0b4c`, native binary binding and diagnostic Mul-None `258e90d6`,
and static NCHW boundaries `7609fc2e`.
The native add/minimum/maximum binding and separate diagnostic Mul-None dependency
have passed independent review. Static NCHW boundaries and their shared
transpose/naming dependency have focused acceptance after the live-observer
correction. The earlier full run failed ten regressions; the fresh full run
passes 6,351 tests, and all three commits passed actual configured hooks.
Public behavior remains LEGACY; no public precision mode,
whole-family admission or corpus-row closure follows from these changes.

## Native binary binding

The existing private `jnp_binding_specs` helper now permits an explicit native
SOURCE opt-in. Only `jnp.add`, `jnp.minimum` and `jnp.maximum` use it. They forward
unchanged arguments to the captured original before custom normalization,
without retrying native errors through legacy precision. The other 46 helper
callers, LEGACY bodies, registrations and raw metadata remain unchanged.

Native JAX now determines weak typing, integer/floating promotion and configured
dtype/rank-promotion errors for these three bindings. Tests compare complete
typed traces against originals captured outside the patched environment. They
also cover local settings, nested Functions/If, cache order without clearing,
shared canonical I32/FP64 constants, IR/protobuf/file returns, gradients and vmap.
Ordinary floating Add uses existing audited rules. Min/max and selected
integer-input promotion cases remain diagnostic-only, with lowering UNVERIFIED.
Binding correctness does not establish target numerical acceptance.

A separate two-file diagnostic correction admits only the observed
`mul(out_dtype=None)` spelling for equal supported floating inputs. Native
abstract evaluation still checks exact output dtype, shape, weakness and effects.
Non-None output types, unknown fields, mixed inputs and explicit nonfloating
forms remain excluded; previously accepted absent-key forms are unchanged.
There is no production lowerer or audited-profile change. Both min/max gradients
now produce real checked artifacts and four validated CPU sample observations
each per stack: original eager/JIT references with optimization disabled/all.

Maintained checks are [binary dispatch](../../../tests/extra_tests/converter/test_source_jnp_binary_dispatch.py)
and [diagnostic boundaries](../../../tests/extra_tests/converter/test_source_precision_diagnostics.py).
The older-stack initial shared-constant test failed because a bare callback
Jaxpr was compared with a ClosedJaxpr carrying constant payloads. The test now
compares bare typed bodies and separately verifies initializer/runtime bytes.
The original 61-pass/one-failure log remains retained; no source value or
production behavior was changed to repair that harness comparison.

## Signed-zero failures remain failed

The unchanged inputs `[-0,+0,-0,+0]` and `[+0,-0,-0,+0]` produce:

| Operation | Original JAX eager/JIT | Recorded CPU ORT output |
| --- | --- | --- |
| minimum | `[-0,-0,-0,+0]` | `[+0,-0,-0,+0]` |
| maximum | `[+0,+0,-0,+0]` | `[+0,-0,-0,+0]` |

There are **eight FAILED observations per stack**: two operations, FP32/FP64 and
optimization disabled/all. Schema and execution checks pass; bitwise numerical
acceptance fails and its acceptance guard raises. The expected-failure controls
pin the actual reference/output hashes, dtype/shape, x86_64 CPU provider and
JAX/ORT pairs 0.11.2/1.31.0 and 0.10.2/1.29.0. Other mismatches fail the tests; a
future genuine exact match takes the normal success path. Passing pytest does
not turn those observations into numerical acceptance. Add is a passing control.
NaN tests retain their separate exact-value/matching-NaNs criterion, without a
new NaN-payload promise.

## Static NCHW boundaries

The frozen boundary candidate composes explicit external NCHW storage with the
original logical NHWC computation. The shared representation descriptor maps
real `(N,H,W,C)` to `(N,C,H,W)` and complex storage `(N,H,W,C,2)` to
`(N,C,H,W,2)`. The complex pair axis stays last; ordinary real trailing-two data
is not inferred to be complex. Canonical inputs and child-scope values retain
their source dtype, shape and weakness.

Selection applies to positional input indices and flattened outputs; real named
Boolean inputs do not shift those indices. Names, aliases, reports and final
IR/protobuf/reloaded-file checks use the same physical descriptor. The layout
test fixture restores all five JAX global settings and enclosing effective
scopes on success/failure.
The [layout tests](../../../tests/extra_tests/converter/test_source_layout_boundaries.py)
include mixed selected/unselected boundaries, canonical alias outputs and the
unchanged source calculation. BF16 evidence is structural/report-only, with
runtime UNVERIFIED. Codec checks for complex storage do not admit every complex
operation or model.

This boundary packet does not close the seven original whole-model NCHW rows.
Symbolic dimensions, normalization selection, raw layout replay and unaudited
clip/full forms remain explicit migration work. The existing 310 symbolic rows
and 41 nondefault-normalization rows remain in the denominator.

## Shared transpose and naming dependency

The unchanged mixed-layout source returning `(x,-x,y)` exposed an existing
shared optimizer defect: a transpose fold removed an input still returned as a
canonical alias. The LEGACY control failed too. The bounded repair checks graph
outputs, uses outside the matched region and nested references before changing
layouts or deleting values. Safe isolated folds and genuinely unused cleanup
remain enabled; this is not a SOURCE-only pass skip.

The later orphan-transpose cleanup required the same observer check: a valid
nested-only capture could survive the fold and then lose its producer. Final
review also found inverse output Transpose deletion leaving distinct lexical
captures unbound. All matching deletion paths now guard those observed outputs
before mutation. This is limited to the confirmed transpose/Add family; pass
order, traversal and unrelated reshape/reduction semantics stay unchanged.

The guarded candidate then exposed an earlier NameFix dependency. The installed
native pass renamed distinct producerless child references independently of
visible ancestor definitions. The wrapper now snapshots only unambiguous
nested node-input aliases, runs the native pass once, and restores their names
from the definitions' final names. Function roots, sibling scopes, local/formal
shadowing and initializers remain distinct. Ambiguity fails before mutation;
there is no graph rewiring or global lexical substitution. Direct outer-value
child graph outputs were rejected by checker/runtime before optimization and
are not treated as supported capture evidence.

The [observer regressions](../../../tests/extra_tests/framework/test_transpose_observer_optimization.py)
retain the failure lineage. Optimizer SHA256 prefixes `383c512a` and `bd81573c`
are superseded candidates: the former missed inverse-output captures; the
latter guarded them but still exposed NameFix. Candidate `bee2340e` passed 783
owner tests per stack, then a precise local variable rename resolved typing.
Candidate `8fd3cff6` passed all 51 owner observer cases and 572 independent
checks per stack, including the bounded semantic review. Its six-case original
witness replay passed pre/post checker and exact CPU disabled/all comparisons
on both stacks. Those prefixes are file hashes, not commits.

The subsequent full run on `8fd3cff6` failed: **10 failed, 6,335 passed, 13,110
warnings in 632.85 seconds**. The unchanged ten baseline failures reproduced on
both stacks. Every failure involved a detached removed Transpose whose lingering
`.uses()` link was incorrectly treated as an observer, retaining an actual orphan
Transpose. The log remains `/tmp/jax2onnx-precision-layout-binary-full-py312.log`;
the earlier focused passes do not establish integrated acceptance.

Final optimizer bytes `0f8b60c5` change one expression to inspect current local
`consumers_in_graph` instead of unrestricted `.uses()`. Graph-output protection,
reachable nested-reference scans, inverse-output guards and NameFix behavior
stay unchanged. One new function supplies six cases for input/intermediate
values with removed local users, removed-parent child users and attached child
captures, with full checker and exact CPU disabled/all checks. The 57 observer
cases include the previous 51 unchanged. Owner and independent final-byte runs
each pass the same 238 cases per stack: ten original failures, 57 observer,
36 layout and 135 related controls. These are separate executions, not additive
coverage counts. The fresh full run passes 6,351 tests; superseded freezes and
the failed full run remain part of the evidence.

## Validation and provenance

| Packet | Bounded evidence per supported stack |
| --- | --- |
| Native binary binder, 62 new cases | Owner: 339 focused plus 27 unchanged generated controls. Independent: 366 combined, all passed. |
| Separate Mul-None diagnostic dependency, 18 new cases | Owner and independent selections: 84 each, all passed. These structural tests alone do not establish runtime support. |
| Static boundaries, 36 new cases, plus observer dependency, 57 new cases | Owner and independent: the same 238 cases each on final `0f8b60c5` bytes, all passed; includes the ten unchanged integration regressions. |

Counts overlap across packets and do not measure operation-family support.
The 57 observer cases include ten collection-only scope/ambiguity controls and
one injected final-name mechanism check; these do not claim runtime evidence.
The executable observer fixtures retain their own checker and exact CPU checks.
The fresh complete primary-stack run passes **6,351 tests**, with 13,110 warnings
in 466.07 seconds and exit status zero. The 6,183-test Dropout/D2a run remains
historical evidence. Root's typing check passes 541 files plus RNG registration,
and global Ruff passes. All three code commits passed actual configured hooks,
without bypasses or formatter upgrades. The 17 frozen code/inventory hashes
match before and after every commit.
The root-owned inventory records 200 handwritten-test files, 1,074 functions
and 322 migration additions, following intermediate refreshes of 198/1,039/287
and 199/1,058/306. Both CLI comparisons pass; all 752 original anchors and 1,802
raw contracts remain unchanged, with no removals. No row is automatically accepted.

The passing full log is `/tmp/jax2onnx-precision-live-observer-full-py312.log`;
its freeze is `/tmp/jax2onnx-precision-live-observer-full-freeze.sha256`.
Actual hook logs are `/tmp/jax2onnx-precision-checkpoint-optimizer-commit.log`,
`/tmp/jax2onnx-precision-checkpoint-binder-commit.log` and
`/tmp/jax2onnx-precision-checkpoint-layout-commit.log`.

Local review/freeze records are `/tmp/jax2onnx-jnp-group-freeze.json`,
`/tmp/jax2onnx-jnp-group-astra-review.md`, `/tmp/jax2onnx-mul-none-freeze.json`,
`/tmp/jax2onnx-mul-none-astra-review.md`,
`/tmp/jax2onnx-c4b-boundary-final-freeze.json`,
`/tmp/jax2onnx-c4b-transpose-observer-final-freeze.json`,
`/tmp/jax2onnx-c4b-astra-review.md`,
`/tmp/jax2onnx-c4b-observer-membership-freeze.json` and
`/tmp/jax2onnx-c4b-live-observer-astra-assessment.md`. They retain exact paths,
hashes, commands, versions and superseded candidates. Standalone probes use explicit checkout
PYTHONPATH and verify imported paths/hashes; the earlier stale-import issue is
not repeated as evidence for these bytes.

The historical [36-cell corpus batch and two-cell supplement](source_corpus_findings.md)
remain unchanged: the supplement passes schema but fails the original Gemm
callback, with no accepted B3 artifact or numerical result. The
[all-22 diagnostic result](fixed_source_compatibility_findings.md) remains three
checked artifacts and 19 pending rows per stack, zero ORT executions and
unverified lowering/runtime/numerics. Matrix subnormal, exact linspace and FP16
rounding failures retain their original criteria and records.

## Next bounded work

The sole next production direction is two equal-shaped inputs `(B,K)` sharing
one named positive batch dimension, one audited Add and a named composition,
with one artifact validated at two concrete batches. It requires actual sample
symbol binding, logical/physical report normalization and correct child-symbol
mapping. It is planned, not implemented, and does not cover symbolic H/W, affine
expressions, general control flow or all 310 raw symbolic rows. Existing positive
dynamic witnesses and fixed sources must remain usable; no new profile choice
or runtime Assert machinery is proposed.

Min/Max numerical alternatives remain feasibility only, with no production
release or admission change. A small standard zero-tie Add/Neg/Where construction
has a semantic argument but fails the measured strict CPU targets. It also
increases recorded non-NaN mismatch counts and changes 21 FP16 non-tie payloads
per operation through signaling-NaN quieting. This establishes neither baseline-byte
preservation nor a new NaN-payload promise; inherited subnormal differences stay
separate. GatherElements preserves the measured signs at a material 12–15-node
and buffer cost; the sparse alternative lacks complete boundary and cost
evidence. Neither is selected for an unconditional rollout. The existing
[target contract](../precision_target_state.md#onnx-feasibility-and-deployment-contract)
permits faithful generic export without target certification. Current export
has no separate runtime-target argument: `validate_source_samples` supplies CPU
evidence and `require_sample_acceptance` rejects failed artifact-bound samples.
A known strict sample remains FAILED; ordinary tolerances cannot erase it or
waive required sign/rounding behavior. A semantic-only change would need its own
reviewed acceptance plan and new artifact evidence. No criterion, public API,
target framework or operation admission changes follow from this feasibility work.
