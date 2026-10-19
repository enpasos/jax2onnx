# Private source-policy integration

Local implementation and review, updated 2026-10-19, beyond `main` at `fae17822`.
The converter has a private SOURCE policy. Public options and legacy defaults
retain their meanings; no public source mode or release is established.
Code checkpoint `7d74a200` has accepted primary-full, secondary affected-area
and finite dual-stack runtime results, with all seven actual commit hooks
passing. The record distinguishes those accepted scopes from earlier failures
and the open full migration. Known runtime rounding failures remain separate
failed numerical acceptance cases.

## MLP state and bounded normalization acceptance

The 2026-10-19 checkpoint has independent acceptance for the finite original
NNX/Linen MLP runtime matrix and selected static normalization examples. Public
behavior remains LEGACY. These results do not complete the existing-functionality
inventory or authorize the final public transition.

The original ordinal-2 MLPs retain seed 17, dimensions 30 → 20 → 10,
running-stat BatchNorm, positive Dropout rate 0.1, the original GELU choice,
parameters and factory dtypes. External pure whole-model preparation exposes
actual state/RNG inputs and outputs while preserving both original branches.
Four independent reference roles are captured before SOURCE activation/export:
original eager, original NNX JIT with its static flag, prepared eager and
prepared JIT. Linen retains its original bridge and key contract.

The finite target matrix covers x64-off FP32, x64-on FP32 and x64-on FP64,
original zero and meaningful nonzero inputs, and B=2/5 where declared. Each
serialized artifact is reused with `ORT_DISABLE_ALL` and `ORT_ENABLE_ALL` on
CPU, on both supported stacks. T/F/F/T controls feed actual returned state into
the next call, with resets and replay. Every production output is compared
with each of the four native references using the original preapproved absolute
bounds, `rtol=0` and no NaN equivalence. All values are finite and retain exact
dtype/shape; zero outputs have exact positive-zero bits. UINT32 state words and
the separately observed Dropout masks match the original native values exactly.
The accepted Phase B packet contains 12 production/diagnostic model pairs and
40 selected PRNG artifacts across both stacks: 128 sessions and 2,144 calls.
Diagnostic Y has an ABI-only contract; direct-mask acceptance does not confer
numerical acceptance on that diagnostic output.

The tracked [permanent runtime regression](../../../tests/extra_tests/converter/test_source_mlp_runtime_acceptance.py)
passes its six NNX/Linen × x64/dtype cases on each stack. Each actual run has six
exports, twelve CPU sessions and 192 target calls, using B=2 and the same four
reference roles, state-feedback/reset/replay checks and frozen criteria. This
minimal CI test does not repeat the separate direct-mask matrix. Source,
configuration, PRNG, callable, policy and observer restoration checks pass for
the recorded successful runs; neither finite packet is a universal PRNG or
model-support guarantee.

The [variance/std checks](../../../tests/extra_tests/converter/test_source_square_variance.py)
and [LayerNorm checks](../../../tests/extra_tests/converter/test_source_layer_norm_exports.py)
add finite static FP16/FP32/FP64 evidence for native square, variance/std and
NNX/Linen/Equinox LayerNorm. They check actual serialized dataflow through
mean/variance, epsilon, square root, reciprocal, affine and residual operations,
including selected inline, named Function and conditional boundaries. Fourteen
valid-ONNX wiring mutations are rejected. The named-boundary proof evaluates the
same captured exported `ClosedJaxpr` eagerly and with fresh JIT against its
preactivation native references, including both conditional inputs.

That proof establishes **actual exported-body eager/JIT equality**. Whole-carrier
JIT equality remains unestablished; the named residual conditional's activated
eager call also triggers compilation and retains its missing-MLIR
`NOT_ESTABLISHED` result. The original failed epoch exits 1 with zero target
sessions/runs. Outer serialized wiring and full-model target comparisons for
both selectors have separate acceptance. Normalized zero rows use their existing
exact signed-zero comparator; nonzero bounds remain unchanged. The two actual
85-case runs are v4 executions (116 sessions/170 calls per stack); v5 acceptance
adds only the independently verified annotation-equivalent change. These results
do not establish complete normalization-family or Functions014/015 acceptance.

The earlier local integration brought 28 reviewed source/test paths and the
manual inventory update into MAIN. Its independent source/inventory review
verified the 880-file source freeze and all prior inventory anchors. The first
joined focused runs recorded 1,027 passes and the same one stale expectation
failure on each stack: the preparation test still expected profile admission
failure after the bounded implementation reached runtime checking. That failed
epoch remains in the record. The narrow companion repair has independent source
acceptance. The later full failure and repairs are recorded separately below;
the earlier review does not certify subsequently changed source bytes.

Frozen acceptance receipts are retained under `/tmp` as local review evidence;
the tracked tests have no dependency on those paths:

| Accepted scope | Independent receipt |
| --- | --- |
| Finite MLP/selected-PRNG target matrix | `/tmp/jax2onnx-mlp-phaseB-actual-result-astra-review-v1/verification.json` |
| Permanent MLP runtime regression | `/tmp/jax2onnx-mlp-permanent-ci-astra-runtime-review-v1/verification.json` |
| Static normalization and actual-body proof | `/tmp/jax2onnx-normalization-candidate-v4-astra-review-v1/verification.json` |
| Local MAIN source and inventory only | `/tmp/jax2onnx-mlp-normalization-MAIN-astra-review-v1/verification.json` |

Earlier sections below retain their original checkpoint evidence. Historical
barriers and failed criteria are not retroactively changed by later acceptance.

## Full-suite failure and finite repair status

The first full primary integration run is **FAILED**: 9,191 passed and 33 failed
out of 9,224 collected tests, with no skips or errors. All 27,672 setup/call/
teardown phases ran. Its actual terminal is session `73239`, chunk `845246`,
exit 1. The follow-up classification records eight stale expectations, three
non-Dropout product defects and twenty-two Dropout control failures. The frozen
failure receipt and source postimages remain historical evidence; later finite
passes do not change this run's result.

### Accepted finite non-Dropout repair

The actual v6 packet passes the same 57 cases and all 171 phases on each
supported stack. Each run records eleven successful exports, five expected
structured exclusions, twenty-four CPU sessions, sixty-eight target calls and
twenty-four validation reports. Actual terminals are primary `59789/c41a76`
and secondary `8795/2cb28f`, both exit 0. Independent review accepts this finite
packet. Its offline audit checks all 136 saved calls and 296 output leaves
against the original criteria: all observed leaves are byte-identical to their
references, including the thirty-two leaves whose approved comparisons remain
tolerant. No tolerance was changed.

The repairs cover the exact native uniform `0`/`1` literal proof for weak FP64,
the selected static rank-one PRNG width of 32, named initializer handling,
malformed-reshape diagnostic classification, and captured typed keys in actual
Functions. Actual key values remain logical source values; selected valid-key
mathematics is unchanged. Unsupported `rbg` keys retain a structured nested
profile exclusion. The rank-one width restriction is a private migration gap
outside this finite implementation, not an ONNX limitation.

Captured-key Function/If artifacts retain the actual key constant and required
nested dataflow. Source/settings/PRNG/policy/callable/observer restoration checks
pass for these actual runs. The earlier failed v4 and v5 epochs remain in the
closed packet, including the raw `ValueError` preceding the structured-error
repair. A subsequent path-header comment is AST-equivalent but was not executed
by v6. Finite diagnostic numerical validation does not establish general product
admission, complete PRNG support or whole-checkpoint acceptance.

### Explicit Dropout state preparation: first runtime failure

The new internal [state preparation helper](../../../tests/extra_tests/converter/source_state_preparation.py)
and expanded [Dropout checks](../../../tests/extra_tests/converter/test_source_dropout_parameters.py)
define an explicit adapter for the declared NNX and Linen Dropout owners. The
helper retains the actual original target code, body, defaults and nonowner
captures. Only the declared NNX owner closure cell is rebound to an owner
reconstructed from its real graph, fixed state, key and counter. Linen retains
its original fixed external key, rather than replacing it with a new state input.

The prepared workflow threads actual returned state through Function, If and
Loop calls. Its independent references use sequential calls to the original
native step, and the unchanged whole original eager/JIT model wherever valid.
Raw positive-rate NNX training with captured mutable state in If/Loop is natively
invalid and remains a separate prerequisite failure. For those cases, sequential
original-step and prepared eager/JIT comparisons supply the later finite evidence;
whole-original native execution is not established. Equinox retains diagnostic
sample-property criteria, without an exact JAX stochastic-mask claim.

The combined 53-case contract plans 67 exports, 134 CPU sessions and 2,308
target calls per stack. The first actual primary run is **FAILED**: 53 cases
were collected, seven executed, six passed and one failed; forty-six remained
unexecuted. All twenty-one phases of those seven cases ran. The actual terminal
is session `17295`, chunk `019246`, exit 1; pytest reported 95.95 seconds.

The original positive case
`test_dropout_uses_real_named_operand_through_nested_scope[positive-rate-if-nnx]`
failed while constructing its first `ORT_DISABLE_ALL` session at x64-off.
ORT reported nested If/Function `TypeInferenceError` at
`_inlfunc__dropout_state_named_node_If_2`, output `cond_then_0/cond_capture_2`:
inferred rank 0 versus declared rank 1. That epoch exposed a representation/
capture defect. The subsequent source-reviewed repair then needed runtime
validation; its later accepted result is recorded below. The original failure
does not establish an ONNX limitation.

The failed run retained thirteen returned exports, twenty-five session
construction attempts (twenty-four returned and one raised) and 384 actual target
calls. No secondary run or rerun followed that epoch, and no source edit was
made as part of closing it. The original numerical comparisons before failure
have not received independent acceptance. The planned full-matrix counts were
not reached, and that failed epoch supplies no runtime acceptance for the
combined packet.

The failure record confirms unchanged 881 guarded Python files, 29 Markdown
files, generated artifacts, dependencies and inventory. Settings, PRNG, policy,
watched callables, original/rebound target captures and the original preparation
body were restored. The observer itself was restored, but its separate
`all_actual_patch_targets_restored` flag is **false**. At that epoch the
original observer did not retain individual differences for its 590 targets;
the later diagnostic below provides separate evidence.
The aggregate `all_sessions_CPU_threads1` flag is also **false** and includes
the failed constructor. That epoch did not resolve its cause; this flag alone
does not establish incorrect thread settings. No all-restoration or all-sessions
acceptance is claimed for that failed epoch.

The helper remains an explicit finite preparation mechanism, not a generic
automatic adapter for arbitrary opaque fixed sources. Supporting unchanged
third-party JAX sources through honest explicit preparation remains part of the
full completion contract. The first failed epoch must remain visible beside
any later repair and successful execution.

### Source-reviewed naming and restoration repairs

A separate actual primary restoration diagnostic executes one case and all
three phases successfully, with two exports, four sessions and sixty-four target
calls. Its outer terminal is nevertheless **FAILED**, session `76426`, chunk
`477293`, exit 1: inherited attributes were restored as new local shadows.
The original record identifies two resolved-identity mismatches in DinoRoPE
classes; the read-only addendum identifies five namespace/descriptor leaks in
total, including three whose resolved identity appeared unchanged. This is a
real restoration defect. The diagnostic's passing pytest case does not establish
numerical or integration acceptance, and its failed outer result is retained.

Naming v2 and restoration v1 now have independent **source-component acceptance**;
their production postimages are retained in local MAIN, with the later reviewed
two-line test-only public-call correction described below. Naming assigns collision-free
local namespaces while preserving external Values and names, and detaches the
shared proto-backed tensor storage whose name setter would otherwise mutate
another owner. Nested owned definitions retain distinct names without changing
stable formals or free captures. The rejected naming v1 and its occupied-name/
shared-storage blockers remain part of the review history.

Restoration now snapshots raw direct class/module attributes, removes introduced
inherited shadows and unwinds leaf patch contexts in reverse order. Factories
still receive resolved originals; the protected activation block also covers
entry failures. Existing instance/slot behavior and Function reference counting
remain intact. The source review used static and standard-library stand-in
checks; it did not execute the proposed real IR or numerical regressions.

The joined 101-case packet combines the original 53 cases with twelve naming
and thirty-six restoration cases. Its first two attempts each record 35 passes,
one failure and 65 unexecuted cases (108 phases). The v3 attempt passes a private
`precision_policy` keyword to public `to_onnx`, which rejects it before any
export/session/run. An independently reviewed two-line test-only deletion
restores the ordinary public call; it changes no production API.

The v4 attempt reaches one delegated serialization and saves its protobuf,
but returns no UI export and opens no session or target call. The external
observer incorrectly requires the entire public IR to remain unchanged during
serialization and records a mismatched inherited-child baseline first observed
while its ancestor patch was active. Its saved BEFORE includes an unnamed
initializer tensor; the installed serializer assigns initializer tensor names.
No actual AFTER snapshot was saved in that failed epoch, so the complete delta
is unestablished. It cannot be retrospectively certified as a name-only change.
These test/observer failures establish neither a new production defect nor an
ONNX limitation. Both original failed closures remain unchanged.

The amended external observer now records actual BEFORE/AFTER transitions,
separate delegate/wrapper outcomes and proven inherited-target baseline lineage.
Its exact v5 prerequisites have independent acceptance. Two installed-IR-only
toy serializations on each stack provide separate observer-mechanics evidence;
they neither reconstruct the missing AFTER nor establish model results.
The subsequent actual dual-stack result and its separate independent acceptance
are recorded next.

### Accepted dual-stack finite runtime result

Independent actual-result review accepts the exact joined101 packet on both
supported stacks. Each completes **101 cases / 303 phases**, with 73 returned
exports (72 protobuf and one IR), 74 serializations, 141 CPU sessions, 2,364
target runs, 18 samples, six validations, eight LEGACY comparisons and 507
original target-factory constructions. The primary terminal is `30813/10bda9`,
secondary `61217/6ec45d`, both exit 0. These actual counts supersede planned
counts for this successful packet; they do not relabel any earlier failure.

The independent audit checks every saved target run against its original
criterion, including 4,096 exact NNX/Linen target-to-original output comparisons
per stack. It preserves original sequential-step, prepared eager/fresh-JIT and
valid whole-body references, T/F/F/T controls, reset/replay and returned UINT32
key/counter chains through Function, If and Loop. Linen's fixed original key
remains `[0,31]`. Equinox training retains its original zero/scaling property
criterion; exact native stochastic-mask equality is not claimed. NNX whole-body
JIT uses the owner-aware route, and raw captured mutable training If/Loop remains
an invalid native prerequisite. The explicit state helper is still finite,
without a generic adapter for arbitrary fixed sources.

All 74 actual serialization transitions and 141 model-blob lineages per stack
pass the frozen contract. The installed serializer synchronizes initializer
`Tensor.name` with `Value.name`; 1,128 permitted name assignments per stack are
witnessed. Returned IR is accepted through its actual same-model transition
chain, without claiming raw IR metadata immutability. Snapshots cover selected
identity, wiring and metadata fields, not arbitrary payload/metadata equality.
The missing actual AFTER snapshot from the earlier v4 failure stays missing.

Restoration passes for 638 canonical patch targets across 44,023 saved records
per stack, using exact local ownership, descriptors, resolved binding and
recorded ancestor provenance. A stricter supplementary snapshot comparison
initially failed on one marshaled-code digest. That diagnostic FAIL remains
recorded: the same function/code identities and every frozen restoration field
match, and stdlib controls on both exact Python runtimes reproduce unstable
marshal digests for unchanged code objects. Actual target marshal byte streams
were not saved, so their precise byte-level cause is unestablished. No frozen
restoration criterion was relaxed. Full current lookup-lineage arrays were not
saved; the audit reconstructs expected lineage from first-seen/ancestor records,
checks current targeted ancestors, and retains the original runtime assertions
for untouched ancestor namespaces.

The eight LEGACY reference arrays were computed after the target runs; their
original tolerant comparisons pass. The named subtraction test retains its
passing separate JIT assertion, while only its eager reference was saved for
independent array review. These limits stay separate from the exact state and
output evidence above. The dual-run closure retains its original pending-review
label; the separate `verification.json` records the later independent **ACCEPT**.
This finite result does not complete all model, PRNG, normalization or deployment
coverage.

### Integration and inventory gates

After the five naming/restoration postimages, actual standalone configured
typing passes for 543 production files plus the RNG trace check. Global Ruff
and pinned Black checking of all 41 changed Python paths also pass. These v4
results apply to the current repaired source; earlier v3 check receipts retain
their original scope. Standalone checks remain separate from actual final hooks.
The fresh primary full run now passes **9,296 tests and all 27,888 phases**
across four workers, with zero failures, errors, skips or omitted cases. It
retains every identity from the earlier 9,036-test baseline, 9,224-test failed
full run and original 1,028 focused cases. Its actual terminal is `58291/4947b5`,
exit 0. The secondary affected-area run passes **1,052 tests and all 3,156
phases**, including every original focused identity, at `27310/5f8aed`, exit 0.
Independent review accepts those actual integration gates and their exact
source/inventory binding. It does not claim a full secondary-stack suite.

The primary audit binds source and test identities; the secondary additionally
checks its recorded settings, PRNG, policy and watched-callable restoration.
All 881 Python, 29 protected Markdown, 26 generated and three dependency files
match the accepted freeze. Earlier failed epochs remain failed history.
The separate actual configured-hook run passes all seven hooks with no source,
documentation, inventory, dependency or staged-path changes (`31329/e7942a`,
exit 0). The subsequent normal local code commit
`7d74a2001ef0a8e20a08151e7d67e20b1ae64150`, parent
`d960d2c50c39da275af364ddadadc7f98d18b1b1`, contains exactly the 42 reviewed
paths and passes all seven actual commit-time hooks without skips or bypass.
Its terminal is `72344/418bb2`, exit 0. Verification confirms matching tested
bytes, unchanged 881 source files, 29 precision Markdown files, 26 generated
files, three dependency files and inventory, with a clean worktree. The code
commit establishes a local checkpoint; no push, release or public transition is
claimed. The strict public MkDocs build also passes on the accepted code with
its 45 public documentation/configuration inputs unchanged. These eight internal
precision documents are outside that build; their content/link review, application
and separate documentation commit remain subsequent gates.

The reviewed inventory is now applied: **228 manual files, 1,460 functions and
708 migration additions**, exactly candidate `bbecd123b81855d694e42e59be2f18126c6fe88e0f1a0bb044ddabfc668fdc53`.
Actual integration review verifies every preceding 1,440 and intermediate 1,448
function identity plus all 752 original anchors, with zero removals. Only manual
rows/provenance and their function count change; all 1,802 raw rows/3,604 cells
and nonmanual fields remain exact. Applied inventory and integration acceptance
and the separately accepted finite runtime result leave the full migration
gates open.

| Evidence scope | Local receipt |
| --- | --- |
| Failed full primary epoch | `/tmp/jax2onnx-mlp-normalization-MAIN-integration-v2/full-primary-failed-verified-v1.json` |
| Accepted actual finite non-Dropout v6 | `/tmp/jax2onnx-full-v2-non-dropout-actual-astra-review-v1/verification.json` |
| Closed dual packet, including v4/v5 failures | `/tmp/jax2onnx-full-v2-bounded-repairs-v1/non-dropout-dual-v6-closed.json` |
| Original combined Dropout contract | `/tmp/jax2onnx-dropout-state-preparation-runtime-v1/closure.json` |
| Failed first combined Dropout primary epoch | `/tmp/jax2onnx-dropout-state-preparation-runtime-v1/combined53-primary-v1-terminal-failed.json` |
| Failed restoration diagnostic and five-leak addendum | `/tmp/jax2onnx-dropout-observer-candidate-v3/restoration-probe1-primary-v1-terminal-failed.json`, `restoration-probe1-five-namespace-leaks-addendum.json` |
| Naming/restoration source components only | `/tmp/jax2onnx-scope-restoration-source-astra-review-v1/verification.json` |
| Current standalone typing, Ruff and Black | `/tmp/jax2onnx-full-v2-bounded-repairs-v1/ROOT-configured-typing-v4-result.json`, `ROOT-style-v4-results.json` |
| Failed joined101 test and observer epochs | `/tmp/jax2onnx-dropout-state-preparation-runtime-v3/joined101-primary-v1-terminal-failed.json`, `/tmp/jax2onnx-dropout-state-preparation-runtime-v4/joined101-primary-v2-terminal-failed.json` |
| Reviewed public-call test correction | `/tmp/jax2onnx-public-legacy-test-correction-astra-v1/verification.json` |
| Joined101 v5 prerequisites only | `/tmp/jax2onnx-joined101-v5-astra-review-v1/verification.json` |
| Actual primary-full/secondary integration and applied inventory | `/tmp/jax2onnx-mlp-normalization-MAIN-actual-integration-astra-v3/verification.json` |
| Seven actual configured hooks | `/tmp/jax2onnx-mlp-normalization-MAIN-integration-v3/ROOT-actual-configured-hooks-v1-terminal.json` |
| Actual joined101 dual execution | `/tmp/jax2onnx-dropout-state-preparation-runtime-v5/joined101-dual-v3-closed.json` |
| Independent finite dual result ACCEPT | `/tmp/jax2onnx-joined101-actual-result-astra-v1/verification.json` |
| Verified 42-path code commit and seven commit hooks | `/tmp/jax2onnx-mlp-normalization-MAIN-integration-v3/ROOT-code-commit-v1-verified.json` |

These local receipt paths support review; tracked code has no dependency on
them. The full B–F contract remains open: all existing functionality, remaining
models and optional environments, wider scope/export combinations, and evidence
for claimed browser or other execution targets must be addressed before the
public transition. The final ordinary `to_onnx(...)` call has one source-preserving
contract, without a public profile choice or primitive-specific legacy fallback.

## What is implemented

[`PrecisionPolicy`](../../../jax2onnx/converter/precision_policy.py) separates
LEGACY and SOURCE. The [shared private harness](../../../tests/extra_tests/converter/precision_test_support.py)
uses the actual tracing helper, IRContext, dispatch, plugins, optimizer,
post-processing and serialization. [source_export.py](source_export.py) is a
compatibility entry point for the internal demos. The bounded harness lives in
test support; it is not a second public export API.

SOURCE preserves traced output dtypes and constant payloads in the audited
paths and suppresses late legacy promotion. Child contexts inherit the policy
and recursive validator. The caller's ambient x64 setting is retained; typed
FP64 inputs under disabled x64 are rejected before tracing.

Both input normalization steps preserve `weak_type`. For scalar `2048` plus
explicitly FP16 `1`, weak inputs return FP16 `2048`; strong inputs with the same
shape and dtype return FP32 or FP64 `2049`. Tests use the original JAX values,
arrays and typed specifications, both x64 states, serialized CPU output, and
actual legacy calls. Shape-only source inputs use the proposed strong FP32
default. Legacy normalization keeps its observed compatibility behavior.

Selected Function and If paths preserve mixed FP64/FP32 outputs, direct and
shared rounded values, static Boolean/string arguments, captures, repeated
results, and tuple/list/dict returns. Output aliases and formal-input
passthrough receive distinct Identity outputs where required by function
signatures. Weak/strong inputs specialize separately. Both If branches are
selected at runtime in the serialized-artifact tests.

## Review findings

The initial 168-case internal suite passed on both stacks but missed the
following cases. The repairs now have durable tests in the regular suite.

### Local function settings must survive body retracing

With ambient x64 disabled, a model called this decorated function inside
`with jax.enable_x64(True)`:

```python
@onnx_function
def cancellation(x):
    wide = x.astype(jnp.float64)
    return ((wide + np.float64(1)) - wide).astype(jnp.float32)
```

For FP32 input `1e8`, JAX returned FP32 `1`; the exported function returned FP32
`0` on both installed runtimes, with optimization off and on. Reversing the
local and ambient x64 settings reversed the discrepancy. A wide-output variant
instead failed the checker. Matching output types could not catch the silent
internal change.

The source call now records x64 and default matrix precision in typed metadata,
replays those settings for body tracing, and includes them in specialization.
Existing typed formals are reconstructed without truncating an already-created
FP64 value; the callee body still runs under its captured settings. Result avals
are checked again after retracing. Tests cover same-signature calls with local
settings, nested functions, failure restoration, and matrix settings whose
explicit algorithm must be rejected rather than lost.

The [context regressions](../../../tests/extra_tests/converter/test_source_function_context.py)
cover x64 and default matrix precision. They do not establish preservation of
every possible JAX setting or concurrent export.

### Cached branch traces must not cross export policies

A conditional branch containing an `onnx_function` could keep the SOURCE
primitive in JAX's trace cache. A later legacy export of the same callable
failed with an unexpected internal output-metadata keyword. Direct
source/legacy/source tests did not exercise that cached branch path.

A private [compatibility helper](../../../jax2onnx/_compat/jax.py) now makes
ordinary execution, LEGACY export and SOURCE export distinct JAX trace/JIT
cache contexts. It restores the prior context after success and failure;
no global cache clearing is used. Both export orders, omitted/false/true
legacy calls, cached conditionals, ordinary eager/JIT execution, and failed
export followed by ordinary execution are checked. The JAX State constructor
used at this compatibility boundary is available in the declared minimum
[JAX 0.8.1 source](https://github.com/jax-ml/jax/blob/jax-v0.8.1/jax/_src/config.py).

### Algebraic fusion must respect the source operation sequence

Admitting Div exposed a legacy rewrite: `(x+x)/float32(2.00001)` became Mean and
returned `1` instead of approximately `0.999995` for `x=1`. SOURCE now retains
Add and Div, as well as separate Mul, ReduceSum and Sqrt operations where
traced. Unvalidated Mean, LpNormalization, ReduceSumSquare and ReduceL2
substitutions remain disabled under SOURCE. LEGACY retains its existing rules.
[Arithmetic regressions](../../../tests/extra_tests/converter/test_source_arithmetic_fusion.py)
check the divisor, overflow, shared rounded values, zero-row behavior and
source/legacy isolation. Removing a converter fusion does not repair a runtime
rounding failure by itself.

### Durable tests now run in the normal CI collection

Input and scope regressions moved from this documentation folder to `tests/`.
New context, matrix, normalization, arithmetic and sample-acceptance regressions
also live there. The existing CI's default pytest collection includes them;
no workflow change was needed. Internal source/UX modules are compatibility
entry points to shared test helpers, and the demo commands still work.
Supplemental PoC diagnostics remain explicitly invoked outside `testpaths`.

## Follow-up review on 2026-10-18

The Astra Ultra review on 2026-10-18 found two additional SOURCE defects after
the preceding checkpoint. Both were implemented with 6.1 Sol Ultra and reviewed
again independently with Astra Ultra, which found no remaining blocker in this
bounded milestone.

With `JAX2ONNX_ENABLE_REWRITE_MUL_RSQRT=1`, FP32
`x * (float32(1.000001) / sqrt(y))` at `x=y=1` became `1` instead of
`1.0000009536743164`. The finalizer now passes its explicit policy to the shared
optimizer. Each registered pass must declare eligible policies; one selection
covers the existing top-level and Function paths. SOURCE excludes reciprocal
reassociation, Swish fusion, ConvTranspose bias folding and transpose/reduction
movement pending numerical audit. Structural cleanup remains enabled. LEGACY
retains its defaults and environment-controlled behavior. See the
[32 optimizer regressions](../../../tests/extra_tests/converter/test_source_optimizer_precision.py).

The function-cache defect involved two registered pytree instances storing
FP64 bias `1e8` with identical shape and bytes but different `weak_type`.
For `(x + bias).astype(float64) * float64(2**-20)` with FP32 input `1`, JAX
returned `(95.367431640625, 95.36743259429932)`; one reused ONNX Function
returned the first value twice. SOURCE keys now retain promotion metadata
recursively through supported static containers, captured arguments and pytree
instance state. The extension applies at primitive argument packing, static
capture keys and unique instance keys. Immutable JAX snapshots retain their
metadata, legacy keys are unchanged, and equivalent states still share a
definition. The [11 captured-state regressions](../../../tests/extra_tests/converter/test_source_captured_precision.py)
cover both orders, nested static arguments, original JAX references, saved
artifacts and both runtime profiles. Removing the new key reproduces four
failures, including the nested-static-argument cases.

## Numerical implementation and acceptance

| Case | Current local result | Limit of the evidence |
| --- | --- | --- |
| FP16 matrix operands, preferred FP32 result | SOURCE emits local FP32 operand casts and FP32 MatMul, returning `2049` for the witness while preserving original FP16 bindings. | Positive static rank-2/rank-3 standard matrix axes and default precision only. Other precision algorithms/types/shapes are explicitly rejected. No performance claim. |
| FP32 squares, selected FP64 variance sum | The actual JAX callable exports inline, in a Function and in both If branches. Exact variance is `8388609`; constant-row output is zero. | Selected fixed CPU samples; independent control remains separate. No general normalization or legacy accuracy-policy adoption. |
| Original FP16 square then sum | The source graph retains Mul then ReduceSum. ORT 1.29 and 1.31 still return zero squares but sum `2**-23`. | Failed numerical acceptance under both tested optimization settings. No automatic repair is established. |
| Explicit wider arithmetic with FP16 intermediate cast | The changed-source sample returns the expected zero squares and sum. | Sample agreement only; not an automatic transformation or equivalence proof. |
| FP32 `x * (1 / sqrt(y))`, `x=3`, `y=2` | Saved source graph and reference retain `2.1213202476501465`. ORT 1.29/1.31 match with optimization off; optimization on returns `2.1213204860687256`. | Failed acceptance under the selected exact-sample criteria with optimization on. Converter preservation does not prevent runtime reassociation. |

An additional ad-hoc probe uses FP16 input `1`, weak FP32 bias `4096`, then
converts the sum to FP32 and scales by `2**-20`. JAX eager/JIT and
ReferenceEvaluator return `0.00390625`; ORT 1.29/1.31 return
`0.003907203674316406` with optimization off and on, losing the intermediate
FP16 Add rounding. This is distinct from square/sum and from successful cache
specialization. The positive x64-off cache regression instead chooses an
epsilon from JAX's promoted dtype and performs FP32 arithmetic. The ad-hoc
failure remains numerical evidence, not a durable acceptance regression.

The exact-one optimizer tests allow a future runtime correction to pass.
Returning the reciprocal as another output preserves the chosen sample under
both current ORT profiles; this is an observation, not a proposed output-interface
change. No automatic runtime repair is established.

The optimizer audit also found an existing LEGACY limitation: moving a
Transpose around ReduceMean with explicit axes inside a Function registers an
initializer that is not serialized as a Function node. Checking then fails
with a missing axes input. SOURCE excludes the pass and its explicit-axis
Function regression passes. The legacy defect remains a separate inventory
item; an invalid artifact is not an accepted compatibility result.

The [CPU sample gate](../../../tests/extra_tests/converter/precision_runtime_support.py)
records the actual ORT version, CPU provider, architecture, optimization and
thread settings, and checks exact output types, shapes and values.
`require_sample_acceptance` rejects FAILED and UNVERIFIED results. Kernel
load/run errors are separate. A future runtime correction may satisfy the
original FP16 sample; a diagnostic that recognizes the current discrepancy
cannot pass acceptance. A report covers only its supplied artifact and samples,
not all model inputs or a deployment generally.

The [numerical findings](numerical_contract_findings.md) describe the unchanged
public matrix gap, new source implementation, controls, runtime failure and
rejection evidence in detail.

## Audited exploration boundary

The harness admits static floating inputs at opsets 23 and 26. Primitive paths
include add/sub/mul/neg, numeric conversions, ReduceSum, equal-width BitCast,
floating gt and the Boolean-to-INT32 conditional selector, two-branch cond,
real onnx_function calls, Div, BroadcastInDim, lax/NumPy Sqrt and the bounded
matrix lowering above. BitCast requires explicit opset 26; different widths
are rejected. Function and branch bodies are checked recursively.

An allowlist is not a support matrix for all compositions. BF16 structural
checks do not establish CPU kernels. JIT regions, loops/scan, symbolic shapes,
effects and unaudited operations remain excluded. Public names/layouts,
parameter materialization and return-mode integration remain outside this
harness. There is no public deployment-target API, generic capability engine,
automatic fallback selection, latency benchmark or memory benchmark.

## Recorded validation

Installed CPU stacks used for these checks:

| Python | JAX / JAXLIB | NumPy | ONNX | ONNX IR | ONNX Runtime |
| --- | --- | --- | --- | --- | --- |
| 3.12.9 | 0.11.2 | 2.5.4 | 1.23.2 | 1.0.0 | 1.31.0 |
| 3.11.12 | 0.10.2 | 2.4.6 | 1.22.0 | 1.0.0 | 1.29.0 |

Durable SOURCE suites pass on both stacks: input properties (46), function/If
scopes (16), call-context/cache isolation (35), preferred-FP32 matrix products
(35), wider normalization (6), arithmetic boundaries (12), and runtime sample
acceptance (9). These 159 cases belong to normal pytest collection. The
supplemental PoC suite contains 107 cases, including expected exclusions and
recognized diagnostics. Passing it is not a numerical certificate.

The follow-up adds 32 optimizer and 11 captured-state regressions to normal
collection, bringing durable SOURCE coverage to 202 cases. Each new suite
passes on both installed stacks.

Final local validation on the updated checkout, 2026-10-18:

- Full repository suite on Python 3.12/JAX 0.11.2: **4662 passed**, using CPU
  and four workers.
- Python 3.11/JAX 0.10.2 focused converter/architecture suite plus supplemental
  PoCs: **616 passed**, using CPU and four workers. This is not a second full
  repository run.
- Supplemental PoCs on Python 3.12: **107 passed**; those same cases also pass
  within the Python 3.11 run above. The numerical demo runs and reports the
  original FP16 sample as rejected by numerical acceptance.
- Repository typing: **538 source files** pass. The four private test-support
  modules also pass a separate mypy check.
- Ruff, IR-builder policy, Black for all **35 changed/new Python files**, and
  the strict MkDocs build pass. Internal Markdown rendering and **143 local
  links/anchors** pass separately because these documents are outside the
  published documentation build. `git diff --check` passes.

The formatting result covers touched files; the preceding implementation
checkpoint recorded pre-existing formatting differences elsewhere in the
repository. These local results establish neither a remote CI result nor a
published capability.

## Remaining migration work

The [next work](../precision_migration_plan.md#next-implementation-work) defines
and enforces a finite initial subset, integrates the existing export boundary
privately, and connects validation evidence to caller-facing diagnostics.
The failed runtime samples remain excluded from acceptance under their recorded
criteria and profiles. Broader operator/scope/target coverage and any public
selection still need their own acceptance. These changes advance Stages 3 and 4 without
completing the migration.


## Exact floating selection

The later two-file SOURCE update repairs the already-admitted Boolean-selector,
two-arm FP16/FP32/FP64 case. It stacks the arms in `[false, true]` order, casts
only the selector to INT32 values 0/1, and copies the chosen value with
`GatherElements(axis=0)` and `Squeeze`. A scalar selector expands to the actual
arm shape. There is no floating cast or arithmetic in this selection. Canonical
arms, siblings and captured values remain intact. Native Where forwarding,
profile admission, public LEGACY behavior, report schema and existing
BOOL/integer/BF16/complex/multiway boundaries retain their meanings. BF16
structural evidence still does not establish CPU kernel support.

Independent review accepts the exact two production/test postimages and the
bounded owner observations. Both supported stacks pass 179 owner cases,
including 38 new cases. Before the repair, the final signed-zero regression
fails on original production on both stacks; afterwards it passes under the
same strict criteria and native reference bytes. The retained witness reports
record twelve converted strict successes per candidate stack, with six direct
old-Where failures and six separate positive-zero diagnostic-reference successes.
Those diagnostic successes do not claim native equivalence. Other focused cases
cover tested NaN payloads, infinities/subnormals, native promotion/broadcasts,
scalar and empty results, B=2/5 with repeated/complementary masks through one
artifact, and Function/If captures and shared outputs. Passing empty-output
cases does not separately prove execution of an empty GatherElements kernel.

The saved production witness evidence contains native arrays, ONNX models and
validator output hashes; it does not retain raw actual target-output arrays for
all focused cases or establish a total runtime-call count. Export reports retain
runtime/numerics UNVERIFIED until separate sample validation. The old Where
artifacts and failures, original callback contracts and strict criteria are
unchanged. The operator-documentation list adds GatherElements; testcase
registration metadata stays identical, while whole-file source provenance
naturally changes.

This representation has a substantial memory cost. The independently reviewed
standalone comparison recorded 576 target calls and 168 sessions: 504 small saved
outputs and 72 realistic A/B/A observations bound to four distinct actual arrays.
At FP32 `[1,12,512,512]`, named logical buffers total 48 MiB for INT32 indices,
versus 60 MiB for INT64 indices and 45 MiB for the tested Compress alternative.
INT32 gave the better measured speed/memory tradeoff in those finite runs. The
2048 attention example still implies 768 MiB of named buffers, including a
384 MiB arm stack, before broadcasts. Process high-water memory includes runtime
arenas, inputs/references and other work; three-call observations are not an
operator-allocation or general-throughput guarantee. No speculative scalar fast
path or memory-cost waiver is part of this change.

The first integrated full run retained all 8,810 expected identities and ended
with 8,721 passes and 89 setup errors, with no test failures or skips. Every
setup error came from the old strict Gather fixture rejecting the new floating
selection representation. That failed run remains immutable; it did not execute
a failing Gather numerical comparison.

The reviewed three-path tooling repair gives the current Gather representation
its own exact 26-node/14-constant template. The historical 20-node/13-constant
template stays literal and is rejected by the current revision. Edge order,
attributes, dtypes and constant bytes remain strict; the original 90 Gather
cases are retained and nine new guards are added. Because the current Dot0
control binds the complete oracle source file, its one current literal hash is
updated to that exact revision. The control identifier and meaning are unchanged;
historical descriptors and failure records are not repinned. There is no
permissive graph fallback or relaxed numerical criterion.

Both MAIN artifact runs preserve the four ordinary controls and repeat all 26
selected cells: 25 diagnostic artifacts plus the unchanged Gather x64-off FP64
configuration conflict. All twelve replay invariants pass, with unchanged
3,578 unselected cells and 126 pending cells. The fresh references and artifacts
match the immediate reviewed owner epoch on both stacks. Gather's original
source, native trace, inputs and references remain bound to the same saved
artifact: the raw callback still fails while SOURCE structure is VALIDATED.
Product lowering, runtime and numerical acceptance remain UNVERIFIED, criteria
and optimization remain None, and these artifact runs attempt no ORT execution.
This does not revise the separately recorded historical matrix-reference issue.

Independent review accepts the exact six-path MAIN integration: two selection
paths, three strict-tooling paths and the manual inventory. Both focused runs
pass 1,911 cases/5,733 phases across eighteen collected modules and fifteen
recorded MAIN imports. The five effective/hidden JAX settings, policy and 25
watched callables are restored; all 871 Python hashes, nineteen checkpoint paths
and 29 Markdown files stay exact. Style, configured typing and both inventory
CLIs pass. The inventory records 220 manual files, 1,372 functions and 620
migration additions, retaining all 1,363 immediate predecessors and 752 original
anchors, zero removals, and 1,802 raw rows/3,604 cells. The fresh complete primary
suite passes exactly all 8,819 expected identities, the earlier 8,810 plus nine
Gather guards, with no failures, errors, skips or omissions. It records 13,773
warnings and 682.91 pytest seconds (682.845 XML seconds); all 871 Python hashes,
nineteen checkpoint paths and 29 Markdown files remain unchanged. Code commit
`778dce3c5f7e799f3288199cd5cdc13b51bb03ed` then passes all seven installed hooks
without bypass, preserving those hashes and committing exactly six changed
paths. The recorded worktree is clean after that commit. The preceding
8,772-test/code-8f checkpoint and the failed first selection/Gather full run
remain historical evidence.

The older-stack FP16 decoder benchmark remains an independently established
baseline failure. This repair does not close it, general selection/provider
support, unchanged MLP training/state behavior or the full precision migration.
At that checkpoint, the separate MLP work had native preparation, a measured
PRNG ownership review and floating criteria frozen before target runs. Its
implementation was outside MAIN and model runtime/numerical acceptance was
pending. The later finite result is recorded in the current acceptance section
above.

| Frozen evidence | SHA256 |
| --- | --- |
| `/tmp/jax2onnx-where-cheaper-probe-astra-review-v1.md` | `2bb8435040ae9fe47d20404402a8f70ba9765e56d8d2b660e368d4684cd94a52` |
| `/tmp/jax2onnx-exact-select-astra-review-v1.md` | `fbcc14994faf36fdc9f8f364c0d0ac092dc513186491df89244b32a06d9e67c6` |
| `/tmp/jax2onnx-exact-select-main-integration-freeze-v1.json` | `5e9b3c140a32ad23c60dbba85cdd6b5f0ec6d03c59b70c0712421a49f36f8a66` |
| `/tmp/jax2onnx-exact-select-main-focused-dual-v1.json` | `555567a2a0f938ecfe3505c80c3e5d02abb7aac2b4390244bc17b01081abf885` |
| `/tmp/jax2onnx-exact-select-main-checks-v1.json` | `fec9aef745cb037f18beab9b6a9f2dad96e6d91d4f1b17a65bb6a291a440bac9` |
| `/tmp/jax2onnx-exact-select-main-full-result-v1.json` | `7dd9ab74cf73fbab7c5538e1e0433307bd1f613b0023ebf674ef638975834847` |
| `/tmp/jax2onnx-gather-selection-code-astra-review-v1.md` | `1163ea5defc57fb39fdd3abce33c6421576becadcaaec45a7350a903096e5b12` |
| `/tmp/jax2onnx-gather-selection-postrun-astra-review-v1.md` | `7bf282c5e6f1cbde0451c3fe063459bc7d614313d50950fd55fbf3ab476c90bf` |
| `/tmp/jax2onnx-gather-selection-live-control-release-v1.json` | `2937b4fafe647fe132f2e5e602e314cd31c3934e43ef7efc79145cdb10f3dbfe` |
| `/tmp/jax2onnx-exact-select-main-integration-freeze-v2.json` | `8411e5cc4c4cd8195f840ae91f9c79fe250a14703c252a257b2a31ff30ac8f1a` |
| `/tmp/jax2onnx-exact-select-main-focused-dual-v2.json` | `c60707b5d4efb8fcffa32382f0eefbe2cbd755cdb826894bfb1a6a1a595517f2` |
| `/tmp/jax2onnx-exact-select-main-checks-v2.json` | `b7f074374f8694963761093474be4048cc6b517014599c8d7fe565fb6dfbe13d` |
| `/tmp/jax2onnx-exact-select-main-artifacts-dual-v2.json` | `aab46edf220965f16bc74109d5bac5b7bee793e456be42eaef09796c4708fe85` |
| `/tmp/jax2onnx-exact-select-main-integration-astra-review-v2.md` | `542ff2a5b9e42a370d6cdcd70ee2f5a51099dd82dfd3ceaa68363045d9decdd5` |
| `/tmp/jax2onnx-exact-select-main-integration-astra-verification-v2.json` | `23ade31437667b95bfca9a02af7f1d1e0f1263aa9073cdc12cae41cb763bf21e` |
| `/tmp/jax2onnx-exact-select-main-full-result-v2.json` | `a05370d277f5e208193085c8a7b8c7b653d0bdc606abf361c0a970cdfa3b3faa` |
| `/tmp/jax2onnx-exact-select-main-code-checkpoint-v2.json` | `4942d59f8a6da92a97db2af0b67ed734ac766170d5a58d244bf961aa09aec441` |
| `/tmp/jax2onnx-exact-select-main-code-commit-v2.log` | `2ab88d93bfbae13995435ea6118fb1ec702a19f4c654aa6bd703611980a26db6` |


## LayerNorm and mean native forwarding

This section records the preceding forwarding-only checkpoint. The later
[MLP and bounded normalization result](#mlp-state-and-bounded-normalization-acceptance)
has separate acceptance and explicit limits.

Independent source/evidence review accepts the finite five-path forwarding
checkpoint. The three framework LayerNorm wrappers and `jax.numpy.mean` now call
the captured original exactly once under private SOURCE policy, before legacy
argument/default processing, input coercion, configuration reads or axis/mask
handling. Original argument identity, omission, exceptions and signatures remain
observable. Complete LEGACY bodies, all four registration ASTs, six raw mean
cases/callbacks, lowerers, abstract evaluation, batching and JVP code are unchanged.
There is no new profile admission, public API change or public precision mode.

The retained NNX witness has FP32 inputs and FP16 affine parameters. Native JAX
first promotes the parameters to FP32. Before the repair, the activated wrapper
instead converted the FP32 input to FP16 and emitted synthetic `nnx.layer_norm`.
The final full-plugin-world trace matches the original native graph on both
supported stacks. This is a concrete removal of exporter-imposed narrowing.

Full-world Equinox testing exposed one additional prerequisite: its native
`reduce_sum → broadcast_in_dim → div` mean was replaced by synthetic
`jax.numpy.mean`. Forwarding mean through its own captured original repairs that
interception. The initial suggestion of an independent var/JIT divergence was
incorrect: complete nested `_var` JIT parameters already matched; only outer SSA
numbering changed after mean collapsed three equations to one. The retained
correction addendum supersedes that diagnosis. No var/JIT or profile change is
part of this checkpoint.

The owner runs pass **127 tests on each supported stack**, retaining all eighteen
previous full-world Equinox failures: sixteen dtype/affine combinations, the
weak-scalar/context case and symbolic batch. Sixteen isolated wrapper-only cases
remain supplemental. The strict-Flax correction compares the actual native
NNX/Linen `TypePromotionError` and message rather than requiring a native success
that does not occur. Failed epochs and the five direct-mean pre-repair failures
per stack remain recorded.

Fresh native-versus-activated comparisons cover typed traces and nested bodies,
actual constants, eager outputs and fresh-JIT outputs, including exact payload
bytes. Inputs, parameters and computation dtypes, fast/slow variance, epsilon,
axes, masks/where, affine options, weak values, contexts, state/key arguments and
native errors are exercised. State snapshots, five global/effective settings,
source bytes, LayerNorm/mean callable identities and policy restore. ORT
constructors and execution entrypoints are denied throughout these owner checks.
The source-derived phase totals are 132 trace, 129 eager and 129 fresh-JIT pairs,
including 32 supplemental pairs per phase; these are test contracts, not separate
observed phase receipts. Separate concrete full-world captures retain three equal
native/activated traces per stack. Existing LEGACY/IR checks pass 30 cases per
stack plus all six original mean graph-checker/callback anchors.

The first MAIN focused selection on JAX 0.11.2 retains **696 passes and one
failure among 697 cases**. The failure is the fixed-source preparation test's
stale expectation of eight current barriers. Native mean forwarding now exposes
already admitted reduction/broadcast/division/subtraction operations in all four
minimum-precision examples. Their FP16/BF16/FP32 inputs produce FP32 outputs;
FP64 produces FP64, all at shape `[1,4]`. Actual artifacts and representation/
lowering reports are saved, while runtime lists stay empty and target numerical
acceptance stays None. Their bitwise native reference criterion is unchanged.
Only the two protected-sum and two HIGHEST-dot barriers remain in that fresh run.
The original twelve-export/four-session/four-run schedule still executes only the
two exact target programs. The independently accepted companion-test repair
updates only the stale live barrier assertions and adds checks for all four
admitted artifacts: native traces and references, report contents, tensor-port
dtypes/shapes, ONNX schema, reload and serialized bytes. Both owner stacks pass
all eleven existing module cases with unchanged identities. No additional target
program runs were added. The original failure, the PoC's literal historical eight
barriers, source programs, inputs, bitwise native criteria and older findings are
retained. This additional mean export progress does not establish LayerNorm
export or new runtime/numerical support.

At this forwarding-only checkpoint, default-audited SOURCE LayerNorm export
stopped at these native operations on both stacks:

| Framework | First barrier at that checkpoint |
| --- | --- |
| NNX | `square`, equation 2 |
| Linen | `square`, equation 0 |
| Equinox | `square`, equation 3 / JIT body / equation 4 |

The profile was unchanged at that checkpoint. These barriers identified
implementation dependencies, not ONNX impossibilities. Original Functions014/015
and whole-model export/runtime/numerical acceptance were **UNVERIFIED**. The
separate MLP implementation was then outside MAIN; its later finite acceptance
and the subsequent bounded normalization result are recorded above. Complete
Functions014/015 acceptance remains open.
The final goal remains every existing capability through one ordinary call that
preserves the source computation, including explicit external JAX preparation
for fixed-source workflows; no permanent public mixture of precision policies
is introduced.

The five reviewed forwarding/test paths, the existing companion test and the
manual inventory update form the seven-file MAIN integration committed at
`ae1d61d1`. Independent review confirms exact accepted postimages across the
872 Python files and preservation of all 29 precision Markdown files during code
validation and commit. The inventory contains **221 manual files, 1,385 functions
and 633 migration additions**. All 1,372 immediate predecessor functions and all
752 original anchors remain, with no removals. Only the new test module's thirteen
functions were added. The 1,802 raw rows, 3,604 cells and all nonmanual fields are
unchanged.

| Frozen evidence | SHA256 |
| --- | --- |
| `/tmp/jax2onnx-layer-norm-mean-native-forwarding-candidate-v1/SHA256SUMS.json` | `7f4c06ae23d3763df62b8fe70d712fc53685d35970e9c92a898c9bfda79dd6ca` |
| `/tmp/jax2onnx-layer-norm-mean-independent-astra-review-v1.md` | `d39f6dd27ab3ee7315ff0aaffd5ebfea6b2e6befd27020e7a637f985b38f293b` |
| `/tmp/jax2onnx-fixed-preparation-companion-candidate-v1/SHA256SUMS.json` | `ab96037ec0c863f35ed1f2eee42c2952f4defcc09c5a3bb2053dcfa5117dfcd0` |
| `/tmp/jax2onnx-fixed-preparation-companion-astra-review-v1.md` | `68d4f8b9df06d60e78d5e4077c1c0df0a18e613fee86cb0281675cd2a9e61913` |
| `/tmp/jax2onnx-layer-norm-main-fixed-preparation-failure-v1.json` | `730b34db1a657d4a4026557b2aa3897dd4696a712ca48022f421c89edc88fcb5` |
| `/tmp/jax2onnx-layer-norm-main-fixed-preparation-failure-v1-closed/manifest.json` | `4ca7cd73584d583ad3146d270bc75a50e4db0a6f00bc7ef9f2045059f53b177e` |
| `/tmp/jax2onnx-layer-norm-main-integration-freeze-v2.json` | `3a68ffcac9d352bcf216a120de3d7687ca7977c2bda67a0a64e0ecd90543474b` |
| `/tmp/jax2onnx-layer-norm-main-focused-dual-v2.json` | `7bcc8b6556e4b91e1b451e8c9e58b74ff0349f5832fe0d8e290c3be0d04707fa` |
| `/tmp/jax2onnx-layer-norm-main-checks-v2.json` | `8ac37ff456beeb15d9899dd68c9ccb2bc401ecfddd55c3ad3cf729cd335f9ab6` |
| `/tmp/jax2onnx-layer-norm-main-integration-astra-review-v2.md` | `6a8715c1e1aecf510be20e3c22bbe119521547beb8d888781b848745158ce2c2` |
| `/tmp/jax2onnx-layer-norm-main-full-result-v2.json` | `1dbbd2fb4ae6de6e80117cb6573de8bd69dc7801f4ae045bbf0f98a814a46678` |
| `/tmp/jax2onnx-layer-norm-main-full-astra-review-v2.md` | `0036b72cccba01480f30d5c92bbb6c5564e14b030af9d5ede920f10690d2e6e8` |
| `/tmp/jax2onnx-layer-norm-main-code-checkpoint-v2.json` | `68b6f539fa687bb535d172290cf61a4a5ecd2e84ce02e64de30c26ff6aed8078` |
| `/tmp/jax2onnx-layer-norm-main-code-commit-v2.log` | `b6d49d1faf06bef9b2d9b42de02a24fef9dfe74701d7af597bc578268c9edaed` |

Validation status: actual MAIN receipts supersede the owner's earlier declared
Ruff/Black/whitespace successes. After the companion repair, both supported JAX
stacks pass the same **697 focused tests**, with **2,091 actually observed
setup/call/teardown reports per stack**. These pytest phases are separate from the
source-derived native comparison totals above. All 31 watched callable identities,
five hidden/global and five effective settings, and the precision policy restore;
21 imports bind each run to MAIN. The refreshed seven quality commands pass:
global Ruff, configured Black version/check for all six Python files, whitespace,
configured typing for 542 files plus registered RNG factories, and both inventory
CLIs. The primary complete suite passes **8,946 unique tests**, retaining all 8,819
previous identities and all 127 new LayerNorm identities, with no missing,
unexpected, failed, errored or skipped cases. Its raw log records 13,609 warnings.
Both the finite integration and actual full-suite result have independent
acceptance. The original 696-pass/one-failure epoch remains retained. Code commit
`ae1d61d14b0122c9f1814c813eba3f495c4365e2` follows predecessor `4f088d82`,
contains exactly the seven accepted paths and completed all seven actual hooks
without skips or bypass. Its terminal receipt records a clean MAIN worktree and
unchanged 872 Python, seven checkpoint and 29 Markdown hashes.


## Native Sum forwarding and measured composition

Independent source/evidence review accepts the native Sum implementation, now
committed with its companion test and manual inventory at `2d1ea944`.
Under private SOURCE policy, the outer wrapper calls its own captured original
exactly once with the original positional and keyword arguments. This occurs
before legacy dtype/axis/default handling, array coercion, `out`/`initial`/`where`
checks or `promote_integers` coercion. Omitted arguments remain omitted; original
return/error identity and signature behavior are preserved. The complete LEGACY
body, all six raw registration cases and callbacks, lowerer/fusion/empty-axis
logic, JVP, batching and custom transpose remain unchanged. This checkpoint does
not change converter core, the SOURCE profile, public defaults or precision flags.

The standalone owner suite passes **90 cases on each supported JAX stack**,
retaining all 70 preparation identities and all eight pre-fix failures. These
are suite cases, including API errors, a captured-callable boundary spy and a
signature check; they are not 90 numerical trace/eager/JIT triples. The coverage
map is source-derived. Seven separately recorded native/SOURCE witnesses per
stack preserve saved typed traces, trace text, closed constants and exact eager
and fresh-JIT payloads. Named-JIT checks reuse the decorated inner function with
fresh outer wrappers, so they do not independently prove fresh inner retracing.
The 22 LEGACY structural/schema/reload artifacts per stack cover six raw anchors,
four fusion/opset cases and twelve empty-axis cases. Their original runtime test
phases remain held. All ORT constructors/runs were denied in these owner checks.

The retained external receipt failures in epochs 1 and 2 came from comparing
current tuple-bearing typed avals with lists restored from JSON. Epoch 3 isolates
ten container differences with identical serialized values. The authorized
epoch-4 correction JSON-normalizes only the current typed manifest for that saved
record comparison; current native
versus SOURCE equality and numerical bytes remain exact. No source, native
criterion or original failure record was changed. Independent review retains the
owner style distinction: Black/Ruff/mypy success stdout exists, while configured-tool
identity and whitespace success lack separate structured process receipts. The
later actual MAIN quality receipts below close that process-evidence gap for the
integrated checkpoint.

The first separate composition used committed LayerNorm/mean code `ae1d61d1`, documents
`21ad27b4` and only the two Sum postimages; it contains no pending MLP source.
Its first **primary-only** run collected 101 tests: **99 passed, one failed and
one was unexecuted**. All 90 Sum tests passed, and 80 actual calls to the unchanged
comparison helper completed. There are 300 observed pytest setup/call/teardown
reports; the native phase semantics remain in the hash-bound helper. The failure
is the existing companion's blanket `profile-validation` assertion at line 192.
The v1 driver correctly stopped because its narrowly defined obsolete-assertion
classifier covered only lines 186/194. Its false acceptance flag is retained;
no source, state or target-schedule guard failed. Secondary was not executed and
no retry or assertion edit was used to relabel that result.

The same four rejected keys now report these actual barriers:

| Cases | Phase/category | Primitive and location |
| --- | --- | --- |
| `exact_original_protected`, `exact_prepared_protected` | `representation` / `representation-limit` | `bitcast_convert_type`, equation 9 in each |
| `linear_original`, `linear_prepared` | `profile-validation` / `profile-exclusion` | HIGHEST `dot_general`, equations 0 and 1 respectively |

Both protected examples request float32-to-uint32 bit reinterpretation. The
unchanged PoC requests **opset 23**; the existing audited representation requires
exactly opset 26 for this BitCast path. An isolated ONNX 1.23.2 schema query confirms
that `BitCast` first appears at 26 and is absent at 23/25. The diagnostic recommends
`opset=26` only if the target supports it and states that export never upgrades
the opset automatically. This identifies the current representation requirement,
not runtime support at opset 26 or an impossibility for every other implementation.
The protected examples remain rejected; this checkpoint does not change their
opset or source to obtain acceptance.

The original twelve native cases and **twelve-export/four-session/four-run**
schedule completed. Eight artifacts were admitted: the two exact programs, two
intermediate-output programs and four minimum-precision examples. All use opset
23. Only the same two exact programs executed, each with disabled/all optimization;
their four target outputs satisfy the unchanged eager/JIT dtype, shape and byte
criterion. The other ten rows retain empty runtime lists and no target numerical
acceptance. All 82 NPY files match the previous accepted companion epoch byte for
byte, including inputs, actual weights, native references and target outputs;
the contract differs only in its epoch identifier. The original native criteria,
including the linear pointwise bounds, and literal historical eight-barrier
`HISTORY` remain unchanged. Sum introduces no new target numerical acceptance.

All 873 Python files, 29 precision and 76 total Markdown files remained unchanged
during that composition run. Five global/local/effective setting values, policy,
61 callable identities including 47 public PRNG callables, and observer wrappers
restore. This is callable preservation, not a separate PRNG-state audit.

The independently accepted existing companion amendment has postimage
`5ccb2e6d75bb0758c8a673bb87b0519f23f78d7b7a2267ecfe8451d9c8f735bb`.
It keeps all four rejected keys and asserts each exact phase/category/primitive,
the protected uint32/opset-26 requirement and conditional/no-auto-upgrade action,
and the linear HIGHEST/default-precision requirement. Original target-output
checks, all four mean artifact/native/report/port/schema/reload/trace checks, the
intermediate/manifest suffix, other tests and original execution limits are exact.
The old driver and failure remain immutable. A new source freeze, strict v2 driver
and separate execution release bind the corrected fixture. Both supported stacks
then pass **all 101 unique cases and 303 observed pytest phases**, with no skip or
obsolete-failure allowance. Each observes 80 completed original comparison-helper
calls, the original 12 native PoC cases and exactly 12 exports/four sessions/four
runs. Both sets of 82 NPY payloads remain byte-identical to the failed v1 epoch;
criteria, protected barriers, eight admitted artifacts and the two original
numerically validated target programs retain the limits above. Five setting
values, policy, 61 callable identities and observer wrappers restore on each stack;
the 47 public PRNG checks remain identity checks only. The 873 Python, 29 precision
and 76 total Markdown hashes remain unchanged during these composition runs.

The four-path Sum code checkpoint `2d1ea944` has parent `21ad27b4`. The MLP
source was outside MAIN and was not a prerequisite for that integration. Default
SOURCE LayerNorm exports then encountered `square`; the later finite MLP and
normalization acceptance is recorded above. Phases B–F and their full
existing-functionality gates remain open. The final goal is one ordinary
source-preserving `to_onnx(...)` call across all existing functionality, including
fixed JAX model sources configured or prepared externally. No public precision
mode or permanent partial-profile choice is introduced.

| Frozen evidence | SHA256 |
| --- | --- |
| `/tmp/jax2onnx-native-sum-forwarding-candidate-v1/SHA256SUMS.json` | `a61a97b9bd7ef3f9fe2ef598e9fa2962da95c831176bf717b52efc39beb93d9e` |
| `/tmp/jax2onnx-native-sum-independent-astra-review-v1.md` | `8c394af1488b458ac25bdf316b686ab008abb4002e8214fe966e31ad0c99eb81` |
| `/tmp/jax2onnx-native-sum-composition-primary-failure-candidate-v1/SHA256SUMS.json` | `d2a13a82178581c974faf7b3c68469df26ea876726f445ded23f90c3975b1d97` |
| `/tmp/jax2onnx-native-sum-composition-primary-failure-candidate-v1/summary.json` | `7daa4fccacff901fb5a8acacbcef59761a0811f1fbfb34712648e5c30ce454f8` |
| `/tmp/jax2onnx-native-sum-composition-astra-review-v1.md` | `2811284b28bea3a6d23d07ad64d9e3d01fd192cc5b6b5b8c2969b1b03f2d7937` |
| `/tmp/jax2onnx-native-sum-composition-onnx-schema-astra-v1.json` | `774ae7d4aed1631e1249c7015f773e37ad6923488782735b6c0271e260257b1b` |
| `/tmp/jax2onnx-native-sum-companion-candidate-v1/SHA256SUMS.json` | `86af7425afa2f0f0ac598c3910a6510bd6cf9931cbf7d0f318df5efcd1739a58` |
| `/tmp/jax2onnx-native-sum-companion-astra-review-v1.md` | `aa84e9a069f3750fffd0610b3a644a07bc6ddcf018901b5e65c54a260f74e7bc` |
| `/tmp/jax2onnx-native-sum-composition-measurement-v2/SHA256SUMS.json` | `b8b8a4f4a8f7efe160aa3c7d175600ce702cf14ab1710133ee3ea0fdd6c0ed72` |
| `/tmp/jax2onnx-native-sum-main-integration-freeze-v1.json` | `4af1cc1045a60136aa04a2e47e025e03df047099fe664b78befeb66214360cad` |
| `/tmp/jax2onnx-native-sum-main-focused-dual-v1.json` | `14f6c67c86a4ecfd0f2134b00ddf4105986cc6728647cca8a6a1dce92669ddea` |
| `/tmp/jax2onnx-native-sum-main-checks-v1.json` | `ad077a5dd71811f398c5d4c08eb0174b3b01355ca289526128763bfeae7748fc` |
| `/tmp/jax2onnx-native-sum-main-full-result-v1.json` | `a9b58ee47dd25bb0915c7656a61f0e157f4a1cb1c47f70f4de992a01b1c0c2a1` |
| `/tmp/jax2onnx-native-sum-main-integration-astra-review-v1.md` | `9eada04c14b7005fd2e5a8cef618d1308ab7faa4798942e04ca02a03e4942a72` |
| `/tmp/jax2onnx-native-sum-main-code-checkpoint-v1.json` | `5d5933ed0b85d814a9940dbfd2590a09f2e7c67d9320067387eb3a5fe3f4e6f8` |
| `/tmp/jax2onnx-native-sum-main-code-commit-v1.log` | `28bd9e2de62d60bb406e310a35e9f190506157c02ca989747332213c6e6118c1` |
| `/tmp/jax2onnx-native-sum-main-apply-enumeration-failure-v1.json` | `7faf06d3a833a11fc9cd545b0a04257af07b6ed2e327c63c852cf3c653f37b9e` |

Validation status — completed code checkpoint: independent review accepts the
four-path MAIN integration (`sum.py`, its new test, the existing PoC companion
and handwritten inventory). The inventory has **222 manual files, 1,401 functions
and 649 migration additions**, retaining all 1,385 predecessor functions and all
752 original anchors. Exactly 16 new test functions are added; all 1,802 raw
rows/3,604 cells and nonmanual fields remain unchanged. Both MAIN focused runs
pass **787 cases and 2,361 observed setup/call/teardown phases**, with all 33
watched callables, five hidden/effective settings and policy restored; 23 recorded
imports resolve to MAIN. Seven actual quality commands exit zero: global Ruff,
Black version/check, whitespace, configured typing plus RNG registration, and both
inventory CLIs. Typing checks 542 source files. Configured Black revision 25.1.0
reports version `0.1.dev1+g8a737e727`; the secondary inventory preserves the existing
absent-on-this-stack `one_minus_square` baseline entry. The fresh complete primary
suite passes **all 9,036 unique expected identities**: the previous 8,946 plus
exactly 90 new Sum cases, with no failures, errors or skips (13,609 warnings;
545.74 pytest seconds). ROOT's earlier apply-enumeration assertion, caused by
ignored debugvenv files, remains recorded; corrected Git enumeration verified
the identical already-applied sources without changing a criterion. Code commit
`2d1ea944adb17f477b76f4951d74aa59cc028cdb` contains exactly the four accepted paths
and passes all seven actual repository hooks without bypass. Its terminal receipt
records a clean MAIN worktree and exact 873 Python, four checkpoint and 29 Markdown
hashes. At that Sum-only snapshot, its companion documentation was a separate
external draft awaiting review, application, build/checks and its own commit.
Those completed code gates do not establish new target/model
numerical support or complete phases B–F.
