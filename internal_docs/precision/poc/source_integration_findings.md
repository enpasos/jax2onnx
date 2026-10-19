# Private source-policy integration

Local implementation and review, updated 2026-10-18, beyond `main` at `fae17822`.
The converter has a private SOURCE policy. Public options and legacy defaults
retain their meanings; no public source mode or release is established.
The review defects described below are repaired locally. Known runtime rounding
failures remain separate failed numerical acceptance cases.

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
The separate MLP work has native preparation, a measured PRNG ownership review
and floating criteria frozen before target runs. Its implementation is not yet
integrated into MAIN, and model runtime/numerical acceptance remains pending.

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

Actual default-audited SOURCE LayerNorm export still stops at these native operations on
both stacks:

| Framework | Current first barrier |
| --- | --- |
| NNX | `square`, equation 2 |
| Linen | `square`, equation 0 |
| Equinox | `square`, equation 3 / JIT body / equation 4 |

The profile remains unchanged. These are remaining implementation dependencies,
not evidence that ONNX cannot represent the models. Original Functions014/015,
whole-model export, runtime and numerical acceptance remain **UNVERIFIED**. The
separate MLP implementation remains outside MAIN and requires its own acceptance
and integration before its prerequisites can be used for those model checks.
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
