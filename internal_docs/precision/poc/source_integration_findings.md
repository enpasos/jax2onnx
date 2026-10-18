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
