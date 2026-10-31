# Computation precision: current state

Updated 2026-10-31. Public `to_onnx` still uses the existing **LEGACY** precision
policy. **SOURCE** is the private migration path that aims to preserve the typed
JAX computation. It has useful, independently reviewed capabilities, but the
complete migration and public transition are unfinished. The unmerged PR may
adopt the new contract incrementally; keeping today's public default is not a
separate requirement. Temporary legacy paths need a concrete CI/reference
purpose and removal criteria. Complete acceptance remains required before merge.

This page explains what exists and what the evidence establishes. The
[target](precision_target_state.md) defines the intended contract, the
[work plan](precision_implementation_work_plan.md) retains the complete remaining
scope, and the [handoff](implementation_handoff.md) preserves detailed results
and failed predecessors. For practical export guidance, read the
[current user view](precision_current_state_user_view.md).

## Published checkpoint and validation limits

The current documented published checkpoint is `62276cd8`. At the
2026-10-31 22:35 UTC observation of
[CI587](https://github.com/enpasos/jax2onnx/actions/runs/41243854804),
the four focused EXTRA cohorts had succeeded. Those are focused regression
results, explicitly `F_ONLY`; they do not establish complete added-suite or
scientific acceptance. Python 3.14 STANDARD qualified individually within the
whole-job budget, and Python 3.12's strict parts and join succeeded. Python
3.11 and 3.13 exceeded their work and whole-job deadlines. The original full
Python jobs were still running. Overall budget qualification and replacement
activation remain open.

The pending scheduling repair reuses the existing balanced parts and strict
join for Python 3.11 and 3.13, preserving their complete STANDARD selections and
ordinary limits. Its local controls and hosted execution are separate gates.
The [CI gate](precision_ci_gate.md) states the exact acceptance rules; the
[execution plan](ci_execution_plan.md) records operational evidence.

Public behavior has not changed. Passing regression tests, a successful export,
and a numerical equivalence result have different meanings. Several tests pass
because they correctly detect a known failure or reject unfinished cases.

## The public precision argument

`enable_double_precision=False` is the current default. The facade temporarily
sets JAX's global x64 value from this argument and restores the previous state,
including failure paths. A model's enclosing local JAX configuration can take
precedence; the ordinary table below assumes no such local override.

| Public call | Global x64 during export | Shape-only input default |
| --- | --- | --- |
| Argument omitted or explicitly false | Disabled | FP32 |
| Explicitly true | Enabled | FP64 |

The flag also affects allocation, constants, consumer adaptation and final
postprocessing. It is therefore more than an x64 permission. Enabling x64 does
not turn an explicitly FP32 JAX operation into FP64, but later LEGACY rules can
widen selected lowered values. FP16/BF16, explicit conversions and selected
constructors have their own paths, so the flag is not a uniform mixed-precision
contract. Relevant owners are [conversion_api.py](../../jax2onnx/converter/conversion_api.py),
[user_interface.py](../../jax2onnx/user_interface.py),
[ir_utils.py](../../jax2onnx/ir_utils.py) and
[ir_postprocess.py](../../jax2onnx/converter/ir_postprocess.py).

Typed specifications initially retain their dtype, but disabled x64 can narrow
an explicit FP64 request during JAX tracing. Historical observations differed
in whether JAX warned. Public LEGACY normalization also loses JAX `weak_type`,
which can change promotion despite equal shape and dtype. SOURCE preparation
preserves that property and rejects explicit requests that JAX would narrow
before model execution. This private correction is not a public flag change.

## Why valid types do not establish source arithmetic

The merged consumer repair uses explicit casts for a consumer requiring DOUBLE
while preserving the original FP32 value for other uses. General lookup does
not mutate the canonical binding. With FP64 input, the LEGACY expression
`-x.astype(jnp.float32)` therefore rounds to FP32, widens that rounded value,
and negates in FP64. A direct output of the rounded value remains FP32.
The [exact-dtype regressions](../../tests/extra_tests/converter/test_exact_dtype_casts.py)
cover selected arithmetic, constructors, shared outputs, functions and layouts.

That is a valid implementation of the existing policy. It does not make later
arithmetic match the original FP32 calculation. The witness
`y = x.astype(jnp.float32); return (y + 1) - y`, with FP64 input `1e8`, produces
zero in the original FP32 JAX calculation and one under the tested LEGACY double
export. A redundant FP32-to-FP32 source cast can disappear during tracing and
then be affected by later promotion; an FP64-to-FP32 cast remains observable.

Constant binding and recursive postprocessing can also change payload dtypes.
Plugin-specific algorithms add further distinctions: the explicit LEGACY
LayerNorm lowering widens selected statistics under its documented local
accuracy policy. That policy is separate from SOURCE preservation of the
framework's original computation. Generated `_f64` tests rewrite selected input
and factory dtypes; they are useful coverage, not independent references for
intentional mixtures of dtypes.

## Implemented private capabilities

The entries below describe bounded capabilities. Their detailed input, dtype,
opset and target limits remain in the linked tests and findings; no entry
claims an entire operator family or arbitrary composition.

| Area | Implemented behavior and evidence boundary |
| --- | --- |
| Shared export pipeline | SOURCE reuses real tracing, primitive dispatch, IR construction, optimization, finalization and serialization. Required finalization errors stop export. Private facade checks cover names, structured/mixed results, reports, proto/IR/file returns and selected web packaging. |
| Input and context preservation | Typed and weak inputs, exact constants and selected caller/model-local x64, matrix and NumPy dtype/rank-promotion settings survive audited paths. Cache specialization distinguishes precision-relevant captures; ordinary JAX, LEGACY and SOURCE traces stay isolated. This is not a guarantee for every JAX setting or concurrent export. |
| Values and scopes | Selected arithmetic, matrices, reductions, normalization, JIT bodies, functions, branches, loops and shared symbolic batches have structural and numerical controls. Required rounding prevents unreviewed algebraic fusion. Wider symbolic, transformed and stateful combinations remain open. |
| Representation | Logical complex values use explicit component storage. Reviewed finite broadcast, conjugation, layout and selection preserve that representation and selected payload semantics. General complex arithmetic, complex multiplication and all indexing are not accepted by those results. |
| Fixed models and preparation | Selected original NNX/Linen MLP state sequences, LayerNorm configurations, GPT_MLP and explicitly prepared FP64 Linear examples have finite evidence. The complete original model, scientific and stochastic contracts remain required. |
| Artifact and sample tools | `get_precision_report` reads private SOURCE metadata; `validate_source_samples` and its guard bind selected CPU comparisons to the actual artifact and supplied criteria. These helpers do not provide public SOURCE export or general deployment certification. BF16 uses actual raw-word transport where NumPy binding is insufficient. |

### Function ownership and NNX isolation

The reviewed Function ownership repair releases model instances and weights
owned only by a completed or failed export, while preserving the ownership
needed by legitimate live JAX traces. It does not clear JAX caches globally.
The accepted controls include completed, failed and still-live trace cases.
Broader mutable state, RNG, automatic differentiation and transformed-scope
acceptance remain separate work.

NNX preparation now rejects unsupported mutable static payloads before running
or changing the original model. The focused rejection/isolation controls
establish containment. They do not establish general state migration: supported
stateful models still require reviewed isolation, genuine state/RNG transport,
feedback, replay and full numerical acceptance.

### Numerical defects and unresolved references

SOURCE floating `reduce_min`/`reduce_max` now preserve the tested NaN propagation
and signed-zero choices, including reciprocal infinity signs. Their strict
default-target gate is still **FAILED**: subnormal results differ between the
official native references and default ORT execution. An explicitly configured
denormal-as-zero ORT profile has finite passing evidence, but does not qualify
the default target or authorize unconditional input flushing.

Separate constant extrema witnesses disagree between official eager and fresh
or warmed JIT on the sign of zero and its reciprocal infinity. The required
native roles must agree. These contradictions remain **FAILED**; choosing the
more convenient role, weakening comparisons or replacing the compiler cannot
close them. The [dependency record](../../docs/about/dependencies.md) distinguishes
these native failures, converter repairs and ORT behavior.

The official-native BF16 TopK counterexample also remains mandatory and FAILED
where observed. The named user exception permits reviewed SOURCE publication;
it does not waive numerical, milestone or final acceptance. Later ordinary CI
passes do not establish that the upstream issue is repaired. Custom native
builds, patches and overlays are prohibited without separate explicit approval.

Other retained witnesses include intermediate FP16 square/sum rounding,
weak-bias/scaling compositions, optimized FP32 reciprocal behavior, exact
sequence results and subnormal matrix cases. Their earlier results keep their
original source, runtime settings and criteria. A green test that detects one
of these differences is evidence about the detector, not acceptance of the
calculation. Valid all-zero outputs remain valid; numerical equality alone does
not check signed zero, and universal NaN-payload equality is not required.

## Inventory and numerical acceptance

The [current inventory](current_inventory_epoch.md) retains **1,856 rows and
3,712 caller obligations**, plus manual, metadata-free, opaque and optional-model
obligations and explicit cohort differences. These are coverage obligations,
not passing scientific cases. Original sources, factory recipes, inputs,
structural callbacks and numerical criteria retain their provenance.

Strict metadata accounting is implemented, but its complete reports retain
**global ERROR and zero accepted scientific cells**. There are still **1,838
missing tolerance pairs**; the existing declarations also need scientific
review. Missing criteria cannot become an implicit exact test, a default
tolerance or a successful numerical outcome. Original callback failures remain
visible beside separately reviewed SOURCE structural counterparts.

A private proof-producer is still being developed. Finite JIT/storage, metadata
and consumer controls have passing evidence. Complete qualification and
production-row scientific acceptance remain open. This
harness work does not accept a production row: an `x + 1` control cannot prove
the original model, criteria, native roles or target results. The broader
Native3/ORT2 obligation requires the unchanged eager, fresh-JIT and warmed-JIT
references and both ORT optimization modes on the bound artifact and feeds.

## ONNX feasibility and runtime evidence

The [central kernel and behavior table](../../docs/about/dependencies.md#onnx-runtime-kernel-gaps)
is the maintained source for checked ORT versions and lowerings. It separates
public LEGACY handling from private SOURCE algorithms. A missing kernel differs
from an incorrect lowering or a numerical failure after successful execution.

Examples include tested CPU gaps for FP64 convolution, BF16 arithmetic and
FP64 transcendental kernels, and `Where` losing selected negative zero. Bitcast
support depends on type widths and opset: finite equal-width SOURCE cases have
opset-26 evidence, while unequal-width/packed-complex forms remain incomplete.
ORT 1.31.0's recorded opset-27 load rejection is a runtime failure; changing the
requested opset silently is not a repair. Locked-runtime evidence does not
qualify ONNX/ORT minimum versions.

Runtime optimization, provider options and hardware belong to numerical
validation. Disabling optional optimization does not prove kernel rounding.
Web packaging is an artifact format, not evidence of browser execution. The
[source contract](precision_target_state.md#onnx-feasibility-and-deployment-contract)
requires separate representation, implementation, execution and numerical
verdicts.

## Work that still blocks completion

The [full work plan](precision_implementation_work_plan.md#full-migration-phases-and-review-gates)
retains all B–F obligations: remaining indexing, complex arithmetic, control
flow, state/RNG/AD, FFT and linear algebra; every existing framework and option;
original models and fixed-source FP64 preparation; current inventory/replay and
scientific criteria; artifact/external-data, minimum-runtime and browser evidence;
and the coordinated public transition.

Original full-model B=2/B=5 inputs, both caller settings, state/reset sequences
and complete references remain required. Historical local resource stops and
crashes do not establish model acceptance or their cause, and hosted successes
do not close local resource holds. Focused examples cannot replace these models.
The next work is to finish the active CI repair, qualify independent production
proofs and resolve concrete semantics and reference failures within official
dependency cohorts. No release or completion date is assigned.
