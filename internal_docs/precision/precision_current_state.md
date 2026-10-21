# Computation Precision Current State

Public-behavior analysis baseline: 2026-10-17. Source snapshot:
[`fae17822e0114e6c00840b8f78adedb7ec36f74e`](https://github.com/enpasos/jax2onnx/commit/fae17822e0114e6c00840b8f78adedb7ec36f74e),
which includes the merged consumer-adaptation fix from PR #275 and the native
typed lowering changes from PR #274.

The authorized support floor is **Python >=3.11 / JAX and JAXLIB >=0.10.2 /
Flax >=0.12.8**. Dependency declarations and both checked lockfiles are applied;
retained metadata from an unpublished pre-restart wheel declares those bounds.
Existing locked package versions and Python variants are retained. Python 3.11 remains supported. The
SECONDARY stack is the exact framework-floor target; its newer ONNX/ORT and
other dependencies do not establish all-direct-dependency-minimum acceptance.
ORT 1.24.1 and the other declared bounds are unchanged.

Earlier JAX 0.8.1/Flax 0.12.1 runs retain their actual outcomes as historical
evidence; they are no longer a supported-floor release gate. This decision
does not fix numerical/provider gaps or remove any existing model, fixed source
or numerical requirement from the single-interface migration target.

The preceding exact native platform-selector and SOURCE `eye` forwarding
checkpoint is committed at **`b7eb923a`**, parent `3d9ba486`.
Its PRIMARY full passes **11,685 unique cases**; the same **110 focused cases**
pass on all three JAX stacks and actual ORT 1.24.1. Strict documentation,
**269 rendered links across nine docs**, all seven normal hooks and exact
qualified-byte verification pass. At commit verification, all **933 Python
inputs** matched the full run and the worktree was clean. The earlier
[platform record](poc/source_integration_findings.md#exact-platform-selector-and-native-eye-forwarding)
retains its failed branch-capture epochs.

The preceding generalized-contraction/support-floor checkpoint is committed at
**`7b70b22f`**. The current private checkpoint forwards Equinox Linear to its
captured original call with exact arguments and native errors, keeping the
original model sources, options, parameter bytes, state and criteria. It extends
the shared native contraction/transpose proof to one actual input-derived `B`
at admitted intermediate axes and binds that symbol at arbitrary admitted
output axes in exact artifact/report/sample declarations. Inputs retain the
positive leading-`B` contract; empty static output tails do not admit empty
symbolic inputs or `B=0`.

Mixed FP32/FP64 dots retain the actual native FP64 result or explicitly
preferred FP32 work after frozen original native inference. A local consumer
conversion occurs before the dot; original physical ports, stored values and
returned aliases retain their declared widths. The reviewed FP64 precision gate
is unchanged. Broader precision algorithms and dtype routes remain unfinished.

Static-empty real/discrete Add validates existing admission, native results and
both physical operands before emitting a fresh Identity or typed empty constant.
Only the output variable is bound. Nonempty and LEGACY arithmetic, symbolic
zero-tail owner gates and the dedicated complex branch retain their behavior.
Empty packed-complex provider behavior remains unqualified.

The actual direct-minimum subset passes **2,908 cases**, zero
failures/errors/skips. Its exact capability ledger is distinct from full
supported-functionality acceptance. SECONDARY affected coverage consists of
**2,527 unchanged-owner passes plus 83 corrected-owner passes** in separate
epochs; the original combined run remains **2,609 PASS / one FAIL of 2,610**.
The original PRIMARY full remains **FAILED: 12,410 PASS / six FAIL of
12,416**, zero errors/skips. Its coverage is accounted for by **12,226 unchanged
other-owner passes plus 190 corrected whole-owner passes** across two epochs.
Those same 190 identities pass on SECONDARY and at actual direct minima,
supplementing the earlier SECONDARY partition and 2,908-case minimum subset.
These are separate source-bound executions, with exact test-only deltas; no
unified green full or combined minimum run is claimed. Earlier dual v2 passes,
the failed minimum-v3 epoch, diagnostics and collection/import repair
remain separate in the [checkpoint record](poc/source_integration_findings.md#equinox-forwarding-shared-symbolic-contraction-and-output-axes).

The shared test evidence helper preserves same-artifact DISABLED acceptance and
specific default-ALL FP64 runtime-load failures, including the unchanged raising
gate and unexecuted later samples. Subnormal bitwise comparisons remain FAILED
under the original target settings. Passing negative-witness tests do not accept
those failed target configurations. Raw callback failures and four source
exclusions remain in the original Eqx/MLP observations; missing runtime/numerical
criteria stay **UNVERIFIED**. No original criterion or public default changes.

The earlier quality-v3 epoch passes before the target evidence helper. The
later quality-v6 marker failure is retained; its two-file comment-only repair
preserves the complete module ASTs and all other source pins. Actual quality-v7
passes all seven checks on that distinct 1,106-pin epoch. Strict docs, seven
normal commit hooks and held-byte/commit checks remain required. Captured outer Function
native JIT retains its specific missing-MLIR evidence gap; original eager/
warmed/fresh JIT and captured leaf checks remain independent. The full product
map, complete models, general fixed sources and B–F gates remain open. Public
behavior stays LEGACY; the future callable-preparation packet is separately
frozen, unapplied and native-unqualified.

The preceding private generalized-contraction implementation uses a shared
validated shape plan: supplied contracting and batch-axis pairing is retained,
free axes retain native order, and only the required consumers transpose or
reshape values. Same-width FP32/FP64 and the existing explicit
FP16-operands/preferred-FP32-result boundary retain native abstract evaluation
and physical metadata checks. Empty reductions produce the native additive
identity; no-reduction products retain signed zero. The established canonical
tensor-by-matrix route remains one MatMul. Individual dimensions outside ONNX
INT64 and an unfinished oversized regrouping strategy have distinct reports.

SOURCE LinearGeneral/DenseGeneral and JAX/NNX/Linen einsum call their captured
native implementations with exact arguments before legacy normalization. Native
promotion, callbacks, parameter storage, aliases, named Functions and JIT
context remain observable. Other arithmetic types, explicit algorithms and
broader symbolic forms remain implementation/qualification work; the finite
boundary is not a permanent capability exclusion.

The v3 selection passed the same **993 focused identities on all three
then-installed JAX stacks**, with zero failures/errors/skips and held sources. The new owner contributes **376**
cases. All three preceding 988-PASS / five-FAIL epochs remain failed history.
The four historical minimum-stack named examples executed and validated against both
native roles under both ORT optimization settings with unchanged bitwise
criteria. Their actual captured function bodies, serialized ports, original
kernel and output aliases are checked. The 22 original dot-row/caller replays
remain separate: missing criteria and failed callbacks do not become numerical
acceptance. Typing passed on **559 source files**; the **17 RNG factories** are
an inventory count. Pinned Black/Ruff and all four policies passed for v3.

The subsequent complete PRIMARY v3 run is **FAILED: 11,992 PASS / 69 FAIL
of 12,061 unique cases**, zero errors/skips, **691.290155 seconds**. All 11,685
preceding IDs, all 993 focused IDs and all 1,098 source/config pins were
retained; the only 376 additions are the generalized-contraction owner.
Both v4 affected runs also remain **FAILED: 1,419 PASS / 21 FAIL of 1,440**,
zero errors/skips, with identical IDs and held sources.

The v5 fixture correction retains each intended corruption using a fresh
native annotation on the same test variable and checks the specific unknown
physical-dimension error. It introduces no production mutable proxy, inference
replacement, skip or changed numerical bound. The same **1,443 affected cases
pass on SECONDARY and PRIMARY**, with zero failures/errors/skips: all 1,440
previous identities plus the three minimum-installer tests. Sources and Git
state hold within both runs; all 934 Python inputs still match after the
separately reviewed lock update.

The completed v5 PRIMARY full run passes **12,061 unique cases**, with zero
failures, errors or skips, in **660.279726 seconds**. Every earlier v3 full
identity and all 1,443 affected identities are present. Its 1,098 source/config
pins matched the held source checkpoint before this five-document status
update; all 934 Python inputs remain unchanged. This installed
PRIMARY environment is Python 3.12.9 / JAX and jaxlib 0.12.0 / Flax 0.12.10;
it is distinct from Poetry's Python 3.12 lock selection.

The original SECONDARY full run was interrupted by the VM restart. Its log
survives, but no terminal receipt or XML exists, so it has no passing result.
The new complete SECONDARY run finishes with **12,054 PASS / two SKIP of
12,056 unique cases**, zero failures/errors, in **1,836.001337 seconds**.
It uses Python 3.11.12 / JAX and jaxlib 0.10.2 / Flax 0.12.8 and exactly the
same 1,098 pinned source/config files as PRIMARY. All 1,443 affected identities
are present and pass. The two skips are the absent optional Keras integration
and JAX's absent `one_minus_square` primitive. Five PRIMARY-only identities
are three feature-gated `no_collectives` context cases and two generated
`one_minus_square` cases; no other identity is missing. These conditional
cases and skips are explicit limits of this installed-stack result.

The separate fresh Python 3.11 all-direct-minimum lane verifies all fifteen
runtime dependencies at their declared lower bounds, including ORT 1.24.1.
Its first actual sixteen-owner CI replay is **FAILED: 464 PASS / one FAIL of
465 cases**, zero errors/skips: the isolated setup omitted the declared
`mypy==2.3.1` development tool required by one typing test. This is a test-tool
setup failure; that guard did not execute its intended typing check. The
failed result is retained without changing source, criteria or runtime floors.
After installing the declared tool, the unchanged local sixteen-owner replay
passes **all 465 unique cases**, zero failures/errors/skips, in **104.421656
seconds**. It retains exactly the failed epoch's 465 identities and the same
1,098 original source/config and 960 execution-Python file hashes. This is a
local replay of the CI selection at all fifteen direct runtime minima, not a
hosted CI run or complete supported-functionality/B–F acceptance.

Checkpoint acceptance requires final quality, strict docs, all seven normal
hooks and qualified-byte verification. Full existing-functionality and B–F
acceptance remain separate obligations.

Phases are **NOT_INSTRUMENTED**.

An external coordinator preserves the full **1,852-row / 3,704-caller**
product plan in bounded fresh processes with resumable durable receipts. Its
historical smoke on immutable `b7eb923a` completed two cells and exactly one new
cell on resume, preserving all prior evidence. All three stop at the existing
Equinox Linear SOURCE barrier. Native results and state are retained; no ONNX
artifact or runtime/numerical acceptance is implied. Those exclusions belong
to that earlier immutable source. The next full map must use the newly
qualified and committed Equinox/shared-symbolic/output-axis checkpoint; its
qualification is still pending. See the [historical smoke record](poc/source_integration_findings.md#generalized-native-contractions-and-product-map-smoke)
and the [current checkpoint](poc/source_integration_findings.md#equinox-forwarding-shared-symbolic-contraction-and-output-axes).

A separate [BF16 elementwise diagnostic](poc/source_integration_findings.md#bf16-elementwise-feasibility-observations)
now has actual PRIMARY CPU observations: direct BF16 models fail to load on
that provider, while FP32 operations with BF16 castback execute but fail the
experiment's frozen bitwise comparisons. Original eager/fresh-JIT references
and all raw artifacts are retained. This finite result does not admit BF16
arithmetic, change missing raw criteria or prove a general target limitation.

The BF16 CPU sample transport checkpoint remains committed at `3d9ba486`, with
its **11,668 PRIMARY full passes**, same **275 focused cases on all three JAX
stacks**, and **41 new cases at actual ORT 1.24.1**. It preserves physical BF16
inputs/outputs through one original execution without numerical conversion,
including dynamic sizes. Missing kernels and failed samples remain distinct
from transport qualification; the [BF16 record](poc/source_integration_findings.md#bf16-cpu-sample-transport-qualification)
retains their exact evidence. Complete supported-floor and B–F acceptance
remain open; public behavior stays **LEGACY**. The authorized support decision
does not establish complete-model or numerical acceptance. Earlier checkpoints
below retain their historical scopes and support policies.

Private same-width complex64/complex128 Add is committed at `18f3fcd7`, with
finite native/CPU qualification for static shapes at opsets 23/26. It retains
declared pair storage and JAX's chosen width. The [latest checkpoint](poc/source_integration_findings.md#complex-add-and-partial-ordering-checkpoint)
separates those samples from the partial initial raw map and the **FAILED**
ordering-kernel diagnostics. Public behavior remains **LEGACY**; all existing
functionality, fixed sources and full B–F acceptance remain open.

The preceding private [CNN, pooling, scientific FP64 and raw-replay checkpoint](poc/source_integration_findings.md#cnn-pooling-scientific-fp64-and-raw-replay-checkpoint)
is committed at **`118ddd8b`** after independent finite-checkpoint acceptance,
corrected dual-stack checks, the complete PRIMARY suite and all seven normal
code hooks. It retains the unchanged full-CNN nonzero comparison, native pooling
and selected scientific FP64 at opset 21. The raw lane's 49 maintained controls
do not run the full 1,852-row / 3,704-caller map. Public behavior stays **LEGACY**;
the full B–F migration remains open.

The preceding explicit FP64 Linear code checkpoint is `b2a866bb`. Its strict
documentation build passed, and its eight-document update is committed at
`97929998`. The canonical findings retain its original runtime and failed epochs.

The public-behavior and implementation analysis below retains that baseline.
The later local branch integration is identified separately: private converter
code now carries `PrecisionPolicy.SOURCE`, while the public API and legacy
defaults remain unchanged. See the
[integration findings](poc/source_integration_findings.md) for its bounded scope.

Current private elementary/reduction evidence: **183 cases / 549 phases PASS
per stack**, zero test failures/errors/skips, independently accepted from actual
artifacts and saved comparisons. Native elementary/reduction bindings and owning
typed lowerers now cover their reviewed finite contracts, including FP32/FP64
Asin using same-width rational coefficients and a half-angle expression. The
original FP32 consumed Asin sum retains exact positive zero against both native
roles and all four target outputs; new shared-B Asin controls cover B=2/5.

Test success is separate from numerical acceptance. Eight original Sign reports
remain **FAILED**, 44 original composition artifacts remain numerically
**UNVERIFIED**, and both original Gather raw callbacks remain **FAILED** despite
separately passing finite numerical predicates. All 318 original elementary
contracts / 636 caller obligations and the 1,852-row / 3,704-caller inventory
remain unchanged. The [current checkpoint record](poc/source_integration_findings.md#elementary-and-reduction-checkpoint)
binds the exact counts, source epochs and retained failures.

The preceding accepted local code checkpoint **`11748451`** is committed. Its
complete PRIMARY suite
passes **10,237 cases / 30,711 phases**, and the focused selection passes
**1,937 cases / 5,811 phases on each supported stack**, with zero test
failures/errors/skips in those executions. All seven normal code hooks pass;
86 committed Python paths (63 production and 23 test/support) match the qualified
bytes, all **923 Python / 163 other input pins** hold through hooks and commit,
and the worktree is clean at commit verification. The focused and full results
have independent finite-checkpoint acceptance. These results retain the separate
failed numerical gates; they do not complete the precision migration.

Generic SOURCE admission accepts explicit opsets **16, 18, 21, 23, 24, 26
and 27**, with operation/type-specific checks. Native forward scan now uses
ordinary admission at **23/26**, and while at **18/23/26**, including the original
opset 18 row. Typed carries, actual JAX 0.11 partitions, recursive bodies, static
complex/empty outputs, runtime flags and native result weakness retain their
finite contracts. Strong static FP32/FP64 zeros are admitted at **23/26** only.

The seven-path loop checkpoint **`c033d301`** is committed after independent
acceptance: **477 cases / 1,431 phases on each stack** and the complete PRIMARY
suite's **10,395 cases / 31,185 phases** pass, with zero failures/errors/skips.
All old 440 focused identities and original metadata remain; the typed-key test
only aligns its message regex. The full collection retains all old 10,237 cases
in order plus 158 additions. All seven normal code hooks pass, and all 1,088
source/input files hold through commit. All **51 raw loop rows** remain
migration obligations rather than a closed family. See the
[loop checkpoint](poc/source_integration_findings.md#loop-admission-and-first-saved-model-samples) for exact gates and retained failed epochs.

The four original GPT_MLP caller/stack cells have independently reviewed
focused15-v2 numerical evidence at unchanged strict **1e-5**: original
768 → 3,072 → 768 dimensions, FP32 model/state, seed, zero Dropout and
approximate GELU. The actual typed `(x*x)*x` expression, coefficients,
second Linear and key-payload provenance are retained. Saved comparison
receipts carry bounded acceptance; export reports keep runtime/numerics
**UNVERIFIED**. This does not close the 44 original composition obligations.

Earlier tensor/index and Erfc evidence retains its recorded scope. Erfc graph/schema checks cover opsets 23/26; configured CPU numerics cover
opset 23 only. Its strict full-seven-output and native-agreement gates remain
**FAILED**, and FP16/BF16 numerics remain **UNVERIFIED**, with no production FTZ
or runtime-policy change.

All six original normalized014/015 source selectors now have independently
accepted structure and exact captured-sample evidence for **eight saved artifacts
per stack**. Actual CPU target runs produce **384 returns / 16 sessions per stack**
(**768 / 32 total**), retaining four native roles, actual state/flags and both
optimization settings. Original014 keeps its reference-first tolerance;
original015 uses the separately accepted strict atol-zero proposal, while its
historical `atol=None` and raw missing criteria remain unchanged.

Each stack retains **six FAILED / two PASSED** raw graph callbacks and unchanged
stored reports with runtime/numerics **UNVERIFIED**. The PRIMARY capture's exit99
and missing final origin record remain failed history, with a later accepted
saved-computation audit; SECONDARY completes its own capture and reviews. These
results bind the frozen `11748451`-derived snapshot, rather than fresh current
MAIN model exports. See the [latest saved-artifact record](poc/source_integration_findings.md#original-six-saved-samples-and-native-conv-integration).

The native FP32 Conv and Scan physical-metadata paths are committed at
**`3223d080`** after independent finite-checkpoint acceptance, over 1,091
frozen source/input files. Both earlier focused failure epochs are retained with corrected secondary counts. Latest PRIMARY passes
687 cases; SECONDARY has 684 passes / three unchanged legacy runtime-fusion
skips, pytest zero but strict-wrapper exit 99. All 586 converter-selection cases
pass on each stack; numerical assertions after those legacy skips did not run.
Production typing on 551 files, 17 RNG factory report entries, and final fixture style/AST checks
pass. The complete PRIMARY suite passes **10,504 cases / 31,512 phases**, with
zero failures/errors/skips. All seven normal code hooks pass; all 1,091 inputs
hold through commit and the worktree is clean at commit verification. The later
CNN/pooling implementation is included in the independently accepted current
code checkpoint linked above. Explicit FP64 Linear preparation
is now applied in eleven paths: seven implementation/PoC/test paths, two
diagnostic-expectation files and two audited test/tool provenance constants.
Accepted dual936 evidence carries on identical production; fresh dual145 and
the full PRIMARY 10,596-case v4 pass. Independent final readiness and all seven
normal code hooks pass; commit `b2a866bb` preserves the eleven qualified paths.
This does not close broader scientific FP64 work.

The unchanged [masked FP64 saved artifact](poc/source_integration_findings.md#masked-fp64-saved-sample)
passes its original 5e-8 / 1e-12 criterion on both ORT stacks using saved PRIMARY
eager/JIT references. Four target returns are bitwise identical; eight
comparisons / 48 scalars have maximum absolute error 4.440892098500626e-16.
That saved-artifact result makes no fresh SECONDARY native claim. The later
scientific selection separately passes 141 cases / 423 phases per stack,
including unchanged Issue52 at opset 21. Broader scientific contracts remain
open; the current checkpoint's focused/full integration gates have passed.

Broader models, the optional 45 MaxText/five MaxDiffusion obligations, general
fixed-source preparation, low precision and scientific FP64 remain open.
Public behavior stays **LEGACY** until the complete B–F contract is met.

For observable behavior and export choices, read the
[user view](precision_current_state_user_view.md). All companion documents are
listed in the [precision overview](README.md).

Computation precision is governed by overlapping tracing, lowering, and
postprocessing rules. The double-precision setting promotes many values, while
explicit casts and selected scopes preserve lower precision. The resulting
behavior is not one uniform dtype contract. The proposed
[target state](precision_target_state.md) and
[migration plan](precision_migration_plan.md) describe how to resolve this.
The merged fix repairs selected consumer type mismatches within the existing
policy; it does not implement that broader target.
The local experiments and committed private integration include core changes. Their successful
samples, known gaps, and separate feasibility controls are summarized below;
they do not establish a released source mode.

## One Argument Controls Several Decisions

The public `enable_double_precision` argument defaults to `False`. Its current
meaning combines tracing configuration, defaults for unspecified dtypes, IR
type selection, constant conversion, and final export postprocessing.

The public facade uses `_temporary_x64`, an alias of
`conversion_api._force_jax_x64`. This context sets `jax_enable_x64` to the
argument's Boolean value and restores the previous setting afterward. Thus
`False` actively disables x64 during tracing, including when the caller had
enabled it. `True` permits JAX to trace FP64 computations; enabling x64 alone
does not make an explicitly FP32 JAX calculation FP64.

`_as_sds_list` assigns FP32 to shape-only input specifications under `False` and
FP64 under `True`. Explicit `ShapeDtypeStruct` dtypes initially survive this
normalization. Later tracing and lowering still apply their own rules, so this
initial preservation does not establish an end-to-end dtype guarantee. See
[conversion_api.py](../../jax2onnx/converter/conversion_api.py) and
[user_interface.py](../../jax2onnx/user_interface.py).

The baseline normalization reconstructs `ShapeDtypeStruct` from shape and dtype
and drops `weak_type`. That JAX property affects type promotion, so retaining
the dtype alone can change the computation JAX traces. Public legacy calls
retain those observed normalization rules. The initial source experiment
inherited the same loss in both the UX wrapper and shared tracing helper.
The later private source policy now preserves the property through both steps,
including direct helper calls, and reports supplied/normalized/traced flags.
The [regression](poc/migration_ux_findings.md#weak-input-properties-preserved-on-the-source-path)
returns FP16 `2048` for a weak scalar plus explicit FP16 `1`, while a strong
input with the same shape/dtype returns FP32 or FP64 `2049`. Both supported
stacks cover this source correction; it does not change legacy normalization.

The [calling experiment](poc/migration_ux_findings.md) compares real public calls
with the argument omitted, explicitly `False`, and explicitly `True`, for
shape-only inputs, typed specifications, NumPy arrays, and JAX arrays under
both ambient x64 settings. Omission and `False` currently select the same
behavior. An FP64 `ShapeDtypeStruct` on those routes is truncated during JAX
tracing: JAX 0.10.2 is silent for the tested case; JAX 0.11.2 warns. Both warn
for a model-internal `.astype(jnp.float64)` under disabled x64. These are
observed version differences, not diagnostics guaranteed by the exporter.

The shared `numpy_dtype_to_ir_with_float_policy` maps FP32 to ONNX `DOUBLE`
when the argument is `True`. It preserves FP16, BF16, and FP64 in its own
mapping. An unspecified dtype also follows the argument. Therefore this helper
does more than translate a dtype already chosen by JAX. Conversely, its FP64
case does not downcast under `False`; tracing and context allocation have
additional behavior. See [ir_utils.py](../../jax2onnx/ir_utils.py).

Historically, this combination was intentional. The
[v0.6 implementation](https://github.com/enpasos/jax2onnx/commit/b646501ae07737e011a4f12e460c9c14f8da4773)
described a double export mode and a single precision downcast mode. Subsequent
fixes introduced important exceptions. Recasting the existing argument as only
an x64 permission would consequently change observable behavior.

## Lowering Has Several Type Authorities

The current native context separates `require_value_for_var` for inputs,
`lookup_value_for_var` for inspection, and `allocate_value_for_var` for writable
outputs. The compatibility helper `get_value_for_var` still reuses an existing
binding or allocates one. Allocating an output no longer means blindly reusing
an input or an already produced result. These ownership improvements do not
remove the precision policy: allocation still consults the Boolean and the
traced `aval.dtype`. See [ir_context.py](../../jax2onnx/converter/ir_context.py) and
[typing_support.py](../../jax2onnx/converter/typing_support.py).

Plugin output types can then follow a traced dtype, an allocated placeholder,
an operand, or a plugin-specific rule. For example,
[`convert_element_type`](../../jax2onnx/plugins/jax/lax/convert_element_type.py)
uses the exact traced target dtype, whereas general context allocation can
promote an FP32 placeholder. Final output binding also keeps existing floating
types in cases where they differ from its policy-derived target. Reading only
the JAXPR or only the final public output type therefore cannot establish the
precision of the whole exported calculation.

The implemented
[`promote_float32_operand`](../../jax2onnx/plugins/_precision_utils.py) helper
now adapts a FLOAT operand when its caller explicitly requires DOUBLE. The
adapter inserts a separate Cast and keeps the original binding available to
other consumers. Native Neg, selected unary and real binary operations,
reduction helpers, and concatenate paths use this adaptation. The explicit
FP64-to-FP32 rounding remains before the consumer's FP32-to-FP64 conversion.
This is a bounded compatibility repair, rather than a global precision rewrite
or a new interpretation of every `astype` expression.

Explicit dtype preservation is a tested boundary. The
[#217 cast fix](https://github.com/enpasos/jax2onnx/commit/e76a90eac15d4cef8dc5f8e5d06d7f70e2720912)
deliberately retained explicit cast targets. Current
[exact-dtype tests](../../tests/extra_tests/converter/test_exact_dtype_casts.py)
require FP32 graph outputs under `enable_double_precision=True` for casts and
selected constructors: zeros, ones, eye, full, linspace, and arange. They do
not establish preservation of every intermediate in arbitrary compositions.
The older bitcast target assertion is structural, and plugin metadata still
skips numerical validation. A new regression additionally checks a complete
exported FP32-to-UINT32 bitcast at opset 26 through the checker and CPU runtime,
while the same FP32 value also feeds a DOUBLE consumer. This is specific runtime
evidence, rather than general bitcast coverage.

FP16 and BF16 need separate consideration. Their shared mapping preserves
their types, and the context's disabled-double output allocation exempts both
from its general floating downcast. The FP16 placeholder regression covers
both Boolean settings. These facts do not prove complete FP16/BF16 constant,
operator, or runtime coverage. In particular, context constant promotion still
widens NumPy floating arrays, including FP16, under double export; the mapper's
exceptions are not universal rules for every path.

## Constants and Nested Graphs Add Further Rules

Closed-over constants are reconciled with their JAX variables and can then be
promoted before binding. `IRContext.bind_const_for_var` applies another
promotion path. At export time, `postprocess_ir_model` promotes FP32
initializer payloads, tensor-valued `Constant` attributes, and constant-backed
node outputs when requested. It recursively visits control-flow subgraphs and
ONNX Function bodies. This occurs after the converter's optimization pass.
See [ir_postprocess.py](../../jax2onnx/converter/ir_postprocess.py).

These paths can turn a precision decision into more than a metadata change:
they alter stored tensor payloads. The postprocessor does not implement a
complete typed graph transformation that reconciles every consumer and cast.
The existing
[constant-promotion tests](../../tests/extra_tests/converter/test_f64_constant_promotion.py)
require DOUBLE initializers and numerical agreement for selected captured
constant expressions. Separate
[postprocessing tests](../../tests/extra_tests/converter/test_ir_postprocess.py)
check direct payload promotion. Neither suite establishes arbitrary mixed
precision correctness.

Child contexts inherit the precision setting. Control-flow contexts also
inherit `_keep_function_float32`; selected scan paths enable that flag to
retain lower precision body values. Function boundaries have explicit formal
inputs, while eligible If branches can capture lexical values. These scope
rules and scan exceptions must be considered together with recursive late
promotion. The
[scan regression](../../tests/extra_tests/framework/test_one_minus_square_scan.py)
exercises a float32 body under double export, so preserving only public
outputs would miss an existing requirement.

## Numerical Accuracy Already Has Local Policies

Storage dtype, operation dtype, and accumulator precision are distinct. The
specialized explicit LEGACY LayerNorm lowering already widens selected internal
work independently of `enable_double_precision`: FP16/BF16 statistics use
FP32, and the FP32 slow-variance reduction accumulates squared centered values
in FP64 before rounding the variance to FP32. The remaining FP32 statistics
and affine computation retain FP32. This is a local numerical algorithm,
rather than a request to turn the entire model into FP64.

The [documented limits](../../docs/user_guide/known_limitations.md) and
[LayerNorm tests](../../tests/extra_tests/test_layer_norm_precision.py) distinguish
error against JAX from error against an independent FP64 reference. Their
fixed bounds cover specified FP32 cases, inputs, opset, runtime provider, and
optimization settings. They provide no general guarantee of bitwise equality,
universal improvement from widening, or equivalent bounds for every
normalization family.

The private SOURCE path preserves the framework's normalization choices.
The later [bounded normalization acceptance](poc/source_integration_findings.md#mlp-state-and-bounded-normalization-acceptance)
adds selected static FP16/FP32/FP64 exports and CPU comparisons. Its named-scope
payload proof evaluates the actual exported body eagerly and with JIT;
whole-carrier JIT equality remains unestablished. The recorded SOURCE criteria
are separate from these legacy bounds and do not certify arbitrary
LayerNorm-containing models.

## What the Precision PoCs Establish

The [numerical findings](poc/numerical_contract_findings.md) compare JAX eager
and JIT execution, ONNX ReferenceEvaluator, and ONNX Runtime CPU with optional
optimization disabled and enabled. Both installed stacks reproduce the
observations: JAX 0.11.2 / ORT 1.31.0 and JAX 0.10.2 / ORT 1.29.0. The findings
record exact versions and inputs. The artifact origin matters:

| Case and origin | ONNX validity | Execution and numerical observation | What remains open |
| --- | --- | --- | --- |
| Public export: FP16 batched `dot_general`, preferred FP32 result | Full checker passes. | JAX returns FP32 `2049`; exported `MatMul` returns FP16 `2048` on ReferenceEvaluator and both ORT settings. | This public legacy result-type gap remains; the private source lowering below handles a bounded subset. |
| Private source export: default-precision FP16 rank-2/rank-3 matrix operands, explicitly preferred FP32 result | Full checker passes. | Operand casts precede an FP32 `MatMul`; the `2049` witness and additional finite-value cases retain FP32 results while shared input values remain FP16. | Other layouts, dtypes, explicit precision algorithms, targets, and performance require separate work. |
| Changed JAX source: explicitly FP32 matrix operands, public export | Full checker passes. | All tested paths return FP32 `2049`. | Sample agreement does not make this a validated automatic lowering or establish its hardware and performance properties. |
| Hand-built normalization control: FP32 centering/squares, FP64 variance sum and division, FP32 output | Full checker passes. | JAX, independent staged reference, ReferenceEvaluator, and both ORT settings agree on variance `8388609`; a NumPy FP32-only sum gives `8388608`. | This original control establishes feasibility; the local source integration below now exports the complete callable. |
| Private source export: the same selected-FP64 normalization, inline, in a Function, and in either conditional branch | Full checker passes after save/reload. | Six boundary/optimization cases retain exact centered values, squares, and variance, with a tolerance on normalized outputs. | This covers the stated calculation and inputs, rather than general normalization families or deployment targets. |
| Source-PoC export: FP16 square then sum | Full checker passes; ReferenceEvaluator agrees with JAX on the sample. | JAX returns zero squares and zero sum. Both ORT settings return zero squares but sum `2**-23`. | The tested runtime fails the sample's required intermediate rounding despite successful execution. |
| Hand-built unfused FP16 `Mul -> ReduceSum` control | Full checker passes. | The same ORT rounding discrepancy occurs. | This rules out blaming converter fusion alone; it does not identify the underlying cause. |
| Changed JAX source: explicit FP32 multiplication, FP16 rounding, FP32 sum, FP16 result; source-PoC export | Full checker passes. | Every tested path returns zero with the original FP16 output signatures. | The arithmetic change is only a bounded source candidate, not a general exporter repair. |

For normalization, exact centered values, squares, and variance are checked;
only the final normalized output uses a tolerance. A broad relative tolerance
on variance would accept losing the very terms the experiment is designed to
preserve. The constant input row also has an exact zero result.

The FP16 discrepancy remains when optional ORT graph optimization is disabled.
That setting does not establish the arithmetic performed inside a kernel.
Neither ReferenceEvaluator agreement nor the unfused control proves a general
rounding guarantee or identifies a converter fusion bug.

The [calling experiment](poc/migration_ux_findings.md) adds input normalization,
actual tracing observations, option conflicts, and warning reporting around
the existing entry points. Explicit FP64 inputs with x64 disabled are rejected before tracing
on the candidate source route; a model-internal FP64 conversion can already
have been truncated by JAX and is reported through its actual warning.

The later [private integration](poc/source_integration_findings.md) replaces
the local allocation subclass with `PrecisionPolicy.SOURCE` in the real
`IRContext`. It reaches audited `onnx_function` and two-branch `cond` children,
preserving selected mixed outputs, shared FP32 values, and FP32/FP64 constants
through serialization and CPU execution. Floating `gt` and Boolean-to-INT32
selector conversion cover the tested condition wiring. Recursive validation
checks branches and actual retraced function bodies. The later Where/JIT
checkpoint adds bounded pure closed JIT bodies; broader loops/JIT forms and
other unaudited operations remain excluded.

The source function variant carries flattened result abstract values and
reconstructs structured returns. It now captures `jax_enable_x64` and
`jax_default_matmul_precision` at each call site, replays them while tracing the
body, and includes them in function cache signatures alongside weak flags,
shape/dtype, and captures. Already-typed FP64 formals survive a local x64-off
scope; recreating the formals must not truncate a value that already existed.
The retraced result signature is checked against the original call.

Review exposed two distinct faults that the local implementation now repairs:
replaying a function under ambient rather than call-site x64 could return FP32
`0` instead of `1`, and JAX's cached conditional branch could retain a source
primitive during a later public legacy export. A private tracing discriminator
now separates ordinary JAX, legacy export, and source export caches. It uses
JAX's trace/JIT cache context through the compatibility layer rather than
clearing caches. The
[context regressions](../../tests/extra_tests/converter/test_source_function_context.py)
cover both local x64 directions, unchanged result dtypes, mixed call-site
settings in one graph, nested functions, scoped matrix requirements, failure
restoration, and export sequences around ordinary eager/JIT execution.

The [input](../../tests/extra_tests/converter/test_source_inputs.py) and
[scope](../../tests/extra_tests/converter/test_source_scopes.py) regressions now
live in the regular repository suite. They cover static arguments, structured
returns, repeated aliases, and input passthrough as well as mixed types.
The bounded source
[matrix lowering](../../tests/extra_tests/converter/test_source_dot_precision.py),
[normalization integration](../../tests/extra_tests/converter/test_source_normalization.py),
and [arithmetic-boundary checks](../../tests/extra_tests/converter/test_source_arithmetic_fusion.py)
also run there. The source path retains `Add -> Div` and `Mul -> ReduceSum`
instead of applying the legacy Mean and ReduceSumSquare substitutions without
an audited numerical equivalence argument.

The [sample acceptance helper](../../tests/extra_tests/converter/precision_runtime_support.py)
records the actual CPU runtime, optimization mode, architecture, thread counts,
and comparison result. It rejects failed or unverified sample acceptance and
keeps execution errors separate. It does not prevent export or implement a
public target-selection API. In particular, the known FP16 square/sum model
still exports and executes, but fails acceptance against its exact reference
on the recorded runtimes. That numerical defect remains unresolved.

Historical validation totals are recorded in the
[integration findings](poc/source_integration_findings.md#recorded-validation);
the [historical option checkpoint](poc/option_and_array_creation_findings.md#validation-and-provenance)
records its complete 5,803-test run. The
[current checkpoint](poc/source_integration_findings.md#tensor-index-erfc-and-optional-inventory-checkpoint)
records the new 10,047-case primary pass and local code checkpoint `1533d60b`
with seven passing normal code hooks and accepted independent final code-evidence
review.
A green suite can include tests that recognize the known numerical failure;
it does not establish model-wide or deployment-wide precision compliance.

## ONNX Feasibility and Runtime Evidence

Precision support has separate schema, model-validity, runtime, and numerical
boundaries. The [ONNX Conv schema](https://onnx.ai/onnx/operators/onnx__Conv.html)
allows DOUBLE tensors, but this does not establish a corresponding kernel in
every execution provider. Focused standalone ONNX probes on 2026-10-17 used
Python 3.12.9, ONNX 1.23.2, ONNX Runtime 1.31.0 CPU, opset 23, IR version 10,
disabled runtime graph optimization, and one thread:

| Minimal operation | Full ONNX checker | ONNX Runtime CPU result |
| --- | --- | --- |
| FP64 `MatMul` | Passed | Loaded and executed |
| FP64 `Conv` | Passed | Load failed with `NOT_IMPLEMENTED` |
| BF16 `Add` and BF16 `Neg` | Passed | Both loads failed with `NOT_IMPLEMENTED` |

The BF16 probes used FLOAT inputs, explicit casts to BF16, the tested operation,
and a cast back to FLOAT. These are narrow kernel availability observations,
not complete jax2onnx exports or universal statements about these dtypes.

The [standard BitCast schema](https://onnx.ai/onnx/operators/onnx__BitCast.html)
starts at opset 26 and requires equal input/output bit widths. A separate
minimal FLOAT `[1]` to INT32 `[1]` reinterpretation failed checker and runtime
loading at opset 23, but passed both and returned `1065353216` for `1.0` at
opset 26. The current
[bitcast plugin](../../jax2onnx/plugins/jax/lax/bitcast_convert_type.py)
emits `BitCast` without an opset gate, while the public default remains opset
23 and its metadata skips numerical validation. The successful probe at 26
and the merged exported-model regression do not establish full JAX bitcast
coverage, especially for unequal bit widths.

Native [LayerNormalization](https://onnx.ai/onnx/operators/onnx__LayerNormalization.html)
also separates data and statistics: its data type parameter includes DOUBLE,
but its statistics type parameter allows FLOAT or BF16. Default `stash_type=1`
specifies FP32 normalization statistics. Thus DOUBLE input/output declarations
alone do not promise FP64 accumulation. This schema observation is not a
demonstrated CPU accuracy regression.

Runtime math settings add another boundary. ONNX Runtime's
[CUDA `use_tf32` option](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#use_tf32)
allows reduced-precision multiplication for certain FP32 matrix products and
convolutions while retaining FP32 tensor types. This is a documented provider
behavior, not a result of the CPU probes above; validation evidence needs the
relevant provider settings as well as the model's tensor dtypes.

Current CPU validation helpers and `export_mode="web"` have narrower roles:
the former execute selected comparisons on CPU, and the latter selects a
serialization profile. Neither is a general runtime capability or numerical
acceptance check. See [user_interface.py](../../jax2onnx/user_interface.py).

## Observable Behavior on Current Main

Focused checks at this snapshot used JAX 0.11.2 on CPU, ONNX 1.23.2,
ONNX Runtime 1.31.0, JAX x64 enabled, and `enable_double_precision=True`.
They show the distinction between source computation and later export promotion:

| Model expression | Tracing input dtype | Exported output dtype |
| --- | --- | --- |
| `x.astype(jnp.float32)` | FP32 | DOUBLE, with no Cast node |
| `x.astype(jnp.float32)` | FP64 | FLOAT, through a Cast node |
| `x.astype(jnp.result_type(x.dtype, jnp.float32))` | FP16 or BF16 | FLOAT, through a Cast node |
| `x.astype(jnp.result_type(x.dtype, jnp.float32))` | FP64 | DOUBLE, with no Cast node |
| `-x.astype(jnp.float32)` | FP64 | DOUBLE, after FP32 rounding and a consumer Cast |

The same-type FP32 cast disappears during JAX tracing. Later promotion therefore
cannot consistently distinguish an explicit source cast from an unchanged
FP32 value. The minimum-precision expression illustrates why its input dtype
must already reflect the intended precision when JAX traces the model. A
literal `astype(jnp.float32)` requests exactly FP32; it cannot also communicate
that FP32 is only a minimum for numerical stability. Preserving that cast
alone therefore does not resolve the minimum-precision intention. The proposed
[FP32 model-side example](precision_target_state_user_view.md#example-exactly-fp32-or-at-least-fp32)
uses computation configuration or a minimum-precision expression before
tracing, rather than having export silently reinterpret an exact conversion.
This is one example of operation-specific intent, rather than a general
requirement to widen lower precision values to FP32.

The original failing example from PR #275 now has consistent operand and
result types. For an FP64 input, `return -x.astype(jnp.float32)` follows:

```text
Cast: DOUBLE input -> FLOAT result
Cast: FLOAT result -> DOUBLE consumer operand
Neg:  DOUBLE input -> DOUBLE result
```

The merged runtime regression checks this result against negation of the input
rounded to FP32 and converted back to FP64. A direct output of the same rounded
value remains FP32, including when a DOUBLE consumer also uses it.

## Completed Legacy Repair and Remaining Boundaries

[PR #275](https://github.com/enpasos/jax2onnx/pull/275) is merged in this snapshot.
Its final implementation uses explicit consumer adapters, including the native
Neg path, rather than promoting values as a side effect of general lookup.
Graph output binding continues to require the existing computed value.

The focused
[exact-dtype suite](../../tests/extra_tests/converter/test_exact_dtype_casts.py)
passed all 26 cases locally at this snapshot. It covers Neg and Add after an
explicit FP32 cast; selected unary, real binary, reduction, and concatenate
consumers; shared direct outputs; function and layout boundaries; boolean and
integer results; and the opset-26 bitcast above. Runtime cases first run the
full ONNX checker, then execute the serialized model on CPU with optimization
disabled and one thread. FP16 Neg has runtime coverage here; BF16 Neg checks
types and model validity because the tested CPU kernel is unavailable.

The tests establish that these selected consumers can compute in DOUBLE after
the required FP32 rounding without changing the canonical FP32 value. They do
not establish source-type preservation for the full program. A fresh check of
`y = x.astype(jnp.float32); return (y + 1) - y` with FP64 input `1e8` produced
zero in JAX eager execution and one in ONNX Runtime CPU under this legacy
policy. This difference illustrates why restoring loadable, internally
consistent types is separate from preserving every source arithmetic step.

Finally, generated `_f64` tests rewrite floating inputs and selected factory
and expected dtypes to FP64. See [t_generator.py](../../tests/t_generator.py).
They supply useful FP64 coverage but cannot substitute for tests that retain
intentional mixtures of dtypes and observable rounding boundaries. The
current system consequently has substantial precision coverage alongside
context-dependent rules; it has no single faithful mixed precision contract
established across the whole pipeline.
