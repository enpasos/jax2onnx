# A simpler precision architecture

Status: first implementation checkpoint in progress, 2 November 2026, on
`precision-contract`. The complete precision transition is not implemented.

The proposed replacement for [PR #277](https://github.com/enpasos/jax2onnx/pull/277)
is one consistent precision rule in the existing exporter, demonstrated by a
small, quantized JAX model that can run independently through ONNX Runtime.
The aim is a simpler user interface and a codebase that remains maintainable.
A successful demonstration alone would not establish the architectural change.

PR #277 and its implementation programme are frozen. Its branch, findings and
failed or incomplete acceptance results remain historical evidence. This
concept starts from `main`; it neither imports that implementation wholesale
nor declares its outstanding requirements satisfied. It develops the direction
discussed in [issue #276](https://github.com/enpasos/jax2onnx/issues/276).

## Goal and user experience

The JAX model defines how each calculation chooses its computation and
accumulation precision, including deliberate conversions and output dtypes.
The export call supplies the inputs and their tracing properties and selects
the ONNX representation. Export preserves the resulting computation
within the capabilities of the selected ONNX version and execution target.

Users configure their model in JAX, Flax or another supported framework, supply
typed inputs, and call the ordinary `to_onnx(...)`. They do not choose an
exporter-wide arithmetic dtype, a SOURCE/LEGACY mode or an operator profile.
Quantization, when wanted, prepares the JAX model before export. Export does not
recalibrate weights or silently choose a different quantization scheme.

For example, a model may use INT8 matrix operands, accumulate in INT32, then
scale the result in FP32. Another model may deliberately round an FP64 value
to FP32 and continue computing in FP32. Both follow the same export rule.
The first example must not become a special Qwix path.

Input declarations retain their types and precision-relevant properties such
as JAX `weak_type`. Shape-only inputs have a documented FP32 default; that
default does not govern the rest of the model. JAX x64 settings belong to model
preparation and tracing. An explicit FP64 input incompatible with those
settings receives a clear error. Export does not silently change the caller's
configuration or lose a local precision setting at a function boundary.

Preserving the computation includes its required conversions and accumulation
behaviour. It does not promise identical floating-point bits across arbitrary
kernels and hardware. Numerical criteria must match the operation and the
execution target. A known violation cannot be hidden by a wider final tolerance.

## Architecture and hard boundaries

Keep the existing [IR converter and plugin architecture](../../docs/developer_guide/architecture.md).
Replace competing precision policies inside it rather than add a second
converter or a separate precision subsystem.

| Existing responsibility | Required rule |
| --- | --- |
| Input preparation and tracing | Preserve declared inputs, relevant JAX configuration and precision information needed by nested traces. |
| Values and constants | Keep one canonical typed binding for each traced value. Constant data and its declared type agree. Looking up a value never changes its type. |
| Operator plugins | Implement source operand, result, conversion and accumulation requirements. Tensor output dtype alone does not prove accumulation precision. |
| Functions and control flow | Preserve the same bindings, signatures and precision rules through functions, branches and loop state. |
| Optimization and serialization | Preserve observable rounding and conversions. Do not globally promote or narrow already typed constants and values. |

A lowering may create explicit intermediate conversions for its operation.
Those conversions must respect the source contract and must not replace the
canonical value seen by another consumer. Unknown types must receive a specific
unsupported-representation error, never silently become FP32.

Apply these rules across existing supported functionality. A missing
implementation is reported as missing; it is not described as an ONNX
impossibility. Supported operations must not alternate between old and new
arithmetic policies. The demonstration does not justify a callable whitelist
or a private export path that ordinary models cannot use.

Use existing diagnostics and validation facilities. Distinguish ONNX validity,
correct lowering, runtime availability and numerical acceptance. A schema-valid
model is not a deployment guarantee. Unknown runtime behaviour stays unverified;
a known failure stays failed. Operator-specific limitations belong with the
operator and the existing dependency documentation.

Only official supported JAX/jaxlib builds are permitted. Compiler patches,
binary overlays and custom JAX, XLA or LLVM builds require separate explicit
permission and are outside this proposal. Respect existing resource limits.

## Practical demonstration

The candidate is a trained MNIST digit classifier with a small
Dense–ReLU–Dense Flax model. Qwix prepares one specified post-training W8A8
configuration: 8-bit weights and activations, INT32 accumulation, and FP32
scales and results. The exact recipe and compatible versions must be established
in the feasibility step; general Qwix support is not implied.

The supplied callable captures prepared weights and accepts typed FP32 image
inputs. Ordinary `to_onnx(...)` produces an artifact that ONNX Runtime CPU can
load and execute in an environment containing neither JAX nor Qwix. The
exporter receives the actual prepared computation, without rewriting it into
a special demonstration implementation.

The intended benefit is portable quantized inference with a smaller artifact
than the corresponding FP32 model. Verify that weights are actually stored in
the quantized representation rather than keeping FP32 weights and quantizing
them on each invocation. Measure actual artifact bytes and runtime memory.
Report latency, but do not promise that a small INT8 model runs faster.

Compare two different properties separately:

- **Export fidelity:** ONNX agrees with the quantized JAX computation under
  predefined numerical criteria, including integer accumulation and scaling.
- **Task quality:** the quantized classifier retains an agreed level of
  accuracy relative to the useful FP32 classifier on held-out data.

Training, calibration and evaluation data are separate. Fix the model,
quantization recipe, data split and numerical and accuracy thresholds before
implementation acceptance. If a threshold fails, investigate the cause rather
than relax it to obtain a green result.

Training and dataset downloads are preparation tasks, not routine CI work.
Use reproducible weights and a small fixed offline regression fixture, with
provenance, licences and checksums. Evaluate task quality on the agreed held-out
set at demonstration acceptance; the small fixture does not replace it.
Large datasets and checkpoints stay outside git. Generated `.onnx` and
`.onnx.data` files must not be committed.

Qwix belongs in a pinned optional example environment. It must not become a
mandatory dependency, load during ordinary converter imports, or silently
change supported dependency cohorts. Whether the proposed example can fit
those constraints is part of feasibility.

## Public transition and scope

The current `enable_double_precision` option changes tracing, defaults,
constants and arithmetic. Replacing that policy is a behaviour change, not
merely removing an argument. Enabling JAX x64 cannot replace every previous
widening effect, especially inside unchanged third-party sources.

Before changing the public API, list affected existing workflows and their
concrete migration paths. Use model configuration, typed parameters and inputs
where they express the intended computation. Do not promise automatic rewriting
of fixed casts. An existing supported workflow without an acceptable migration
path is an unresolved decision, not permission to remove it silently.

The intended end state is one rule at an explicit, announced API version
boundary, with migration examples and an actionable error for the obsolete
argument. The released API stays unchanged until that transition is qualified.
Any temporary compatibility code needs a removal point before the new contract
is released; permanent dual arithmetic modes are not the solution.

This project does not include redesigning `input_params`, general stateful-model
adapters, automatic precision rewriting of external model sources, a new
artifact-certification framework, or support for every quantization format.
FP8, FP4, INT4 and additional quantized deployment targets wait for concrete
requirements. Existing non-precision features and integrations retain their
regression coverage; these scope limits do not authorize breaking them.

## Acceptance criteria

All of the following are needed to complete a replacement implementation:

1. **One public rule.** The ordinary export API follows the model's precision
   choices throughout its supported computation. There is no hidden per-model
   or per-operator switch back to legacy arithmetic. Migration documentation,
   public API tests and release notes describe the same behaviour.
2. **General invariants.** Focused regressions cover source casts and rounding,
   weak inputs, exact constants, shared consumers, mixed outputs, functions,
   branches and loops. Include the known FP32-rounding counterexample from
   #277. Tests inspect relevant intermediates and their consumers where final
   outputs alone would miss a violation.
3. **Operation correctness.** INT8 products retain INT32 accumulation, including
   inputs that distinguish it from an inappropriate floating implementation.
   Check shapes, dtypes and discrete values exactly, and use operation-specific
   criteria for floating results. Check zero signs, NaN locations and exceptional
   values where the operation's contract requires them; all-zero results are
   not universally invalid.
4. **A complete useful example.** The trained classifier exports through the
   normal API, executes on the pinned CPU runtime without the training stack,
   meets quantized-JAX fidelity and task-quality criteria, and demonstrates
   the claimed artifact-size benefit. Report measured performance separately.
5. **Existing capability.** Affected operators and complete model workflows
   pass their regressions. Run the full suite, pinned hooks and documentation
   checks at implementation milestones and before release across the supported
   Python/JAX cohorts. Core and operator regressions cover those cohorts;
   the optional Qwix demonstration initially qualifies its one pinned supported
   environment. Do not drop supported cases or shrink model inputs to obtain a
   pass. Any intentional API change is reviewed explicitly.
6. **Practical CI.** Preserve main-equivalent routine coverage. Every Linux
   STANDARD job finishes within 20 minutes from job start through final cleanup,
   excluding only queue time. Added expensive validation has a separate,
   explicit integration job and runs at relevant milestones and final acceptance.
   Report aggregate work as well; splitting jobs must not conceal growing cost.
7. **Maintainability.** Independent review confirms that implicit promotion
   paths and competing policy branches have been removed, additions have clear
   owners in the existing pipeline, and ordinary converter use has no Qwix
   dependency. A contributor can explain the precision rule without a second
   parallel implementation or a large catalogue of special cases.

For claimed numerical acceptance, retain matching official JAX eager,
fresh-JIT and warmed-JIT references and validate the runtime settings being
claimed. Matching means meeting the predefined operation-specific criteria,
including exact properties where required, rather than universal bitwise
identity. A contradiction between required references remains a failure; it is
not repaired by choosing the convenient result. Historical failures in #277
remain recorded as such, and are not retroactively passed by this narrower
proposal. Green diagnostic tests do not establish numerical correctness.

## First implementation checkpoint

The first change preserves JAX `weak_type` through ordinary input preparation,
named functions, captures and function-cache keys. This is one shared tracing
property, not a new precision mode. The accompanying operator change represents
8-bit integer `dot_general` operands with INT32 accumulation using
`MatMulInteger`. Existing pure value lookup already preserves canonical
bindings and does not need another implementation.

On unchanged `main` (`ff012402`), official JAX 0.10.2 CPU returns FP16 `2048`
for a weak FP32 `2048.0` plus a typed FP16 `1`. All three native roles agree;
both ORT optimization modes instead return FP32 `2049`. The strong FP32
counterpart correctly returns FP32 `2049`. An INT8 contraction of 1,041
`127 * 127` products returns INT32 `16,790,289` in all native roles, while export
emits an invalid INT8 `Gemm`. ONNX Runtime 1.31.0 CPU executes direct
`MatMulInteger` models for all four signed/unsigned operand combinations.

Acceptance for these discrete and FP16 witnesses is exact, including output
shape and dtype. Integer accumulation is inspected before downstream FP32
scaling. The optional Qwix FP32 example uses fixed `rtol=1e-6`, `atol=1e-6`
for matching native references and runtime results, with output structure,
shape and dtype checked separately. Each claimed result needs eager, fresh-JIT
and warmed-JIT references and both claimed ORT optimization settings.

Reuse the small weak-input and function-specialization corrections and
`2048.0` counterexamples from `precision-migration` at `3aaed9ff`, adapting
them to the ordinary pipeline. Do not copy its SOURCE profiles, tracing
controllers or deployment-evidence infrastructure. Its integer matrix path
rejects integer operands, so it supplies no INT8 implementation to reuse.
Exact initializer-payload comparisons are a useful later candidate, not part
of this checkpoint without a separate concrete need.

The unchanged optional Qwix recipe now reaches two integer products through
ordinary export. Its first attempted export exposed the Dense shortcut's
flat-array assumption and floating-to-8-bit overflow differences. Both repairs
are generic: structured Dense parameters use the existing original-call
fallback, and FP16/BF16/FP32/FP64 conversions sanitize NaN and saturate before
casting. No Qwix recognition is present in the converter. Repeated empty
contractions also exposed uninitialized runtime output; static zeros and local
padding for potentially empty symbolic contractions avoid that kernel path.

This checkpoint preserves the current explicit precision-flag behaviour.
The known double-export example still returns FP64 `1` where the model's FP32
arithmetic returns `0`; removing that global promotion and preserving local
x64 scopes remain future transition work. The next compatibility decisions are:

| Existing workflow | Decision required before the public transition |
| --- | --- |
| Omitted precision flag | Preserve the caller's x64 configuration instead of implicitly disabling it. |
| Shape-only inputs with double export | Replace the implicit FP64 default with explicit typed FP64 inputs; the proposed shape-only default stays FP32. |
| Typed FP64 input while JAX x64 is disabled | Reject the incompatible input before tracing instead of silently narrowing it or changing the caller's configuration. |
| Explicit old precision flag | Choose and announce the API version boundary and migration error, with tested equivalents for the claimed workflows. |
| Fixed FP32 casts in third-party sources | Keep their meaning. A workflow relying on global widening needs a real framework configuration or an explicit unresolved compatibility decision. |
| A local x64 scope around a named function | Preserve its effective tracing context and cache identity before removing the old policy. |

The bounded follow-up is to resolve those cases with concrete examples, then
remove the shared tracing/default and constant/payload promotion rules together
with their affected plugin consumers. Do not add a parallel compatibility
engine. The trained-classifier accuracy and size demonstration follows that
decision; this small fixed-weight integration is not its substitute.

The focused regressions pass on the locked JAX 0.10.2 and 0.11.2 cohorts and
with minimum ONNX Runtime 1.24.1. Pinned hooks and the strict documentation
build pass. Independent review found no blocking production defect for this
checkpoint; complete-suite and hosted timing acceptance remain open.

The local full-suite attempt was stopped at the existing 4 GiB process-memory
limit after 82 seconds. A separate unchanged-main control hit the same limit
at the large GPT test after 67 seconds. Both runs are incomplete, not passes;
neither establishes a memory regression or a complete comparison. The existing
hosted full-suite jobs must supply the missing results without reducing model
inputs or coverage. Their exact new commit and job durations must be checked.

Local focused runs take seconds; they do not forecast hosted setup or
full-suite cost. A reliable total estimate for the public transition depends
on the unresolved compatibility decisions above. The initial one-to-three-day
estimate applied only to this first checkpoint and is not a completion
forecast for the migration.

## How to proceed and when to stop

**First establish feasibility.** The bounded first checkpoint is authorized;
the feasibility results above establish its change boundary. They do not
authorize an open-ended migration. Agree a separate effort budget before the
public transition and record time and compute cost.

Use unchanged `main` as the baseline. Confirm the actual Qwix recipe and traced
operations, reproduce the export gap, and identify the precision-policy sites
that must change for the general contract. The result should be a concrete
change boundary, a compatibility decision list, proposed fixed numerical
criteria and a remaining-effort estimate. Reuse #277's findings and small
counterexamples before repeating investigations.

**Then make a go/no-go decision.** Continue only with a bounded implementation
plan and an agreed effort budget. Stop for reassessment if the common contract
requires another converter, model-specific tracing patches, compiler changes,
or an expanding compatibility programme. If the useful example already works,
document that result; it does not by itself justify a redesign.

**Implement through the existing pipeline.** Work from `main` in small,
reviewable changes. Separate behaviour-preserving cleanup from the explicit
contract transition. Keep incomplete work unreleased until the general
invariants, compatibility decision and complete demonstration are accepted.
Borrow code from #277 only after identifying its necessary responsibility and
validating it in the smaller design.

**Finish with an independent review of the result.** Review the public UX,
the removed as well as added complexity, numerical evidence and CI cost.
The project is complete only when both the general architecture and the
practical benefit pass acceptance. A successful toy graph or a cleaner diagram
is not sufficient.

## Lessons and references

PR #277 established useful counterexamples but coupled precision preservation
to a much broader migration and evidence programme. Preserve its findings,
separate exporter defects from upstream/runtime behaviour, and keep decisions
and evidence proportionate to the release claim. Maintain this one concept
document and ordinary focused tests rather than duplicate current-state,
handoff and execution histories.

- [#275: explicit FP32 values and double export](https://github.com/enpasos/jax2onnx/pull/275)
  shows the conflict between source rounding and global promotion.
- [#278: stable elementary functions](https://github.com/enpasos/jax2onnx/pull/278)
  and [#279: attention and low-precision types](https://github.com/enpasos/jax2onnx/pull/279)
  show the value of operation-specific fixes within the existing architecture.
- [Qwix](https://github.com/google/qwix) documents W8A8 model preparation;
  [its accumulator selection](https://github.com/google/qwix/blob/main/qwix/_src/core/qarray.py)
  separates integer accumulation from the floating result.
- [ONNX MatMulInteger](https://onnx.ai/onnx/operators/onnx__MatMulInteger.html)
  provides a candidate representation for 8-bit operands and INT32 output.
  That schema does not establish complete Qwix export or runtime performance.
- [JAX numerical differences under JIT](https://docs.jax.dev/en/latest/faq.html#jit-changes-the-exact-numerics-of-outputs)
  explain why reference disagreement must be distinguished from an export bug.
