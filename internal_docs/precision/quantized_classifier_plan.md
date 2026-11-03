# A trained quantized classifier through ordinary export

Status: implementation and acceptance contract for the bounded follow-up to merged precision
Steps 1 and 2 ([#287](https://github.com/enpasos/jax2onnx/pull/287),
[#288](https://github.com/enpasos/jax2onnx/pull/288)). The remaining work is the
practical demonstration in the [architecture concept](precision_architecture_concept.md#practical-demonstration).
The architecture and API migration are complete; #277 is closed without merge.
The goal below governs autonomous implementation on `precision-quantized-example`.
Checkpoint 1 found a numerical failure in the original dynamic-activation
recipe. On 3 November 2026, the user approved the one alternative described
below: fixed power-of-two activation scales prepared through public Qwix rules.
All numerical, accuracy, size and resource gates remain in force. Training,
calibration and full held-out deployment have passed checkpoints 1–3.
Final integration and merge readiness are established by independent review
and checks on the current head of [PR #289](https://github.com/enpasos/jax2onnx/pull/289).

## Goal and deliverables

Train one useful MNIST classifier, prepare real W8A8 weights with optional Qwix,
export its actual computation through ordinary `to_onnx(...)`, and demonstrate
accurate independent ONNX Runtime CPU inference with a smaller artifact.
Measure memory and latency without promising improvements in either.

Prefer two simple scripts for preparation/export and standalone deployment,
with a small example-local model helper only where sharing requires it.
Add a user-guide recipe, optional-environment pins, focused tests and measured
acceptance results with reproducible commands. Do not introduce a public API,
special converter path, general model framework or reporting infrastructure.

## Fixed model, data and preparation protocol

- 784 inputs → Linen Dense(128) → ReLU → Dense(10), both layers without bias.
  Inputs, parameters, scales and final logits use FP32. Convert original uint8
  pixels to `float32 / 255`, flatten to 784 features and preserve pixel order.
- Use seed 0 only. JAX parameter construction starts from its seed-0 key;
  split before separate consumers. Use one NumPy `Generator(PCG64(0))`:
  its first permutation of the canonical 60,000 training indices assigns the
  first 55,000 to training and last 5,000 to development. Continue that generator
  for each epoch's training shuffle; record the split indices and their hash.
- Train exactly 20 epochs with Adam at `1e-3`, mean softmax cross-entropy,
  batch size 128 including the remainder. Record the pinned Optax Adam defaults.
  No alternate seeds, hyperparameter sweeps, fallback candidates,
  development-based checkpoint selection or early stopping. Use the final
  weights. Development data supplies the approved activation calibration and
  diagnoses the fixed candidate; it is not training.
- Keep the official 10,000 test images untouched until weights and recipe are
  frozen and hashed. Evaluate every image at inference batch size 1, without
  resizing, dropping or padding examples. Never tune against test results.
- Establish authoritative MNIST provenance, redistribution/licence terms and
  checksums before reuse; start with the [canonical dataset description](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data),
  without adding TensorFlow. Verify actual distribution URLs and checksums at
  checkpoint 1. Record source locations, preprocessing, splits,
  training configuration, software versions and checkpoint/recipe hashes.
  Keep datasets and checkpoints outside git, including small checkpoints.
- Reuse Qwix 0.1.8 at commit
  `9ee1079245682ed82e5da00ce855e4daf77a086b`: `dot_general`, INT8 weights and
  activations, weight `absmax`, layer-specific fixed symmetric activation ranges,
  `act_static_scale=False`,
  `tile_size=None`, `PtqProvider(..., disable_jit=True)`, no zero points.
  Prepare INT8 weights once; accumulate in INT32 and scale/return in FP32.
  As approved on 3 November, choose activation scales as powers of two from
  the known input range and the 5,000 development images. Set each Qwix fixed
  symmetric range to `127.5 * scale`. Calibration is model preparation before
  export; freeze its results before official test access. The original dynamic
  recipe remains a recorded failed candidate.
- Pin the optional CPU preparation environment to Python 3.12, JAX/jaxlib
  0.10.2, Flax 0.12.8, NumPy 2.4.6, Optax 0.2.8, ONNX 1.23.2 and ORT 1.31.1.
  Retain the immutable Qwix commit and existing `--no-deps` installation recipe;
  verify installed metadata and dependency consistency. No default dependency
  change or supported-cohort substitution is authorized.
- Export fresh serving callables capturing the actual prepared parameters,
  with typed FP32 inputs of shape `(1, 784)` and opset 23. Export before the
  first native trace of each serving callable to avoid cached Linen shortcuts.
  Then obtain official native eager, fresh-JIT and warmed-JIT references.
  Do not rewrite the quantized computation as a demonstration implementation.

## Measurable acceptance

These criteria are fixed before candidate experiments; a failure is investigated,
not repaired by relaxing a bound, changing the recipe or selecting another seed.

1. **Useful classifier:** FP32 native and FP32 ORT each achieve at least 97%
   top-1 accuracy on all 10,000 official test images. Quantized JAX and ORT each
   lose at most 0.5 percentage points, or 50 additional errors, relative to their
   respective FP32 reference. Report error counts, accuracies and differences.
2. **Quantized export fidelity:** eager, fresh-JIT, warmed-JIT and ORT with both
   `ORT_DISABLE_ALL` and `ORT_ENABLE_ALL` agree at the existing W8A8
   `rtol=1e-6`, `atol=1e-6`, with identical class predictions, exact shapes and
   dtypes, and finite logits. Check the complete held-out set. Native-reference
   contradictions remain failures. Report FP32 native/ORT logit differences
   separately: long FP32 reductions do not inherit the quantized logit bound.
   Existing exact integer-accumulation witnesses retain their checks. Legitimate
   all-zero inputs and outputs are valid; no blanket nonzero-output gate.
3. **Real quantized storage:** both model contractions use `MatMulInteger` with
   INT8 operands and INT32 results. Match the stored INT8 kernel payloads exactly
   to Qwix's prepared weights; check FP32 scales/results and absence of zero
   points. No retained FP32 weight matrices or per-invocation weight quantization.
   Run ONNX full checking and shape inference as well as runtime validation.
4. **Smaller artifact:** total W8A8 artifact bytes, including external tensor
   data if any, are at most 40% of the matching FP32 artifact. Compare the same
   architecture, export format, inputs and metadata policy. Count files actually
   required for inference, not array sizes, checkpoint compression or estimates.
5. **Independent deployment:** run the complete held-out ORT evaluation in a
   clean fresh process/environment containing NumPy and ORT, without installed
   JAX, Flax, Qwix or jax2onnx and without checkout `PYTHONPATH`. Verify that these
   packages are absent. The standalone script must not import package helpers
   that load the training stack. Record runtime pins and provider/settings.
6. **Honest measurements:** evaluate FP32 and W8A8 in separate serial fresh
   processes. Record import baseline RSS, session/inference RSS and peak RSS
   with units, session creation, first inference and warmed latency, hardware,
   versions, threads, repetitions and graph optimization level. Exclude training,
   export and JAX from runtime memory/timing. Only artifact size must improve;
   report actual memory and speed even when quantization makes them worse.

## Four delivery checkpoints

1. **Qualify the fixed candidate.** Confirm data provenance and optional pins;
   reuse the existing tiny Qwix witness, qualify actual example graph/storage
   and fresh serving traces on development inputs, then perform the one fixed
   training run. Freeze final weights, splits and recipe hashes before test-set
   results. Astra independently reviews feasibility, criteria and ownership.
   Reassess the effort estimate here. If the candidate fails or needs broader
   work, document the cause and escalate before expanding scope.
2. **Establish fidelity and task quality.** Export FP32 and Qwix computations
   through ordinary APIs. Validate the stored kernels, integer products,
   full ONNX checks, all required native/ORT references and all held-out quality
   gates against the frozen artifacts. Keep export fidelity and task quality
   separate. A failed gate stays failed; no test-guided retraining.
3. **Demonstrate deployment and measured benefit.** Run the independent minimal
   environment on all 10,000 examples. Measure real artifact bytes, separate
   process memory and latency with reproducible commands. Explain deployment
   dependencies and hardware-specific observations without unsupported claims.
4. **Integrate, review and qualify the final head.** Finish the user guide,
   environment recipe, targeted regressions and release note. Preserve all
   existing capability and cheap witnesses. Complete independent Astra review,
   required checks, commits and the one draft PR. Inspect exact-head CI results
   and job durations; completion requires all gates as well as passing CI.

## Validation and resource boundaries

Keep `tests/extra_tests/test_qwix_quantized_export.py` and its criteria intact.
Add a small offline fixture only for a demonstrated regression gap; tests should
exercise actual example-script functions without training or downloads. Do not
commit datasets, checkpoints, ONNX files, ONNX data or large fixtures. Reuse
synthetic inputs/weights where they establish the particular regression.

Full held-out evaluation and measurement have an explicit separate invocation
or integration job outside STANDARD. A manually dispatched job may prepare
verified external data; PR jobs must not acquire a new compulsory download or
training phase. Reuse the existing optional Qwix integration job for cheap work.
Document preparation and integration commands so acceptance is repeatable.

Retain main-equivalent STANDARD cases, shapes, tolerances and all mandatory
Python 3.11–3.14 cohorts, actual minimum-dependency checks and applicable support
jobs. Run focused regressions, pinned hooks, typing, strict documentation and
complete required suites at reviewed milestones and final acceptance. Every
Linux STANDARD job must finish within 20 minutes from setup through cleanup,
excluding queue time only; record aggregate cost without concealing growth by
splitting jobs. Do not repeat full runs without changes or unresolved concerns.

Use official JAX/jaxlib only, one local native worker and at most 4 GiB RAM.
Preserve user changes and machine stability. Run training, export and runtime
processes serially. Hosted parallelism does not authorize local concurrency.
Small source-faithful repairs belong to existing converter/plugin owners and
require independent review plus focused checks on both supported JAX cohorts.
Beyond the approved fixed-range calibration, a different model/recipe, a general exporter path,
nonlocal repair, weakened checks or resource/runtime expansion requires escalation.
Do not reopen the frozen precision migration or import #277 infrastructure;
only small independently relevant witnesses may be reused.

Provisional estimate: **3–5 engineering days**, including qualification and
review; reassess after checkpoint 1. This is an estimate, not a token/spend cap.
Report failed, incomplete and unverified work honestly; do not declare completion
because the estimate has elapsed or CI alone is green.

## Checkpoint 1 finding and approved change

The existing small Qwix witness passes in the pinned environment. The actual
784–128–10 model, initialized with seed 0, fails its first development-image
comparison at the unchanged `1e-6` tolerances. This is development image 4243
from the original training archive, not an official test image. The first
output comparison differs by up to `0.00195623` in both ORT optimization modes.

Early preparation guards had 13 passing offline tests covering typed preprocessing,
checksum-before-array-access and frozen checkpoint/recipe/split integrity. A
fresh isolated runtime process also verifies that JAX, jaxlib, Flax, Qwix and
jax2onnx are neither installed nor importable. These checks qualify the
preparation guards and environment isolation only; at that stage classifier
fidelity, accuracy, artifact benefit and independent inference were incomplete.

The first mismatch occurs before the first integer matrix product. Both
implementations use the same FP32 activation scale. For one pixel, division
gives `1.4999998807907104` in JAX and `1.5` in ORT. Nearest-even rounding then
produces INT8 values 1 and 2. A separate `Div → Round → INT8` witness reproduces
this on official JAX/jaxlib 0.10.2 and 0.12.0.1, in eager, fresh-JIT and warmed-JIT
references. ORT 1.31.1 agrees between optimization modes. The emitted FP32
`Div` and nearest-even `Round` match the requested operations; the discrepancy
does not justify changing generic division or biasing rounding in the exporter.
Explicit strict fast-math and platform-dependent-math flags did not remove it
in the tested JAX 0.10.2 process. At that point no training or official test
evaluation had been performed; this original result remains failed.

On 3 November 2026 the user approved Astra's one model-preparation alternative:
layer-specific fixed symmetric activation ranges through Qwix's existing public
`QuantizationRule`. Choose each scale as the smallest power of two covering its
calibration maximum under Qwix's 127.5 bound, and set the fixed range to
`127.5 * scale`. Use the known input range `[0, 1]` for the first layer (scale
`1/64`) and the FP32 model's post-ReLU hidden maximum over all 5,000 development
examples for the second. Calibration makes no guarantee against clipping of
unseen or quantized activations; the full quality gates still decide acceptance.
This introduces explicit
calibration while retaining `/255`, the model, training protocol, INT8 weights,
INT32 accumulation, ordinary export and every acceptance threshold.

First verify the intended scale representation and the original
development probe on an untrained model against every native reference and
both ORT settings. Only a passing probe permits the single fixed training run,
followed by one deterministic calibration and frozen weights/scales before
official test access. There is no range search or further fallback candidate.
The approval amends the recipe requirement of the active goal; it does not
establish a successful numerical result. The original activation-absmax failure
and its diagnostics remain failed historical evidence.

The approved untrained candidate passes the first 16 development images plus
the zero image against eager, fresh-JIT and warmed-JIT JAX and both ORT
optimization settings. Both actual activation denominators are exactly `1/64`.
The development calibration uses all 5,000 prescribed images. Its untrained
artifacts are 407,044 bytes (FP32) and 106,082 bytes (W8A8), a ratio of 0.2606.
These results qualify proceeding to training, not final task accuracy or the
trained-model artifact claim. Astra's independent review accepts this gate.

The single prescribed training run subsequently completed all 20 epochs. Its
frozen checkpoint is
`8d0802292268f6f051714720519666dffdba6c5be84788d98dc46be1977d559d`.
The final 5,000-image calibration chose hidden scale `1/8` from maximum
`15.180764198303223`. Both artifacts and scales were hashed before test access.
On all 10,000 test images, FP32 has 9,796 correct predictions and W8A8 has
9,793, consistently across the required native and independent ORT modes.
All W8A8 logits match exactly. The trained artifacts retain the byte counts
above. On the measured CPU, W8A8 is about eight times slower with similar memory
use; the demonstrated advantage is file size. The user guide records the
measurement protocol and limitations. Astra accepts checkpoints 1–3 and the
implementation review, subject to the required integration checks and final
PR qualification. Passing CI alone does not replace the scientific gates.

Checkpoint 1 effort reassessment: the approved variant required no exporter
changes, and training plus qualification fits comfortably within the existing
local resource bounds. The initial 3–5-day estimate no longer represents the
remaining work; that work is integration, documentation and the final CI run.
There is no justification for expanding the architecture or the model scope.

## Active implementation goal

> Implement the bounded trained W8A8 MNIST classifier demonstration described in
> `internal_docs/precision/quantized_classifier_plan.md` in
> `/home/enpasos/projects/jax2onnx`, starting from merged
> main on `precision-quantized-example`. Treat its fixed protocol, six measurable
> acceptance gates, four checkpoints and boundaries as binding. The precision
> architecture and API migration in #287/#288 are complete; #277 is closed
> without merge and its broader implementation remains frozen historical work.
>
> Use GPT-6.1 Sol at Ultra for implementation and tests, and GPT-6 Astra at Ultra
> for planning and independent review at checkpoints, final acceptance and
> material uncertainty. Train only the specified seed-0, 20-epoch, no-bias
> 784–128–10 Linen classifier; use the pinned Qwix recipe with the user-approved
> fixed power-of-two activation scales and development-only calibration,
> untouched test set, frozen hashes, numerical/quality/size gates and ordinary
> export path. Complete independent NumPy/ORT-only deployment and honest measured
> artifact, memory and latency evidence. Keep datasets, checkpoints and generated
> ONNX artifacts out of git; do not build a new framework or public API.
>
> Work autonomously through the four reviewed checkpoints. Preserve supported
> Python 3.11–3.14 cohorts, minimum/support checks, existing coverage/tolerances
> and the 20-minute complete Linux STANDARD-job limit. Keep expensive held-out
> work separate. Use official JAX/jaxlib, one local native worker and 4 GiB RAM.
> Preserve user changes. Reassess the provisional 3–5-day estimate at checkpoint 1;
> document and escalate failed candidates or required scope expansion rather
> than silently changing criteria, model, calibration or architecture.
>
> Ordinary commits, non-force pushes to `precision-quantized-example`, and
> opening/updating its one draft implementation PR are authorized. After every
> push inspect actual checks on that exact head; repair regressions before
> further feature pushes. Necessary CI repair pushes are authorized. Do not
> merge, force-push, release, post unsolicited comments, close issue #276 or make
> other GitHub writes. The final validated draft PR is for the user to merge.
>
> Finish only when all four checkpoints and acceptance gates are complete,
> documented with metrics and reproducible commands, independently reviewed by
> Astra, and committed, with all applicable final exact-head checks at SUCCESS
> and the STANDARD runtime bound verified. Passing CI alone is insufficient.
> Report limitations and failures honestly and keep the goal active while
> required work remains. No token budget is requested.
