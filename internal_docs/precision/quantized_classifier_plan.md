# A trained quantized classifier through ordinary export

Status: implementation plan for the bounded follow-up to merged precision
Steps 1 and 2 ([#287](https://github.com/enpasos/jax2onnx/pull/287),
[#288](https://github.com/enpasos/jax2onnx/pull/288)). The remaining work is the
practical demonstration in the [architecture concept](precision_architecture_concept.md#practical-demonstration).
The architecture and API migration are complete; #277 is closed without merge.
The goal below governs autonomous implementation on `precision-quantized-example`.

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
  weights. Development data diagnoses the fixed candidate; it is not training.
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
  activations, weight/activation `absmax`, `act_static_scale=False`,
  `tile_size=None`, `PtqProvider(..., disable_jit=True)`, no zero points.
  Prepare INT8 weights once; accumulate in INT32 and scale/return in FP32.
  This dynamic activation recipe has no dataset-fitted calibration phase.
  Do not invent one or claim that development evaluation calibrates scales.
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
Need for a different model/recipe, static calibration, a general exporter path,
nonlocal repair, weakened checks or resource/runtime expansion requires escalation.
Do not reopen the frozen precision migration or import #277 infrastructure;
only small independently relevant witnesses may be reused.

Provisional estimate: **3–5 engineering days**, including qualification and
review; reassess after checkpoint 1. This is an estimate, not a token/spend cap.
Report failed, incomplete and unverified work honestly; do not declare completion
because the estimate has elapsed or CI alone is green.

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
> 784–128–10 Linen classifier; preserve the pinned dynamic-activation Qwix recipe,
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
