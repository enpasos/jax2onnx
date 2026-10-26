# Precision migration handoff — 2026-10-26

This continues diagnostic commit `3d023218`, after native-file diagnostic
`b8888c40` and annotation repair `87b2b3c6`. The user has explicitly requested
merging current main `48054bb3` into this branch; its reviewed integration and
finite qualification are recorded below. The observer-off control is pushed,
but precise same-call unprofiled word evidence and the full CI gate remain open.
Continue on branch
`precision-migration` in `/home/enpasos/projects/jax2onnx`.

## Objective and immediate priority

Complete the existing precision target and migration plan, including phases B–F,
existing functionality, and fixed or third-party JAX sources. The final public
behavior must have one consistent source-preserving precision contract, with
evidenced technical limits and no permanent legacy/SOURCE mixture or feature
loss. Keep public defaults LEGACY until the coordinated transition is qualified.

First restore **all eight checks on the exact latest head of PR #277**, including
the full Python 3.11–3.14 jobs. Preserve original model sizes, inputs, cases and
numerical/bitwise criteria. Keep CI passing after subsequent pushes. A green
earlier head or a local subset does not satisfy this gate.

Use `gpt-6.1-sol` with ultra reasoning for implementation/tests and `gpt-6-astra`
with ultra reasoning for architecture and independent reviews. Ordinary commits
and non-force pushes to the existing branch/PR are authorized when continuing
the implementation. Do not merge PR #277 into main, mark it ready, enable
automerge, force-push, release, post unsolicited comments, or cancel/rerun CI.

## What this checkpoint contains

- Preserve the original class compiler context when preparing pinned NNX
  methods on Python 3.14. Code-contract comparison, including method flags,
  remains strict. Generic callable compilation retains its existing default.
- Add compiler/provenance regression tests and retain all original model tests.
- Retain cold annotations through `AnnotationState.attach`, derive capture ports
  from pinned parameters/native defaults, and avoid redundant class signature
  reads. Prepared VALUE reads retain the existing shared-cache/stale guards.
- Add passive, exact-node FP64-to-FP16 diagnostics to the existing batch runner,
  with outcome/profile noninterference tests. These diagnostics do not change
  conversion, inputs, references, assertions, test selection or acceptance.

`main` was already merged **into** this PR branch as `ee45c6dc`; it was not a
merge of the PR into main. At the earlier `70ea34ad` checkpoint, `ee45c6dc` was
the last verified remote PR head. Later observations are recorded below;
refresh GitHub state before relying on any CI status.

## Verified local repair qualification

The locked framework cohorts are Python **3.14.9 / JAX 0.12.0.1 / Flax 0.12.10**
and Python **3.12.9 / JAX 0.10.2 / Flax 0.12.8**. Each complete preparation pair
passed **125 tests**, with the original Transformer cases first in fresh workers,
at source-files `29dc0cefd33effbb6bad07ba38988844c4f53109199bc4ebeaba65a7ad3c108c`.
The final complete 17-owner affected selection passed **410 tests on each cohort**
at source-files
`a16fd89ea3fecde320fa5b7e4645734221c5e52d4f641fb51e119e25a48ada01`.
Both pairs have zero failures/errors/skips, complete JUnit, held source/Git/index
and environment metadata, and empty owned-process cleanup. The 410-case pair is
the final code qualification; the earlier 125-case source epoch stays distinct.

The unchanged original FP64-to-FP16 tie node passed through the actual batch
runner on each cohort. Both v2 diagnostics are **ORIGINAL_CALL_WORDS_OBSERVED**,
with complete artifact/runtime/comparison words and provenance, restored
profile/settings, and held source/code identity. SciPy is explicitly
**NOT_LOADED**, rather than imported for observation. This repairs diagnostic
coverage; it changes no numerical conversion and does not explain the hosted
bitwise disagreement.

At source-files `08e434e7deee1680e5fe3d97ef72b01770fc1bdeb4f207f6cf7e18c325f5f0d8`,
all seven actual pinned hooks pass over all files, including modern-stack mypy.
Older-stack package mypy passes **579 files**; both RNG reports contain the
same **17 entries**. Strict MkDocs also passes. These runs retain clean cleanup,
source/Git/index and environment-metadata fences under the same resource caps.
The final documentation-only closeout and normal commit hook retain their own
command receipts; none of these results establishes full-suite CI success.

## Actual hosted FP64-to-FP16 failure

Repair `87b2b3c6` was stored with all seven normal commit hooks passing and a
passing strict documentation build on the committed tree, then pushed without
force. PR #277 and [CI run #536](https://github.com/enpasos/jax2onnx/actions/runs/40377107687)
both identify that exact head; its observed checkout/event merge is
`d7aa9052be9160beacd626036b532ef6a90ebafe`. At **18:50 UTC**, four support jobs
were **SUCCESS**, Python 3.11 was **FAILURE**, and the other three full jobs
were still running. No later head inherits those results.

The unchanged `ties-float16-float64` case fails bitwise acceptance on hosted
Python 3.11.17/JAX 0.10.2/ORT 1.31.0. Its complete original-call diagnostic has
no errors and all five fences held. For input word `0x3ff0020000000001`
(`1 + 2^-11 + 2^-52`), eager and fresh-JIT JAX return FP16 `0x3c01`; ORT returns
`0x3c00`. The other five words agree. The first disabled/eager comparison fails;
the original test does not reach the other three comparisons.

The hosted model SHA is
`f48469e1bf98d5d1f7f2b1b59a1c1b1c172237eccb778aa68f7d802b9dfadd8e`,
identical to the successful local artifact. Complete FP64 input bytes and ORT
output bytes also match. A fresh local **Python 3.11.17** locked main/dev cohort
passes the unchanged actual node with all four comparisons: local eager/JIT and
ORT all return `0x3c00`. This rules out Python minor version alone; it does not
isolate CPU causality. Python build, native binaries and compiler settings have
their own evidence limits. Local installed native files have matching RECORD
hashes; hosted native files had not yet been measured at that checkpoint. The
later #537 measurements below supersede that evidence gap.

The [ORT Cast implementation](https://raw.githubusercontent.com/microsoft/onnxruntime/v1.31.0/onnxruntime/core/providers/cpu/tensor/cast_op.cc)
uses its [pinned Eigen half constructor](https://raw.githubusercontent.com/eigen-mirror/eigen/1d8b82b0740839c0de7f1242a3585e3390ff5f33/Eigen/src/Core/arch/Default/Half.h)
with an FP32 intermediate; [StableHLO convert](https://openxla.org/stablehlo/spec#convert)
leaves inexact float-to-float behavior unresolved. These source-level facts
do not themselves identify the loaded hosted machine instructions. A universal
single-rounding replacement would disagree with the current local native
reference. Production conversion, six inputs, bitwise criteria and the existing
success assertion therefore remain unchanged pending further contract review.

The diagnostic follow-up records installed native-file provenance passively
and replays retained artifact/input/reference bytes in a separate failure-only
CI step. It must distinguish supplementary replay from the original call,
record all four actual outcomes and preserve the original failed job. Its
normal hooks, documentation and commit closeout retain their own receipts.

At source-files `c9c7fe2bffee1833fef2591a690efc4ced423fc660949ebe07246071b527aafd`,
the two complete diagnostic owners pass **107 identical cases on each cohort**:
Python 3.14.9/JAX 0.12.0.1 and Python 3.11.17/JAX 0.10.2, with no failures,
errors or skips. Actual original-node batch integrations pass **1 case each**,
with complete word records and verified installed-file manifests: **58 files /
436,383,598 bytes** on the modern cohort and **57 files / 458,165,086 bytes** on
3.11. These are installed RECORD/file checks, not executed-kernel claims.

The new replay CLI also runs against the untouched archived hosted failure on
both local cohorts. Each reports four actual runtime-valid/numerically-failed
comparisons with `SampleAcceptanceError`, preserving original evidence and
separate replay context. A real passing original-node receipt returns
**NOT_APPLICABLE**. All guarded qualifications retain source/Git/environment
metadata and clean cleanup. Independent code review accepts this finite
diagnostic packet. It does not repair the numerical mismatch or qualify the
unattempted hosted comparisons; those still need actual CI evidence.

## Hosted follow-up and the observer-off control

Diagnostic commit `b8888c40` was stored with all seven normal hooks passing,
unchanged tested file contents and strict-docs qualification, then pushed
without force. [CI #537](https://github.com/enpasos/jax2onnx/actions/runs/40386799137)
is bound to that exact head; its checkout/event merge is `71b03a7869875ea537747d76cc36e25e98831fa9`.
Python 3.11 again fails the original tie case. The new observer supplies complete
original word evidence and **57 verified installed files / 458,165,086 bytes**;
all measured paths, sizes and SHA values match the local 3.11 manifest.
Its separate **hosted** replay completes all four optimization/reference
combinations: runtime execution VALIDATED, numerical acceptance FAILED and
`SampleAcceptanceError`, with held original-evidence/source/artifact fences.
The original call still contains only its first failed comparison.

A further causal confound remains: the earlier green `70ea34ad` run had failed
observer initialization, so its cast executed without this profiler. Its CPU
and native context were not captured. Current installed-file agreement and
target replays do not establish fresh unprofiled JAX behavior on the failing
worker. Preserve the original success assertion while investigating this.

The next controlled experiment uses one supplementary fresh child with the
exact current failed request, full ordered collection, assigned items, pytest
arguments and predecessor history. For #537 these are 15,119 collected items,
390 assigned items and 323 predecessors before the target; derive counts from
the actual new failure, rather than hardcoding them. Only an exact diagnostic
descriptor disables observer activation in that child. Canonical execution
keeps the observer and rejects supplementary receipts as completion evidence.

Capture startup environment **before pytest/full collection**, then collection
and target-entry/exit contexts separately. Full collection itself sets
`JAX_PLATFORMS=cpu` and `JAX_ENABLE_X64=True`; copying those target-time values
to startup changes the experiment. Missing startup evidence stays UNVERIFIED.
Use no source wrappers, new native oracle, CPU/math flags or retry loop. Keep
the original failed job and all original inputs/criteria unchanged.

Matching unprofiled bitwise failure excludes observer activation as necessary
for that reproduced failure. Target success establishes only an association
and requires further paired investigation. Earlier failure, changed collection
or predecessors, profile present, timeout, interruption or bad fences stay
UNVERIFIED. Preserve `-x` and any unattempted suffix. No assertion amendment or
numerical production rewrite is authorized by this diagnostic plan.

At source-files
`d761acce0350de0b1d0f9ef0cb1fb9a55876c3cf7450338005f31dad00c256c1`,
the three complete diagnostic owners pass **139 identical ordered unique cases
on each cohort**: Python 3.14.9/JAX 0.12.0.1 and Python 3.11.17/JAX 0.10.2.
There are no failures/errors/skips; source/Git/index and environment metadata
hold, and process cleanup is empty. Independent Astra review accepts the code
and both actual local ON/OFF integrations. The predecessor modern run remains
**138 passed / one failed**, with its original rejected receipt and a separate
owned-process cleanup record; it does not qualify the final source epoch.

The actual unchanged cast node also passes **once with the observer on and
once with it off on each cohort**. Each pair has exactly equal startup,
collection and target-before/after metadata, absent profiles at those boundaries,
JAX NOT_LOADED at startup and native JAX config OBSERVED at the later boundaries.
OFF produces no observer artifact, remains explicitly supplemental, and is
rejected by canonical completion validation. Its passing parent is
NOT_APPLICABLE to the failure-only coordinator, with no child launched. These
one-node passing-baseline integrations qualify branch/config plumbing only;
they are not the full hosted failed-history experiment or a numerical repair.

A complete byte-equal extraction of #537's original failed batch is rejected
as UNVERIFIED before a child or framework import because it lacks the new
canonical/startup metadata. The controller does not infer missing context.
An earlier local check used a sparse extraction and established only
missing-file rejection; its scope correction is retained separately.

## Requested merge of current main

On 2026-10-26 the user explicitly requested **main into precision-migration**.
The fetched main tip is `48054bb35681ecd18e4a8376bb7ccdeb8fe7aa8c`
(#279), with merge base `4add8d2d` and branch parent `3d023218`.
This is branch synchronization, not a merge of PR #277 into main.

The two conflicts are resolved by combining both contracts. SOURCE initializer
validation retains exact typed storage and duplicate handling; LEGACY takes
main's float64-only narrowing, preserving float16 data. Where retains its
SOURCE dtype/weak-type overrides and strict output aval while taking JAX's
weak-type promotion and traced output dtype for LEGACY. Incoming attention
lowering is retained; its existing SOURCE wrapper still calls original JAX.
Independent Astra review accepts these resolutions.

One old LEGACY fixture expected float16 widening to float32, contrary to #279's
documented fix. Its expectation now narrows only float64; all original test IDs,
inputs, dtype/shape/storage comparisons and SOURCE assertions remain. The first
modern run remains **159 passed / one failed**, with its original receipt.
This adaptation does not amend the separate FP64-to-FP16 parity gate.

At source-files
`240c8d92f0edad68541169d68b99317d9a457a01c1c616aa40faf473b83e3e8a`
and index tree `790cfaff606fc8374fcadc9130ea16038fb989e4`, **1,550 identical
ordered unique cases pass on each locked cohort** (Python 3.14.9/JAX 0.12.0.1
and Python 3.11.17/JAX 0.10.2), with zero failures/errors/skips. All 29 selected
complete owners/classes run in four serial batches, preserving both merge
parents, source/Git/index, environment metadata and clean process cleanup.
These include all affected generated attention/Where/BitCast classes and the
relevant SOURCE/builder owners. BF16 schema-only cases are not numerical proof;
optional MaxText model loading and full exact-head CI remain separate gates.

Run #539 is bound to head `3d023218`; its Python 3.12 cast case explicitly passes
in the log. That full job later receives a runner shutdown during batch 30/49:
the full-check step is CANCELLED and the job concludes FAILURE. Control/replay
and artifact upload are skipped; no original failed-cast witness or ZIP is
available for that job. It does not satisfy full-suite acceptance. Predecessor
#537 is terminal seven SUCCESS and its recorded Python 3.11 FAILURE.

The in-flight postmortem helper/worker changes are separately preserved while
the merge is qualified. The next diagnostic reads retained NumPy references and
the existing validation report **after** an unprofiled failure, without source
execution, JAX input materialization, profiling or additional ORT execution. Generic OFF
bitwise failure alone does not prove the same HIGH reference words. Require
same-call original producer/code/source bindings, both complete native HIGH
vectors and original input/model/target typed hashes before making that precise
word-specific inference. No assertion amendment is authorized.

The historical full B–F inventory denominators remain historical. Main adds
metadata/test rows; reconcile the new epoch explicitly rather than silently
replacing old obligations. Public defaults remain LEGACY, and full migration,
original model/resource gates and coordinated public transition remain open.

## Preserved predecessor outcomes

- Full NNX Transformer and generic callable preparation owners on actual Python
  3.14.8: **119 passed, 3 failed, no errors/skips**. Those failures were
  both original native Transformer order cases and the first named original
  opset-23 case. Cold annotation metadata is materialized by eager signature or
  annotation reads. A previous focused success had warmed this metadata and
  therefore did not establish first-call correctness.
- Batch-runner regression owner: **31 passed, no failures/errors/skips** on the
  primary stack. Process cleanup and source fences held.
- The unchanged FP64-to-FP16 tie case passed once through the actual batch hook
  on the secondary stack, with all three pytest phases. Its diagnostic remains
  **UNVERIFIED** at that epoch: a JAX config value did not satisfy the observer's
  literal-string-or-None schema. No comparison words were captured by that
  integration. The later actual-hook v1 diagnostics also remain **UNVERIFIED**:
  their finishing provenance lookup required an unloaded runtime module. Their
  observed comparison words do not replace the complete v2 provenance result.
- Earlier standalone observers passed locally on both JAX cohorts. They do not
  explain the hosted Python 3.12 bitwise failure. Do not invent a rounding fix
  or relax the bitwise test before observing the actual hosted disagreement.
- Latest observed remote run `40361320761` (#533) had four support checks green,
  Python 3.14 failed and the other three full jobs running. This is historical
  status, not current verification. The implementation checkpoint `70ea34ad`
  was subsequently observed as the PR head. The later observed head is
  `3869ba31`: at **2026-10-26 18:09 UTC**, four support checks were **SUCCESS**
  and all four full Python jobs were **IN PROGRESS**. These are time-bound
  observations, not terminal success for the repair or a later pushed head.
- A later terminal observation of #533 records **seven SUCCESS / one FAILURE**:
  Python 3.14 remains failed. Python 3.13 completed all 47 batches and all 15,025
  collected items: 15,014 passed and 11 original skips (one Keras/backend gate
  and ten web gates). This older full-suite result does not qualify repair #536.

## Next concrete work

Complete normal merge hook/documentation closeout, store and push the requested
main integration, restore the separate in-flight diagnostic and inspect all
eight checks on the exact new head. Continue qualifying the precise same-call
postmortem before relying on any HIGH-specific causal inference.
Preserve original models, cases, inputs, assertions and criteria; a diagnostic
commit alone does not repair the numerical mismatch.

Only after the CI gate is green, resume the held shared-comparison work and the
remaining full migration. The full-model resource gate and final public
transition are still open; narrower profiles are intermediate milestones.

## Local evidence and execution constraints

Current receipts and complete JUnit are under
`/tmp/jax2onnx-goal-validation`: `cold-annotations-{modern,minimum-jax}-v1`,
`function-and-diagnostic-{modern,minimum-jax}-v2`, and
`actual-cast-{modern,minimum-jax}-batch-v2`. The historical
`/tmp/jax2onnx-restart-checkpoint-20261021` directory is absent in this container;
its older receipts/candidate references remain historical, not reconstructed
evidence. B-only candidates remain unapplied/unqualified until rechecked.

Current hosted originals are under `ci87-python311/`; cross-checks include
`host-vs-local-python311-word-comparison.json`. The fresh local integration is
`cast-hook-python311-v1` / `actual-cast-python311-batch-v1`. Its optional local
code-generation observation is `cast-hook-python311-dump-v1` and
`cast-original-jit311/`; dump-enabled and dump-free artifact/reference/runtime
words agree. LLVM `fptrunc` alone does not prove final machine rounding. The
local emitted object calls `__truncdfhf2`; installed-helper and executed-symbol
attribution remain separate evidence. The supplementary local target replay
uses frozen hosted references, not another native source execution; its v1
audit failed by confusing raw and typed array hashes, while corrected v2
records four runtime-valid/numerically-failed reports and acceptance errors.
It does not establish the three unattempted hosted comparisons.

The follow-up proof directories are `diagnostic-owners-{modern,python311}-v1`,
`actual-cast-{modern,python311}-batch-v3`, and
`diagnostic-real-hosted-replay-{modern,python311}-v1`. Supplemental outputs stay
under `ci87-python311/replay-current-{modern,python311}-v1.json`; they do not
overwrite original files. The passing-node early exit has its separate
`diagnostic-real-passing-node-not-applicable-v1` receipt.

Actual #537 originals, native manifests, supplementary hosted replay and
independent archive comparisons are under `cib888-python311/`.
The complete original failed-batch extraction used for the control's rejection
check is `control-archive-b888-complete-v1/`. Current control qualifications are
`control-owners-modern-v2`, `control-owners-python311-v1`,
`control-real-{modern,python311}-{on,off}-v1` and
`control-real-archived-rejection-v2`; their aggregate is
`control-packet-qualification-v1.json`. The real ON/OFF branch audits are
`actual-control-{modern,python311}-off-v1/qualification.json`.

Requested-main merge proof is under `main-merge-20261026T212012/`,
`main-merge-modern-v2` and `main-merge-python311-v1` (with combined JUnit and the
separate canonical batch folders). That merge directory also contains exact
backups of the two in-flight files and the WIP patch; their scoped stash is
`fcbeb13b5b7ee9123a03eb47e9ba9f9152ef2e26` until verified restoration.
The #539 Python 3.12 interruption log and compact triage are under
`ci3d023218-python312/`.

Preserve machine stability: one native test/build/hook job at a time, one pytest
worker, native thread counts of one, CPUs 4–7, a 4 GiB aggregate process guard,
12 GiB available memory at startup/8 GiB live, and 4 GiB free disk. Reuse the
replacement `/tmp/jax2onnx-goal-validation/run_guarded.py` with these same caps.
It records environment metadata fences, not complete installed-binary integrity
or wheel archive verification. Do not change the caps or allocator.
Do not mutate source/Git/environments during a native run. Require terminal
receipts, complete JUnit, cleanup and source fences before the next mutation.

Preserve Python 3.11 and the locked dependency cohorts. Route version-sensitive
JAX APIs through `_compat/jax.py`. Read `AGENTS.md`, the precision target/user
view and migration plans before implementation. Do not commit ONNX artifacts.
