# Precision migration handoff — 2026-10-26

This is an intentionally incomplete local checkpoint, committed at the user's
request. It is not a green-CI or migration-completion claim. Continue on branch
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
- Add passive, exact-node FP64-to-FP16 diagnostics to the existing batch runner,
  with outcome/profile noninterference tests. These diagnostics do not change
  conversion, inputs, references, assertions, test selection or acceptance.

`main` was already merged **into** this PR branch as `ee45c6dc`; it was not a
merge of the PR into main. That was also the last verified remote PR head before
this local checkpoint. Refresh GitHub state before relying on any CI status.

## Verified results and remaining failures

- Full NNX Transformer and generic callable preparation owners on actual Python
  3.14.8: **119 passed, 3 failed, no errors/skips**. The remaining failures are
  both original native Transformer order cases and the first named original
  opset-23 case. Cold annotation metadata is materialized by eager signature or
  annotation reads. A previous focused success had warmed this metadata and
  therefore did not establish first-call correctness.
- Batch-runner regression owner: **31 passed, no failures/errors/skips** on the
  primary stack. Process cleanup and source fences held.
- The unchanged FP64-to-FP16 tie case passed once through the actual batch hook
  on the secondary stack, with all three pytest phases. Its diagnostic remains
  **UNVERIFIED**: a JAX config value does not satisfy the observer's current
  literal-string-or-None schema. No comparison words were captured by that
  integration. Fix the diagnostic and require actual word/provenance records.
- Earlier standalone observers passed locally on both JAX cohorts. They do not
  explain the hosted Python 3.12 bitwise failure. Do not invent a rounding fix
  or relax the bitwise test before observing the actual hosted disagreement.
- Latest observed remote run `40361320761` (#533) had four support checks green,
  Python 3.14 failed and the other three full jobs running. This is historical
  status, not current verification. The implementation checkpoint `70ea34ad`
  was subsequently observed as the PR head; inspect its actual checks rather
  than carrying over the previous head's results.

## Next concrete work

Finish and independently review the cold-annotation repair. Reuse the existing
`AnnotationState.attach` boundary, derive pinned capture-port information
without evaluating annotations, and avoid the redundant signature query for
registered class targets whose receiver handling is already known. Do not
prewarm original annotations or weaken stale-metadata guards. Run the full NNX
owner in a fresh worker, including the first original cases before added tests.

Correct the diagnostic config schema using actual loaded JAX values; preserve
the original call and all outcome/profile/source fences. Validate its complete
word schema through the actual batch runner, not only synthetic controls.
Then run relevant dual-JAX checks and normal pinned hooks, commit/push the
reviewed repairs, and inspect all eight checks on that exact head.

Only after the CI gate is green, resume the held shared-comparison work and the
remaining full migration. The full-model resource gate and final public
transition are still open; narrower profiles are intermediate milestones.

## Local evidence and execution constraints

Detailed receipts, environments, candidate patches and handoff journals are in
`/tmp/jax2onnx-restart-checkpoint-20261021` (local temporary evidence, not required
for understanding the committed code). Start with `root-draft-pr-current-handoff-v35.json`
and the later checkpoint handoff there. B-only candidate patches are not applied
or qualified merely because they exist; recheck preimages and review state.

Preserve machine stability: one native test/build/hook job at a time, one pytest
worker, native thread counts of one, CPUs 4–7, a 4 GiB aggregate process guard,
12 GiB available memory at startup/8 GiB live, and 4 GiB free disk. Reuse the
sealed `source-resource-protected-serial-runner-sealed-v1/run_serial_v9_resource_protected.py`
and ROOT wrappers in that evidence directory; do not change its caps or allocator.
Do not mutate source/Git/environments during a native run. Require terminal
receipts, complete JUnit, cleanup and source fences before the next mutation.

Preserve Python 3.11 and the locked dependency cohorts. Route version-sensitive
JAX APIs through `_compat/jax.py`. Read `AGENTS.md`, the precision target/user
view and migration plans before implementation. Do not commit ONNX artifacts.
