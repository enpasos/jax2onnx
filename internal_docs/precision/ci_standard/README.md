# Private STANDARD provisioning

This is private test-execution tooling, not a public SOURCE API or an activated
workflow. Real rendering requires a committed source tree whose HEAD and every
tracked file match an independently supplied event SHA. These inputs do not
qualify a hosted job against the complete 1,200-second budget.

## Reviewed cohort inputs

All profiles preserve literal ordered main `81ad6ca` identities. Generated
bindings include the two empty package markers; they describe the complete
output of the unchanged generator, not a subset of its modules.

| Input | Main identities | Generated files | Declared bootstrap pip | Optional integrations |
| --- | ---: | ---: | --- | --- |
| `py311.json` | 4,571 | 26 | 25.1.1 | off |
| `py312.json` | 4,621 | 28 | 26.2.2 | original 45 MaxText + 5 MaxDiffusion |
| `py313.json` | 4,573 | 26 | 26.2.2 | off |
| `py314.json` | 4,573 | 26 | 25.1.1 | off |

The modern profiles share only the hash-bound literal identities/generated map
in `modern-baseline.json`. Their package, plugin, environment, JAX transition and
provenance records remain separate. Their observed global PR collections differ;
no global inventory is transported from one cohort to another.

Every dependency version must occur in the bound Poetry lock. The editable
project version comes from the bound `pyproject.toml`; pip is an explicit
bootstrap pin. The complete installed distribution set and complete `pytest11`
entry-point set must match the reviewed profile: missing, different, duplicate
or unknown distributions/plugins fail admission. The renderer installs nothing.

Environment/JAX literals come from each profile's reviewed observations. The
portable Python 3.11 CPU profile explicitly changes `LD_LIBRARY_PATH` to genuinely
unset. Its local render-only admission passed; actual collection/native-runtime
qualification remains open. Historical FULL/context receipts do not establish
equivalent loader/runtime conditions.

## Private commands

First establish the exact declared CPU/environment profile using the supported
official interpreter, then run the original `scripts/generate_tests.py` once.
Generation is a separate child; its environment changes do not rewrite the
coordinator's startup profile. Preserve existing generated artifacts before its
normal cleanup. Do not add framework flags, package overlays or cache clearing.

With that profile already established, the Python 3.11 command shape is:

```sh
python -B scripts/prepare_ci_standard.py \
  --cohort internal_docs/precision/ci_standard/py311.json \
  --event-sha "$EXPECTED_CHECKOUT_SHA" \
  --run-dir "$RUNNER_TEMP/standard-run" \
  --output "$RUNNER_TEMP/standard-main.json"

python -B scripts/run_pytest_batches.py \
  --batch-size 500 --output-dir "$RUNNER_TEMP/standard-run" \
  --ci-lane standard --ci-main-baseline "$RUNNER_TEMP/standard-main.json" \
  --_ci-provision-campaign internal_docs/precision/ci_standard/py311.json \
  -- --durations=50 --durations-min=1
```

Set `EXPECTED_CHECKOUT_SHA` from the external workflow event expression
`${{ github.event.pull_request.head.sha || github.sha }}`: the PR head for a pull
request, or the event SHA for a push. `GITHUB_SHA` can identify a PR merge event
instead of the checked-out PR head. Deriving the expectation from the current
Git HEAD does not establish external event authority. A reviewer-supplied local
SHA qualifies only local admission, not GitHub event provenance.
Select the matching cohort file for other supported
interpreters. Python 3.12 additionally supplies `--maxtext-source` and
`--maxdiffusion-source` to the renderer, pointing to clean checkouts at the
profile's pinned refs with both original `MODELS=all` settings. Normal plugin
autoload and the profile's other exact settings remain required.

Only four path values are rendered: `${CHECKOUT}` for `PYTHONPATH`,
`${MAXTEXT_SOURCE}` and `${MAXDIFFUSION_SOURCE}` for optional source directories,
and `${RUN}/native-model-artifacts` for child artifact expectations. Other
environment/JAX literals are preserved. The native-evidence key must be genuinely
absent in the parent; the existing transport creates and owns its predicted child
directory before loading lane authority. The future run and external manifest
destinations must be fresh, including no broken symlink leaves. Canonical path
checks reject aliases into the checkout.

The renderer compares all tracked bytes to the event's Git objects, rejects
unreviewed untracked files, and checks the full generated output set and bytes.
The external manifest binds the tracked/generated union without a self-hash
cycle. Drift, missing output, extra output or a package/plugin/profile mismatch
fails instead of refreshing authority from the observation.

## First-collection provisioning and limits

`--_ci-provision-campaign` is mutually exclusive with the existing
`--ci-campaign-contract` route. It requires full-global canonical standard
selection, freezes source-bound skip allowances before collection, and runs the
existing canonical collection exactly once. After closed-worker hashes,
exit status, zero execution and the independent environment/JAX transition pass,
it writes the existing campaign-v1 contract against that untouched raw receipt
and main manifest, then validates it before the first batch. Ordered main S,
complete/disjoint extras A, every later collection, native phases, JUnit and final
source/context closures retain their existing checks and worker request formats.

Allowances are exact source/line/reason/phase records, not required skips. A
complete ordinary pass remains valid. Python 3.11 permits the reviewed scan and
Keras skips; Python 3.12 permits only the scan skip, since its Main Keras case
passed; each modern profile permits only its independently observed Keras skip.
Unexpected skips and any native `wasxfail` presence fail campaign validation.

Focused v3 controls passed 47 cases; the preceding 40-case result and its sole
path-containment review finding remain preserved. Complete component/process
qualification passed 373 cases / 1,119 phases on each official Python 3.11 and
3.14 family, including all six opt-in synthetic xdist controls. Receipts are in
`portable-standard-provision-v1-cb_dzqs5/complete-owners-v3/` below
`/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/` (pair freeze `237b62f3…`).
All four local render-only admissions passed at clean `11d5739f`, with each
original generator run once, own package/plugin expectations, framework-unloaded
renderer and exact Root/Main restoration. The packet is
`portable-standard-render-only-11d5739f-8sxiaspv/aggregate-freeze.json` under the
same evidence root (hash `4b5560aa…`). This uses a reviewer-supplied **local SHA**,
not GitHub event provenance. Saved cohort epochs remain distinct from restored
Root28; no collection, runtime or FULL followed these admissions.

## Controlled-command resource observer

`scripts/run_with_resources.py` is a standalone Linux observer; it does not
change runner requests, authority schemas, profiles or child argv/env/cwd/streams.
For example, with an independently approved command and fresh external output:

```sh
python -B scripts/run_with_resources.py \
  --output-dir "$RUNNER_TEMP/command-resources" --timeout-seconds 900 \
  -- python -B -m mkdocs build --strict \
    --site-dir "$RUNNER_TEMP/documentation-site"
```

It samples itself and owned PID/starttime descendants every 0.2 s through
cleanup, including separately sessioned/adopted children. Defaults are 4 GiB
aggregate RSS, 2 GiB host headroom and 4 GiB declared-filesystem free space;
existing local outer guards retain their stricter 12/8 GiB host floors. Required
read/accounting failures and observed breaches cannot pass. Optional/shared
kernel and cgroup counters retain unavailable/reset/ownership labels, without
inferring OOM from a signal or replacing unknown values with zero.

Pair every receipt with the actual command exit/signal. The summary separates
the main child result, guard result, adopted wait outcomes and planned wrapper
termination. It claims only sampled controlled-command compliance. External
Actions resource intervals remain UNOBSERVED, not an automatic activation block;
the separate whole-job 1,200-second requirement still includes those actions.

The dedicated owner passes 28 cases / 84 phases on each official 3.11/3.14 family,
including an ordinary exact CLI, detached descendants, PID reuse and preserved
failed controls. Evidence is `owned-resource-wrapper-v2-owners-xgispep0/`
(pair `931cafa8…`). No candidate workflow, numerical migration, full B–F or hosted
timing/resource qualification is activated by these finite results.
