# Private STANDARD and support provisioning

This is private test-execution tooling, not a public SOURCE API or an activated
workflow. Real rendering requires a committed source tree whose HEAD and every
tracked file match an independently supplied event SHA. These inputs do not
qualify a hosted job against the complete 1,200-second budget.

The Main-owner tooling was published at `b39e8662`. Its local locked312 attempt
failed at the work deadline after 10 closed S passes. At that head,
[CI 575](https://github.com/enpasos/jax2onnx/actions/runs/41154521659) has six successful
candidates, locked312 RSS FAILURE and locked311 deadline CANCELLED. Four
authoritative support jobs passed; four authoritative full jobs remain pending
in this snapshot. All-eight qualification failed.

This repair revision uses `py311-owners.json` / `py312-owners.json` only for the
explicitly named temporary locked311/312 Main-owner qualification candidates.
Both render `--collection-scope main-owners` and provision with all 188/190 literal
module paths, retaining Main S=4,571/4,621 and the complete C/S/O/A contract below.
The eight authoritative jobs, other six candidates, original 45+5 optional cases,
inputs, skip rules and all budgets remain unchanged. The original full-global
and support routes remain available. Owner acceptance and authoritative/public
SOURCE activation stay **HOLD**, with no measured saving or memory-fit claim.
The [handoff](../implementation_handoff.md#current-checkpoint-2026-10-30) separates
these failed campaigns from accepted finite protocol/quality checks.

Historical checkpoint: at `56c6a24f`, [CI 571](https://github.com/enpasos/jax2onnx/actions/runs/41060058197)
passes all four candidate support jobs. All four global candidates fail:
locked312 detects MaxText leaking `XLA_FLAGS`; locked311/313/314 reject ten
unchanged missing-npm call skips omitted from their allowances. Remaining
assigned cases were unexecuted. These failures remain recorded; the source
repairs require fresh qualification. Gate activation and complete job-budget
acceptance remain open.

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

The support declarations retain their original ordered commands and independent
installation maps:

| Input | Role | Main identities | Generated files | Bootstrap pip |
| --- | --- | ---: | ---: | --- |
| `minimum312.json` | minimum, Python 3.12 | 475 | 26 | 24.3.1 |
| `normalization312.json` | normalization, Python 3.12 | 102 | 0 | 26.2.2 |
| `normalization313.json` | normalization, Python 3.13 | 102 | 0 | 26.2.2 |
| `node312.json` (candidate) | node-smoke, Python 3.12 | 50 | 26 | 26.2.2 |

The normalization declarations share only `normalization-baseline.json`: the
literal Main IDs for layer-norm then cosine, with an empty generated map. Their
package and boundary-context records remain separate. Actual absence of both
generated directories is still required at setup and closure; `{}` alone does
not prove absence. All support skip-allowance lists are empty. The committed
Node declaration, `node-baseline.json` and `node-runtime-reference.json` retain
their separate local and hosted admission evidence.

The modern profiles share only the hash-bound literal identities/generated map
in `modern-baseline.json`. Their package, plugin, environment, JAX transition and
provenance records remain separate. Their observed global PR collections differ;
no global inventory is transported from one cohort to another.

Locked STANDARD, normalization and node-smoke versions must occur in the bound
Poetry lock. A minimum-support cohort instead freezes an independently reviewed
complete floor installation and additionally binds
`scripts/install_minimum_dependencies.py`; it does not infer floors from the
running environment. The editable project version comes from the bound
`pyproject.toml`; pip is an explicit bootstrap pin. The complete installed
distribution set and complete `pytest11`
entry-point set must match the reviewed profile: missing, different, duplicate
or unknown distributions/plugins fail admission. The renderer installs nothing.

Environment/JAX literals come from each profile's reviewed observations. The
portable Python 3.11 CPU profile explicitly changes `LD_LIBRARY_PATH` to genuinely
unset. Its local render-only admission passed; actual collection/native-runtime
qualification remains open. Historical FULL/context receipts do not establish
equivalent loader/runtime conditions.

## Private commands

First establish the exact declared CPU/environment profile using the supported
official interpreter. Run the original `scripts/generate_tests.py` once for
STANDARD and support commands whose original setup includes generation.
Normalization retains no generation and a clean checkout with both ignored
`tests/primitives` and `tests/examples` directories absent.
Generation is a separate child; its environment changes do not rewrite the
coordinator's startup profile. Preserve existing generated artifacts before its
normal cleanup. Do not add framework flags, package overlays or cache clearing.

With that profile already established, the original global Python 3.11 command shape is:

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

For Python profiles, four path values are rendered: `${CHECKOUT}` for `PYTHONPATH`,
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
`--ci-campaign-contract` route. The original STANDARD schema requires full-global
canonical selection; only the explicit owner variant below changes that scope.
The support variant retains its exact original command. Global provisioning
freezes source-bound skip allowances before collection and runs the
existing canonical collection exactly once. After closed-worker hashes,
exit status, zero execution and the independent environment/JAX transition pass,
it writes the existing campaign-v1 contract against that untouched raw receipt
and main manifest, then validates it before the first batch. Ordered main S,
complete/disjoint extras A, every later collection, native phases, JUnit and final
source/context closures retain their existing checks and worker request formats.

Allowances are exact source/line/reason/phase records, not required skips. A
complete ordinary pass remains valid. All four global profiles preserve ten
original missing-npm call-skip conditions. Their owner bytes and original setup
match Main; hosted Main's 12/11/11/11 aggregate skips support this parity, while
its individual phases/reasons were not recorded. Exact phases/locations/reasons
were observed in CI 571 on locked311/313/314. This corrects an omitted conditional
contract, without changing tests or adding skips. Python 3.11 also permits the
reviewed scan and Keras skips; Python 3.12 retains only its scan allowance beyond
these ten, since its Main Keras case passed; each modern profile retains its
independently observed Keras allowance. The dedicated original Node lane still
executes all ten, with no allowances. Unexpected skips and any native
`wasxfail` presence fail campaign validation; allowed skips are not numerical
acceptance.

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

## Reviewed Main-owner variant

The [v2 wire amendment](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/standard-owner-scope-amendment-v2-3mme80px/WIRE-DRAFT.md)
extends the existing renderer and campaign validator. It introduces no second
executor. `jax2onnx-ci-main-owner-cohort-v1` adds only literal ordered
`collection_args` to the STANDARD cohort fields. The reviewed profiles preserve
their original Main identities, inputs, skip allowances and other authority:

| Input | Main S | Complete owner modules |
| --- | ---: | ---: |
| `py311-owners.json` | 4,571 | 188 |
| `py312-owners.json` | 4,621 | 190 |
| `py313-owners.json` | 4,573 | 188 |
| `py314-owners.json` | 4,573 | 188 |

Selectors are the frozen relative `.py` paths in profile order, not inferred from
the current items. Partial function/class/parameter selectors, globs and duplicate
paths are rejected. Every worker uses ordinary collectors and Items, retains
successful/empty collector reports, and collects every added case in those owners
before projecting its assigned literal S after normal collection/ordering hooks.

The owner-rendered baseline carries hash-bound
`collection_scope: "main-owners"`. The same profile's global rendering has empty
collection arguments and no scope marker; both retain the same declared context
transition. Workers read scope from the baseline, never infer it from selectors.
Original global/support authorities reject the owner marker.

Execution is **provisioning-only** through `--_ci-provision-campaign`, using the
matching owner profile and repeating its literal ordered arguments as
`--test-path`. A supplied owner `--ci-campaign-contract` is rejected before RUN
creation or child/context validation. Existing global/support supplied contracts
remain supported. The owner campaign closes genuine complete global C at
`RUN/global-collection/receipt.json`, then complete owner O at
`RUN/collection/receipt.json`. Its distinct
`jax2onnx-ci-main-owner-campaign-v1` binds both manifests and both receipts to
this fresh RUN; no old execution nonce or cross-run C cache is accepted.

O must equal the ordered C projection for the approved modules, and every worker
must reproduce complete O and its collector chronology. Complete extras are
**A=C−S**, including both O−S and C−O; all remain separately assigned. Global
collection errors block success. Global-only skips remain global observations,
not fabricated owner-child events. Source/package/environment/JAX checks,
ordinary terminal hooks, native phases/JUnit, artifact ownership and final raw
receipt-byte checks remain mandatory. Inventory equality alone does not prove
parameter semantics, import effects or numerical equivalence.

All C discovery and O collection costs count toward the unchanged **1,200-second
whole-job, 4-GiB owned-RSS, 160 candidate job-minute and 100-MiB successful-evidence
limits**. The complete Main mapping, original 45+5 optional models, support
contexts, full extras and exact-final all-cohort suites remain required. Finite
protocol controls do not qualify production import order, model numerics, memory
savings or activation; follow the [execution plan](../ci_execution_plan.md#reviewed-main-owner-collection-amendment).

## Private support variant

`jax2onnx-ci-support-cohort-v1` adds a required `support_role` and ordered
`collection_args`: minimum and node-smoke require Python 3.12; normalization
permits 3.12/3.13. Repeat each original selector as `--test-path` in that exact
order. Other selection overrides remain prohibited. The support campaign uses
`jax2onnx-ci-support-campaign-v1` with `command_inventory`; the unchanged global
route retains `global_inventory`. Both supplied-contract and first-collection
routes keep closed-worker, complete command C, ordered S/disjoint A, native
phase/skip/`wasxfail`, JUnit and final source/context checks. The declarations
activate no workflow or hosted acceptance.

At committed `c338c33f`, the first-collection route passes normalization312 and
normalization313 **C=S=102 / 306 native phases each**, and minimum312
**C=615 / S=475 / 1,425 phases / A=140 UNEXECUTED**. All three have zero skips
or raw `wasxfail`, closed package/source/resource checks, and 18 actual boundary
comparisons including `${RUN}/native-model-artifacts`. Astra independently
verified all 384 frozen files (combined summary `52cea604…`). This is local
committed-data qualification of the original criteria; native transport remains
`SUBSET_TRANSPORT_ONLY`. It establishes no GitHub event, complete hosted timing
or full SOURCE/B–F acceptance. Earlier raw observations did not activate the
native-evidence key and retain their separate epochs and failure evidence.

The five-file extension passes 447 controls / 1,341 phases on official 3.13;
the prior 373 controls remain as 362 literal IDs plus 11 explicit global
renamings/parameterizations, with 74 additional cases. Its preserved
`closure-error.json` records the rejected literal-ID bookkeeping assumption.
After the test-only refinement, all 447 controls pass on 3.11, including a
genuine unsupported-normalization-role rejection before any coordinator C/batch;
the two affected positive routes pass again on 3.13. Rejection is not positive
support qualification. Final packet `support-provision-test-v2-alqe7ufe/`
(pair freeze `2635bb2d…`) is below the evidence root above.

### Private portable Node binding

Only `node-smoke` on Python 3.12 requires the optional `node_runtime` block;
its rendered manifest also carries the paired `support_role` discriminator.
Supply `--node-prefix` explicitly. Source/cohort bindings cover the package lock
and both executed JS helpers. The reviewed reference fixes official
archive/binary/npm bytes and the complete `node_modules` regular-file/link
inventory. Admission resolves
bare `node` through the actual PATH, checks its canonical target, executable bit
and independent binary hash, and validates official npm links without escapes.
Portable prefix moves are allowed. Python/global profiles retain their shapes
and environment rules.

The Node block declares absence of Node/npm overlays, including lowercase
aliases, except explicit portable cache/config paths. The candidate uses
`${NODE_PREFIX_PARENT}/npm-cache` and empty
`${NODE_PREFIX_PARENT}/config/{user,global}.npmrc`. Full audits stream all bound
bytes at existing source-audit boundaries; per-test checks inspect resolution
and overlays. Startup, collection and final Node observations plus terminal
rechecks are mandatory. Missing observations or drift fail admission. The
reference is never refreshed from the current host during admission.

The five-file code/test epoch is `38dfb1ff` plus patch `010719c8…`, AST-equivalent
to reviewed V3 `c32a81d8…`, before these docs/data. Ordinary official Python 3.12
passes 53 focused controls and all 494 owner controls, with zero top-level
skips/xfails; full outer cost is **1,006.50 s / 342.02 MiB**. Pinned formatting,
Ruff, mypy and policy checks pass (receipt `24a0c16b…`, freeze `f04a9ee3…`).
These controls establish no real Node admission or other-minor qualification.
The unchanged reference `47cb4f12…` authenticates the official archive and
current retained installation. Complete module JS/WASM hashes were first
observed at that extraction; earlier raw S50 receipts bound package manifests
only. New Node observations and native transport require committed-data
qualification. Trust assumes the same invocations and trusted loader; dynamic
library loads and swap/restore TOCTOU behavior are not attested.

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
