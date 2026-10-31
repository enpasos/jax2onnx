# Precision migration CI execution plan

## CI584 native-receipt capacity repair

At published `7fcfa114`, CI584's five locked compute jobs failed during campaign
initialization with `worker context file exceeds bound`; the strict 3.12 join
also failed. Three independently verified full-C receipts were
16,845,963–16,883,869 bytes, exceeding 16,777,216 bytes. Their native collection workers closed successfully and
phase-free, before any assigned scientific batch. The seven pinned all-files
hooks passed on the three independently checked runtime profiles
([primary accounting](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/hosted-ci-7fcfa114-watch-ovtbutTK/receipt-overflow-primary-accounting.json)).

One finite 64 MiB native-receipt limit now matches existing transport and Actions
restore capacity across campaign inventory/rechecks, strict closed-worker decode
and part-join read/hash paths. Manifest, request, source, JUnit and profile bounds,
primary-copy limits, artifact budgets, full C/O/S and resource/time/numerical
criteria remain unchanged. All original byte/source/context/phase/JUnit checks
remain fail-closed; overflow diagnostics include the exact path and limit.

Local focused qualification closed **21 PASS / zero skips** on each approved
official 3.11/3.13 family, including genuine global/Main-owner receipts above
16 MiB and strict duplicate/hash/source/phase rechecks. The existing real
3.12.9 source-root and serial-parts/join controls closed **two PASS / zero skips**,
with all original receipt read/hash paths observed at 64 MiB
([local controls](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/ci584-native-receipt-capacity-NHht46OA/receipt-capacity-controls-closure.json)).
Synthetic Actions snapshots remain unit controls, distinct from hosted 3.12.15
qualification.

CI584 remains FAILED. Whole hosted execution, complete EXTRA profiles and
scientific/final B–F acceptance remain separate; local controls do not qualify
them. The mandatory native BF16 TopK failure is preserved under unchanged
runtime, inputs and criteria.

## Focused EXTRA projection — local verifier controls

The canonical runner extends local `e11918ad` with a
`--ci-extra-scope all|focused` option. The default remains complete **A=C−S**;
an explicit scope requires a strict EXTRA campaign. Focused execution accepts
only full-global authority, never Main-owner or support campaigns. After the
genuine full collection and original S/A checks, literal **F** must be nonempty,
unique, entirely in A and in exact collection order. Missing, reordered,
duplicated or S IDs fail; they are never intersected away. The existing serial
planner executes F and checks its exact completion. Reports retain complete A,
F and **R=A−F**, so newly collected unmatched IDs remain owned by complete EXTRA.

A focused PASS is explicitly **F_ONLY**, with full A **UNEXECUTED**, numerical
acceptance **NOT_ASSERTED** and full B–F **UNVERIFIED**. A passing rejection test
does not establish numerical correctness. Complete EXTRA still executes all A,
including the existing empty-A collection-only case. All source/context,
phase/JUnit, wasxfail and final-byte audits remain. Optional
`extra_allowed_skips` uses the existing exact source/line/reason/phase record,
checked against actual A. It cannot excuse failed scientific phases, another
case, or the real fixed-parts smoke on Python 3.12.

No actual F list, cohort allowance, EXTRA profile or workflow activation is
qualified by these controls. Freeze those only after the combined source head
and genuine collection, including the accepted ownership integration's cases.
Separate EXTRA profiles must bind `JAX2ONNX_TEST_CI_LANE_XDIST=1` and, where
applicable, `JAX2ONNX_RUN_FIXED_PARTS_SMOKE=1` in startup/collection/final
contexts. The six existing xdist controls retain their reviewed synthetic
two-worker exception inside serial EXTRA; no xdist run occurred here.
Qualified STANDARD profile data is unchanged.

The ownership plan remains four routine focused cohorts, with four complete-A
cohorts replacing focused work for milestone/final campaigns; preserve the
additional complete minimum311 job, minimum312 support's full A (currently 140),
and nightly Node/Chrome contexts with exact active-branch SHA wiring.
Retirement requires retained original outcomes, complete ownership and exact-head
replacement qualification with independent review; preserve still-live original
jobs without cancellation. Old-eight SUCCESS is not a blanket precondition for
independently reviewed CI repair publication. CI583's native BF16 TopK failure
remains FAILED; choosing F cannot hide or resolve it. Gate activation stays HOLD.
At **2026-10-31 16:02:41 UTC**, the user explicitly allowed reviewed SOURCE
publication while the named BF16 failure remains mandatory and FAILED. This
exception does not waive new CI regressions or scientific/final acceptance and
does not authorize custom runtimes, changed inputs or weakened criteria.

The complete changed runner, lane-selection and renderer files account for
**731 IDs per official 3.11/3.13 family: 725 PASS and six unchanged opt-in xdist
skips** ([local controls](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/ci-focused-extra-e119-yeb98sy3/focused-extra-controls-closure.json)).
They share 730 literal IDs; the remaining existing provisioning role is
`unsupported-normalization-role` on 3.11 and `first-collection` on 3.13.
The prior e119 scope included 60 unchanged cache controls; this scope instead
includes all 100 renderer controls and 75 net additions to runner/lane coverage.
Two old Main-owner rejection identities expand into six explicit scope controls.
Eight genuine pytest skip controls per family cover supplied/provisioned authority,
all/focused execution, exact setup/call/teardown and JUnit/source/line/reason,
and wrong-allowance rejection. Expected rejects do not establish numerical PASS.
The existing real official 3.12 source-root and serial-parts/join controls also
closed **two PASS / zero skips**. Local 3.12.9 and synthetic Actions observations
remain distinct from hosted 3.12.15 qualification.

Initial fixture-admission and launcher failures remain in the local evidence.
The 3.11 runner's 278 PASS execution initially retained 71,184,384 allocated
bytes, exceeding its 64 MiB limit. Its original failure receipt is immutable;
verified PAX preservation corrected retained storage without replaying tests.
This is a retention correction, not a retroactive budget PASS.

## Earlier complete-EXTRA repair — local e11918ad

At exact `6c444f3b`, CI583's original Python 3.12 job ended **FAILURE** at
**2026-10-31 14:20:27 UTC** in BF16 TopK `finite-inline`, batch 31/55
([primary excerpt](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/hosted-ci-6c444f3b-watch-niR5o14v/original312-terminal-bf16-topk-failure-excerpt.log)).
Independent review classifies this as the documented CI549/551 official-native
JAX/jaxlib 0.10.2 failure: first fresh JIT returns positive BF16 words
`0001, 007f` instead of descending `007f, 0001`. They differ by a factor of
127, so this is not a tie or oracle defect. Warm/eager/ONNX/ORT and later
batches were not reached. Exact CPU/compiler cause remains unproved. The case
and overall CI remain **FAILED**; resolving it requires an explicitly reviewed
official-runtime profile decision. Pins, inputs and criteria remain unchanged;
no custom native patch/build is authorized.

At the observed [CI583](https://github.com/enpasos/jax2onnx/actions/runs/41206202756)
checkpoint on exact `6c444f3b`, all nine candidate compute jobs and the strict
locked312 join succeeded: maximum whole job **857 s**, aggregate **4,421 s**,
successful artifacts **64,156,009 bytes**. This qualifies that candidate
native-pytest/support/resource scope. It omits Main's whole-tree pinned hooks
and complete EXTRA execution. Four original support jobs succeeded; the four
original full Python jobs were still in progress at that earlier observation.
The terminal failure above precludes overall CI acceptance. Their authoritative
coverage remains required; replacement activation stays **HOLD**.

The current repair extends the existing strict campaign to canonical
global and support **EXTRA A = C − S**, preserving complete ordered collection,
source/context checks, genuine phases/JUnit and final byte audits. Both supplied
and provisioned campaigns require explicit lane and baseline. Main-owner-only
EXTRA rejects because it omits outside-owner extras; extra skips, xfail and
incomplete phases cannot establish PASS. Reports retain numerical acceptance
**NOT_ASSERTED** and full B–F **UNVERIFIED**. The three changed complete CI
control owners closed **610 PASS / six unchanged opt-in xdist skips** on each
approved official Python 3.11/3.13 family, with 616 accounted IDs per family
and 615 shared literal IDs. Real strict EXTRA controls include a passing
expected-rejection body and empty A with complete collection and no execution.
The actual official Python 3.12.9 source-root and real fixed-parts join controls
closed **two PASS / zero skips** ([local closure](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/ci-extra-quality-source-IQcm5H/controls-closure.json)).
Their synthetic Actions data do not qualify hosted Python 3.12.15 timing,
complete cohort A, numerical equivalence or B–F completion.

Candidate monitored WORK now includes the original pinned
`poetry run pre-commit run --all-files`, preceded by the original marker fix,
once per locked Python cohort at the base-lock point before optional dependency
installation and generation. Locked312 part a owns it; part b avoids duplicate
hooks. Support jobs retain their original responsibilities. Hook failures fail
WORK, and all quality costs count toward the unchanged 1,100-second WORK and
1,200-second whole-job budgets. CI583's timings cannot qualify this added work:
fresh exact-head hosted quality-inclusive qualification is still required.

Before any activation, retain the original minimum-dependency Python 3.11 job
as additional required coverage; execute minimum312 **A=140** through strict
support EXTRA; assign literal focused EXTRA IDs for extrema, ownership,
mutable-static rejection and their consumers; assign all remaining complete A
to milestone/final execution across supported cohorts; and preserve nightly Web's original
contexts plus eventual exact active-branch SHA wiring. These executable owners
and missing migration verifiers are not supplied by this repair. Keep the
original jobs enabled until exact-head replacement qualification and independently
accepted STANDARD/EXTRA ownership close. Routine CI may transition at that point;
separate milestone/final full B–F obligations and known FAILED/UNVERIFIED science
gates remain open. Public LEGACY remains unchanged.

## Historical CI582 admission repair — qualification was PENDING

The two-part controller is published at `1ff4c9c6`; **hosted qualification is PENDING**.

At exact `1ff4c9c6`, [CI582](https://github.com/enpasos/jax2onnx/actions/runs/41205139374)
was rejected before any job: five join job-level environment expressions used
an unavailable `runner` context. The same paths are now initialized from real
`RUNNER_TEMP` in the first join step through `GITHUB_ENV`; the targeted check
rejects this context in workflow/job environment maps. No model/work result
is inferred from that failed admission, and fresh hosted qualification remains pending.

CI581 at exact `8f113ea8` [locked312 job](https://github.com/enpasos/jax2onnx/actions/runs/41189302941) ended CANCELLED after 1,252 seconds whole-job:
WORK was 1,105.4059 seconds and packing reached its 90-second deadline.
Its observed aggregate peak was 4,429,303,808 bytes; the observed worker
peak was 4,308,254,720 bytes. All 50 original optional cases, including Omni,
closed genuine setup/call/teardown PASS with JUnit accounting.
Only 1,708/4,621 Main S cases closed. Full STANDARD and whole-job acceptance
were not established.

The approved plan uses two fixed SERIAL locked312 parts, 216/4,405 cases.
Each repeats original full C/O and keeps original 500-case batching, inputs
and criteria. Their normal envelope is 5 GiB RSS, fresh available RAM >=12 GiB,
live available RAM >=8 GiB and free disk >=4 GiB.
Other candidate profiles retain their serial 4-GiB envelope.
The strict join requires the same head/attempt, runtime, sources and context,
genuine phases/JUnit, and a disjoint ordered union equal to full original S.
An individual part PASS cannot qualify locked312.

Nine compute candidates plus one small fixed join retain 1,100-second WORK,
1,200-second whole-job, 160-job-minute aggregate and 100-MiB successful-upload
budgets. The original eight authoritative jobs remain enabled until candidate
qualification.
Forecast WORK is 750.049/902.935 seconds plus ten untimed Python 3.12 Web cases.
These are forecasts, not measured fit.
The 250-case/8-GiB parallel draft was rejected and remains private; it creates
no published infrastructure obligation.

The stale-assertion checkpoint `858bf548` closed 25 PASS on each official
Python 3.11/3.13/3.14 family. Initial policy controls closed 100 PASS and one
explicit 3.12-only skip. The actual official Python 3.12.9 control then closed
2 PASS (`8642f73a`): two tiny serial native campaigns, their strict join,
and semantic/source/profile/resource/time/assignment rejection controls.
Its Actions observations are synthetic unit controls, not hosted timing proof;
hosted Python 3.12.15 remains separately qualified. Earlier disk-censored and
join-source-root failures remain retained. The seven complete adjacent CI test
files closed 805 PASS and 7 explicit skips on each official 3.11/3.13 family
(811 shared JUnit IDs; one existing cohort-dependent provisioning role).
No full hosted acceptance, numerical conclusion or B–F completion is asserted.
Historical failed runs and their evidence remain below.

## Historical CI580 admission correction — superseded diagnostic policy

At exact `027d92eb`, [CI580](https://github.com/enpasos/jax2onnx/actions/runs/41188009953)
stopped locked312 after 14 seconds: the initial 12-GiB capacity check passed,
but the observer's unchanged CLI ceiling rejected 5,120 MiB before setup/C/O
or any model execution ([primary packet](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/hosted-ci-027d92eb-watch-zMeCkRQP/locked312-early-admission-failure.json)).
There has been **no complete 5-GiB model observation**. The existing observer
now requires explicit diagnostic opt-in, enforces 12-GiB initial/8-GiB live
availability, a 4-GiB disk floor and at most 1,100 seconds, and records the
actual 5,120-MiB execution limit. It remains **UNQUALIFIED** and exits nonzero
even for child success. Default STANDARD admission stays at 4,096 MiB and
1,200 seconds. The workflow passes this opt-in only to locked312; original
inputs, C/O/S, all batches, criteria and one-observation scope are unchanged.

The complete observer test file and eight affected workflow branches have
**46 PASS / zero skips** ([focused closure](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/ci580-observer-cli-focused311-xdo5sp0l/qualification-closure.json)).
A separate unmodified official312 CLI probe completed a trivial child with
raw 5,120-MiB limits, child exit zero, observer exit 125 and UNQUALIFIED status.
This verifies admission and closure only; full Omni/hosted RSS acceptance remains open.

## CI579 follow-up: unchanged acceptance, one observational execution

Published `a612d5e4` / [CI579](https://github.com/enpasos/jax2onnx/actions/runs/41182937914)
still fails locked312's aggregate 4-GiB criterion. First WORK crossing is
4,297,863,168 bytes; the cleanup peak is 4,303,409,152 bytes. Original Omni
has setup PASS only, with no retained Omni/temp ONNX under the inventoried
roots. Its later completion/peak remains unknown. C/O/S/A is
19,564/4,850/4,621/14,943; 147 S cases close and all 50 original optional IDs
remain assigned ([primary packet](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/hosted-ci-a612d5e4-watch-xi9acp2x/locked312-failure-trigger-phase-accounting.json)).

The next existing locked312 job is explicitly a **one-shot DIAGNOSTIC**:
execution limit 5,120 MiB, initial/before-work available RAM at least 12 GiB,
live available RAM 8 GiB, disk floor 4 GiB, 1,100-second work and 1,200-second
whole-job bounds. Its raw summary honestly records 5,120 MiB. STANDARD
acceptance stays **4,096 MiB / UNQUALIFIED** and this job is forced non-success
even if all original C/O/S batches finish. No new job, controller, profiler,
case/input/parameter/RNG change, skip, or local recollection is introduced.
After this one run, the diagnostic mode must be explicitly removed or reviewed
again before another push; it is not a recurring qualification policy.

Original Python 3.11 separately stopped before batch 12: native artifact-root
free space was 7,286,153,216 bytes versus the unchanged necessary lower bound
7,486,178,400 bytes, including 3,124,102,240 bytes of remaining original raw
outputs ([primary packet](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/hosted-ci-a612d5e4-watch-xi9acp2x/original311-disk-failure-accounting.json)).
Hosted full-check preparation now releases only validated pip/Poetry download
caches after all installs and generation, before pytest; it records actual
paths, allocated/download bytes, filesystem capacity and runtime identity
before/after. Virtualenvs, models, receipts, TMPDIR, locks and native floors
are preserved. Local invocation performs no user-cache cleanup.

## Earlier CI578 repair status — 2026-10-31

Published `8578c47a` / [CI578](https://github.com/enpasos/jax2onnx/actions/runs/41173158463)
has **seven STANDARD successes / locked312 FAILURE**. Locked311/313/314 whole
jobs succeed in **1,061/993/1,049 seconds**, each below 1,200 seconds
([job API snapshot](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/hosted-ci-8578c47a-watch-uJvj63pA/jobs-20261031T073238Z.json)).
Locked312 first crosses 4 GiB during WORK at **4,303,532,032 bytes**; its later
cleanup peak is **4,310,863,872 bytes**. Original Omni has setup PASS only;
**147 S cases close / 4,474 remain unclosed**. Cleanup is confirmed; OOM cause
remains unverified. All 50 original optional IDs are assigned: 44 MaxText and
five MaxDiffusion cases close, while Omni remains incomplete
([primary accounting](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/hosted-ci-8578c47a-watch-uJvj63pA/locked312-failure-trigger-phase-accounting.json)).

The strict single-decode C/O repair (`33ce072f` runner /
`b58b6ae9` tests) removes a duplicate temporary decoded graph while preserving
all native, original-byte, phase and context checks. The **complete controller
test file has 212 PASS / zero failures, errors or skips on each official311/313
family**, retaining all previous 203 controls plus nine new cases
([311](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/ci578-single-decode-controls311-ff3661a0bd/qualification-closure.json),
[313](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/ci578-single-decode-controls313-04444304f1/qualification-closure.json)).
There are **211 shared literal IDs**; one existing provisioning-role case differs
by cohort. Both resource, PID and protected-source/artifact closures pass.
This establishes local validation semantics; **hosted RSS savings and exact-head
full campaign acceptance remain PENDING**. Earlier bound-`e6f1120c`
[preparation](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/ci578-actual-preparation-probe-ahuxa5wk/preparation-diagnostic-closure.json) and
[context replay](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/ci578-genuine-context-replay-5rtvil1_/replay-diagnostic-closure.json)
diagnose allocation overlap, not this candidate's acceptance.

Preserve Main **S=4,621**, all 50 original optional IDs, original inputs/RNG42,
criteria, cohorts, 4-GiB and 1,200-second budgets, and all audits. Next run ordinary
quality and strict docs, make one normal seven-hook repair commit, then qualify
**one full hosted campaign on that exact head**. No extra 500-second local
recollection or A/B run is needed. All-eight acceptance remains FAILED; full
B–F remains open and public LEGACY/gate activation stays HOLD.

## Earlier CI577 checkpoint — retained source epoch

At published `506903cc`, [CI 577](https://github.com/enpasos/jax2onnx/actions/runs/41169144281)
has seven successful STANDARD candidates and locked312 FAILURE. Four original
support jobs succeeded; four original full-suite jobs were still LIVE at the
**2026-10-31 05:27:02 UTC** observation. Locked311/313/314 close their original
Main S=4,571 / 4,573 / 4,573 in **882 / 894 / 1,017-second whole jobs**
([bounded successful-job accounting](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/hosted-ci577-506903cc-watch-jfyzgsw7/locked311-313-314-success-accounting.json)).
All-eight STANDARD qualification remains **FAILED**.

Locked312's first WORK crossing is **4,295,016,448 bytes > 4 GiB** at
644.033 seconds; the subsequent cleanup sampled peak is **4,300,374,016 bytes**.
That simultaneous peak includes the controller's **114.129 MiB** and Omni
worker's **3,969.777 MiB**. Only 147 S cases close, including 44 original
MaxText and five MaxDiffusion cases. Original Omni has setup PASS only;
call/teardown and cleanup are not closed, and OOM cause remains unverified.
The censored worker trace cannot establish standalone completion below 4 GiB
([failure accounting](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/hosted-ci577-506903cc-watch-jfyzgsw7/locked312-failure-trigger-phase-accounting.json),
`52ac19bd…`). Its whole job is 742 seconds; time compliance does not repair the
RSS/closure failure or qualify the earlier finite `98a34705` proof as a campaign.

The scoped inventory-lifetime candidate releases complete decoded C/O receipt
graphs before child launch, retaining bound paths/digests and ordered IDs. It
rereads checked local inventories for required comparisons and preserves full
initial projection, context/native, deep accounting, phase/JUnit and final
receipt-byte audits
([source plan](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/campaign-inventory-lifetime-source-_ito3jun/source-plan.json)).
The complete controller test file has **203 PASS / zero skips per official311/313 family**
([311](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/campaign-inventory-lifetime-controls311-9ev9c4de/qualification-closure.json),
`b398abed…`;
[313](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/campaign-inventory-lifetime-controls313-retry-r10s8az3/qualification-closure.json),
`ca67718d…`), retaining one existing cohort-specific provisioning role. All three
new real Main-owner multibatch/lifetime/late C/O tampering controls pass.
This establishes object release and audit semantics, **not hosted RSS savings**.
The earlier 313 result, **137 PASS / 1 FAIL / 65 unexecuted**, remains a retained
unchanged 4-GiB+64-MiB disk-prerequisite failure
([failure](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/campaign-inventory-lifetime-controls313-_jtmwdr5/qualification-failure-closure.json)).
Verified PAX preservation of two completed task-owned scratches recovered
54,296,576 allocated bytes before the unchanged retry
([preservation](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/campaign-inventory-scratch-preservation-m0sq7deb/preservation-retirement-closure.json)).

Original **45+5 optional cases / Main S=4,621**, inputs, criteria, lock profiles,
4-GiB and 1,200-second limits remain unchanged. **Exact-head hosted RSS and full
campaign acceptance remain PENDING**; no benchmark or finite Omni repeat is
needed before that qualification. Public LEGACY/gate activation remains HOLD.

Previous published snapshot: `22a7b4d0` / [CI 576](https://github.com/enpasos/jax2onnx/actions/runs/41158051464)
had seven successful candidates and locked312's **4,499,320,832-byte** RSS failure
with unfinished Omni/cleanup
([accounting](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/hosted-ci576-locked312-failure-8eop7iow/locked312-actual-failure-accounting.json),
[resources](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/hosted-ci576-locked312-failure-8eop7iow/ci-candidate-locked312-work-resources__summary.json)).
Locked311 closed S=4,571 in an 806-second whole job
([finite qualification](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/hosted-ci576-locked311-success-7ddyjp6h/finite-locked311-qualification.json),
`4633f91e…`); its four original support jobs succeeded and four full suites were
still running at that snapshot's 02:21:45 UTC observation.

The reviewed local scheduler checkpoint `7b528095` gives the exact original Omni
case a fresh execution worker after its 44 predecessors. The batcher preserves
their order, the full
190-owner collection/context, all **45 MaxText + five MaxDiffusion cases** and
literal **S=4,621**. It adds one process boundary without changing model
parameters, inputs, RNG seed 42, numerical criteria or skip rules. Twelve
planner controls pass on official Python 3.11; they establish scheduling and
inventory preservation, not Omni memory fit
([focused closure](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/omni-isolation-planner311-retry-__sq33rd/closure.json)).
Its callbackless canonical singleton diagnostic **FAILED**: a fresh complete
**O=4,850** closed, and the singleton validated the same owner collection, but
only Omni setup passed. Sampled aggregate owned RSS reached
**4,230.984375 MiB > 4,096 MiB**. The measured command entry through postflight
was **554.060 seconds**; worker/JUnit closure is incomplete and observer cleanup
is **UNCONFIRMED**. A later passive PID/start check found no live owned process;
that does not upgrade the failed cleanup receipt
([failure closure](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/omni-isolation-canonical7b528-g_k84sts/primary-finite-failure-closure.json),
`5d7af940…`). Singleton isolation remains **UNQUALIFIED**. The finite diagnostic
omitted global C and establishes neither the campaign partition nor full S.

The additive observer update retains each already sampled process's
PID/start/parent/RSS/state vector and the vector/time/stage at the aggregate
peak. Sampling, limits, the 16-MiB sample-evidence cap and cleanup are unchanged.
All **29 observer-owner cases PASS** on official Python 3.11
([qualification](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/resource-process-vector-observer311-sw63cksp/qualification-closure.json),
`ff9e4dbc…`). This qualifies diagnostic serialization, not an RSS repair.
The subsequent canonical attribution at clean local `d02cec65` also
**FAILED/censored**: fresh **O=4,850** closed, but the original Omni case had
only setup PASS. At the simultaneous aggregate peak of **4,103.277 MiB**, the
worker held **4,043.445 MiB** and the observer/driver held **59.832 MiB**.
Observer cleanup remains **UNCONFIRMED**; passive owned PID/start identities
were later absent and source/output restoration held. The censored trace does
not establish the complete model peak or that the worker alone would finish
below 4 GiB
([failure closure](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/omni-peak-attribution-d02cec65-it219t20/peak-attribution-failure-closure.json),
`6ae33866…`).

The initial local checkpoint `b6ce1fba` waits for the resolved MaxText
`DenseGeneral` constructor only inside `get_maxtext_model`'s `from_config`
scope. It calls the original constructor once, then uses public `nnx.state` and
`jax.block_until_ready`; any tracer leaf bypasses the whole wait. The exact
constructor is restored on normal, exceptional and nested exits. Initializer
math, parameter values/dtypes/shapes, arguments, inputs and RNG consumption are
unchanged; no state is retained and forward tracing gains no readiness barrier.
All **36 controls PASS on official311**
([closure](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/maxtext-ready-controls311-hilb8lnz/qualification-closure.json),
`b824b815…`) and **36 PASS on official313**
([closure](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/maxtext-ready-controls313-1_6f_w2z/qualification-closure.json),
`8b328bcb…`). The mandatory tiny real `DenseGeneral`/initializer probe also
**PASSES on the unchanged full optional312 profile with zero skips**
([closure](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/maxtext-ready-real312-kpgomylk/qualification-closure.json),
`a27dc0c4…`). These are finite byte/dtype/shape/signed-zero, RNG, identity,
tracing and restoration checks.

Before any canonical model run at `b6ce1fba`, the pinned-source audit found that
its uppercase-first resolver did not establish binding to AudioEncoder's
lowercase `maxtext` imports. Case-varied paths on D share filesystem identities;
distinct Python classes were possible, not measured. The correction prefers
`maxtext.layers.linears.DenseGeneral` with the existing uppercase fallback,
without rewriting package aliases. The audio constructor builds all **32
layers**, with **1,280 model / 5,120 FFN dimensions** unchanged; its direct
projections, attention and MLP use this class. Layer norms and three convolutions
remain outside the candidate's boundary
([source finding](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/maxtext-ready-canonical-class-source-x5mzyw1y/source-finding.json),
`8eb934d9…`). All **37 controls PASS on official311/313**, including the
distinct-module/class regression
([311](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/maxtext-canonical-class-controls311-idwoc675/qualification-closure.json),
`a4f1dc62…`;
[313](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/maxtext-canonical-class-controls313-crnbehx3/qualification-closure.json),
`2e4a89ae…`). The strengthened real312 probe **PASSES with zero skips**, binding
the selected class and wrapped constructor to the actual audio, attention and
MLP references, plus the original initializer/state/parameter/RNG assertions
([closure](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/maxtext-canonical-class-real312-35qqbudv/qualification-closure.json),
`b65ae1bd…`). No canonical model ran at `b6ce1fba`.

At actual clean corrected `98a34705`, the original canonical Omni case now
**PASSES finite resource qualification**: fresh 190-owner **O=4,850** closes,
the worker validates the same O, and original setup/call/teardown plus
**JUnit 1 PASS / zero skips** close. Sampled aggregate owned RSS peaks at
**4,248,961,024 bytes / 4,052.125 MiB < 4,096 MiB**, leaving about **43.9 MiB**
sampled headroom. Work takes **628.533639 seconds**; actual command entry through
postflight takes **737.830522 seconds < 1,200**. A separate zero-native launch
preflight failure retains its measured **0.95722525 seconds**; no retry of native
work occurred. Resource/transport closure and cleanup pass, recorded owned
PID/start identities are absent, and source/runtime/generated bytes/modes,
prior outputs and the original ONNX link target are preserved
([qualification closure](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/omni-construction-ready-canonical98a34705-j4ehlypc/canonical-omni-qualification-closure.json),
`5d57d182…`). The written **688,657-byte** model is byte-identical to the prior
retained artifact (`c244349d…`); this run adds no numerical validation or checker
execution. Actual asynchronous backlog was not measured. This closes one full
original case, with its original 32 audio layers/parameters/inputs/RNG 42 and
criteria. **All 45+5 optional cases, full S, global C/campaign partition and the
whole hosted job remain PENDING**; the finite headroom is not a complete-cohort
resource result.
The **1,200-second whole-job / 4-GiB / 160-job-minute / 100-MiB** limits remain
unchanged, as do the original profiles, coverage and all 50 optional-case
obligations. Complete exact-head cohort validation remains required. Public
LEGACY and gate activation remain **HOLD**; this checkpoint has not passed
complete-cohort CI qualification and has not been pushed.

Previous published snapshot: `b39e8662`, [CI 575](https://github.com/enpasos/jax2onnx/actions/runs/41154521659),
has terminal candidate outcomes of **six SUCCESS, locked312 FAILURE from the
owned-RSS guard, and locked311 CANCELLED after work/pack deadlines**. The four
authoritative support jobs succeeded; the four authoritative full-suite jobs
remain pending in this snapshot. All-eight STANDARD qualification is **FAILED**.

This repair revision configures only locked311 and locked312 as explicitly named,
temporary Main-owner qualification candidates, using the existing provisioning
route and the [reviewed C/S/O/A contract](ci_standard/README.md#reviewed-main-owner-variant).
The eight authoritative jobs, other six candidates, complete Main coverage,
original inputs/skip rules and all resource/artifact limits remain unchanged.
Owner acceptance and authoritative/public SOURCE activation remain **HOLD**;
no time or memory saving is established by this configuration.

The [locked312 evidence](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/hosted-ci575-b39-observation-95edj9mj/locked312-extracted-evidence-summary.json)
records 4,326.086 MiB owned RSS and incomplete Omni/cleanup closure. The
[locked311 evidence](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/hosted-ci575-b39-observation-95edj9mj/locked311-actual-failure.json)
records a 1,227-second whole job with an unfinished final batch, below the RSS cap.

Previous published snapshot: `a0debb7f` [CI 574](https://github.com/enpasos/jax2onnx/actions/runs/41101017340) is terminal with **11 SUCCESS and five FAILURE** ([retained final record](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/hosted-ci574-a0de-7271vsuc/final-terminal-205648/closed-final-ci574.json)). Seven candidates succeed, including locked311/313/314 in **1,041 / 988 / 1,096 s**; locked312's **4,298.11-MiB** owned-RSS breach and incomplete worker/cleanup leave all-eight qualification **FAILED**. Candidate totals remain **4,611 s / 76.85 job-minutes** and **42,692,635 successful-archive bytes**. The coordinator cleanup's earlier **24 controller passes on each official 3.11/3.13 family** and approximately 35-MiB reachable-object census remain [finite evidence](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/hosted-ci574-a0de-7271vsuc/coordinator-lifetime-source-qmrp_dyv/closed-focused-qualification.json), not measured RSS savings or a canonical memory repair.

The latest local locked312 Main-owner campaign at committed `b39e8662`
**FAILED/incomplete at its 1,100-second work deadline**. The complete launcher
returned exit 1 after **1,136.576 s**: C=19,533, O=4,850, S=4,621 and A=14,912;
only **10 S cases closed PASS**. Eight more passed phase triplets and a ninth
setup remain in an interrupted worker without terminal closure. All original
45+5 optional cases were unexecuted. Source/process/artifact preservation closed;
this is not a successful STANDARD run or a measured memory repair
([closed attempt](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/canonical-owner312-final-08w09nhf/external-launcher-closure.json)).

The earlier full-global `b63ee7f4` attempt remains a separate deadline failure:
98 closed PASS, 4,523 unexecuted and all original 45+5 unexecuted
([earlier closure](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/canonical-standard-b63-locked312-sourceplan-7yx7JTF1/closed-canonical-standard312.json)).
The committed owner tooling retains its accepted finite official311/313 protocol
controls, separate opt-in controls and strict docs/normal-hook checkpoint. Scan54
is qualified only for its selected official311 cases. Full current cohort S,
production context, remaining Scan/SOURCE and complete B–F/resource gates stay
open as recorded in the [handoff](implementation_handoff.md#current-checkpoint-2026-10-30).
Public LEGACY and the authoritative checks remain unchanged; **activation is HOLD**.

Predecessor status snapshot — 2026-10-30: Published `c5d60d4c` [CI 573](https://github.com/enpasos/jax2onnx/actions/runs/41076579765) has **10 SUCCESS, three FAILURE, one CANCELLED and two old full jobs in progress** in the [frozen job snapshot](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/hosted-ci573-c5d60-aw64dezy/jobs-final-repair-prep.json) (`3856753b…`). Old 3.13/3.14 fail the same stale negative-test injection; old 3.11/3.12 are still running. Six of eight candidates succeed; locked311's **1,227-second** whole-job span and locked312's owned-RSS guard stop leave all-eight qualification failed. The eight candidate spans total **4,301 s / 71.68 job-minutes**, and the six successful archives total **30,775,351 bytes**; these partial successes do not satisfy all-eight acceptance ([frozen budgets](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/hosted-ci573-c5d60-aw64dezy/candidate-terminal-budget.json), `6027acf6…`). CI 571/572 failures and their source repairs remain historical evidence below and in the handoff.

The original Omni case separately passes an export with an external CALL profiler in a different import/session context; its retained artifact independently passes the full ONNX checker ([checker closure](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/omni-checker-only-nxi1sakk/run/closed-checker-only.json), `9080c14a…`). Numerics, canonical memory repair, the complete original 45+5 and full S remain unqualified. The `1421bd8d`/`ff52d925` repair qualification is finite: the unchanged final **945 Main cases / 2,835 phases PASS** in the two changed packets at **753.21 s / 3,825.53 MiB** ([closure](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/ci311-classpack-qual-aofrkz2x/tail-witness/closed-original-945-tail.json), `7d18902f…`), while full S remains **FAIL/incomplete**, 3,626 cases are unexecuted and all 12 native owners remain **UNVERIFIED**. The negative-test-only replay repair separately passes all 98 owner cases on both official framework families ([closure](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/ci314-raw-bridge-repair-GyA0d5/closed-qualification.json), `6a3fa1ce…`). These repairs are absent from the frozen `c5d60d4c` snapshot; no whole-job saving or canonical memory repair is inferred. The unchanged 1,200-second, 4-GiB, 160-job-minute and 100-MiB requirements, full B–F and gate activation remain open.

## Binding user clarification — 2026-10-29

> die Standardprüfung sollte den Status Quo wie auf main prüfen, damit wir nichts gegenüber diesem Stand verlieren, aber auch nicht länger laufen. Zusatzprüfungen wären on top und dann auch gern extra.

The latest user amendment replaces relative runtime comparison with an absolute
limit: **every successful Linux STANDARD job must finish within 1,200 seconds
(20 minutes)**. Measure from the actual job start through checkout, setup,
generation, collection, execution, aggregation, upload and final cleanup;
external queue time is excluded. Timeout, interruption, a guard stop or incomplete
evidence fails qualification even if the observed prefix is shorter than 20 minutes.

Preserve main's complete existing coverage. Every added SOURCE, migration or
regression check belongs to a separately identified extra lane, complete and
disjoint from standard. Those checks remain required at their assigned trigger;
separating them does not delete, weaken or waive them. Extra cost remains visible.
Keep the activated layout bounded to **four Python cohorts plus four support
jobs**. The temporary qualification overlap below is a separate, one-milestone
amendment. Sequential canonical packets are work inside those jobs, not permission
to multiply hosted jobs or apply the limit separately to each packet/process.

This amendment supersedes relative NoIncrease, paired main/PR timing campaigns,
median/worst comparative gates and the mandatory original two-worker main
benchmark. No new main benchmark or replacement diagnostic is required.
Historical comparative measurements and failed attempts remain labeled evidence;
they are not active acceptance prescriptions. Existing ordinary CI and runners
remain unchanged until separately reviewed activation.

This clarification supersedes the previously accepted `C(p) - H(p)` routine
selection and its **35-minute / 160-minute standard runtime budgets**. Neither
selecting nearly the entire expanded PR suite nor reducing standard coverage to
two representative Python cohorts satisfies the new instruction. Acceptance of
the earlier `C(p) - H(p)` and comparative architecture is historical;
it does not activate the new plan. Independent Astra review accepts the narrow
absolute-20-minute amendment and diagnostic retirement; execution qualification
remains separate. No standard/extra workflow gate or required check is activated.

Independent Astra Ultra review on 2026-10-29 by
`/root/astra_ci_plan_review/main_baseline_architecture_review`: **ACCEPT for
architecture**, including the subsequent one-to-one/phase-F clarification.
The reviewed candidate permits incremental implementation. Literal selection,
main-to-PR contract mapping, collection-context equivalence, measured whole-job
20-minute runtime and all gate-activation evidence remain unqualified. This review does
not accept missing migration verifiers or claim performance results.

## Reviewed Main-owner collection amendment

The [reviewed v2 amendment](/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/standard-owner-scope-amendment-v2-3mme80px/AMENDMENT.md)
permits a bounded extension of the existing runner. It preserves every Main
counterpart and changes worker collection only through explicit owner authority:

| Inventory | Required meaning |
| --- | --- |
| C | Genuine complete global PR collection for this exact cohort and campaign. |
| S | Literal total one-to-one Main mapping: 4,571 / 4,621 / 4,573 / 4,573 on locked311/312/313/314. |
| O | Genuine complete ordinary collection of approved ordered Main-owner modules, including their added cases. |
| A = C − S | All extras, including O − S and C − O, assigned to executable lanes and triggers. |

Discover C once inside each fresh campaign; there is no cross-run C cache.
Every owner worker collects complete O with normal hooks and actual Items, then
selects its assigned S. O is never labeled C. Reconcile its ordered
nodeid/module/class inventory against C; those fields do not replace source,
parameter, callable, input or numerical-contract review. Preserve the original
45 MaxText and five MaxDiffusion obligations and each cohort's own context.

The [provisioning-only route](ci_standard/README.md#reviewed-main-owner-variant)
binds separate current-RUN C/O receipts, explicit scope, exact source/generated
bytes, packages/plugins, optional inputs and the existing context transitions.
Global/support routes retain their stricter original contracts. Terminal hooks,
native phases, artifact ownership and final persisted-byte audits remain required.
All discovery and collection costs count toward the same whole-job limits;
this amendment claims no memory saving or Omni fit.

Source approval and finite controls do not activate the variant. Current
collector/import/registry qualification, full per-cohort S, support/integration
and complete extra coverage, exact-final supported-cohort suites, quality checks
and measured budgets remain required before independent acceptance and explicit
workflow activation. The existing eight plus temporary eight layout is unchanged.

## Earlier private qualification — 2026-10-30

In [CI 571](https://github.com/enpasos/jax2onnx/actions/runs/41060058197), the
same-head candidate normalization312/313, minimum312 and Node312 jobs pass in
**65 / 74 / 134 / 125 s**, totaling **398 job-seconds / 2,506,645 artifact bytes**.
This qualifies those four support observations only, not all-eight acceptance.
The locked312 worker stops after 79 of 176 assigned cases in batch 0003 when
MaxText initialization leaves its HLO-dump `XLA_FLAGS` in the process environment.
Locked311/313/314 instead reject ten original missing-npm call skips; their
exact native phases, source bytes, lines and reasons agree. Later assigned
cases remain unexecuted, and all four actual global results remain failed.

The four global profiles now restore only those ten omitted conditional
allowances. Owner bytes and npm setup match frozen Main, whose actual hosted
global jobs recorded 12/11/11/11 aggregate skips; historical per-ID skip phases
remain inferred, not observed. The unchanged dedicated Node lane executes all
ten with zero allowances. Source repairs do not waive context checks, change
inputs or comparisons, establish numerical acceptance, or qualify a new head.

The [private support variant](ci_standard/README.md#private-support-variant)
retains exact original selectors, installation maps and first-collection
provisioning. At committed `c338c33f`, normalization312 and normalization313 each
pass **C=S=102 / 306 phases**; minimum312 passes **C=615 / S=475 / 1,425 phases**
with **A=140 UNEXECUTED**. All three have zero skips/raw `wasxfail`, 18 actual
environment/JAX comparisons and complete local closures. Astra verified all 384
frozen files (combined summary `52cea604…`). Native artifact transport remains
`SUBSET_TRANSPORT_ONLY`. The retained Linux/D-mode inspector failure did not
repeat native execution. This local SHA qualification supplies no GitHub event,
hosted job budget or full SOURCE/B–F acceptance.

The [private Node binding](ci_standard/README.md#private-portable-node-binding)
has reviewed source and ordinary312 synthetic controls: 53 focused and 494 full
owner controls pass, no top-level skip/xfail, **1,006.50 s / 342.02 MiB** outer
cost. This binds `38dfb1ff` plus five-file patch `010719c8…`, before new docs/data
(receipt `24a0c16b…`, freeze `f04a9ee3…`). The reviewed cohort/baseline and
unchanged reference `47cb4f12…` are committed at `9fb5bb68`. At that exact head,
local canonical admission passes **C=S=50 / A=0, 150 phases and 50 JUnit cases**,
with six actual Node/environment/JAX observations, the new native-evidence key,
and closed source/package/resource/retention checks. The controlled sequence
costs **809.36 s / 1,010.27 MiB** (summary `9b75973c…`, independent acceptance
`fb1f24bf…`). This excludes earlier tool installation and proves no hosted
whole-job budget or complete SOURCE/B–F acceptance; not all fifty owners are
numerical tests. Earlier raw Node50/150 phases and failed capture/diagnostic
evidence remain distinct. Never refresh expected runtime hashes from the host.

Earlier portable controls/render-only/resource qualification remains in its
recorded epochs (`237b62f3…`, `4b5560aa…`, `931cafa8…`); the 447-case support
controls retain genuine unsupported-role rejection and their failed bookkeeping
record (`2635bb2d…`). The [handoff](implementation_handoff.md#current-checkpoint-2026-10-30)
retains the earlier raw observations and unresolved numerical failures.

Published `234df50e` has **all eight SUCCESS**, including all exact-head checkout
steps, in [CI 569](https://github.com/enpasos/jax2onnx/actions/runs/41024691391)
at **11:43:05 UTC on 30 October**. Full 3.11/3.12/3.13/3.14 spans are
**11,024 / 12,556 / 10,263 / 10,169 s**, aggregate **44,327 job-seconds** for all
eight. Predecessor `10c46eb2` also has all eight success in
[CI 568](https://github.com/enpasos/jax2onnx/actions/runs/41014331472), terminal
09:24:47 UTC. These old mixed-suite successes do not meet the 20-minute target
or constitute a numerical repair.

Frozen main remains `81ad6ca`. Earlier published `cd5cb4be` has **all eight SUCCESS** in
[CI 40964061166](https://github.com/enpasos/jax2onnx/actions/runs/40964061166),
terminal **01:05:49 UTC on 30 October**. Its old mixed-suite full-job spans are
**9,525–10,394 s**, support **54–137 s**, aggregate **40,191 job-seconds
(11h09m51s)**. This is neither 20-minute qualification nor a numerical repair.
`e8bed493` and `450732ad` each retain seven successes and a Python 3.11 native
BF16 TopK failure before export, with unchanged mathematics and unresolved
cause; AMD success does not prove causation.

The optional strict campaign contract and per-audit exact-path memo remain
private and unused by ordinary CI. Every role check and source/package/context
checkpoint stays intact; no memo survives an audit. Complete owners pass
**336 cases / 1,008 phases each** on official Python 3.11
(179.61 s / 325.57 MiB) and 3.14 (214.35 s / 359.05 MiB), including all opt-in
synthetic xdist controls. Current PR312 has **C=18,862 / S=4,621 / A=14,241**,
all original **45+5 optional cases** in S, and only the exact Main312 scan
setup-skip/teardown-pass allowance at line 17. Main's passing Keras case supplies
no allowance. Unprofiled collection is **218.07 s / 1,189.66 MiB**; the original
first ten Eqx cases pass 30 native phases and ten JUnit cases under a
**220.11 s / 1,315.77 MiB** guard. Accepted freezes are `b1f8ad67…`,
`8c6c2e3d…` and `4af8b725…`, linked in the [handoff](implementation_handoff.md#current-checkpoint-2026-10-30).
FULL312 was not launched; these costs do not establish a viable 1,200-second
FULL job. No environment changed: cached home312 has 170 versus qualified 156
distributions (27 missing, 15 different, 41 extra), with no qualified safe disk
headroom for replacement. Other cohorts/support, portable manifests, resource
ownership and full B–F remain open.

## Temporary transition amendment — 2026-10-30

Astra accepts **one PR qualification milestone with at most eight fixed candidate
jobs alongside the untouched authoritative old eight**: at most sixteen jobs in
that temporary overlap. Candidates use the existing runner and complete main S,
original inputs/assertions and all 45+5 Python 3.12 optional cases. The fixed
roles are four locked Python 3.11–3.14 jobs, minimum Python 3.12, normalization
Python 3.12 and 3.13, and Node 22 smoke on Python 3.12. Retain minimum Python
3.11 as additional assigned coverage; it is not dropped.

Every successful candidate's **whole completed job span is at most 1,200 s**,
including checkout/setup, generation, collection, validation, aggregation,
uploads and cleanup, excluding queue time. Timeout, interruption or incomplete
evidence fails. Preserve the existing aggregate memory/headroom and artifact
budgets below. Candidate aggregate cost is at most **160 job-minutes**, with
actual legacy-job and failed-attempt costs reported in addition. Do not shard,
apply the budget per packet, revive main-comparison benchmarks, dispatch runs
or introduce a new controller.

Support reuse requires the **same head, complete context/ownership and resource
evidence**. Current telemetry is incomplete, so reuse is not assumed. All old
eight outcomes remain required on the qualification head. Independently review
the concrete workflow, complete mapping and actual results before activation;
retire the candidates at activation and verify the activated head.

The concrete V2 wiring and failure-retention paths are independently accepted
(workflow patch `e8778409…`, review `95defced…`). Exactly eight fixed candidates
are appended to the same workflow for PR #277; the eight old jobs stay
unchanged and authoritative. Each candidate uses the original setup and first
complete canonical C→S, with private Linux temporary capture, complete retained
output inventories, separate failed model/temp payloads and raw upload fallback.
Successful receipts/logs retain seven days and share the **100-MiB aggregate**
compressed-artifact budget; failure/retry payloads and old-job costs are separate.
This is temporary qualification wiring, **not coordinated standard/extra gate
activation**. Successful same-head sixteen-job qualification and complete
candidate timing/resource/artifact evidence remain open. Before retiring old jobs,
assign minimum311 additional coverage, minimum312 **A=140** and every other
extra/integration/quality obligation to explicit executable owners. Public
LEGACY behavior and all full B–F/final acceptance obligations remain unchanged.

## Historical checkout and CI observations — 2026-10-29

The reference main commit, supplied by root's fresh remote/local observation, is
`81ad6ca2592635c8d2ced6dbd77292c487f0351b`. Freeze this SHA as the initial baseline;
record and independently reconcile later main changes instead of silently
rebasing or shrinking the baseline. At **16:58:48 UTC on 2026-10-29**, published
PR head `14439a075f18979510ba4a0e6e97fb7342e6c800` has four successful support
jobs and four full Python jobs still in progress in
[CI 40910187730](https://github.com/enpasos/jax2onnx/actions/runs/40910187730).
This is run-association evidence. The one inspected terminal normalization
Python 3.12 job checks out merge commit
`ea7579bb658e4b8b5a6703d06ad81d7d8b8ac55f`, with parents frozen main `81ad6ca`
and head `14439a07`. Its tree `73d9e6290dda54c6ed85559ba8743ba83e7a5aa1` is
identical to the head tree; preserve its genuine same-tree pass without calling
it a literal head checkout. No assertion about the other seven job checkouts
follows. Evidence:
`/tmp/jax2onnx-exact-head-checkout-audit-20261029-334e87_v/files.json`
(SHA256 `58c175305849816afcdc83d3b0e69d1f7149e872c305788f581c01d92862b02b`).

The proposed repair pins each of the four checkout blocks to
`${{ github.event.pull_request.head.sha || github.sha }}` and immediately
verifies `HEAD^{commit}` against the expected peeled commit before setup.
Missing objects and mismatches fail the existing job; annotated tags remain
supported. `GITHUB_SHA` stays the event commit. Successor CI must prove the
actual checkout; check names, cohorts, selection and budgets are unchanged.

The earlier published-head `9e37792c` observation retains four successful
support jobs and four full Python jobs in progress in
[CI 40891049655](https://github.com/enpasos/jax2onnx/actions/runs/40891049655)
at **16:05:36 UTC on 2026-10-29**. Predecessor `9a0bc7fb` retains four successful support
jobs and a successful full Python 3.11 job, with the other three full jobs running in
[CI 40875416260](https://github.com/enpasos/jax2onnx/actions/runs/40875416260).
Its independently accepted storage/Linen repair
passes the four complete related owners on every supported Python cohort,
all seven normal pinned hooks and strict documentation. Predecessor `4ec9fad4`
retains four successful support jobs and all four failed Python full jobs in
[CI 40847090831](https://github.com/enpasos/jax2onnx/actions/runs/40847090831);
every failure is the NumPy diagnostic mismatch repaired by `9a`. Predecessor `afa6d16e` had four successful support
jobs and four failed full Python jobs in
[CI 40837446240](https://github.com/enpasos/jax2onnx/actions/runs/40837446240):
all four stopped at the original NNX Transformer static-preparation defect;
later assigned cases were unexecuted. Historical head `0a9638d` had eight
successful ordinary checks. These results do not qualify the current worktree,
the new selection or any performance budget.

Use supported official packages. Main's modern cohort used JAX/JAXLIB 0.11.2;
the current PR locks use 0.12.0.1 there. Preserve each revision's own lock
versions and markers in isolated installations. No custom native builds,
compiler patches, overlays or substitutions are authorized. The complete B–F
contract and exact-final-commit acceptance remain unchanged.

## Historical bounded qualification — 2026-10-29

These source-scoped records remain intact. Their earlier relative NoIncrease or
paired-comparison conclusions describe the superseded timing contract. Workload
measurements such as 854.56 s and 1,120.59 s do not retroactively prove the new
checkout-through-cleanup whole-job limit.

### Retired frozen-main comparative diagnostic

The non-required PR #277 `main-baseline-feasibility.yml` diagnostic and its
observer, authority JSON and component owner are retired. They have no remaining
code consumers outside that family and none existed in frozen main `81ad6ca`.
Their removal changes no ordinary CI job, main test, converter or existing runner.
The earlier one-eligible-transition proposal is historical, not a remaining gate
or an authorization to rerun the original two-worker benchmark.

The bounded component owner passed **77 tests on each official Python 3.11/3.14
family**, including a two-worker non-native pytest fixture and stdlib ownership
controls. Its guards measured 2.28 s / 213.00 MiB and 2.89 s / 246.95 MiB.
Candidate telemetry was stubbed inside those fixtures; they did not qualify real
hosted safety or main/model execution. Preserve those receipts at
`/mnt/d/jax2onnx-hosted-observer-v3-corrected-py311-gtpovn31/` and
`/mnt/d/jax2onnx-hosted-observer-v3-corrected-py314-8tpkc9vb/`, together with the
first **76 PASS / 1 FAIL** toy skip-line predecessor at
`/mnt/d/jax2onnx-hosted-observer-v3-focused-py311-a9gfz7di/`.

The actual hosted [run 40946430156](https://github.com/enpasos/jax2onnx/actions/runs/40946430156)
at driver `e8bed493` and frozen subject `81ad6ca` remains
**FAILED_OR_INCOMPLETE**. Setup passed in **32.92 s**. The workload guard stopped
at **169.80 s / 4,103.03 MiB**, with **817 complete passes and two setup-only
outcomes**. All 4,571 collected identities are not completed outcomes; final
worker closures/controller completion are absent. Preserve the scoped archive
and `/tmp/jax2onnx-hosted-main-e8bed-failure-20261029-eq16fy64/diagnosis.json`
(SHA256 `52b5067cb7f7edec1d93fe913810cd7e2cdfac92ee90e5b81e9437b698f23d95`).
This failure does not qualify the new whole-job 20-minute rule or direct xdist.
It must not be rerun or given a larger cap merely for comparative timing.

The earlier local original-main pilot also remains
**RESOURCE_FAILED/INCOMPLETE**, **52.26 s / 4,103.35 MiB**, with no terminal
JSON/JUnit. Its separate prelaunch bookkeeping ERROR executed zero native tests.
Keep `/mnt/d/jax2onnx-main-original-two-worker-pilot-7rm_yxaa/` and all earlier
source/package/swap/restoration evidence. The retired sources remain in commit
history and external source freezes; their prior design limits are historical.

Astra independently accepts the finite mapped-main-owner Python 3.11 campaign
at clean `14439a075f18979510ba4a0e6e97fb7342e6c800`: **4,569 PASS plus two
exact original main skips**, all 4,571 IDs in order, all fourteen actual
assignments, fifteen closed workers, 45 context snapshots and 13,712 phase
reports, with no `wasxfail` fields. The unchanged guard completes in
**854.56 s / 3,229.37 MiB**, without a trigger. Receipt:
`/tmp/jax2onnx-owner188-checkpoint14439a07-full-review-receipt-v2.json`
(SHA256 `140921bdca6e7aa8e4d9faf3990d5edd79da7fce6d98da8555d988a1bd1bc0e5`).
Its **8,569,970-byte** archive covers canonical/qualification receipts, guards
and metadata, not all model artifacts. This closes the next finite owner
qualification at this checkpoint only; whole-job timing, all cohorts/support,
Full44, full B–F, literal exact-head CI and gate activation remain open.

An independently accepted **temporary candidate**, using existing
`collection_args` and canonical lifecycle checks, collects all 188 mapped main
owner modules. At `9e37792c`, separate phase-free controls observe global
`C=18,689`, `S=4,571`, `A=14,118` and complete owner inventory of 4,796
(`O` in the current amendment), including all
225 additions inside those owners. Both collection costs are charged:
23.267989 s globally and 8.079359 s for owners. Three declared context boundaries
match the fresh main control except the bound artifact path. A separate staged
collection plus the original first ten tests passes all 30 native phases.
Evidence: `/tmp/jax2onnx-owner188-current9e-prepare-review-receipt-v1.json` and
`/tmp/jax2onnx-owner188-current9e-qualify-review-receipt-v1.json`.
This is finite collection/execution evidence; dynamic import-order independence,
the remaining standard cases, other cohorts and comparable runtime are still
unqualified. The current full-C implementation and CI gates remain in place.

The current CSE capture repair and original Issue52 outcome are recorded in the
[handoff](implementation_handoff.md).
Its full binding scans are batched without changing the CSE algorithm or
post-pass validation. The complete original Issue52 payload and all six affected
complete ViT cases pass on each official framework family. Independent review
accepts this finite checkpoint; residual membership costs, Full44 and broader
performance gates remain open. The current finite mapped-owner campaign above
supersedes the next-qualification step for Python 3.11 at `14439a07` only. It
does not establish the whole-job 20-minute limit or authorize gate activation
or replay feature expansion. The former matched timing campaign is superseded.

The complete locked Python 3.11 standard campaign at old head `9520dfb9`
finishes the exact original 4,571-case order: **4,569 PASS and two unchanged
main skips**. Both skip counterparts reproduce on frozen main with its own
lock. All fifteen closed workers retain full `C=18,647`, exact assignments,
source/package bindings and startup, collected and post-pytest context.
Astra independently accepts this finite functional/resource evidence:
1,120.59 s, 3,237.78 MiB and at most four owned processes, within the unchanged
1,200-s/4-GiB guard. Evidence:
`/tmp/jax2onnx-canonical-standard311-full-v4-iykuhoky` and
`/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/canonical-standard-full311-v4-20261029-qualification/full-summary.json`.
The campaign does not qualify a later integrated head, the other cohorts,
SOURCE numerical behavior, comparable runtime or gate activation.

A separate three-arm tail trial preserves all original identities and
compares the existing 246/332 packets with their exact 578-case concatenation.
Child elapsed time is 145.946805 s separately and 125.256561 s combined.
The combined peak is 3,865.64 MiB, only 230.36 MiB below the cap. Astra accepts
the literal outcomes, context and evidence comparisons; one fixed-order trial
does not establish general savings or safe scheduling. **Keep the existing
split and default planner.** Evidence:
`/tmp/jax2onnx-standard-tail-boundary-comparison-v2.json`. The initial control13
caller omitted the explicit suite environment and failed precheck without
starting a worker; that failed predecessor remains recorded.

The former paired main-versus-PR timing prescription is **SUPERSEDED**.
It required matched filesystem/cache/setup contexts and at least three
alternating campaigns with median/worst critical-path and aggregate comparisons.
Those comparisons were not qualified and are no longer a required gate. Keep
historical measurements tied to their original epochs; do not schedule paired
campaigns or an original two-worker benchmark to satisfy the active rule.
Whole-job 1,200-second qualification remains pending under the preserved coverage
and resource requirements.

## Accepted incremental work and measured starting point

The optional-model duplicate-removal edit retains pinned preparation, test
generation and the existing four-cohort full runner. Its fresh Python 3.12
50-identity/outcome qualification remains **UNVERIFIED**; the historical 3m19s
of duplicate execution is not a measured saving from the edit.

Astra independently accepted the additive phase-duration checkpoint. Timings
are separate from canonical phase outcomes; semantic comparisons and existing
checkpoint order are preserved. Supplied completed checks: Python 3.11 runner
100 PASS; Python 3.11 transport/control/replay 165 PASS; Python 3.14 runner plus
selected context/cache regressions 141 PASS, 68 deselected. This acceptance
survives the new lane design. It establishes no scheduling change or runtime
saving. Ordinary commits must invoke the installed hooks without bypass;
applicable hooks must pass and legitimate no-matching-file SKIPs are recorded.

The following completed GitHub job measurements are observational. Test counts,
scheduling and the modern JAX version differ; hardware was not experimentally
controlled. They are historical orientation, not a paired no-increase proof.

| Python | Main job | Published PR job | PR Original22 call time, all 44 cases |
| --- | --- | --- | --- |
| 3.11 | 11m22s | 2h16m51s | 65m05s |
| 3.12 | 18m04s | 2h25m12s | 58m30s |
| 3.13 | 9m31s | 2h11m51s | 66m47s |
| 3.14 | 10m15s | 1h38m09s | 47m29s |

Sources: [main 81ad6ca, CI 40641446266](https://github.com/enpasos/jax2onnx/actions/runs/40641446266)
and [published 0a9638d, CI 40686137924](https://github.com/enpasos/jax2onnx/actions/runs/40686137924).
All eight ordinary jobs passed in each run. Aggregate ordinary job time rose
from 54m30s to 8h37m08s, excluding failed custom-compiler builds. The old full
suite used two xdist workers. The PR used 54 serial processes per cohort, each
recollecting approximately 17,700 tests; summed startup-to-collection-log time
was 8m33s, 13m30s, 8m34s and 6m34s respectively. This is not pure tracing time.
One 467-item batch held all 44 Original22 cases and took 48–68m. Twelve batches
held one case each.

The four PR archives totaled approximately 13.3 GB, about 3.33 GB per cohort,
with 103–127s upload time each. MaxText's 45 and MaxDiffusion's five cases also
ran separately before the full suite, costing 3m19s on Python 3.12. All-files
pre-commit ran four times. Remove duplication only after preserving the affected
coverage, environment and outcome accounting.

## Standard ownership: main's full contract, executed on PR code

For each supported Python/dependency cohort `p`, define `M(p)` as the complete
ordered collection on the frozen main revision under its original official
locked setup, test generation, optional integrations and pytest configuration.
Record main's support-job commands, conditions, environments and outcomes too;
`M(p)` alone does not describe the whole standard campaign.

Collect `C(p)` on the PR with its own supported locked setup. Build a reviewed
total mapping `F(p)` from **every** main test identity/contract into the
corresponding PR test identity, retaining a distinct reviewed counterpart for
each original obligation. The standard pytest selection is the mapped baseline
`S(p)`. Additions are `A(p) = C(p) - S(p)`.
Missing, ambiguous, changed or unsupported baseline mappings block activation;
never implement this as the intersection of two node-ID lists. A main identity
must stay visible even when its equivalent has a different PR ID. Keep this
mapping one-to-one at the original parameter-contract level; do not collapse
several original obligations into one passing result. Counts cannot establish
equivalence.

The mapping must bind more than names:

- Preserve ordered node IDs, parametrizations, original callable/model identity,
  full inputs, dtypes/shapes, x64 and runtime settings, numerical/structural
  criteria, setup/call/teardown outcomes and exact collection skips.
- For handwritten tests, retain the original qualified function/class and the
  reviewed source/fixture/parameter changes. A file allowlist includes new
  functions in old files; a matching function or node ID can conceal changed
  inputs or weakened assertions. Neither is sufficient by itself.
- For generated tests, bind registry origin, raw case/variant and original
  metadata/factory/input/criterion contract to the actually collected PR item.
  Reuse the existing generator and inventory representations where applicable;
  generated display names and counts are not provenance. Missing optional
  registrations or generator name collisions remain explicit blockers.
- New cases or added migration/regression scenarios in existing files belong to
  extras. Where an old test now embeds separable new checks, move only those
  additions into a distinct extra owner while retaining every original baseline
  assertion. An inseparable changed owner needs review before lane assignment;
  it cannot silently broaden standard or weaken the baseline to fit its budget.

Execute the standard selection against **PR production code, PR tests/fixtures
and PR generated metadata**. The clean main checkout supplies the control and
provenance. Never copy main production or test files over the PR to manufacture
a baseline pass. Hash/source changes identify review work; a hash mismatch is
not by itself a lost capability, and a matching ID is not equivalence proof.

Reconcile JAX 0.11.2 versus 0.12.0.1 differences explicitly per cohort. Retain a
main-only ID as an unresolved obligation until a reviewed PR equivalent proves
its original contract, or a separately authorized support change resolves it.
New version-conditional PR cases belong to extras. Do not downgrade PR locks or
normalize removed/renamed cases into apparent agreement. Python 3.11–3.14,
main's actual minimum-dependency installation check, optional model ownership,
normalization gates and web smoke all retain their standard owners. Dependency-
sensitive checks retain main's cohort coverage unless consolidation is proven
equivalent; platform-independent checks may run once after that review.

Main's existing all-files lint, policy and typing checks still inspect the
current PR tree under their original contract. Do not freeze their inspected
file set to main or exclude newly added production code. Their expanded cost
counts in the complete standard job's 1,200-second accounting. Added pytest identities/scenarios and new
standalone checks are the extra work; this does not narrow existing whole-tree
quality checks.

Apply the same per-item ownership to pytest-based support jobs. A broad existing
file command must not silently absorb added tests. Preserve separate main job
contexts where the same baseline ID runs with different environments/settings.

The current public LEGACY default requires the same main-suite contracts. This
baseline does not freeze that policy permanently or cancel the coordinated
phase-F SOURCE transition. A legitimate version-sensitive or phase-F API change
needs an independently accepted one-to-one counterpart mapping that preserves
the original sources, full inputs, criteria and features. Retain each original
obligation and its evidence through the transition; do not keep a permanent
public LEGACY/SOURCE mixture or silently weaken an assertion to retain a name.

Use one compact per-cohort baseline/mapping artifact and a small literal lane
table in the existing runner. No service, registry framework, test DSL or second
executor is required. Cache immutable baseline collection evidence only by base
SHA, Python/cohort, lock/setup hashes, optional checkout/selection hashes and
collector version. PR mapping is bound separately to the exact PR SHA. Validate
artifact hashes on reuse; a changed input invalidates the relevant cached proof.
Package-download caches are distinct from evidence caches. Installed environment
caches must distinguish locked, optional and actual minimum-version installs.

## Concrete baseline collection and first implementation step

**Smallest next step: obtain and reconcile main/PR collection evidence, before
changing scheduling.** Root prepares separate disposable checkouts and isolated
official locked environments for each cohort, preserving each revision's own
workflow setup. Source trees remain unchanged except ordinary generated tests
and local ignored artifacts in disposable collection copies; the reference main
checkout remains an immutable evidence source. Main has no PR batch runner;
do not assume that entrypoint exists there or run PR's observer-bearing worker
against main.

After the original pinned optional setup for the appropriate cohort, the
existing commands in each prepared checkout are:

```sh
poetry run python scripts/generate_tests.py
poetry run pytest --collect-only -q
```

Capture command, stdout/stderr, exit status, actual resolved packages, source SHA,
lock/setup hashes, generation output and original pytest options in a fresh
external evidence directory for each revision/cohort. Use the same generation
and optional setup as that revision's ordinary workflow; do not use one PR
virtualenv for both checkouts. These commands collect, not execute, and cannot
prove assertions, runtime skips or collection-context equivalence. Repeat the
existing support selectors in their respective environment inventories.

If machine-readable records need a hook, make the next code patch a small
**collection-only observer in the existing runner**, reused as a pytest plugin
from an explicit reviewed file path. It must observe actual items and collection
reports, not enable cast/model observers, import a second converter, install
framework patches or execute selected tests. Assert that converter/test modules
resolve inside the intended checkout. This hook, its concrete invocation and
the main-to-PR mapping/verifier are **NOT IMPLEMENTED**; review them and their
small synthetic checks before use. Plain collection transcripts are initial
evidence, not a substitute for the eventual fail-closed selection verifier.

Publish exact main and PR manifests, support-command ownership and every mapping
exception before proposing a workflow diff. Final executable selection requires
literal resolved node IDs/contracts and the reviewed mapping artifact, not the
symbols `M`, `S`, `A`, file globs or placeholders in this document.

### Initial source-contract reconciliation — 2026-10-29

The initial modern source mapping received independent Astra Ultra review. All
3,124 generated main identities have same-ID counterparts with the original
criteria. The generator's ordinary path only adds a dictionary copy before
metadata enrichment. These eleven generated counterparts require explicit
annotations rather than an assertion of byte-identical source inputs:

- Deferred construction, preserving model arguments, seeds, full inputs and
  criteria: `tests/primitives/test_eqx.py::Test_conv::test_eqx_conv2d_nchw`,
  `Test_conv::test_eqx_conv2d_batched_nchw`,
  `Test_gru_cell::test_eqx_gru_cell_basic`,
  `Test_lstm_cell::test_eqx_lstm_cell_basic`,
  `Test_spectral_norm::test_eqx_spectral_norm_linear_inference`,
  `Test_weight_norm::test_eqx_weight_norm_linear_dynamic`,
  `Test_weight_norm::test_eqx_weight_norm_linear`, and
  `Test_weight_norm::test_eqx_weight_norm_conv2d` in that same file.
- Previously accepted deterministic Gaussian-fixture stabilization: full
  `tests/examples/test_gpt.py::Test_GPT_Attention::test_gpt_attention` and
  `tests/primitives/test_nnx.py::Test_dot_product_attention::test_dpa_with_causal_mask`
  with its `_f64` counterpart. Local seed-zero successive Gaussian draws replace
  global unseeded draws; shapes, dtypes, masks, callables and criteria remain.
  Input bytes deliberately change and are not described as identical.

Seven pre-existing handwritten bodies change. Both optimizer-failure tests in
`test_io_names.py` retain assertions while their stub accepts `precision_policy`.
Both requested-instance tests in `test_internal_improvements.py` retain original
instances/inputs/criteria and use actual lifetime registration keys. The minimum
dependency extraction test retains the extraction contract against current
declared floors and explicitly covers `jaxlib`. The Issue52 helper's native-schema
adaptation preserves its recorded payloads, inputs and criteria.

The other two bodies and one shared fixture required repairs before selection:

- The original Dropout baseline identity again owns the complete opset-21
  `(3,30) → Gemm → BatchNormalization → Dropout → Gelu → Gemm → (3,10)`
  corridor. Legal weights, BN parameters and a Constant tensor attribute repair
  the old invalid fixture. Removal of `Not`, exactly one Dropout, connectivity,
  ratio retention and deterministic-input pruning remain asserted. An omitted
  training-mode input is the reviewed semantic counterpart of explicit false.
  The smaller direct-Dropout example has a separate added identity.
- The baseline tile AD test keeps native forwarding or the official older-stack
  decomposition counterpart. Four hostile-binding scenarios now have a separate
  added identity, retaining restoration and rejection checks.
- The primitive-recording runtime-session prohibition applies only to added
  SOURCE tests, preserving the original baseline test's setup.

These repairs have independent bounded Astra ACCEPT and pass all **90 cases in
the three complete owners on each official framework family**. They change no
production code. Collection evidence and its exact source epoch are recorded in
the handoff; matching IDs alone still do not prove numerical, context or runtime
equivalence. Remaining cohorts, optional integrations, support-command ownership,
selector qualification and comparable performance remain open.

### Local preparation and unused selector qualification

The ordinary main inventories now exist on their own official locked setups:
Python 3.11 has 4,571 IDs; Python 3.12 with both pinned optional checkouts has
4,621, including all 45 MaxText and five MaxDiffusion IDs; Python 3.13 and 3.14
each have 4,573. The modern lists retain the two native `one_minus_square`
variants. Preserve actual final-session order: the existing JSON report's
collector-tree order is earlier than pytest's ordering hooks. The initial
Python 3.11 and 3.14 PR projections preserve every baseline ID and its relative
final-session order. This remains collection evidence, not execution acceptance.

Local Poetry 2.1.3 initially re-resolved the Python 3.12 setup to different
locked variants. That setup is **UNQUALIFIED**. The subsequent installation used
`POETRY_INSTALLER_RE_RESOLVE=false`, verified the target virtualenv before any
installation, and checked the actual marker-selected JAX/jaxlib 0.10.2, Flax
0.12.8, NumPy 2.4.6 and SciPy 1.17.1 versions plus `pip check`. This follows
[Poetry's documented marker-based installation](https://python-poetry.org/docs/configuration/#installerre-resolve).
Neither lock nor reference-runtime source was changed. Retain both setup
receipts; the initial install's exit zero is not cohort qualification.

The existing `scripts/run_pytest_batches.py` now contains an **unused**, explicit
pytest plugin candidate for standard/extra projection. Canonical script mode,
batch requests, observers and artifact transport remain unchanged. Activation
requires all three `--ci-lane`, `--ci-main-baseline` and `--ci-lane-report`
options; loading the plugin alone leaves ordinary pytest behavior intact.
Literal, separately qualified cohort/context manifests bind ordered baseline
IDs, actual source/fixture/helper/configuration files, environment and declared
package versions. Collection retains all modules; filtering occurs after
ordinary ordering and before xdist publishes worker IDs. Report directories are
fresh and external; worker inventories are separate from native phase outcomes.

Independent Astra review accepted this unused stage after two repairs: final
context auditing catches lazy imports, and the controller validates every final
worker receipt and propagates terminal failures, including after passing native
reports. **46 synthetic cases pass on each official framework family**, including
the bounded opt-in xdist witness and these regressions. Initial 39-pass/one-skip
and 40-pass runs did not exercise the subsequently identified defects and do not
replace the repaired qualification. Evidence is recorded in the handoff.
Durable CI manifests, support selectors, execution/context equivalence, CI
activation and runtime parity remain **UNVERIFIED**.
Subsequent temporary-manifest checks now establish complete/disjoint standard
and extra **collection** on every cohort, retaining exact ordered main IDs:
3.11 `18,498 = 4,571 + 13,927`; pinned-optional 3.12
`18,552 = 4,621 + 13,931`; 3.13 `18,503 = 4,573 + 13,930`; 3.14
`18,500 = 4,573 + 13,927`. Extra identities differ legitimately by native
typing/AxisData/context schema; preserve each literal cohort list. Receipts are
**COLLECTED**, with zero executed phases. This closes this finite collection
partition check, not durable-manifest acceptance, runtime/context equivalence,
support-job mapping, performance qualification or gate activation.

## Support-owner source mapping — 2026-10-29

The canonical-runner integration has independent Astra **architecture and
bounded static-context source ACCEPT**. Real-main context/runtime qualification
remains pending. It keeps the existing
two-key collection and five-key batch request schemas, carrying lane, manifest
path and expected raw SHA through private worker options. Collection retains
complete `C`; the coordinator partitions once, plans only the chosen ordered
lane and requires the flattened/completed assignments to match it exactly.
Workers use the existing assignment projection, shared qualification and
terminal source/context audits. The independent direct projector is not loaded
in canonical workers. Transport still receives full `C`; unassigned native
model obligations remain explicitly unverified.

Canonical qualification binds the actual child environment after transport
setup, including its run-owned native-artifact directory. Whole DINO owners
require prefix flushes to preserve baseline order; the noncontiguous 136-case
diagnostic is not an ordered campaign. Defaults, original fail-fast options,
observer controls and replay acceptance remain unchanged. No workflow, durable
manifest or gate activation is included in this stage.

The first implementation passes 198 existing/new runner and selector cases on
each family, but independent review found a collection-purity gap. A hook could
clear `collectonly` and execute cases during the collection child. Both new
counterexamples fail against that implementation because the inner processes
incorrectly return success:
`/tmp/jax2onnx-ci-collection-mode-predecessor-v1-py311-fzdman78`.
The amended source rejects collection execution before any test protocol and
independently rejects recorded collection phases, completions and timings at
worker, closed-receipt and coordinator boundaries. Direct collection mode is
also bound against late changes. Independent Astra review accepts v2, and
**203 tests pass on each family**; the two override witnesses and three
semantic-corruption cases are included. V2 evidence is
`/tmp/jax2onnx-ci-canonical-lanes-integration-v2-py311-b8dfuvy8` and
`/tmp/jax2onnx-ci-canonical-lanes-integration-v2-py314-tuwdqeh4`.
Existing artifact transport, observer-off control and frozen replay consumers
also pass **165 tests on each family** at v2. Evidence:
`/tmp/jax2onnx-ci-canonical-lanes-consumers-v2-py311-t_4ynj6b` and
`/tmp/jax2onnx-ci-canonical-lanes-consumers-v2-py314-dcoxcslu`.
Real ordered execution, resource qualification and paired performance remain
pending; the v1 passes do not qualify those gates.

Before the first full campaign, independent review rejects the temporary
helper's proposed startup defaults as standard-context evidence. Original 3.11
startup has `JAX_PLATFORMS` and `JAX_ENABLE_X64` unset; ordinary collection later
sets them to `cpu` and `True`. Predeclaring those values before JAX import can
change collection-time computation. The existing 136-case receipt records both
stages, including JAX initially unloaded. The complete observed transition also
includes `TF_CPP_MIN_LOG_LEVEL`; bind every observed field. Do not ignore fields or predeclare
them to make an audit pass. A reviewed explicit startup/post-collection binding
is pending; the temporary full-4571 helper has not run. Preserve unset optional
integration refs as unset, and distinguish current-CI `PYTEST_ADDOPTS` from saved
profiles that did not declare it. Direct pytest configuration does not prove
pre-import process startup; missing startup history remains unverified.

The bounded canonical integration is committed locally at `35f80783`, with all
seven actual pinned commit hooks passing (26.10 s, 453 MiB) and strict docs
passing (4.79 s, 138 MiB). It remains unpublished and does not activate a gate.
The next canonical-only amendment has independent Astra architecture ACCEPT:
an optional exact `collection_transition` binds startup JAX metadata plus the
complete collected environment and JAX metadata. Existing `environment` stays
the complete startup binding. Prove startup before importing pytest, require
the explicit collected stage before projection/execution, and recheck runtime,
terminal and closed-receipt context. Static defaults stay strict; direct/xdist
transition profiles fail explicitly because they lack canonical startup proof.
No field may be ignored, predeclared or rewritten. Both natural-transition counterexamples fail
against the preceding source with unsupported-manifest-schema diagnostics;
retain `/tmp/jax2onnx-ci-staged-context-predecessor-v1-py311-h5yqr43g`.

Independent Astra source review now accepts the frozen staged amendment in
`/tmp/jax2onnx-ci-staged-context-source-v1.json`. All **246 runner/selector tests
pass on each JAX family**, including the explicitly enabled synthetic xdist
cases: minimum 76.33/76.66 s, 308 MiB; modern 138.64/139.50 s, 347 MiB.
Evidence: `/tmp/jax2onnx-ci-staged-context-integration-v1-py311-6ijhcsgc` and
`/tmp/jax2onnx-ci-staged-context-integration-v1-py314-shxvrxdh`.
Existing artifact/control/replay consumers also pass **165 tests on each
family**: 28.84/29.39 s, 278 MiB and 48.66/49.68 s, 303 MiB; evidence
`/tmp/jax2onnx-ci-staged-context-consumers-v1-py311-yzxfbssx` and
`/tmp/jax2onnx-ci-staged-context-consumers-v1-py314-txi9dbr_`.
Protocol checks reuse bounded environment/JAX readers without adding per-case
CPU, Git or source hashing. These are bounded infrastructure results; original
main context, ordered full execution, resource/performance qualification and
CI activation remain open.

The independently reviewed passive frozen-main probe now passes in main's own
Python 3.11 lock: exact **4,571 original ordered IDs**, zero executed phases,
unchanged source/package epochs, 51.41 s and 601 MiB. Evidence:
`/tmp/jax2onnx-main311-passive-context-iy33qqup`; guard
`/tmp/jax2onnx-main311-passive-context-v1-zr_zxw83`. Startup JAX is NOT_LOADED;
collected JAX has x64 enabled with `jax_platforms=None`. The complete observed
environment transition also adds `PYTEST_VERSION=9.1.1` during pytest import,
alongside the three previously observed JAX/TF changes. Preserve its unset
startup value. This explicitly controlled CPU/thread/current-CI-ADDOPTS profile
differs from historical main's unspecified ADDOPTS; it is context evidence, not
a historical execution or paired-performance certificate. Fresh current-tree
context qualification remains required.

The staged implementation is committed locally at `0ccd9041`; all seven normal
pinned hooks pass (25.92 s, 453 MiB) and strict docs pass (4.79 s, 139 MiB).
Preflight review of the real-campaign helper found a separate existing default
collection gap: without lane authority, a hook could disable `collectonly` and
run a body. The new unchanged default-mode witness actually creates its body
sentinel and fails against that source. Preserve
`/tmp/jax2onnx-ci-default-collection-purity-predecessor-v1-py311-mvxpte1f`.
Astra requires the existing protocol guard for every canonical collection
request before preparing the real manifest; the two/five-key wires stay intact.
The unexecuted temporary v2 helper also needs its first packet's literal owner
corrected to `test_eqx.py`, with Eqx DINO in packet two and NNX DINO in packet
four. Neither predecessor is relabeled as successful qualification.

Astra accepts the minimal unconditional protocol guard and corrected unexecuted
helper v3. The amended source is frozen in
`/tmp/jax2onnx-ci-staged-context-source-v2.json`. All **412 runner, selector,
artifact, observer-control and replay cases pass together on each family**,
including enabled synthetic xdist variants: minimum 104.70/105.32 s, 309 MiB;
modern 188.46/189.55 s, 348 MiB. Evidence:
`/tmp/jax2onnx-ci-default-collection-purity-qualified-v2-py311-u0og9f1e` and
`/tmp/jax2onnx-ci-default-collection-purity-qualified-v2-py314-7b1kzce2`.
These results do not qualify the real campaign, context counterpart or runtime
budget. Helper v1/v2 remain preserved and unexecuted.

The guard repair is locally committed at `6ce69f12`; all seven pinned hooks
pass (23.88 s, 453 MiB), as does strict documentation (4.77 s, 139 MiB).
Fresh helper-v3 PREPARE observes **18,630 = 4,571 + 14,059**, with exact ordered
baseline, complementary extras, zero execution and unchanged package/source
epochs (25.84 s, 696 MiB). Both JAX snapshots match the controlled main probe;
the complete environment maps differ only in the explicitly bound run-owned
artifact directory. Evidence is under
`/mnt/d/jax2onnx-ci-baseline-20261029-6dufZ6/canonical-standard-full311-v3-20261029-qualification`.
The staged QUALIFY attempt **FAILS before the first packet**, because the flat
source binding omits imported `internal_docs/precision/poc/fixed_source_preparation.py`
(24.98 s, 737 MiB). Bind that source instead of exempting it. This is the actual
predecessor cause; do not relabel it as a terminal-context failure.

Independent review also finds that pytest restores its temporary
`PYTEST_VERSION` after unconfiguration: main's collected E1 contains `9.1.1`,
but its post-pytest E2 omits the variable. A separate tiny cold-start witness
actually fails on this distinction, with zero phases and observed
`unset -> 9.1.1 -> unset` (1.02/1.45 s, 147 MiB):
`/tmp/jax2onnx-ci-pytest-terminal-lifecycle-predecessor-v1-py311-6p__mt5m`.
Earlier synthetic staged tests inherited this variable and do not qualify
natural cleanup. Astra requires a bounded third snapshot: add exact
`post_pytest_environment` and `post_pytest_jax_config` to the transition, retain
E1 checks through unconfiguration, and explicitly validate E2 after return and
observer cleanup. No ignored/restored-field exception or startup predeclaration
is accepted. Regenerate/review a fresh five-field manifest with all imported
`internal_docs` Python sources; existing helper-v3 outcomes remain preserved.

Independent Astra review accepts the three-boundary implementation and helper
v4. The schema has exactly the five stated fields; collected and post-pytest
completion flags advance only at their explicit lifecycle boundaries. Default
workers record actual final context without changing numerical outcomes or
the two/five-key requests. The amended source in
`/tmp/jax2onnx-ci-staged-context-source-v3.json` passes **429 combined cases on
each family**: minimum 110.04/110.79 s, 311 MiB; modern 196.63/197.85 s,
348 MiB. Evidence:
`/tmp/jax2onnx-ci-pytest-lifecycle-three-stages-v3-py311-tcojb26g` and
`/tmp/jax2onnx-ci-pytest-lifecycle-three-stages-v3-py314-8r9fbj_w`.
Coverage includes the truly unset startup variable, masked unconfigure mutation,
invalid post-pytest metadata, completion flags and semantic receipt corruption.
The new helper observes full final E2/J2 and hashes imported `internal_docs`
Python sources. Fresh real PREPARE/QUALIFY and complete standard/performance
qualification remain pending; prior three-field passes do not establish them.

Read-only comparison of main `81ad6ca` with PR `17585d7a` accepts the source
counterparts below. This is **not support-context collection or execution
acceptance**. Main's three ordinary support jobs run on every existing CI trigger:
pushes to `main`, `v*` tags, pull requests targeting `main`, and manual dispatch.
All use `ubuntu-latest`; retain their separate job contexts and outcomes.

| Support owner | Original setup and settings to retain | Mapping and outstanding qualification |
| --- | --- | --- |
| Minimum dependencies, 35-minute timeout | Main used Python 3.12, `poetry install --no-interaction`, `scripts/install_minimum_dependencies.py`, `pip check`, version reporting and test generation; pytest uses `JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=True` | The installer is unchanged and still installs every declared direct runtime lower bound. Current published PR uses Python 3.11 and JAX/JAXLIB 0.10.2 / Flax 0.12.8. Preserve the logical minimum-installation coverage at these current floors. Finite current-floor 3.12 collection and separate standard/extra execution pass below; historical minimum-context equivalence, full support qualification and runtime parity remain **UNQUALIFIED**. |
| Normalization accuracy, two separate 30-minute jobs | Python 3.12 and 3.13, each revision's own official lock, version/CPU/provider profile, `JAX_PLATFORMS=cpu`; `test_layer_norm_precision.py` followed by `test_cos_precision.py` | Both test files are byte-identical, with 91 and 11 ordinary-collection candidate IDs and no additions. Preserve every parametrization, including the full `[257, 384]` seed-0 input, epsilon `1e-5`, opset 23, all four implementations, three modes, both ORT optimization settings and original float64-reference/JAX error bounds. Python 3.13's JAX 0.11.2 to 0.12.0.1 change remains explicit. Separate support-context qualification is pending. |
| onnxruntime-web smoke, 20-minute timeout | Python 3.12, Node 22, `poetry install --no-interaction`, `npm ci`, generation, then `scripts/run_onnxruntime_web_smoke.sh` | Script, package files, handwritten owners and generated case contracts are unchanged. The script sets `JAX2ONNX_VALIDATE_ONNXRUNTIME_WEB=1`, defaults `JAX2ONNX_ONNXRUNTIME_WEB_RUNNER=node`, and does **not** set `JAX2ONNX_EXPORT_MODE=web`. Fifty ordinary-collection candidate IDs retain the original sources and criteria; actual Node/WASM support collection and execution remain **UNQUALIFIED**. |

The minimum-job context ownership is explicit: retain Python **3.12** as the
standard main counterpart, installing the PR's current declared minima. Keep
the existing Python **3.11** minimum check as additional coverage with separate
cost and outcomes until an independently accepted equivalence permits
consolidation. Neither an ordinary locked run nor identical selected IDs proves
that equivalence; no historical framework downgrade is part of this mapping.

Finite current-minimum Python **3.12** qualification now passes the actual
declared-floor installation and `pip check`, then preserves all seventeen
original selectors under `JAX_PLATFORMS=cpu` and `JAX_ENABLE_X64=True`.
Two phase-free collections partition **615 = 475 candidate main obligations +
140 extras**, with identical complete inventories, exact order and no collection
skips: `/tmp/jax2onnx-minimum312-support-v1-wge0dtqn`
(94.83 s, 490 MiB for generation and both collections). Against the unchanged
frozen source/package/environment manifest, the standard candidates pass
**475 cases / 1,425 phases** (98.56/106.43 s, 2,809 MiB), and a separate extra
run passes **140 cases / 420 phases** (36.39/43.04 s, 552 MiB).
Guarded execution evidence:
`/tmp/jax2onnx-minimum312-support-standard-execution-v1-z32erjil` and
`/tmp/jax2onnx-minimum312-support-extra-execution-v1-exezp9dw`.
Candidate identities derive from main's ordinary own-lock collection; the
corresponding source review is separate from historical minimum-context
equivalence. These results qualify this finite current-minimum selection and
execution, not paired runtime parity, historical framework equivalence, the
complete standard campaign or any gate activation.

A bounded selector schema amendment has **independent source-review ACCEPT**:
optional `collection_args` records
the exact original ordered literal pytest positional selectors. Absent or `[]`
retains the existing full-repository restriction. Require exact equality with
effective `config.option.file_or_dir`; never infer, expand or add selectors.
Use checkout-relative file selectors and literal qualified cases, with no glob,
directory, option token or path escape. All other prefilter/context prohibitions
and membership/order/source checks remain. Record the arguments in each receipt
and retain the controller/worker manifest binding.

The unused amendment passes **68 synthetic cases on each official Python
3.11/3.14 family**, including the five explicitly enabled bounded xdist cases:
`/tmp/jax2onnx-ci-qualified-context-v9-py311-fvr2oeg7`
(20.36/20.84 s, 185 MiB) and
`/tmp/jax2onnx-ci-qualified-context-v9-py314-1uiakm31`
(31.06/31.85 s, 201 MiB). Independent review identified and root reproduced a
directory-resolution gap: a nested invocation could validate a root-relative
file but execute a different file. The preserved predecessor is
`/tmp/jax2onnx-ci-invocation-context-predecessor-v1-s626xizl` (one failed
regression, incorrectly successful inner execution). The final amendment
requires checkout-root invocation for nonempty selectors, rechecks that context
and selector equality after collection, and has direct regressions for both.
Absent/empty selectors retain their original behavior. These synthetic results
do not qualify actual support commands, runtime savings or any gate activation.

For these support contexts, the full inventory is the complete collection of
that original command, before standard/extra projection. Thus new cases inside
an originally selected file become support extras, while cases outside an
original exact-case selector do not silently enter that command. Collecting the
whole repository instead would change the original imports and collection
context. Qualify exact-file and exact-case positives, default strictness and
reordered/missing/injected-argument negatives before use. This reuses the existing
executor and adds no workflow activation; the remaining support contexts,
historical equivalence and runtime comparison remain pending.

The minimum command's original owners, in command order, are listed below.
Counts are observations from the actual main Python 3.12 ordinary collection,
not assertions about the unexecuted historical main minimum-installation
context. The eventual support manifest must contain the expanded literal IDs,
not these file selectors.

| Original selector | Candidate baseline IDs | Added current cases excluded from this standard owner |
| --- | ---: | ---: |
| `tests/extra_tests/framework/test_jax_compat.py` | 11 | 132 |
| `tests/extra_tests/framework/test_minimum_dependency_installer.py` | 3 | 0 |
| `tests/extra_tests/framework/test_onnx_ir_surface.py` | 9 | 0 |
| `tests/extra_tests/converter/test_onnx_function_if_captures.py` | 6 | 0 |
| `tests/extra_tests/converter/test_onnx_function_transformed_captures.py` | 14 | 0 |
| `tests/extra_tests/converter/test_onnx_function_static_captures.py` | 17 | 0 |
| `tests/extra_tests/converter/test_onnx_function_loop_call_params.py` | 9 | 0 |
| `tests/extra_tests/converter/test_typed_builder.py` | 12 | 0 |
| `tests/extra_tests/converter/test_context_typing_compat.py` | 3 | 0 |
| `tests/extra_tests/converter/test_native_context_typing.py` | 1 | 0 |
| `tests/extra_tests/converter/test_native_function_scope.py` | 3 | 8 |
| `tests/extra_tests/loop/test_fori_loop_captures.py` | 7 | 0 |
| `tests/primitives/test_linen.py` | 224 | 0 |
| `tests/extra_tests/test_bfloat16_capability_matrix.py` | 23 | 0 |
| `tests/extra_tests/test_cos_precision.py` | 11 | 0 |
| `tests/extra_tests/test_float16_cpu_inference.py` | 18 | 0 |
| `tests/extra_tests/test_log1p_expm1_precision.py` | 104 | 0 |

The two expanded files must retain only these original qualified test IDs in
the minimum standard owner; all their other current cases belong to extras:

```text
tests/extra_tests/framework/test_jax_compat.py::test_jax_compat_exports_core_types
tests/extra_tests/framework/test_jax_compat.py::test_jax_compat_literal_falls_back_to_jax_src_core
tests/extra_tests/framework/test_jax_compat.py::test_jax_compat_exposes_not_mapped_alias
tests/extra_tests/framework/test_jax_compat.py::test_jax_compat_exposes_shape_equality_helper
tests/extra_tests/framework/test_jax_compat.py::test_jax_compat_concrete_or_error_returns_static_value
tests/extra_tests/framework/test_jax_compat.py::test_fresh_var_like_uses_current_jax_constructor
tests/extra_tests/framework/test_jax_compat.py::test_fresh_var_like_preserves_legacy_quantization_metadata
tests/extra_tests/framework/test_jax_compat.py::test_scan_arity_without_xs
tests/extra_tests/framework/test_jax_compat.py::test_scan_arity_with_consts_carry_and_xs
tests/extra_tests/framework/test_jax_compat.py::test_scan_arity_falls_back_to_legacy_params
tests/extra_tests/framework/test_jax_compat.py::test_scan_arity_rejects_inconsistent_arity
tests/extra_tests/converter/test_native_function_scope.py::test_function_finalization_owns_its_body_and_preserves_metadata
tests/extra_tests/converter/test_native_function_scope.py::test_function_scope_finalizes_once_and_registry_retains_the_native_definition
tests/extra_tests/converter/test_native_function_scope.py::test_repeated_calls_publish_the_same_native_definition_stored_in_the_registry
```

Web smoke's original command order is `test_quickstart.py` (four candidates),
`test_return_modes.py` (17), generated `tests/examples/test_lax.py` (eight),
`tests/examples/test_jnp.py` (20), then the single exact ID
`tests/primitives/test_nn.py::Test_gelu::test_jaxnn_gelu`. The first two files are
under `tests/extra_tests/`. There are no added current cases in these selectors;
do not silently include the separate `test_jaxnn_gelu_f64` case.

Candidate literal lists are reproducible from
`/tmp/jax2onnx-own-lock-collection-a3442583/nodeids.json`: concatenate each
original selector's matching IDs in command order, retaining their within-owner
order. SHA-256 of `json.dumps(ids, separators=(",", ":")).encode()` is
`a5b8da3d2f91ee87d9ec185f5bdeb60ca707ff20ca8f250b8e2bc726b9c9ed77` for
the 475 minimum candidates,
`886a55c8afbc94ede8b9dd8150486d1cc3cf7e9bd5e4792040dc1e43c180157e` for
the 102 normalization candidates, and
`bab141cec69c0f8cc95b9db1821837b78cd8838c89de0515a683063bed1ecf6f` for
the 50 smoke candidates. These external collection snapshots are derivation
receipts, not durable or executable support manifests. Actual support collection
must reconcile version-dependent items, skips and ordering in its own setup.

The unchanged `nightly-qs.yml` remains separate: schedule `0 2 * * *` or manual
dispatch, concurrency `nightly-qs-${{ github.ref }}` with cancellation disabled,
and a 26-hour recent-commit check bypassed by manual dispatch. Both full-web jobs
use Python 3.12, Node 22, `npm ci`, generation and `pytest-xdist`; Chrome also
installs Playwright Chromium. Node has a 180-minute timeout and Chrome 240.
Their scripts set `JAX2ONNX_EXPORT_MODE=web` and web validation, with distinct
`node`/`chrome` runners. Original arguments are `-n auto --dist=loadscope -q --ff -x`.
These are recorded baseline settings, not approved local execution commands.
Full-web collection/worker equivalence is **UNQUALIFIED**; keep its complete main
counterparts and added SOURCE cases in separate ownership. The optional Python
3.12 CPU inventory is not a substitute for either full-web context.

The 45 MaxText and five MaxDiffusion CPU cases have independently accepted
same-ID source counterparts. Their complete plugin packages and shared
construction/RNG/registration/execution helpers are unchanged; full inputs,
seeds, variants and criteria remain. In particular MaxText's original
`skip_numeric_validation=True` remains a limitation, not numerical acceptance.
Both contexts use `MODELS=all`, clean tracked external sources, and unchanged
pins `17d805e3488104b5de96bd19be09491ff73c57c1` /
`b4f95730bf4f00c4fd9e3dd2fdda1b50484afda2`. The literal 50-ID snapshot
`/tmp/jax2onnx-main312-optional-baseline-ids-v3.json` has file SHA-256
`217fb9736a7117424aa27f4138ce4f52603878a39fd9179dcc586cc2bf62e92a`;
all 50 appear in PR312 standard collection and none in extras. This accepts
source and selection equivalence only; duplicate-removal execution and measured
savings remain unqualified. No workflow changed during this review.

## Separate lanes after qualification

| Lane | Exact ownership to implement | Trigger and acceptance meaning |
| --- | --- | --- |
| Standard main baseline | `S(p)` on all four locked Python 3.11–3.14 PR cohorts; main's full support coverage, actual minimum-dependency job and original integration ownership | Every PR update and main push under main's existing relevant conditions. Same baseline capabilities, inputs and criteria; each successful Linux STANDARD job must finish within 1,200 seconds from job start through final cleanup/upload, excluding queue time. `standard-main` success is only this scope. |
| Focused extra regressions | Explicit reviewed IDs in `A(p)` for confirmed extrema, completed-export ownership and mutable-static/callback regressions, relevant SOURCE semantics, changed owners and their consumer/integration regressions | Required on every affected PR update on applicable official framework families/settings. Freeze literal selectors before activation. A missing required extra job is not replaced by standard success. |
| Remaining extra CPU coverage | Every item in `A(p)` not assigned to the focused extra lane, including the complete added SOURCE corpus and full-model cases | Dedicated complete extra acceptance at significant milestones, whenever assumptions change, and on the exact final commit across all supported cohorts. Nightly/manual execution may add evidence. No unmatched addition disappears. |
| Main full web baseline | The complete mapped main Node/Chrome integration selections and original conditions | Preserve main's nightly/manual ownership, separately from newly added web cases. Qualify worker/isolation configuration and collection mapping before replacing its execution. |
| Extra integration and migration acceptance | Added web checks and the individually owned inventory/replay/model/runtime/resource obligations below | Relevant milestone and exact-final-SHA acceptance. Missing implementations, references, criteria or verifiers remain explicit blockers. New strict docs (`poetry run mkdocs build --strict --site-dir <fresh-external-dir>`) is an extra check; main-only deployment was not a PR docs gate. |

The exhaustive CPU union is `S(p) + A(p) = C(p)`, with each assigned item executed
exactly once in its canonical campaign. Repeated baseline support jobs remain
separately identified when their environments/conditions differ. Standard and
extras must have distinct check names, artifacts, aggregate times and verdicts;
extras cannot be hidden inside standard's runtime or silently made optional.
All added tests default to `A`, including additions in old files and generated
metadata. The focused subset is explicit; all other additions still have the
complete extra lane as their owner.

The two previously identified heavy owners stay intact in extra acceptance:
`test_source_original22_state_preparation.py::test_original22_source_state_preparation_cpu_feedback_and_reset`
and `test_source_model_native_bindings_exports.py::test_original_full_model_native_bindings_cpu_gate`
under `tests/extra_tests/converter/`. At the measured epoch these had 44 and ten
cases. Their negatives and every other added case also remain owned. Preserve
original callables, B=2/B=5 inputs, caller x64 variants, state/reset sequences,
native roles and both ORT modes; counts are observations, not future limits.

Nightly acceptance must resolve and record one explicit SHA of the active
migration branch while PR #277 remains open, then main after transition. Change
detection and every checkout use that same SHA. Final acceptance checks the
supplied final commit itself; a PR merge-ref run is additional evidence. This
wiring and the new lane selections are **NOT IMPLEMENTED**. The published gate
remains effective until concrete changes and evidence are independently accepted.

### Migration command and acceptance ownership

Commands below document existing entrypoints; none was executed for this plan.
Angle-bracket paths denote required evidence or fresh external outputs, not
existing files. Resolve them to reviewed literal paths before enabling a job.
Use supported official packages and the cohort's isolated installation. Full
inventory observations require both pinned optional checkouts and `MODELS=all`:
MaxText `17d805e3488104b5de96bd19be09491ff73c57c1`, MaxDiffusion
`b4f95730bf4f00c4fd9e3dd2fdda1b50484afda2`. Record older-cohort conditional
differences; do not normalize a contract mismatch into agreement.

| Obligation / owner lane | Existing command and required evidence | Acceptance boundary and current blocker |
| --- | --- | --- |
| Current inventory identities / migration inventory | `poetry run python scripts/source_precision_inventory.py --output internal_docs/precision/source_migration_inventory_current.json` on both official framework families with complete optional setup; retain stdout, manifest hash and cohort. Without `--refresh` this checks rather than rewrites the manifest. | Identity agreement only. The committed manifest has 316 manual rows; the separately reported later 319-file AST observation also predates new owners. Current identities need reviewed reconciliation, including authorized experimental removals. **UNVERIFIED**; never auto-refresh to obtain green. |
| Current caller-obligation plan / migration inventory | `poetry run python scripts/source_precision_probe.py --baseline internal_docs/precision/source_migration_inventory_current.json --output <fresh>/plan.json` | Planning executes no models. Preserve the modern 1,856 rows / 3,712 caller obligations, manual anchors, optional models and explicit cohort differences. A generated plan is not execution or numerical acceptance. |
| Current replay adapters and artifact checks / migration replay | Existing execution interface: `poetry run python scripts/source_precision_probe.py --baseline internal_docs/precision/source_migration_inventory_current.json --replay-adapters <reviewed-current-adapters.json> --stage product --limit <reviewed-positive-batch-size> --output <fresh>/product.json --evidence-directory <fresh>/artifacts`; `--stage prepare` is preparation only. | Current-epoch adapters and a complete current replay schedule/verifier are **UNVERIFIED / not supplied**. Default baseline/adapters are historical. Product and diagnostic main paths return zero even with failed/pending cells; their exit status cannot be an acceptance gate. Require a verifier of every assigned outcome, scope and artifact, not just counts or process exit. |
| Historical 26-cell corpus supplement / migration replay | `poetry run python scripts/source_precision_probe.py --corpus-replay --limit 26 --history <original36.json> --history <two-dot.json> --output <fresh>/corpus.json --preparation-output <fresh>/preparation.json` | Hard-pins inventory from `158ce625` and both unchanged same-stack history files. Earlier `/tmp` proofs are unavailable; locate authenticated retained bytes or remain **UNVERIFIED**. This invariant check is historical finite evidence, not a replay of the current inventory. |
| Finite 24-cell numerical supplement / migration numerics | `poetry run python scripts/source_precision_numerical_probe.py --approval <approved.json> --accepted-artifact <accepted-artifact.json> --criterion-version source-corpus-fp32-conjunction-v1 --limit 24 --execute --output <fresh>/numerics.json --preparation-output <fresh>/preparation.json` with the exact pinned approvals/artifact history on each applicable cohort | Only 24 selected cells. Require `validated-scoped-samples`, empty `pending`, all invariants and reviewed actual comparisons. Exit zero without `--execute` can mean `prepared-no-runtime`. Required historical inputs and fresh execution are **UNVERIFIED**; neither result closes full SOURCE numerics. |
| Complete current metadata SOURCE/numerical acceptance / migration numerics | The current plan above defines the denominator. Existing finite probes can supply only their explicitly matched cells. | **UNVERIFIED acceptance blocker:** no complete current 3,712-obligation execution/criteria/verdict verifier is supplied. Missing native references, callbacks, criteria or runtime execution remain gaps. Implement and review the missing coverage/verifier before claiming an executable full gate; no diagnostic PASS can replace it. |
| Original22 full-model/resource closure / migration models | Full CPU collection must execute every case in `tests/extra_tests/converter/test_source_original22_state_preparation.py`; the existing full entrypoint is `poetry run python scripts/run_pytest_batches.py -- --durations=50 --durations-min=1`. Retain complete case receipts, JUnit, native/runtime results and resource observations. | Hosted 44-case predecessor results do not qualify the current SHA or close the first-B=5 local 4 GiB censor. **UNVERIFIED resource blocker:** original 22 owners, both caller settings and all inputs stay required; current exact-SHA/resource verifier evidence is missing. Do not schedule a known oversized local trial until a reviewed safe resource solution exists. |
| Native-binding full-owner/resource closure / migration models | The same full CPU entrypoint must retain all 12 cases in `tests/extra_tests/converter/test_source_model_native_bindings_exports.py`, including ten full-model cases and both negatives. | **LOCAL_UNVERIFIED_RESOURCE_HELD** remains a separate acceptance blocker. Keep isolated original invocations and complete artifact transport/replay. Hosted success is separate from the unresolved local 4 GiB hold and does not provide a missing resource-closure verifier. |
| Complete web behavior / integration | Current commands are `scripts/run_onnxruntime_web_full.sh -n auto --dist=loadscope -q --ff -x` and `scripts/run_onnxruntime_web_chrome_full.sh -n auto --dist=loadscope -q --ff -x`, after Python 3.12, Node 22, `npm ci`, generation and Chromium setup for Chrome. | Existing commands are recorded, **not approved bounded execution**. Qualify a serial or explicitly bounded worker configuration with complete selection/outcome equivalence and memory measurements. Preserve the current limits until independently reviewed; no new resource acceptance follows from these commands. |

Each missing verifier above is an implementation obligation in its named lane;
the table is documentation, not a new registry or evidence system. Existing
receipt/report types should carry the eventual verdicts. Product/migration
obligations remain intact when the explicitly prohibited compiler experiments
and their experiment-only tests are removed. Keep all superseded or failing
observations with their actual source epoch and meaning.

Full acceptance is mandatory before finishing each significant B–F milestone,
before public SOURCE transition and before declaring this goal complete. Changes
to framework/ORT locks, original sources/inputs/criteria, state preparation,
Function ownership, precision rules, graph rewrites, replay or lane selection
invalidate affected exhaustive evidence. The corresponding full acceptance must
run at that milestone before it is accepted or the next dependent milestone
starts. Routine fixes within the unfinished milestone need not rerun an already
passing unrelated exhaustive packet after every edit. Final acceptance always
reruns the entire accepted suite on the exact final SHA across all cohorts.

## Collection context, outcomes and evidence

Initial mapping observes the full PR collection. Collecting only baseline modules
can change import-time registrations, native configuration, fixtures or ordering,
including through SOURCE code imported from an existing module. Added imports do
not become baseline tests, but their cost still counts if standard requires them.
Moving extra execution to another lane does not remove its collection overhead.

Global C remains the complete inventory authority. The reviewed owner variant
above is the explicit exception to recollecting full C in every STANDARD worker;
it does not relax existing global/support routes. Qualify concrete current
collector, first/transitive/lazy-import, registry and fixture dependencies under
the exact source/cohort/process context, then validate every supported cohort's
manifest and native outcomes. Main Module completion order and matching boundary
snapshots alone do not prove import equivalence. Preserve ordered projections,
required effects, original assertions and phase outcomes. Retain
`test_each_real_batch_preserves_unassigned_module_import_effects` and
`test_unassigned_collection_drift_stops_before_assigned_execution`; any replacement
mechanism needs review and meaningful counterexamples, including the owner
control that rejects a missing global import effect before native test phases.
Finite source and process-order evidence is required, not a universal proof of
import independence. Required full discovery is charged to STANDARD even in a
separate prerequisite; collection manifests alone cannot qualify a cheaper context.

Derive and freeze the old ordered predecessor packets for the f64-to-f16 tie case
and BF16 native TopK case. Preserve the ten native-binding model cases in separate
fresh processes with full collection until equivalence is proven. Check packet
intersections with both `S(p)` and `A(p)`: these intersections and the exact packets
are **UNVERIFIED**. Assign each intact packet explicitly. If a standard test needs
an added case as an execution predecessor, repair/qualify that dependency or
resolve ownership before activation; silently running extras inside standard or
omitting predecessors is forbidden. Additional context-sensitive failures join
this explicit set pending repair. Supplementary solo passes do not replace failed
original-context results.

Bounded execution, collection-context equivalence and resources must be
qualified before standard activation. There is no mandatory original two-worker
benchmark or inherited concurrency prescription. Reuse existing pytest/xdist or
the existing runner only after its concrete context, ownership and descendant
resource behavior are accepted; **direct xdist remains unqualified**. Do not
restore `-n auto`, multiply hosted jobs or increase local resource caps. Keep the
four-Python-plus-four-support layout initially bounded, with canonical packets
sequential inside jobs. Extra native/model packets retain serial isolation.
Only after per-case duration and descendant memory are measured should Original22
be split into bounded complete-case packets; four cases / 12 predicted minutes
remain a candidate, not an approved allocation. Never split a case's internal
inputs, references or comparison phases to fit a budget.

Each run records PR/base SHA, package/setup provenance, manifest/mapping hashes,
assigned IDs, canonical phase outcomes, separate timings and resource evidence.
Use the existing receipts/JUnit and ordinary Actions artifacts. Aggregation rejects
missing jobs/IDs, duplicates, unassigned outcomes, changed contracts, collection
errors and interrupted phases. Require exact cohort/node/outcome/reason allowances
for existing skips/xfails; collect-only does not authorize runtime skips. Retain
collection skips from unselected modules and all failed/superseded observations.
Allowed skips, expected-rejection tests and diagnostic successes are not numerical
acceptance. Keep native reference, ONNX schema, runtime and numerical verdicts
separate, with applicable dtype/shape/tree, NaN-mask, zero-sign, infinity, rounding
and tolerance checks. Official-native failures remain failed or unverified;
they cannot authorize compiler substitution or weaker comparisons.

## Absolute standard-job runtime and separate extra budgets

Every **successful Linux STANDARD job must complete in at most 1,200 seconds**
from its actual job start through the end of final cleanup and upload. Include
checkout, dependency/setup work, generation, collection, mapping validation,
execution, shared prerequisites, internal waits, aggregation and artifact upload.
Exclude only external runner queue time. Use actual job start/end evidence;
a pytest duration, workload guard or sum of selected child times is insufficient.

Timeout, interruption, guard termination, missing phases/receipts or otherwise
incomplete evidence cannot pass. A short failed prefix is not a timing success.
Record each cohort/support job separately so no slower job is hidden by aggregate
or critical-path summaries. Report aggregate and standard-plus-extra costs for
visibility without reinstating relative main comparisons.

Relative NoIncrease, paired main/PR timing campaigns, median/worst comparative
gates and the original two-worker benchmark are superseded. No matched main
campaign is a prerequisite. Preserve historical main/PR results and failed
attempts; their old timing conclusions do not qualify this absolute rule.
Setup/cache provenance and every existing resource/context obligation remain.
Shared standard costs cannot be reassigned to extras or removed to fit the cap.
The separate extra targets below remain provisional and cannot extend a
STANDARD job's 20-minute limit or replace milestone/final acceptance.

| Scope | Runtime acceptance | Memory target | Uploaded evidence target |
| --- | --- | --- | --- |
| Standard main baseline, all four cohorts/support | Every successful Linux STANDARD job <=1,200 s from actual job start through checkout/setup/generation/collection/execution/aggregation/upload/final cleanup; queue excluded; timeout/interruption/incomplete fails | For all controlled project-work intervals: sampled aggregate owned-process RSS <=4 GiB, host headroom >=2 GiB, and no observed OOM/swap growth; fixed worker count fits the aggregate target | <=100 MiB compressed successful-campaign receipts/logs; seven days. Report complete failure/retry payloads separately without truncation. |
| Focused extras | Provisional <=35m slowest job and <=160m aggregate, charged separately from standard; freeze actual selection and measure before qualification | One native worker per runner; <=4 GiB owned-process RSS and >=2 GiB headroom | <=100 MiB compressed successful receipts/logs; seven days; complete failure evidence retained |
| Complete CPU acceptance, `S + A`, all four cohorts | Provisional <=45m slowest job, <=90m execution makespan with eight available slots, <=480m aggregate including prerequisites; these targets are not achieved results | One native worker per extra model runner; model packets <=12 GiB owned-process RSS with >=2 GiB host headroom; no OOM/swap growth | <=3.5 GiB per cohort / <=14 GiB total compressed, with complete required payloads; 14 days, final evidence retained through review |
| Full web integrations | Main web baseline and added web costs reported separately. Historical 180m Node / 240m Chrome limits do not qualify a replacement's runtime | Current `-n auto` configuration is unqualified under the new resource plan; bounded execution and descendant/headroom measurements need review | Numeric budget UNVERIFIED; retain complete failure/replay evidence and report compressed bytes separately |

Sample every controlled setup, generation, admission, execution,
aggregation/compression and cleanup interval with owned descendants. An observed
breach, OS kill or missing required interval receipt fails resource qualification.
List pinned GitHub-managed checkout/setup/upload/post-hook intervals separately
as resource **UNOBSERVED**, never zero. These service-only gaps do not themselves
block activation; the complete Actions job span must still be <=1,200 s, and no
project work may be hidden in them. Optional unavailable unowned kernel counters
remain labelled unavailable, without a blanket failure when required owned/proc
accounting is complete. Measure owned RSS, shared cgroup counters and actual host
headroom separately; filesystem cache is not model RSS. Full hosted workflow
qualification remains pending.
Budget failure does not erase completed semantic results. An OS kill, timeout or
missing receipt remains incomplete. Existing local one-job/one-worker, one-thread,
CPUs 4–7, 4 GiB aggregate, 12 GiB startup/8 GiB live available-memory and 4 GiB
free-disk guards remain unchanged. Hosted success does not close local model holds.

Independent Astra review permits one finite harness exception for selector
qualification: one outer pytest job may run the opt-in stdlib/pytest/xdist-only
synthetic with a child controller and two workers (at least four Python test
processes including the outer harness). It executes zero JAX/native-framework
workers. Monitor the complete process tree under the same aggregate memory,
headroom, CPU, thread and disk guards. This does not relax the one-native-worker
rule, qualify real-suite xdist resource use or change any CI gate. The test is
assigned to its explicit qualification lane, without a user-approval flow.

## Implementation and activation order

1. Independent Astra architecture review is complete and recorded above. Keep
   the old `C - H` acceptance historical. No workflow or check-name change follows
   from this document or architecture-only acceptance.
2. Prepare isolated main/PR collection evidence using the commands above; review
   the smallest collection-only observation/mapping patch if necessary. Resolve
   every original baseline identity/contract, generated variant, optional case,
   support owner and cross-version difference. Keep the main-to-PR proof readable
   and bounded; no generic capability database is needed.
3. Publish literal standard/extra selectors and implement them incrementally in
   the existing runner/workflow. Test complete/disjoint ownership, missing and
   changed IDs/contracts, skip changes, collection failures, interruption and
   aggregation using small synthetic suites. Run real context comparisons and
   relevant regression owners under the unchanged machine guards. Keep the
   published gate authoritative during this qualification.
4. Complete one exact-revision all-cohort union/equivalence campaign including
   the original full models, and qualify every successful Linux STANDARD job
   against the complete 1,200-second job span.
   Preserve original inputs/criteria and complete payloads; record actual time,
   descendant memory and uploaded compressed size. Missing B–F verifiers remain
   separately named acceptance blockers; they do not prevent these CI repairs.
5. Astra reviews the concrete workflow, total main-to-PR mapping, collection/
   outcome proofs, exact-SHA wiring, skip allowances and measured results. Only
   then activate distinct standard and extra checks. Standard green cannot
   replace a required extra check, and an exhaustive CPU pass cannot close an
   unimplemented migration verifier. Verify every required check at each pushed
   SHA; a red pushed gate takes priority over further feature publication.
6. Retain complete exhaustive validation at significant B–F milestones and when
   its assumptions change. Final acceptance reruns all accepted standard, extra,
   integration and migration obligations on the exact final commit across all
   supported cohorts. The absolute standard-job limit, separately reviewed extra
   budgets, independent numerical review and documentation must pass. Invoke
   normal installed hooks without bypass; applicable hooks must pass and
   legitimate no-matching-file SKIPs remain recorded.

The original architecture review changed only this plan. Subsequent preparation,
source repairs and unused selector qualification are recorded above and in the
handoff. No standard/extra workflow gate is activated. Complete mapping,
execution/context equivalence and performance acceptance remain UNVERIFIED;
bounded source/selector acceptance does not discharge those gates.
