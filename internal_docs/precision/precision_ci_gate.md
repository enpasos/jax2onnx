# CI gates for the precision migration

These are the acceptance rules for PR #277, updated 2026-11-01. They preserve
the complete [migration work plan](precision_implementation_work_plan.md).
The [execution plan](ci_execution_plan.md) owns literal selections, commands,
resource measurements and activation evidence; the [handoff](implementation_handoff.md)
records implementation and failed predecessors. A passing CI lane establishes
only the work assigned to that lane.

## What must pass before completion

The complete accepted suite must pass on the **exact final commit**, across
all supported Python and official framework cohorts. This includes the main
baseline, added SOURCE tests, complete models, inventory and replay, numerical
acceptance, minimum dependencies, browser integrations and normal quality checks.
Independent Astra review must accept the implementation and its evidence. A
result from another revision, a smaller example or a green diagnostic test
cannot close a missing final obligation.

At the documented published checkpoint, ordinary export still uses **LEGACY**,
the existing precision policy, and **SOURCE** names the private implementation
of the intended contract. This is the current implementation, not a requirement
to preserve the old default throughout the unmerged PR. The branch may adopt the
new contract incrementally through normal reviewed code, tests and documentation.
Complete capability, compatibility and scientific acceptance remain merge and
release requirements. These documentation changes switch no code or CI gate.

## Current qualification

Published `574b277e` has independently verified routine coverage in
[CI597](https://github.com/enpasos/jax2onnx/actions/runs/41274200501): sixteen
STANDARD jobs and five EXTRA jobs, including the strict part joins, minimum
Python 3.11 coverage and complete minimum 3.12 support EXTRA, meet their measured
whole-job, aggregate-time and successful-evidence budgets. This establishes
routine pytest and resource coverage only. Complete added coverage, numerical
acceptance and full integration acceptance remain separate obligations.

The latest observed published checkpoint, `7d33b9d1`,
[CI604](https://github.com/enpasos/jax2onnx/actions/runs/41296409719), finished
with 25 successful checks and six failed checks. All three strict STANDARD
joins succeeded. The four full Python jobs stopped at outdated Eqx state-record
assertions; full Node and Chrome stopped at a BitCast test that supplied a
synthetic equation where a native JAX equation is required. Overall acceptance
remains open.

This branch integrates current main and the four dependency updates, including
mechanical formatting and corrections to three test files. Bounded local runtime
checks pass on both framework variants, the existing ONNX Runtime minimum and
the updated Node/Chrome stack. Exact-head hosted acceptance and the precision
merge gates remain required.

Earlier failed and interrupted campaigns remain recorded in the execution plan
and handoff with their original causes and outcomes.

The tracked activation switch remains false, so original and replacement owners
run together. Expanded focused selections and browser obligations need actual
current-head qualification and independent review before routine retirement.
Green routine checks do not close full added coverage, numerical acceptance or
the complete migration.

## The 20-minute STANDARD budget

Every successful Linux STANDARD job must finish in **at most 1,200 seconds**.
Measure from actual job start through checkout, dependency setup, generation,
collection, hooks, execution, aggregation, upload and final cleanup. Only the
external runner queue is excluded. Shared prerequisites retain their proper
cost; moving them to another job cannot hide STANDARD work.

A pytest duration or a successful work segment is insufficient. Timeouts,
interruption, cancellation, resource-guard stops and missing evidence fail
qualification. Report every cohort and support job, aggregate runner time,
owned-process memory and uploaded bytes. The former relative timing campaigns,
35-minute STANDARD allowance and mandatory two-worker benchmark are superseded.

| Resource | Binding boundary |
| --- | --- |
| Ordinary STANDARD execution | Existing 1,100-second monitored-work deadline; 1,200-second whole-job cap; 4 GiB aggregate owned-process RSS. |
| Reviewed Python 3.12 optional-model exception | Its existing 5 GiB limit and two-part join remain specific to that profile. This does not raise other cohorts' limits. |
| STANDARD campaign | Existing 160 job-minute aggregate and 100 MiB compressed successful-evidence budgets; record complete failure evidence separately. |
| Local native execution | Retain the existing one-job/one-native-worker, thread, CPU, available-memory and free-disk guards. Hosted success does not close a local model resource hold. |
| EXTRA | Focused jobs retain the separate provisional 35-minute limit; minimum 3.12 complete support EXTRA keeps the ordinary 20-minute limit. Both share the separate 160 job-minute aggregate / 100 MiB successful-evidence targets and never extend STANDARD. |
| Expensive complete validation | Separate measured runtime, memory and artifact budgets in the execution plan. Provisional targets are not achieved results or permission to raise local caps. |

Measure all controlled project-work intervals and their descendants. Label
unobserved GitHub-managed service intervals explicitly; never record them as
zero. A successful job still needs its complete wall-clock span. Resource
stops do not prove an OOM, cache leak or compiler cause.

## Coverage ownership

For each cohort, `C` is the complete collection, `S` is the preserved main
baseline, and `A = C − S` is every added test. An explicitly reviewed focused
subset `F` belongs to `A`; `R = A − F` remains assigned to complete EXTRA.
The letters describe test ownership, not numerical acceptance.

| Lane | Required work and trigger |
| --- | --- |
| STANDARD | Main-equivalent tests and support coverage on Python 3.11–3.14, including original optional-model inputs and assertions, minimum dependencies and original integration ownership. Required on routine updates under the original conditions. |
| Focused EXTRA | Literal reviewed regressions for extrema, Function ownership, NNX static rejection, affected SOURCE semantics and relevant consumers on all applicable cohorts. Required on affected routine updates. |
| Complete EXTRA | All of `A`, replacing focused-only execution at significant milestones and final acceptance; no unmatched addition disappears. Preserve all original model inputs, state sequences, references and criteria. |
| Other integration and migration lanes | Complete Node/Chrome coverage, current inventory/replay and numerical verdicts, model/resource closure, strict docs and remaining accepted obligations. Each has an executable owner and the required milestone/final trigger. |

Keep legacy code only where an identified CI or regression/reference comparison
needs it, with [explicit removal criteria](precision_implementation_work_plan.md#temporary-legacy-bridges).
That temporary need does not require every public default or lowering to remain
unchanged throughout development. Changes to the public contract still need
normal reviewed implementation, tests and migration documentation.

Retain original checks while replacements are qualified. Retirement requires
complete ownership, exact-head collection/outcome equivalence, measured budgets
and independent review. A new green check name does not prove that the former
coverage ran. Keep minimum-Python coverage, minimum-runtime support's complete
added selection, optional environments and active-branch SHA wiring explicit.

The exhaustive union is exactly `S + A = C`, with no duplicates or omissions
within its canonical campaign. Preserve collection imports, registration,
fixtures, process order and context-sensitive predecessors. An accepted
Main-owner collection route is a scoped exception, not permission to stop full
inventory discovery. Part executions qualify only with the strict complete join.
Keep each test case intact; splitting its internal samples or comparison phases
to fit a budget is not allowed. Direct native xdist remains unqualified.

Receipts and JUnit must account for collection and setup/call/teardown outcomes,
assigned and completed IDs, source/cohort/context, exact existing skip allowances,
and final artifacts. Missing phases, changed contracts, unexpected skips,
xfails, duplicate IDs or unassigned work cannot pass. Use existing runners and
reports; no additional controller or evidence framework is required.

## Publication and review

Before a feature push, complete focused regressions, relevant supported-family
checks, normal pinned hooks, documentation checks and independent review. After
every push, inspect the required terminal outcomes for that exact revision.
Repair a new red check before publishing further features; CI-repair pushes are
authorized while the previous head is red. Local tests and review do not preclaim
hosted success.

The user's named **BF16 publication exception** permits independently reviewed
SOURCE pushes despite the retained official-native BF16 TopK failure. That case
remains mandatory and **FAILED** where observed. The exception waives neither a
new CI regression nor scientific, milestone or final acceptance. A later passing
sample does not erase the failed predecessor or establish an upstream repair.

The workflow's explicit full mode and standalone reviewed commit marker
`Precision-Acceptance: full` select the original exhaustive Python contexts and
full browser route at one exact commit. Unknown or changed production/runtime/
reference/criterion inputs conservatively select that route. Reviewed routine
orchestration or documentation changes may reuse unrelated exhaustive evidence.
Minimum Python 3.11 keeps all original selectors under STANDARD resources;
minimum 3.12 also owns its complete support EXTRA selection. New wiring and
resource behavior need actual qualification before routine checks retire.
The tracked activation switch stays false during the nonreplacement campaign;
all old/new owners and focused work run together. Only a later independently
accepted ordinary commit may enable retirement. Preserved raw full-C execution
still requires strict receipt/skip/xfail review; zero exit is not strict A or
numerical acceptance.

Full validation is required at significant migration milestones, when changed
assumptions invalidate it, and on the exact final commit. Ordinary fixes need
not rerun unrelated successful exhaustive packets after every edit. Implementation
uses Sol Ultra; architecture and independent acceptance use Astra Ultra.

## Dependencies and numerical verdicts

Use supported official JAX/jaxlib releases and preserve both lockfiles' versions
and cohort markers. Custom builds, patches, binary overlays or substitutions of
JAX, jaxlib, XLA or LLVM require separate explicit user approval and are currently
prohibited. Permission to implement, repair CI, commit or push does not grant it.
The unapproved compiler infrastructure has been removed; retain the concise
[upstream counterexample](../../docs/about/dependencies.md).

Schema validity, source semantics, runtime execution and numerical acceptance
are separate gates. The current user decision keeps native eager/JIT
contradictions **FAILED** until the required official roles agree. Changing that
reference rule would require a separate explicit decision. Missing
references, criteria or numerical execution never count as acceptance. Enforce
output dtype/shape/structure, exact discrete results, required NaN masks, zero
signs and downstream reciprocal signs; apply bitwise checks where justified.
Valid all-zero results are allowed. NaN payload equality is not a universal
output criterion. See the [numerical contract](precision_target_state.md#numerical-correctness).

Ordinary commits and non-force pushes to the existing branch and PR are
authorized. Merge, force-push, release publication and unrequested external
messages are outside this task.
