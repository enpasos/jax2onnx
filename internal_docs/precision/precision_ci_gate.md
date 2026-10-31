# CI gate for the precision migration

This binding addendum records the revised user objective of 2026-10-29: repair
actual failures, keep required checks passing, and reduce validation cost without
losing product or migration obligations. It supplements the
[complete implementation work plan](precision_implementation_work_plan.md);
it does not replace or reduce the precision migration objective.

The latest user timing instruction on **2026-10-29** requires **each successful
Linux STANDARD job to finish within 1,200 seconds (20 minutes)**, measured from
actual job start through checkout, setup, generation, collection, execution,
aggregation, upload and final cleanup. External runner queue time is excluded.
Timeout, interruption, a guard stop or incomplete evidence fails qualification.
A short pytest/workload duration does not establish the complete job budget.

Preserve the complete frozen `main` baseline, including original Python 3.12
optional models, inputs, assertions and exact skip allowances. Added checks run
in separately executable, complete/disjoint extra lanes and retain their
milestone/final obligations. Keep the initially bounded four-Python-plus-four-
support layout, existing resource limits and context requirements.

This absolute rule supersedes relative NoIncrease, paired main/PR timing
campaigns, median/worst comparative gates, mandatory original two-worker
benchmarking and the older 35-minute standard budget. Retain earlier results as
historical, superseded evidence. The comparative diagnostic is retired without
a replacement framework; its hosted failure stays FAILED_OR_INCOMPLETE.
Mapping, collection context, resource and direct-xdist qualification remain
pending. This document changes no runner, selector, ordinary workflow or timeout
gate and grants no activation or standard 20-minute acceptance.

## Immediate acceptance

The published workflow still runs all eight ordinary checks, including the four
complete Python jobs. The [execution plan](ci_execution_plan.md) has independent
Astra architecture acceptance for incremental implementation. Its scheduling
and routine/exhaustive gate split are not implemented or qualified. The user's
revised objective permits that split after independent review of the concrete
implementation and its activation evidence.

All eight PR #277 checks must reach terminal **SUCCESS** on the exact current
head, including the Python 3.11–3.14 jobs and every support job. Record the commit
and run links with their actual terminal outcomes. Pending, interrupted,
cancelled or skipped work is not a successful check; an older head's green run
does not qualify the current revision. Local checks, review and remote CI remain
separate evidence.

Repair the actual cause while preserving Python's 3.11 support floor, the
JAX/JAXLIB 0.10.2 and Flax/NNX 0.12.8 floors, lockfile versions/markers and all
supported cohorts. Preserve original tests, meaningful negative controls,
criteria, fixed sources/callables, models and full inputs. Skips, removals,
weakened assertions, relaxed tolerances/thresholds, reduced samples or raised
resource budgets cannot substitute for a repair.

## Gate for subsequent changes

1. Before pushing a feature change, complete its focused tests, required local
   checks and review. Keep significant changes in meaningful commits with the
   normal installed hooks invoked without bypass. Applicable hooks must pass;
   legitimate no-matching-file SKIPs remain recorded.
2. After every push, verify the terminal CI outcomes for that exact revision.
   Retain failed, superseded and cancelled predecessor outcomes as they occurred;
   do not report them as green through a later revision's success.
3. Publish additional features only after the current head satisfies all routine
   requirements of the independently accepted, implemented CI plan; until then,
   the existing eight-check gate applies. If a pushed head is red, prioritize its repair immediately.
   Necessary CI-repair pushes may proceed while the preceding head is red.

Implementation uses Sol Ultra and significant changes receive Astra Ultra
review. Passing local checks or review never preclaims remote CI success.

The revised objective no longer requires every exhaustive obligation after
every push. Full acceptance remains mandatory at significant migration
milestones, when affected assumptions change, and on the exact final commit
across every supported cohort. Changes invalidate affected prior evidence;
unrelated successful packets need not be repeated after each edit. A changed
check name or a green subset never proves that former full coverage executed.
The proposed lane mapping and budgets remain unverified until actual runs and
independent review establish them.

## Official dependency boundary

Use supported official JAX/jaxlib releases with the existing versions and lock
markers. Custom builds, dependency patches, binary overlays or substitutions
of JAX, jaxlib, XLA or LLVM require separate explicit user approval. Ordinary
implementation, CI repair, commit and push authorization does not grant it.
The unapproved compiler experiment is being removed; its tests are not product
acceptance obligations. Keep the native BF16 counterexample and distinguish it
from converter and execution-target failures.

## Authorized scope and continuing objective

The user authorizes scoped CI repairs and ordinary pushes to the existing PR
branch that restore and maintain this gate. This does not authorize merging,
force-pushing, unrequested messages or broader publication.

The full B–F precision migration remains unfinished until its original
acceptance gates are met, including fixed original sources, all existing
functionality and one coherent public SOURCE UX without permanent selectors,
fallbacks or capability loss. Public defaults remain LEGACY until the reviewed
phase-F transition. CI health at one head qualifies that head's checks; it does
not close the remaining model, capability or migration obligations.

Until a reviewed scheduling change is implemented, full-suite CI uses fresh
serial pytest batches. Every batch verifies the same
full ordered collection before selecting its assigned items, preserving
collection-time imports and registrations; each item executes intact in one
process. PASS requires the completed node IDs to form an exact, disjoint union of
the original collection, with original pytest outcome rules and complete
phase/JUnit accounting. Receipts and passive resource observations document
execution; they do not establish OOM as the cause of earlier interruptions or
close the separate model resource gate.
