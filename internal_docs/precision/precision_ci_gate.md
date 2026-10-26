# CI gate for the precision migration

This binding addendum records the user's priority: fix PR #277 CI now and keep CI
passing on later commits. It supplements the
[complete implementation work plan](precision_implementation_work_plan.md);
it does not replace or reduce the precision migration objective.

## Immediate acceptance

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
   normal commit hooks run unskipped.
2. After every push, verify the terminal CI outcomes for that exact revision.
   Retain failed, superseded and cancelled predecessor outcomes as they occurred;
   do not report them as green through a later revision's success.
3. Publish additional features only after the current head satisfies the
   complete CI gate. If a pushed head is red, prioritize its repair immediately.
   Necessary CI-repair pushes may proceed while the preceding head is red.

Implementation uses Sol Ultra and significant changes receive Astra Ultra
review. Passing local checks or review never preclaims remote CI success.

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
