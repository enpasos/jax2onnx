# Precision migration CI execution plan

Status: **Main-baseline architecture accepted; implementation, gate activation and measured qualification pending.**

## Binding user clarification — 2026-10-29

> die Standardprüfung sollte den Status Quo wie auf main prüfen, damit wir nichts gegenüber diesem Stand verlieren, aber auch nicht länger laufen. Zusatzprüfungen wären on top und dann auch gern extra.

The standard campaign must preserve main's complete existing coverage and must
not take longer than a comparable main campaign. Every added SOURCE, migration
or regression check belongs to a separately identified extra lane. Those checks
remain required at their assigned trigger; separating them does not delete,
weaken or waive them. Extra work and its runtime must remain visible.

This clarification supersedes the previously accepted `C(p) - H(p)` routine
selection and its **35-minute / 160-minute standard runtime budgets**. Neither
selecting nearly the entire expanded PR suite nor reducing standard coverage to
two representative Python cohorts satisfies the new instruction. The earlier
architecture acceptance is historical; it is not acceptance of this amendment.
No workflow or required check has changed for this revision.

Independent Astra Ultra review on 2026-10-29 by
`/root/astra_ci_plan_review/main_baseline_architecture_review`: **ACCEPT for
architecture**, including the subsequent one-to-one/phase-F clarification.
The reviewed candidate permits incremental implementation. Literal selection,
main-to-PR contract mapping, collection-context equivalence, measured no-increase
runtime and all gate-activation evidence remain unqualified. This review does
not accept missing migration verifiers or claim performance results.

The reference main commit, supplied by root's fresh remote/local observation, is
`81ad6ca2592635c8d2ced6dbd77292c487f0351b`. Freeze this SHA as the initial baseline;
record and independently reconcile later main changes instead of silently
rebasing or shrinking the baseline. Current published PR head `4ec9fad4` has
four successful support jobs and four full Python jobs still in progress in
[CI 40847090831](https://github.com/enpasos/jax2onnx/actions/runs/40847090831).
Its independently accepted NNX repair has dual-framework focused evidence and
all seven normal pinned hooks. Predecessor `afa6d16e` had four successful support
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
saving. Ordinary commit hooks remain required and unskipped.

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
counts in standard's comparison. Added pytest identities/scenarios and new
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
| Standard main baseline | `S(p)` on all four locked Python 3.11–3.14 PR cohorts; main's full support coverage, actual minimum-dependency job and original integration ownership | Every PR update and main push under main's existing relevant conditions. Same baseline capabilities, inputs and criteria; comparable standard runtime must not increase. `standard-main` success is only this scope. |
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

Keep full collection authoritative during qualification. Compare full-collection
baseline execution with the proposed cheaper context on an immutable PR revision
and both official framework families, then validate every supported cohort's
manifest/outcomes. Preserve ordered projections, registry/configuration effects,
original assertions and phase outcomes. Retain
`test_each_real_batch_preserves_unassigned_module_import_effects` and
`test_unassigned_collection_drift_stops_before_assigned_execution`; any replacement
mechanism needs review and meaningful counterexamples. A full collection still
required by standard is charged to its measured time, even if performed in a
shared prerequisite. Collection manifests alone cannot qualify a cheaper context.

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

The standard candidate begins from main's documented fixed concurrency (two
xdist workers in the measured main run), not the PR's arbitrary 500-item serial
packet layout. Reuse existing pytest/xdist or the existing runner as appropriate;
no scheduler service is needed. Qualify resource use, isolation and collection
semantics before selecting worker/shard counts. Do not restore `-n auto` or
increase local resource caps. Extra native/model packets retain serial isolation.
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

## No-increase standard runtime and separate extra budgets

Standard acceptance is **no increase versus comparable main**, not 35m per job
or 160m aggregate. Use at least three complete paired main/PR-standard campaigns
with frozen source SHAs, each revision's approved lock/setup, the same standard
hosted runner class/recorded CPU features, available concurrency and matched
cold/warm-cache treatment. Pair runs closely and alternate order. Report cohort
job times, workflow critical path, aggregate charged job time and all raw pairs.
A framework-version difference remains an explicit confounder; it does not grant
a runtime allowance or permission to replace PR locks.

Require both the median and worst observed **standard critical path and aggregate
job time** to be no higher than the corresponding main measurements, with no
unreviewed tolerance. Report each cohort/support comparison so a slower owner
cannot be hidden by renamed checks. The initial historical 18m04s slowest job
and 54m30s aggregate are orientation only, not controlled thresholds or proof of
makespan. If comparable evidence is unavailable or the no-increase test fails,
performance acceptance remains UNVERIFIED/FAILED; optimize without removing
coverage or obtain an explicit reviewed change to the measurement plan.

Charge dependency installation, test generation, collection, mapping validation,
shared prerequisites, support, aggregation and uploads to the campaign that
requires them. Shared costs required for standard cannot be reassigned to extras
to make standard appear faster. Record baseline-bootstrap/cache-miss costs and
cache-hit steady-state results separately; qualify both applicable operating
modes. Count all failed attempts and retries additionally. External runner queue
time is reported separately; internal dependency waits and scheduling waves
remain part of execution makespan. Report combined standard-plus-extra cost too.

| Scope | Runtime acceptance | Memory target | Uploaded evidence target |
| --- | --- | --- | --- |
| Standard main baseline, all four cohorts/support | No increase in comparable main critical path and aggregate time under the paired rule above | <=4 GiB total owned-process RSS per ordinary runner, >=2 GiB host headroom, no OOM/swap growth; fixed worker count must fit this aggregate target | <=100 MiB compressed successful-campaign receipts/logs; seven days. Report complete failure/retry payloads separately without truncation. |
| Focused extras | Provisional <=35m slowest job and <=160m aggregate, charged separately from standard; freeze actual selection and measure before qualification | One native worker per runner; <=4 GiB owned-process RSS and >=2 GiB headroom | <=100 MiB compressed successful receipts/logs; seven days; complete failure evidence retained |
| Complete CPU acceptance, `S + A`, all four cohorts | Provisional <=45m slowest job, <=90m execution makespan with eight available slots, <=480m aggregate including prerequisites; these targets are not achieved results | One native worker per extra model runner; model packets <=12 GiB owned-process RSS with >=2 GiB host headroom; no OOM/swap growth | <=3.5 GiB per cohort / <=14 GiB total compressed, with complete required payloads; 14 days, final evidence retained through review |
| Full web integrations | Main web baseline and added web costs reported separately. Historical 180m Node / 240m Chrome limits do not qualify a replacement's runtime | Current `-n auto` configuration is unqualified under the new resource plan; bounded execution and descendant/headroom measurements need review | Numeric budget UNVERIFIED; retain complete failure/replay evidence and report compressed bytes separately |

Measure owned process trees, cgroup peaks and actual host headroom separately;
filesystem cache is not model RSS. Missing/incomplete telemetry makes resource
acceptance UNVERIFIED, never zero usage. The runner's existing parent-PID samples
do not establish descendant peaks; full hosted qualification remains pending.
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
   the original full models, then the paired standard no-increase measurements.
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
   supported cohorts. No-increase standard and separately reviewed extra budgets,
   independent numerical review, documentation and normal hooks must all pass.

The original architecture review changed only this plan. Subsequent preparation,
source repairs and unused selector qualification are recorded above and in the
handoff. No standard/extra workflow gate is activated. Complete mapping,
execution/context equivalence and performance acceptance remain UNVERIFIED;
bounded source/selector acceptance does not discharge those gates.
